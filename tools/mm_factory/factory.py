"""One-click motion-matching set generator for kimodo.cpp (prototype).

Generates a small, phase-consistent locomotion set for one style by driving
`kmd-generate --server` directly (never the web demo, so nothing lands in
demo-output/):

  Idle loop  -> anchors the idle pose used by every start, stop and turn
  Walk loops -> one per direction, pinned seamless cycles (two-pass recipe)
  Starts     -> idle pose ... first frames of that direction's loop
  Stops      -> first frames of the loop ... idle pose
  Turns      -> idle pose ... idle pose rotated (turn-in-place recipe)

Every clip is measured and checked (floating start, foot skate, path and
heading tracking, style signature, similarity to the hero loop); failures are
regenerated with a new seed. Output: GLBs in category folders, manifest.json
and report.md.

usage: python -I tools/mm_factory/factory.py --style zombie [--out factory-output/zombie]
       [--dirs Fwd,Bwd,Left,Right] [--turns L90,R90,L180,R180] [--seed 1] [--retries 3]
       [--soma-assets DIR] [--min-free-mib 3000]
"""
import argparse
import json
import math
import os
import queue
import shutil
import subprocess
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import clips as C  # noqa: E402
import skeleton as S  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
BIN = os.path.join(REPO, "build-win")
DEFAULT_MOTION = os.path.join(REPO, "models", "kimodo-soma-rp-v1.1-f32.gguf")
DEFAULT_TEXT = os.path.join(REPO, "Llama-3-Kimodo-Q8_0.gguf")
TURNS = {"L90": math.radians(90), "R90": math.radians(-90), "L180": math.radians(180), "R180": math.radians(-180)}


# ---------------------------------------------------------------- generator process

class Generator:
    def __init__(self, motion, text, timeout=900):
        env = dict(os.environ)
        env["PATH"] = os.path.join(BIN, "bin") + os.pathsep + BIN + os.pathsep + env.get("PATH", "")
        self.timeout = timeout
        self.log = open(os.path.join(os.environ.get("MMF_LOG_DIR", "."), "kmd-generate.log"), "ab")
        self.p = subprocess.Popen([os.path.join(BIN, "kmd-generate.exe"), "--server", motion, text], cwd=BIN, env=env,
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log)
        self.lines = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for raw in self.p.stdout:
            self.lines.put(raw.decode("utf-8", "replace").rstrip("\r\n"))
        self.lines.put(None)

    def generate(self, req):
        out = os.path.normcase(os.path.abspath(req["output"]))
        if "demo-output" in out:
            raise RuntimeError("refusing to write into demo-output: " + out)
        t = time.perf_counter()
        self.p.stdin.write((json.dumps(req, separators=(",", ":")) + "\n").encode("utf-8"))
        self.p.stdin.flush()
        try:
            ans = self.lines.get(timeout=self.timeout)
        except queue.Empty:
            ans = "TIMEOUT"
        if ans is None:
            raise RuntimeError("kmd-generate exited (see kmd-generate.log)")
        return ans.startswith("OK\t"), ans, time.perf_counter() - t

    def close(self):
        try:
            self.p.stdin.close(); self.p.wait(timeout=60)
        except Exception:
            self.p.kill()


SOMA_BASE = "https://raw.githubusercontent.com/nv-tlabs/kimodo/main/kimodo/assets/skeletons/somaskel77/"
SOMA_FILES = {  # sha256 of the copies validated on 2026-10-09 (docs/motion-stage-research/07-soma-mesh-export-and-blender.md)
    "skin_standard.npz": "90ee2cdf50f168382a7dd7e9c88b118298aba0aca4b72022080c89c4bab0ceb2",
    "relaxed_hands_rest_pose.npy": "64a3828e0d1ef1f1de8228c74eba8040c0810d898169f1892d8997a213b2b64c",
    "somaskel77_standard_tpose.bvh": "3e8cdaf72d2b12a25450ff1af7da261175a830e186c1e07ff094f99ef604d85b",
}


def ensure_soma_assets(d):
    """Download upstream Kimodo's SOMA body-mesh assets (Apache-2.0) once, verifying hashes. No pickles."""
    import hashlib
    import urllib.request
    os.makedirs(d, exist_ok=True)
    for name, sha in SOMA_FILES.items():
        p = os.path.join(d, name)
        if not os.path.exists(p):
            print(f"downloading {name} ...", flush=True)
            with urllib.request.urlopen(SOMA_BASE + name, timeout=60) as r, open(p + ".part", "wb") as f:
                f.write(r.read())
            os.replace(p + ".part", p)
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        if h != sha:
            raise SystemExit(f"{name}: sha256 {h} does not match the validated asset; delete it or pass --soma-assets")
    return d


def free_vram_mib():
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, check=True).stdout.split()[0]
        return int(q)
    except Exception:
        return None


# ---------------------------------------------------------------- checks

def common_metrics(m, P, xz, hd):
    r = {"pelvis_max_m": float(P[:, S.HIPS, 1].max()),
         "foot0_m": float(P[0, S.FEET, 1].min()),
         "skate_cms": S.skate_cms(P, m["contacts"]) if m["contacts"] is not None else float("nan")}
    if xz is not None:
        e = np.linalg.norm(P[:, S.HIPS][:, [0, 2]] - xz[:len(P)], axis=1)
        r["path_err_mean_cm"] = float(e.mean() * 100); r["path_err_max_cm"] = float(e.max() * 100)
    if hd is not None:
        he = np.degrees(np.abs(S.wrap(S.heading(P) - hd[:len(P)])))
        r["heading_err_mean_deg"] = float(he.mean()); r["heading_err_max_deg"] = float(he.max())
    return r


def check(name, value, lo=None, hi=None):
    ok = (lo is None or value >= lo) and (hi is None or value <= hi)
    lim = (f">= {lo}" if lo is not None else "") + (" and " if lo is not None and hi is not None else "") + (f"<= {hi}" if hi is not None else "")
    return {"check": name, "value": round(float(value), 3), "limit": lim, "ok": bool(ok)}


def signature_checks(style, post):
    sig = style.get("signature", {})
    out = []
    for c in sig.get("all", []):
        out.append(check("signature " + c["metric"], post[c["metric"]], c.get("min"), c.get("max")))
    anys = sig.get("any", [])
    if anys:
        res = [check(c["metric"], post[c["metric"]], c.get("min"), c.get("max")) for c in anys]
        ok = any(r["ok"] for r in res)
        out.append({"check": "signature any of " + ", ".join(r["check"] + " " + r["limit"] for r in res),
                    "value": "; ".join(f"{r['check']}={r['value']}" for r in res), "limit": "one must pass", "ok": ok})
    return out


# ---------------------------------------------------------------- factory

class Factory:
    def __init__(self, args, style_name, style):
        self.args, self.style_name, self.style = args, style_name, style
        self.out = os.path.abspath(args.out)
        self.work = os.path.join(self.out, "_work")
        os.makedirs(self.work, exist_ok=True)
        os.environ["MMF_LOG_DIR"] = self.work
        self.gen = None
        self.clips = []      # manifest entries
        self.timings = []
        self.subject = style["subject"]
        self.speed = float(style.get("speed", 1.2))
        self.check_cfg = dict({"max_pelvis_m": 1.15, "max_foot0_m": 0.10, "max_skate_cms": 15.0,
                               "max_path_err_mean_cm": 7.0, "max_heading_err_deg": 12.0, "turn_final_err_deg": 6.0,
                               "turn_overshoot_deg": 25.0}, **style.get("checks", {}))
        self.label = style.get("label", style_name.capitalize())
        self.anchor = None
        ha = style.get("hand_anchor")
        if ha:
            d = ha["pose_dir"] if os.path.isabs(ha["pose_dir"]) else os.path.join(HERE, ha["pose_dir"])
            am = S.load_motion(d)
            self.anchor = C.pose_from(am, 0)
            # normalise the carry to heading 0 so hand pins never fight the root2d heading
            h0 = float(S.heading(S.fk(am["root"][:1], am["rot"][:1])[0])[0])
            self.anchor["q"][0] = S.qmul(S.yaw_quat(-h0), self.anchor["q"][0])
            off = S.rot_y(self.anchor["pelvis"][[0, 2]] - self.anchor["smooth_xz"], -h0)
            self.anchor["pelvis"][[0, 2]] = self.anchor["smooth_xz"] + off
            self.anchor_every = int(ha.get("every", 8))
        self.signature_on = set(style.get("signature_on", ["loop"]))
        self.carry_edit = style.get("carry_edit")  # {"right": [dx,dy,dz], "left": [...]} in the character frame, applied at export

    def apply_carry_edit(self, root, rot):
        """Deterministic arm-IK layer: move each hand by a fixed offset in the character frame on every frame.
        Spine, legs and root are untouched; identical input poses give identical outputs, so joins stay exact."""
        if not self.carry_edit:
            return rot
        out = rot.copy()
        P, _ = S.fk(root, rot)
        h = S.heading(P)
        for t in range(len(root)):
            c, s = math.cos(h[t]), math.sin(h[t])
            tg = {}
            for side, j in (("right", S.RHAND), ("left", S.LHAND)):
                o = self.carry_edit.get(side)
                if o:
                    tg[side] = P[t, j] + np.array([c * o[0] + s * o[2], o[1], -s * o[0] + c * o[2]])
            out[t] = S.ik_hands(root[t], rot[t], tg)
        return out

    def standing_lift(self):
        """Pelvis lift (m) that turns the walking-height anchor into a standing one (style idle_pelvis_h)."""
        if self.anchor is None or self.style.get("idle_pelvis_h") is None:
            return 0.0
        return float(self.style["idle_pelvis_h"]) - float(self.anchor["pelvis"][1])

    def hand_pins(self, xz, hd, exclude=(), lift=0.0):
        """left-hand + right-hand constraints holding the anchor carry every N frames (prop poses like rifles)."""
        if self.anchor is None:
            return []
        T = len(xz)
        frames = [t for t in range(0, T, self.anchor_every) if t not in set(exclude)]
        if (T - 1) not in exclude and (T - 1) - frames[-1] >= self.anchor_every // 2:
            frames.append(T - 1)
        frames = frames[:19]  # stay under the ~20 constrained frames per type guideline
        lift = np.broadcast_to(np.asarray(lift, np.float64), (T,))
        placed = []
        for t in frames:
            p = C.place(self.anchor, xz[t], float(hd[t]))
            p["root"][1] += float(lift[t])  # hands follow the pelvis (FK from the root), so the carry is kept
            placed.append(p)
        out = []
        for kind in ("left-hand", "right-hand"):
            c = C.fullbody(frames, placed)
            c["type"] = kind
            out.append(c)
        return out

    def log(self, msg):
        print(msg, flush=True)
        with open(os.path.join(self.work, "factory.log"), "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    def start(self):
        free = free_vram_mib()
        if free is not None and free < self.args.min_free_mib:
            raise SystemExit(f"only {free} MiB VRAM free (< --min-free-mib {self.args.min_free_mib}); close something or lower the gate")
        self.log(f"VRAM free: {free} MiB; starting kmd-generate --server")
        self.gen = Generator(self.args.motion, self.args.text)

    # ---- one job with retries
    def job(self, name, build, evaluate):
        """build(seed, outdir) -> (request, xz, headings); evaluate(m, P, xz, hd) -> (metrics, checks)."""
        best = None
        attempts = []
        for k in range(self.args.retries):
            seed = self.args.seed + 101 * k
            outdir = os.path.join(self.work, name, f"seed{seed}")
            req, xz, hd = build(seed, outdir)
            req_path = os.path.join(outdir, "request.json")
            cached = False
            if not self.args.regenerate and os.path.exists(os.path.join(outdir, "local_rotations_xyzw.f32")) and os.path.exists(req_path):
                with open(req_path, encoding="utf-8") as f:
                    cached = json.load(f) == json.loads(json.dumps(req))
            if cached:
                ok, ans, dt = True, "cached", 0.0
            else:
                os.makedirs(outdir, exist_ok=True)
                with open(os.path.join(outdir, "request.json"), "w", encoding="utf-8") as f:
                    json.dump(req, f)
                ok, ans, dt = self.gen.generate(req)
                self.timings.append(dt)
            if not ok:
                self.log(f"  {name} seed {seed}: generator error {ans}")
                attempts.append({"seed": seed, "error": ans})
                continue
            m = S.load_motion(outdir)
            P, _ = S.fk(m["root"], m["rot"])
            metrics, checks = evaluate(m, P, xz, hd)
            fails = [c for c in checks if not c["ok"]]
            score = (len(fails), metrics.get("path_err_mean_cm", 0.0))
            attempts.append({"seed": seed, "seconds": round(dt, 1), "failed": [c["check"] for c in fails]})
            self.log(f"  {name} seed {seed}: {dt:5.1f}s  " + ("PASS" if not fails else "FAIL " + ", ".join(f"{c['check']}={c['value']}" for c in fails)))
            cand = {"name": name, "seed": seed, "dir": outdir, "m": m, "P": P, "metrics": metrics, "checks": checks,
                    "passed": not fails, "request": req, "xz": xz, "hd": hd, "score": score}
            if best is None or score < best["score"]:
                best = cand
            if not fails:
                break
        if best is None:
            raise RuntimeError(f"{name}: every attempt failed in the generator")
        best["attempts"] = attempts
        return best

    def base_eval(self, kind, m, P, xz, hd, post_slice=slice(15, None)):
        metrics = common_metrics(m, P, xz, hd)
        metrics.update(S.posture(P, post_slice))
        cc = self.check_cfg
        checks = [check("pelvis max (floating)", metrics["pelvis_max_m"], hi=cc["max_pelvis_m"]),
                  check("lowest foot at frame 0", metrics["foot0_m"], hi=cc["max_foot0_m"]),
                  check("foot skate", metrics["skate_cms"], hi=cc["max_skate_cms"])]
        if "path_err_mean_cm" in metrics:
            checks.append(check("path error mean", metrics["path_err_mean_cm"], hi=cc["max_path_err_mean_cm"]))
        if kind in ("loop", "start", "stop") and "heading_err_max_deg" in metrics:
            checks.append(check("heading error max", metrics["heading_err_max_deg"], hi=cc["max_heading_err_deg"]))
        if kind != "loop" and kind in self.signature_on:  # loops add theirs in loop()
            checks += signature_checks(self.style, metrics)
        if kind == "idle" and self.style.get("idle_signature"):
            checks += signature_checks({"signature": self.style["idle_signature"]}, metrics)
        return metrics, checks

    # ---- clip builders
    def idle(self):
        T = 90
        xz = np.zeros((T, 2)); hd = np.zeros(T)
        prompt = self.style.get("prompt_idle", "{subject} stands still").format(subject=self.subject)

        def b1(seed, out):
            return C.request(out, prompt, T, seed, [C.root2d(xz, hd)] + self.hand_pins(xz, hd, lift=self.standing_lift()), self.style), xz, hd
        p1 = self.job("idle_pass1", b1, lambda m, P, x, h: self.base_eval("idle", m, P, x, h))
        f = S.calmest_frame(p1["P"])
        a, a1 = C.pose_from(p1["m"], f), C.pose_from(p1["m"], f + 1)
        self.idle_pose = (a, a1)
        self.idle_prev = C.pose_from(p1["m"], f - 1)  # so clips can END exactly on idle frame 0

        def b2(seed, out):
            pins = C.fullbody([0, 1, T - 2, T - 1], [C.place(a, (0, 0)), C.place(a1, (0, 0)), C.place(a, (0, 0)), C.place(a1, (0, 0))])
            return C.request(out, prompt, T, seed, [pins, C.root2d(xz, hd)] + self.hand_pins(xz, hd, (0, 1, T - 2, T - 1), lift=self.standing_lift()), self.style), xz, hd
        p2 = self.job("idle_loop", b2, lambda m, P, x, h: self.base_eval("idle", m, P, x, h, slice(0, None)))
        self.idle_ref = S.posture(p2["P"])
        self.add_clip(p2, "Idle", f"{self.label}_Idle_Loop", 0, T - 2, kind="idle_loop", direction=None, loop=True,
                      extra={"idle_pose_source_frame": f})

    def loop(self, d):
        v = self.speed * self.style.get("speed_factor", {}).get(d, C.SPEED_FACTOR[d])
        word = C.DIR_WORDS[d]
        prompt = self.style.get("prompt_loop", "{subject} walks {dir}").format(subject=self.subject, dir=word)
        T1 = 120
        xz1 = C.straight(v, C.DIRS[d], T1); hd1 = np.zeros(T1)

        def ev(m, P, x, h, sl=slice(15, None)):
            metrics, checks = self.base_eval("loop", m, P, x, h, sl)
            post = metrics
            checks += signature_checks(self.style, post)
            if d != "Fwd" and getattr(self, "hero_post", None):
                tol = dict({"pelvis_h_m": 0.08, "head_h_m": 0.10, "hands_dist_m": 0.15, "hands_h_m": 0.15}, **self.style.get("similarity", {}))
                for key, t in tol.items():
                    checks.append(check(f"similar to hero {key}", abs(post[key] - self.hero_post[key]), hi=t))
            return metrics, checks

        def b1(seed, out):
            return C.request(out, prompt, T1, seed, [C.root2d(xz1, hd1)] + self.hand_pins(xz1, hd1), self.style), xz1, hd1
        p1 = self.job(f"loop_{d}_pass1", b1, ev)
        cyc = S.best_cycle(p1["P"])
        a, L = cyc["a"], cyc["L"]
        M = 15
        F = L + 2 * M
        xz2 = C.straight(v, C.DIRS[d], F); hd2 = np.zeros(F)
        pa, pb = C.pose_from(p1["m"], a), C.pose_from(p1["m"], a + 1)

        def b2(seed, out):
            pins = C.fullbody([M, M + 1, M + L, M + L + 1],
                              [C.place(pa, xz2[M]), C.place(pb, xz2[M + 1]), C.place(pa, xz2[M + L]), C.place(pb, xz2[M + L + 1])])
            return C.request(out, prompt, F, seed, [pins, C.root2d(xz2, hd2)] + self.hand_pins(xz2, hd2, (M, M + 1, M + L, M + L + 1)), self.style), xz2, hd2
        p2 = self.job(f"loop_{d}", b2, lambda m, P, x, h: ev(m, P, x, h, slice(M, M + L)))
        if d == "Fwd":
            self.hero_post = {k: p2["metrics"][k] for k in ("pelvis_h_m", "head_h_m", "hands_dist_m", "hands_h_m")}
        self.loop_pose[d] = (C.pose_from(p2["m"], M - 1), C.pose_from(p2["m"], M), C.pose_from(p2["m"], M + 1), v)
        self.add_clip(p2, "Loops", f"{self.label}_Walk_{d}_Loop", M, M + L, kind="loop", direction=d, loop=True,
                      extra={"speed_mps": v, "cycle_frames": L, "pass1_seed": p1["seed"], "pass1_seam_pose_cm": round(cyc["seam_pose_cm"], 2)})

    def start_stop(self, d, kind):
        a, a1 = self.idle_pose
        ap = self.idle_prev
        lp, la, lb, v = self.loop_pose[d]
        word = C.DIR_WORDS[d]
        if kind == "start":
            T, lead = 60, 12
            speed = C.spring_speed(T, lead, self.style.get("start_halflife", 0.3), 0.0, v)
            prompt = self.style.get("prompt_start", "{subject} is standing still, then starts walking {dir}").format(subject=self.subject, dir=word)
        else:
            T, lead = 72, 8
            speed = C.spring_speed(T, lead, self.style.get("stop_halflife", 0.3), v, 0.0)
            prompt = self.style.get("prompt_stop", "{subject} walks {dir}, then stops and stands still").format(subject=self.subject, dir=word)
        xz = C.path_from_speed(speed, C.DIRS[d]); hd = np.zeros(T)

        def b(seed, out):
            if kind == "start":
                # starts on idle frames 0,1 and ENDS on the loop's frame 0 (preceded by its natural frame -1)
                pins = C.fullbody([0, 1, T - 2, T - 1], [C.place(a, xz[0]), C.place(a1, xz[1]), C.place(lp, xz[T - 2]), C.place(la, xz[T - 1])])
            else:
                # starts on the loop's frames 0,1 and ENDS on idle frame 0
                pins = C.fullbody([0, 1, T - 2, T - 1], [C.place(la, xz[0]), C.place(lb, xz[1]), C.place(ap, xz[T - 2]), C.place(a, xz[T - 1])])
            return C.request(out, prompt, T, seed, [pins, C.root2d(xz, hd)] + self.hand_pins(xz, hd, (0, 1, T - 2, T - 1), lift=self.standing_lift() * (1 - speed / v)), self.style), xz, hd
        p = self.job(f"{kind}_{d}", b, lambda m, P, x, h: self.base_eval(kind, m, P, x, h))
        cat = "Starts" if kind == "start" else "Stops"
        link = {"enters" if kind == "start" else "leaves": f"{self.label}_Walk_{d}_Loop"}
        self.add_clip(p, cat, f"{self.label}_Walk_{d}_{kind.capitalize()}", 0, T - 1, kind=kind, direction=d, loop=False,
                      extra={"speed_mps": v, **link})

    def turn(self, key):
        theta = TURNS[key]
        a, a1 = self.idle_pose
        ap = self.idle_prev
        T = 75 if abs(theta) < 3 else 90
        t0, t1 = 10, T - 14
        xz = np.zeros((T, 2)); hd = C.eased_heading(T, t0, t1, theta)
        side = "left" if theta > 0 else "right"
        amount = "90 degrees" if abs(theta) < 3 else "around"
        prompt = self.style.get("prompt_turn", "{subject} stands still and then turns {amount} to his {side}").format(
            subject=self.subject, amount=amount, side=side)

        def b(seed, out):
            pins = C.fullbody([0, 1, T - 14, T - 2, T - 1],
                              [C.place(a, (0, 0)), C.place(a1, (0, 0)), C.place(a, (0, 0), theta), C.place(ap, (0, 0), theta), C.place(a, (0, 0), theta)])
            return C.request(out, prompt, T, seed, [pins, C.root2d(xz, hd)] + self.hand_pins(xz, hd, (0, 1, T - 14, T - 2, T - 1), lift=self.standing_lift()), self.style), xz, hd

        def ev(m, P, x, h):
            metrics, checks = self.base_eval("turn", m, P, x, h)
            hu = np.unwrap(S.heading(P))
            final = float(np.degrees(hu[-10:].mean()))
            metrics["final_heading_deg"] = final
            metrics["overshoot_deg"] = float(max(0.0, np.degrees(np.max(np.abs(hu))) - abs(math.degrees(theta))))
            checks.append(check("final heading error", abs(final - math.degrees(theta)), hi=self.check_cfg["turn_final_err_deg"]))
            checks.append(check("turn overshoot", metrics["overshoot_deg"], hi=self.check_cfg["turn_overshoot_deg"]))
            return metrics, checks
        p = self.job(f"turn_{key}", b, ev)
        self.add_clip(p, "TurnInPlace", f"{self.label}_TurnInPlace_{key}", 0, T - 1, kind="turn", direction=None, loop=False,
                      extra={"turn_deg": math.degrees(theta)})

    # ---- export
    def add_clip(self, res, category, name, f0, f1, kind, direction, loop, extra):
        m = res["m"]
        root = m["root"][f0:f1 + 1].copy(); rot = m["rot"][f0:f1 + 1].copy()
        smooth = m["smooth"][f0:f1 + 1].copy() if m["smooth"] is not None else None
        contacts = m["contacts"][f0:f1 + 1].copy() if m["contacts"] is not None else None
        o = (smooth[0] if smooth is not None else root[0]).copy(); o[1] = 0.0
        root -= o
        if smooth is not None:
            smooth -= o
        rot = self.apply_carry_edit(root, rot)
        md = os.path.join(self.work, "final", name)
        S.save_motion(md, root, rot, contacts, smooth)
        rel = os.path.join(category, name + ".glb")
        dst = os.path.join(self.out, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        subprocess.run([sys.executable, "-I", os.path.join(REPO, "scripts", "export_glb.py"), "--motion-dir", md, "--output", dst],
                       check=True, capture_output=True)
        files = {"glb": rel.replace("\\", "/")}
        if self.args.soma_assets:
            srel = os.path.join("SOMA", category, name + ".glb")
            sdst = os.path.join(self.out, srel)
            os.makedirs(os.path.dirname(sdst), exist_ok=True)
            subprocess.run([sys.executable, "-I", os.path.join(HERE, "soma_glb.py"), self.args.soma_assets, md, sdst, "--name", name],
                           check=True, capture_output=True)
            files["glb_soma"] = srel.replace("\\", "/")
        P, _ = S.fk(root, rot)
        hu = np.unwrap(S.heading(P))
        final_post = {k: round(v, 3) for k, v in S.posture(P).items()}
        entry = {"name": name, "category": category, "kind": kind, "direction": direction, "loop": loop,
                 "frames": int(len(root)), "duration_s": round((len(root) - 1) / S.FPS, 3), "fps": 30,
                 "files": files, "seed": res["seed"], "attempts": res["attempts"], "passed": res["passed"],
                 "prompt": res["request"]["segments"][0]["prompt"],
                 "root_displacement_xz_m": [round(float(root[-1, 0] - root[0, 0]), 3), round(float(root[-1, 2] - root[0, 2]), 3)],
                 "net_yaw_deg": round(float(np.degrees(hu[-1] - hu[0])), 2),
                 "metrics": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res["metrics"].items()},
                 "checks": res["checks"], "source": os.path.relpath(res["dir"], self.out).replace("\\", "/"),
                 "window": [f0, f1], "final_posture": final_post, "carry_edit": self.carry_edit, **extra}
        if loop:
            X = S.local_pose(P)
            entry["seam_pose_cm"] = round(float(np.linalg.norm(X[-1] - X[0], axis=1).mean() * 100), 3)
        self.clips.append(entry)
        self.log(f"  -> {rel}  ({entry['frames']} frames, {'PASS' if res['passed'] else 'BEST-EFFORT'})")

    def continuity(self):
        """Pose (cm, mean joint distance in a pelvis-relative, heading-aligned frame) and pelvis-velocity
        (cm/frame) differences at every join a motion-matching set relies on."""
        cache = {}

        def P(name):
            if name not in cache:
                m = S.load_motion(os.path.join(self.work, "final", name))
                cache[name] = S.fk(m["root"], m["rot"])[0]
            return cache[name]

        def diff(a, fa, b, fb):
            Pa, Pb = P(a), P(b)
            X = S.local_pose(np.stack([Pa[fa], Pb[fb]]))
            pose = float(np.linalg.norm(X[0] - X[1], axis=1).mean() * 100)
            def vel(Pm, f, out):  # pelvis velocity entering (out=False) or leaving (out=True) frame f, heading-aligned
                f = f % len(Pm)
                v = (Pm[f + 1, 0] - Pm[f, 0]) if out else (Pm[f, 0] - Pm[f - 1, 0])
                h = S.heading(Pm[f:f + 1])[0]
                return np.array([math.cos(-h) * v[0] + math.sin(-h) * v[2], v[1], -math.sin(-h) * v[0] + math.cos(-h) * v[2]])
            dv = float(np.linalg.norm(vel(Pa, fa, False) - vel(Pb, fb, True)) * 100)
            return round(pose, 3), round(dv, 3)

        names = {c["name"] for c in self.clips}
        idle = f"{self.label}_Idle_Loop"
        rows = []
        for c in self.clips:
            n, d = c["name"], c["direction"]
            loop = f"{self.label}_Walk_{d}_Loop" if d else None
            if c["kind"] == "start" and loop in names:
                rows.append((f"{n} last -> {loop} first",) + diff(n, -1, loop, 0))
                if idle in names:
                    rows.append((f"{idle} first -> {n} first (pose only)", diff(idle, 0, n, 0)[0], None))
            if c["kind"] == "stop" and loop in names:
                rows.append((f"{loop} first -> {n} first (pose only)", diff(loop, 0, n, 0)[0], None))
                if idle in names:
                    rows.append((f"{n} last -> {idle} first",) + diff(n, -1, idle, 0))
            if c["kind"] == "turn" and idle in names:
                rows.append((f"{idle} first -> {n} first (pose only)", diff(idle, 0, n, 0)[0], None))
                rows.append((f"{n} last -> {idle} first (rotated)",) + diff(n, -1, idle, 0))
            if c["loop"]:
                rows.append((f"{n} wrap (last -> first)",) + diff(n, -1, n, 0))
        return [{"join": r[0], "pose_diff_cm": r[1], "velocity_diff_cmf": r[2]} for r in rows]

    def write_outputs(self, elapsed):
        man = {"generator": "kimodo.cpp tools/mm_factory (prototype)", "style": self.style_name, "style_config": self.style,
               "model": os.path.basename(self.args.motion), "fps": 30, "conventions": {
                   "up": "+Y", "facing_at_frame_0": "+Z", "left": "+X", "units": "metres",
                   "root_motion": "translation on Hips; clips re-based so the smooth root starts at the origin",
                   "loops": "last key == first key + one stride of root motion (duplicate end frame)"},
               "elapsed_s": round(elapsed, 1), "generations": len(self.timings), "clips": self.clips}
        try:
            joins = self.continuity()
        except Exception as e:  # never lose the manifest over the report
            joins = [{"join": f"continuity check failed: {e}", "pose_diff_cm": None, "velocity_diff_cmf": None}]
        man["continuity"] = joins
        with open(os.path.join(self.out, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(man, f, indent=1)
        lines = [f"# Motion-matching set: {self.label}", "",
                 f"{len(self.clips)} clips, {len(self.timings)} generations, {elapsed / 60:.1f} min. Style subject: \"{self.subject}\", speed {self.speed} m/s.", "",
                 "| Clip | Frames | Status | Seed (attempts) | Path err mean (cm) | Heading err max (deg) | Pelvis max (m) | Skate (cm/s) | Notes |",
                 "|---|---|---|---|---|---|---|---|---|"]
        for c in self.clips:
            mt = c["metrics"]
            notes = []
            if c["loop"]:
                notes.append(f"seam {c['seam_pose_cm']:.2f} cm")
            if c["kind"] == "turn":
                notes.append(f"final {mt['final_heading_deg']:.1f} deg, overshoot {mt['overshoot_deg']:.1f}")
            fails = [x["check"] for x in c["checks"] if not x["ok"]]
            if fails:
                notes.append("failed: " + ", ".join(fails))
            lines.append(f"| `{c['name']}` | {c['frames']} | {'PASS' if c['passed'] else '**BEST-EFFORT**'} | {c['seed']} ({len(c['attempts'])}) | "
                         f"{mt.get('path_err_mean_cm', float('nan')):.1f} | {mt.get('heading_err_max_deg', float('nan')):.1f} | "
                         f"{mt['pelvis_max_m']:.2f} | {mt['skate_cms']:.1f} | {'; '.join(notes)} |")
        lines += ["", "## Posture per clip (style consistency)", "",
                  "| Clip | Pelvis h | Head h | Hands dist | Hands h | Hands fwd (cm) | Arm swing (cm) | Torso pitch (deg) |",
                  "|---|---|---|---|---|---|---|---|"]
        if self.carry_edit:
            lines.insert(len(lines) - 2, f"Values below are the EXPORTED clips, after the carry edit {self.carry_edit} (checks above ran on the generated carry).")
            lines.insert(len(lines) - 2, "")
        for c in self.clips:
            mt = c["final_posture"] if self.carry_edit else c["metrics"]
            lines.append(f"| `{c['name']}` | {mt['pelvis_h_m']:.2f} | {mt['head_h_m']:.2f} | {mt['hands_dist_m']:.2f} | {mt['hands_h_m']:.2f} | "
                         f"{mt['hands_fwd_cm']:.1f} | {mt['arm_swing_cm']:.1f} | {mt['torso_pitch_deg']:.1f} |")
        lines += ["", "## Continuity at the joins", "",
                  "Pose diff = mean joint distance in a pelvis-relative, heading-aligned frame; velocity diff = pelvis velocity "
                  "entering the first frame vs leaving the second (cm/frame; one frame of normal walking moves the pelvis ~4 cm).", "",
                  "| Join | Pose diff (cm) | Velocity diff (cm/frame) |", "|---|---|---|"]
        for j in joins:
            vd = "-" if j["velocity_diff_cmf"] is None else f"{j['velocity_diff_cmf']:.2f}"
            pd = "-" if j["pose_diff_cm"] is None else f"{j['pose_diff_cm']:.3f}"
            lines.append(f"| {j['join']} | {pd} | {vd} |")
        lines += ["", "By construction: Starts end exactly on their Loop's frame 0, Stops begin on it, and Starts, Stops, Turns "
                  "and the Idle loop share one idle pose (full-body pins).", ""]
        with open(os.path.join(self.out, "report.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--style", required=True)
    ap.add_argument("--styles", default=os.path.join(HERE, "styles.json"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--dirs", default="Fwd,Bwd,Left,Right")
    ap.add_argument("--turns", default="L90,R90,L180,R180")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--soma", action="store_true", help="also export GLBs with the SOMA body mesh (downloads upstream assets once)")
    ap.add_argument("--soma-assets", default=None, help="existing folder with skin_standard.npz etc. (implies --soma)")
    ap.add_argument("--min-free-mib", type=int, default=3000)
    ap.add_argument("--motion", default=DEFAULT_MOTION)
    ap.add_argument("--text", default=DEFAULT_TEXT)
    ap.add_argument("--regenerate", action="store_true", help="ignore cached generations in _work/")
    args = ap.parse_args()
    styles = json.load(open(args.styles, encoding="utf-8"))
    if args.style not in styles:
        raise SystemExit(f"unknown style {args.style}; available: {', '.join(k for k in styles if not k.startswith('_'))}")
    style = styles[args.style]
    if args.seed is None:
        args.seed = int(style.get("seed", 1))
    if args.out is None:
        args.out = os.path.join(REPO, "factory-output", args.style)
    if args.soma and not args.soma_assets:
        args.soma_assets = ensure_soma_assets(os.path.join(REPO, "factory-output", "_assets", "somaskel77"))
    if "demo-output" in os.path.normcase(os.path.abspath(args.out)):
        raise SystemExit("refusing to write into demo-output")
    dirs = [d for d in args.dirs.split(",") if d]
    if "Fwd" not in dirs:
        dirs = ["Fwd"] + dirs  # the forward loop is the style hero
    for d in dirs:
        if d not in C.DIRS:
            raise SystemExit(f"unknown direction {d}")
    turns = [t for t in args.turns.split(",") if t]
    fac = Factory(args, args.style, style)
    fac.loop_pose = {}
    t0 = time.perf_counter()
    fac.start()
    try:
        fac.log(f"[{args.style}] idle"); fac.idle()
        for d in dirs:
            fac.log(f"[{args.style}] loop {d}"); fac.loop(d)
        for d in dirs:
            fac.log(f"[{args.style}] start {d}"); fac.start_stop(d, "start")
            fac.log(f"[{args.style}] stop {d}"); fac.start_stop(d, "stop")
        for t in turns:
            fac.log(f"[{args.style}] turn {t}"); fac.turn(t)
    finally:
        if fac.gen:
            fac.gen.close()
        fac.write_outputs(time.perf_counter() - t0)
    n_ok = sum(c["passed"] for c in fac.clips)
    fac.log(f"done: {len(fac.clips)} clips ({n_ok} passed every check) in {(time.perf_counter() - t0) / 60:.1f} min -> {fac.out}")


if __name__ == "__main__":
    main()
