"""Screen prompt wordings x seeds for a style signature before building a set.

Each candidate walks forward on a dense straight path (heading 0) and is scored
against a style's signature from styles.json (or --signature-from). Prints a
ranked table and writes screen.json; the best (prompt, seed) is what you put in
styles.json as subject/prompt_loop + seed.

usage: python -I tools/mm_factory/screen_styles.py --style rifle_low_ready --seeds 1,2,3
         --prompt "a man walks forward holding a rifle ..." [--prompt ...] [--speed 1.1] [--out DIR]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import clips as C  # noqa: E402
import skeleton as S  # noqa: E402
from factory import DEFAULT_MOTION, DEFAULT_TEXT, REPO, Generator, free_vram_mib  # noqa: E402


def range_distance(v, lo, hi):
    if lo is not None and v < lo:
        return lo - v
    if hi is not None and v > hi:
        return v - hi
    return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", required=True)
    ap.add_argument("--styles", default=os.path.join(HERE, "styles.json"))
    ap.add_argument("--prompt", action="append", required=True)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--speed", type=float, default=None)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    style = json.load(open(a.styles, encoding="utf-8"))[a.style]
    sig = style.get("signature", {}).get("all", [])
    v = a.speed if a.speed is not None else float(style.get("speed", 1.2))
    out = os.path.abspath(a.out or os.path.join(REPO, "factory-output", "_screen", a.style))
    os.makedirs(out, exist_ok=True)
    os.environ["MMF_LOG_DIR"] = out
    print(f"VRAM free: {free_vram_mib()} MiB")
    gen = Generator(DEFAULT_MOTION, DEFAULT_TEXT)
    xz = C.straight(v, C.DIRS["Fwd"], a.frames); hd = np.zeros(a.frames)
    rows = []
    try:
        for pi, prompt in enumerate(a.prompt):
            for seed in [int(s) for s in a.seeds.split(",") if s]:
                d = os.path.join(out, f"p{pi}_s{seed}")
                req = C.request(d, prompt, a.frames, seed, [C.root2d(xz, hd)], style)
                if not os.path.exists(os.path.join(d, "local_rotations_xyzw.f32")):
                    ok, ans, dt = gen.generate(req)
                    if not ok:
                        print("ERR", prompt, seed, ans); continue
                m = S.load_motion(d); P, _ = S.fk(m["root"], m["rot"])
                post = S.posture(P, slice(15, None))
                fails = [c["metric"] for c in sig if range_distance(post[c["metric"]], c.get("min"), c.get("max")) > 0]
                dist = sum(range_distance(post[c["metric"]], c.get("min"), c.get("max")) / max(1e-6, abs(c.get("max", c.get("min", 1)) or 1)) for c in sig)
                floating = float(P[:, 0, 1].max()) > 1.15
                rows.append({"prompt_index": pi, "prompt": prompt, "seed": seed, "dir": d, "fails": fails, "distance": dist,
                             "floating": floating, "posture": post})
                print(f"p{pi} s{seed}: {'PASS' if not fails else 'fail ' + ','.join(fails)}"
                      f"  rh {post['rhand_h_m']:.2f} lh {post['lhand_h_m']:.2f} down {post['handline_down_deg']:.0f} fwd {post['handline_fwd_cm']:.0f}"
                      f" dist {post['hands_dist_m']:.2f} swing {post['arm_swing_cm']:.1f} pitch {post['torso_pitch_deg']:.0f}", flush=True)
    finally:
        gen.close()
    rows.sort(key=lambda r: (r["floating"], len(r["fails"]), r["distance"]))
    json.dump({"style": a.style, "speed": v, "signature": sig, "ranked": rows}, open(os.path.join(out, "screen.json"), "w"), indent=1)
    print("\nranked:")
    for r in rows:
        print(f"  fails={len(r['fails'])} dist={r['distance']:.3f} p{r['prompt_index']} s{r['seed']}: {r['prompt']}")


if __name__ == "__main__":
    main()
