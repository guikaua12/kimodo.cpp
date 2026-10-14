"""Extract the SOMA-30 data of the web editor's pose presets (demo/editor/poses.js).

Writes demo/editor/pose_library.js:
- arms: local rotations of the twelve arm joints under Chest (LeftShoulder ..
  RightHandMiddleEnd) for each rifle carry (the pose layer's targets), and
- bodies: whole poses in the preset_poses.js form {rot, pelvis}, turned so the
  hip line faces exactly +Z (engine heading 0): the hand-pin anchor and the
  crouch bodies.

Each carry is a git-tracked anchor under tools/mm_factory/anchors (a 2-frame
motion dir and anchor.json, built by tools/mm_factory/anchor.py from a
generated clip):
- rifle_collapsed_low_ready / rifle_collapsed_walk: Kimodo's own reading of
  "low ready" (anchors/rifle_low_ready_natural: "a man walks forward holding a
  rifle in a low ready position with the muzzle pointed down", seed 2, frame
  51). The rifle runs diagonally across the body, muzzle down-left: grip hand
  1.22 m at the chest, support hand 1.00 m at the left hip. This is the
  collapsed low ready.
- rifle_low_ready: the muzzle-forward low ready (anchors/rifle_low_ready_forward),
  arms only. Kimodo never generates this carry (probe/poses/report_low_ready.md),
  so it is authored from its standing high-port clip and only used as a layer
  target; the hand pins of both rifle poses use rifle_collapsed_walk.
- crouch_idle {prev, pose, next}, crouch_walk and crouch_walk_next (the next
  frame, for the fullbody walk pair), read from the crouch GPU research
  (docs/motion-stage-research/probe/poses/crouch_poses.json) when it exists:
  "A person stays in place, holding still." / "A person walks forward at a
  steady pace." with dense Hips pins at 0.66 / 0.70 m (no crouch words).

usage: python -I scripts/extract_pose_library.py [--anchors tools/mm_factory/anchors]
       [--crouch docs/motion-stage-research/probe/poses/crouch_poses.json]
       [--out demo/editor/pose_library.js]
"""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_preset_poses import J, axis_angle, heading, load, pose  # noqa: E402

ARM_JOINTS = ["LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "LeftHandThumbEnd", "LeftHandMiddleEnd",
              "RightShoulder", "RightArm", "RightForeArm", "RightHand", "RightHandThumbEnd", "RightHandMiddleEnd"]
ARM_INDICES = list(range(10, 22))  # src/skeleton.hpp soma30_names
# (arms key, body key, anchor dir, required)
CARRIES = [("rifle_collapsed_low_ready", "rifle_collapsed_walk", "rifle_low_ready_natural", True),
           ("rifle_low_ready", None, "rifle_low_ready_forward", True)]


def r6(v):
    return round(float(v), 6)


def arms(m, f=0):
    return [[r6(c) for c in axis_angle(m["rot"][f][j])] for j in ARM_INDICES]


def check_body(name, p, tolerance=0.05):
    """Re-derive the engine heading of a stored pose (~0 after the correction; companion
    frames such as prev/next share their key frame's correction, so they get a wider bound)."""
    v = np.asarray(p["rot"][0], np.float64)
    a = np.linalg.norm(v)
    q = np.array([0.0, 0.0, 0.0, 1.0]) if a < 1e-12 else np.concatenate([v / a * math.sin(a / 2), [math.cos(a / 2)]])
    h = math.degrees(heading(q))
    if abs(h) > tolerance:
        raise SystemExit(f"{name}: heading {h:+.3f} deg after correction (expected 0)")
    if len(p["rot"]) != J or len(p["pelvis"]) != 3:
        raise SystemExit(f"{name}: expected {J} rotations and a pelvis [dx, y, dz]")


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", default=os.path.join(here, "tools", "mm_factory", "anchors"))
    ap.add_argument("--crouch", default=os.path.join(here, "docs", "motion-stage-research", "probe", "poses", "crouch_poses.json"))
    ap.add_argument("--out", default=os.path.join(here, "demo", "editor", "pose_library.js"))
    args = ap.parse_args()

    data = {"arms": {"joints": ARM_JOINTS}, "bodies": {}}
    sources = []
    for arms_key, body_key, anchor, required in CARRIES:
        d = os.path.join(args.anchors, anchor)
        if not os.path.exists(os.path.join(d, "local_rotations_xyzw.f32")):
            if required:
                raise SystemExit(f"missing anchor {d}")
            print(f"no anchor {anchor}; {arms_key} left out")
            continue
        m = load(d)
        h = heading(m["rot"][0][0])
        data["arms"][arms_key] = arms(m)
        if body_key:
            data["bodies"][body_key] = pose(m, 0, -h)
        info = {}
        if os.path.exists(os.path.join(d, "anchor.json")):
            with open(os.path.join(d, "anchor.json"), encoding="utf-8") as fh:
                info = json.load(fh)
        src = str(info.get("source", "?")).replace("\\", "/").split("/")[-1]
        sources.append(f"// {arms_key}{' / ' + body_key if body_key else ' (arms only)'}: tools/mm_factory/anchors/{anchor} ({src}, frame {info.get('frame', '?')}).")
        print(f"{arms_key}: heading correction {math.degrees(-h):+.2f} deg" + (f", pelvis {data['bodies'][body_key]['pelvis']}" if body_key else ""))

    crouch_src = None
    if os.path.exists(args.crouch):
        with open(args.crouch, encoding="utf-8") as fh:
            crouch = json.load(fh)
        idle = crouch["crouch_idle"]
        data["bodies"]["crouch_idle"] = {k: {"rot": idle[k]["rot"], "pelvis": idle[k]["pelvis"]} for k in ("prev", "pose", "next")}
        for key in ("crouch_walk", "crouch_walk_next"):
            data["bodies"][key] = {"rot": crouch[key]["rot"], "pelvis": crouch[key]["pelvis"]}
        crouch_src = crouch.get("_provenance")
        print(f"crouch: idle pelvis {idle['pose']['pelvis']}, walk pelvis {crouch['crouch_walk']['pelvis']}")
    else:
        print(f"no crouch poses at {args.crouch}; crouch bodies left out")

    for name, body in data["bodies"].items():
        for k, p in (body.items() if "prev" in body else [("", body)]):
            companion = k in ("prev", "next") or name.endswith("_next")
            check_body(f"{name}{'.' + k if k else ''}", p, tolerance=1.0 if companion else 0.05)

    body = json.dumps(data, separators=(",", ":"))
    lines = [
        "// SOMA-30 data of the pose presets (poses.js).",
        "// GENERATED by scripts/extract_pose_library.py; do not edit by hand.",
        "//",
        "// arms: local axis-angle rotations of the twelve joints under Chest (joints",
        "// 10-21, names in `arms.joints`) for each rifle carry; bodies: whole poses",
        "// {rot: [30][3] local axis-angle (joint 0 = Hips, global), pelvis: [dx, y, dz]",
        "// (offset from the smooth root, height)}, turned so the hip line faces exactly",
        "// +Z (engine heading 0). rifle_collapsed_walk (Kimodo's own carry, mid-stride)",
        "// is the hand-pin anchor of both rifle poses; it is used only for hand and Hips pins.",
        *sources,
    ]
    if crouch_src:
        idle, walk = crouch_src["crouch_idle"], crouch_src["crouch_walk"]
        lines.append("// crouch_idle / crouch_walk / crouch_walk_next: docs/motion-stage-research/probe/poses/crouch_poses.json,")
        lines.append(f"// idle = {idle['source'].split('/')[-1]} frames {idle['frames']} (\"{idle['prompt']}\", seed {idle['seed']}),")
        lines.append(f"// walk = {walk['source'].split('/')[-1]} frames {walk['frame']}, {walk['next_frame']} (\"{walk['prompt']}\", seed {walk['seed']}).")
    lines.append(f"export const POSE_LIBRARY = Object.freeze({{ soma30: {body} }});")
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
