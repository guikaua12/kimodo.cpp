"""Build a hand-anchor pose for prop carries: pick the most typical frame of a source clip and move the
hands with arm IK (offsets in the character frame: x left, y up, z forward).

usage: python -I tools/mm_factory/anchor.py SOURCE_MOTION_DIR OUT_DIR --right DX,DY,DZ --left DX,DY,DZ
Writes OUT_DIR/{root_positions,local_rotations_xyzw,smooth_root}.f32 (2 identical frames) + anchor.json.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import skeleton as S  # noqa: E402


def typical_frame(P, start=15):
    X = S.local_pose(P)[start:]
    H = X[:, [S.LHAND, S.RHAND]].reshape(len(X), -1)
    return start + int(np.argmin(np.linalg.norm(H - H.mean(0), axis=1)))


def build(src, right, left):
    m = S.load_motion(src)
    P, _ = S.fk(m["root"], m["rot"])
    f = typical_frame(P)
    h = S.heading(P[f:f + 1])[0]
    c, s = np.cos(h), np.sin(h)
    def world(off):  # character frame -> world (x left, z forward; heading h)
        return np.array([c * off[0] + s * off[2], off[1], -s * off[0] + c * off[2]])
    targets = {"right": P[f, S.RHAND] + world(right), "left": P[f, S.LHAND] + world(left)}
    q = S.ik_hands(m["root"][f], m["rot"][f], targets)
    P2, _ = S.fk(m["root"][f][None], q[None])
    err = {k: float(np.linalg.norm(P2[0, S.RHAND if k == "right" else S.LHAND] - v)) for k, v in targets.items()}
    smooth = m["smooth"][f] if m["smooth"] is not None else m["root"][f]
    return m, f, q, err, smooth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source"); ap.add_argument("out")
    ap.add_argument("--right", default="0,-0.2,0.02"); ap.add_argument("--left", default="0,-0.15,0.02")
    a = ap.parse_args()
    right = [float(x) for x in a.right.split(",")]; left = [float(x) for x in a.left.split(",")]
    m, f, q, err, smooth = build(a.source, right, left)
    root = m["root"][f]
    S.save_motion(a.out, np.stack([root, root]), np.stack([q, q]), None, np.stack([smooth, smooth]))
    P, _ = S.fk(np.stack([root, root]), np.stack([q, q]))
    post = S.posture(P, slice(0, None))
    info = {"source": os.path.abspath(a.source), "frame": f, "right_offset": right, "left_offset": left,
            "ik_error_m": err, "posture": post}
    json.dump(info, open(os.path.join(a.out, "anchor.json"), "w"), indent=1)
    print(json.dumps({"frame": f, "ik_error_mm": {k: round(v * 1000, 2) for k, v in err.items()},
                      **{k: round(post[k], 3) for k in ("rhand_h_m", "lhand_h_m", "hands_dist_m", "handline_down_deg",
                                                         "handline_fwd_cm", "handline_left_cm", "rhand_fwd_cm", "rhand_side_cm")}}, indent=1))


if __name__ == "__main__":
    main()
