"""Root paths and kimodo constraint builders for motion-matching clips.

All paths are canonical: frame 0 at the origin facing +Z (kimodo ignores a
non-zero frame-0 heading). Facing is held at 0 for every locomotion clip, so
strafes and backward walks keep facing forward; only turn-in-place clips
change heading.
"""
import math

import numpy as np

from skeleton import FPS, quat_to_axis_angle, qmul, rot_y, yaw_quat

S2 = math.sqrt(0.5)
# Movement direction in xz (the character's left is +X when facing +Z).
DIRS = {"Fwd": (0.0, 1.0), "Bwd": (0.0, -1.0), "Left": (1.0, 0.0), "Right": (-1.0, 0.0),
        "FwdLeft": (S2, S2), "FwdRight": (-S2, S2), "BwdLeft": (S2, -S2), "BwdRight": (-S2, -S2)}
DIR_WORDS = {"Fwd": "forward", "Bwd": "backward", "Left": "sideways to his left", "Right": "sideways to his right",
             "FwdLeft": "diagonally forward and to the left", "FwdRight": "diagonally forward and to the right",
             "BwdLeft": "diagonally backward and to the left", "BwdRight": "diagonally backward and to the right"}
# Default speed factor per direction (people walk slower backward and sideways).
SPEED_FACTOR = {"Fwd": 1.0, "Bwd": 0.8, "Left": 0.85, "Right": 0.85,
                "FwdLeft": 0.9, "FwdRight": 0.9, "BwdLeft": 0.8, "BwdRight": 0.8}


# ---------------------------------------------------------------- paths

def straight(v, d, T):
    t = np.arange(T)[:, None] / FPS
    return t * v * np.asarray(d)[None]


def spring_speed(T, t0, halflife, v0, v1):
    """Speed per frame: v0 until frame t0, then a critically damped approach to v1
    (Holden-style spring with zero initial acceleration)."""
    y = 2 * math.log(2) / halflife
    s = np.full(T, float(v0))
    tau = (np.arange(T) - t0) / FPS
    m = tau > 0
    s[m] = v1 + (v0 - v1) * (1 + y * tau[m]) * np.exp(-y * tau[m])
    return s


def path_from_speed(speed, d):
    step = 0.5 * (speed[1:] + speed[:-1]) / FPS
    dist = np.concatenate([[0.0], np.cumsum(step)])
    return dist[:, None] * np.asarray(d)[None]


def eased_heading(T, t0, t1, theta):
    """0 until t0, cosine ease to theta at t1, theta after."""
    h = np.zeros(T)
    for t in range(T):
        if t <= t0:
            h[t] = 0.0
        elif t >= t1:
            h[t] = theta
        else:
            u = (t - t0) / (t1 - t0)
            h[t] = theta * 0.5 * (1 - math.cos(math.pi * u))
    return h


# ---------------------------------------------------------------- constraints

def root2d(xz, headings=None, frames=None):
    xz = np.asarray(xz, np.float64)
    frames = list(range(len(xz))) if frames is None else list(frames)
    c = {"type": "root2d", "frame_indices": frames, "smooth_root_2d": [[float(xz[f, 0]), float(xz[f, 1])] for f in frames]}
    if headings is not None:
        c["global_root_heading"] = [[math.cos(headings[f]), math.sin(headings[f])] for f in frames]
    return c


def pose_from(motion, f):
    """A pose sampled from a generated clip: local quats, pelvis position, smooth root."""
    smooth = motion["smooth"][f] if motion["smooth"] is not None else motion["root"][f]
    return {"q": motion["rot"][f].copy(), "pelvis": motion["root"][f].copy(), "smooth_xz": smooth[[0, 2]].copy()}


def place(pose, target_xz, yaw=0.0):
    """Put a sampled pose at a smooth-root target, rotated by yaw about +Y."""
    q = pose["q"].copy()
    off = pose["pelvis"][[0, 2]] - pose["smooth_xz"]
    if yaw:
        q[0] = qmul(yaw_quat(yaw), q[0])
        off = rot_y(off, yaw)
    return {"aa": quat_to_axis_angle(q), "root": [float(target_xz[0] + off[0]), float(pose["pelvis"][1]), float(target_xz[1] + off[1])],
            "smooth": [float(target_xz[0]), float(target_xz[1])]}


def fullbody(frames, placed):
    return {"type": "fullbody", "frame_indices": list(frames),
            "local_joints_rot": [p["aa"].tolist() for p in placed],
            "root_positions": [p["root"] for p in placed],
            "smooth_root_2d": [p["smooth"] for p in placed]}


def hips_pin(frame, placed):
    """Frame-0 Hips end-effector: pins root height (prevents floating starts)."""
    return {"type": "end-effector", "joint_names": ["Hips"], "frame_indices": [frame],
            "local_joints_rot": [placed["aa"].tolist()], "root_positions": [placed["root"]],
            "smooth_root_2d": [placed["smooth"]]}


def request(output, prompt, frames, seed, constraints, style, steps=100):
    return {"output": output.replace("\\", "/"), "seed": int(seed), "steps": steps, "transition_frames": 5,
            "text_cfg": float(style.get("text_cfg", 2.0)), "constraint_cfg": float(style.get("constraint_cfg", 2.0)),
            "post_processing": True, "root_margin": float(style.get("root_margin", 0.06)),
            "segments": [{"frames": int(frames), "prompt": prompt}], "constraints": constraints}
