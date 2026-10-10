"""SOMA-30 skeleton, FK, quaternion helpers and motion metrics for kimodo.cpp outputs.

Conventions (kimodo.cpp, checked on device 2026-10-09):
- Y up, 30 fps, frame 0 faces +Z, the character's left is +X.
- heading theta = atan2(d.z, -d.x) with d = RightLeg - LeftLeg; +theta turns LEFT;
  forward = (sin theta, 0, cos theta); global_root_heading = [cos theta, sin theta].
- Quaternions are xyzw; joint 0 (Hips) local rotation is global.
"""
import math
import os

import numpy as np

FPS = 30.0
NAMES = ["Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head", "Jaw",
         "LeftEye", "RightEye", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
         "LeftHandThumbEnd", "LeftHandMiddleEnd", "RightShoulder", "RightArm", "RightForeArm",
         "RightHand", "RightHandThumbEnd", "RightHandMiddleEnd", "LeftLeg", "LeftShin", "LeftFoot",
         "LeftToeBase", "RightLeg", "RightShin", "RightFoot", "RightToeBase"]
PARENTS = [-1, 0, 1, 2, 3, 4, 5, 6, 6, 6, 3, 10, 11, 12, 13, 13, 3, 16, 17, 18, 19, 19, 0, 22, 23, 24, 0, 26, 27, 28]
# Rest offsets from src/skeleton.hpp (metres); Hips offset is unused (root position is explicit).
OFFSETS = np.array([
    [0, 0, 0], [-.00013727, .0500376256, -.00053726669], [-1.86574103e-9, .0712530139, -.000298248546],
    [-5.75188398e-9, .0755006305, -.00815970992], [-.00181676517, .263112953, -.00553348292],
    [-2.85102231e-8, .0770939664, .0230258546], [-4.5975437e-8, .0612891595, .0195370861],
    [2.63687901e-5, .0047559225, .0309494062], [.0320638079, .0538020513, .0758688308],
    [-.0322244017, .05361869, .0755823359], [.0162165175, .232371641, .0511341324],
    [.149198457, 2.19397873e-8, -.0550232576], [.287393078, 2.50268389e-9, -2.58787737e-5],
    [.270939812, -7.06625108e-9, 2.60897248e-5], [.122686267, -.0322017573, .0483306876],
    [.190119595, -.00312878387, -.000339570373], [-.0138011824, .231803086, .0521415786],
    [-.150371962, 1.17387901e-7, -.0554560437], [-.287366393, 1.87628082e-8, -2.59709359e-5],
    [-.271336198, -1.16767401e-9, 2.61269368e-5], [-.122642483, -.0321145448, .0480403904],
    [-.190005945, -.00306615542, -.0003157343], [.10043214, -.0843452671, .0259565473],
    [-1e-8, -.432217537, -.00802912805], [1e-8, -.421550959, -.0348152298],
    [0, -.0505947206, .132315294], [-.10047278, -.0829525995, .0262031695],
    [1e-8, -.433622059, -.00805555828], [2e-8, -.421173943, -.0347839785],
    [-3.42907669e-9, -.0507960932, .132841956]], dtype=np.float64)
J = 30
HIPS, NECK1, HEAD, LHAND, RHAND = 0, 4, 6, 13, 19
LLEG, LFOOT, LTOE, RLEG, RFOOT, RTOE = 22, 24, 25, 26, 28, 29
FEET = [LFOOT, LTOE, RFOOT, RTOE]  # order of foot_contacts.f32


# ---------------------------------------------------------------- I/O

def load_motion(d):
    def rd(name, cols):
        p = os.path.join(d, name)
        if not os.path.exists(p):
            return None
        a = np.fromfile(p, dtype="<f4").astype(np.float64)
        return a.reshape(-1, *cols)
    root = rd("root_positions.f32", (3,))
    m = {"root": root, "rot": rd("local_rotations_xyzw.f32", (J, 4)),
         "contacts": rd("foot_contacts.f32", (4,)), "smooth": rd("smooth_root.f32", (3,))}
    m["T"] = len(root)
    return m


def save_motion(d, root, rot, contacts=None, smooth=None):
    os.makedirs(d, exist_ok=True)
    np.asarray(root, "<f4").tofile(os.path.join(d, "root_positions.f32"))
    np.asarray(rot, "<f4").tofile(os.path.join(d, "local_rotations_xyzw.f32"))
    if contacts is not None:
        np.asarray(contacts, "<f4").tofile(os.path.join(d, "foot_contacts.f32"))
    if smooth is not None:
        np.asarray(smooth, "<f4").tofile(os.path.join(d, "smooth_root.f32"))


# ---------------------------------------------------------------- quaternions (xyzw)

def qnorm(q):
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def qmul(a, b):
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz], -1)


def yaw_quat(theta):
    """Rotation by theta about +Y (positive theta turns the character LEFT)."""
    return np.array([0.0, math.sin(theta / 2), 0.0, math.cos(theta / 2)])


def quat_to_axis_angle(q):
    q = qnorm(np.asarray(q, np.float64))
    q = np.where(q[..., 3:4] < 0, -q, q)
    v = q[..., :3]
    s = np.linalg.norm(v, axis=-1, keepdims=True)
    ang = 2 * np.arctan2(s, q[..., 3:4])
    with np.errstate(invalid="ignore", divide="ignore"):
        axis = np.where(s > 1e-12, v / np.maximum(s, 1e-12), 0.0)
    return axis * ang


def quat_to_mat(q):
    q = qnorm(q)
    x, y, z, w = np.moveaxis(q, -1, 0)
    m = np.empty(q.shape[:-1] + (3, 3))
    m[..., 0, 0] = 1 - 2 * (y * y + z * z); m[..., 0, 1] = 2 * (x * y - z * w); m[..., 0, 2] = 2 * (x * z + y * w)
    m[..., 1, 0] = 2 * (x * y + z * w); m[..., 1, 1] = 1 - 2 * (x * x + z * z); m[..., 1, 2] = 2 * (y * z - x * w)
    m[..., 2, 0] = 2 * (x * z - y * w); m[..., 2, 1] = 2 * (y * z + x * w); m[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return m


def rot_y(xz, theta):
    """Rotate xz points about the origin like R_y(theta) (maps +Z to (sin, cos))."""
    c, s = math.cos(theta), math.sin(theta)
    xz = np.asarray(xz, np.float64)
    return np.stack([c * xz[..., 0] + s * xz[..., 1], -s * xz[..., 0] + c * xz[..., 1]], -1)


# ---------------------------------------------------------------- FK and metrics

def fk(root, rot):
    T = len(root)
    L = quat_to_mat(rot)
    G = np.empty_like(L)
    P = np.empty((T, J, 3))
    for j in range(J):
        p = PARENTS[j]
        if p < 0:
            G[:, j] = L[:, j]; P[:, j] = root
        else:
            G[:, j] = G[:, p] @ L[:, j]
            P[:, j] = P[:, p] + np.einsum("tab,b->ta", G[:, p], OFFSETS[j])
    return P, G


def heading(P):
    d = P[:, RLEG] - P[:, LLEG]
    return np.arctan2(d[:, 2], -d[:, 0])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def local_pose(P):
    """Joint positions relative to the pelvis xz, rotated so the character faces +Z."""
    h = heading(P)
    c, s = np.cos(-h), np.sin(-h)
    X = P.copy()
    X[:, :, 0] -= P[:, :1, 0]; X[:, :, 2] -= P[:, :1, 2]
    x, z = X[:, :, 0].copy(), X[:, :, 2].copy()
    X[:, :, 0] = c[:, None] * x + s[:, None] * z
    X[:, :, 2] = -s[:, None] * x + c[:, None] * z
    return X


def posture(P, sl=slice(None)):
    """Style-bearing posture metrics over frames `sl` (means)."""
    h = heading(P)
    T = len(P)
    fwd = np.stack([np.sin(h), np.zeros(T), np.cos(h)], 1)
    pel = P[:, HIPS]
    r = {}
    r["pelvis_h_m"] = float(pel[sl, 1].mean())
    r["head_h_m"] = float(P[sl, HEAD, 1].mean())
    r["hands_dist_m"] = float(np.linalg.norm(P[sl, LHAND] - P[sl, RHAND], axis=1).mean())
    r["hands_h_m"] = float(0.5 * (P[sl, LHAND, 1] + P[sl, RHAND, 1]).mean())
    sw, fo = [], []
    for hj in (LHAND, RHAND):
        a = np.einsum("tj,tj->t", P[:, hj] - pel, fwd)[sl]
        sw.append(a.std()); fo.append(a.mean())
    r["arm_swing_cm"] = float(np.mean(sw) * 100)
    r["hands_fwd_cm"] = float(np.mean(fo) * 100)
    v = P[:, NECK1] - pel
    r["torso_pitch_deg"] = float(np.degrees(np.arctan2(np.einsum("tj,tj->t", v, fwd), v[:, 1]))[sl].mean())
    # Two-handed prop proxy: the right-hand -> left-hand line in the character frame (x left, y up, z forward).
    # For a right-handed rifle hold the line runs from the grip (right hand) toward the muzzle (support hand).
    X = local_pose(P)[sl]
    r["rhand_h_m"] = float(P[sl, RHAND, 1].mean())
    r["lhand_h_m"] = float(P[sl, LHAND, 1].mean())
    d = X[:, LHAND] - X[:, RHAND]
    r["handline_fwd_cm"] = float(d[:, 2].mean() * 100)
    r["handline_left_cm"] = float(d[:, 0].mean() * 100)
    r["handline_down_deg"] = float(np.degrees(np.arctan2(-d[:, 1], np.linalg.norm(d[:, [0, 2]], axis=1))).mean())
    r["rhand_fwd_cm"] = float(X[:, RHAND, 2].mean() * 100)
    r["feet_width_cm"] = float(np.abs(X[:, LFOOT, 0] - X[:, RFOOT, 0]).mean() * 100)
    r["rhand_side_cm"] = float(X[:, RHAND, 0].mean() * 100)
    return r


def skate_cms(P, contacts, sl=slice(None)):
    T = len(P)
    fv = np.zeros((T, 4))
    for k, j in enumerate(FEET):
        fv[1:, k] = np.linalg.norm(np.diff(P[:, j][:, [0, 2]], axis=0), axis=1) * FPS
    c = contacts > 0.5
    fv, c = fv[sl], c[sl]
    return float(fv[c].mean() * 100) if c.any() else float("nan")


def best_cycle(P, Lmin=24, Lmax=90, edge=10):
    """Window [a, a+L) whose wrap (pose(a+L) vs pose(a), plus velocity) matches best."""
    X = local_pose(P)
    T = len(X)
    best = None
    for a in range(edge, T - edge):
        for L in range(Lmin, Lmax + 1):
            b = a + L
            if b + 1 > T - edge:
                break
            pe = np.linalg.norm(X[b] - X[a], axis=1).mean()
            ve = np.linalg.norm((X[b] - X[b - 1]) - (X[a + 1] - X[a]), axis=1).mean()
            sc = pe + 2 * ve
            if best is None or sc < best[0]:
                best = (sc, a, L, pe, ve)
    return {"a": best[1], "L": best[2], "seam_pose_cm": best[3] * 100, "seam_vel_cmf": best[4] * 100}


def calmest_frame(P, edge=15):
    """Frame with the least joint motion (for picking an idle pose)."""
    X = local_pose(P)
    sp = np.linalg.norm(np.diff(X, axis=0), axis=2).mean(1)
    sp = np.convolve(sp, np.ones(7) / 7, "same")
    lo, hi = edge, len(sp) - edge
    return int(lo + np.argmin(sp[lo:hi]))


# ---------------------------------------------------------------- pose editing (arm IK)

def mat_to_quat(m):
    """3x3 rotation -> xyzw quaternion."""
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def rot_between(u, v):
    """Minimal rotation matrix taking direction u to direction v."""
    u = u / np.linalg.norm(u); v = v / np.linalg.norm(v)
    c = float(np.dot(u, v)); w = np.cross(u, v); s = np.linalg.norm(w)
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        a = np.cross(u, [1.0, 0, 0]) if abs(u[0]) < 0.9 else np.cross(u, [0, 1.0, 0])
        a /= np.linalg.norm(a)
        return 2 * np.outer(a, a) - np.eye(3)
    k = w / s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    ang = math.atan2(s, c)
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K)


ARMS = {"left": (10, 11, 12, 13), "right": (16, 17, 18, 19)}  # shoulder, arm (upper-arm joint), forearm (elbow), hand (wrist)


def ik_hands(root, quats, targets):
    """Move wrists to world targets with analytic two-bone IK, keeping elbow plane and hand global rotation.
    root [3], quats [30,4] xyzw local; targets {"left": xyz, "right": xyz}. Returns new local quats [30,4]."""
    q = quats.copy()
    for side, tgt in targets.items():
        sh, arm, fore, hand = ARMS[side]
        P, G = fk(root[None], q[None])
        P, G = P[0], G[0]
        S_, E, W = P[arm], P[fore], P[hand]
        Wt = np.asarray(tgt, np.float64)
        a, b = np.linalg.norm(E - S_), np.linalg.norm(W - E)
        dvec = Wt - S_
        d = float(np.clip(np.linalg.norm(dvec), abs(a - b) + 1e-4, a + b - 1e-4))
        dirn = dvec / np.linalg.norm(dvec)
        n = np.cross(W - S_, E - S_)
        if np.linalg.norm(n) < 1e-8:
            n = np.cross(dirn, [0, 1.0, 0])
        n /= np.linalg.norm(n)
        perp = np.cross(n, dirn); perp /= np.linalg.norm(perp)
        if np.dot(perp, E - S_) < 0:
            perp = -perp
        cos_a = (a * a + d * d - b * b) / (2 * a * d)
        sin_a = math.sqrt(max(0.0, 1 - cos_a * cos_a))
        Et = S_ + dirn * a * cos_a + perp * a * sin_a
        Wt_eff = S_ + dirn * d
        G_arm = rot_between(E - S_, Et - S_) @ G[arm]
        G_fore_tmp = G_arm @ quat_to_mat(q[fore])
        cur = G_fore_tmp @ OFFSETS[hand]
        G_fore = rot_between(cur, Wt_eff - Et) @ G_fore_tmp
        G_hand = G[hand]  # keep the hand's world orientation (grip on the prop)
        q[arm] = mat_to_quat(G[sh].T @ G_arm)
        q[fore] = mat_to_quat(G_arm.T @ G_fore)
        q[hand] = mat_to_quat(G_fore.T @ G_hand)
    return q
