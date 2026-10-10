"""Export a kimodo.cpp SOMA-30 motion as a GLB skinned with the real SOMA body mesh.

Mirrors upstream kimodo/viz/soma_skin.py: the 30 predicted local rotations are
written by name into the 77-joint rig, every other joint keeps the upstream
relaxed-hands rest pose, FK runs on the 77-joint neutral skeleton (offsets from
somaskel77_standard_tpose.bvh, which equal the bind-matrix-derived offsets) and
the mesh is linear-blend skinned with inverse(bind_rig_transform), all 8
influences per vertex (JOINTS_0/1, WEIGHTS_0/1).

Assets (upstream nv-tlabs/kimodo, Apache-2.0, kimodo/assets/skeletons/somaskel77):
skin_standard.npz, relaxed_hands_rest_pose.npy, somaskel77_standard_tpose.bvh.
No pickle files are read.

usage: python -I soma_glb.py ASSET_DIR MOTION_DIR OUT.glb [--name CLIP] [--fps 30]
"""
import argparse
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import soma77 as S  # noqa: E402

NAMES30 = ["Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head", "Jaw", "LeftEye", "RightEye",
           "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "LeftHandThumbEnd", "LeftHandMiddleEnd",
           "RightShoulder", "RightArm", "RightForeArm", "RightHand", "RightHandThumbEnd", "RightHandMiddleEnd",
           "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase", "RightLeg", "RightShin", "RightFoot", "RightToeBase"]


def mat_to_quat(m):
    """3x3 rotation -> xyzw quaternion."""
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def quat_to_mat(q):
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def load_rig(asset_dir):
    sk = S.load_skin(asset_dir)
    names = [str(n) for n in sk["rig_joint_names"]]
    parent = [-1] * len(names)
    for p, c in sk["rig_joint_connections"]:
        parent[int(c)] = int(p)
    bnames, bpar, boff = S.parse_bvh(os.path.join(asset_dir, "somaskel77_standard_tpose.bvh"))
    bi = {n: i for i, n in enumerate(bnames)}
    offsets = np.array([boff[bi[n]] for n in names])
    relaxed = np.load(os.path.join(asset_dir, "relaxed_hands_rest_pose.npy"), allow_pickle=False).astype(np.float64)
    return {"names": names, "parent": parent, "offsets": offsets, "relaxed": relaxed.reshape(len(names), 3, 3),
            "bind": sk["bind_rig_transform"].astype(np.float64), "vertices": sk["bind_vertices"].astype(np.float64),
            "faces": sk["faces"].astype(np.int64), "lbs_i": sk["lbs_indices"].astype(np.int64),
            "lbs_w": sk["lbs_weights"].astype(np.float64)}


def local_rots77(rig, rot30):
    """[T,30,4] xyzw -> [T,77,3,3] local rotation matrices (upstream to_SOMASkeleton77)."""
    T = rot30.shape[0]
    idx = [rig["names"].index(n) for n in NAMES30]
    R = np.repeat(rig["relaxed"][None], T, 0).copy()
    for k, j in enumerate(idx):
        R[:, j] = np.stack([quat_to_mat(q) for q in rot30[:, k]])
    return R


def fk77(rig, R, root):
    """Global rotations [T,77,3,3] and positions [T,77,3] (upstream batch_rigid_transform)."""
    T, J = R.shape[:2]
    G = np.empty_like(R); P = np.empty((T, J, 3))
    for j in range(J):
        p = rig["parent"][j]
        if p < 0:
            G[:, j] = R[:, j]; P[:, j] = root
        else:
            G[:, j] = G[:, p] @ R[:, j]
            P[:, j] = P[:, p] + np.einsum("tab,b->ta", G[:, p], rig["offsets"][j])
    return G, P


def skin_vertices(rig, G, P, frame):
    """Upstream SOMASkin.lbs for one frame -> [V,3]."""
    J = G.shape[1]
    X = np.tile(np.eye(4), (J, 1, 1)); X[:, :3, :3] = G[frame]; X[:, :3, 3] = P[frame]
    A = X @ np.linalg.inv(rig["bind"])
    v = np.concatenate([rig["vertices"], np.ones((len(rig["vertices"]), 1))], 1)
    out = np.zeros((len(v), 3))
    for k in range(rig["lbs_i"].shape[1]):
        Ak = A[rig["lbs_i"][:, k]]
        out += rig["lbs_w"][:, k:k + 1] * np.einsum("vab,vb->va", Ak[:, :3, :], v)
    return out


def write_glb(rig, root, rot30, out_path, clip_name, fps=30.0):
    T = root.shape[0]
    J = len(rig["names"])
    names = rig["names"]
    # local quaternions per joint, hemisphere-continuous
    R = local_rots77(rig, rot30)
    Q = np.zeros((T, J, 4))
    for j in range(J):
        for t in range(T):
            Q[t, j] = mat_to_quat(R[t, j])
            if t and np.dot(Q[t, j], Q[t - 1, j]) < 0:
                Q[t, j] = -Q[t, j]

    verts = rig["vertices"].astype(np.float32)
    faces = rig["faces"]
    # outward winding check + area-weighted normals
    fn = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]], verts[faces[:, 2]] - verts[faces[:, 0]])
    nrm = np.zeros_like(verts, dtype=np.float64)
    for k in range(3):
        np.add.at(nrm, faces[:, k], fn)
    centroid = verts.mean(0)
    if np.mean(np.einsum("va,va->v", nrm, verts - centroid)) < 0:
        faces = faces[:, [0, 2, 1]]; nrm = -nrm
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    lbs_i, lbs_w = rig["lbs_i"], rig["lbs_w"].copy()
    lbs_w /= lbs_w.sum(1, keepdims=True)
    lbs_i = np.where(lbs_w > 0, lbs_i, 0)
    ibm = np.linalg.inv(rig["bind"]).astype(np.float32)

    blob = bytearray(); views = []; accessors = []

    def add(data, target=None):
        while len(blob) % 4:
            blob.append(0)
        off = len(blob); b = data.tobytes(); blob.extend(b)
        v = {"buffer": 0, "byteOffset": off, "byteLength": len(b)}
        if target:
            v["target"] = target
        views.append(v)
        return len(views) - 1

    def acc(data, ctype, typ, minmax=False, target=None):
        view = add(data, target)
        count = data.shape[0]
        a = {"bufferView": view, "componentType": ctype, "count": count, "type": typ}
        if minmax:
            flat = data.reshape(count, -1)
            a["min"] = flat.min(0).astype(float).tolist(); a["max"] = flat.max(0).astype(float).tolist()
        accessors.append(a)
        return len(accessors) - 1

    a_pos = acc(verts, 5126, "VEC3", True, 34962)
    a_nrm = acc(nrm.astype(np.float32), 5126, "VEC3", False, 34962)
    a_j0 = acc(lbs_i[:, :4].astype(np.uint16), 5123, "VEC4", False, 34962)
    a_w0 = acc(lbs_w[:, :4].astype(np.float32), 5126, "VEC4", False, 34962)
    a_j1 = acc(lbs_i[:, 4:8].astype(np.uint16), 5123, "VEC4", False, 34962)
    a_w1 = acc(lbs_w[:, 4:8].astype(np.float32), 5126, "VEC4", False, 34962)
    idx_dtype, idx_ct = (np.uint16, 5123) if verts.shape[0] < 65536 else (np.uint32, 5125)
    a_idx = acc(faces.reshape(-1).astype(idx_dtype), idx_ct, "SCALAR", False, 34963)
    a_ibm = acc(np.ascontiguousarray(np.transpose(ibm, (0, 2, 1))).reshape(J, 16), 5126, "MAT4")
    a_time = acc((np.arange(T) / fps).astype(np.float32), 5126, "SCALAR", True)
    a_root = acc(root.astype(np.float32), 5126, "VEC3")
    a_rot = [acc(Q[:, j].astype(np.float32), 5126, "VEC4") for j in range(J)]

    nodes = []
    for j in range(J):
        node = {"name": names[j], "rotation": Q[0, j].astype(float).tolist()}
        node["translation"] = (root[0] if rig["parent"][j] < 0 else rig["offsets"][j]).astype(float).tolist()
        kids = [c for c in range(J) if rig["parent"][c] == j]
        if kids:
            node["children"] = kids
        nodes.append(node)
    mesh_node = len(nodes)
    nodes.append({"name": "SOMA_body", "mesh": 0, "skin": 0})
    hips = rig["parent"].index(-1)
    samplers = [{"input": a_time, "output": a_root, "interpolation": "LINEAR"}]
    channels = [{"sampler": 0, "target": {"node": hips, "path": "translation"}}]
    for j in range(J):
        samplers.append({"input": a_time, "output": a_rot[j], "interpolation": "LINEAR"})
        channels.append({"sampler": len(samplers) - 1, "target": {"node": j, "path": "rotation"}})
    gltf = {
        "asset": {"version": "2.0", "generator": "kimodo.cpp SOMA mesh exporter (upstream kimodo soma_skin.py semantics)"},
        "scene": 0, "scenes": [{"nodes": [hips, mesh_node]}], "nodes": nodes,
        "meshes": [{"name": "SOMA_body", "primitives": [{"attributes": {"POSITION": a_pos, "NORMAL": a_nrm,
                    "JOINTS_0": a_j0, "WEIGHTS_0": a_w0, "JOINTS_1": a_j1, "WEIGHTS_1": a_w1},
                    "indices": a_idx, "material": 0}]}],
        "materials": [{"name": "SOMA_skin", "pbrMetallicRoughness": {"baseColorFactor": [0.6, 0.74, 1.0, 1.0],
                       "metallicFactor": 0.0, "roughnessFactor": 0.7}}],
        "skins": [{"name": "SOMA77", "joints": list(range(J)), "inverseBindMatrices": a_ibm, "skeleton": hips}],
        "animations": [{"name": clip_name, "samplers": samplers, "channels": channels}],
        "accessors": accessors, "bufferViews": views, "buffers": [{"byteLength": len(blob)}],
    }
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * ((4 - len(js) % 4) % 4)
    while len(blob) % 4:
        blob.append(0)
    with open(out_path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(blob)))
        f.write(struct.pack("<II", len(js), 0x4E4F534A)); f.write(js)
        f.write(struct.pack("<II", len(blob), 0x004E4942)); f.write(bytes(blob))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("assets"); ap.add_argument("motion"); ap.add_argument("out")
    ap.add_argument("--name", default=None); ap.add_argument("--fps", type=float, default=30.0)
    a = ap.parse_args()
    root = np.fromfile(os.path.join(a.motion, "root_positions.f32"), "<f4").reshape(-1, 3).astype(np.float64)
    rot = np.fromfile(os.path.join(a.motion, "local_rotations_xyzw.f32"), "<f4").reshape(len(root), 30, 4).astype(np.float64)
    rig = load_rig(a.assets)
    write_glb(rig, root, rot, a.out, a.name or os.path.splitext(os.path.basename(a.out))[0], a.fps)
    print(f"wrote {a.out} ({len(root)} frames, {len(rig['names'])} joints, {len(rig['vertices'])} vertices)")


if __name__ == "__main__":
    main()
