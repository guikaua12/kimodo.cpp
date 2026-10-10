"""SOMA77 rig helpers: BVH parsing (text only) and bind-matrix analysis. No pickle loading."""
import os, re
import numpy as np

def parse_bvh(path):
    names, parents, offsets, stack = [], [], [], []
    for line in open(path):
        t = line.strip().split()
        if not t: continue
        if t[0] in ("ROOT", "JOINT"):
            names.append(t[1]); parents.append(stack[-1] if stack else -1); offsets.append(None); cur = len(names) - 1
            pending = cur
        elif t[0] == "End":
            pending = None
        elif t[0] == "{":
            stack.append(pending if pending is not None else -2)
        elif t[0] == "}":
            stack.pop()
        elif t[0] == "OFFSET" and pending is not None and offsets[pending] is None:
            offsets[pending] = [float(x) for x in t[1:4]]
        elif t[0] == "MOTION":
            break
    parents = [p if p >= 0 else -1 for p in parents]
    return names, parents, np.array(offsets) / 100.0

def load_skin(d):
    z = np.load(os.path.join(d, "skin_standard.npz"), allow_pickle=False)
    return {k: z[k] for k in z.files}
