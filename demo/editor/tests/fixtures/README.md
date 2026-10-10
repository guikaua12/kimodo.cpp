# Test fixtures

- `kimodo-soma-rp/*.constraints.json`: verbatim copies of
  `kimodo/assets/demo/examples/kimodo-soma-rp/<example>/constraints.json` from
  NVIDIA Kimodo (Apache-2.0, upstream commit `58e7818`). 30-joint SOMA files.
- `smooth_root_upstream.json`: reference outputs of upstream
  `kimodo/motion_rep/smooth_root.py` (`smooth_signal`, `TrajectorySmoother`, and the
  demo's linear-interpolation + smoothing dense path), computed in float64 (numpy 2.4,
  scipy 1.18) by `gen_smooth_root_upstream.py`, which imports the unmodified upstream
  file (torch stubbed out):
  `py -3 gen_smooth_root_upstream.py <upstream-kimodo clone> smooth_root_upstream.json`.
  Regenerate only if the upstream smoother changes.
- `soma30.mjs`: the soma30 `/api/models` entry (names/parents/offsets mirror
  `demo/skeletons_extra.go`; hips/chains from upstream `kimodo/skeleton/definitions.py`).
