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
- `pose_layer.json`: inputs and expected outputs of the pose layer (`poses.js` `applyPoseLayer`),
  written by `gen_pose_layer.mjs` (`node demo/editor/tests/fixtures/gen_pose_layer.mjs`). The Go twin's test
  (`demo/pose_layer_test.go`) checks the same numbers, so regenerate it only when the layer maths change on both sides.
