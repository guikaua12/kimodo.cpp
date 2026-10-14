# Constraint test data

The `*.json` files are unmodified copies of the `constraints.json` examples
shipped with NVIDIA Kimodo (https://github.com/nv-tlabs/kimodo, commit
`58e781898b3d7e328a676a75d3e338c45dce3ad9`), Copyright (c) 2026 NVIDIA
CORPORATION & AFFILIATES, licensed under the Apache License 2.0 (see the
repository `LICENSE`).

| file | upstream path |
|---|---|
| `soma_03_full_body_keyframes.json` | `kimodo/assets/demo/examples/kimodo-soma-rp/03_full_body_keyframes/constraints.json` |
| `soma_04_ee_constraint.json` | `kimodo/assets/demo/examples/kimodo-soma-rp/04_ee_constraint/constraints.json` |
| `soma_06_root_waypoints.json` | `kimodo/assets/demo/examples/kimodo-soma-rp/06_root_waypoints/constraints.json` |
| `soma_07_mixed_constraints.json` | `kimodo/assets/demo/examples/kimodo-soma-rp/07_mixed_constraints/constraints.json` |
| `g1_04_ee_constraint.json` | `kimodo/assets/demo/examples/kimodo-g1-rp/04_ee_constraint/constraints.json` |

The matching prompts and durations are in each example's `meta.json`
upstream. `kimodo-constraints-test` parses them, round-trips them through
`constraints_to_json` and checks their condition masks.
