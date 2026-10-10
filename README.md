# kimodo.cpp

GGML/C++ implementation of NVIDIA's Kimodo text-to-motion model.

## Status

The five released Kimodo motion checkpoints accept either a UTF-8 prompt or a
precomputed LLM2Vec embedding and generate local rotations plus root
translations on CPU or Vulkan:

- SMPL-X RP v1: 22 joints
- SOMA RP/SEED v1.1: the predicted compact 30-joint control skeleton
- G1 RP/SEED v1: 34 Unitree G1 joints

NVIDIA's Python API expands SOMA's predicted 30 joints to a relaxed-hand
77-joint presentation skeleton. The native API currently returns the 30 joints
the model actually predicts. The text encoder uses eight-layer Vulkan chunks by
default; set `KIMODO_TEXT_LAYER_CHUNK=1..32` to tune VRAM use.

Included: checked GGUF loading, safetensors conversion, DDIM sampling, C/C++
APIs, conditioned multi-prompt transitions, kinematic constraints (full-body
keyframes, hand/foot end-effectors, 2D root waypoints and paths with optional
heading) with upstream `constraints.json` import/export, NVIDIA MotionCorrection
post-processing (x86/x86-64 only: the solver needs SSE4.1; configure with
`-DKIMODO_ENABLE_POSTPROCESS=OFF` to leave it out), CPU/Vulkan parity tests,
skeleton-only GLB export, selective LLM2Vec quantisation, divergence
reporting, a local web constraint editor and comparison viewer. 77-joint SOMA
output expansion, skinned-mesh GLB export, and motion-denoiser quantisation
are not implemented yet.

## Build and test on Linux

Install a C++23 compiler, CMake 3.25+, Ninja, Python 3 with the Hugging Face
CLI (`pip install huggingface_hub`), and the Vulkan loader/headers for Vulkan
support. GGML is a pinned Git submodule:

```sh
git submodule update --init --recursive
scripts/download_gguf_weights.sh --output "$PWD" --model soma-rp-v1.1
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

The standard test suite requires the local motion GGUF, text bundle, and
fixtures. It never downloads weights by itself. `release`, `asan-ubsan`, and
`fuzz` presets are also available.

Nix is optional and provides these dependencies reproducibly:

```sh
nix develop path:. --command cmake --preset debug
nix develop path:. --command cmake --build --preset debug
```

For sanitizer work:

```sh
nix develop path:. --command cmake --preset asan-ubsan
nix develop path:. --command cmake --build --preset asan-ubsan
nix develop path:. --command env \
  LD_LIBRARY_PATH="$PWD/build/asan-ubsan/ggml/src:$PWD/build/asan-ubsan/ggml/src/ggml-vulkan:$LD_LIBRARY_PATH" \
  ASAN_OPTIONS=detect_leaks=0:abort_on_error=1 UBSAN_OPTIONS=print_stacktrace=1 \
  ctest --preset asan-ubsan --output-on-failure
```

Leak detection is disabled because Vulkan loader/driver allocations are global
to the process. The GGUF parser fuzzer requires Clang.

## API

`include/kimodo/kimodo_capi.h` is the C API. Model loading checks the motion
GGUF and text model before inference. The text model can be a monolithic GGUF
beside `tokenizer.gguf` or the legacy component directory. Use
`kimodo_generate_embedding` for
4096 F32 values or `kimodo_generate` for text. Both return the selected model's
root translations and local XYZW rotations; query the joint count from the
result rather than assuming a fixed skeleton.

## Constraints

Constraints use NVIDIA Kimodo's `constraints.json` format: a JSON array of
`root2d`, `fullbody`, `left-hand`, `right-hand`, `left-foot`, `right-foot` and
`end-effector` objects with global 30 fps `frame_indices`, axis-angle
`local_joints_rot` on the model's control skeleton (SOMA files with 77 joints
are reduced to the 30 predicted joints), pelvis `root_positions` and optional
`smooth_root_2d`/`global_root_heading`. Files saved by the official demo or CLI
load unchanged; see `tests/data/constraints` for examples and
[`docs/IMPLEMENTATION.md`](docs/IMPLEMENTATION.md#constraints) for the exact
semantics. `kmd-generate` takes one JSON request:

```sh
cat > request.json <<'JSON'
{"output": "out/walk", "seed": 43, "steps": 100, "transition_frames": 5,
 "text_cfg": 2.0, "constraint_cfg": 2.0, "post_processing": true,
 "root_margin": 0.04,
 "segments": [{"frames": 150, "prompt": "A person walks forward and picks something up"}],
 "constraints_file": "tests/data/constraints/soma_03_full_body_keyframes.json"}
JSON
build/debug/kmd-generate models/kimodo-soma-rp-v1.1-f32.gguf Llama-3-Kimodo-Q8_0.gguf \
  --request request.json
```

It writes `root_positions.f32` [T,3], `local_rotations_xyzw.f32` [T,J,4],
`foot_contacts.f32` [T,4] and `smooth_root.f32` [T,3] (little-endian float32).
The same object (with `"constraints": [...]` inline if preferred) drives
`kimodo_generate_request` in the C API, `model::parse_request` plus
`model::generate` in C++, and JSON lines of `kmd-generate --server`. Omitted
fields keep their defaults; post-processing defaults to off in the API and is
never applied to G1, like upstream. In a build without the post-processor
(`kimodo_post_processing_available()` returns 0) a request that sets it fails
before any sampling.

## Demo

After building the debug preset and downloading the native GGUF bundle:

```sh
go run ./demo -addr 0.0.0.0:8094
```

On Windows, `start-kimodo-web.cmd` in the repository root starts the
`build-win` server (`build-win\kimodo-demo.exe -generator
build-win\kmd-generate.exe`) on `http://127.0.0.1:8094` and opens the
browser; close its window to stop the server.

Open `http://localhost:8094` for the web constraint editor described below.
Generated animations are kept in `demo-output/` and listed under "Past
animations". Every successful animation also writes a standalone
`animation.glb` beside its raw streams, for example
`demo-output/<animation-id>/animation.glb`. It contains the selected animated
node hierarchy (no mesh), ready to copy into a Three.js project. It is also
available from `/api/animations/<animation-id>/animation.glb` while the demo
is running.

The demo keeps all 32 layers of its default Q8 text encoder in VRAM for maximum
throughput, while executing them as bounded eight-layer GGML graphs. Its 10 GiB
residency ceiling makes the larger BF16 reference stream in bounded groups on a
16 GiB GPU. Library and command-line callers retain the lower-memory eight-layer
default; set `KIMODO_TEXT_LAYER_CHUNK=32` for residency and optionally set
`KIMODO_TEXT_RESIDENT_LIMIT_MIB` to enforce a VRAM-safe bundle-size ceiling.
The demo reuses one native worker while the selected motion model and text
quantization remain unchanged, preserving both weight sets across requests.
Profiling controls, measurements, and the next optimization targets are in
[`docs/PROFILING.md`](docs/PROFILING.md).

### Web constraint editor

The page follows NVIDIA's Kimodo demo:

- **Tabs.** *Generate* holds the motion model, text-encoder quantization,
  prompt segments, seed, diffusion steps, text and constraint CFG weights,
  transition frames, Post Processing (with its root margin), the constraint
  panel and the past animations. *Visualize* toggles the mesh, skeleton, foot
  contacts, labels and starting direction. *Instructions* explains the editor.
- **Timeline.** The Prompts row holds one block per text segment: drag a
  boundary or the last edge to change durations, Shift+click to split,
  right-click to delete or merge, double-click to edit the text. Seven tracks
  hold the constraints: **Full-Body**, **2D Root**, **Left/Right Hand**,
  **Left/Right Foot** and **Hips** (a generic `end-effector` on the root
  joint: it pins the pelvis position, hip height and pelvis rotation). Click a track to add a keyframe snapshotted from the
  displayed motion (keep the button held to drag it), Ctrl/Cmd-drag to add an
  interval or stretch an item, drag to move, right-click to delete. A hand or
  foot keyframe also fixes the pelvis position, hip height and facing of its
  stored pose. Constraints past the end of a shortened clip are kept (drawn
  faded) but not sent.
- **Editing Mode** shows the skeleton: click a joint for its rotation gizmo
  (Local or World space), drag the ground gizmo under the character to move
  the whole pose (W/E switch the hips between moving and rotating), and set an
  optional facing direction on a 2D Root waypoint. Edits update the
  constraints at that frame; Reset Constraint, Undo Move and Snap to
  Constraint are available.
- **Make Smooth Path** turns the 2D Root waypoints into a dense, smoothed root
  path anchored at the first and last frame.
- **Path presets** (*Browse presets…*) opens a "Parametric
  presets" dialog with 13 presets: Acceleration, Constant Speed,
  Strafe, Constant speed/Arc, Forward turn, Strafe turn, Start, Start turn,
  Stop, Stop turn, Spin and Turn in place, plus Run in a circle (a closed
  circle of a set radius and number of laps, running at 4 m/s by default).
  Each preset has a few parameters (speed, angle, side, timings, spring
  half-lives; radius and laps for the circle) and a live top-down preview. Apply
  turns it into constraints: a 2D Root path with a heading on every frame
  (simulated with critically damped springs), a Hips pin at frame 0 against
  floating starts, SOMA idle-pose pins for starts, stops and in-place turns,
  a matching prompt, Post Processing on and root margin 0.06. The constraints
  land on the timeline like any others, so they can be edited before
  generating. The pose pins need a SOMA model; Apply switches to one when it
  is installed. Apply leaves the seed, steps and CFG weights alone; the
  presets were measured with 100 steps and CFG 2/2
  ([experiment log](docs/motion-stage-research/09-experiment-log.md)).
  Run in a circle passed every GPU check wherever its dialog shows no
  warning. It warns above 4 m/s (the feet slide), above 90°/s of turn (the
  body lags the path) and on running circles wider than 5 m (they float).
- **Save / Load Constraints** read and write the official Kimodo
  `constraints.json`. SOMA files are written with the 30 control joints, like
  the shipped SOMA examples; 77-joint files saved by the official demo are
  accepted.
- **Post Processing** (foot-skate cleanup and exact constraint enforcement) is
  on by default and hidden for G1 models.
- Selecting a past animation restores its prompts, model, settings and
  constraints. Settings an older animation did not record (CFG weights, Post
  Processing, root margin) keep their current values.
- Tips: keep fewer than 20 constrained frames per constraint type (dense 2D
  root paths are fine) and at most 10 s per prompt.

Deviations from the upstream demo:

- There is no CFG-enable checkbox: the engine always runs separated text and
  constraint CFG, so only the two weights are exposed.
- SOMA hand-end joints (`*HandMiddleEnd`, `*HandThumbEnd`) are placed with the
  30-joint control skeleton's FK, like upstream's CLI. The Hugging Face demo
  uses 77-joint FK, so `*HandMiddleEnd` differs there by a few centimetres.
- G1 motions are never post-processed, like upstream.
- The Hips track and the path presets are additions; upstream's demo has
  neither (its `constraints.json` format already allows a Hips end-effector).

Editor development notes (serving it from the checkout with `-web-dir demo`,
the viewport harness, module overview) are in
[`demo/editor/README.md`](demo/editor/README.md). The demo's tests run with
`go test ./demo` and `node --test demo/editor/tests/`.

Quantisation comparisons produced by the workflow in
[`docs/QUANTIZATION.md`](docs/QUANTIZATION.md) can be opened at
`http://localhost:8094/compare`. Supply their parent directory with the demo's
`--comparisons` option. The comparison view overlays every variant on a shared
timeline and provides world-space and root-position-aligned modes.

## Weights

Ready-to-run native GGML weights are published under the Hugging Face
`LocalAI-io` organisation (not GitHub's `localai-org`). The reusable
[Llama-3-Kimodo-GGML](https://huggingface.co/LocalAI-io/Llama-3-Kimodo-GGML)
text encoder is separate from the four redistributable motion repositories,
each of which preserves a one-to-one relationship to its NVIDIA upstream:

- [Kimodo-SOMA-RP-v1.1-GGML](https://huggingface.co/LocalAI-io/Kimodo-SOMA-RP-v1.1-GGML)
- [Kimodo-SOMA-SEED-v1.1-GGML](https://huggingface.co/LocalAI-io/Kimodo-SOMA-SEED-v1.1-GGML)
- [Kimodo-G1-RP-v1-GGML](https://huggingface.co/LocalAI-io/Kimodo-G1-RP-v1-GGML)
- [Kimodo-G1-SEED-v1-GGML](https://huggingface.co/LocalAI-io/Kimodo-G1-SEED-v1-GGML)

Download one or repeat `--model` to install several:

```sh
scripts/download_gguf_weights.sh --output "$PWD" \
  --model soma-rp-v1.1 --model g1-rp-v1
```

Q8_0 is the default text encoder. Pass `--text-quantization` with `bf16`,
`q8_0`, `q6_k`, `q5_k`, `q4_k`, or `q4_k_m` to install another level. Each
level is one weight GGUF and all levels share `tokenizer.gguf`; the runtime can
still selectively stream layer tensors from the monolith. The installer
verifies each published manifest and SHA-256 hashes. Use
`--motion-only` when supplying a precomputed 4096-float LLM2Vec embedding.
SMPL-X RP is deliberately absent from the published-weight installer: its
internal-R&D licence prohibits distributing derivative models, so it must be
converted locally after the user obtains the upstream checkpoint under its
gated terms.

The text bundle includes converted Meta Llama 3 material and retains its
separate terms. Review every selected model card before downloading or
redistributing.

## License

The C++ port and its original tooling are licensed under Apache-2.0; see
[LICENSE](LICENSE). GGML and the model weights retain their respective
licences.

| Motion checkpoint | Upstream terms | Commercial use |
| --- | --- | --- |
| Kimodo-SMPLX-RP-v1 | [NVIDIA Internal Scientific Research and Development Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-internal-scientific-research-and-development-model-license/) | No; internal, non-production R&D only; derivative model redistribution is prohibited |
| SOMA RP/SEED v1.1 | [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) | Permitted by the model licence |
| G1 RP/SEED v1 | [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) | Permitted by the model licence |

The SMPL-X warning is about NVIDIA's trained Kimodo checkpoint, not the mere
fact that its output uses an SMPL-X-shaped 22-joint hierarchy. Converting that
checkpoint to GGUF is a new runtime representation of the same weights and does
not replace its licence. Skeleton names, parent links, and the Apache-2.0 port
source do not by themselves make the SOMA or G1 checkpoints non-commercial.
The SMPL-X Hugging Face metadata, model card, and access terms identify the
internal-R&D licence; treat those restrictive terms as controlling even though
an apparently inconsistent `LICENSE` file has also appeared in that upstream
repository.

### Regenerating the bundle

This is only needed to reproduce a conversion. The SMPL-X checkpoint and Llama
base model are gated. After accepting their Hugging Face licences and
authenticating, download the exact revisions and hash manifests with:

```sh
nix develop path:. --command hf auth login
scripts/download_weights.sh --output "$PWD/models" --with-text \
  --model smplx-rp-v1 --model soma-rp-v1.1 --model soma-seed-v1.1 \
  --model g1-rp-v1 --model g1-seed-v1
```

Convert the local LLM2Vec model to the native component bundle with:

```sh
nix develop path:. --command scripts/convert_llm2vec_bundle.sh \
  "$PWD/models/llama3-8b-instruct-base" "$PWD/generated/llm2vec-text-bundle"
```

Quantize and pack the release variants as described in
[`docs/QUANTIZATION.md`](docs/QUANTIZATION.md). `kmd-pack-text` streams tensor
payloads between files, so creating the BF16 monolith does not require enough
RAM to hold the complete encoder.

Validate a prospective release without network access, then explicitly upload
it from an account allowed to publish to `LocalAI-io`:

```sh
nix develop path:. --command python scripts/publish_gguf.py --component motion \
  --motion-model soma-rp-v1.1
nix develop path:. --command python scripts/publish_gguf.py --component motion \
  --motion-model soma-rp-v1.1 --upload --confirm-upstream-licences
nix develop path:. --command python scripts/publish_gguf.py --component text \
  --upload --confirm-upstream-licences
```
