# mm_factory: one-click motion-matching sets from Kimodo (prototype)

`factory.py` generates a small locomotion set for one style. Every clip is phase-consistent with the others and every join is exact. It drives `build-win\kmd-generate.exe --server` directly, so nothing goes through the web demo and nothing lands in `demo-output/`.

```bat
:: one click: double-click (asks for a style), or
tools\mm_factory\make-mm-set.cmd zombie

:: or directly
python -I tools\mm_factory\factory.py --style zombie --soma
python -I tools\mm_factory\factory.py --style rifle --dirs Fwd,Bwd,Left,Right,FwdLeft,FwdRight --turns L90,R90,L180,R180
```

Default set: 17 clips from 22 generations, about 6–10 minutes on an RTX 5070 with Unreal open. Each generation adds about 1.9 GB of VRAM.

| Category | Clips | How it is made |
|---|---|---|
| `Idle/` | `<Style>_Idle_Loop` | A standing clip; its calmest frame becomes **the idle pose**. Regenerated with that pose pinned on frames 0, 1 and T−2, T−1, so the loop closes exactly |
| `Loops/` | `<Style>_Walk_<Dir>_Loop` for Fwd, Bwd, Left, Right (diagonals optional) | Two passes ([06-seamless-loops.md](../../docs/motion-stage-research/06-seamless-loops.md)). Pass 1: a free walk on a dense straight path with the heading held at 0, from which the best gait cycle is picked. Pass 2: regenerated with that cycle's start pose pinned at both ends. The last key equals the first key plus one stride |
| `Starts/` | `<Style>_Walk_<Dir>_Start` | Idle pose on frames 0–1, a critically damped speed ramp (half-life 0.3 s), and the loop's frames −1 and 0 pinned at the end. **The last frame is exactly the loop's first frame** |
| `Stops/` | `<Style>_Walk_<Dir>_Stop` | The loop's frames 0–1 at the start, a braking ramp, and the idle pose at the end. **The first frame is exactly the loop's first frame; the last frame is exactly idle frame 0** |
| `TurnInPlace/` | `<Style>_TurnInPlace_L90/R90/L180/R180` | Verified turn-in-place recipe: idle pose pinned at the start; idle pose rotated by the turn angle at T−14, T−2 and T−1; dense heading eased from 0 to the angle. **It ends on the idle pose, rotated** |
| `SOMA/…` (with `--soma`) | The same clips | Skinned with the real SOMA body mesh ([07](../../docs/motion-stage-research/07-soma-mesh-export-and-blender.md)). The assets are downloaded once from upstream Kimodo (Apache-2.0) into `factory-output/_assets/` and checked against fixed hashes |

All locomotion keeps facing +Z: strafes and backward walks hold the heading constraint on every frame. Shared seeds and shared pinned poses keep the style consistent across clips. The style comes from `styles.json` (prompt subject, speed, signature checks).

## Checks and regeneration

Every generation is measured; if a check fails, the clip is regenerated with seed + 101·k, up to `--retries` (default 3). The best attempt is kept and marked `BEST-EFFORT` in the report.

| Check | Default | Why |
|---|---|---|
| Pelvis max | ≤ 1.15 m | Catches the "floating start" failure (pelvis 1.8–2.3 m) |
| Lowest foot at frame 0 | ≤ 0.10 m | Same failure |
| Foot skate on contact frames | ≤ 15 cm/s | Zombie is about 8 cm/s by style |
| Path error mean (pelvis vs requested path) | ≤ 7 cm | Post-processing clamps the pelvis to `root_margin` (0.06) anyway |
| Heading error max (loops, starts, stops) | ≤ 12° | Strafes and backward walks must keep facing forward |
| Turn final heading error / overshoot | ≤ 6° / ≤ 25° | Text alone overshoots turns |
| Style signature (loops) | Per style in `styles.json` | Example: rifle hands ≤ 0.40 m apart, ≥ 1.05 m high, in front of the body. This rejects the measured "dropped rifle" seed |
| Similarity to the hero Fwd loop (other loops) | Pelvis h ±0.08, head h ±0.10, hands dist and height ±0.15 | Seed variance is the main measured style risk |

`report.md` lists every clip, its checks, posture metrics and **a continuity table** of pose and velocity differences at every join (loop wraps, start → loop, loop → stop, stop → idle, idle → turn → idle). `manifest.json` carries everything machine-readable: files, frames, seeds, attempts, prompts, root displacement, net yaw, metrics, checks, links and continuity.

Results are cached in `_work/<clip>/seed<N>/` together with their `request.json`. A rerun only regenerates clips whose request changed, and `--regenerate` forces all of them.

## Conventions of the exported clips

- 30 fps, Y-up, metres; frame 0 faces +Z; the character's left is +X.
- Root motion (translation and yaw) lives on the `Hips` bone, the same as `scripts/export_glb.py` and the existing Unreal import pipeline (`docs/how_to_setup_with_unreal_in_windows.md`). Each clip is re-based so its smooth root starts at the origin.
- Loops include the duplicate end frame (last = first + one stride), the convention Unreal's looping expects.

## Styles

Edit `styles.json`. The fields are:

| Field | Meaning |
|---|---|
| `subject` | Spliced into the prompt templates, e.g. "a zombie walks sideways to his left". Keep speed words out: the path sets the speed |
| `speed` | Forward loop speed in m/s. Other directions use `clips.SPEED_FACTOR` (Bwd 0.8, strafes 0.85, diagonals 0.8–0.9) unless `speed_factor` overrides it |
| `seed` | Base seed |
| `signature` | `all` / `any` posture checks |
| `checks` | Overrides of the defaults above |
| `similarity` | Overrides of the hero-similarity tolerances |
| `prompt_idle`, `prompt_loop`, `prompt_start`, `prompt_stop`, `prompt_turn` | Template overrides |
| `start_halflife`, `stop_halflife` | Speed-ramp half-lives |

Measured natural speeds are a guide: a free zombie walks at about 0.57 m/s ([09](../../docs/motion-stage-research/09-experiment-log.md)).

## Prop carries (rifle poses): screen → anchor → carry edit

Kimodo has no props, so a carry such as a rifle at **collapsed low ready** can only show in the hands. Text alone does not produce it: Kimodo's "low ready" keeps the grip hand at chest height (about 1.22 m). This is how the `rifle_low_ready` style was built (2026-10-10):

1. **Screen** wordings × seeds against a numeric signature, then render the best by eye. The numbers alone miss "hands clasped behind the back".
   ```bat
   python -I tools\mm_factory\screen_styles.py --style rifle_low_ready --seeds 1,2,3 --prompt "a man walks forward holding a rifle in a low ready position with the muzzle pointed down" --prompt "..."
   "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" -b --factory-startup --python tools\mm_factory\preview_blender.py -- OUT 6 front CLIP.glb
   ```
   The signature uses the hand-line metrics from `skeleton.posture`:

   | Metric | Meaning |
   |---|---|
   | `rhand_h_m`, `lhand_h_m` | Hand heights |
   | `handline_down_deg`, `handline_fwd_cm`, `handline_left_cm` | Direction of the right-hand → left-hand line, which stands in for the rifle |
   | `feet_width_cm` | Stance width |

2. **Anchor.** Take the most typical frame of the best walk (`anchor.py`) and pin both hands to it every 8 frames in every clip (`hand_anchor`). This keeps the carry identical across the set.
   - Pin Kimodo's *own* carry. Pinning hands lower than Kimodo likes makes it reach them by leaning about 20° forward.
   - On standing frames, lift the anchor to standing pelvis height (`idle_pelvis_h`). Otherwise Kimodo spreads the feet to about 50 cm and bows the head to hit the walking-height pins.
   - Starts and stops blend the lift by speed.
3. **Carry edit at export** (`carry_edit`): a deterministic two-bone arm IK layer applied to every frame.
   - The grip hand moves −18 cm (to belt height) and the support hand −9 cm (to the hip).
   - Spine, legs and root are untouched, and identical poses give identical results, so the joins stay exact.
   - Checks run on the generated carry; `report.md` shows the exported posture.

Result: idle, start, loop and stop all carry the rifle the same way, measured on the exported clips (grip 1.04–1.08 m, support 0.91–0.95 m, hand line 23° down and 30 cm to the left, hands 33 cm apart), with an upright torso (lean −0.9° to +5.4°), the head up, and exact joins.

Style keys used for this:

| Key | Meaning |
|---|---|
| `hand_anchor {pose_dir, every}` | Hand pins every N frames to the anchor pose |
| `carry_edit {right, left}` | Character-frame hand offsets applied by IK at export |
| `idle_pelvis_h` | Standing pelvis height for idle, turn and start/stop pins |
| `signature_on` | Clip kinds the signature applies to |
| `idle_signature` | Extra idle checks (head up, stance, upright) |
| `prompt_*` | Templates |

## Limits of the prototype

- **Only one gait (walk)**, 4 or 8 directions, starts, stops and turns in place. Not yet included: pivots or plants, turns while walking (arcs), circle loops, speed transitions, runs and sprints, phase-shifted variants (the other foot leading) and mirroring.
- **Style consistency is checked but not guaranteed.** Clips that fail after all retries are kept as `BEST-EFFORT`; read `report.md`.
- **Starts and stops in other directions use a direction-worded prompt.** Whether a strafe start reads as a natural step-off has not been watched.
- **There is no separate root bone yet** (Hips carries the root motion) and no FBX. Retarget to Manny in Unreal as before.
- **Visual quality is only checked by metrics.** Watch the GLBs in Blender or Unreal; the SOMA ones show the body mesh.

## Where this should live

- **For now:** here, as a standalone tool that only talks to Kimodo through the generator protocol (JSON lines over `kmd-generate --server`), so it can move out unchanged.
- **Long term:** a separate Motion-Stage or Factory project consuming that protocol, or a small HTTP service around it. The engine features it would benefit from belong in kimodo.cpp: an optional `root_height` on `root2d`, embedding requests, and per-style guidance passes ([04 gap table](../../docs/motion-stage-research/04-kimodo-internals-constraints-and-gaps.md)).
- **Avoid the web demo's `/api/generate` for this.** It writes into `demo-output/`, defaults to the non-commercial SMPL-X model, rejects segments under 60 frames, and its queue blocks after 32 waiting jobs.

## Files

| File | Purpose |
|---|---|
| `factory.py` | CLI and orchestration: jobs, retries, checks, export, manifest, report, continuity |
| `clips.py` | Paths (straight, spring ramps, eased heading) and constraint builders (`root2d`, `fullbody` pins, Hips pin, requests) |
| `skeleton.py` | SOMA-30 FK, quaternion helpers, posture and skate metrics, cycle search |
| `styles.json` | Style library |
| `soma_glb.py`, `soma77.py` | SOMA body-mesh GLB exporter (77-joint rig, 8-weight skinning, validated in Blender to 0.00 mm) |
| `make-mm-set.cmd` | One-click launcher |
