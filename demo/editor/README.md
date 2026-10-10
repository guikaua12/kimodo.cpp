# Constraint editor modules

ES modules served at `/editor/` by the demo server (embedded with `go:embed`).
Each module's header documents its API, conventions and deviations from upstream;
`tests/` holds `node --test` unit tests for the DOM-free modules.

## DOM-free logic (run unchanged in the browser and in node)

| Module | Contents |
|---|---|
| `math.js` | xyzw quaternions, row-major 3x3 matrices, axis-angle (upstream `matrix_to_axis_angle` canonical form), vec3 helpers |
| `skeleton.js` | `Skeleton` from an `/api/models` entry: FK, `localFromGlobals`, `restPose`, `heading`, chains/hips, hidden gizmo joints |
| `motion.js` | `Motion`: per-frame local rotations + root with cached FK and editing helpers (`setJointGlobalRotation`, ...) |
| `smooth_root.js` | exact port of upstream `smooth_signal` / `get_smooth_root_pos` (ADMM multigrid, banded Cholesky) |
| `constraints.js` | `ConstraintStore`: the six timeline tracks, write-through, Make Smooth Path, `toModelJSON` / `loadModelJSON` |
| `soma77.js` | SOMA77 names and the 30-from-77 slice for importing 77-joint constraint files |

Each file's header lists its conventions and the deliberate deviations from upstream.

## Browser modules

| Module | Contents |
|---|---|
| `viewport.js` | three.js view: mannequin, skeleton overlay, constraint ghosts / waypoints / paths, labels |
| `editing.js` | Editing mode: joint rotate gizmos, root ground gizmo, heading handle, undo / reset / snap |
| `timeline.js` | Canvas timeline: ruler + playhead, Prompts row, six constraint tracks and their gestures |
| `app.js` | Page shell for `index.html`: sidebar tabs, generation requests, gallery, playback, wiring |
| `style.css` | Page styles (LocalAI house style) |

`window.__kimodo` (set by `app.js`) exposes the state, store, timeline, viewport and
editor for automated browser tests (`window.__kimodoEditor` is the editor); gallery items
carry `data-id`; `Timeline#frameToX` / `#rowY` give click positions.

`dev/viewport-harness.html` is a developer page for `viewport.js` + `editing.js` (one real
animation, example constraints, every option and editing action; `window.__h` for scripts).
Open it through the demo server at `/editor/dev/viewport-harness.html?id=<animation id>`
(start the server with `-web-dir demo` to pick up edits without rebuilding). It reads only
the server API (`/api/models`, `/api/animations` and that animation's motion files),
never the gallery directory.

## Tests

```
node --test demo/editor/tests/
```

(`node --test` inside `demo/editor`, or the glob `"demo/editor/tests/*.test.mjs"`, run
the same files.) Fixtures and their provenance: `tests/fixtures/README.md`.
