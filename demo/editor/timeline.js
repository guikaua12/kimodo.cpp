// Canvas timeline for the constraint editor: frame ruler + playhead, the
// Prompts row (one block per text segment) and the seven constraint tracks
// (constraints.js TRACKS; the Hips row is ours, upstream has six).
// Port of the kimodo-viser timeline widget used by the upstream NVIDIA Kimodo
// demo (src/viser/client/src/Timeline.tsx, timeline/draw.ts, styles.ts);
// gestures and limits as in the upstream demo (kimodo/demo/ui.py timeline callbacks).
//
// The widget owns no constraint or prompt data. setData() hands it a view of
// the app state; every edit is reported through a callback and the app answers
// with a new setData(). While a gesture is in progress the widget draws a
// local preview (the dragged item / resized prompts) and reports the result
// once, on pointer-up.
//
// Callbacks (all optional):
//   seek(f)                       header click / drag (always enabled)
//   add(track, f) -> id|null      click on an empty track spot; when an id is
//                                 returned the new keyframe keeps following the
//                                 pointer while the button is held
//   addInterval(track, s, e)      Ctrl/Cmd-drag on an empty spot (s === e on a plain Ctrl-click)
//   move(id, s, e)                drag / Ctrl-edge resize of an item (on release)
//   remove(id)                    right-click an item
//   promptsChange(segments)       [{prompt, frames}] after any prompt edit
//
// Gestures:
//   tracks  click empty = add keyframe (+ drag while held); drag keyframe or
//           interval body = move; Ctrl/Cmd-drag empty = new interval;
//           Ctrl/Cmd-drag an item edge = resize (keyframe -> interval, flips
//           past the opposite edge); right-click item = delete. Locked items
//           (Smooth Path anchors / band) are not editable; the band lets
//           clicks through so keyframes can still be added on top of it.
//   prompts drag a boundary = resize the left segment and shift the rest;
//           drag the last block's right edge = resize it; double-click = edit
//           text (Enter commits, Escape cancels); Shift+click = split;
//           right-click a block = delete it (not the last one); right-click a
//           boundary = merge; click after the last block = append a segment.
//   view    wheel = pan, Shift+wheel = zoom (anchored on the playhead when it
//           is visible, else on the pointer); double-click the ruler = fit.
// Frame edits are clamped to [0, numFrames-1]; segment lengths to
// [minFrames, maxFrames]. With enabled = false only ruler scrubbing works.
// Items past the end of the clip (kept by the store when the clip shrinks, not
// sent) are drawn faded and dashed; they can still be dragged back in or
// right-clicked away.
//
// Extra API beyond the core one: frameToX(f), xToFrame(x), rowY(track|'prompts'|'header'),
// fit(), flush(), destroy(); constructor options {minFrames, maxFrames, maxSegments,
// defaultPrompt, defaultFrames, fps}.

import { TRACKS, TRACK_LABELS, TRACK_COLORS } from './constraints.js';

/** Prompt block colours, cycled per segment (kimodo-viser timeline/styles.ts PROMPT_COLORS). */
export const PROMPT_COLORS = Object.freeze([
  [82, 133, 166], [239, 68, 68], [249, 115, 22], [234, 179, 8],
  [34, 197, 94], [59, 130, 246], [168, 85, 247], [236, 72, 153],
]);

// Geometry (CSS pixels). Upstream: ruler 35, prompt row 40, tracks 40 * 0.5.
const HEADER_H = 35;
const PROMPT_H = 40;
const TRACK_H_MIN = 20;          // track rows grow to fill the container (max 28); 7 rows need >= 215 px
const LABEL_W = 96;          // track label column
const PAD_L = 12;            // gap between the label column and frame 0 (room for a keyframe dot)
const PAD_R = 14;
const KF_R = 6;              // keyframe radius
const HIT_R = 10;            // keyframe hit radius
const EDGE_HIT = 6;          // Ctrl-resize grab distance from an item edge
const BOUNDARY_HIT = 5;      // prompt boundary grab distance
const DRAG_THRESHOLD = 3;    // px before a press on an item counts as a drag
const FONT = "Inter, system-ui, -apple-system, 'Segoe UI', sans-serif";

// LocalAI dark palette (index.html --la-*) with upstream's timeline accents.
const C = {
  bg: '#0e1318',
  header: '#0b1014',
  labelBg: '#10161c',
  line: '#232e37',
  tick: '#1a232b',
  text: '#8b9aa5',
  label: '#c4ced6',
  playhead: '#4f9cf7',
  highlight: '#f4d03f',
  border: '#f3f6f4',
  outside: 'rgba(0, 0, 0, 0.28)',
  disabled: 'rgba(8, 11, 15, 0.55)',
  warn: '#ffb04c',
  preview: 'rgba(59, 211, 208, 0.9)',
};

// Tooltip notes: what a keyframe of an end-effector track holds beyond its joints.
const LIMB_NOTE = 'also holds the pelvis position, hip height and facing of the stored pose';
const HIPS_NOTE = 'pins the pelvis position, hip height and pelvis rotation of the stored pose';

const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
const rgba = (c, a) => `rgba(${c[0]}, ${c[1]}, ${c[2]}, ${a})`;
const isCtrl = (e) => e.ctrlKey || e.metaKey;

export class Timeline {
  constructor(container, callbacks = {}, options = {}) {
    this.container = container;
    this.callbacks = callbacks;
    this.opts = {
      minFrames: 60, maxFrames: 300, maxSegments: 16,
      defaultPrompt: 'A person walks forward.', defaultFrames: 180, fps: 30,
      ...options,
    };
    this.data = { numFrames: 1, frame: 0, items: [], prompts: [], warnings: {}, counts: null, enabled: true };
    // View: first visible frame (float) and visible frame count. `fit` re-fits
    // the whole clip whenever its length changes, until the user zooms.
    this.view = { start: 0, frames: 100, fit: true };
    this.drag = null;       // the gesture in progress (see _onDown)
    this.hover = null;      // last hit-test under the pointer (cursor + highlight)
    this.edit = null;       // inline prompt text editor {index, input}
    this._ctrl = false;
    this._shift = false;    // Shift held: split preview on the Prompts row
    this._raf = 0;
    this._autopan = 0;

    container.classList.add('kt-timeline');
    if (getComputedStyle(container).position === 'static') container.style.position = 'relative';
    this.canvas = document.createElement('canvas');
    this.canvas.className = 'kt-canvas';
    this.canvas.style.cssText = 'display:block;width:100%;height:100%;touch-action:none;user-select:none';
    this.canvas.setAttribute('role', 'application');
    this.canvas.setAttribute('aria-label', 'Timeline: prompts and constraint tracks');
    container.append(this.canvas);
    this.ctx = this.canvas.getContext('2d');
    this.width = 0; this.height = 0; this.dpr = 1;
    this.trackH = TRACK_H_MIN;

    const c = this.canvas;
    this._listeners = [
      [c, 'pointerdown', (e) => this._onDown(e)],
      [c, 'pointermove', (e) => this._onMove(e)],
      [c, 'pointerup', (e) => this._onUp(e)],
      [c, 'pointercancel', () => this._cancelDrag()],
      [c, 'pointerleave', () => { if (!this.drag) { this.hover = null; this._setCursor('default'); this._invalidate(); } }],
      [c, 'contextmenu', (e) => this._onContextMenu(e)],
      [c, 'dblclick', (e) => this._onDblClick(e)],
      [c, 'wheel', (e) => this._onWheel(e), { passive: false }],
      [window, 'keydown', (e) => this._onKey(e, true)],
      [window, 'keyup', (e) => this._onKey(e, false)],
      [window, 'blur', () => { this._ctrl = false; this._shift = false; this._invalidate(); }],
    ];
    for (const [target, type, fn, opt] of this._listeners) target.addEventListener(type, fn, opt);
    this._resizeObserver = new ResizeObserver(() => this._resize());
    this._resizeObserver.observe(container);
    this._resize();
  }

  // ---------------------------------------------------------------------------
  // public API
  // ---------------------------------------------------------------------------

  /**
   * Updates the displayed state. Accepts a partial object; omitted keys keep
   * their value. {numFrames, frame, items: [{id, track, start, end, locked,
   * opacity}], prompts: [{prompt, frames}], warnings: {track: bool},
   * counts?: {track: n}, enabled}.
   */
  setData(patch) {
    const prevFrames = this.data.numFrames;
    Object.assign(this.data, patch);
    this.data.numFrames = Math.max(1, Math.round(this.data.numFrames || 1));
    if (this.data.numFrames !== prevFrames) {
      if (this.view.fit) this.fit();
      else this._clampView();
    }
    if (patch.enabled === false && this.drag && this.drag.kind !== 'scrub') this._cancelDrag();
    if (patch.prompts && this.edit && this.edit.index >= this.data.prompts.length) this._closeEditor(false);
    this._invalidate();
  }

  /**
   * Draws now when a redraw is pending. The app calls it from its own
   * animation frame after a frame change, so the playhead is drawn in the same
   * frame as the 3D view and the frame readout (not one frame later).
   */
  flush() {
    if (!this._raf) return;
    cancelAnimationFrame(this._raf);
    this._raf = 0;
    this._draw();
  }

  /** Fits the whole clip (plus a little room to extend it) into the view. */
  fit() {
    const n = this.data.numFrames;
    this.view.frames = Math.max(30, Math.ceil(n * 1.08) + 4);
    this.view.start = 0;
    this.view.fit = true;
    this._invalidate();
  }

  /** CSS-pixel x (relative to the canvas) of frame f. */
  frameToX(f) {
    return LABEL_W + PAD_L + (f - this.view.start) * this._ppf();
  }

  /** Nearest frame at CSS-pixel x (not clamped). */
  xToFrame(x) {
    return Math.round((x - LABEL_W - PAD_L) / this._ppf() + this.view.start);
  }

  /** Vertical centre (CSS px) of a track row, of 'prompts' or of 'header'. */
  rowY(row) {
    if (row === 'header') return HEADER_H / 2;
    if (row === 'prompts') return HEADER_H + PROMPT_H / 2;
    const i = TRACKS.indexOf(row);
    if (i < 0) throw new Error(`unknown row ${row}`);
    return HEADER_H + PROMPT_H + i * this.trackH + this.trackH / 2;
  }

  destroy() {
    for (const [target, type, fn, opt] of this._listeners) target.removeEventListener(type, fn, opt);
    this._resizeObserver.disconnect();
    cancelAnimationFrame(this._raf);
    cancelAnimationFrame(this._autopan);
    this._closeEditor(false);
    this.canvas.remove();
  }

  // ---------------------------------------------------------------------------
  // geometry helpers
  // ---------------------------------------------------------------------------

  _ppf() {
    const w = Math.max(40, this.width - LABEL_W - PAD_L - PAD_R);
    return w / this.view.frames;
  }

  /** Start frame of each segment plus the clip length, for a prompt list. */
  _segmentStarts(prompts) {
    const starts = [];
    let t = 0;
    for (const p of prompts) { starts.push(t); t += p.frames; }
    return { starts, total: t };
  }

  /** The prompt list to draw: the drag preview while a prompt gesture runs. */
  _prompts() {
    return this.drag?.preview ?? this.data.prompts;
  }

  /** Total frames including an in-progress prompt resize (for pan limits). */
  _contentFrames() {
    const p = this._prompts();
    const total = p.length ? this._segmentStarts(p).total : this.data.numFrames;
    return Math.max(this.data.numFrames, total);
  }

  /** Clamps the pan so frame 0 and the end of the clip stay reachable. */
  _clampView() {
    const v = this.view;
    v.frames = clamp(v.frames, 12, Math.max(60, this._contentFrames() * 3, 1200));
    const end = Math.max(this._contentFrames() * 1.08 + 4, v.frames);
    v.start = clamp(v.start, 0, Math.max(0, end - v.frames));
  }

  /** Rows: {kind: 'header'|'prompts'|'track'|'none', track?} at CSS-pixel y. */
  _row(y) {
    if (y < HEADER_H) return { kind: 'header' };
    if (y < HEADER_H + PROMPT_H) return { kind: 'prompts' };
    const i = Math.floor((y - HEADER_H - PROMPT_H) / this.trackH);
    if (i >= 0 && i < TRACKS.length) return { kind: 'track', track: TRACKS[i] };
    return { kind: 'none' };
  }

  /** Items of a track, with the dragged one at its preview position. */
  _trackItems(track) {
    const out = [];
    const d = this.drag;
    let seen = false;
    for (const it of this.data.items) {
      if (it.track !== track) continue;
      if (d?.kind === 'item' && it.id === d.id) { out.push({ ...it, start: d.s, end: d.e, dragging: true }); seen = true; } else out.push(it);
    }
    // A keyframe added on pointer-down may not be in `items` yet.
    if (d?.kind === 'item' && d.track === track && !seen) out.push({ id: d.id, track, start: d.s, end: d.e, locked: false, opacity: 1, dragging: true });
    return out;
  }

  /**
   * Hit test on a track at x: {item, part: 'key'|'body'|'start'|'end'} or null.
   * Keyframes win over intervals; locked / invisible items are skipped.
   */
  _hitItem(track, x) {
    const items = this.data.items.filter((it) => it.track === track && !it.locked && (it.opacity ?? 1) > 0);
    let best = null, bestD = Infinity;
    for (const it of items) {
      if (it.start !== it.end) continue;
      const d = Math.abs(this.frameToX(it.start) - x);
      if (d <= HIT_R && d < bestD) { best = it; bestD = d; }
    }
    if (best) return { item: best, part: 'key' };
    for (const it of items) {
      if (it.start === it.end) continue;
      const xs = this.frameToX(it.start), xe = this.frameToX(it.end);
      if (x < xs - KF_R - 2 || x > xe + KF_R + 2) continue;
      if (Math.abs(x - xs) <= EDGE_HIT) return { item: it, part: 'start' };
      if (Math.abs(x - xe) <= EDGE_HIT) return { item: it, part: 'end' };
      return { item: it, part: 'body' };
    }
    return null;
  }

  /**
   * Hit test on the Prompts row: {part: 'boundary', index} (between index and
   * index+1), {part: 'lastEdge'}, {part: 'block', index}, {part: 'after'} or null.
   */
  _hitPrompt(x) {
    const prompts = this.data.prompts;
    if (!prompts.length) return { part: 'after' };
    const { starts, total } = this._segmentStarts(prompts);
    for (let i = 1; i < prompts.length; i++) {
      if (Math.abs(x - this.frameToX(starts[i])) <= BOUNDARY_HIT) return { part: 'boundary', index: i - 1 };
    }
    const endX = this.frameToX(total - 1);
    if (Math.abs(x - endX) <= BOUNDARY_HIT + 1) return { part: 'lastEdge' };
    for (let i = 0; i < prompts.length; i++) {
      const xs = this.frameToX(starts[i]);
      const xe = i + 1 < prompts.length ? this.frameToX(starts[i + 1]) : endX;
      if (x >= xs && x <= xe) return { part: 'block', index: i };
    }
    if (x > endX) return { part: 'after' };
    return null;
  }

  _local(e) {
    const r = this.canvas.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  }

  _clampFrame(f) {
    return clamp(f, 0, this.data.numFrames - 1);
  }

  // ---------------------------------------------------------------------------
  // pointer gestures
  // ---------------------------------------------------------------------------

  _onDown(e) {
    if (e.button !== 0) return;
    if (this.edit) this._closeEditor(true);
    const { x, y } = this._local(e);
    const row = this._row(y);
    if (row.kind === 'header') {
      if (x < LABEL_W) return;
      this._startDrag(e, { kind: 'scrub' });
      this._seek(x);
      return;
    }
    if (!this.data.enabled || x < LABEL_W) return;
    const f = this.xToFrame(x);

    if (row.kind === 'prompts') {
      const hit = this._hitPrompt(x);
      if (!hit) return;
      const segs0 = this.data.prompts.map((p) => ({ ...p }));
      if (hit.part === 'boundary') {
        this._startDrag(e, { kind: 'boundary', index: hit.index, segs0, preview: segs0.map((p) => ({ ...p })), frame: null });
      } else if (hit.part === 'lastEdge') {
        this._startDrag(e, { kind: 'lastEdge', segs0, preview: segs0.map((p) => ({ ...p })), frame: null });
      } else if (hit.part === 'block' && e.shiftKey) {
        this._split(hit.index, f);
      } else if (hit.part === 'after') {
        this._append();
      }
      return;
    }

    if (row.kind !== 'track') return;
    const track = row.track;
    const hit = this._hitItem(track, x);
    const ctrl = isCtrl(e);
    const n = this.data.numFrames;
    if (hit && ctrl && hit.part !== 'body') {
      // Resize: the opposite edge stays put (a keyframe grows either way).
      const it = hit.item;
      const anchor = hit.part === 'start' ? it.end : it.start;
      this._startDrag(e, { kind: 'item', mode: 'resize', id: it.id, track, s0: it.start, e0: it.end, s: it.start, e: it.end, anchor, x0: x, moved: false });
    } else if (hit) {
      const it = hit.item;
      this._startDrag(e, { kind: 'item', mode: 'move', id: it.id, track, s0: it.start, e0: it.end, s: it.start, e: it.end, grab: f - it.start, x0: x, moved: false });
    } else if (ctrl) {
      const a = this._clampFrame(f);
      this._startDrag(e, { kind: 'band', track, anchor: a, s: a, e: a });
    } else if (f >= 0 && f <= n - 1) {
      const id = this.callbacks.add?.(track, f);
      if (id !== null && id !== undefined && id !== false) {
        // Upstream keeps dragging the new keyframe while the button is held.
        this._startDrag(e, { kind: 'item', mode: 'move', id, track, s0: f, e0: f, s: f, e: f, grab: 0, x0: x, moved: true, isNew: true });
      }
    }
    this._invalidate();
  }

  _startDrag(e, drag) {
    this.drag = drag;
    this.drag.pointerId = e.pointerId;
    this.drag.lastX = this._local(e).x;
    try { this.canvas.setPointerCapture(e.pointerId); } catch { /* synthetic events */ }
    this._updateCursor();
  }

  _onMove(e) {
    const { x, y } = this._local(e);
    const d = this.drag;
    if (!d) {
      this._updateHover(x, y, e);
      return;
    }
    d.lastX = x;
    this._applyDrag(x);
    if (d.kind === 'boundary' || d.kind === 'lastEdge' || d.kind === 'scrub') this._maybeAutopan();
  }

  /** Updates the gesture in progress for pointer x. */
  _applyDrag(x) {
    const d = this.drag;
    if (!d) return;
    const f = this.xToFrame(x);
    const n = this.data.numFrames;
    if (d.kind === 'scrub') {
      this._seek(x);
    } else if (d.kind === 'item') {
      if (!d.moved && Math.abs(x - d.x0) < DRAG_THRESHOLD) return;
      d.moved = true;
      if (d.mode === 'move') {
        const len = d.e0 - d.s0;
        d.s = clamp(f - d.grab, 0, Math.max(0, n - 1 - len));
        d.e = d.s + len;
      } else {
        const m = this._clampFrame(f);
        d.s = Math.min(d.anchor, m);
        d.e = Math.max(d.anchor, m);
      }
    } else if (d.kind === 'band') {
      const m = this._clampFrame(f);
      d.s = Math.min(d.anchor, m);
      d.e = Math.max(d.anchor, m);
    } else if (d.kind === 'boundary') {
      // Resize the left segment; everything to the right shifts with it.
      const { starts } = this._segmentStarts(d.segs0);
      const i = d.index;
      const frames = clamp(f - starts[i], this.opts.minFrames, this.opts.maxFrames);
      d.preview = d.segs0.map((p) => ({ ...p }));
      d.preview[i].frames = frames;
      d.frame = starts[i] + frames;
    } else if (d.kind === 'lastEdge') {
      const { starts } = this._segmentStarts(d.segs0);
      const i = d.segs0.length - 1;
      const frames = clamp(f - starts[i] + 1, this.opts.minFrames, this.opts.maxFrames);
      d.preview = d.segs0.map((p) => ({ ...p }));
      d.preview[i].frames = frames;
      d.frame = starts[i] + frames - 1;
    }
    this._invalidate();
  }

  _onUp(e) {
    const d = this.drag;
    if (!d) return;
    this.drag = null;
    cancelAnimationFrame(this._autopan); this._autopan = 0;
    try { this.canvas.releasePointerCapture(e.pointerId); } catch { /* already released */ }
    if (d.kind === 'item') {
      if (d.moved && (d.s !== d.s0 || d.e !== d.e0)) this.callbacks.move?.(d.id, d.s, d.e);
    } else if (d.kind === 'band') {
      this.callbacks.addInterval?.(d.track, d.s, d.e);
    } else if (d.kind === 'boundary' || d.kind === 'lastEdge') {
      const changed = d.preview.some((p, i) => p.frames !== d.segs0[i].frames);
      if (changed) this.callbacks.promptsChange?.(d.preview);
    }
    const { x, y } = this._local(e);
    this._updateHover(x, y, e);
    this._invalidate();
  }

  _cancelDrag() {
    if (!this.drag) return;
    try { this.canvas.releasePointerCapture(this.drag.pointerId); } catch { /* ignore */ }
    this.drag = null;
    cancelAnimationFrame(this._autopan); this._autopan = 0;
    this._invalidate();
  }

  _seek(x) {
    const f = this._clampFrame(this.xToFrame(x));
    if (f !== this.data.frame) {
      this.data.frame = f;
      this.callbacks.seek?.(f);
      this._invalidate();
    }
  }

  /** Pans while a prompt edge / the playhead is dragged past the view edges. */
  _maybeAutopan() {
    if (this._autopan) return;
    const step = () => {
      this._autopan = 0;
      const d = this.drag;
      if (!d) return;
      const x = d.lastX;
      const left = LABEL_W + PAD_L, right = this.width - 6;
      let dir = 0;
      if (x > right - 10) dir = 1;
      else if (x < left && this.view.start > 0) dir = -1;
      if (!dir) return;
      this.view.start += dir * Math.max(0.5, this.view.frames / 120);
      this.view.fit = false;
      this._clampView();
      this._applyDrag(x);
      this._autopan = requestAnimationFrame(step);
    };
    this._autopan = requestAnimationFrame(step);
  }

  _onContextMenu(e) {
    e.preventDefault();
    if (!this.data.enabled || this.drag) return;
    const { x, y } = this._local(e);
    if (x < LABEL_W) return;
    const row = this._row(y);
    if (row.kind === 'track') {
      const hit = this._hitItem(row.track, x);
      if (hit) this.callbacks.remove?.(hit.item.id);
    } else if (row.kind === 'prompts') {
      const hit = this._hitPrompt(x);
      if (hit?.part === 'boundary') this._merge(hit.index);
      else if (hit?.part === 'block' || hit?.part === 'lastEdge') this._deleteSegment(hit.part === 'block' ? hit.index : this.data.prompts.length - 1);
    }
  }

  _onDblClick(e) {
    const { x, y } = this._local(e);
    const row = this._row(y);
    if (row.kind === 'header') { this.fit(); return; }
    if (!this.data.enabled || row.kind !== 'prompts') return;
    const hit = this._hitPrompt(x);
    if (hit?.part === 'block') this._openEditor(hit.index);
  }

  _onWheel(e) {
    e.preventDefault();
    if (this.drag && this.drag.kind !== 'scrub') return;
    const { x } = this._local(e);
    // Chrome/Windows turns Shift+wheel into a horizontal delta.
    const delta = Math.abs(e.deltaY) >= Math.abs(e.deltaX) ? e.deltaY : e.deltaX;
    const scale = e.deltaMode === 1 ? 33 : e.deltaMode === 2 ? this.width : 1;
    if (e.shiftKey) {
      const ppf0 = this._ppf();
      const playX = this.frameToX(this.data.frame);
      const inView = playX >= LABEL_W + PAD_L && playX <= this.width - PAD_R;
      const anchorX = inView ? playX : clamp(x, LABEL_W + PAD_L, this.width);
      const anchorF = (anchorX - LABEL_W - PAD_L) / ppf0 + this.view.start;
      this.view.frames *= Math.exp(delta * scale * 0.0015);
      this.view.fit = false;
      this._clampView();
      this.view.start = anchorF - (anchorX - LABEL_W - PAD_L) / this._ppf();
    } else {
      this.view.start += (delta * scale) / this._ppf() * 0.5;
      if (this.view.start !== 0) this.view.fit = false;
    }
    this._clampView();
    this._invalidate();
  }

  _onKey(e, down) {
    const ctrl = e.ctrlKey || e.metaKey;
    if (ctrl !== this._ctrl) { this._ctrl = ctrl; if (!this.drag && this.hover) this._updateCursor(); }
    if (e.shiftKey !== this._shift) {
      this._shift = e.shiftKey;
      if (this.hover?.prompt) { this._updateCursor(); this._invalidate(); }
    }
    if (down && e.key === 'Escape' && this.drag && this.drag.kind !== 'scrub') {
      // Escape aborts a drag; an item added on pointer-down stays where it was added.
      this._cancelDrag();
    }
  }

  _updateHover(x, y, e) {
    const row = this._row(y);
    let hover = { row, x };
    if (row.kind === 'track' && x >= LABEL_W) hover.hit = this._hitItem(row.track, x);
    else if (row.kind === 'prompts' && x >= LABEL_W) hover.prompt = this._hitPrompt(x);
    this._ctrl = isCtrl(e);
    // The Shift split preview follows the pointer inside a prompt block.
    const splitMoved = (e.shiftKey || this._shift) && hover.prompt?.part === 'block' && this.xToFrame(x) !== this.xToFrame(this.hover?.x ?? -1);
    this._shift = e.shiftKey;
    const changed = splitMoved || JSON.stringify([this.hover?.hit?.item?.id, this.hover?.hit?.part, this.hover?.prompt]) !== JSON.stringify([hover.hit?.item?.id, hover.hit?.part, hover.prompt]);
    this.hover = hover;
    this._updateCursor();
    this._updateTitle(row, x);
    if (changed) this._invalidate();
  }

  _updateCursor() {
    const d = this.drag;
    if (d) {
      const cur = d.kind === 'scrub' ? 'ew-resize' : d.kind === 'boundary' || d.kind === 'lastEdge' || d.mode === 'resize' || d.kind === 'band' ? 'col-resize' : 'grabbing';
      this._setCursor(cur);
      return;
    }
    const h = this.hover;
    if (!h) return this._setCursor('default');
    if (h.row.kind === 'header') return this._setCursor(h.x >= LABEL_W ? 'pointer' : 'default');
    if (h.x < LABEL_W) return this._setCursor('default');
    if (!this.data.enabled && (h.row.kind === 'prompts' || h.row.kind === 'track')) return this._setCursor('not-allowed');
    if (h.row.kind === 'prompts') {
      const p = h.prompt?.part;
      if (p === 'block' && this._shift) return this._setCursor('crosshair'); // Shift+click splits here
      return this._setCursor(p === 'boundary' || p === 'lastEdge' ? 'col-resize' : p === 'after' ? 'copy' : 'default');
    }
    if (h.row.kind === 'track') {
      if (h.hit) return this._setCursor(this._ctrl && h.hit.part !== 'body' ? 'col-resize' : 'grab');
      return this._setCursor('crosshair');
    }
    this._setCursor('default');
  }

  _setCursor(c) {
    if (this.canvas.style.cursor !== c) this.canvas.style.cursor = c;
  }

  /** Native tooltip for the label column (counts / >= 20 frame warning) and items. */
  _updateTitle(row, x) {
    let title = '';
    if (row.kind === 'track') {
      const t = row.track;
      const n = this.data.counts?.[t];
      if (x < LABEL_W) {
        title = `${TRACK_LABELS[t]}${n !== undefined ? `: ${n} constrained frame${n === 1 ? '' : 's'}` : ''}`;
        if (t === 'hips') title += ` — ${HIPS_NOTE}`;
        if (this.data.warnings?.[t]) title += t === 'root2d'
          ? ' (20+ waypoints: fewer keyframes usually work better; dense paths are fine)'
          : ' (20+ frames: fewer than 20 constrained frames per type usually work better)';
      } else if (this.hover?.hit) {
        const it = this.hover.hit.item;
        const n = this.data.numFrames;
        title = `${TRACK_LABELS[t]} @ ${it.start === it.end ? it.start : `[${it.start}, ${it.end}]`}`;
        // A hand/foot keyframe holds the whole stored pose's root as well; a Hips keyframe pins exactly that.
        if (t === 'hips') title += ` (${HIPS_NOTE})`;
        else if (t !== 'fullbody' && t !== 'root2d') title += ` (${LIMB_NOTE})`;
        if (it.start >= n) title += ` — past the end of the clip (frame ${n - 1}): kept, but not sent; drag it back or lengthen the clip`;
        else if (it.end >= n) title += ` — frames after ${n - 1} are past the end of the clip and not sent`;
      }
    } else if (row.kind === 'prompts' && x >= LABEL_W) {
      const h = this.hover?.prompt;
      if (h?.part === 'block') title = this.data.prompts[h.index]?.prompt ?? '';
    }
    if (this.canvas.title !== title) this.canvas.title = title;
  }

  // ---------------------------------------------------------------------------
  // prompt edits
  // ---------------------------------------------------------------------------

  _emitPrompts(prompts) {
    this.callbacks.promptsChange?.(prompts.map((p) => ({ prompt: p.prompt, frames: p.frames })));
  }

  /** Shift+click: split block i at frame f; both halves keep the text (upstream). */
  _split(i, f) {
    const prompts = this.data.prompts.map((p) => ({ ...p }));
    if (prompts.length >= this.opts.maxSegments) return;
    const { starts } = this._segmentStarts(prompts);
    const seg = prompts[i];
    const min = this.opts.minFrames;
    // Too-short halves are lengthened (the clip may grow), as upstream does.
    const left = clamp(f - starts[i], min, this.opts.maxFrames);
    const right = Math.max(min, seg.frames - left);
    prompts.splice(i, 1, { prompt: seg.prompt, frames: left }, { prompt: seg.prompt, frames: right });
    this._emitPrompts(prompts);
  }

  /** Right-click a boundary: merge i and i+1 (left text kept, length capped). */
  _merge(i) {
    const prompts = this.data.prompts.map((p) => ({ ...p }));
    if (i + 1 >= prompts.length) return;
    const frames = Math.min(this.opts.maxFrames, prompts[i].frames + prompts[i + 1].frames);
    prompts.splice(i, 2, { prompt: prompts[i].prompt, frames });
    this._emitPrompts(prompts);
  }

  _deleteSegment(i) {
    if (this.data.prompts.length <= 1) return;
    const prompts = this.data.prompts.map((p) => ({ ...p }));
    prompts.splice(i, 1);
    this._emitPrompts(prompts);
  }

  _append() {
    const prompts = this.data.prompts.map((p) => ({ ...p }));
    if (prompts.length >= this.opts.maxSegments) return;
    prompts.push({ prompt: this.opts.defaultPrompt, frames: this.opts.defaultFrames });
    this._emitPrompts(prompts);
  }

  _openEditor(i) {
    this._closeEditor(true);
    const prompts = this.data.prompts;
    const { starts, total } = this._segmentStarts(prompts);
    const xs = Math.max(LABEL_W + 2, this.frameToX(starts[i]));
    const xe = Math.min(this.width - 2, i + 1 < prompts.length ? this.frameToX(starts[i + 1]) : this.frameToX(total - 1));
    const input = document.createElement('input');
    input.type = 'text';
    input.className = 'kt-prompt-input';
    input.value = prompts[i].prompt;
    input.setAttribute('aria-label', `Prompt ${i + 1} text`);
    const col = PROMPT_COLORS[i % PROMPT_COLORS.length];
    input.style.cssText = `position:absolute;left:${xs}px;top:${HEADER_H + 3}px;width:${Math.max(160, xe - xs)}px;height:${PROMPT_H - 6}px;`
      + `box-sizing:border-box;z-index:3;border:2px solid ${C.highlight};border-radius:6px;padding:0 10px;`
      + `background:${rgba(col.map((v) => v * 0.55), 0.98)};color:#fff;font:12px ${FONT};outline:none;text-align:center`;
    this.container.append(input);
    this.edit = { index: i, input, original: prompts[i].prompt };
    input.addEventListener('keydown', (e) => {
      e.stopPropagation(); // keep Space / arrows away from the playback shortcuts
      if (e.key === 'Enter') { e.preventDefault(); this._closeEditor(true); }
      else if (e.key === 'Escape') { e.preventDefault(); this._closeEditor(false); }
    });
    input.addEventListener('blur', () => this._closeEditor(true));
    input.focus();
    input.select();
  }

  _closeEditor(commit) {
    const ed = this.edit;
    if (!ed) return;
    this.edit = null;
    ed.input.remove();
    const text = ed.input.value.trim();
    if (commit && text && text !== ed.original && ed.index < this.data.prompts.length) {
      const prompts = this.data.prompts.map((p) => ({ ...p }));
      prompts[ed.index].prompt = text;
      this._emitPrompts(prompts);
    }
  }

  // ---------------------------------------------------------------------------
  // drawing
  // ---------------------------------------------------------------------------

  _resize() {
    const r = this.container.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.floor(r.width)), h = Math.max(1, Math.floor(r.height));
    if (w === this.width && h === this.height && dpr === this.dpr) return;
    this.width = w; this.height = h; this.dpr = dpr;
    this.trackH = clamp(Math.floor((h - HEADER_H - PROMPT_H) / TRACKS.length), TRACK_H_MIN, 28);
    this.canvas.width = Math.round(w * dpr);
    this.canvas.height = Math.round(h * dpr);
    if (this.view.fit) this.fit();
    this._clampView();
    this._draw();
  }

  _invalidate() {
    if (this._raf) return;
    this._raf = requestAnimationFrame(() => { this._raf = 0; this._draw(); });
  }

  _draw() {
    const ctx = this.ctx;
    const W = this.width, H = this.height;
    if (!ctx || W < 2) return;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);

    const n = this.data.numFrames;
    const ppf = this._ppf();
    const tracksTop = HEADER_H + PROMPT_H;
    const tracksBottom = tracksTop + TRACKS.length * this.trackH;
    const x0 = LABEL_W;

    // Frame grid: "nice" label spacing of at least ~56 px.
    const steps = [1, 2, 5, 10, 15, 30, 60, 90, 150, 300, 600, 900, 1800];
    const step = steps.find((s) => s * ppf >= 56) ?? 3600;
    const first = Math.max(0, Math.floor(this.view.start / step) * step);
    const last = this.view.start + this.view.frames + step;

    ctx.save();
    ctx.beginPath(); ctx.rect(x0, 0, W - x0, H); ctx.clip();

    // Ruler background.
    ctx.fillStyle = C.header;
    ctx.fillRect(x0, 0, W - x0, HEADER_H);
    // Grid lines and labels.
    ctx.font = `10px ${FONT}`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (let f = first; f <= last; f += step) {
      const x = Math.round(this.frameToX(f)) + 0.5;
      if (x < x0 - 30 || x > W + 30) continue;
      ctx.strokeStyle = C.tick;
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x, HEADER_H); ctx.lineTo(x, H); ctx.stroke();
      ctx.strokeStyle = C.line;
      ctx.beginPath(); ctx.moveTo(x, HEADER_H - 6); ctx.lineTo(x, HEADER_H); ctx.stroke();
      // Skip labels that would be cut by the label column.
      if (x - ctx.measureText(String(f)).width / 2 < x0 + 2) continue;
      ctx.fillStyle = f < n ? C.text : 'rgba(139, 154, 165, 0.45)';
      ctx.fillText(String(f), x, HEADER_H / 2 - 2);
    }

    // Outside the clip: shade frames past the end, and draw the end line.
    const endX = this.frameToX(n - 1) + KF_R + 2;
    if (endX < W) {
      ctx.fillStyle = C.outside;
      ctx.fillRect(endX, HEADER_H, W - endX, H - HEADER_H);
      ctx.strokeStyle = C.line;
      ctx.beginPath(); ctx.moveTo(Math.round(endX) + 0.5, HEADER_H); ctx.lineTo(Math.round(endX) + 0.5, H); ctx.stroke();
    }

    // Row separators.
    ctx.strokeStyle = C.line;
    ctx.lineWidth = 1;
    for (let k = 0; k <= TRACKS.length; k++) {
      const y = Math.round(tracksTop + k * this.trackH) + 0.5;
      ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(W, y); ctx.stroke();
    }
    ctx.beginPath(); ctx.moveTo(x0, HEADER_H + 0.5); ctx.lineTo(W, HEADER_H + 0.5); ctx.stroke();

    this._drawPrompts(ctx);
    for (let k = 0; k < TRACKS.length; k++) this._drawTrack(ctx, TRACKS[k], tracksTop + k * this.trackH);

    // Rubber-band interval preview.
    const d = this.drag;
    if (d?.kind === 'band') {
      const k = TRACKS.indexOf(d.track);
      const col = TRACK_COLORS[d.track];
      this._intervalShape(ctx, d.s, d.e, tracksTop + k * this.trackH, rgba(col, 0.45), C.preview, [4, 3]);
    }

    if (!this.data.enabled) {
      ctx.fillStyle = C.disabled;
      ctx.fillRect(x0, HEADER_H + 1, W - x0, tracksBottom - HEADER_H);
    }

    // Playhead line.
    const px = this.frameToX(this.data.frame);
    ctx.save();
    ctx.strokeStyle = C.playhead;
    ctx.lineWidth = 2;
    ctx.shadowColor = 'rgba(0, 0, 0, 0.45)';
    ctx.shadowBlur = 4;
    ctx.beginPath(); ctx.moveTo(px, HEADER_H - 2); ctx.lineTo(px, H); ctx.stroke();
    ctx.restore();
    ctx.restore(); // clip

    this._drawLabels(ctx, tracksTop);

    // Playhead box (drawn over the label column so frame 0 stays readable).
    ctx.save();
    ctx.beginPath(); ctx.rect(x0 - 26, 0, W - x0 + 26, HEADER_H); ctx.clip();
    this._frameBox(ctx, px, this.data.frame, C.playhead, '#fff');
    // While dragging an item / prompt edge: the frame under the cursor in yellow.
    let dragFrame = null;
    if (d?.kind === 'item' && d.moved) dragFrame = d.mode === 'resize' ? (d.s === d.anchor ? d.e : d.s) : d.s;
    else if (d?.kind === 'band') dragFrame = d.s === d.anchor ? d.e : d.s;
    else if ((d?.kind === 'boundary' || d?.kind === 'lastEdge') && d.frame !== null) dragFrame = d.frame;
    if (dragFrame !== null) this._frameBox(ctx, this.frameToX(dragFrame), dragFrame, C.highlight, '#000');
    ctx.restore();
  }

  _frameBox(ctx, x, frame, fill, textColor) {
    const w = Math.max(40, 14 + String(frame).length * 7), h = 24, y = 5;
    ctx.save();
    ctx.shadowColor = 'rgba(0, 0, 0, 0.35)';
    ctx.shadowBlur = 5;
    ctx.shadowOffsetY = 2;
    ctx.fillStyle = fill;
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.85)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.roundRect(x - w / 2, y, w, h, 4);
    ctx.fill();
    ctx.stroke();
    ctx.shadowBlur = 0; ctx.shadowOffsetY = 0;
    ctx.fillStyle = textColor;
    ctx.font = `bold 11px ${FONT}`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(String(frame), x, y + h / 2 + 0.5);
    ctx.restore();
  }

  _drawPrompts(ctx) {
    const prompts = this._prompts();
    if (!prompts.length) return;
    const { starts, total } = this._segmentStarts(prompts);
    const y = HEADER_H + 3, h = PROMPT_H - 6;
    const editing = this.edit?.index;
    const hoverP = !this.drag && this.data.enabled ? this.hover?.prompt : null;
    for (let i = 0; i < prompts.length; i++) {
      const xs = this.frameToX(starts[i]);
      const xe = i + 1 < prompts.length ? this.frameToX(starts[i + 1]) : this.frameToX(total - 1);
      if (xe < LABEL_W - 4 || xs > this.width + 4) continue;
      const col = PROMPT_COLORS[i % PROMPT_COLORS.length];
      const bx = xs + 1, bw = Math.max(2, xe - xs - 2);
      ctx.save();
      ctx.beginPath();
      ctx.roundRect(bx, y, bw, h, 5);
      const g = ctx.createLinearGradient(0, y, 0, y + h);
      g.addColorStop(0, rgba(col, 0.92));
      g.addColorStop(1, rgba(col.map((v) => v * 0.78), 0.92));
      ctx.fillStyle = g;
      ctx.fill();
      ctx.strokeStyle = '#05080b';
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.beginPath();
      ctx.roundRect(bx + 1, y + 1, bw - 2, h - 2, 4);
      ctx.strokeStyle = hoverP?.part === 'block' && hoverP.index === i ? 'rgba(255, 255, 255, 0.45)' : 'rgba(255, 255, 255, 0.15)';
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.clip();
      if (editing !== i) {
        // Duration pill (seconds), then the centred prompt text.
        const visL = Math.max(bx, LABEL_W), visR = Math.min(bx + bw, this.width);
        const secs = `${(prompts[i].frames / this.opts.fps).toFixed(1)} s`;
        ctx.font = `11px ${FONT}`;
        const pw = ctx.measureText(secs).width + 12, ph = 16;
        let textL = visL + 10;
        if (visR - visL > pw + 24) {
          ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
          ctx.beginPath(); ctx.roundRect(visL + 6, y + (h - ph) / 2, pw, ph, 8); ctx.fill();
          ctx.fillStyle = '#fff';
          ctx.textAlign = 'left';
          ctx.textBaseline = 'middle';
          ctx.fillText(secs, visL + 12, y + h / 2 + 0.5);
          textL = visL + 6 + pw + 8;
        }
        const avail = visR - 10 - textL;
        if (avail > 24) {
          ctx.font = `12px ${FONT}`;
          ctx.fillStyle = '#fff';
          ctx.shadowColor = 'rgba(0, 0, 0, 0.5)';
          ctx.shadowBlur = 2;
          ctx.shadowOffsetY = 1;
          const text = ellipsize(ctx, prompts[i].prompt, avail);
          const tw = ctx.measureText(text).width;
          const cx = clamp((visL + visR) / 2, textL + tw / 2, visR - 10 - tw / 2);
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(text, cx, y + h / 2 + 0.5);
        }
      }
      ctx.restore();
    }
    // Grab handles: highlight the boundary / last edge under the pointer or being dragged.
    const d = this.drag;
    const markX = (x) => {
      ctx.fillStyle = C.highlight;
      ctx.fillRect(Math.round(x) - 1.5, y + 3, 3, h - 6);
    };
    if (d?.kind === 'boundary') markX(this.frameToX(starts[d.index + 1]));
    else if (d?.kind === 'lastEdge') markX(this.frameToX(total - 1));
    else if (hoverP?.part === 'boundary') markX(this.frameToX(starts[hoverP.index + 1]));
    else if (hoverP?.part === 'lastEdge') markX(this.frameToX(total - 1));
    else if (hoverP?.part === 'block' && this._shift) {
      // Split preview (upstream: a line where Shift+click would cut the block;
      // the cut honours the minimum segment length like _split does).
      const i = hoverP.index;
      const cut = starts[i] + clamp(this.xToFrame(this.hover.x) - starts[i], this.opts.minFrames, this.opts.maxFrames);
      const sx = Math.round(this.frameToX(cut)) + 0.5;
      ctx.save();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.moveTo(sx, y - 1); ctx.lineTo(sx, y + h + 1); ctx.stroke();
      ctx.restore();
    }
  }

  _drawTrack(ctx, track, top) {
    const items = this._trackItems(track);
    const col = TRACK_COLORS[track];
    const hoverId = !this.drag && this.data.enabled ? this.hover?.hit?.item?.id : null;
    const n = this.data.numFrames;
    // Constraints kept past the end of the clip (not sent) are drawn faded and
    // dashed from the clip's end line on.
    const endX = this.frameToX(n - 1) + KF_R + 2;
    // Locked bands first (Smooth Path), then intervals, then keyframes on top.
    const order = (it) => (it.locked ? 0 : it.start === it.end ? 2 : 1);
    items.sort((a, b) => order(a) - order(b));
    for (const it of items) {
      const opacity = it.opacity ?? 1;
      if (opacity <= 0) continue; // invisible Smooth Path anchors
      const hl = it.dragging || it.id === hoverId;
      const border = it.locked ? 'rgba(243, 246, 244, 0.35)' : hl ? C.highlight : C.border;
      const draw = (fade, dash) => {
        if (it.start === it.end && !it.locked) {
          const x = this.frameToX(it.start), cy = top + this.trackH / 2;
          if (x < LABEL_W - KF_R || x > this.width + KF_R) return;
          ctx.save();
          if (dash) ctx.setLineDash(dash);
          ctx.beginPath();
          ctx.arc(x, cy, hl ? KF_R + 1 : KF_R, 0, Math.PI * 2);
          ctx.fillStyle = rgba(col, opacity * fade);
          ctx.fill();
          ctx.lineWidth = 1.5;
          ctx.strokeStyle = border;
          ctx.stroke();
          ctx.restore();
        } else {
          this._intervalShape(ctx, it.start, it.end, top, rgba(col, (it.locked ? Math.min(opacity, 0.3) : opacity) * fade), border, dash);
        }
      };
      if (it.locked || it.end < n) { draw(1, null); continue; }
      if (it.start >= n) { draw(0.3, [3, 3]); continue; } // entirely past the end
      // Crossing the end: the part inside the clip as usual, the rest faded.
      const clipped = (fromX, toX, fade, dash) => {
        ctx.save();
        ctx.beginPath(); ctx.rect(fromX, top, toX - fromX, this.trackH); ctx.clip();
        draw(fade, dash);
        ctx.restore();
      };
      clipped(LABEL_W, endX, 1, null);
      clipped(endX, this.width, 0.3, [3, 3]);
    }
  }

  _intervalShape(ctx, s, e, top, fill, stroke, dash) {
    const xs = this.frameToX(s) - KF_R, xe = this.frameToX(e) + KF_R;
    if (xe < LABEL_W || xs > this.width) return;
    const cy = top + this.trackH / 2;
    ctx.save();
    ctx.beginPath();
    ctx.roundRect(xs, cy - KF_R, Math.max(2 * KF_R, xe - xs), 2 * KF_R, KF_R);
    ctx.fillStyle = fill;
    ctx.fill();
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = stroke;
    if (dash) ctx.setLineDash(dash);
    ctx.stroke();
    ctx.restore();
  }

  _drawLabels(ctx, tracksTop) {
    const H = this.height;
    ctx.fillStyle = C.labelBg;
    ctx.fillRect(0, 0, LABEL_W, H);
    ctx.fillStyle = C.header;
    ctx.fillRect(0, 0, LABEL_W, HEADER_H);
    ctx.strokeStyle = C.line;
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(LABEL_W + 0.5, 0); ctx.lineTo(LABEL_W + 0.5, H); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, HEADER_H + 0.5); ctx.lineTo(LABEL_W, HEADER_H + 0.5); ctx.stroke();

    ctx.textBaseline = 'middle';
    // Header: clip length.
    ctx.font = `10px ${FONT}`;
    ctx.fillStyle = C.text;
    ctx.textAlign = 'left';
    ctx.fillText(`${this.data.numFrames} f · ${(this.data.numFrames / this.opts.fps).toFixed(1)} s`, 10, HEADER_H / 2);

    ctx.font = `600 12px ${FONT}`;
    ctx.fillStyle = C.label;
    ctx.fillText('Prompts', 10, HEADER_H + PROMPT_H / 2);
    ctx.beginPath(); ctx.moveTo(0, tracksTop + 0.5); ctx.lineTo(LABEL_W, tracksTop + 0.5); ctx.stroke();

    TRACKS.forEach((t, k) => {
      const cy = tracksTop + k * this.trackH + this.trackH / 2;
      ctx.fillStyle = rgba(TRACK_COLORS[t], this.data.enabled ? 1 : 0.5);
      ctx.beginPath(); ctx.arc(14, cy, 3.5, 0, Math.PI * 2); ctx.fill();
      ctx.font = `11px ${FONT}`;
      ctx.fillStyle = this.data.enabled ? C.label : C.text;
      ctx.textAlign = 'left';
      ctx.fillText(TRACK_LABELS[t], 23, cy + 0.5);
      if (this.data.warnings?.[t]) {
        // Soft >= 20 frame warning (upstream docs, limitations): amber "!" badge.
        const bx = LABEL_W - 13;
        ctx.fillStyle = C.warn;
        ctx.beginPath(); ctx.arc(bx, cy, 6, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = '#1a1206';
        ctx.font = `bold 10px ${FONT}`;
        ctx.textAlign = 'center';
        ctx.fillText('!', bx, cy + 0.5);
      }
      ctx.strokeStyle = C.line;
      ctx.beginPath(); ctx.moveTo(0, tracksTop + (k + 1) * this.trackH + 0.5); ctx.lineTo(LABEL_W, tracksTop + (k + 1) * this.trackH + 0.5); ctx.stroke();
    });
  }
}

/** Longest prefix of text (with an ellipsis) that fits maxWidth at the current font. */
function ellipsize(ctx, text, maxWidth) {
  if (ctx.measureText(text).width <= maxWidth) return text;
  let lo = 0, hi = text.length;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (ctx.measureText(`${text.slice(0, mid)}…`).width <= maxWidth) lo = mid; else hi = mid - 1;
  }
  return lo > 0 ? `${text.slice(0, lo).trimEnd()}…` : '';
}
