// Application shell for the Kimodo text-to-motion demo page (index.html):
// sidebar (Generate | Visualize | Instructions), 3D viewport, playback bar and
// timeline, wired to the Go demo server's API. Behaviour follows the upstream
// NVIDIA Kimodo / Hugging Face demo (kimodo/demo/ui.py, app.py, generation.py)
// in the LocalAI house style.
//
// State model
// - `segments` [{prompt, frames}] is the single source of truth for the prompt
//   sequence; the sidebar list and the timeline Prompts row both render it.
//   numFrames = sum of segment frames.
// - `motion` is the working motion: a clone of the selected animation (or the
//   rest pose when none is selected). Editing mode, Smooth Path and Snap mutate
//   it in place; the server data is never touched. When the duration changes
//   the motion is resized: frames are kept, re-extension takes frames from the
//   selected animation, and frames past its end hold the last frame (upstream
//   pads constraint snapshots past the motion end with the last frame).
// - `store` (ConstraintStore) holds every constraint; its 'change' event
//   refreshes the timeline and the constraint panel immediately and the
//   viewport ghosts on the next animation frame. Shortening the clip keeps the
//   constraints past the new end (upstream); they are not sent until it grows.
//
// Port decisions (deviations from the upstream demo):
// - CFG: only the two weight sliders. The engine runs separated CFG only, so
//   there is no "Enable CFG" checkbox (upstream's no-CFG mode has no equivalent).
// - G1 models: Post Processing hidden and sent as false; gizmo space forced Local.
// - Timeline edits are re-enabled after a failed generation (upstream left the
//   timeline disabled until the page was reloaded).
// - Snap to Constraint is re-evaluated on every frame / constraint change.
// - While a generation runs the gallery is locked like the other editing
//   controls, so the request's constraints stay in the store until the new
//   animation is shown with them.
// - Selecting a past animation restores its prompts, model, settings and its
//   constraints.json; one without constraints keeps the current constraints
//   (upstream "Load motion" keeps them). Settings a record does not carry
//   (records made before the constraint editor have no CFG weights, Post
//   Processing or root margin; G1 records always store Post Processing off)
//   keep their current values. After our own Generate the new animation is
//   selected and the store is kept as it was sent. On page load the newest
//   animation is shown (as the previous viewer did) with its prompts and model
//   only; the generation settings keep their defaults.
// - Editing mode: the Editor (editing.js) owns the gizmos, the skeleton /
//   opacity switch and the Smooth Path re-application on exit and after a
//   ground-gizmo release; the app mirrors the options in the Visualize tab.
//
// Debug handle for automated tests: window.__kimodo.

import { Skeleton } from './skeleton.js';
import { Motion } from './motion.js';
import { ConstraintStore, TRACKS, TRACK_LABELS, TRACK_COLORS, WARN_FRAMES } from './constraints.js';
import { Timeline, PROMPT_COLORS } from './timeline.js';
import { Viewport } from './viewport.js';
import { Editor } from './editing.js';

const FPS = 30;
const MIN_FRAMES = 60;
const MAX_FRAMES = 300;
const MAX_SEGMENTS = 16;
const DEFAULT_PROMPT = 'A person walks forward.';
const DEFAULT_FRAMES = 180;
const POLL_MS = 2500;
const GALLERY_PAGE = 40;
const VIS_KEY = 'kimodo.editor.visualize';

const $ = (id) => document.getElementById(id);
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const el = (tag, props = {}, ...children) => {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === 'class') node.className = v;
    else if (k === 'style') node.style.cssText = v;
    else if (k.startsWith('on')) node.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null && v !== false) node.setAttribute(k, v === true ? '' : v);
  }
  for (const c of children) if (c !== null && c !== undefined && c !== false) node.append(c);
  return node;
};

// ---------------------------------------------------------------------------
// state
// ---------------------------------------------------------------------------

const state = {
  models: [],
  quantizations: [],
  model: null,          // /api/models entry of the selected motion model
  skeleton: null,
  store: null,
  motion: null,         // working motion (mutated by editing)
  source: null,         // pristine motion of the selected animation
  selected: null,       // selected animation record (null: rest pose)
  segments: [{ prompt: DEFAULT_PROMPT, frames: DEFAULT_FRAMES }],
  frame: 0,
  playing: false,
  speed: 1,
  acc: 0,
  lastTime: 0,
  animations: [],
  gallerySig: '',
  galleryLimit: GALLERY_PAGE,
  active: null,         // generation in flight {id, started}
  editing: false,
  undoAvailable: false,
  visBeforeEdit: null,
  vis: {
    showMesh: true, meshOpacity: 1, showSkeleton: false, showContacts: false,
    showLabels: true, showStartDirection: true, onlyCurrent: false, follow: true,
  },
};
let viewport = null;
let editor = null;
let timeline = null;
let selectToken = 0;
// What the RAF loop last pushed to the viewport / timeline.
const seen = { motion: null, motionVersion: -1, frame: -1, store: null, storeVersion: -1, numFrames: -1, readoutFrames: -1 };

const numFrames = () => state.segments.reduce((n, s) => n + s.frames, 0);
const isG1 = () => (state.model?.skeleton_key || '').startsWith('g1');

// ---------------------------------------------------------------------------
// toasts
// ---------------------------------------------------------------------------

function toast(title, { body = '', kind = 'info', timeout } = {}) {
  const node = el('div', { class: `toast ${kind}`, role: kind === 'error' ? 'alert' : 'status' }, el('b', {}, title), body ? el('span', {}, body) : null);
  $('toasts').append(node);
  const ms = timeout ?? (kind === 'error' ? 9000 : kind === 'warn' ? 6000 : 3200);
  setTimeout(() => {
    node.classList.add('leaving');
    setTimeout(() => node.remove(), 300);
  }, ms);
  while ($('toasts').children.length > 5) $('toasts').firstChild.remove();
}

const errorText = (err) => (err && err.message) || String(err);

// ---------------------------------------------------------------------------
// fetch helpers
// ---------------------------------------------------------------------------

async function fetchJSON(url, { optional = false } = {}) {
  const r = await fetch(url);
  if (optional && r.status === 404) return null;
  if (!r.ok) throw new Error(`${url}: ${r.status} ${(await r.text()).trim()}`);
  return r.json();
}

async function fetchF32(url, { optional = false } = {}) {
  const r = await fetch(url);
  if (optional && r.status === 404) return null;
  if (!r.ok) throw new Error(`${url}: ${r.status} ${(await r.text()).trim()}`);
  const buf = await r.arrayBuffer();
  if (buf.byteLength % 4) throw new Error(`${url}: not a float32 file`);
  return new Float32Array(buf);
}

// ---------------------------------------------------------------------------
// model / skeleton
// ---------------------------------------------------------------------------

function newStore(skeleton) {
  const store = new ConstraintStore(skeleton);
  store.setNumFrames(numFrames());
  store.addEventListener('change', onStoreChange);
  return store;
}

/**
 * Selects a motion model. A different skeleton starts over (upstream clears
 * motions and constraints on a model switch): new store, rest-pose motion.
 * Models sharing the skeleton keep the motion and the constraints.
 */
function setModel(entry) {
  const sameSkeleton = state.skeleton && state.model && state.model.skeleton_key === entry.skeleton_key;
  state.model = entry;
  $('model').value = entry.id;
  updateModelHint();
  if (!sameSkeleton) {
    if (state.editing) exitEditing({ quiet: true });
    state.skeleton = new Skeleton(entry);
    state.store = newStore(state.skeleton);
    state.source = null;
    state.selected = null;
    // Skeleton first, then the motion built on it.
    if (viewport) viewport.setSkeleton(state.skeleton);
    else viewport = new Viewport($('viewport'), state.skeleton);
    setMotion(Motion.rest(state.skeleton, numFrames()));
    onStoreChange();
    updateDownload();
  }
  // G1: no post-processing (forced off upstream) and a Local-only gizmo.
  $('postFolder').hidden = isG1();
  $('gizmoSpaceRow').hidden = isG1();
  if (isG1()) $('gizmoSpace').value = 'local';
}

function setMotion(motion) {
  state.motion = motion;
  if (viewport) viewport.setMotion(motion);
  seen.motion = motion;
  seen.motionVersion = -1;
  if (state.frame >= motion.frames) state.frame = motion.frames - 1;
}

function updateModelHint() {
  const model = state.model;
  if (!model) return;
  $('eyebrow').textContent = `${model.label} · Vulkan`;
  const hint = $('modelHint');
  hint.classList.toggle('license-warning', !model.commercial);
  const terms = model.commercial ? 'commercial use permitted under NVIDIA Open Model License' : '⚠ non-commercial research use only';
  const detail = model.available ? `${model.skeleton} · ${model.upstream} · ` : `${model.skeleton} · ${model.reason || 'not installed'} · `;
  const link = el('a', { href: model.license_url, target: '_blank', rel: 'noreferrer' }, terms);
  hint.replaceChildren(document.createTextNode(detail), link);
}

function updateQuantizationHint() {
  const q = state.quantizations.find((item) => item.id === $('quantization').value);
  if (!q) return;
  const size = q.bytes ? `${(q.bytes / 1073741824).toFixed(2)} GiB · ` : '';
  $('quantizationHint').textContent = q.available ? `${size}${q.description}` : q.reason;
}

function populateSelects() {
  const ms = $('model');
  ms.replaceChildren(...state.models.map((m) => el('option', { value: m.id, disabled: !m.available }, `${m.label}${m.available ? '' : ' — not installed'}`)));
  const qs = $('quantization');
  qs.replaceChildren(...state.quantizations.map((q) => el('option', { value: q.id, disabled: !q.available }, `${q.label}${q.available ? '' : ' — unavailable'}`)));
  const preferred = state.quantizations.find((q) => q.id === 'q8_0' && q.available) ?? state.quantizations.find((q) => q.available);
  if (preferred) qs.value = preferred.id;
  updateQuantizationHint();
}

// ---------------------------------------------------------------------------
// prompt segments
// ---------------------------------------------------------------------------

const normalizeSegments = (segments) => segments.slice(0, MAX_SEGMENTS).map((s) => ({
  prompt: String(s.prompt ?? ''),
  frames: clamp(Math.round(Number(s.frames) || DEFAULT_FRAMES), MIN_FRAMES, MAX_FRAMES),
}));

/** Replaces the prompt sequence; resizes the working motion when the length changes. */
function setSegments(segments, { renderList = true } = {}) {
  const before = numFrames();
  state.segments = normalizeSegments(segments.length ? segments : [{ prompt: DEFAULT_PROMPT, frames: DEFAULT_FRAMES }]);
  if (renderList) renderSegments();
  updateTotal();
  const n = numFrames();
  timeline?.setData({ prompts: state.segments.map((s) => ({ ...s })), numFrames: n });
  if (state.motion && (n !== before || state.motion.frames !== n)) resizeMotion(n);
}

function renderSegments() {
  const list = $('segments');
  const disabled = !!state.active;
  list.replaceChildren(...state.segments.map((seg, i) => {
    const color = PROMPT_COLORS[i % PROMPT_COLORS.length];
    const area = el('textarea', { rows: 2, 'aria-label': `Prompt ${i + 1}`, placeholder: 'Describe the motion', disabled });
    area.value = seg.prompt;
    area.addEventListener('input', () => {
      state.segments[i].prompt = area.value;
      autoGrow(area);
      timeline.setData({ prompts: state.segments.map((s) => ({ ...s })) });
    });
    const frames = el('input', { type: 'number', min: MIN_FRAMES, max: MAX_FRAMES, step: 1, value: seg.frames, 'aria-label': `Prompt ${i + 1} frames`, title: `Frames (${MIN_FRAMES}–${MAX_FRAMES})`, disabled });
    const secs = el('small', {}, `${(seg.frames / FPS).toFixed(1)} s`);
    frames.addEventListener('change', () => {
      const next = state.segments.map((s) => ({ ...s }));
      next[i].frames = clamp(Math.round(Number(frames.value) || seg.frames), MIN_FRAMES, MAX_FRAMES);
      frames.value = next[i].frames;
      setSegments(next);
    });
    const remove = el('button', {
      type: 'button', class: 'secondary icon', title: 'Remove segment', 'aria-label': `Remove prompt ${i + 1}`,
      disabled: disabled || state.segments.length <= 1,
      onclick: () => setSegments(state.segments.filter((_, k) => k !== i)),
    }, '×');
    return el('div', { class: 'segment' },
      el('span', { class: 'swatch', style: `background: rgb(${color.join(',')})` }),
      area,
      el('div', { class: 'frames' }, frames, secs),
      remove);
  }));
  for (const area of list.querySelectorAll('textarea')) autoGrow(area);
  $('addSegment').disabled = disabled || state.segments.length >= MAX_SEGMENTS;
  $('segmentCount').textContent = `${state.segments.length} segment${state.segments.length === 1 ? '' : 's'}`;
}

function autoGrow(area) {
  area.style.height = '0px';
  area.style.height = `${Math.max(58, area.scrollHeight + 2)}px`;
}

function updateTotal() {
  const n = numFrames();
  $('totalDuration').replaceChildren('Total duration: ', el('b', {}, `${(n / FPS).toFixed(1)} s`), el('small', {}, ` · ${n} frames`));
}

/**
 * Resizes the working motion to n frames: existing frames (with any edits)
 * are kept, re-extension uses the selected animation's frames, and frames past
 * both hold the last frame.
 */
function resizeMotion(n) {
  const cur = state.motion;
  if (!cur || cur.frames === n) {
    state.store.setNumFrames(n, cur);
    return;
  }
  const sk = state.skeleton, J = sk.J, src = state.source;
  const local = new Float32Array(n * J * 4), root = new Float32Array(n * 3);
  const withContacts = !!(cur.contacts || src?.contacts);
  const contacts = withContacts ? new Float32Array(n * 4) : null;
  for (let f = 0; f < n; f++) {
    const from = f < cur.frames ? [cur, f] : src && f < src.frames ? [src, f] : null;
    if (from) {
      const [m, g] = from;
      local.set(m.local.subarray(g * J * 4, (g + 1) * J * 4), f * J * 4);
      root.set(m.root.subarray(g * 3, g * 3 + 3), f * 3);
      if (contacts && m.contacts) contacts.set(m.contacts.subarray(g * 4, g * 4 + 4), f * 4);
    } else {
      local.copyWithin(f * J * 4, (f - 1) * J * 4, f * J * 4);
      root.copyWithin(f * 3, (f - 1) * 3, f * 3);
      if (contacts) contacts.copyWithin(f * 4, (f - 1) * 4, f * 4);
    }
  }
  setMotion(new Motion(sk, n, local, root, { contacts }));
  state.store.setNumFrames(n, state.motion);
  if (state.store.smoothPath) state.store.applyDensePath(state.motion, n);
}

// ---------------------------------------------------------------------------
// generation parameters
// ---------------------------------------------------------------------------

/** Links a range slider with its number box. */
function linkSlider(rangeId, numberId, { min, max, decimals = 0, onChange } = {}) {
  const range = $(rangeId), num = $(numberId);
  const set = (v, from) => {
    let x = Number(v);
    if (!Number.isFinite(x)) x = Number(range.value);
    x = clamp(x, min, max);
    x = decimals ? Number(x.toFixed(decimals)) : Math.round(x);
    if (from !== range) range.value = x;
    if (from !== num || Number(num.value) !== x) num.value = x;
    onChange?.(x);
  };
  range.addEventListener('input', () => set(range.value, range));
  num.addEventListener('change', () => set(num.value, null));
  return (v) => set(v, null);
}

const params = {};

function setupParams() {
  const summary = () => {
    $('paramsSummary').textContent = `seed ${$('seed').value} · ${$('steps').value} steps · CFG ${$('textCfg').value}/${$('constraintCfg').value}`;
    $('postSummary').textContent = $('postProcessing').checked ? `on · ${$('rootMargin').value} m` : 'off';
  };
  params.steps = linkSlider('stepsRange', 'steps', { min: 2, max: 1000, onChange: summary });
  params.textCfg = linkSlider('textCfgRange', 'textCfg', { min: 0, max: 5, decimals: 1, onChange: summary });
  params.constraintCfg = linkSlider('constraintCfgRange', 'constraintCfg', { min: 0, max: 5, decimals: 1, onChange: summary });
  params.transition = linkSlider('transitionRange', 'transition', { min: 1, max: 10, onChange: summary });
  $('seed').addEventListener('change', () => {
    $('seed').value = clamp(Math.round(Number($('seed').value) || 0), 0, Number.MAX_SAFE_INTEGER);
    summary();
  });
  $('randomSeed').addEventListener('click', () => {
    $('seed').value = Math.floor(Math.random() * 2 ** 31);
    summary();
  });
  const post = () => {
    $('rootMarginRow').hidden = !$('postProcessing').checked;
    summary();
  };
  $('postProcessing').addEventListener('change', post);
  $('rootMargin').addEventListener('change', () => {
    const v = Number($('rootMargin').value);
    $('rootMargin').value = Number.isFinite(v) ? clamp(v, 0, 1) : 0.04;
    summary();
  });
  params.summary = () => { post(); };
  post();
}

/**
 * Restores the generation settings recorded on an animation; with
 * all = false only the text quantization (the startup auto-selection keeps
 * the page defaults, like the previous viewer). Settings the record does not
 * carry keep their current values: records made before the constraint editor
 * have no CFG weights, Post Processing or root margin, and the server stores
 * Post Processing off for every G1 request (call after setModel, so isG1()
 * describes the record's model).
 */
function restoreParams(a, { all = true } = {}) {
  const q = state.quantizations.find((item) => item.id === a.text_quantization && item.available);
  if (q) { $('quantization').value = q.id; updateQuantizationHint(); }
  if (!all) return;
  if (Number.isSafeInteger(a.seed) && a.seed >= 0) $('seed').value = a.seed;
  else if (Number.isFinite(a.seed)) toast('Seed not restored', { body: 'This animation’s seed is above 2^53 and cannot be edited here; the current seed is kept.', kind: 'warn' });
  if (a.diffusion_steps) params.steps(a.diffusion_steps);
  if (Number.isFinite(a.text_cfg)) params.textCfg(a.text_cfg);
  if (Number.isFinite(a.constraint_cfg)) params.constraintCfg(a.constraint_cfg);
  if (a.transition_frames) params.transition(a.transition_frames);
  if (!isG1()) {
    if (typeof a.post_processing === 'boolean') $('postProcessing').checked = a.post_processing;
    if (Number.isFinite(a.root_margin)) $('rootMargin').value = a.root_margin;
  }
  params.summary();
}

function requestBody() {
  const n = numFrames();
  return {
    segments: state.segments.map((s) => ({ prompt: s.prompt.trim(), frames: s.frames })),
    model: state.model.id,
    text_quantization: $('quantization').value,
    seed: Number($('seed').value) || 0,
    steps: Number($('steps').value),
    transition_frames: Number($('transition').value),
    text_cfg: Number($('textCfg').value),
    constraint_cfg: Number($('constraintCfg').value),
    post_processing: isG1() ? false : $('postProcessing').checked,
    root_margin: Number($('rootMargin').value),
    constraints: state.store.toModelJSON(n),
  };
}

// ---------------------------------------------------------------------------
// generation
// ---------------------------------------------------------------------------

async function generate() {
  if (state.editing || state.active) return;
  const empty = state.segments.findIndex((s) => !s.prompt.trim());
  if (empty >= 0) {
    toast('Prompt missing', { body: `Segment ${empty + 1} has no text.`, kind: 'warn' });
    return;
  }
  if (!state.model?.available) {
    toast('Model not available', { body: state.model?.reason || 'The selected motion model is not installed.', kind: 'error' });
    return;
  }
  let body;
  try {
    body = requestBody();
  } catch (err) {
    toast('Could not build the constraints', { body: errorText(err), kind: 'error' });
    return;
  }
  setStatus('Submitting generation…', 'progress');
  $('generate').disabled = true;
  try {
    const r = await fetch('/api/generate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    if (!r.ok) throw new Error((await r.text()).trim() || `HTTP ${r.status}`);
    const a = await r.json();
    state.active = { id: a.id, started: Date.now(), constraints: body.constraints.length };
    setGenerating(true);
    toast('Waiting for the GPU…', { body: body.constraints.length ? `${a.constraint_count ?? ''} constrained frames sent with the request.`.trim() : '' });
    await refreshGallery();
  } catch (err) {
    setStatus(`Generation failed: ${errorText(err)}`, 'error');
    toast('Generation failed!', { body: errorText(err), kind: 'error' });
    setGenerating(false);
  }
}

/** Locks everything but scrubbing / playback / visualisation while generating. */
function setGenerating(on) {
  timeline?.setData({ enabled: !on });
  for (const id of ['model', 'addSegment', 'editMode', 'clearConstraints', 'loadConstraints', 'smoothPath']) $(id).disabled = on;
  renderSegments();
  renderGallery(true); // gallery items are locked while generating
  updateButtons();
  if (!on) { $('model').disabled = state.editing; }
}

function setStatus(text, kind = '') {
  const s = $('status');
  s.className = `status-line ${kind}`;
  s.textContent = text;
}

function showProgress() {
  const job = state.active;
  if (!job) return;
  const a = state.animations.find((item) => item.id === job.id);
  const seconds = Math.max(0, Math.floor((Date.now() - job.started) / 1000));
  if (!a) { setStatus(`Waiting for the server… ${seconds}s`, 'progress'); return; }
  if (a.status === 'ready') {
    state.active = null;
    setGenerating(false);
    setStatus(`Generated ${a.frames} frames in ${seconds}s.`);
    toast('Motion generation finished!', { kind: 'success' });
    // The store still holds the constraints that were sent (the gallery and
    // every constraint control were locked during the run); only the motion
    // and the record's settings change.
    selectAnimation(a, { loadConstraints: false, quiet: true });
    return;
  }
  if (a.status === 'failed') {
    state.active = null;
    setGenerating(false); // upstream left the timeline disabled here
    setStatus(`Generation failed: ${a.error || 'unknown error'}`, 'error');
    toast('Generation failed!', { body: a.error || '', kind: 'error' });
    return;
  }
  const what = a.status === 'running' ? (a.progress || 'Generating motion') : 'Queued for generation';
  setStatus(`${what}… ${seconds}s elapsed`, 'progress');
}

// ---------------------------------------------------------------------------
// gallery
// ---------------------------------------------------------------------------

async function refreshGallery() {
  try {
    state.animations = await fetchJSON('/api/animations');
  } catch (err) {
    console.warn(err);
    return;
  }
  renderGallery();
  showProgress();
}

function animationText(a) {
  const segs = a.segments?.length ? a.segments : [{ prompt: a.prompt }];
  return segs.map((s) => s.prompt).join(' → ');
}

function renderGallery(force = false) {
  const query = $('gallerySearch').value.trim().toLowerCase();
  const onlyConstrained = $('galleryConstrained').checked;
  let list = state.animations;
  if (query) list = list.filter((a) => animationText(a).toLowerCase().includes(query));
  if (onlyConstrained) list = list.filter((a) => a.constraint_count > 0);
  const shown = list.slice(0, state.galleryLimit);
  const sig = JSON.stringify([shown.map((a) => [a.id, a.status, a.progress, a.error]), list.length, state.selected?.id, state.active?.id]);
  $('galleryCount').textContent = `${list.length}${list.length !== state.animations.length ? ` / ${state.animations.length}` : ''}`;
  if (!force && sig === state.gallerySig) return;
  state.gallerySig = sig;
  const gallery = $('gallery');
  if (!state.animations.length) { gallery.replaceChildren(el('span', { class: 'readout' }, 'No animations yet.')); return; }
  if (!list.length) { gallery.replaceChildren(el('span', { class: 'readout' }, 'No matching animations.')); return; }
  const items = shown.map((a) => {
    const segs = a.segments?.length || 1;
    const selectable = a.status === 'ready' || a.status === 'failed';
    const meta = el('div', { class: 'item-meta' },
      a.status !== 'ready' ? el('span', { class: `chip ${a.status === 'failed' ? 'failed' : 'running'}` }, a.status) : null,
      el('span', { class: 'chip' }, a.model || 'model?'),
      segs > 1 ? el('span', { class: 'chip' }, `${segs} segments`) : null,
      a.constraint_count > 0 ? el('span', { class: 'chip constraints', title: (a.constraint_types || []).join(', ') }, `◆ ${a.constraint_count} constrained frame${a.constraint_count === 1 ? '' : 's'}`) : null,
      el('span', {}, `${a.frames} frames · ${a.diffusion_steps} steps · ${a.text_quantization || 'bf16'} · seed ${a.seed}`));
    const b = el('button', {
      type: 'button',
      'data-id': a.id,
      class: `item ${a.status}${state.selected?.id === a.id ? ' active' : ''}`,
      // Locked during a run: selecting would replace the prompts, settings and
      // constraints the running request was made with.
      disabled: !selectable || !!state.active,
      title: state.active && selectable ? 'Wait for the current generation to finish'
        : a.status === 'failed' ? 'Restore this request (prompts, settings, constraints)' : a.status === 'ready' ? 'Play this animation and restore its settings' : 'Waiting for the generator',
      onclick: () => selectAnimation(a),
    }, el('p', { class: 'item-prompt' }, animationText(a)), meta, a.error ? el('div', { class: 'error' }, a.error) : null);
    return b;
  });
  if (list.length > shown.length) {
    items.push(el('button', {
      type: 'button', class: 'secondary more',
      onclick: () => { state.galleryLimit += GALLERY_PAGE; renderGallery(true); },
    }, `Show more (${list.length - shown.length} more)`));
  }
  gallery.replaceChildren(...items);
}

/**
 * Plays a past animation and restores its prompts / model / settings. Ready:
 * loads the motion and, unless loadConstraints is false, its constraints.json
 * (foot contacts load on demand). Failed: restores the request without a
 * motion. restoreSettings = false keeps seed / steps / CFG / post-processing.
 */
async function selectAnimation(a, { loadConstraints = true, quiet = false, restoreSettings = true } = {}) {
  if (a.status !== 'ready' && a.status !== 'failed') return;
  // Never while a generation runs (the gallery is locked then); the finished
  // run is selected after state.active is cleared.
  if (state.active) return;
  const entry = state.models.find((m) => m.id === a.model);
  if (!entry) {
    toast('Unknown motion model', { body: `${a.model} is not served by this demo.`, kind: 'error' });
    return;
  }
  const token = ++selectToken;
  const base = `/api/animations/${encodeURIComponent(a.id)}`;
  let root = null, rot = null, constraints = null;
  try {
    // constraints.json exists only for requests that carried constraints; foot
    // contacts are fetched on demand (ensureContacts), older records lack them.
    [root, rot, constraints] = await Promise.all([
      a.status === 'ready' ? fetchF32(`${base}/root.f32`) : null,
      a.status === 'ready' ? fetchF32(`${base}/rotations.f32`) : null,
      loadConstraints && a.constraint_count > 0 ? fetchJSON(`${base}/constraints.json`, { optional: true }) : null,
    ]);
  } catch (err) {
    if (token === selectToken) toast('Could not load the animation', { body: errorText(err), kind: 'error' });
    return;
  }
  if (token !== selectToken || state.active) return; // a newer selection or a generation won
  if (state.editing) exitEditing({ quiet: true });

  setModel(entry);
  restoreParams(a, { all: restoreSettings });
  const segments = a.segments?.length ? a.segments : [{ prompt: a.prompt, frames: a.frames }];

  if (a.status === 'ready') {
    let motion;
    try {
      motion = Motion.fromBuffers(state.skeleton, root, rot);
    } catch (err) {
      toast('Invalid motion data', { body: errorText(err), kind: 'error' });
      return;
    }
    state.source = motion;
    state.selected = a;
    state.contactsFor = null;
    setMotion(motion.clone());
    state.frame = 0;
    state.acc = 0;
    state.playing = true;
  }
  // Prompts after the motion: resizes the clone only if the record disagrees.
  setSegments(segments);
  state.store.setNumFrames(numFrames(), state.motion);

  if (constraints) {
    try {
      loadConstraintArray(constraints, { source: `animation ${a.id.slice(0, 8)}`, quiet });
    } catch (err) {
      toast('Could not load the animation’s constraints', { body: errorText(err), kind: 'error' });
    }
  }
  updateDownload();
  renderGallery(true);
  ensureContacts();
  if (!quiet) {
    if (a.status === 'failed') toast('Request restored', { body: 'Prompts, settings and constraints of the failed run are back; fix them and Generate again.' });
  }
}

/**
 * Loads foot_contacts.f32 of the selected animation once "Show foot contacts"
 * is on (records generated before contacts were saved have none).
 */
async function ensureContacts() {
  const a = state.selected;
  if (!a || !state.vis.showContacts || !state.vis.showSkeleton || state.contactsFor === a.id) return;
  state.contactsFor = a.id;
  let data = null;
  try {
    data = await fetchF32(`/api/animations/${encodeURIComponent(a.id)}/contacts.f32`, { optional: true });
  } catch (err) {
    console.warn(err);
  }
  if (state.selected?.id !== a.id || !state.source) return;
  if (!data || data.length !== state.source.frames * 4) {
    toast('No foot contacts for this animation', { body: 'It was generated before foot contacts were saved.', kind: 'warn' });
    return;
  }
  state.source.contacts = data;
  const m = state.motion;
  const c = new Float32Array(m.frames * 4);
  for (let f = 0; f < m.frames; f++) {
    const g = Math.min(f, state.source.frames - 1); // past the end: hold the last frame
    c.set(data.subarray(g * 4, g * 4 + 4), f * 4);
  }
  m.contacts = c;
  viewport.setMotion(m); // redraw with the contacts
}

function updateDownload() {
  const a = state.selected;
  const link = $('download');
  link.hidden = !a;
  $('selectedLabel').textContent = a ? animationText(a) : 'Rest pose · no animation selected';
  $('selectedLabel').title = a ? `${a.id} · ${animationText(a)}` : '';
  if (a) {
    link.href = `/api/animations/${encodeURIComponent(a.id)}/animation.glb`;
    link.download = `kimodo-${a.id}.glb`;
    link.title = 'Download the selected motion as an animated GLB';
  }
}

// ---------------------------------------------------------------------------
// constraints
// ---------------------------------------------------------------------------

function onStoreChange() {
  const store = state.store;
  if (!store) return;
  const counts = {};
  for (const t of TRACKS) counts[t] = store.frameCount(t);
  const warnings = store.warnings();
  timeline?.setData({ items: store.items(), warnings, counts });
  $('counts').replaceChildren(...TRACKS.map((t) => el('div', { class: `count${warnings[t] ? ' warn' : ''}`, title: warnings[t] ? `${WARN_FRAMES}+ constrained frames` : '' },
    el('span', { class: 'dot', style: `background: rgb(${TRACK_COLORS[t].join(',')})` }),
    el('span', { class: 'name' }, TRACK_LABELS[t]),
    el('span', { class: 'n' }, String(counts[t])))));
  $('countWarning').hidden = !TRACKS.some((t) => warnings[t]);
  const total = TRACKS.reduce((n, t) => n + counts[t], 0);
  $('constraintSummary').textContent = total ? `${total} frame${total === 1 ? '' : 's'}` : '';
  $('smoothPath').checked = store.smoothPath;
  updateButtons();
}

/** Refreshes the dense Smooth Path on the motion after a 2D-root edit (upstream timing). */
function afterRootEdit() {
  if (state.store.smoothPath) state.store.applyDensePath(state.motion, numFrames());
}

const itemById = (id) => state.store.items().find((it) => it.id === id);

const timelineCallbacks = {
  seek(f) { setFrame(f); },
  add(track, f) {
    const id = state.store.addKeyframe(track, f, state.motion);
    if (id && track === 'root2d') afterRootEdit();
    return id;
  },
  addInterval(track, s, e) {
    const id = state.store.addInterval(track, s, e, state.motion);
    if (id && track === 'root2d') afterRootEdit();
    return id;
  },
  move(id, s, e) {
    const item = itemById(id);
    if (state.store.move(id, s, e, state.motion) && item?.track === 'root2d') afterRootEdit();
  },
  remove(id) {
    const item = itemById(id);
    if (state.store.remove(id) && item?.track === 'root2d') afterRootEdit();
  },
  promptsChange(segments) { setSegments(segments); },
};

/** Loads a constraints.json array into the store (replacing it). */
function loadConstraintArray(array, { source = 'file', quiet = false } = {}) {
  if (array && !Array.isArray(array) && Array.isArray(array.constraints)) array = array.constraints;
  const result = state.store.loadModelJSON(array, state.motion);
  // Frames past the clip end: grow the last segment when it can. What still
  // does not fit stays on the timeline past the end and is not sent.
  const maxFrame = state.store.items().reduce((m, it) => Math.max(m, it.end), -1);
  const n = numFrames();
  if (maxFrame >= n) {
    const segs = state.segments.map((s) => ({ ...s }));
    const last = segs[segs.length - 1];
    last.frames = Math.min(MAX_FRAMES, last.frames + (maxFrame + 1 - n));
    setSegments(segs);
    if (maxFrame >= numFrames()) toast('Some constraints are past the end of the clip', { body: `Frames after ${numFrames() - 1} do not fit in the prompt segments. They stay on the timeline but are not sent; add a segment to use them.`, kind: 'warn' });
  }
  state.store.setNumFrames(numFrames(), state.motion);
  const frames = TRACKS.reduce((k, t) => k + state.store.frameCount(t), 0);
  if (!quiet) toast('Constraints loaded!', { body: `${result.objects} constraint set${result.objects === 1 ? '' : 's'}, ${frames} constrained frames from the ${source}.`, kind: 'success' });
  return result;
}

function saveConstraints() {
  const array = state.store.toModelJSON(numFrames());
  if (!array.length) {
    toast('Nothing to save', { body: 'Add constraints on the timeline first.', kind: 'warn' });
    return;
  }
  const blob = new Blob([JSON.stringify(array)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = el('a', { href: url, download: 'constraints.json' });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
  toast('Constraints saved!', { body: `${array.length} constraint set${array.length === 1 ? '' : 's'} in constraints.json.`, kind: 'success' });
}

async function loadConstraintsFile(file) {
  try {
    const array = JSON.parse(await file.text());
    loadConstraintArray(array, { source: file.name });
  } catch (err) {
    toast('Failed to load constraints', { body: errorText(err), kind: 'error', timeout: 10000 });
  }
}

// ---------------------------------------------------------------------------
// editing mode
// ---------------------------------------------------------------------------

function enterEditing() {
  if (state.editing || state.active) return;
  try {
    editor.enter(isG1() ? 'local' : $('gizmoSpace').value);
  } catch (err) {
    toast('Could not enter editing mode', { body: errorText(err), kind: 'error' });
    return;
  }
  state.editing = true;
  state.undoAvailable = false;
  state.playing = false;
  // Upstream: skeleton on, mesh opacity 0.65 while editing.
  state.visBeforeEdit = { showSkeleton: state.vis.showSkeleton, meshOpacity: state.vis.meshOpacity };
  setVis({ showSkeleton: true, meshOpacity: 0.65 }, { persist: false });
  toast('Entered editing mode', { body: 'You can now modify pose or path constraints.' });
  updateButtons();
}

function exitEditing({ quiet = false } = {}) {
  if (!state.editing) return;
  try { editor.exit(); } catch (err) { console.error(err); }
  state.editing = false;
  state.undoAvailable = false;
  if (state.visBeforeEdit) setVis({ ...state.visBeforeEdit }, { persist: false });
  state.visBeforeEdit = null;
  // (Editor#exit re-applies the Smooth Path to the motion, as upstream does.)
  if (!quiet) toast('Exited editing mode');
  updateButtons();
}

function updateButtons() {
  const editing = state.editing, busy = !!state.active;
  const gen = $('generate');
  gen.disabled = editing || busy;
  gen.textContent = editing ? 'Generate Disabled In Editing Mode' : busy ? 'Generating…' : 'Generate';
  const edit = $('editMode');
  edit.textContent = editing ? 'Exit Editing Mode' : 'Enter Editing Mode';
  edit.classList.toggle('active-mode', editing);
  edit.disabled = busy && !editing;
  $('gizmoSpace').disabled = editing;
  $('resetPose').disabled = !editing || editor?.canReset === false;
  $('undoMove').disabled = !editing || !(editor?.canUndo ?? state.undoAvailable);
  $('model').disabled = editing || busy;
  $('modeBanner').hidden = !editing;
  updateSnap();
  updateHeadingRow();
}

/**
 * Root 2D options -> heading row: the optional heading of the 2D Root waypoint
 * at the current frame. "Edit heading" (editing mode) shows the editor's
 * heading ring - the same as clicking the waypoint ring in the view, which the
 * ground gizmo usually covers; "Clear" removes the heading.
 */
function updateHeadingRow() {
  const r = state.store?.entriesAt(state.frame).root2d;
  const deg = (rad) => `${Math.round((rad * 180) / Math.PI)}°`;
  $('headingReadout').textContent = !r ? 'No 2D Root at this frame'
    : r.heading === null ? `2D Root @ ${state.frame}: no heading` : `2D Root @ ${state.frame}: heading ${deg(r.heading)}`;
  const selected = !!editor?.headingSelected;
  $('editHeading').disabled = !r || !state.editing;
  $('editHeading').textContent = selected ? 'Done' : 'Edit heading';
  $('editHeading').classList.toggle('active-mode', selected);
  $('clearHeading').disabled = !r || r.heading === null || !!state.active;
}

function updateSnap() {
  $('snap').disabled = !state.store || !state.store.hasConstraintAt(state.frame) || !!state.active;
}

// ---------------------------------------------------------------------------
// visualisation options
// ---------------------------------------------------------------------------

const VIS_INPUTS = {
  showMesh: 'showMesh', showSkeleton: 'showSkeleton', showContacts: 'showContacts', showLabels: 'showLabels',
  showStartDirection: 'showStartDirection', follow: 'follow',
};

function setVis(patch, { persist = true } = {}) {
  Object.assign(state.vis, patch);
  for (const [key, id] of Object.entries(VIS_INPUTS)) $(id).checked = !!state.vis[key];
  $('onlyCurrentV').checked = $('onlyCurrentC').checked = !!state.vis.onlyCurrent;
  $('meshOpacity').value = state.vis.meshOpacity;
  $('meshOpacityOut').textContent = Number(state.vis.meshOpacity).toFixed(2);
  // Upstream shows "Show foot contacts" only together with the skeleton.
  $('showContacts').disabled = !state.vis.showSkeleton;
  $('showContactsRow').classList.toggle('disabled', !state.vis.showSkeleton);
  viewport?.setOptions({ ...state.vis, showContacts: state.vis.showContacts && state.vis.showSkeleton });
  ensureContacts();
  if (persist) {
    try {
      const { showSkeleton, meshOpacity, ...rest } = state.vis;
      const saved = state.editing && state.visBeforeEdit ? { ...rest, ...state.visBeforeEdit } : { ...rest, showSkeleton, meshOpacity };
      localStorage.setItem(VIS_KEY, JSON.stringify(saved));
    } catch { /* storage unavailable */ }
  }
}

function loadVis() {
  try {
    const saved = JSON.parse(localStorage.getItem(VIS_KEY) || 'null');
    if (saved && typeof saved === 'object') {
      for (const k of Object.keys(state.vis)) if (typeof saved[k] === typeof state.vis[k]) state.vis[k] = saved[k];
    }
  } catch { /* ignore */ }
}

// ---------------------------------------------------------------------------
// playback
// ---------------------------------------------------------------------------

function setFrame(f) {
  const n = state.motion?.frames ?? 1;
  state.frame = clamp(Math.round(f), 0, n - 1);
  state.acc = 0;
}

function setPlaying(on) {
  state.playing = on;
  state.acc = 0;
  updatePlayButton();
}

function updatePlayButton() {
  const b = $('play');
  const playing = state.playing;
  b.setAttribute('aria-label', playing ? 'Pause' : 'Play');
  b.innerHTML = playing
    ? '<svg viewBox="0 0 14 14" aria-hidden="true"><rect x="2" y="1.5" width="3.6" height="11" rx="1"/><rect x="8.4" y="1.5" width="3.6" height="11" rx="1"/></svg>'
    : '<svg viewBox="0 0 14 14" aria-hidden="true"><path d="M3 1.6v10.8a.8.8 0 0 0 1.2.7l8.6-5.4a.8.8 0 0 0 0-1.4L4.2.9A.8.8 0 0 0 3 1.6z"/></svg>';
}

function setSpeed(speed) {
  state.speed = speed;
  for (const b of document.querySelectorAll('[data-speed-group] button')) b.setAttribute('aria-pressed', String(Number(b.dataset.speed) === speed));
}

let shownPlaying = null;

function updateReadout() {
  const n = state.motion?.frames ?? 0;
  $('frameReadout').innerHTML = `frame ${state.frame} / ${n} <small>· ${(state.frame / FPS).toFixed(2)} s</small>`;
}

let loopErrors = 0;

function tick(time) {
  try {
    const dt = state.lastTime ? Math.min(0.1, (time - state.lastTime) / 1000) : 0;
    state.lastTime = time;
    const motion = state.motion;
    const n = motion.frames;
    if (state.playing && n > 1) {
      state.acc += dt * FPS * state.speed;
      const steps = Math.floor(state.acc);
      if (steps > 0) {
        state.acc -= steps;
        state.frame = (state.frame + steps) % n; // loops at the end (upstream)
      }
    }
    if (state.frame >= n) state.frame = n - 1;
    if (shownPlaying !== state.playing) { shownPlaying = state.playing; updatePlayButton(); }

    const frameChanged = state.frame !== seen.frame;
    if (motion !== seen.motion) { viewport.setMotion(motion); seen.motion = motion; seen.motionVersion = -1; }
    if (frameChanged || motion.version !== seen.motionVersion) {
      viewport.setFrame(state.frame);
      editor.refresh?.();
      seen.motionVersion = motion.version;
    }
    if (frameChanged) {
      seen.frame = state.frame;
      timeline.setData({ frame: state.frame });
      timeline.flush(); // playhead in step with the 3D view / readout
      if (state.editing) updateButtons(); // Reset depends on the frame
      else { updateSnap(); updateHeadingRow(); }
    }
    if (frameChanged || n !== seen.readoutFrames) {
      seen.readoutFrames = n;
      updateReadout();
    }
    const store = state.store;
    if (store !== seen.store || store.version !== seen.storeVersion || n !== seen.numFrames) {
      viewport.setConstraints(store, n);
      seen.store = store; seen.storeVersion = store.version; seen.numFrames = n;
      updateSnap();
    }
    viewport.render();
  } catch (err) {
    // Keep the loop alive; report the first few failures.
    if (loopErrors++ < 3) {
      console.error(err);
      toast('Viewer error', { body: errorText(err), kind: 'error' });
    }
  }
  requestAnimationFrame(tick);
}

// ---------------------------------------------------------------------------
// wiring
// ---------------------------------------------------------------------------

function isTyping(target) {
  if (!target || !(target instanceof Element)) return false;
  if (target.isContentEditable) return true;
  const tag = target.tagName;
  if (tag === 'TEXTAREA' || tag === 'SELECT') return true;
  if (tag === 'INPUT') return !['checkbox', 'radio', 'button', 'submit', 'reset', 'file'].includes(target.type);
  return false;
}

function wireUI() {
  // Tabs
  const tabs = ['generate', 'visualize', 'instructions'];
  for (const name of tabs) {
    $(`tab-${name}`).addEventListener('click', () => {
      for (const other of tabs) {
        $(`tab-${other}`).setAttribute('aria-selected', String(other === name));
        $(`panel-${other}`).hidden = other !== name;
      }
    });
  }

  $('model').addEventListener('change', () => {
    const entry = state.models.find((m) => m.id === $('model').value);
    if (!entry || entry.id === state.model?.id) return;
    const switching = entry.skeleton_key !== state.model?.skeleton_key;
    setModel(entry);
    if (switching) toast('Model switched', { body: `${entry.skeleton}: motion and constraints were reset.` });
  });
  $('quantization').addEventListener('change', updateQuantizationHint);
  $('addSegment').addEventListener('click', () => {
    if (state.segments.length >= MAX_SEGMENTS) return;
    setSegments([...state.segments, { prompt: DEFAULT_PROMPT, frames: DEFAULT_FRAMES }]);
    const areas = $('segments').querySelectorAll('textarea');
    areas[areas.length - 1]?.focus();
    areas[areas.length - 1]?.select();
  });
  $('generate').addEventListener('click', generate);

  // Constraints panel
  $('editMode').addEventListener('click', () => (state.editing ? exitEditing() : enterEditing()));
  $('snap').addEventListener('click', () => {
    let kind = null;
    try {
      // In editing mode the editor snaps (gizmos, undo); otherwise the store does.
      kind = editor.active ? editor.snapToConstraint() : state.store.snapToConstraint(state.frame, state.motion);
    } catch (err) {
      toast('Snap failed', { body: errorText(err), kind: 'error' });
      return;
    }
    // An End-Effector slot holding only a Hips pin is named like its viewport label.
    const eeTypes = kind === 'ee' ? state.store.entriesAt(state.frame).ee?.types : null;
    const eeName = eeTypes && eeTypes.size === 1 && eeTypes.has('hips') ? 'Hips' : 'end-effector';
    if (kind) toast('Snapped to constraint', { body: typeof kind === 'string' ? `Pose copied from the ${kind === 'ee' ? eeName : kind === 'root2d' ? '2D root' : 'full-body'} constraint at frame ${state.frame}.` : '' });
  });
  $('resetPose').addEventListener('click', () => {
    try {
      if (editor.resetPose() === false) toast('Nothing to reset', { body: 'The pose at this frame has not been edited in this session.' });
    } catch (err) {
      toast('Reset failed', { body: errorText(err), kind: 'error' });
    }
    updateButtons();
  });
  $('undoMove').addEventListener('click', () => {
    try { editor.undoDrag(); } catch (err) { toast('Undo failed', { body: errorText(err), kind: 'error' }); }
    state.undoAvailable = false;
    updateButtons();
  });
  $('editHeading').addEventListener('click', () => {
    if (editor.headingSelected) editor.deselect();
    else if (!editor.selectWaypoint()) toast('No 2D Root here', { body: 'Add a 2D Root keyframe at this frame first.', kind: 'warn' });
    updateButtons();
  });
  $('clearHeading').addEventListener('click', () => {
    // The editor's version also re-syncs its gizmos; outside editing the store is enough.
    const ok = editor.active ? editor.clearRootHeading() : state.store.setRootHeading(state.frame, null);
    if (ok) toast('Heading cleared', { body: `2D Root @ ${state.frame} constrains the position only.` });
    updateButtons();
  });
  $('smoothPath').addEventListener('change', () => {
    try {
      state.store.setSmoothPath($('smoothPath').checked, state.motion, numFrames());
    } catch (err) {
      toast('Smooth path failed', { body: errorText(err), kind: 'error' });
    }
    $('smoothPath').checked = state.store.smoothPath;
  });
  for (const id of ['onlyCurrentC', 'onlyCurrentV']) $(id).addEventListener('change', () => setVis({ onlyCurrent: $(id).checked }));
  $('clearConstraints').addEventListener('click', () => {
    const had = state.store.items().length;
    state.store.clear();
    if (had) toast('All constraints cleared');
  });
  $('saveConstraints').addEventListener('click', saveConstraints);
  $('loadConstraints').addEventListener('click', () => $('constraintsFile').click());
  $('constraintsFile').addEventListener('change', () => {
    const file = $('constraintsFile').files?.[0];
    $('constraintsFile').value = '';
    if (file) loadConstraintsFile(file);
  });

  // Gallery
  $('gallerySearch').addEventListener('input', () => { state.galleryLimit = GALLERY_PAGE; renderGallery(); });
  $('galleryConstrained').addEventListener('change', () => { state.galleryLimit = GALLERY_PAGE; renderGallery(); });

  // Visualize
  for (const [key, id] of Object.entries(VIS_INPUTS)) $(id).addEventListener('change', () => setVis({ [key]: $(id).checked }));
  $('meshOpacity').addEventListener('input', () => setVis({ meshOpacity: Number($('meshOpacity').value) }));
  const setFov = (deg) => {
    viewport.camera.fov = deg;
    viewport.camera.updateProjectionMatrix();
    $('fov').value = deg;
    $('fovOut').textContent = String(Math.round(deg));
  };
  $('fov').addEventListener('input', () => setFov(Number($('fov').value)));
  $('resetCamera').addEventListener('click', () => {
    viewport.resetCamera();
    setFov(viewport.camera.fov); // the upstream view: 45 degrees
  });

  // Playback
  $('play').addEventListener('click', () => setPlaying(!state.playing));
  for (const b of document.querySelectorAll('[data-speed-group] button')) b.addEventListener('click', () => setSpeed(Number(b.dataset.speed)));
  window.addEventListener('keydown', (e) => {
    if (e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey || isTyping(e.target)) return;
    if (e.code === 'Space' || e.key === ' ') {
      e.preventDefault();
      if (!e.repeat) setPlaying(!state.playing);
    } else if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
      e.preventDefault();
      setPlaying(false);
      setFrame(state.frame + (e.key === 'ArrowLeft' ? -1 : 1));
    }
  });
}

function wireEditor() {
  editor = new Editor(viewport, () => ({ motion: state.motion, frame: state.frame, store: state.store, numFrames: numFrames() }));
  editor.addEventListener('dragstart', () => {
    state.undoAvailable = true;
    state.playing = false;
    updateButtons();
  });
  // The editor re-applies the Smooth Path itself after a ground-gizmo release.
  editor.addEventListener('dragend', (e) => {
    updateButtons();
    if (e.detail.heightUnconstrained) {
      toast('Height not constrained', {
        body: `Only a Hips, Full-Body or hand/foot keyframe at frame ${e.detail.frame} sends the hip height to the model (2D Root is ground-plane only).`,
        kind: 'warn',
      });
    }
  });
  editor.addEventListener('change', () => updateButtons());
  editor.addEventListener('select', () => updateHeadingRow());
}

// ---------------------------------------------------------------------------
// startup
// ---------------------------------------------------------------------------

async function main() {
  loadVis();
  setupParams();
  wireUI();
  updatePlayButton();
  try {
    [state.models, state.quantizations, state.animations] = await Promise.all([
      fetchJSON('/api/models'),
      fetchJSON('/api/text-quantizations'),
      fetchJSON('/api/animations').catch(() => []),
    ]);
  } catch (err) {
    document.body.append(el('div', { class: 'fatal' }, `Could not reach the demo server: ${errorText(err)}`));
    throw err;
  }
  if (!state.models.length) {
    document.body.append(el('div', { class: 'fatal' }, 'The server lists no motion models.'));
    return;
  }
  populateSelects();

  // Start on the newest ready animation (as the previous viewer did), else on
  // the rest pose of the first installed model.
  const initial = state.animations.find((a) => a.status === 'ready' && state.models.some((m) => m.id === a.model));
  const model = state.models.find((m) => m.id === initial?.model) ?? state.models.find((m) => m.available) ?? state.models[0];
  setModel(model);
  wireEditor();
  timeline = new Timeline($('timeline'), timelineCallbacks, {
    minFrames: MIN_FRAMES, maxFrames: MAX_FRAMES, maxSegments: MAX_SEGMENTS,
    defaultPrompt: DEFAULT_PROMPT, defaultFrames: DEFAULT_FRAMES, fps: FPS,
  });
  setSegments(state.segments);
  onStoreChange();
  setVis({}, { persist: false });
  setSpeed(1);
  updateButtons();
  updateReadout();
  updateDownload();
  renderGallery(true);
  requestAnimationFrame(tick);
  if (initial) await selectAnimation(initial, { quiet: true, restoreSettings: false });
  setInterval(refreshGallery, POLL_MS);
  setInterval(() => { if (state.active) showProgress(); }, 1000);
}

// Debug / test handles (window.__kimodoEditor is the Editor instance).
window.__kimodo = {
  state,
  get viewport() { return viewport; },
  get editor() { return editor; },
  get timeline() { return timeline; },
  get store() { return state.store; },
  setFrame,
  selectAnimation,
};
Object.defineProperty(window, '__kimodoEditor', { get: () => editor, configurable: true });

main().catch((err) => console.error(err));
