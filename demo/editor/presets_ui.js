// Parametric path presets dialog (Motorica-style "Parametric presets") for the
// Kimodo demo page: a native modal <dialog> with a searchable grid / list of the
// presets of presets.js and a detail view (live top-down preview, a form
// generated from the preset's ParamDefs, the editable prompt, a summary, the
// warnings) that hands a Built preset to the app (app.js applyPreset). Browser
// only; the preset maths (simulation, pins, prompts, SVG) is the DOM-free
// presets.js.
//
// API
//   new PresetsDialog({onApply, canGenerate, canApply, skeletonKey, pose, keepVisible})
//     onApply(built, {generate})  Built preset of poses.js buildPosedPreset (a
//                                 presets.js Built + `pose`) with `prompt` = the
//                                 dialog's prompt text; return false to keep the
//                                 dialog open (refused/failed).
//     canGenerate()               false disables "Apply & Generate" (generation
//                                 running or editing mode).
//     canApply()                  optional; false disables both buttons (e.g. a
//                                 generation is running). Default: always true.
//     skeletonKey()               optional; the skeleton key the preset will be
//                                 built for (default 'soma30'). A non-SOMA key
//                                 previews the path-only build and its warning.
//     pose()                      optional; the selected pose {id, params} or null:
//                                 presets are built (and previewed) with it.
//     keepVisible                 optional elements (the toast stack) moved into
//                                 the dialog while it is open, so they render
//                                 above the modal backdrop (top layer).
//   #open(state?)                 opens the dialog: the preset list, or with
//                                 state = {id, params, prompt?} the detail view
//                                 of a previously applied preset.
//   #close(), #isOpen, #element (the <dialog>), #syncGenerate() (re-reads
//   canApply / canGenerate; the app calls it whenever they may change).
//   presetReadout(preset)         'Forward turn · 90° left · 4.0 s' for
//                                 {id, params, label}.
//
// Conventions
// - The grid | list choice is kept in localStorage (VIEW_KEY); every access is
//   inside try/catch and the dialog works without storage.
// - Focus: the search box on open, the Back button in the detail view, the
//   preset's card after Back, the opener once the dialog closes. Esc closes
//   (native cancel), as do the x button and a click on the backdrop. Key
//   events stop at the dialog, so the page's window shortcuts stay quiet.
// - Every control writes into the detail's params; each change rebuilds with
//   buildPreset (which normalizes) and writes the normalized values back into
//   the controls (sub-durations are fitted into the clip, so e.g. a shorter
//   Duration pulls "Turn at" in). A number box being typed in is not rewritten
//   until it commits (change). Range sliders span the legal values given the
//   other params (min and max).
// - The prompt follows the generated text until the user edits it; "Use
//   generated prompt" returns to it. An empty prompt applies the generated one.

import { FPS, PRESETS, presetById, defaultParams, normalizeParams, simulatePreset, presetSVG } from './presets.js';
import { buildPosedPreset, stripPosePhrases } from './poses.js';

const VIEW_KEY = 'kimodo.editor.presetsView';
/** Collapsible ParamDef groups shown after the 'main' params. */
const GROUPS = Object.freeze([['dynamics', 'Dynamics'], ['pins', 'Pose pins']]);
/** SVG sizes (viewBox units; CSS scales them): card thumbnails and the detail preview. */
const THUMB = Object.freeze({ width: 176, height: 110 });
const PREVIEW = Object.freeze({ width: 520, height: 360 });
const ICONS = Object.freeze({
  grid: '<svg viewBox="0 0 16 16" aria-hidden="true"><rect x="1.5" y="1.5" width="5.5" height="5.5" rx="1.2"/><rect x="9" y="1.5" width="5.5" height="5.5" rx="1.2"/><rect x="1.5" y="9" width="5.5" height="5.5" rx="1.2"/><rect x="9" y="9" width="5.5" height="5.5" rx="1.2"/></svg>',
  list: '<svg viewBox="0 0 16 16" aria-hidden="true"><rect x="1.5" y="2" width="13" height="2.6" rx="1.1"/><rect x="1.5" y="6.7" width="13" height="2.6" rx="1.1"/><rect x="1.5" y="11.4" width="13" height="2.6" rx="1.1"/></svg>',
});

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

let uid = 0;
const nextId = (prefix) => `${prefix}-${++uid}`;
const stepDecimals = (step) => (String(step).split('.')[1] || '').length;
/** A number shown with the decimals of its ParamDef step (4 -> '4.0' for step 0.1). */
const fixed = (v, step) => Number(v).toFixed(stepDecimals(step));

function loadView() {
  try {
    return localStorage.getItem(VIEW_KEY) === 'list' ? 'list' : 'grid';
  } catch {
    return 'grid';
  }
}

function saveView(view) {
  try { localStorage.setItem(VIEW_KEY, view); } catch { /* storage unavailable */ }
}

/** Sidebar readout of an applied preset {id, params, label}: 'Forward turn · 90° left · 4.0 s'. */
export function presetReadout(preset) {
  if (!preset) return '';
  let def;
  try { def = presetById(preset.id); } catch { return preset.label || ''; }
  const p = normalizeParams(preset.id, preset.params);
  const parts = [def.label];
  if ('laps' in p) parts.push(`${p.speed} m/s`, `${p.laps} ${p.laps === 1 ? 'lap' : 'laps'} ${p.side}, ${p.radius} m radius`);
  else if ('angle' in p) parts.push(`${p.angle}° ${p.side}`);
  else if ('rate' in p) parts.push(`${p.rate}°/s ${p.side}`);
  else if ('speed2' in p) parts.push(`${p.speed} → ${p.speed2} m/s`);
  else if ('speed' in p) parts.push(`${p.speed} m/s`);
  parts.push(`${(Math.round(p.duration * FPS) / FPS).toFixed(1)} s`);
  return parts.join(' · ');
}

/** "ends 2.56 m forward, 2.20 m left" (+X is the character's left). */
function endText([x, z]) {
  const parts = [];
  if (Math.abs(z) >= 0.005) parts.push(`${Math.abs(z).toFixed(2)} m ${z > 0 ? 'forward' : 'back'}`);
  if (Math.abs(x) >= 0.005) parts.push(`${Math.abs(x).toFixed(2)} m ${x > 0 ? 'left' : 'right'}`);
  return parts.length ? parts.join(', ') : 'on the start point';
}

function turnText(deg) {
  if (Math.abs(deg) < 0.5) return 'none';
  return `${Math.round(Math.abs(deg))}° ${deg > 0 ? 'left' : 'right'}`;
}

function summaryParts(built) {
  const s = built.summary;
  const stat = (...children) => el('span', { class: 'pd-stat' }, ...children);
  return [
    stat(el('b', {}, String(built.frames)), ' frames'),
    stat(el('b', {}, s.duration_s.toFixed(1)), ' s'),
    stat(el('b', {}, s.distance_m.toFixed(2)), ' m travelled'),
    stat('ends ', el('b', {}, endText(s.end))),
    stat('net turn ', el('b', {}, turnText(s.net_turn_deg))),
  ];
}

function pinsText(built) {
  const parts = [`2D root path on all ${built.frames} frames with a heading`];
  const hips = built.pins.filter((p) => p.kind === 'hips').map((p) => p.frame);
  const idle = built.pins.filter((p) => p.kind === 'fullbody' && !/\.walk/.test(p.pose)).map((p) => p.frame);
  const walk = built.pins.filter((p) => p.kind === 'fullbody' && /\.walk/.test(p.pose)).map((p) => p.frame);
  if (hips.length) parts.push(`Hips pin @ ${hips.join(', ')}`);
  if (idle.length) parts.push(`idle pose @ ${idle.join(', ')}`);
  if (walk.length) parts.push(`walk pose @ ${walk.join(', ')}`);
  const hands = built.pins.filter((p) => p.kind === 'hands').map((p) => p.frame);
  if (hands.length) parts.push(`hand guide pins @ ${hands.join(', ')}`);
  const pose = built.pose ? ` · pose: ${built.pose.label}${built.pose.layer ? ' (+ pose layer after generation)' : ''}` : '';
  return `Constraints: ${parts.join(' · ')}${pose}`;
}

export class PresetsDialog {
  constructor({ onApply, canGenerate = () => true, canApply = () => true, skeletonKey = () => 'soma30', pose = () => null, keepVisible = [] } = {}) {
    this.onApply = onApply;
    this.pose = pose;
    this.canGenerate = canGenerate;
    this.canApply = canApply;
    this.skeletonKey = skeletonKey;
    this.keepVisible = keepVisible.filter(Boolean);
    this.view = loadView();
    this.current = null; // detail view: {def, params, built, promptDirty, controls: Map, groups: []}
    this.opener = null;
    this.cards = new Map(); // preset id -> card button (built on the first open)
    this._homes = []; // [node, parent, nextSibling] of the keepVisible elements while open
    this._build();
  }

  get isOpen() { return this.dialog.open; }
  get element() { return this.dialog; }

  /** Opens the list, or the detail view of state = {id, params, prompt?}. */
  open(state = null) {
    if (!this.dialog.open) {
      this.opener = document.activeElement instanceof HTMLElement && document.activeElement !== document.body ? document.activeElement : null;
      this._ensureCards();
      this._adoptOverlays();
      this.dialog.showModal();
    }
    let detail = false;
    if (state && state.id) {
      try {
        this._showDetail(state.id, state);
        detail = true;
      } catch (err) {
        console.warn(err);
      }
    }
    if (!detail) {
      this._showBrowse();
      this.search.focus();
      this.search.select();
    }
    this.syncGenerate();
  }

  close() {
    if (this.dialog.open) this.dialog.close();
  }

  /** Re-reads canApply / canGenerate into the footer buttons. */
  syncGenerate() {
    const ready = !!this.current?.built;
    const applyOk = !!this.canApply();
    const genOk = applyOk && !!this.canGenerate();
    this.applyBtn.disabled = !ready || !applyOk;
    this.applyBtn.title = applyOk ? 'Load the path, pose pins, prompt and settings into the editor' : 'Wait for the current generation to finish';
    this.generateBtn.disabled = !ready || !genOk;
    this.generateBtn.title = genOk ? 'Apply the preset and start generating'
      : applyOk ? 'Generating is disabled in editing mode: use Apply, then Generate' : 'Wait for the current generation to finish';
    this.footNote.textContent = applyOk ? '' : 'A generation is running; presets can be applied when it finishes.';
  }

  // -------------------------------------------------------------------------
  // DOM
  // -------------------------------------------------------------------------

  _build() {
    const titleId = nextId('pd-title');
    this.title = el('h2', { id: titleId }, 'Parametric presets');
    this.count = el('span', { class: 'pd-count' }, `${PRESETS.length} presets`);
    this.badge = el('span', { class: 'chip experimental', hidden: true }, 'Experimental');
    this.back = el('button', {
      type: 'button', class: 'secondary small pd-back', hidden: true, 'aria-label': 'Back to all presets',
      onclick: () => this._showBrowse({ focus: true }),
    }, '← All presets');
    const close = el('button', { type: 'button', class: 'secondary icon pd-close', title: 'Close (Esc)', 'aria-label': 'Close', onclick: () => this.close() }, '×');
    const head = el('header', { class: 'pd-head' }, this.back, el('div', { class: 'pd-titles' }, this.title, this.count, this.badge), close);

    // ---- browse view
    this.search = el('input', { type: 'search', placeholder: 'Search presets…', 'aria-label': 'Search presets', autocomplete: 'off', spellcheck: 'false' });
    this.search.addEventListener('input', () => this._filter());
    this.viewButtons = {};
    const toggle = el('div', { class: 'seg-toggle', role: 'group', 'aria-label': 'Layout' }, ...['grid', 'list'].map((v) => {
      const name = v === 'grid' ? 'Grid view' : 'List view';
      const b = el('button', { type: 'button', 'aria-label': name, title: name, 'aria-pressed': 'false', onclick: () => this._setView(v) });
      b.innerHTML = ICONS[v];
      this.viewButtons[v] = b;
      return b;
    }));
    this.items = el('div', { class: 'pd-items' });
    this.empty = el('p', { class: 'pd-empty', hidden: true });
    this.browse = el('section', { class: 'pd-browse', 'aria-label': 'All presets' },
      el('div', { class: 'pd-toolbar' }, this.search, toggle),
      el('div', { class: 'pd-scroll' }, el('div', { class: 'pd-pad' }, this.items, this.empty)));

    // ---- detail view
    this.preview = el('div', { class: 'pd-preview' });
    const legend = el('div', { class: 'pd-legend' },
      el('span', {}, el('i', { class: 'lg-start' }), 'start'),
      el('span', {}, el('i', { class: 'lg-end' }), 'end'),
      el('span', {}, el('i', { class: 'lg-facing' }), 'facing every 0.5 s'),
      el('span', {}, 'top view · forward is up, the character’s left is left · 1 m grid'));
    this.summary = el('div', { class: 'pd-summary' });
    this.pinsLine = el('p', { class: 'pd-pins' });
    this.warnings = el('ul', { class: 'pd-warnings', hidden: true });
    this.experimental = el('p', { class: 'pd-experimental', hidden: true },
      el('b', {}, 'Experimental. '), 'Checked on the GPU at the default parameters (2 seeds and the mirrored side); expect more variation across seeds and parameters than with the core presets.');
    this.desc = el('p', { class: 'pd-desc' });
    this.form = el('div', { class: 'pd-form' });
    const promptId = nextId('pd-prompt');
    this.promptArea = el('textarea', { id: promptId, rows: 3 });
    this.promptArea.addEventListener('input', () => this._promptEdited());
    this.promptState = el('span', { class: 'pd-prompt-state' });
    this.promptReset = el('button', { type: 'button', class: 'secondary small', onclick: () => this._useGeneratedPrompt() }, 'Use generated prompt');
    const prompt = el('div', { class: 'pd-prompt' },
      el('div', { class: 'pd-prompt-head' }, el('label', { for: promptId }, 'Prompt'), this.promptState),
      this.promptArea,
      el('div', { class: 'pd-prompt-foot' },
        el('span', { class: 'hint' }, 'Generated from the parameters. The path and its headings steer the motion; the prompt mostly sets the gait, so avoid words that contradict the path.'),
        this.promptReset));
    this.detailScroll = el('div', { class: 'pd-scroll' },
      el('div', { class: 'pd-detail-grid' },
        el('div', { class: 'pd-left' }, this.preview, legend, this.summary, this.pinsLine, this.warnings, this.experimental),
        el('div', { class: 'pd-right' }, this.desc, this.form, prompt)));
    this.detail = el('section', { class: 'pd-detail', hidden: true, 'aria-labelledby': titleId }, this.detailScroll);

    // ---- footer (detail view)
    this.resetBtn = el('button', { type: 'button', class: 'secondary', title: 'Every parameter back to its default', onclick: () => this._resetParams() }, 'Reset to defaults');
    this.footNote = el('span', { class: 'pd-foot-note', role: 'status' });
    this.applyBtn = el('button', { type: 'button', class: 'secondary', onclick: () => this._apply(false) }, 'Apply');
    this.generateBtn = el('button', { type: 'button', onclick: () => this._apply(true) }, 'Apply & Generate');
    this.foot = el('footer', { class: 'pd-foot', hidden: true }, this.resetBtn, this.footNote, this.applyBtn, this.generateBtn);

    // tabindex -1: a click on plain text keeps the focus (and the keys) in the dialog.
    this.dialog = el('dialog', { class: 'presets-dialog', 'aria-labelledby': titleId, tabindex: '-1' }, head, this.browse, this.detail, this.foot);
    this.dialog.addEventListener('close', () => this._onClose());
    // The modal owns the keyboard: the page's shortcuts (playback, editing-mode
    // W / E / Esc) listen on window and must not see keys typed here. Native
    // behaviour (Esc closes, Space presses a button) is not a listener and stays.
    this.dialog.addEventListener('keydown', (e) => e.stopPropagation());
    this.dialog.addEventListener('keyup', (e) => e.stopPropagation());
    // A click on the backdrop closes; both press and release must be on it (a
    // slider drag released outside the dialog also targets the dialog).
    let downOnBackdrop = false;
    this.dialog.addEventListener('pointerdown', (e) => { downOnBackdrop = e.target === this.dialog; });
    this.dialog.addEventListener('click', (e) => {
      if (e.target === this.dialog && downOnBackdrop) this.close();
      downOnBackdrop = false;
    });
    document.body.append(this.dialog);
    this._setView(this.view, { persist: false });
  }

  /** The 12 cards with their default-parameter thumbnails (once). */
  _ensureCards() {
    if (this.cards.size) return;
    for (const def of PRESETS) {
      const thumb = el('span', { class: 'preset-thumb' });
      try {
        thumb.innerHTML = presetSVG(simulatePreset(def.id, defaultParams(def.id)), { ...THUMB });
      } catch (err) {
        console.warn(err);
      }
      const card = el('button', { type: 'button', class: 'preset-card', 'data-id': def.id, title: def.description, onclick: () => this._showDetail(def.id) },
        thumb,
        el('span', { class: 'pc-text' },
          el('span', { class: 'pc-label' }, def.label, def.experimental ? el('span', { class: 'chip experimental' }, 'Experimental') : null),
          el('span', { class: 'pc-desc' }, def.description)));
      this.cards.set(def.id, card);
      this.items.append(card);
    }
  }

  /** Moves keepVisible (the toasts) into the dialog so they show above the backdrop. */
  _adoptOverlays() {
    this._homes = this.keepVisible.map((node) => [node, node.parentNode, node.nextSibling]);
    for (const node of this.keepVisible) this.dialog.append(node);
  }

  _releaseOverlays() {
    for (const [node, parent, next] of this._homes) {
      if (!parent) continue;
      if (next && next.parentNode === parent) parent.insertBefore(node, next);
      else parent.append(node);
    }
    this._homes = [];
  }

  _onClose() {
    this._releaseOverlays();
    const opener = this.opener;
    this.opener = null;
    if (opener && opener.isConnected && !opener.disabled) opener.focus();
  }

  _setView(view, { persist = true } = {}) {
    this.view = view === 'list' ? 'list' : 'grid';
    this.items.classList.toggle('grid', this.view === 'grid');
    this.items.classList.toggle('list', this.view === 'list');
    for (const [v, b] of Object.entries(this.viewButtons)) b.setAttribute('aria-pressed', String(v === this.view));
    if (persist) saveView(this.view);
  }

  _filter() {
    const raw = this.search.value.trim();
    const q = raw.toLowerCase();
    let shown = 0;
    for (const def of PRESETS) {
      const hit = !q || def.label.toLowerCase().includes(q) || def.description.toLowerCase().includes(q);
      const card = this.cards.get(def.id);
      if (card) card.hidden = !hit;
      if (hit) shown++;
    }
    this.empty.hidden = shown > 0;
    this.empty.textContent = shown ? '' : `No preset matches “${raw}”.`;
    this.count.textContent = q ? `${shown} of ${PRESETS.length} presets` : `${PRESETS.length} presets`;
  }

  // -------------------------------------------------------------------------
  // views
  // -------------------------------------------------------------------------

  _showBrowse({ focus = false } = {}) {
    const from = this.current?.def.id;
    this.current = null;
    this.browse.hidden = false;
    this.detail.hidden = true;
    this.foot.hidden = true;
    this.back.hidden = true;
    this.title.textContent = 'Parametric presets';
    this.count.hidden = false;
    this.badge.hidden = true;
    this._filter();
    if (focus) {
      const card = from ? this.cards.get(from) : null;
      (card && !card.hidden ? card : this.search).focus();
    }
  }

  /** Detail view of preset `id`; params / prompt restore a previous state. Throws for an unknown id. */
  _showDetail(id, { params = null, prompt = null } = {}) {
    const def = presetById(id);
    this.current = { def, params: normalizeParams(id, params ?? defaultParams(id)), built: null, promptDirty: false, controls: new Map(), groups: [] };
    this.browse.hidden = true;
    this.detail.hidden = false;
    this.foot.hidden = false;
    this.back.hidden = false;
    this.title.textContent = def.label;
    this.count.hidden = true;
    this.badge.hidden = !def.experimental;
    this.desc.textContent = def.description;
    this.experimental.hidden = !def.experimental;
    this.form.replaceChildren(...this._formRows(def));
    this.detailScroll.scrollTop = 0;
    this._update();
    const text = typeof prompt === 'string' ? prompt.trim() : '';
    if (text && this.current.built && text !== this.current.built.prompt) {
      this.current.promptDirty = true;
      this.promptArea.value = text;
      this._syncPromptState();
    }
    this.back.focus();
  }

  // -------------------------------------------------------------------------
  // form
  // -------------------------------------------------------------------------

  _formRows(def) {
    const c = this.current;
    const rows = def.params.filter((pd) => !pd.group || pd.group === 'main').map((pd) => this._control(pd));
    for (const [group, title] of GROUPS) {
      const defs = def.params.filter((pd) => pd.group === group);
      if (!defs.length) continue;
      const extra = el('span', { class: 'summary-extra' });
      c.groups.push({ group, defs, extra });
      const note = group === 'dynamics'
        ? el('div', { class: 'hint' }, 'Spring half-lives (seconds) that shape the path: smaller values give sharper changes.')
        : null;
      rows.push(el('details', { class: 'folder pd-group' },
        el('summary', {}, title, extra),
        el('div', { class: 'folder-body' }, ...defs.map((pd) => this._control(pd)), note)));
    }
    return rows;
  }

  _control(pd) {
    const ctl = pd.type === 'number' ? this._numberControl(pd)
      : pd.type === 'side' ? this._toggleControl(pd)
        : pd.type === 'select' ? this._selectControl(pd)
          : this._boolControl(pd);
    this.current.controls.set(pd.key, { pd, set: ctl.set });
    return ctl.row;
  }

  _numberControl(pd) {
    const label = pd.unit ? `${pd.label} (${pd.unit})` : pd.label;
    const hint = pd.hint ? el('span', { class: 'sr-only', id: nextId('pd-hint') }, ` ${pd.hint}`) : null;
    const described = hint ? hint.id : null;
    const range = el('input', { type: 'range', min: pd.min, max: pd.max, step: pd.step, 'aria-label': label, 'aria-describedby': described });
    const box = el('input', { type: 'number', min: pd.min, max: pd.max, step: pd.step, 'aria-label': `${label} value`, 'aria-describedby': described });
    range.addEventListener('input', () => this._change(pd.key, Number(range.value)));
    // Live while typing (the box itself is not rewritten), normalized on commit.
    box.addEventListener('input', () => {
      const v = Number(box.value);
      if (box.value.trim() !== '' && Number.isFinite(v)) this._change(pd.key, v, { typing: box });
    });
    box.addEventListener('change', () => {
      const v = Number(box.value);
      this._change(pd.key, box.value.trim() !== '' && Number.isFinite(v) ? v : this.current.params[pd.key]);
    });
    const row = el('div', { class: 'slider-row', title: pd.hint }, el('span', {}, label, hint), range, box);
    const set = (v, { typing = null, min = pd.min, max = pd.max } = {}) => {
      range.min = min;
      box.min = min;
      range.max = max;
      box.max = max;
      range.value = v;
      if (typing !== box) box.value = fixed(v, pd.step);
    };
    return { row, set };
  }

  _toggleControl(pd) {
    const options = pd.options ?? [];
    const buttons = options.map((o) => el('button', { type: 'button', 'aria-pressed': 'false', onclick: () => this._change(pd.key, o.value) }, o.label));
    const row = el('div', { class: 'pd-field', title: pd.hint },
      el('span', {}, pd.label),
      el('div', { class: 'seg-toggle', role: 'group', 'aria-label': pd.label }, ...buttons));
    const set = (v) => options.forEach((o, i) => buttons[i].setAttribute('aria-pressed', String(o.value === v)));
    return { row, set };
  }

  _selectControl(pd) {
    const select = el('select', { 'aria-label': pd.label }, ...(pd.options ?? []).map((o) => el('option', { value: o.value }, o.label)));
    select.addEventListener('change', () => this._change(pd.key, select.value));
    const row = el('div', { class: 'pd-field', title: pd.hint }, el('span', {}, pd.label), select);
    return { row, set: (v) => { select.value = v; } };
  }

  _boolControl(pd) {
    const input = el('input', { type: 'checkbox' });
    input.addEventListener('change', () => this._change(pd.key, input.checked));
    const row = el('div', { class: 'pd-bool' },
      el('label', { class: 'check' }, input, pd.label),
      pd.hint ? el('div', { class: 'hint' }, pd.hint) : null);
    return { row, set: (v) => { input.checked = !!v; } };
  }

  _change(key, value, opts = {}) {
    const c = this.current;
    if (!c) return;
    c.params = { ...c.params, [key]: value };
    this._update(opts);
  }

  _resetParams() {
    const c = this.current;
    if (!c) return;
    c.params = defaultParams(c.def.id);
    this._update();
  }

  /** Rebuilds the preset from the form and refreshes every part of the detail view. */
  _update({ typing = null } = {}) {
    const c = this.current;
    if (!c) return;
    const id = c.def.id;
    let built;
    try {
      const pose = this.pose() || null;
      built = buildPosedPreset(id, c.params, pose?.id ?? null, pose?.params, { skeletonKey: this.skeletonKey() || 'soma30' });
    } catch (err) {
      console.error(err);
      c.built = null;
      this.warnings.replaceChildren(el('li', {}, `Could not build this preset: ${err.message || err}`));
      this.warnings.hidden = false;
      this.syncGenerate();
      return;
    }
    c.built = built;
    // While a number box is being typed in, its raw value stays in params (an
    // intermediate keystroke must not refit the other durations).
    if (!typing) c.params = { ...built.params };
    const p = built.params;
    for (const [key, ctl] of c.controls) {
      const pd = ctl.pd;
      // (only Run in a circle raises a minimum: laps for 60 frames)
      const min = pd.type === 'number' ? normalizeParams(id, { ...p, [key]: pd.min })[key] : undefined;
      const max = pd.type === 'number' ? normalizeParams(id, { ...p, [key]: pd.max })[key] : undefined;
      ctl.set(p[key], { typing, min, max });
    }
    for (const g of c.groups) {
      g.extra.textContent = g.group === 'dynamics'
        ? `${g.defs.map((pd) => fixed(p[pd.key], pd.step)).join(' / ')} s`
        : g.defs.map((pd) => (p[pd.key] ? 'on' : 'off')).join(' / ');
    }
    this.preview.innerHTML = presetSVG(built.sim, { ...PREVIEW, grid: true, title: `${c.def.label}: top view of the path` });
    this.summary.replaceChildren(...summaryParts(built));
    this.pinsLine.textContent = pinsText(built);
    this.warnings.replaceChildren(...built.warnings.map((w) => el('li', {}, w)));
    this.warnings.hidden = !built.warnings.length;
    if (!c.promptDirty) this.promptArea.value = built.prompt;
    this._syncPromptState();
    this.syncGenerate();
  }

  // -------------------------------------------------------------------------
  // prompt
  // -------------------------------------------------------------------------

  _promptEdited() {
    const c = this.current;
    if (!c?.built) return;
    c.promptDirty = this.promptArea.value.trim() !== c.built.prompt;
    this._syncPromptState();
  }

  _syncPromptState() {
    const dirty = !!this.current?.promptDirty;
    const empty = !this.promptArea.value.trim();
    this.promptState.textContent = !dirty ? 'generated' : empty ? 'empty: the generated prompt is used' : 'edited';
    this.promptState.classList.toggle('edited', dirty);
    this.promptReset.disabled = !dirty;
  }

  _useGeneratedPrompt() {
    const c = this.current;
    if (!c?.built) return;
    c.promptDirty = false;
    this.promptArea.value = c.built.prompt;
    this._syncPromptState();
    this.promptArea.focus();
  }

  // -------------------------------------------------------------------------
  // apply
  // -------------------------------------------------------------------------

  _apply(generate) {
    const c = this.current;
    if (!c?.built) return;
    this.syncGenerate();
    if (this.applyBtn.disabled || (generate && this.generateBtn.disabled)) return;
    const text = this.promptArea.value.trim();
    // basePrompt: the prompt without pose words (an edited prompt counts as the user's own).
    const edited = c.promptDirty && !!text;
    const built = { ...c.built, prompt: text || c.built.prompt, basePrompt: edited ? stripPosePhrases(text).trim() : c.built.basePrompt };
    let ok = false;
    try {
      ok = this.onApply ? this.onApply(built, { generate }) : false;
    } catch (err) {
      console.error(err);
      ok = false;
    }
    if (ok !== false) this.close();
    else this.syncGenerate();
  }
}
