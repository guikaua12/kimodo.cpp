// Pose presets dialog for the Kimodo demo page: a native modal <dialog> with
// the poses of poses.js as cards ("No pose" first) and a detail view (front
// and side stick figures of the pose, what it adds to a generation, its
// parameters) that hands the choice to the app (app.js applyPose). It shares
// the look of the path presets dialog (presets_ui.js: the .presets-dialog and
// .pd-* styles). Browser only; the pose logic is the DOM-free poses.js.
//
// API
//   new PosesDialog({onApply, canApply, canGenerate, context, skeleton, keepVisible})
//     onApply(choice, {generate})  choice = {id, params} or null (no pose); return
//                                  false to keep the dialog open (refused/failed).
//     canApply(), canGenerate()    as in PresetsDialog.
//     context()                    {preset, editedPreset, skeletonKey, somaSwitch,
//                                  somaModel, densePath}: what the pose will apply
//                                  to (shown as notes; app.js poseContext).
//     skeleton()                   a SOMA-30 skeleton.js Skeleton for the drawings,
//                                  or null (cards then show no figure).
//     keepVisible                  elements moved into the dialog while open (toasts).
//   #open(current?)                the list, or the detail view of current = {id, params}.
//   #close(), #isOpen, #element, #syncGenerate()
//   poseReadout(choice)            sidebar text: 'Rifle: collapsed low ready · hold 100% · sway 100%'.

import { POSES, poseById, defaultPoseParams, normalizePoseParams, poseAvailable, poseSVG } from './poses.js';

const THUMB = Object.freeze({ width: 176, height: 110 });
const PREVIEW = Object.freeze({ width: 520, height: 330 });
const GROUP_TITLES = Object.freeze({ pins: 'Guide pins (generation)', layer: 'Pose layer (after generation)' });

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

/** Sidebar readout of a pose choice {id, params}. */
export function poseReadout(choice) {
  if (!choice) return '';
  let def;
  try { def = poseById(choice.id); } catch { return ''; }
  const p = normalizePoseParams(choice.id, choice.params);
  const parts = [def.label];
  if (def.kind === 'arms') {
    if (!p.guidePins) parts.push('no hand pins');
    parts.push(p.hold > 0 ? `hold ${p.hold}%` : 'no layer');
    if (p.hold > 0) parts.push(`sway ${p.sway}%`);
  }
  return parts.join(' · ');
}

/** What a pose adds to a generation, as short lines for the detail view. */
function effects(def, p) {
  const out = [def.prompt.phrase ? `Prompt: “… ${def.prompt.phrase} …”`
    : 'Prompt: no crouch words (they made Kimodo hunch 60-80 degrees); with a path preset, “stands still” becomes “stays still”.'];
  if (def.kind === 'arms') {
    out.push(p.guidePins
      ? "Guide pins: both hands pinned to Kimodo's own rifle carry (the collapsed carry, which keeps the body upright) every ~8 frames along the path (sided Left/Right Hand constraints), lifted to standing height where the path stands still."
      : 'Guide pins: off. The body is shaped by the prompt only, and the layer places the arms.');
    out.push('Path presets: their idle pins hold the idle pose with that carry; the frame-0 Hips pin uses its walk pose.');
    out.push(p.hold > 0
      ? `Pose layer: after generation the server holds the arms on the ${def.id === 'rifle_collapsed_low_ready' ? 'collapsed (cross-body)' : 'low-ready (muzzle forward)'} carry on every frame (${p.hold}%), keeping ${p.sway}% of the model's own arm sway. The GLB download includes it.`
      : 'Pose layer: off (the arms are exactly what the model generated).');
  } else {
    out.push('Path presets: their idle pins hold a crouch idle pose (standing starts and ends, turns in place). Presets that start moving pin a crouch walk pose on the first two frames' + (p.endAnchor ? ', and on the last two when still moving there.' : '.'));
    out.push('Pelvis pins every ~10 frames at 0.66 m (standing) to 0.70 m (walking), tilted 15 degrees forward, instead of the frame-0 Hips pin.');
    out.push('No pose layer: the legs and torso come from the model. Without a path preset the crouch adds no pins and no words.');
  }
  return out;
}

export class PosesDialog {
  constructor({ onApply, canApply = () => true, canGenerate = () => true, context = () => ({}), skeleton = () => null, keepVisible = [] } = {}) {
    this.onApply = onApply;
    this.canApply = canApply;
    this.canGenerate = canGenerate;
    this.context = context;
    this.skeleton = skeleton;
    this.keepVisible = keepVisible.filter(Boolean);
    this.current = null; // detail: {def: PoseDef | null, params, controls: Map}
    this.cards = new Map();
    this.opener = null;
    this._homes = [];
    this._build();
  }

  get isOpen() { return this.dialog.open; }
  get element() { return this.dialog; }

  /** Opens the list, or the detail view of current = {id, params}. */
  open(current = null) {
    if (!this.dialog.open) {
      this.opener = document.activeElement instanceof HTMLElement && document.activeElement !== document.body ? document.activeElement : null;
      this._ensureCards();
      this._homes = this.keepVisible.map((node) => [node, node.parentNode, node.nextSibling]);
      for (const node of this.keepVisible) this.dialog.append(node);
      this.dialog.showModal();
    }
    this._markCurrent(current);
    if (current && current.id) {
      try {
        this._showDetail(current.id, current.params);
      } catch (err) {
        console.warn(err);
        this._showBrowse();
      }
    } else {
      this._showBrowse();
      this.items.querySelector('.preset-card')?.focus();
    }
    this.syncGenerate();
  }

  close() {
    if (this.dialog.open) this.dialog.close();
  }

  syncGenerate() {
    const ready = !!this.current && (!this.current.def || poseAvailable(this.current.def.id));
    const applyOk = !!this.canApply();
    const genOk = applyOk && !!this.canGenerate();
    this.applyBtn.disabled = !ready || !applyOk;
    this.generateBtn.disabled = !ready || !genOk;
    this.applyBtn.title = applyOk ? 'Use this pose for the next generations' : 'Wait for the current generation to finish';
    this.generateBtn.title = genOk ? 'Use this pose and start generating'
      : applyOk ? 'Generating is disabled in editing mode: use Apply, then Generate' : 'Wait for the current generation to finish';
    this.footNote.textContent = applyOk ? '' : 'A generation is running; poses can be changed when it finishes.';
  }

  // -------------------------------------------------------------------------
  // DOM
  // -------------------------------------------------------------------------

  _build() {
    const titleId = nextId('pod-title');
    this.title = el('h2', { id: titleId }, 'Poses');
    this.count = el('span', { class: 'pd-count' }, `${POSES.length} poses`);
    this.badge = el('span', { class: 'chip experimental', hidden: true }, 'Experimental');
    this.back = el('button', { type: 'button', class: 'secondary small pd-back', hidden: true, 'aria-label': 'Back to all poses', onclick: () => this._showBrowse({ focus: true }) }, '← All poses');
    const close = el('button', { type: 'button', class: 'secondary icon pd-close', title: 'Close (Esc)', 'aria-label': 'Close', onclick: () => this.close() }, '×');
    const head = el('header', { class: 'pd-head' }, this.back, el('div', { class: 'pd-titles' }, this.title, this.count, this.badge), close);

    this.items = el('div', { class: 'pd-items grid pose-items' });
    this.browse = el('section', { class: 'pd-browse', 'aria-label': 'All poses' },
      el('p', { class: 'pd-desc pose-intro' }, 'A pose is held for the whole generated clip and works with the path presets. The prompt alone does not pin a pose down: Kimodo reads “low ready” differently from seed to seed. So each pose also adds constraints to the request, and the rifle poses add a layer the server applies after generation.'),
      el('div', { class: 'pd-scroll' }, el('div', { class: 'pd-pad' }, this.items)));

    this.preview = el('div', { class: 'pd-preview pose-preview' });
    const legend = el('div', { class: 'pd-legend' },
      el('span', {}, 'front (its left on the right) · side (facing right)'),
      el('span', {}, el('i', { class: 'lg-prop' }), 'rifle line, grip → support hand'));
    this.effects = el('ul', { class: 'pose-effects' });
    this.notes = el('ul', { class: 'pd-warnings', hidden: true });
    this.experimental = el('p', { class: 'pd-experimental', hidden: true },
      el('b', {}, 'Experimental. '), 'Checked on the GPU on Constant Speed, Strafe, Forward turn, Start, Stop and Turn in place at 1.0 m/s (3 seeds each with the start walk pair; the end pair, on by default, on 2 seeds). Idle, stop and turn clips lean about 38-41 degrees; moving clips 40-49 degrees and read as a creeping crouch. Crouch starts sometimes sink a few centimetres into the floor.');
    this.desc = el('p', { class: 'pd-desc' });
    this.form = el('div', { class: 'pd-form' });
    this.detailScroll = el('div', { class: 'pd-scroll' },
      el('div', { class: 'pd-detail-grid' },
        el('div', { class: 'pd-left' }, this.preview, legend, this.notes, this.experimental),
        el('div', { class: 'pd-right' }, this.desc, this.form, el('div', { class: 'pose-effects-box' }, el('b', {}, 'Adds to a generation'), this.effects))));
    this.detail = el('section', { class: 'pd-detail', hidden: true, 'aria-labelledby': titleId }, this.detailScroll);

    this.resetBtn = el('button', { type: 'button', class: 'secondary', onclick: () => this._resetParams() }, 'Reset to defaults');
    this.footNote = el('span', { class: 'pd-foot-note', role: 'status' });
    this.applyBtn = el('button', { type: 'button', class: 'secondary', onclick: () => this._apply(false) }, 'Apply');
    this.generateBtn = el('button', { type: 'button', onclick: () => this._apply(true) }, 'Apply & Generate');
    this.foot = el('footer', { class: 'pd-foot', hidden: true }, this.resetBtn, this.footNote, this.applyBtn, this.generateBtn);

    this.dialog = el('dialog', { class: 'presets-dialog poses-dialog', 'aria-labelledby': titleId, tabindex: '-1' }, head, this.browse, this.detail, this.foot);
    this.dialog.addEventListener('close', () => this._onClose());
    this.dialog.addEventListener('keydown', (e) => e.stopPropagation());
    this.dialog.addEventListener('keyup', (e) => e.stopPropagation());
    let downOnBackdrop = false;
    this.dialog.addEventListener('pointerdown', (e) => { downOnBackdrop = e.target === this.dialog; });
    this.dialog.addEventListener('click', (e) => {
      if (e.target === this.dialog && downOnBackdrop) this.close();
      downOnBackdrop = false;
    });
    document.body.append(this.dialog);
  }

  _svg(id, size, title) {
    const sk = this.skeleton();
    if (!sk) return '';
    try {
      return poseSVG(sk, id, { ...size, title });
    } catch (err) {
      console.warn(err);
      return '';
    }
  }

  _ensureCards() {
    if (this.cards.size) return;
    const card = (id, label, desc, extra = null) => {
      const thumb = el('span', { class: 'preset-thumb pose-thumb' });
      thumb.innerHTML = this._svg(id, THUMB);
      const node = el('button', { type: 'button', class: 'preset-card', 'data-id': id ?? 'none', title: desc, onclick: () => this._showDetail(id) },
        thumb,
        el('span', { class: 'pc-text' },
          el('span', { class: 'pc-label' }, label, extra, el('span', { class: 'chip current', hidden: true }, 'In use')),
          el('span', { class: 'pc-desc' }, desc)));
      this.cards.set(id ?? 'none', node);
      this.items.append(node);
    };
    card(null, 'No pose', 'Generate from the prompt and constraints only.');
    for (const def of POSES) {
      const missing = !poseAvailable(def.id);
      card(def.id, def.label, def.description,
        missing ? el('span', { class: 'chip failed' }, 'Data missing') : def.experimental ? el('span', { class: 'chip experimental' }, 'Experimental') : null);
    }
  }

  _markCurrent(current) {
    const key = current?.id ?? 'none';
    for (const [id, node] of this.cards) node.querySelector('.chip.current').hidden = id !== key;
  }

  _onClose() {
    for (const [node, parent, next] of this._homes) {
      if (!parent) continue;
      if (next && next.parentNode === parent) parent.insertBefore(node, next);
      else parent.append(node);
    }
    this._homes = [];
    const opener = this.opener;
    this.opener = null;
    if (opener && opener.isConnected && !opener.disabled) opener.focus();
  }

  _showBrowse({ focus = false } = {}) {
    const from = this.current ? (this.current.def?.id ?? 'none') : null;
    this.current = null;
    this.browse.hidden = false;
    this.detail.hidden = true;
    this.foot.hidden = true;
    this.back.hidden = true;
    this.title.textContent = 'Poses';
    this.count.hidden = false;
    this.badge.hidden = true;
    if (focus) (this.cards.get(from) ?? this.items.querySelector('.preset-card'))?.focus();
  }

  _showDetail(id, params = null) {
    const def = id == null ? null : poseById(id);
    this.current = { def, params: def ? normalizePoseParams(def.id, params ?? defaultPoseParams(def.id)) : {}, controls: new Map() };
    this.browse.hidden = true;
    this.detail.hidden = false;
    this.foot.hidden = false;
    this.back.hidden = false;
    this.title.textContent = def ? def.label : 'No pose';
    this.count.hidden = true;
    this.badge.hidden = !def?.experimental;
    this.experimental.hidden = !def?.experimental;
    this.desc.textContent = def ? def.description : 'Generations use the prompt and the constraints as they are. Any pose phrase is removed from the prompts, and an applied path preset is rebuilt without pose pins.';
    this.resetBtn.hidden = !def || !def.params.length;
    this.form.replaceChildren(...(def ? this._formRows(def) : []));
    this.preview.innerHTML = this._svg(id, PREVIEW, def ? `${def.label}: front and side view` : 'No pose: the idle pose');
    this.detailScroll.scrollTop = 0;
    this._update();
    this.back.focus();
  }

  _formRows(def) {
    const rows = [];
    for (const group of ['main', 'pins', 'layer']) {
      const defs = def.params.filter((pd) => (pd.group || 'main') === group);
      if (!defs.length) continue;
      if (group !== 'main') rows.push(el('div', { class: 'subhead' }, GROUP_TITLES[group]));
      for (const pd of defs) rows.push(this._control(pd));
    }
    return rows;
  }

  _control(pd) {
    if (pd.type === 'bool') {
      const input = el('input', { type: 'checkbox' });
      input.addEventListener('change', () => this._change(pd.key, input.checked));
      this.current.controls.set(pd.key, (v) => { input.checked = !!v; });
      return el('div', { class: 'pd-bool' }, el('label', { class: 'check' }, input, pd.label), pd.hint ? el('div', { class: 'hint' }, pd.hint) : null);
    }
    const label = pd.unit ? `${pd.label} (${pd.unit})` : pd.label;
    const range = el('input', { type: 'range', min: pd.min, max: pd.max, step: pd.step, 'aria-label': label });
    const box = el('input', { type: 'number', min: pd.min, max: pd.max, step: pd.step, 'aria-label': `${label} value` });
    range.addEventListener('input', () => this._change(pd.key, Number(range.value)));
    box.addEventListener('change', () => this._change(pd.key, Number(box.value)));
    this.current.controls.set(pd.key, (v) => { range.value = v; box.value = v; });
    return el('div', { class: 'pd-bool' },
      el('div', { class: 'slider-row', title: pd.hint }, el('span', {}, label), range, box),
      pd.hint ? el('div', { class: 'hint pose-hint' }, pd.hint) : null);
  }

  _change(key, value) {
    const c = this.current;
    if (!c?.def) return;
    c.params = normalizePoseParams(c.def.id, { ...c.params, [key]: value });
    this._update();
  }

  _resetParams() {
    const c = this.current;
    if (!c?.def) return;
    c.params = defaultPoseParams(c.def.id);
    this._update();
  }

  _update() {
    const c = this.current;
    if (!c) return;
    for (const [key, set] of c.controls) set(c.params[key]);
    this.effects.replaceChildren(...(c.def ? effects(c.def, c.params) : ['Nothing: no phrase, no pins, no layer.']).map((t) => el('li', {}, t)));
    const ctx = this.context() || {};
    const notes = [];
    if (c.def) {
      if (!poseAvailable(c.def.id)) notes.push('This pose’s data is missing from pose_library.js (scripts/extract_pose_library.py); it cannot be used yet.');
      if (ctx.skeletonKey && ctx.skeletonKey !== 'soma30') {
        notes.push(ctx.somaSwitch ? `Poses need the SOMA skeleton: Apply switches to ${ctx.somaSwitch}.`
          : ctx.somaModel ? `Poses need the SOMA skeleton, and your constraints belong to the current one: switch the motion model to ${ctx.somaModel} first. Until then only the prompt phrase is used.`
            : 'Poses need the SOMA skeleton and no SOMA model is installed: only the prompt phrase is used.');
      }
      if (ctx.preset) notes.push(`The applied path preset “${ctx.preset}” is rebuilt with this pose; its preview replaces the viewport until you Generate.`);
      else if (ctx.editedPreset) notes.push(`The path preset “${ctx.editedPreset}” was edited after it was applied, so it is not rebuilt: the pose is added to your prompts and constraints. Re-apply the preset to rebuild it with the pose.`);
      else if (c.def.kind === 'arms' && c.params.guidePins && ctx.denseNoHeadings) notes.push('No path preset is applied, and your 2D root has no headings: hand guide pins go only where the facing is known, so set headings on its waypoints. Until then this pose adds the prompt phrase and the pose layer.');
      else if (c.def.kind === 'arms' && c.params.guidePins && !ctx.densePath) notes.push('No path preset is applied. Hand guide pins need a 2D root on every frame with headings (a path preset, or Make Smooth Path with waypoint headings), so this pose adds only the prompt phrase and the pose layer.');
      else if (c.def.kind === 'stance' && !ctx.preset) notes.push('No path preset is applied: the crouch adds no pins and no words. Its idle, walk and pelvis pins come with the path presets.');
    } else if (ctx.preset) {
      notes.push(`The applied path preset “${ctx.preset}” is rebuilt without a pose.`);
    } else {
      notes.push('Hand constraints already on the timeline stay and are still sent.');
    }
    this.notes.replaceChildren(...notes.map((t) => el('li', {}, t)));
    this.notes.hidden = !notes.length;
    this.syncGenerate();
  }

  _apply(generate) {
    const c = this.current;
    if (!c) return;
    this.syncGenerate();
    if (this.applyBtn.disabled || (generate && this.generateBtn.disabled)) return;
    const choice = c.def ? { id: c.def.id, params: { ...c.params } } : null;
    let ok = false;
    try {
      ok = this.onApply ? this.onApply(choice, { generate }) : false;
    } catch (err) {
      console.error(err);
      ok = false;
    }
    if (ok !== false) this.close();
    else this.syncGenerate();
  }
}
