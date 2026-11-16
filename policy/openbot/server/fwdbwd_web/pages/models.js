// Models: every model folder with its best-val numbers, and what is published to the phone.
import { h, clear, api, attempt, fmt, pct, fmtDate, fmtAgo } from '../lib.js';

const DIRS = ['all', 'fwd', 'bwd', 'both', 'other'];

export async function render(view) {
  const filt = { dir: localStorage.getItem('fb.mdir') || 'all', model: '', q: '' };
  const pubBox = h('div');
  const tableBox = h('div', {}, h('div', { class: 'empty' }, 'Loading…'));
  const modelSel = h('select', { onchange: () => { filt.model = modelSel.value; draw(); } });
  const search = h('input', { type: 'text', placeholder: 'search names, e.g. s4-5 or flip', oninput: () => { filt.q = search.value.toLowerCase(); draw(); } });
  const seg = h('div', { class: 'seg' });
  view.append(
    h('div', { class: 'page-head' }, h('h1', {}, 'Models'), h('span', { class: 'spacer' }), seg, modelSel, search),
    pubBox, tableBox);
  let data = { models: [], published: [] };

  function drawSeg() {
    clear(seg, DIRS.map((d) => h('button', { class: filt.dir === d ? 'on' : '', onclick: () => { filt.dir = d; localStorage.setItem('fb.mdir', d); drawSeg(); draw(); } }, d)));
  }

  function draw() {
    const types = [...new Set(data.models.map((m) => m.model).filter(Boolean))].sort();
    clear(modelSel, h('option', { value: '' }, 'all models'), types.map((t) => h('option', { value: t, selected: t === filt.model }, t)));
    const rows = data.models.filter((m) =>
      (filt.dir === 'all' || (filt.dir === 'other' ? !m.direction : m.direction === filt.dir))
      && (!filt.model || m.model === filt.model)
      && (!filt.q || m.name.toLowerCase().includes(filt.q)));
    clear(tableBox, h('div', { class: 'card flush' }, h('table', {},
      h('thead', {}, h('tr', {},
        h('th', {}, 'Model folder'), h('th', {}, 'Type'), h('th', {}, 'Sessions'), h('th', { class: 'num' }, 'Epochs'),
        h('th', { class: 'num' }, 'Best'), h('th', { class: 'num' }, 'val loss'), h('th', { class: 'num' }, 'val angle'),
        h('th', { class: 'num' }, 'val dir'), h('th', { class: 'num' }, 'val MAE'), h('th', {}, 'Trained'))),
      h('tbody', {}, rows.map((m) => {
        const b = m.best || {};
        return h('tr', { class: 'click', onclick: () => { location.hash = `#/model?name=${encodeURIComponent(m.name)}`; } },
          h('td', { class: 'name' }, m.direction ? h('span', { class: `tag ${m.direction}`, style: { marginRight: '6px' } }, m.direction) : null, m.name,
            m.published.length ? h('div', {}, m.published.map((p) => h('span', { class: 'tag good', style: { marginRight: '4px' } }, '📱 ' + p))) : null),
          h('td', { class: 'nowrap' }, m.model || h('span', { class: 'muted' }, '?')),
          h('td', { class: 'nowrap' }, m.sessions.length ? m.sessions.join(', ') : h('span', { class: 'muted' }, '–')),
          h('td', { class: 'num' }, m.epochs),
          h('td', { class: 'num' }, m.best ? b.epoch + 1 : '–'),
          h('td', { class: 'num' }, fmt(b.val_loss)),
          h('td', { class: 'num' }, pct(b.val_angle_metric)),
          h('td', { class: 'num' }, pct(b.val_direction_metric)),
          h('td', { class: 'num' }, fmt(b.val_mean_absolute_error)),
          h('td', { class: 'nowrap muted' }, fmtDate(m.mtime)));
      }))),
      h('div', { class: 'muted', style: { padding: '8px 12px', fontSize: '12px' } },
        `${rows.length} of ${data.models.length}. Best = the epoch with the lowest val loss. angle = share of frames with steering within 0.1, dir = steering on the right side. Compare loss only within one model type; memory models' val numbers are teacher-forced.`)));
  }

  function drawPublished() {
    const p = data.published;
    clear(pubBox, h('div', { class: 'card' },
      h('h2', {}, `On the phone (${p.length})`),
      h('div', { class: 'muted', style: { fontSize: '12px', marginBottom: '6px' } },
        'models/*.tflite: the phone downloads these within ~10 s while connected to this server, as CMDNAV; set memory/seq models to CMDNAV_MEM / CMDNAV_SEQ in its Model Management. Removing one here doesn\'t delete the phone\'s copy.'),
      p.length ? h('table', {}, h('tbody', {}, p.map((x) => h('tr', {},
        h('td', { class: 'mono' }, x.file),
        h('td', {}, x.model ? h('a', { href: `#/model?name=${encodeURIComponent(x.model)}`, class: 'mono', style: { fontSize: '12px' } }, `${x.model} / ${x.checkpoint}`) : h('span', { class: 'muted' }, 'origin unknown')),
        h('td', { class: 'num muted' }, fmtAgo(x.time || x.mtime)),
        h('td', { class: 'right' }, h('button', { class: 'small danger', onclick: async () => {
          if (!confirm(`Stop serving ${x.file}?`)) return;
          await attempt(() => api(`published/${encodeURIComponent(x.name)}/delete`, {}), 'Unpublished');
          load();
        } }, 'Unpublish'))))))
        : h('div', { class: 'muted' }, 'Nothing published. Open a model and press Publish.')));
  }

  async function load() {
    data = await api('models');
    drawPublished();
    draw();
  }
  drawSeg();
  await load();
}
