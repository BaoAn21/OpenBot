// One model: metrics, curves, the training plots, publishing to the phone, check_tflite,
// notes and the data it was trained on.
import { h, clear, api, attempt, fmt, pct, fmtDate, modelFileUrl, jobLog, toast } from '../lib.js';
import { lineChart, legend } from '../charts.js';

const METRICS = [
  ['loss', 'loss', (v) => fmt(v)],
  ['angle_metric', 'steering within 0.1', pct],
  ['direction_metric', 'steering direction', pct],
  ['mean_absolute_error', 'mean abs error', (v) => fmt(v)],
];

function defaultTarget(m) {
  const s = m.sessions.length ? '_s' + (m.sessions.length > 3 ? `${m.sessions[0]}-${m.sessions[m.sessions.length - 1]}` : m.sessions.join('-')) : '';
  const short = (m.model || 'model').replace('cil_mobile_fast', 'cil').replace('donkey_', 'dk_');
  return `${m.direction || 'model'}${s}_${short}`;
}

function manifestSummary(text) {
  const counts = {};
  for (const line of text.split('\n')) {
    const m = /^(train|test)\/(fwd|bwd)\/session_(\d+)\//.exec(line);
    if (!m) continue;
    const k = `${m[1]} ${m[2]}`;
    counts[k] = counts[k] || {};
    counts[k][m[3]] = (counts[k][m[3]] || 0) + 1;
  }
  return Object.entries(counts).map(([k, v]) =>
    h('div', {}, h('b', {}, k), ': ', Object.entries(v).map(([s, n]) => `s${s} (${n} folders)`).join(', ')));
}

export async function render(view, params) {
  const name = params.name;
  let m;
  try { m = await api('models/' + encodeURIComponent(name)); } catch (e) {
    view.append(h('div', { class: 'card note bad' }, e.message));
    return;
  }
  const charts = [];
  const col = (k) => (m.header.includes(k) ? m.rows.map((r) => r[m.header.indexOf(k)]) : null);
  const x = m.rows.map((r) => r[0] + 1);
  const bestEpoch = m.best ? m.best.epoch + 1 : null;

  // metrics table
  const metricRows = METRICS.filter(([k]) => m.header.includes('val_' + k)).map(([k, label, f]) => h('tr', {},
    h('td', {}, label),
    h('td', { class: 'num' }, m.best ? f(m.best['val_' + k]) : '–'),
    h('td', { class: 'num' }, m.last ? f(m.last['val_' + k]) : '–'),
    h('td', { class: 'num muted' }, m.best ? f(m.best[k]) : '–'),
    h('td', { class: 'num muted' }, m.last ? f(m.last[k]) : '–')));

  // curve with a metric picker
  const metricSel = h('select', {}, METRICS.filter(([k]) => m.header.includes(k)).map(([k, label]) => h('option', { value: k }, label)));
  const chartBox = h('div');
  function drawChart() {
    const k = metricSel.value;
    const opts = {
      x, xMin: 1, xMax: Math.max(2, x[x.length - 1] || 2), height: 240,
      series: [
        { label: 'train', color: '--series-1', values: col(k) || [], dots: x.length < 40 },
        { label: 'val', color: '--series-2', values: col('val_' + k) || [], dots: x.length < 40 },
      ],
      xFormat: (v) => 'epoch ' + Math.round(v), xTick: (v) => String(Math.round(v)),
      yFormat: k === 'loss' || k === 'mean_absolute_error' ? (v) => String(+v.toPrecision(3)) : (v) => (100 * v).toFixed(0) + '%',
      markers: bestEpoch ? [{ x: bestEpoch, label: 'best val loss' }] : [],
    };
    if (!charts[0]) { charts[0] = lineChart(opts); clear(chartBox, legend([{ label: 'train', color: '--series-1' }, { label: 'val', color: '--series-2' }]), charts[0]); }
    else charts[0].update(opts);
  }
  metricSel.onchange = drawChart;

  // publish
  const ckpt = h('select', {}, m.tflite.map((t) => h('option', { value: t, selected: t === 'best.tflite' }, t)));
  const target = h('input', { type: 'text', value: defaultTarget(m), style: { width: '260px' } });
  const pubList = h('div');
  const drawPub = () => clear(pubList, m.published.length
    ? m.published.map((p) => h('div', {}, h('span', { class: 'tag good' }, `📱 ${p.file}`), h('span', { class: 'muted' }, ` from ${p.checkpoint}, ${fmtDate(p.time || p.mtime)}`)))
    : h('div', { class: 'muted' }, 'Not published.'));
  drawPub();
  async function publish(overwrite) {
    try {
      const r = await api(`models/${encodeURIComponent(name)}/publish`, { checkpoint: ckpt.value, target: target.value, overwrite });
      toast(`Published ${r.name}.tflite; the phone picks it up within ~10 s`, 'good');
      m = { ...m, published: (await api('models/' + encodeURIComponent(name))).published };
      drawPub();
    } catch (e) {
      if (/already published/.test(e.message) && confirm(e.message + '. Replace it?')) return publish(true);
      toast(e.message, 'bad');
    }
  }

  // check_tflite
  const checkBox = h('div');
  const notes = h('textarea', { rows: 4, placeholder: 'how it drove, what to try next…' }, m.notes);

  view.append(
    h('div', { class: 'page-head' }, h('a', { href: '#/models' }, '← Models')),
    h('div', { class: 'page-head' },
      h('h1', { class: 'mono', style: { fontSize: '16px', wordBreak: 'break-all' } }, m.name)),
    h('div', { class: 'row', style: { marginBottom: '14px' } },
      m.direction ? h('span', { class: 'tag ' + m.direction }, m.direction) : null,
      m.model ? h('span', { class: 'tag' }, m.model) : null,
      h('span', { class: 'tag' }, 'phone: ' + m.phone_type),
      m.sessions.length ? h('span', { class: 'tag' }, 'sessions ' + m.sessions.join(', ')) : null,
      h('span', { class: 'tag' }, `${m.epochs} epochs`),
      m.best ? h('span', { class: 'tag good' }, `best val loss @ epoch ${bestEpoch}`) : null,
      h('span', { class: 'muted' }, fmtDate(m.mtime))),
    h('div', { class: 'grid2' },
      h('div', {},
        h('div', { class: 'card' }, h('h2', {}, 'Publish to the phone'),
          h('div', { class: 'field' }, h('label', {}, 'Checkpoint'), ckpt),
          h('div', { class: 'field' }, h('label', {}, 'File name'), h('div', { class: 'row' }, target, h('span', { class: 'muted' }, '.tflite'))),
          m.seq_tag ? h('div', { class: 'muted', style: { fontSize: '12px' } }, `${m.seq_tag} is kept in the name: the phone reads the frame offsets from it.`) : null,
          m.steer_only ? h('div', { class: 'muted', style: { fontSize: '12px' } }, '_steeronly is kept in the name: the phone drives it only with Fixed or Manual throttle.') : null,
          m.phone_type !== 'CMDNAV' ? h('div', { class: 'note warn', style: { margin: '6px 0' } }, `The phone adds downloads as CMDNAV: set this one to ${m.phone_type} in Model Management.`) : null,
          h('div', { class: 'row', style: { marginTop: '8px' } }, h('button', { class: 'primary', disabled: !m.tflite.length, onclick: () => publish(false) }, 'Publish')),
          h('div', { style: { marginTop: '8px' } }, pubList),
          h('div', { class: 'muted', style: { fontSize: '12px', marginTop: '6px' } }, 'best.tflite = best-val.tflite, the checkpoint with the lowest val loss.')),
        h('div', { class: 'card' }, h('h2', {}, 'Metrics'),
          metricRows.length ? h('table', {},
            h('thead', {}, h('tr', {}, h('th', {}, ''), h('th', { class: 'num' }, 'val best'), h('th', { class: 'num' }, 'val last'), h('th', { class: 'num' }, 'train best'), h('th', { class: 'num' }, 'train last'))),
            h('tbody', {}, metricRows)) : h('div', { class: 'muted' }, 'No log.csv yet.')),
        h('div', { class: 'card' }, h('h2', {}, 'Check the TFLite files'),
          h('div', { class: 'dim', style: { marginBottom: '8px' } }, 'check_tflite.py: every .tflite in checkpoints/ against its Keras checkpoint on real test frames (match, stateless). Runs on the CPU, so it doesn\'t disturb a training run.'),
          h('button', { onclick: async () => {
            const job = await attempt(() => api(`models/${encodeURIComponent(name)}/check`, {}));
            if (job) clear(checkBox, jobLog(job));
          } }, 'Run check_tflite'), checkBox),
        h('div', { class: 'card' }, h('h2', {}, 'Notes'), notes,
          h('div', { class: 'row', style: { marginTop: '6px' } },
            h('button', { onclick: () => attempt(() => api(`models/${encodeURIComponent(name)}/notes`, { notes: notes.value }), 'Notes saved') }, 'Save'),
            h('span', { class: 'muted' }, 'kept in the model folder as notes.txt'))),
        m.manifest ? h('details', { class: 'card' }, h('summary', {}, 'Trained on (data_manifest.txt)'),
          h('div', { class: 'stack', style: { margin: '6px 0' } }, manifestSummary(m.manifest)),
          h('div', { class: 'log' }, m.manifest)) : null),
      h('div', {},
        h('div', { class: 'card' }, h('div', { class: 'row' }, h('h2', { style: { margin: 0 } }, 'Training curves'), h('span', { class: 'spacer', style: { flex: 1 } }), metricSel),
          m.rows.length ? chartBox : h('div', { class: 'muted' }, 'No epochs logged yet.')),
        m.plots.length ? h('div', { class: 'plots' }, m.plots.map((p) => h('figure', {},
          h('a', { href: modelFileUrl(`${name}/logs/${p}`), target: '_blank' }, h('img', { src: modelFileUrl(`${name}/logs/${p}`) + `?t=${Math.round(m.mtime)}`, loading: 'lazy', alt: p })),
          h('figcaption', {}, p)))) : null)));
  if (m.rows.length) drawChart();
  return () => { charts.forEach((c) => c.destroy()); checkBox.firstChild?.stop?.(); };
}
