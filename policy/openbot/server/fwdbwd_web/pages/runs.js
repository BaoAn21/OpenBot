// Autopilot runs: Session logs, takeovers per minute per model, the report plots, and your
// takeovers extracted into the inbox as training data.
import { h, clear, api, attempt, fmtS, recDate, fileUrl, jobLog } from '../lib.js';
import { dropzone } from '../upload.js';

export async function render(view) {
  const selected = new Set();
  const out = h('div');
  const body = h('div', {}, h('div', { class: 'empty' }, 'Loading…'));
  view.append(
    h('div', { class: 'page-head' }, h('h1', {}, 'Autopilot runs'),
      h('span', { class: 'muted' }, 'Autopilot Session logs (Logging switch). Takeovers per minute of model driving is the number to compare models by.')),
    body);
  let data = { runs: [], zips: [] };

  async function job(path, payload) {
    const j = await attempt(() => api(path, payload));
    if (!j) return;
    out.firstChild?.stop?.();
    clear(out, h('div', { class: 'card' }, jobLog(j, { tall: true, onDone: () => load() })));
    out.scrollIntoView({ behavior: 'smooth' });
  }

  function perModel(runs) {
    const agg = new Map();
    for (const r of runs) for (const p of r.per_model) {
      const k = `${p.direction} ${p.model}`;
      const a = agg.get(k) || { direction: p.direction, model: p.model, model_s: 0, human_s: 0, takeovers: 0, runs: 0 };
      a.model_s += p.model_s; a.human_s += p.human_s; a.takeovers += p.takeovers; a.runs += 1;
      agg.set(k, a);
    }
    return [...agg.values()].sort((a, b) => b.model_s - a.model_s);
  }

  function draw() {
    const runs = data.runs;
    const pool = selected.size ? runs.filter((r) => selected.has(r.name)) : runs;
    const agg = perModel(pool);
    const names = [...selected];
    const addCard = h('div', { class: 'card' }, h('h2', {}, 'Add runs'),
      dropzone({ hint: 'Drop autopilot_*.zip from the phone (Documents/OpenBot/) here, or', onDone: load }),
      data.zips.length ? h('div', { class: 'row', style: { marginTop: '8px' } },
        h('span', { class: 'tag warn' }, `${data.zips.length} zip(s) not unpacked`),
        h('button', { class: 'small', onclick: async () => { await attempt(() => api('runs/unpack', {}), 'Unpacked'); load(); } }, 'Unpack')) : null);
    const actionsCard = h('div', { class: 'card' }, h('h2', {}, 'Actions'),
      h('div', { class: 'dim', style: { marginBottom: '8px' } }, selected.size ? `${selected.size} run(s) selected.` : 'Select runs in the table below (Report works on all when none is selected).'),
      h('div', { class: 'row' },
        h('button', { onclick: () => job('runs/report', { runs: names }) }, 'Report (plots)'),
        h('button', { disabled: !selected.size, onclick: () => job('runs/extract', { runs: names, dry_run: true }) }, 'Preview extract'),
        h('button', { class: 'primary', disabled: !selected.size, onclick: () => {
          if (confirm(`Extract your takeovers from ${selected.size} run(s) into the inbox?`)) job('runs/extract', { runs: names });
        } }, 'Extract takeovers → inbox')),
      h('div', { class: 'muted', style: { fontSize: '12px', marginTop: '8px' } },
        'Extract keeps only the frames you drove, minus 0.3 s at each end of a takeover. Split them as their own session (test share 0) so you choose which models train on them.'));
    const aggCard = h('div', { class: 'card' }, h('h2', {}, `Per model (${selected.size ? selected.size + ' selected runs' : 'all runs'})`),
      agg.length ? h('table', {},
        h('thead', {}, h('tr', {}, h('th', {}, 'Model'), h('th', { class: 'num' }, 'runs'), h('th', { class: 'num' }, 'model drove'), h('th', { class: 'num' }, 'you drove'), h('th', { class: 'num' }, 'takeovers'), h('th', { class: 'num' }, 'takeovers/min'))),
        h('tbody', {}, agg.map((a) => h('tr', {},
          h('td', {}, h('span', { class: `tag ${a.direction}` }, a.direction), ' ', h('span', { class: 'mono' }, a.model || '(none)')),
          h('td', { class: 'num' }, a.runs),
          h('td', { class: 'num' }, fmtS(a.model_s)), h('td', { class: 'num' }, fmtS(a.human_s)),
          h('td', { class: 'num' }, a.takeovers),
          h('td', { class: 'num' }, h('b', {}, a.model_s ? (a.takeovers / (a.model_s / 60)).toFixed(1) : '–'))))))
        : h('div', { class: 'muted' }, 'No runs.'),
      h('div', { class: 'muted', style: { fontSize: '12px', marginTop: '6px' } }, 'Short runs that start just before a hard turn inflate the rate; compare models on similar runs.'));
    const allSelected = runs.length > 0 && selected.size === runs.length;
    const table = h('div', { class: 'card flush' }, h('table', {},
      h('thead', {}, h('tr', {},
        h('th', {}, h('input', { type: 'checkbox', checked: allSelected, onchange: (e) => { runs.forEach((r) => (e.target.checked ? selected.add(r.name) : selected.delete(r.name))); draw(); } })),
        h('th', {}, 'Run'), h('th', {}, 'Models'), h('th', { class: 'num' }, 'length'), h('th', { class: 'num' }, 'model'),
        h('th', { class: 'num' }, 'you'), h('th', { class: 'num' }, 'takeovers'), h('th', { class: 'num' }, '/min'), h('th', {}, ''))),
      h('tbody', {}, runs.map((r) => h('tr', {},
        h('td', {}, h('input', { type: 'checkbox', checked: selected.has(r.name), onchange: (e) => { e.target.checked ? selected.add(r.name) : selected.delete(r.name); draw(); } })),
        h('td', { class: 'nowrap' }, h('a', { href: `#/play?path=${encodeURIComponent(r.path)}&back=runs` }, recDate(r.name) || r.name),
          r.extracted ? h('div', {}, h('span', { class: 'tag good', title: r.extracted }, '✓ takeovers extracted')) : null),
        h('td', {}, r.per_model.map((p) => h('div', { class: 'nowrap' }, h('span', { class: `tag ${p.direction}` }, p.direction), ' ', h('span', { class: 'mono', style: { fontSize: '12px' } }, p.model || '(none)')))),
        h('td', { class: 'num' }, fmtS(r.seconds)),
        h('td', { class: 'num' }, fmtS(r.model_s)), h('td', { class: 'num' }, fmtS(r.human_s)),
        h('td', { class: 'num' }, r.takeovers), h('td', { class: 'num' }, r.per_min ?? '–'),
        h('td', { class: 'nowrap' },
          h('a', { href: `#/play?path=${encodeURIComponent(r.path)}&back=runs` }, '▶ play'), ' ',
          r.report ? h('a', { href: fileUrl(r.report), target: '_blank', title: 'autopilot_report.png' }, 'plot') : null))))));
    clear(body,
      h('div', { class: 'grid2' }, h('div', {}, addCard, actionsCard), aggCard),
      out,
      table);
  }

  async function load() {
    data = await api('runs');
    for (const n of [...selected]) if (!data.runs.some((r) => r.name === n)) selected.delete(n);
    draw();
  }
  await load();
  return () => out.firstChild?.stop?.();
}
