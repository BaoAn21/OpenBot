// Jobs: every script run from the app, with its live log.
import { h, clear, api, every, fmtS, fmtDate, jobLog, statusTag } from '../lib.js';

export async function render(view, params) {
  let current = params.id || null;
  const list = h('div', { class: 'card flush' });
  const detail = h('div');
  view.append(
    h('div', { class: 'page-head' }, h('h1', {}, 'Jobs'),
      h('span', { class: 'muted' }, 'Splits, training runs, checks and reports. Logs live in dataset/fwdbwd/.webapp/jobs/.')),
    h('div', { class: 'grid2' }, list, detail));
  let shown = null;

  function open(j) {
    current = j.id;
    if (shown) shown.stop();
    shown = jobLog(j, { tall: true });
    clear(detail, h('div', { class: 'card' }, shown, h('div', { class: 'muted mono', style: { marginTop: '8px', fontSize: '11.5px', wordBreak: 'break-all' } }, j.command)));
    history.replaceState(null, '', `#/jobs?id=${j.id}`);
  }

  async function refresh() {
    const jobs = await api('jobs');
    clear(list, jobs.length ? h('table', {}, h('tbody', {}, jobs.map((j) => h('tr', {
      class: 'click', style: j.id === current ? { background: 'var(--surface-2)' } : {}, onclick: () => { open(j); refresh(); },
    },
      h('td', {}, statusTag(j.status)),
      h('td', {}, h('div', {}, j.label), h('div', { class: 'muted', style: { fontSize: '12px' } }, `${j.kind} · ${fmtDate(j.created)}`)),
      h('td', { class: 'num muted' }, j.started ? fmtS((j.ended || Date.now() / 1000) - j.started) : '')))))
      : h('div', { class: 'empty' }, 'No jobs yet.'));
    if (current && !shown) {
      const j = jobs.find((x) => x.id === current);
      if (j) open(j);
    }
  }
  const stop = every(3000, refresh);
  return () => { stop(); shown?.stop(); };
}
