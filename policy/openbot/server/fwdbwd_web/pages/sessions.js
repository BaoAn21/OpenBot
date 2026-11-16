// Sessions: what each split batch holds, its note, and every recording with its parts.
import { h, clear, api, attempt, recDate } from '../lib.js';

const play = (path) => `#/play?path=${encodeURIComponent(path)}&back=sessions`;

function details(s) {
  return h('table', {},
    h('thead', {}, h('tr', {}, h('th', {}, 'Recording'), h('th', {}, 'Split'), h('th', {}, 'Forward runs'), h('th', {}, 'Backward'), h('th', {}, ''))),
    h('tbody', {}, s.recordings.map((r) => {
      const fwd = r.parts.filter((p) => p.direction === 'fwd');
      const bwd = r.parts.filter((p) => p.direction === 'bwd');
      const sum = (ps) => ps.reduce((a, p) => a + p.frames, 0);
      return h('tr', {},
        h('td', { class: 'mono' }, h('a', { href: play(r.path) }, r.name), ' ', h('span', { class: 'muted' }, recDate(r.name))),
        h('td', {}, h('span', { class: 'tag ' + (r.split === 'test' ? 'warn' : '') }, r.split)),
        h('td', {}, fwd.length ? h('span', {}, `${fwd.length} run${fwd.length > 1 ? 's' : ''}, ${sum(fwd)} frames `,
          fwd.map((p) => h('a', { href: play(p.path), class: 'mono', style: { marginRight: '6px', fontSize: '12px' }, title: p.name }, p.name.match(/_f\d+$/)?.[0].slice(1) || p.name))) : h('span', { class: 'muted' }, '–')),
        h('td', {}, bwd.length ? h('a', { href: play(bwd[0].path) }, `${sum(bwd)} frames`) : h('span', { class: 'muted' }, '–')),
        h('td', { class: 'right' }, h('a', { href: play(r.path) }, '▶ raw')));
    })));
}

function noteCell(s, refresh) {
  const td = h('td', {});
  const show = () => clear(td,
    h('span', { class: s.note ? '' : 'muted' }, s.note || 'no note'), ' ',
    h('button', { class: 'small', onclick: (e) => { e.stopPropagation(); edit(); } }, 'Edit'));
  const edit = () => {
    const ta = h('textarea', { rows: 2 }, s.note);
    clear(td, ta, h('div', { class: 'row', style: { marginTop: '4px' } },
      h('button', { class: 'small primary', onclick: async (e) => {
        e.stopPropagation();
        const ok = await attempt(() => api(`sessions/${s.number}/note`, { note: ta.value }), 'Note saved');
        if (ok !== undefined) { s.note = ta.value.trim(); refresh(); }
      } }, 'Save'),
      h('button', { class: 'small', onclick: (e) => { e.stopPropagation(); show(); } }, 'Cancel')));
    ta.addEventListener('click', (e) => e.stopPropagation());
    ta.focus();
  };
  show();
  return td;
}

export async function render(view) {
  const body = h('div', {}, h('div', { class: 'empty' }, 'Loading…'));
  view.append(
    h('div', { class: 'page-head' }, h('h1', {}, 'Sessions'),
      h('span', { class: 'muted' }, 'Each split of the inbox is one session; train on any choice of them.'),
      h('span', { class: 'spacer' }), h('a', { href: '#/train' }, 'Train on them →')),
    body);
  const open = new Set();

  let sessions = [];
  async function refresh() {
    sessions = await api('sessions');
    draw();
  }
  function draw() {
    const tot = { fwd_train: 0, fwd_test: 0, bwd_train: 0, bwd_test: 0, recs: 0 };
    for (const s of sessions) { for (const k of Object.keys(s.counts)) tot[k] += s.counts[k]; tot.recs += s.recordings.length; }
    const rows = [];
    for (const s of sessions.slice().reverse()) {
      const tr = h('tr', { class: 'click', onclick: () => { open.has(s.number) ? open.delete(s.number) : open.add(s.number); draw(); } },
        h('td', { class: 'nowrap' }, (open.has(s.number) ? '▾ ' : '▸ ') + `session ${s.number}`),
        noteCell(s, draw),
        h('td', { class: 'num' }, s.recordings.length),
        h('td', { class: 'num' }, s.counts.fwd_train), h('td', { class: 'num' }, s.counts.fwd_test || h('span', { class: 'muted' }, '0')),
        h('td', { class: 'num' }, s.counts.bwd_train), h('td', { class: 'num' }, s.counts.bwd_test || h('span', { class: 'muted' }, '0')),
        h('td', { class: 'num' }, s.test));
      rows.push(tr);
      if (open.has(s.number)) rows.push(h('tr', { class: 'sub' }, h('td', { colspan: 8 }, details(s))));
    }
    clear(body, h('div', { class: 'card flush' }, h('table', {},
      h('thead', {}, h('tr', {},
        h('th', {}, 'Session'), h('th', {}, 'Note'), h('th', { class: 'num' }, 'Recordings'),
        h('th', { class: 'num' }, 'fwd train'), h('th', { class: 'num' }, 'fwd test'),
        h('th', { class: 'num' }, 'bwd train'), h('th', { class: 'num' }, 'bwd test'), h('th', { class: 'num' }, 'test recs'))),
      h('tbody', {}, rows),
      h('tfoot', {}, h('tr', {},
        h('th', {}, 'all'), h('th', {}, ''), h('th', { class: 'num' }, tot.recs),
        h('th', { class: 'num' }, tot.fwd_train), h('th', { class: 'num' }, tot.fwd_test),
        h('th', { class: 'num' }, tot.bwd_train), h('th', { class: 'num' }, tot.bwd_test), h('th', {}, '')))),
      h('div', { class: 'muted', style: { padding: '8px 12px', fontSize: '12px' } }, 'Frame counts. Click a session for its recordings; every recording and part plays.')));
  }
  await refresh();
}
