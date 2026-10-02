// The page's behaviour: copy BibTeX, scroll to top, episode clips that play only while on screen, the call-by-call
// comparison of the two agents (#duel), and the code viewer of the gallery (#viewer). Data for the last two is
// embedded in the page as JSON (#duel-data, #gallery-data); the code in it is already syntax-highlighted HTML.

function copyBibTeX() {
  const code = document.getElementById('bibtex-code');
  const button = document.querySelector('.copy-bibtex-btn');
  const label = button.querySelector('.copy-text');
  const done = () => {
    button.classList.add('copied');
    label.textContent = 'Copied';
    setTimeout(() => { button.classList.remove('copied'); label.textContent = 'Copy'; }, 2000);
  };
  const fallback = () => {
    const area = document.createElement('textarea');
    area.value = code.textContent;
    document.body.appendChild(area);
    area.select();
    document.execCommand('copy');
    document.body.removeChild(area);
    done();
  };
  if (navigator.clipboard && window.isSecureContext) {
    navigator.clipboard.writeText(code.textContent).then(done, fallback);
  } else {
    fallback();
  }
}

function scrollToTop() {
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

window.addEventListener('scroll', function () {
  const button = document.querySelector('.scroll-to-top');
  if (button) button.classList.toggle('visible', window.pageYOffset > 300);
});

// ---- clips: play while on screen ----------------------------------------------------------------------------------
const REDUCED = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
let clipObserver = null;

function watchClip(v) {
  if (REDUCED || !clipObserver) {
    v.controls = true;
    return;
  }
  clipObserver.observe(v);
}

// ---- helpers ------------------------------------------------------------------------------------------------------
function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

function fmtTokens(n) {
  if (n >= 1e6) return (n / 1e6).toFixed(2) + 'M';
  return Math.round(n / 1000) + 'k';
}

function printedBlock(text, more) {
  const d = el('details', 'printed');
  d.appendChild(el('summary', null, 'What it printed'));
  const pre = el('pre', null, text || '(nothing)');
  d.appendChild(pre);
  if (more) d.appendChild(el('p', 'more-note', '… ' + more + ' more line' + (more === 1 ? '' : 's') + ' left out'));
  return d;
}

// ---- one task, two agents -----------------------------------------------------------------------------------------
function setupDuel() {
  const root = document.getElementById('duel');
  const dataEl = document.getElementById('duel-data');
  if (!root || !dataEl) return;
  const data = JSON.parse(dataEl.textContent);
  const agents = [['tool', data.tool], ['code', data.code]];
  const total = Math.max(data.tool.calls.length, data.code.calls.length);
  const maxCum = Math.max(data.tool.calls[data.tool.calls.length - 1].cum, data.code.calls[data.code.calls.length - 1].cum);
  let k = 1;
  let timer = null;

  // controls
  const controls = el('div', 'duel-controls');
  const prev = el('button', 'duel-btn', '◀ Previous');
  const play = el('button', 'duel-btn play', '▶ Play');
  const next = el('button', 'duel-btn', 'Next ▶');
  const label = el('span', 'duel-k');
  [prev, play, next].forEach(b => { b.type = 'button'; controls.appendChild(b); });
  controls.appendChild(label);
  root.appendChild(controls);

  // one row of squares per agent: a square per LLM call
  const strips = el('div', 'duel-strips');
  const boxes = {};
  agents.forEach(([key, a]) => {
    const row = el('div', 'strip ' + key);
    row.appendChild(el('span', 'strip-label', a.name));
    const cells = el('div', 'strip-boxes');
    boxes[key] = a.calls.map(c => {
      const b = el('button', 'callbox' + (c.images ? ' img' : '') + (c.k === a.solved ? ' solved' : ''), String(c.k));
      b.type = 'button';
      b.title = 'LLM call ' + c.k + (c.images ? ' · ' + c.images + ' camera image' + (c.images === 1 ? '' : 's') : '');
      b.addEventListener('click', () => { stop(); show(c.k); });
      cells.appendChild(b);
      return b;
    });
    row.appendChild(cells);
    strips.appendChild(row);
  });
  root.appendChild(strips);

  // the two panels
  const panels = el('div', 'duel-panels');
  const refs = {};
  agents.forEach(([key, a]) => {
    const p = el('div', 'duel-panel ' + key);
    const head = el('div', 'panel-head');
    head.appendChild(el('span', 'agent-tag ' + key, a.name));
    const pk = el('span', 'panel-k');
    head.appendChild(pk);
    p.appendChild(head);
    const views = el('div', 'panel-views');
    const img = el('img');
    img.alt = a.name + ': agent-view camera after this LLM call';
    const wrist = el('img');
    wrist.alt = a.name + ': wrist camera after this LLM call';
    views.appendChild(img);
    views.appendChild(wrist);
    p.appendChild(views);
    const meter = el('div', 'panel-meter');
    const bar = el('div', 'meter-bar');
    const fill = el('div', 'meter-fill');
    bar.appendChild(fill);
    const tok = el('span', 'meter-text');
    meter.appendChild(bar);
    meter.appendChild(tok);
    p.appendChild(meter);
    const status = el('div', 'panel-status');
    p.appendChild(status);
    const action = el('div', 'panel-action');
    p.appendChild(action);
    panels.appendChild(p);
    refs[key] = { a, pk, img, wrist, fill, tok, status, action };
  });
  root.appendChild(panels);

  function renderAgent(key) {
    const r = refs[key];
    const a = r.a;
    const n = a.calls.length;
    const kk = Math.min(k, n);
    const c = a.calls[kk - 1];
    const images = a.calls.slice(0, kk).reduce((s, x) => s + (x.images || 0), 0);
    r.pk.textContent = 'LLM call ' + kk + ' of ' + n;
    r.img.src = c.frame;
    r.wrist.src = c.wrist;
    r.fill.style.width = (100 * c.cum / maxCum).toFixed(1) + '%';
    r.tok.textContent = fmtTokens(c.cum) + ' tokens so far · ' + images + ' camera image' + (images === 1 ? '' : 's');
    r.status.textContent = '';
    r.status.className = 'panel-status';
    if (k > n) {
      r.status.textContent = 'Finished after ' + n + ' LLM calls; solved at call ' + a.solved + '.';
      r.status.classList.add('done');
    } else if (kk === a.solved) {
      r.status.textContent = '✓ This call solved the task.';
      r.status.classList.add('solved');
    } else if (kk > a.solved) {
      r.status.textContent = 'Solved at call ' + a.solved + '; closing calls after it.';
      r.status.classList.add('done');
    }
    r.action.innerHTML = '';
    if (key === 'tool') {
      if (c.tools.length) {
        const list = el('ul', 'tool-list');
        c.tools.forEach(([name, count, args]) => {
          const li = el('li');
          li.appendChild(el('code', null, name));
          if (count > 1) li.appendChild(el('span', 'times', ' ×' + count));
          const shown = args.filter(Boolean);
          if (shown.length) li.appendChild(el('span', 'args', '  ' + shown.join(' · ') + (count > shown.length ? ' · …' : '')));
          list.appendChild(li);
        });
        r.action.appendChild(list);
      } else {
        r.action.appendChild(el('p', 'note', 'No tool call: the final message.'));
      }
      if (c.images) r.action.appendChild(el('p', 'returned', '+ ' + c.images + ' camera images returned'));
      if (c.note) r.action.appendChild(el('p', 'note', '“' + c.note + '”'));
    } else {
      if (c.code) {
        const headText = 'Python cell · ' + c.lines + ' line' + (c.lines === 1 ? '' : 's') +
          (c.images ? ' · asked for ' + c.images + ' image' + (c.images === 1 ? '' : 's') : ' · no image');
        r.action.appendChild(el('p', 'cell-head', headText));
        const pre = el('pre', 'code-block small');
        const code = el('code');
        code.innerHTML = c.code;
        pre.appendChild(code);
        r.action.appendChild(pre);
        if (c.printed) r.action.appendChild(printedBlock(c.printed, c.more));
      } else {
        r.action.appendChild(el('p', 'note', 'No code: the final message.'));
      }
    }
  }

  function show(newK) {
    k = Math.max(1, Math.min(total, newK));
    label.textContent = 'LLM call ' + k + ' of ' + total;
    agents.forEach(([key, a]) => {
      boxes[key].forEach((b, i) => b.classList.toggle('current', i + 1 === Math.min(k, a.calls.length) && k <= a.calls.length));
      renderAgent(key);
    });
    prev.disabled = k === 1;
    next.disabled = k === total;
  }

  function stop() {
    if (timer) clearInterval(timer);
    timer = null;
    play.textContent = '▶ Play';
  }

  prev.addEventListener('click', () => { stop(); show(k - 1); });
  next.addEventListener('click', () => { stop(); show(k + 1); });
  play.addEventListener('click', () => {
    if (timer) { stop(); return; }
    if (k === total) show(1);
    play.textContent = '❚❚ Pause';
    timer = setInterval(() => { if (k >= total) { stop(); return; } show(k + 1); }, REDUCED ? 2600 : 1400);
  });
  root.tabIndex = 0;
  root.addEventListener('keydown', e => {
    if (e.key === 'ArrowRight') { stop(); show(k + 1); e.preventDefault(); }
    if (e.key === 'ArrowLeft') { stop(); show(k - 1); e.preventDefault(); }
  });
  show(1);
}

// ---- gallery code viewer ------------------------------------------------------------------------------------------
function setupViewer() {
  const root = document.getElementById('viewer');
  const dataEl = document.getElementById('gallery-data');
  if (!root || !dataEl) return;
  const data = JSON.parse(dataEl.textContent);
  const tiles = document.querySelectorAll('.gallery-item');

  function open(i, scroll) {
    const ep = data[i];
    tiles.forEach(t => t.classList.toggle('active', Number(t.dataset.ep) === i));
    root.innerHTML = '';
    const head = el('div', 'viewer-head');
    head.appendChild(el('span', 'bench', ep.bench));
    head.appendChild(el('span', 'task', ep.task));
    head.appendChild(el('span', 'count', ep.cells.length + ' cell' + (ep.cells.length === 1 ? '' : 's') + ', task solved'));
    root.appendChild(head);
    const body = el('div', 'viewer-body');
    const left = el('div', 'viewer-video');
    const v = el('video', 'inview');
    v.muted = true;
    v.loop = true;
    v.playsInline = true;
    v.setAttribute('playsinline', '');
    v.preload = 'metadata';
    v.poster = ep.poster;
    const src = el('source');
    src.src = ep.video;
    src.type = 'video/mp4';
    v.appendChild(src);
    left.appendChild(v);
    if (ep.speed) left.appendChild(el('p', 'speed-note', 'Sped up ' + ep.speed + '.'));
    body.appendChild(left);
    // one cell at a time: a tab per cell, previous / next, the code in a fixed-height box
    const right = el('div', 'viewer-cells');
    const tabs = el('div', 'cell-tabs');
    tabs.setAttribute('role', 'tablist');
    const pane = el('div', 'cell-pane');
    const nav = el('div', 'cell-nav');
    const back = el('button', 'duel-btn', '◀ Previous cell');
    const fwd = el('button', 'duel-btn', 'Next cell ▶');
    back.type = 'button';
    fwd.type = 'button';
    nav.appendChild(back);
    nav.appendChild(fwd);
    const tabButtons = ep.cells.map((c, j) => {
      const t = el('button', 'cell-tab', 'Cell ' + (j + 1));
      t.type = 'button';
      t.setAttribute('role', 'tab');
      t.addEventListener('click', () => showCell(j));
      tabs.appendChild(t);
      return t;
    });
    function showCell(j) {
      const c = ep.cells[j];
      tabButtons.forEach((t, i) => {
        t.classList.toggle('active', i === j);
        t.setAttribute('aria-selected', i === j ? 'true' : 'false');
      });
      pane.innerHTML = '';
      pane.appendChild(el('p', 'cell-head', 'Cell ' + (j + 1) + ' of ' + ep.cells.length +
        (c.images ? ' · asked for ' + c.images + ' image' + (c.images === 1 ? '' : 's') : ' · no image')));
      const pre = el('pre', 'code-block small');
      const code = el('code');
      code.innerHTML = c.code;
      pre.appendChild(code);
      pane.appendChild(pre);
      if (c.printed) pane.appendChild(printedBlock(c.printed, c.more));
      if (j === ep.cells.length - 1 && ep.final) {
        const f = el('p', 'final');
        f.appendChild(el('strong', null, 'The agent’s final report: '));
        f.appendChild(document.createTextNode(ep.final));
        pane.appendChild(f);
      }
      back.disabled = j === 0;
      fwd.disabled = j === ep.cells.length - 1;
      back.onclick = () => showCell(j - 1);
      fwd.onclick = () => showCell(j + 1);
    }
    right.appendChild(tabs);
    right.appendChild(pane);
    if (ep.cells.length > 1) right.appendChild(nav);
    showCell(0);
    body.appendChild(right);
    root.appendChild(body);
    watchClip(v);
    if (scroll) root.scrollIntoView({ behavior: REDUCED ? 'auto' : 'smooth', block: 'nearest' });
  }

  document.querySelectorAll('.code-btn').forEach(b => {
    b.addEventListener('click', () => open(Number(b.dataset.ep), true));
  });

  // a tab per benchmark: show its grid, and its first episode in the viewer
  const benchTabs = document.querySelectorAll('.bench-tab');
  const grids = document.querySelectorAll('.gallery-grid[data-group]');
  function showGroup(group) {
    benchTabs.forEach(t => {
      const on = t.dataset.group === group;
      t.classList.toggle('active', on);
      t.setAttribute('aria-selected', on ? 'true' : 'false');
    });
    let first = null;
    grids.forEach(g => {
      const on = g.dataset.group === group;
      g.hidden = !on;
      if (!on) g.querySelectorAll('video').forEach(v => v.pause());
      if (on && first === null) first = Number(g.querySelector('.gallery-item').dataset.ep);
    });
    if (first !== null) open(first, false);
  }
  benchTabs.forEach(t => t.addEventListener('click', () => showGroup(t.dataset.group)));
  open(0, false);
}

document.addEventListener('DOMContentLoaded', function () {
  if (!REDUCED && 'IntersectionObserver' in window) {
    clipObserver = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        const v = entry.target;
        if (entry.isIntersecting) {
          if (v.preload === 'none') v.preload = 'auto';
          const p = v.play();
          if (p && p.catch) p.catch(function () { v.controls = true; });
        } else {
          v.pause();
        }
      });
    }, { threshold: 0.35 });
  }
  document.querySelectorAll('video.inview').forEach(watchClip);
  setupDuel();
  setupViewer();
  setupToc();
});

// ---- contents bar: highlight the section being read ---------------------------------------------------------------
function setupToc() {
  const links = [...document.querySelectorAll('.toc a')];
  const sections = links.map(a => document.querySelector(a.getAttribute('href'))).filter(Boolean);
  if (!sections.length) return;
  let last = null;
  function update() {
    // the last section whose top has passed a line a third of the way down the window
    const line = window.innerHeight / 3;
    let current = null;
    sections.forEach(s => { if (s.getBoundingClientRect().top <= line) current = s; });
    if (current === last) return;
    last = current;
    links.forEach(a => {
      const on = current !== null && a.getAttribute('href') === '#' + current.id;
      a.classList.toggle('active', on);
      // on a narrow bar, bring the highlighted entry into view
      const bar = a.parentElement;
      if (on && (a.offsetLeft < bar.scrollLeft || a.offsetLeft + a.offsetWidth > bar.scrollLeft + bar.clientWidth)) {
        bar.scrollTo({ left: a.offsetLeft - 16 });
      }
    });
  }
  let pending = false;
  window.addEventListener('scroll', () => {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => { pending = false; update(); });
  }, { passive: true });
  update();
}
