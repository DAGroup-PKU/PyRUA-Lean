// The page's behaviour: the top bar that highlights the section being read, reveal on scroll, the bar charts that grow
// into view with their arrows (#rejudge), the demo videos (play muted while in view; a click opens them in a player with
// controls; tabs swap the episode), copy BibTeX and back to top.
// The <head> marks <html> with .js, so content starts hidden for the reveal only when this script runs.

const REDUCED = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

function copyBibTeX() {
  const code = document.getElementById('bibtex-code');
  const button = document.querySelector('.copy-bibtex');
  const label = button.querySelector('span');
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

// ---- top bar: highlight the section being read; back-to-top button ------------------------------------------------
function setupBar() {
  const links = [...document.querySelectorAll('.bar-nav a')];
  const sections = links.map(a => document.querySelector(a.getAttribute('href'))).filter(Boolean);
  const top = document.querySelector('.to-top');
  let last = null;
  function update() {
    if (top) top.classList.toggle('visible', window.pageYOffset > 400);
    // the last section whose top has passed a line a third of the way down the window
    const line = window.innerHeight / 3;
    let current = null;
    sections.forEach(s => { if (s.getBoundingClientRect().top <= line) current = s; });
    if (current === last) return;
    last = current;
    links.forEach(a => a.classList.toggle('active', current !== null && a.getAttribute('href') === '#' + current.id));
  }
  let pending = false;
  window.addEventListener('scroll', () => {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => { pending = false; update(); });
  }, { passive: true });
  if (top) top.addEventListener('click', () => window.scrollTo({ top: 0, behavior: REDUCED ? 'auto' : 'smooth' }));
  update();
}

// ---- reveal, bar charts and type bars: start when they scroll into view --------------------------------------------
function setupReveal() {
  const items = [...document.querySelectorAll('.rv, .bars-plot')];
  const show = el => { el.classList.add('in'); if (el._play) el._play(); };
  if (REDUCED || !('IntersectionObserver' in window)) { items.forEach(show); return; }
  // what is already on screen at load is shown at once; everything is shown when printed
  items.forEach(el => { if (el.getBoundingClientRect().top < window.innerHeight) show(el); });
  window.addEventListener('beforeprint', () => items.forEach(show));
  const seen = new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (!e.isIntersecting) return;
      show(e.target);
      seen.unobserve(e.target);
    });
  }, { threshold: 0.15 });
  items.forEach(el => { if (!el.classList.contains('in')) seen.observe(el); });
}

// ---- videos: muted loop while in view; a click opens the player; tabs swap the episode -----------------------------
function setupVideos() {
  const stages = [...document.querySelectorAll('.stage')];
  const tryPlay = v => { const p = v.play(); if (p && p.catch) p.catch(() => {}); };
  if (!REDUCED && 'IntersectionObserver' in window) {
    const seen = new IntersectionObserver(entries => {
      entries.forEach(e => {
        const v = e.target.querySelector('video');
        if (e.isIntersecting) tryPlay(v); else v.pause();
      });
    }, { threshold: 0.3 });
    stages.forEach(s => seen.observe(s));
  }

  const dialog = document.getElementById('player');
  const big = dialog.querySelector('video');
  const title = dialog.querySelector('.player-top b');
  const cap = dialog.querySelector('.player-cap');
  function open(stage) {
    const v = stage.querySelector('video');
    v.pause();
    title.textContent = stage.dataset.title || '';
    const c = document.getElementById(stage.dataset.cap);
    cap.innerHTML = c ? c.innerHTML : '';
    big.poster = v.poster;
    big.src = v.querySelector('source').src;
    if (dialog.showModal) dialog.showModal(); else dialog.setAttribute('open', '');
    big.currentTime = 0;
    tryPlay(big);
  }
  function close() {
    big.pause();
    big.removeAttribute('src');
    big.load();
    if (dialog.close) dialog.close(); else dialog.removeAttribute('open');
  }
  stages.forEach(s => {
    s.addEventListener('click', () => open(s));
    s.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(s); } });
  });
  dialog.querySelector('.player-close').addEventListener('click', close);
  // a click on the backdrop, outside the box, closes the player; Esc closes it too
  dialog.addEventListener('click', e => { if (e.target === dialog) close(); });
  dialog.addEventListener('cancel', e => { e.preventDefault(); close(); });

  document.querySelectorAll('.demo').forEach(demo => {
    const tabs = [...demo.querySelectorAll('.tabs button[data-video]')];
    const stage = demo.querySelector('.stage');
    const video = stage.querySelector('video');
    tabs.forEach(t => t.addEventListener('click', () => {
      if (t.classList.contains('active')) return;
      tabs.forEach(u => {
        const on = u === t;
        u.classList.toggle('active', on);
        u.setAttribute('aria-selected', on ? 'true' : 'false');
        document.getElementById(u.dataset.cap).hidden = !on;
      });
      stage.dataset.cap = t.dataset.cap;
      stage.dataset.title = t.dataset.title;
      video.poster = t.dataset.video + '.jpg';
      video.querySelector('source').src = t.dataset.video + '.mp4';
      video.load();
      if (!REDUCED) tryPlay(video);
    }));
  });
}

// ---- H2 bar charts, after Figures 4 and 5 of the paper: a light bar (judged by one criteria) and a solid bar (the same
// episodes judged by the other). When the chart scrolls into view (setupReveal calls _play) both bars rise to the first
// verdict; the old level stays as a dashed line, then the second bar slides to the new verdict, darkening, with an
// arrow and the change in points that count along with it. Replay plays it again.
const SVGNS = 'http://www.w3.org/2000/svg';

function svg(tag, attrs, parent) {
  const e = document.createElementNS(SVGNS, tag);
  Object.entries(attrs || {}).forEach(([k, v]) => e.setAttribute(k, v));
  if (parent) parent.appendChild(e);
  return e;
}

function lighten(hex, share) {
  // the colour mixed with white, keeping `share` of the colour
  const n = parseInt(hex.slice(1), 16);
  const mix = c => Math.round(c * share + 255 * (1 - share));
  return 'rgb(' + mix(n >> 16) + ',' + mix((n >> 8) & 255) + ',' + mix(n & 255) + ')';
}

function label(parent, x, y, item, cls) {
  // "read C^orig": a plain prefix, then the criteria symbol with its superscript
  const t = svg('text', { x: x, y: y, 'text-anchor': 'middle', class: cls }, parent);
  t.appendChild(document.createTextNode(item.text));
  if (item.crit) {
    svg('tspan', { class: 'crit' }, t).textContent = 'C';
    const sup = svg('tspan', { dy: -7, 'font-size': '10.5' }, t);
    sup.textContent = item.crit;
  }
  return t;
}

// progress of `t` through [t0, t0 + dur], clamped to 0..1, and the easings of the timeline
const phase = (t, t0, dur) => Math.min(1, Math.max(0, (t - t0) / dur));
const easeOut = p => 1 - Math.pow(1 - p, 3);
const easeInOut = p => (p < 0.5 ? 4 * p * p * p : 1 - Math.pow(-2 * p + 2, 3) / 2);

function drawChart(root) {
  const spec = JSON.parse(root.dataset.bars);
  // on a phone the chart is drawn on a narrower canvas, so its text is not scaled down as much
  const narrow = root.clientWidth < 460;
  const W = narrow ? 380 : 520, H = 336, L = narrow ? 36 : 44, R = 8, top = 78, bottom = 282, max = 80;
  const y = v => bottom - (bottom - top) * v / max;
  const s = svg('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': root.getAttribute('aria-label') || '' });
  for (let v = 0; v <= max; v += 20) {
    svg('line', { x1: L, x2: W - R, y1: y(v), y2: y(v), class: v ? 'grid' : 'axis' }, s);
    svg('text', { x: L - 10, y: y(v) + 4, 'text-anchor': 'end', class: 'tick' }, s).textContent = v;
  }
  const yx = narrow ? 6 : 14;
  const yl = svg('text', { x: yx, y: (top + bottom) / 2, class: 'ylab', transform: `rotate(-90 ${yx} ${(top + bottom) / 2})`, 'text-anchor': 'middle' }, s);
  yl.textContent = 'SR (%)';
  const span = (W - R - L) / spec.groups.length, bw = narrow ? 50 : 66, gap = narrow ? 8 : 12;
  const groups = spec.groups.map((g, k) => {
    const cx = L + span * (k + 0.5);
    const xa = cx - (bw + gap) / 2, xb = cx + (bw + gap) / 2, ax = xb + bw / 2 + 10;
    const e = { g: g, k: k, cx: cx, xa: xa, xb: xb, ax: ax };
    e.barA = svg('rect', { x: xa - bw / 2, width: bw, rx: 5, fill: lighten(g.c, 0.34) }, s);
    e.barB = svg('rect', { x: xb - bw / 2, width: bw, rx: 5 }, s);
    // the old level, over the bars, from the light bar across to the arrow
    e.ghost = svg('line', { x1: xa + bw / 2, x2: xa + bw / 2, y1: y(g.a), y2: y(g.a), class: 'ghost' }, s);
    e.valA = svg('text', { x: xa, 'text-anchor': 'middle', class: 'val', fill: '#5B6577' }, s);
    e.valB = svg('text', { x: xb, 'text-anchor': 'middle', class: 'val', fill: g.c }, s);
    e.shaft = svg('line', { x1: ax, x2: ax, class: 'arw', stroke: g.c }, s);
    e.head = svg('path', { fill: g.c }, s);
    e.pill = svg('g', { class: 'delta' }, s);
    svg('rect', { x: -30, y: -12, width: 60, height: 24, rx: 12, fill: g.c }, e.pill);
    e.dtxt = svg('text', { x: 0, y: 5, 'text-anchor': 'middle', class: 'dtxt' }, e.pill);
    e.py = Math.min(y(g.a), y(g.b)) - 46;
    label(s, cx, bottom + 26, g, 'xlab');
    return e;
  });
  const fmt = v => v.toFixed(1);
  const signed = d => (d > 0 ? '+' : d < 0 ? '−' : '') + Math.abs(d).toFixed(1);
  // one frame of the timeline at `t` seconds; each group starts a little after the one before
  function frame(t) {
    groups.forEach(e => {
      const g = e.g, o = e.k * 0.18, d = g.b - g.a;
      const grow = easeOut(phase(t, o, 0.9));
      const ghost = easeOut(phase(t, o + 1.0, 0.35));
      const m = phase(t, o + 1.35, 1.0), morph = easeInOut(m);
      const bump = Math.sin(Math.PI * phase(t, o + 2.35, 0.3));
      const va = g.a * grow, vb = grow < 1 ? va : g.a + d * morph;
      e.barA.setAttribute('y', y(va));
      e.barA.setAttribute('height', bottom - y(va));
      e.barB.setAttribute('y', y(vb));
      e.barB.setAttribute('height', bottom - y(vb));
      e.barB.setAttribute('fill', lighten(g.c, 0.34 + 0.66 * morph));
      [[e.valA, e.xa, va], [e.valB, e.xb, vb]].forEach(([el, x, v]) => {
        el.setAttribute('y', y(v) - 8);
        el.setAttribute('opacity', Math.min(1, grow * 3));
        el.textContent = fmt(v);
      });
      e.ghost.setAttribute('x2', e.xa + bw / 2 + (e.ax + 6 - e.xa - bw / 2) * ghost);
      e.ghost.setAttribute('opacity', ghost ? 1 : 0);
      // the arrow runs from the old level to the bar's top as it moves; hidden while too short to read
      const y0 = y(g.a), y1 = y(vb), dir = d > 0 ? 1 : -1, len = Math.abs(y1 - y0);
      const shown = len > 10 ? 1 : 0;
      e.shaft.setAttribute('y1', y0);
      e.shaft.setAttribute('y2', y1 + dir * 7);
      e.shaft.setAttribute('opacity', shown);
      e.head.setAttribute('d', `M ${e.ax - 6} ${y1 + dir * 9} L ${e.ax + 6} ${y1 + dir * 9} L ${e.ax} ${y1} Z`);
      e.head.setAttribute('opacity', shown);
      e.dtxt.textContent = signed(d * morph);
      e.pill.setAttribute('opacity', Math.min(1, m * 4));
      e.pill.setAttribute('transform', `translate(${e.cx} ${e.py + 8 * (1 - Math.min(1, m * 2.5))}) scale(${1 + 0.12 * bump})`);
    });
  }
  const end = 0.18 * (groups.length - 1) + 2.7;
  let raf = 0;
  root._play = () => {
    cancelAnimationFrame(raf);
    if (REDUCED) { frame(end); return; }
    const t0 = performance.now();
    const step = now => {
      const t = (now - t0) / 1000;
      frame(t);
      if (t < end) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
  };
  frame(0);
  const plot = root.querySelector('.bars-plot');
  plot._play = root._play;
  plot.appendChild(s);
  window.addEventListener('beforeprint', () => { cancelAnimationFrame(raf); frame(end); });
  const replay = root.querySelector('.replay');
  if (replay) replay.addEventListener('click', root._play);
}

function setupCharts() {
  document.querySelectorAll('.bars[data-bars]').forEach(drawChart);
}

document.addEventListener('DOMContentLoaded', function () {
  setupBar();
  setupCharts();
  setupReveal();
  setupVideos();
  const copy = document.querySelector('.copy-bibtex');
  if (copy) copy.addEventListener('click', copyBibTeX);
});
