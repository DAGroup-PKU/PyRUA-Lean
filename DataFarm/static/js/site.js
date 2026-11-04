// DataFarm project page. Showcase with action-label trajectories, view-contrast panels, drag-to-compare player,
// close-up zoom, real-robot chart, cost chart, task tabs, scroll reveal.
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
const SVGNS = 'http://www.w3.org/2000/svg';
const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
function svgEl(tag, attrs, text) {
  const e = document.createElementNS(SVGNS, tag);
  Object.entries(attrs || {}).forEach(([k, v]) => e.setAttribute(k, v));
  if (text !== undefined) e.textContent = text;
  return e;
}
function playSafe(v) { const p = v.play(); if (p && p.catch) p.catch(() => { v.controls = true; }); }
// draw loop driven by decoded frames when the browser supports it
function onFrames(v, draw) {
  if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) {
    const cb = (now, meta) => { draw(meta.mediaTime); v.requestVideoFrameCallback(cb); };
    v.requestVideoFrameCallback(cb);
  }
  const tick = () => { if (!('requestVideoFrameCallback' in HTMLVideoElement.prototype) || v.paused) draw(v.currentTime); requestAnimationFrame(tick); };
  requestAnimationFrame(tick);
}
function setSource(v, src) {
  const s = v.querySelector('source');
  if (s.getAttribute('src') === src) return false;
  v.poster = src.replace(/\.mp4$/, '.jpg');
  s.src = src; v.load(); playSafe(v);
  return true;
}
function tabs(group, attr, pick, first) {
  const box = $(`.tabs[data-group="${group}"]`);
  if (!box) return () => {};
  const btns = $$('button', box);
  const go = key => { btns.forEach(b => b.classList.toggle('active', b.dataset[attr] === key)); pick(key); };
  btns.forEach(b => b.addEventListener('click', () => go(b.dataset[attr])));
  go(first || btns[0].dataset[attr]);
  return go;
}
// play while visible (keeps many clips from decoding at once)
const vio = 'IntersectionObserver' in window ? new IntersectionObserver(es => es.forEach(e => {
  const v = e.target; if (e.isIntersecting) playSafe(v); else v.pause();
}), { threshold: 0.15 }) : null;
function watchVideo(v) { if (vio) vio.observe(v); else playSafe(v); }

/* ---------------- showcase: simulator | DataFarm, three views, trajectories from the action labels ---------------- */
async function setupShowcase() {
  const v = $('#show-src'), cv = $('#show-canvas'), svg = $('#show-svg');
  if (!v) return;
  const ctx = cv.getContext('2d');
  let segs = null;
  try { segs = await (await fetch('static/demo/show.json')).json(); } catch (e) { segs = null; }
  const ARM = { L: '#E26A58', R: '#5EC8DC' };
  const order = ['towel', 'stack', 'drawer'];
  let key = 'towel', vertical = false;
  const W = 800, H = 676, HH = H * 2 / 3;                       // one T panel: head on top (2/3), wrists below
  function layout() {
    vertical = cv.parentElement.clientWidth < 640;
    const w = vertical ? W : 2 * W, h = vertical ? 2 * H : H;
    if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; svg.setAttribute('viewBox', `0 0 ${w} ${h}`); }
  }
  const off = k => (vertical ? [0, k * H] : [k * W, 0]);       // panel k: 0 = simulator, 1 = DataFarm
  // static layer per panel: view seams, view chips, panel tag; dynamic layer: trails, current points, events
  function staticLayer() {
    svg.innerHTML = '';
    const seg = segs && segs[key];
    for (let k = 0; k < 2; k++) {
      const [x, y] = off(k), g = svgEl('g', { transform: `translate(${x},${y})` });
      g.appendChild(svgEl('rect', { x: 0, y: HH - 1.5, width: W, height: 3, fill: '#0A0E16' }));
      g.appendChild(svgEl('rect', { x: W / 2 - 1.5, y: HH, width: 3, height: H - HH, fill: '#0A0E16' }));
      [['Head', 12, 12, '#F3F5F8'], ['Left wrist', 12, HH + 10, ARM.L], ['Right wrist', W / 2 + 12, HH + 10, ARM.R]].forEach(([t, cx, cy, c]) => {
        const w = t.length * 7.4 + 18;
        g.appendChild(svgEl('rect', { x: cx, y: cy, width: w, height: 24, rx: 5, fill: 'rgba(10,14,22,0.72)', stroke: 'rgba(255,255,255,0.14)' }));
        g.appendChild(svgEl('text', { x: cx + 9, y: cy + 16.5, fill: c, 'font-size': 13, 'font-weight': 600, 'font-family': 'Inter, sans-serif' }, t));
      });
      const tag = k ? 'DataFarm render' : `Simulation · ${seg ? seg.task : ''}`;
      const tw = tag.length * 7.6 + 22;
      g.appendChild(svgEl('rect', { x: W - tw - 12, y: 12, width: tw, height: 26, rx: 6, fill: k ? 'rgba(139,20,20,0.92)' : 'rgba(60,70,88,0.92)' }));
      g.appendChild(svgEl('text', { x: W - tw - 1, y: 30, fill: '#fff', 'font-size': 13.5, 'font-weight': 700, 'font-family': 'Inter, sans-serif' }, tag));
      if (seg) ['L', 'R'].forEach(a => {
        const pts = seg.trails[a].filter(Boolean);
        if (pts.length < 2) return;
        const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
        if (Math.max(...xs) - Math.min(...xs) + Math.max(...ys) - Math.min(...ys) <= 0.04) return;   // arm that does not move
        g.appendChild(svgEl('polyline', { points: pts.map(p => `${(p[0] * W).toFixed(1)},${(p[1] * HH).toFixed(1)}`).join(' '),
          fill: 'none', stroke: 'white', 'stroke-opacity': 0.28, 'stroke-width': 1.6, 'stroke-dasharray': '2 7', 'stroke-linecap': 'round' }));
      });
      g.appendChild(svgEl('g', { class: 'dyn' }));
      svg.appendChild(g);
    }
  }
  function drawDyn(j) {
    const seg = segs && segs[key];
    $$('g.dyn', svg).forEach(dyn => {
      dyn.innerHTML = '';
      if (!seg) return;
      ['L', 'R'].forEach(a => {
        const pts = seg.trails[a], live = pts.filter(Boolean);
        if (live.length < 2) return;
        const xs = live.map(p => p[0]), ys = live.map(p => p[1]);
        if (Math.max(...xs) - Math.min(...xs) + Math.max(...ys) - Math.min(...ys) <= 0.04) return;
        const L = 75;
        for (let k = Math.max(1, j - L); k <= Math.min(j, pts.length - 1); k++) {
          const p = pts[k - 1], q = pts[k];
          if (!p || !q) continue;
          const o = Math.pow(1 - (j - k) / L, 1.6);
          dyn.appendChild(svgEl('line', { x1: p[0] * W, y1: p[1] * HH, x2: q[0] * W, y2: q[1] * HH, stroke: ARM[a], 'stroke-width': 1.5 + 3.5 * o,
            'stroke-opacity': 0.95 * o, 'stroke-linecap': 'round' }));
        }
        const c = pts[Math.min(j, pts.length - 1)];
        if (c) {
          const pulse = 0.5 + 0.5 * Math.sin(j / 4);
          dyn.appendChild(svgEl('circle', { cx: c[0] * W, cy: c[1] * HH, r: 13 + 6 * pulse, fill: 'none', stroke: ARM[a], 'stroke-opacity': 0.45 * (1 - pulse) + 0.15, 'stroke-width': 2 }));
          dyn.appendChild(svgEl('circle', { cx: c[0] * W, cy: c[1] * HH, r: 6.5, fill: ARM[a], stroke: 'white', 'stroke-width': 2.4 }));
        }
      });
      // grasp / release: the wrist view of that arm flashes, with the event name
      seg.events.filter(e => j >= e.f && j < e.f + 36).forEach(e => {
        const p = 1 - (j - e.f) / 36, x = e.arm === 'L' ? 0 : W / 2;
        dyn.appendChild(svgEl('rect', { x: x + 2, y: HH + 2, width: W / 2 - 4, height: H - HH - 4, fill: 'none', stroke: ARM[e.arm],
          'stroke-width': 2 + 3 * p, opacity: p }));
        const t = e.kind, bx = x + 12 + ((e.arm === 'L' ? 'Left wrist' : 'Right wrist').length * 7.4 + 18) + 8;
        dyn.appendChild(svgEl('rect', { x: bx, y: HH + 10, width: t.length * 8 + 18, height: 24, rx: 5, fill: ARM[e.arm], opacity: Math.min(1, p * 3) }));
        dyn.appendChild(svgEl('text', { x: bx + 9, y: HH + 26.5, fill: '#0A0E16', 'font-size': 13, 'font-weight': 700, opacity: Math.min(1, p * 3),
          'font-family': 'Inter, sans-serif' }, t));
      });
    });
  }
  function draw(t) {
    if (v.readyState < 2) return;
    for (let k = 0; k < 2; k++) { const [x, y] = off(k); ctx.drawImage(v, k * W, 0, W, H, x, y, W, H); }
    const seg = segs && segs[key];
    drawDyn(seg ? Math.min(seg.frames - 1, Math.floor(t * 30 + 1e-3)) : 0);
  }
  const pick = tabs('show', 'key', k => {
    key = k; layout(); staticLayer();
    setSource(v, `static/demo/show_${k}.mp4`) || (v.currentTime = 0, playSafe(v));
  }, 'towel');
  v.addEventListener('ended', () => pick(order[(order.indexOf(key) + 1) % order.length]));   // cycle through the episodes
  window.addEventListener('resize', () => { const was = vertical; layout(); if (was !== vertical) staticLayer(); });
  onFrames(v, draw);
  playSafe(v);
}

/* ---------------- contrast: four renderers of the same episode, a row (desktop) or 2x2 (phone) ---------------- */
function setupViews(data) {
  const v = $('#views-src'), cv = $('#views-canvas');
  if (!v) return;
  const ctx = cv.getContext('2d'), PW = 480, PH = 404;
  let grid = false;
  function layout() {
    grid = cv.parentElement.clientWidth < 760;
    const w = grid ? 2 * PW : 4 * PW, h = grid ? 2 * PH : PH;
    if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
    $('.views-frames').style.gridTemplateRows = grid ? '1fr 1fr' : '1fr';
  }
  function draw() {
    if (v.readyState < 2) return;
    for (let k = 0; k < 4; k++) {
      const x = grid ? (k % 2) * PW : k * PW, y = grid ? Math.floor(k / 2) * PH : 0;
      ctx.drawImage(v, k * PW, 0, PW, PH, x, y, PW, PH);
    }
  }
  tabs('views', 'task', t => { layout(); setSource(v, `static/videos/views_${t}.mp4`); $('#views-note').textContent = data.views[t]; }, 'drawer');
  window.addEventListener('resize', layout);
  onFrames(v, draw);
  watchVideo(v);
}

/* ---------------- drag-to-compare: simulator | DataFarm (side-by-side source, split on a canvas) ---------------- */
function setupWipe(data) {
  const box = $('#wipe'), v = $('#wipe-src'), cv = $('#wipe-canvas');
  if (!box) return;
  const ctx = cv.getContext('2d'), handle = $('.wipe-handle', box);
  let split = 0.5, touched = false, t0 = performance.now();
  const W = 960, H = 540;
  function draw() {
    if (!touched && !reduced) split = 0.5 + 0.28 * Math.sin((performance.now() - t0) / 1400);   // idle sweep invites a drag
    handle.style.left = (split * 100) + '%';
    box.setAttribute('aria-valuenow', Math.round(split * 100));
    if (v.readyState < 2) return;
    const sx = Math.round(split * W);
    ctx.drawImage(v, 0, 0, W, H, 0, 0, W, H);                                 // simulator, full frame
    if (sx < W) ctx.drawImage(v, W + sx, 0, W - sx, H, sx, 0, W - sx, H);       // DataFarm right of the handle
  }
  const setFrom = e => { const r = box.getBoundingClientRect(); split = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)); touched = true; };
  let drag = false;
  box.addEventListener('pointerdown', e => { drag = true; box.setPointerCapture(e.pointerId); setFrom(e); });
  box.addEventListener('pointermove', e => { if (drag) setFrom(e); });
  box.addEventListener('pointerup', () => { drag = false; });
  box.addEventListener('keydown', e => {
    if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') { touched = true; split = Math.min(1, Math.max(0, split + (e.key === 'ArrowLeft' ? -0.05 : 0.05))); e.preventDefault(); }
  });
  tabs('wipe', 'task', t => {
    setSource(v, `static/videos/gen_${t}.mp4`);
    $('#wipe-line').textContent = data.wipe[t].line;
    const sp = $('#wipe-speed'); sp.hidden = !data.wipe[t].speed; sp.textContent = data.wipe[t].speed;
  }, 'cups');
  (function loop() { draw(); requestAnimationFrame(loop); })();
  watchVideo(v);
}

/* ---------------- real robot: trials + chart ---------------- */
function setupTrials(data) {
  tabs('real', 'task', t => {
    const i = data.keys.indexOf(t);
    setSource($('#real-teleop'), `static/videos/real_${t}_teleop.mp4`);
    setSource($('#real-ours'), `static/videos/real_${t}_ours.mp4`);
    $('#real-teleop-score').textContent = `${data.ood.teleop[i]}/20`;
    $('#real-ours-score').textContent = `${data.ood.ours[i]}/20`;
  }, 'cups');
}
function setupChart(data) {
  const host = $('#rr-chart'), note = $('#rr-note');
  if (!host) return;
  const W = 900, H = 330, L = 44, R = 10, T = 28, B = 46;
  const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img' });
  host.appendChild(svg);
  const ph = H - T - B, n = data.tasks.length, gw = (W - L - R) / n, bw = Math.min(30, gw * 0.24);
  [0, 25, 50, 75, 100].forEach(v => {
    const y = T + ph * (1 - v / 100);
    svg.appendChild(svgEl('line', { x1: L, x2: W - R, y1: y, y2: y, stroke: 'rgba(255,255,255,0.08)' }));
    svg.appendChild(svgEl('text', { x: L - 8, y: y + 4, 'text-anchor': 'end', 'font-size': 11, fill: '#5E6A7C' }, v + '%'));
  });
  svg.appendChild(svgEl('rect', { x: L + (n - 1) * gw + 6, y: T - 16, width: gw - 12, height: ph + 16, rx: 8, fill: 'rgba(139,20,20,0.18)' }));
  const fill = { teleop: '#A3ADB8', simdr: '#B39250', ours: '#C8372D' }, ink = { teleop: '#C3CBD5', simdr: '#D2B270', ours: '#FF8A78' };
  const bars = [];
  data.tasks.forEach((t, i) => {
    const cx = L + (i + 0.5) * gw;
    ['teleop', 'simdr', 'ours'].forEach((k, j) => {
      const x = cx + (j - 1.5) * (bw + 4) + 2;
      const r = svgEl('rect', { class: 'bar', x, width: bw, y: T + ph, height: 0, rx: 3, fill: fill[k] });
      const tip = svgEl('title'); r.appendChild(tip);
      const lb = svgEl('text', { x: x + bw / 2, y: T + ph, 'text-anchor': 'middle', 'font-size': 12.5, 'font-weight': 700, fill: ink[k] });
      svg.appendChild(r); svg.appendChild(lb); bars.push({ r, tip, lb, i, k });
    });
    svg.appendChild(svgEl('text', { x: cx, y: H - B + 22, 'text-anchor': 'middle', 'font-size': 13.5, 'font-weight': i === n - 1 ? 800 : 600, fill: '#F3F5F8' }, t));
    if (i === n - 1) svg.appendChild(svgEl('text', { x: cx, y: H - B + 38, 'text-anchor': 'middle', 'font-size': 11, fill: '#9AA5B4' }, '100 trials'));
  });
  function show(cond) {
    const d = data[cond];
    bars.forEach(b => {
      const v = d[b.k][b.i], m = d.n[b.i], h = ph * v / m, y = T + ph - h;
      b.r.setAttribute('y', y); b.r.setAttribute('height', h);
      b.lb.setAttribute('y', y - 6); b.lb.textContent = v;
      b.tip.textContent = `${data.tasks[b.i]}, ${{ ours: 'DataFarm', simdr: 'Sim + DR', teleop: 'teleop data' }[b.k]}: ${v} / ${m}`;
    });
    note.textContent = d.note;
    $$('.seg-btn').forEach(b => b.classList.toggle('active', b.dataset.cond === cond));
  }
  $$('.seg-btn').forEach(b => b.addEventListener('click', () => show(b.dataset.cond)));
  // bars grow when the chart scrolls into view
  if ('IntersectionObserver' in window) {
    const io = new IntersectionObserver(es => { if (es[0].isIntersecting) { show('ood'); io.disconnect(); } }, { threshold: 0.3 });
    io.observe(host);
  } else show('ood');
}

/* ---------------- cost: farm cost per episode for N episodes of one task (band = range over tasks; flat lines = teleoperation estimates) ---------------- */
function setupCost(data) {
  const host = $('#cost-chart'), read = $('#cost-read');
  if (!host) return;
  const c = data.cost, W = 640, H = 380, L = 56, R = 18, T = 20, B = 46, N0 = c.nmin, N1 = c.nmax, K = c.adapt.length;
  const per = (i, n) => c.adapt[i] / n + c.m[i];
  const mean = n => c.adapt.reduce((s, _, i) => s + per(i, n), 0) / K;
  const lo = n => Math.min(...c.adapt.map((_, i) => per(i, n))), hi = n => Math.max(...c.adapt.map((_, i) => per(i, n)));
  const mMean = c.m.reduce((s, v) => s + v, 0) / K, ymax = Math.ceil(hi(N0));
  const X = n => L + (W - L - R) * Math.log10(n / N0) / Math.log10(N1 / N0), Y = v => T + (H - T - B) * (1 - v / ymax);
  const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'Farm cost per episode for N episodes of one task, against a teleoperation labour estimate and a quote' });
  for (let v = 0; v <= ymax; v += 1) {
    svg.appendChild(svgEl('line', { x1: L, x2: W - R, y1: Y(v), y2: Y(v), stroke: 'rgba(255,255,255,0.07)' }));
    svg.appendChild(svgEl('text', { x: L - 8, y: Y(v) + 4, 'text-anchor': 'end', 'font-size': 12, fill: '#7D8899' }, '$' + v));
  }
  [10, 20, 50, 100, 200, 500, 1000].forEach(n => svg.appendChild(svgEl('text', { x: X(n), y: H - B + 18, 'text-anchor': 'middle', 'font-size': 12, fill: '#7D8899' }, n.toLocaleString('en-US'))));
  svg.appendChild(svgEl('text', { x: (L + W - R) / 2, y: H - 8, 'text-anchor': 'middle', 'font-size': 12, fill: '#9AA5B4' }, 'episodes of one task (log scale)'));
  const ns = []; for (let k = 0; k <= 60; k++) ns.push(N0 * Math.pow(N1 / N0, k / 60));
  const pts = (f, arr) => arr.map(n => `${X(n).toFixed(1)},${Y(f(n)).toFixed(1)}`).join(' ');
  svg.appendChild(svgEl('polygon', { points: pts(hi, ns) + ' ' + pts(lo, ns.slice().reverse()), fill: 'rgba(226,106,88,0.16)' }));
  svg.appendChild(svgEl('line', { x1: X(N0), x2: X(N1), y1: Y(mMean), y2: Y(mMean), stroke: '#A3ADB8', 'stroke-dasharray': '6 6', 'stroke-width': 1.6 }));
  svg.appendChild(svgEl('text', { x: X(N1) - 6, y: Y(mMean) + 18, 'text-anchor': 'end', 'font-size': 12.5, 'font-weight': 700, fill: '#C3CBD5' }, `per-episode cost $${mMean.toFixed(2)}`));
  svg.appendChild(svgEl('line', { x1: X(N0), x2: X(N1), y1: Y(c.teleop), y2: Y(c.teleop), stroke: '#E8EDF4', 'stroke-dasharray': '10 4 2 4', 'stroke-width': 1.8 }));
  svg.appendChild(svgEl('text', { x: X(N1) - 6, y: Y(c.teleop) - 8, 'text-anchor': 'end', 'font-size': 12.5, 'font-weight': 700, fill: '#E8EDF4' }, `teleoperation labour, North America*: $${c.teleop.toFixed(2)}`));
  svg.appendChild(svgEl('line', { x1: X(N0), x2: X(N1), y1: Y(c.teleopCn), y2: Y(c.teleopCn), stroke: '#F0B35A', 'stroke-dasharray': '10 4 2 4', 'stroke-width': 1.8 }));
  svg.appendChild(svgEl('text', { x: X(N0) + 6, y: Y(c.teleopCn) - 8, 'font-size': 12.5, 'font-weight': 700, fill: '#F0B35A' }, `teleop. quote, China: $${c.teleopCn.toFixed(2)}`));
  svg.appendChild(svgEl('polyline', { points: pts(mean, ns), fill: 'none', stroke: '#E26A58', 'stroke-width': 3.4, 'stroke-linecap': 'round' }));
  svg.appendChild(svgEl('text', { x: X(12), y: Y(3.5), 'font-size': 12.5, 'font-weight': 700, fill: '#FF8A78' }, 'DataFarm, average per episode'));
  [100, 1000].forEach(n => {
    svg.appendChild(svgEl('line', { x1: X(n), x2: X(n), y1: Y(mean(n)), y2: Y(0), stroke: 'rgba(226,106,88,0.5)', 'stroke-dasharray': '3 4' }));
    svg.appendChild(svgEl('circle', { cx: X(n), cy: Y(mean(n)), r: 6, fill: '#0A0E16', stroke: '#E26A58', 'stroke-width': 2.5 }));
    svg.appendChild(svgEl('text', { x: X(n) + (n === N1 ? -10 : 9), y: n === N1 ? Y(c.teleopCn) - 7 : Y(mean(n)) - 12, 'text-anchor': n === N1 ? 'end' : 'start', 'font-size': 12.5, 'font-weight': 700, fill: '#FF8A78' }, `$${mean(n).toFixed(2)} at ${n.toLocaleString('en-US')}`));
  });
  const cursor = svgEl('line', { y1: T, y2: H - B, stroke: 'rgba(255,255,255,0.35)', opacity: 0 });
  svg.appendChild(cursor);
  host.insertBefore(svg, read);
  svg.addEventListener('pointermove', e => {
    const r = svg.getBoundingClientRect(), px = (e.clientX - r.left) / r.width * W;
    const f = Math.min(1, Math.max(0, (px - L) / (W - L - R))), n = Math.round(N0 * Math.pow(N1 / N0, f));
    cursor.setAttribute('x1', X(n)); cursor.setAttribute('x2', X(n)); cursor.setAttribute('opacity', 1);
    read.innerHTML = `<b>${n.toLocaleString('en-US')} episodes</b><br>$${mean(n).toFixed(2)} per episode on average ($${lo(n).toFixed(2)}&ndash;${hi(n).toFixed(2)}) &middot; $${(mean(n) * n).toFixed(0)} in total`;
    read.style.left = Math.min(r.width - 300, Math.max(8, e.clientX - r.left + 12)) + 'px'; read.style.top = '14px';
    read.classList.add('on');
  });
  svg.addEventListener('pointerleave', () => { read.classList.remove('on'); cursor.setAttribute('opacity', 0); });
}

/* ---------------- small things ---------------- */
function setupReveal() {
  const els = $$('.rv');
  if (!('IntersectionObserver' in window) || reduced) { els.forEach(e => e.classList.add('in')); return; }
  const io = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); } }), { threshold: 0.12 });
  els.forEach(e => io.observe(e));
}
function setupCloseups() {
  const cus = $$('.cu');
  if (!('IntersectionObserver' in window)) { cus.forEach(c => c.classList.add('zoom')); return; }
  const io = new IntersectionObserver(es => es.forEach(e => {
    if (e.isIntersecting) setTimeout(() => e.target.classList.add('zoom'), 700); else e.target.classList.remove('zoom');
  }), { threshold: 0.6 });
  cus.forEach(c => io.observe(c));
}
function setupCounts() {
  $$('[data-count]').forEach(el => {
    const to = +el.dataset.count, pre = el.dataset.prefix || '';
    if (reduced) return;
    const t0 = performance.now(), d = 1100;
    (function step(t) { const p = Math.min(1, (t - t0) / d), e = 1 - Math.pow(1 - p, 3); el.textContent = pre + Math.round(to * e); if (p < 1) requestAnimationFrame(step); })(t0);
  });
}
function setupTyped(data) {
  const el = $('#typed');
  if (!el) return;
  const items = data.appearance; let i = 0, k = 0, back = false;
  (function tick() {
    const s = items[i];
    if (!back) { k++; if (k >= s.length) { back = true; setTimeout(tick, 1800); el.textContent = s; return; } }
    else { k--; if (k <= 0) { back = false; i = (i + 1) % items.length; } }
    el.textContent = s.slice(0, k);
    setTimeout(tick, back ? 25 : 55);
  })();
}
function setupNav() {
  const links = $$('.bar-nav a'), secs = links.map(a => $(a.getAttribute('href'))).filter(Boolean);
  const top = $('#to-top');
  function update() {
    const y = window.scrollY + 140; let cur = null;
    secs.forEach(s => { if (s.offsetTop <= y) cur = s; });
    links.forEach(a => a.classList.toggle('active', !!cur && a.getAttribute('href') === '#' + cur.id));
    top.classList.toggle('on', window.scrollY > 600);
  }
  window.addEventListener('scroll', update, { passive: true }); update();
  top.addEventListener('click', () => window.scrollTo({ top: 0, behavior: reduced ? 'auto' : 'smooth' }));
}
function setupBib() {
  const b = $('#copy-bib');
  b.addEventListener('click', () => {
    const t = $('#bibtex-code').textContent;
    if (navigator.clipboard) navigator.clipboard.writeText(t).then(() => { b.classList.add('copied'); b.querySelector('span').textContent = 'Copied'; setTimeout(() => { b.classList.remove('copied'); b.querySelector('span').textContent = 'Copy'; }, 1600); });
  });
}

document.addEventListener('DOMContentLoaded', () => {
  const data = JSON.parse($('#site-data').textContent);
  setupNav(); setupReveal(); setupCounts();
  setupShowcase(); setupViews(data); setupWipe(data);
  setupTrials(data); setupChart(data); setupCost(data);
  setupCloseups(); setupTyped(data); setupBib();
  // remaining clips play only while on screen
  $$('video').forEach(v => { if (!['show-src', 'views-src', 'wipe-src'].includes(v.id)) watchVideo(v); });
});
