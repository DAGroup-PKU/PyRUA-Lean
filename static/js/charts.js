// Interactive charts of the project page. Data: <script type="application/json" id="site-data">, built from the paper's
// own sources. Every chart has a tooltip on its marks.
(function () {
  'use strict';
  const NS = 'http://www.w3.org/2000/svg';
  const C = { base: '#A3AEB8', third: '#3B6FB6', ours: '#B3261E', grid: '#E6E9ED', axis: '#C9CFD6',
              ink: '#1e293b', ink2: '#475569', muted: '#64748b' };

  function svgEl(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function htmlEl(tag, cls, text, parent) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined && text !== null) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }
  function text(parent, x, y, s, attrs) {
    const t = svgEl('text', Object.assign({ x: x, y: y, fill: C.ink2, 'font-size': 12 }, attrs || {}), parent);
    t.textContent = s;
    return t;
  }
  // a bar with a 4px rounded data end, square at the baseline (horizontal: the end is on the right)
  function hbar(parent, x, y, w, h, fill) {
    const r = Math.min(4, w / 2, h / 2);
    const d = w <= 0 ? '' : `M${x},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + h - r}Q${x + w},${y + h} ${x + w - r},${y + h}H${x}Z`;
    return svgEl('path', { d: d, fill: fill }, parent);
  }
  function vbar(parent, x, base, w, h, fill) {
    const r = Math.min(4, w / 2, h / 2);
    const top = base - h;
    const d = h <= 0 ? '' : `M${x},${base}V${top + r}Q${x},${top} ${x + r},${top}H${x + w - r}Q${x + w},${top} ${x + w},${top + r}V${base}Z`;
    return svgEl('path', { d: d, fill: fill }, parent);
  }

  // ---- one tooltip for the page ---------------------------------------------------------------------------------
  const tip = htmlEl('div', 'chart-tip', null, document.body);
  tip.setAttribute('role', 'status');
  function showTip(evt, rows, title) {
    tip.replaceChildren();
    if (title) htmlEl('div', 'chart-tip-title', title, tip);
    rows.forEach(r => {
      const row = htmlEl('div', 'chart-tip-row', null, tip);
      const key = htmlEl('span', 'chart-tip-key', null, row);
      key.style.background = r.color;
      htmlEl('strong', null, r.value, row);
      htmlEl('span', 'chart-tip-label', r.label, row);
    });
    tip.style.display = 'block';
    moveTip(evt);
  }
  function moveTip(evt) {
    let x, y;
    if (evt && evt.clientX !== undefined && evt.type.startsWith('pointer')) { x = evt.clientX; y = evt.clientY; }
    else if (evt && evt.target && evt.target.getBoundingClientRect) {
      const b = evt.target.getBoundingClientRect(); x = b.right; y = b.top + b.height / 2;
    } else return;
    const w = tip.offsetWidth, h = tip.offsetHeight;
    let left = x + 14, top = y - h - 10;
    if (left + w > window.innerWidth - 8) left = x - w - 14;
    if (top < 8) top = y + 16;
    tip.style.left = left + 'px';
    tip.style.top = top + 'px';
  }
  function hideTip() { tip.style.display = 'none'; }
  // a transparent hit area bigger than the mark, with the same readout on hover and on keyboard focus
  function hit(parent, x, y, w, h, rows, title, mark) {
    const r = svgEl('rect', { x: x, y: y, width: Math.max(w, 1), height: h, fill: 'transparent', tabindex: 0,
                               class: 'chart-hit' }, parent);
    r.setAttribute('aria-label', (title ? title + ': ' : '') + rows.map(o => o.label + ' ' + o.value).join(', '));
    const on = e => { showTip(e, rows, title); if (mark) mark.forEach(m => m.classList.add('is-hover')); };
    const off = () => { hideTip(); if (mark) mark.forEach(m => m.classList.remove('is-hover')); };
    r.addEventListener('pointerenter', on);
    r.addEventListener('pointermove', moveTip);
    r.addEventListener('pointerleave', off);
    r.addEventListener('focus', on);
    r.addEventListener('blur', off);
    return r;
  }

  // ---- controls ----------------------------------------------------------------------------------------------------
  function segmented(parent, label, options, value, onChange) {
    const g = htmlEl('div', 'seg', null, parent);
    g.setAttribute('role', 'group');
    g.setAttribute('aria-label', label);
    const btns = options.map(o => {
      const b = htmlEl('button', 'seg-btn', o.label, g);
      b.type = 'button';
      b.setAttribute('aria-pressed', String(o.value === value));
      b.addEventListener('click', () => {
        btns.forEach(x => x.setAttribute('aria-pressed', String(x === b)));
        onChange(o.value);
      });
      return b;
    });
    return g;
  }
  function legend(parent, items, onToggle) {
    const l = htmlEl('div', 'chart-legend', null, parent);
    items.forEach(it => {
      const tag = onToggle ? 'button' : 'span';
      const e = htmlEl(tag, 'legend-item' + (onToggle ? ' is-toggle' : ''), null, l);
      if (onToggle) { e.type = 'button'; e.setAttribute('aria-pressed', 'true'); }
      const sw = htmlEl('span', 'legend-swatch' + (it.line ? ' is-line' : ''), null, e);
      sw.style.background = it.color;
      htmlEl('span', null, it.label, e);
      if (onToggle) e.addEventListener('click', () => {
        const on = e.getAttribute('aria-pressed') !== 'true';
        e.setAttribute('aria-pressed', String(on));
        onToggle(it.key, on);
      });
    });
    return l;
  }
  function onResize(fn) {
    let w = window.innerWidth, t = null;
    window.addEventListener('resize', () => {
      if (Math.abs(window.innerWidth - w) < 8) return;
      w = window.innerWidth;
      clearTimeout(t); t = setTimeout(fn, 120);
    });
  }
  function niceMax(v) {
    const p = Math.pow(10, Math.floor(Math.log10(v)));
    for (const m of [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]) if (m * p >= v) return m * p;
    return 10 * p;
  }
  const fmtTokens = v => v >= 1e6 ? (v / 1e6).toFixed(2).replace(/\.?0+$/, '') + 'M' : Math.round(v / 1e3) + 'k';


  // ---- columns in the manner of a results panel: baseline against ours per group, a gain pill above, tabs of views,
  // bars that grow when the panel comes into view, and a group that lifts while the rest fade under the pointer --------
  const REDUCED = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  function columnsPanel(root, views, opts) {
    opts = opts || {};
    let vi = 0, seen = false;
    const top = htmlEl('div', 'chart-controls', null, root);
    const tabs = htmlEl('div', 'rx-tabs', null, top);
    if (views.length < 2) top.style.display = 'none';
    tabs.setAttribute('role', 'tablist');
    const legendEl = htmlEl('div', 'rx-legend', null, root);
    const frame = htmlEl('div', 'rx', null, root);
    const note = htmlEl('p', 'chart-note', null, root);
    views.forEach((v, i) => {
      const b = htmlEl('button', 'rx-tab', v.name, tabs);
      b.type = 'button';
      b.setAttribute('role', 'tab');
      b.addEventListener('click', () => { vi = i; render(); });
    });
    function render() {
      const v = views[vi];
      Array.prototype.forEach.call(tabs.children, (b, j) => { b.classList.toggle('on', j === vi); b.setAttribute('aria-selected', String(j === vi)); });
      legendEl.replaceChildren();
      [[v.baseLabel || 'Baseline', 'base'], [v.oursLabel || 'PyRUA-Lean', 'ours']].forEach(([t, c]) => {
        const sp = htmlEl('span', 'rx-key ' + c, null, legendEl);
        htmlEl('i', null, null, sp);
        htmlEl('span', null, t, sp);
      });
      note.textContent = v.note || '';
      frame.replaceChildren();
      hideTip();
      const n = v.groups.length, dense = n > 8;
      frame.classList.toggle('dense', dense);
      const scroll = htmlEl('div', 'rx-scroll', null, frame);
      const inner = htmlEl('div', 'rx-inner', null, scroll);
      if (dense) inner.style.minWidth = (n * 46) + 'px';
      const plot = htmlEl('div', 'rx-plot', null, inner);
      const xs = htmlEl('div', 'rx-x', null, inner);
      plot.style.gridTemplateColumns = xs.style.gridTemplateColumns = 'repeat(' + n + ', 1fr)';
      const max = v.max || 100, bars = [];
      v.groups.forEach((g, i) => {
        const grp = htmlEl('div', 'rx-g', null, plot);
        grp.tabIndex = 0;
        [[g.base, 'base'], [g.ours, 'ours']].forEach(([val, cls], k) => {
          const b = htmlEl('div', 'rx-b ' + cls, null, grp);
          if (val === null || val === undefined) { b.style.visibility = 'hidden'; return; }
          const lab = htmlEl('b', null, (v.fmt || fmtPct)(val), b);
          if (dense && cls === 'base') lab.classList.add('quiet');
          b.style.transitionDelay = (REDUCED ? 0 : Math.min(0.6, i * 0.04) + k * 0.12) + 's';
          bars.push([b, Math.max(0, Math.min(1, val / max))]);
        });
        if (v.pills !== false && g.base !== null && g.base !== undefined) {
          const d = htmlEl('div', 'rx-d', g.pill || (v.gain || gainPp)(g), grp);
          d.style.setProperty('--v', Math.max(g.ours, g.base) / max);
          if (dense) d.classList.add('small');
        }
        const x = htmlEl('span', null, null, xs);
        htmlEl('strong', null, g.label, x);
        if (g.sub && !dense) htmlEl('small', null, g.sub, x);
        const rows = [{ color: C.ours, value: (v.fmt || fmtPct)(g.ours), label: v.oursLabel || 'PyRUA-Lean' }];
        if (g.base !== null && g.base !== undefined) rows.push({ color: C.base, value: (v.fmt || fmtPct)(g.base), label: g.who || v.baseLabel || 'Baseline' });
        (g.extra || []).forEach(e => rows.push(e));
        const on = e => { plot.classList.add('hovering'); grp.classList.add('hot'); showTip(e, rows, g.label + (g.sub ? ' · ' + g.sub : '')); };
        const off = () => { plot.classList.remove('hovering'); grp.classList.remove('hot'); hideTip(); };
        grp.addEventListener('pointerenter', on);
        grp.addEventListener('pointermove', moveTip);
        grp.addEventListener('pointerleave', off);
        grp.addEventListener('focus', on);
        grp.addEventListener('blur', off);
      });
      const grow = () => { frame.classList.add('in'); bars.forEach(([b, f]) => { b.style.height = 'calc(var(--plot-h) * ' + f + ')'; }); };
      const visible = () => { const r = frame.getBoundingClientRect(); return r.top < window.innerHeight && r.bottom > 0; };
      if (REDUCED || seen || !('IntersectionObserver' in window) || visible()) { seen = true; requestAnimationFrame(() => requestAnimationFrame(grow)); }
      else {
        const io = new IntersectionObserver(es => { if (es.some(e => e.isIntersecting)) { seen = true; io.disconnect(); grow(); } }, { threshold: 0.2 });
        io.observe(frame);
        // a browser that never reports the intersection (some headless or embedded views) still gets its bars
        setTimeout(() => { if (!seen && visible()) { seen = true; io.disconnect(); grow(); } }, 1500);
      }
    }
    render();
  }
  const fmtPct = v => (v >= 99.95 ? '100' : v.toFixed(1));
  const gainPp = g => (g.ours - g.base >= 0 ? '+' : '') + (g.ours - g.base).toFixed(1);


  // ---- Figure 1 of the paper, drawn in the page: (a) a radar of success per sub-suite, (b, c) tokens and dollars per
  // solved episode, (d) RoboDojo per dimension and the head of the board, (e) the Coder on unseen tasks -----------------
  const SERIF = "'Times New Roman', Times, 'Liberation Serif', serif";
  const F = { tool: '#8C9AA6', toolFill: '#B9C2CB', red: '#94070A', redSoft: 'rgba(148, 7, 10, 0.10)', tint: '#FBF2F3', grid: '#E3E7EB', ink: '#101A2B', muted: '#7A8794' };
  function ftext(parent, x, y, s, attrs) {
    return text(parent, x, y, s, Object.assign({ 'font-family': SERIF, fill: F.ink, 'font-size': 15 }, attrs || {}));
  }
  function panel(parent, title, cls) {
    const p = htmlEl('div', 'f1-panel ' + (cls || ''), null, parent);
    htmlEl('div', 'f1-title', title, p);
    return p;
  }
  function phead(parent, label, arrow, claim) {
    const h = htmlEl('div', 'f1-head', null, parent);
    htmlEl('span', 'f1-label', label + ' ' + arrow, h);
    htmlEl('span', 'f1-claim', claim, h);
    return h;
  }
  // a radar: axes clockwise from the top, a dashed hollow baseline and a filled red line for ours
  function radar(parent, cfg) {
    const W = 560, H = cfg.height || 470, n = cfg.axes.length, R = cfg.radius || 150;
    const svg = cfg.svg || svgEl('svg', { viewBox: `0 0 ${W} ${H}`, class: 'f1-svg', role: 'img', 'aria-label': cfg.label }, parent);
    const cx = cfg.cx !== undefined ? cfg.cx : W / 2, cy = cfg.cy !== undefined ? cfg.cy : H / 2 + (cfg.dy || 0);
    const fs = cfg.fontScale || 1;
    const ang = i => -Math.PI / 2 + (2 * Math.PI * i) / n;
    const pt = (i, v) => [cx + Math.cos(ang(i)) * R * (v / cfg.max), cy + Math.sin(ang(i)) * R * (v / cfg.max)];
    cfg.rings.forEach(v => svgEl('circle', { cx: cx, cy: cy, r: R * v / cfg.max, fill: 'none', stroke: F.grid, 'stroke-width': 1 }, svg));
    cfg.rings.forEach(v => ftext(svg, cx + 4, cy - R * v / cfg.max + 13, String(v), { 'font-size': 11, fill: F.muted }));
    cfg.axes.forEach((a, i) => { const [x, y] = pt(i, cfg.max); svgEl('line', { x1: cx, y1: cy, x2: x, y2: y, stroke: F.grid, 'stroke-width': 1 }, svg); });
    (cfg.groups || []).forEach(g => {           // the benchmarks, as arcs outside the labels
      const r = R + (cfg.arcGap || 86), a0 = ang(g.from) - Math.PI / n * 0.8, a1 = ang(g.to) + Math.PI / n * 0.8;
      const p0 = [cx + Math.cos(a0) * r, cy + Math.sin(a0) * r], p1 = [cx + Math.cos(a1) * r, cy + Math.sin(a1) * r];
      svgEl('path', { d: `M${p0[0]},${p0[1]}A${r},${r} 0 ${a1 - a0 > Math.PI ? 1 : 0} 1 ${p1[0]},${p1[1]}`, fill: 'none', stroke: '#C9CFD6', 'stroke-width': 1.6 }, svg);
      const am = (a0 + a1) / 2, rr = r + 16;
      ftext(svg, cx + Math.cos(am) * rr, cy + Math.sin(am) * rr + 5, g.name, { 'text-anchor': 'middle', fill: F.muted, 'font-size': 16 });
    });
    const poly = (vals, attrs) => svgEl('polygon', Object.assign({ points: vals.map((v, i) => pt(i, v).join(',')).join(' ') }, attrs), svg);
    poly(cfg.axes.map(a => a.ours), { fill: F.redSoft, stroke: 'none' });
    poly(cfg.axes.map(a => a.base), { fill: 'none', stroke: F.tool, 'stroke-width': 2, 'stroke-dasharray': '6 4' });
    poly(cfg.axes.map(a => a.ours), { fill: 'none', stroke: F.red, 'stroke-width': 2.5, 'stroke-linejoin': 'round' });
    cfg.axes.forEach((a, i) => {
      const [bx, by] = pt(i, a.base), [ox, oy] = pt(i, a.ours);
      svgEl('circle', { cx: bx, cy: by, r: 5, fill: '#fff', stroke: F.tool, 'stroke-width': 2 }, svg);
      svgEl('circle', { cx: ox, cy: oy, r: 5.5, fill: F.red, stroke: '#fff', 'stroke-width': 1.5 }, svg);
      // the label: the axis, what it tests, and the gain
      const [lx, ly] = pt(i, cfg.max * (1 + (cfg.labelGap || 0.2)));
      const anchor = Math.abs(lx - cx) < 12 ? 'middle' : lx > cx ? 'start' : 'end';
      const top = ly < cy - R * 0.9 ? -24 : ly > cy + R * 0.9 ? 6 : -10;
      ftext(svg, lx, ly + top, a.axis, { 'text-anchor': anchor, 'font-weight': 700, 'font-size': 16 * fs });
      if (a.tag) ftext(svg, lx, ly + top + 16 * fs, a.tag, { 'text-anchor': anchor, 'font-style': 'italic', fill: F.muted, 'font-size': 13 * fs });
      ftext(svg, lx, ly + top + (a.tag ? 32 : 16) * fs, a.gain + ' pp', { 'text-anchor': anchor, fill: F.red, 'font-size': 13.5 * fs });
      const hx = (bx + ox) / 2, hy = (by + oy) / 2;
      hit(svg, Math.min(bx, ox) - 14, Math.min(by, oy) - 14, Math.abs(bx - ox) + 28, Math.abs(by - oy) + 28,
          [{ color: F.red, value: a.ours.toFixed(1) + '%', label: cfg.oursLabel }, { color: F.tool, value: a.base.toFixed(1) + '%', label: a.who || cfg.baseLabel }],
          a.axis + (a.note ? ' · ' + a.note : ''));
      void hx; void hy;
    });
    return svg;
  }
  // paired horizontal bars, the baseline above ours, with the ratio on the right and the mean in a tinted row
  function pairBars(parent, rows, fmt, label) {
    const W = 560, rowH = 50, top = 22, H = top + rows.length * rowH + 4, x0 = 150, rx = W - 46, plotW = rx - x0 - 70;
    const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, class: 'f1-svg', role: 'img', 'aria-label': label }, parent);
    ftext(svg, W - 4, 14, 'Ratio', { 'text-anchor': 'end', 'font-size': 14, fill: F.ink });
    const max = Math.max(...rows.map(r => r.base));
    rows.forEach((r, i) => {
      const y = top + i * rowH;
      if (r.avg) {
        svgEl('rect', { x: 0, y: y - 4, width: W, height: rowH - 2, fill: F.tint }, svg);
        svgEl('line', { x1: 0, x2: W, y1: y - 6, y2: y - 6, stroke: '#D5DAE0', 'stroke-width': 1 }, svg);
      }
      ftext(svg, x0 - 10, y + 24, r.label, { 'text-anchor': 'end', 'font-size': 15.5, 'font-weight': r.avg ? 700 : 400 });
      const wb = plotW * r.base / max, wo = plotW * r.ours / max;
      svgEl('rect', { x: x0, y: y + 4, width: wb, height: 16, fill: F.toolFill }, svg);
      svgEl('rect', { x: x0, y: y + 22, width: wo, height: 16, fill: F.red }, svg);
      ftext(svg, x0 + wb + 6, y + 17, fmt(r.base), { 'font-size': 13, fill: F.muted });
      ftext(svg, x0 + wo + 6, y + 35, fmt(r.ours), { 'font-size': 13.5, fill: F.red, 'font-weight': 700 });
      ftext(svg, W - 4, y + 27, r.ratio + '×', { 'text-anchor': 'end', 'font-size': 16, fill: F.red });
      svgEl('line', { x1: x0, x2: x0, y1: y, y2: y + 42, stroke: '#9AA5B1', 'stroke-width': 1 }, svg);
      hit(svg, 0, y - 2, W, rowH - 4, [{ color: F.red, value: fmt(r.ours), label: 'PyRUA-Lean' }, { color: F.tool, value: fmt(r.base), label: 'Tool calling (RPent)' }],
          r.label + ' · ' + r.ratio + '×');
    });
    return svg;
  }
  function figureOne(root, data) {
    const I = data.interface, by = b => I.find(r => r.bench === b);
    // top: the interface against tool calling
    const top = panel(root, 'Fewer tokens, better action', 'f1-top');
    const grid = htmlEl('div', 'f1-top-grid', null, top);
    const a = htmlEl('div', 'f1-cell', null, grid), bc = htmlEl('div', 'f1-cell', null, grid);
    const all = by('All');
    phead(a, '(a) Success rate', '↑', all.success[0].toFixed(1) + '% → ' + all.success[1].toFixed(1) + '% overall');
    radar(a, { label: 'Success rate per sub-suite, tool calling against PyRUA-Lean', max: 100, rings: [25, 50, 75, 100], radius: 145, height: 480, dy: 6,
               oursLabel: 'PyRUA-Lean', baseLabel: 'Tool calling (RPent)',
               axes: data.radar.map(r => ({ axis: r.axis, tag: r.tag, gain: r.gain, base: r.tool, ours: r.code, note: r.bench + ', ' + r.n + ' instances' })),
               groups: [{ name: 'LIBERO-PRO', from: 0, to: 3 }, { name: 'RoboTwin', from: 4, to: 6 }, { name: 'RC365', from: 7, to: 8 }] });
    const lg = htmlEl('div', 'f1-legend', null, a);
    [['Tool calling (RPent)', 'base'], ['Code (PyRUA-Lean, ours)', 'ours']].forEach(([t, c]) => { const e = htmlEl('span', 'f1-key ' + c, null, lg); htmlEl('i', null, null, e); htmlEl('span', null, t, e); });
    const order = ['LIBERO-PRO', 'RoboCasa365 composite', 'RoboTwin 2.0', 'RoboCasa365 atomic'];
    const short = b => b.replace('RoboCasa365', 'RC365');
    phead(bc, '(b) Token usage', '↓', Math.round(100 * (1 - all.tokens[1] / all.tokens[0])) + '% fewer tokens on average');
    pairBars(bc, order.map(b => ({ label: short(b), base: by(b).tokens[0], ours: by(b).tokens[1], ratio: by(b).ratio.tokens }))
      .concat([{ label: 'Avg', base: all.tokens[0], ours: all.tokens[1], ratio: all.ratio.tokens, avg: true }]), fmtTokens, 'Input tokens per solved episode');
    phead(bc, '(c) Cost', '↓', all.ratio.cost + '× cheaper on average');
    pairBars(bc, order.map(b => ({ label: short(b), base: by(b).cost[0], ours: by(b).cost[1], ratio: by(b).ratio.cost }))
      .concat([{ label: 'Avg', base: all.cost[0], ours: all.cost[1], ratio: all.ratio.cost, avg: true }]), v => '$' + v.toFixed(2), 'Dollars per solved episode');
    htmlEl('p', 'f1-foot', 'Costs at list prices. pp: percentage points. Tokens, calls and costs on the instances both agents solved.', top);
    // the name
    const band = htmlEl('div', 'f1-band', null, root);
    const img = htmlEl('img', null, null, band); img.src = 'static/images/logo.svg'; img.alt = 'PyRUA-Lean: Python for Lean Robot-Use Agents';
    // bottom left: RoboDojo
    const bottom = htmlEl('div', 'f1-bottom', null, root);
    const d = panel(bottom, 'State of the art on RoboDojo', 'f1-d');
    const B = data.board.best, ours = data.board.entries[0];
    const nTasks = dim => new Set(data.tasks.filter(t => t.dim === dim).map(t => t.task.replace(/_random$/, ''))).size;
    const nEntries = data.board.entries.length - 1;
    phead(d, '(d) Success rate', '↑', ours.average.sr.toFixed(1) + '% average, above all ' + nEntries + ' board entries');
    const DIMS = [['generalization', 'Generalization'], ['precision', 'Precision'], ['longHorizon', 'Long-Horizon'], ['memory', 'Memory'], ['open', 'Open']];
    const who = n => n === 'GPT-6-Astra' ? 'GPT-6 Astra, direct control' : n;
    const head4 = data.board.entries.slice(1).sort((p, q) => q.average.sr - p.average.sr).slice(0, 4);
    {
      const W = 640, H = 360;
      const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, class: 'f1-svg', role: 'img', 'aria-label': 'RoboDojo: success rate per dimension against the best board entry in it, and the average against the head of the board' }, d);
      radar(null, { svg: svg, cx: 150, cy: 188, radius: 92, max: 80, rings: [40, 80], labelGap: 0.36, fontScale: 0.95,
                    oursLabel: 'PyRUA-Lean (ours)', baseLabel: 'Best board entry',
                    axes: DIMS.map(([k, l]) => ({ axis: l, tag: nTasks(l) + ' tasks', gain: (B[k].ours - B[k].sr >= 0 ? '+' : '') + (B[k].ours - B[k].sr).toFixed(1), base: B[k].sr, ours: B[k].ours, who: 'Best board entry: ' + who(B[k].name) })) });
      const x0 = 452, plotW = 120, max = ours.average.sr;
      const rows = [{ name: 'PyRUA-Lean', v: ours.average.sr, ours: true }].concat(head4.map((e, i) => ({ name: e.name, v: e.average.sr, rank: i + 1 })));
      rows.forEach((r, i) => {
        const y = 92 + i * 46 + (r.ours ? 0 : 18), h = r.ours ? 32 : 22, w = plotW * r.v / max;
        if (r.rank) ftext(svg, 338, y + h - 5, String(r.rank), { 'font-style': 'italic', fill: F.muted, 'font-size': 15 });
        ftext(svg, x0 - 10, y + h - (r.ours ? 9 : 5), r.name, { 'text-anchor': 'end', 'font-size': r.ours ? 18 : 16, fill: r.ours ? F.red : F.ink, 'font-weight': r.ours ? 700 : 400 });
        svgEl('rect', { x: x0, y: y, width: w, height: h, fill: r.ours ? F.red : F.toolFill }, svg);
        ftext(svg, x0 + w + 7, y + h - (r.ours ? 7 : 5), r.v.toFixed(2), { 'font-size': r.ours ? 24 : 15, fill: r.ours ? F.red : F.muted, 'font-weight': r.ours ? 700 : 400 });
        hit(svg, 330, y - 6, W - 330, h + 12, [{ color: r.ours ? F.red : F.tool, value: r.v.toFixed(2) + '%', label: 'average success rate' }], r.ours ? 'PyRUA-Lean (ours, seed 0)' : '#' + r.rank + ' ' + r.name);
      });
      const xa = x0 + plotW * head4[0].average.sr / max, xb = x0 + plotW;
      svgEl('path', { d: `M${xa},72V80M${xa},76H${xb}M${xb},72V80`, stroke: F.red, 'stroke-width': 1.6, fill: 'none' }, svg);
      ftext(svg, (xa + xb) / 2, 64, '+' + (ours.average.sr - head4[0].average.sr).toFixed(1) + ' pp', { 'text-anchor': 'middle', fill: F.red, 'font-weight': 700, 'font-size': 17 });
      ftext(svg, 452 + 60, 336, 'average SR (%), board snapshot of ' + (([y, m, dd]) => ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'][m - 1] + ' ' + dd + ', ' + y)(data.board.stamp.split('.').map(Number)), { 'text-anchor': 'middle', 'font-style': 'italic', fill: F.muted, 'font-size': 13 });
    }
    const lgd = htmlEl('div', 'f1-legend', null, d);
    [['Best board entry', 'base'], ['PyRUA-Lean (ours)', 'ours']].forEach(([t, c]) => { const e = htmlEl('span', 'f1-key ' + c, null, lgd); htmlEl('i', null, null, e); htmlEl('span', null, t, e); });
    // bottom right: the Coder on what it never saw
    const e = panel(bottom, 'PyRUA-Lean Coder v0.1-4B', 'f1-e');
    const cod = data.coder, base = cod.models[0].values, mine = cod.models[2].values;
    phead(e, '(e) Generalization to unseen tasks', '↑', base[5].toFixed(1) + '% → ' + mine[5].toFixed(1) + '% on unseen instances');
    {
      const groups = [['LIBERO-PRO', 'unseen tasks', 1], ['RoboTwin 2.0', 'unseen benchmark', 2], ['RC365 atomic', 'unseen benchmark', 3],
                      ['RC365 composite', 'unseen benchmark', 4], ['All unseen', '655 instances', 5]];
      const W = 640, H = 330, left = 46, base0 = 262, plotH = 210, gw = (W - left - 6) / groups.length;
      const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, class: 'f1-svg', role: 'img', 'aria-label': 'PyRUA-Lean Coder v0.1-4B against Qwen3.5-4B on unseen tasks' }, e);
      svgEl('rect', { x: left + 4 * gw, y: base0 - plotH - 10, width: gw, height: plotH + 64, fill: F.tint }, svg);
      [0, 50, 100].forEach(t => {
        const y = base0 - plotH * t / 100;
        svgEl('line', { x1: left, x2: W - 6, y1: y, y2: y, stroke: t ? F.grid : '#9AA5B1', 'stroke-width': 1 }, svg);
        ftext(svg, left - 8, y + 4, t === 100 ? '100%' : String(t), { 'text-anchor': 'end', 'font-size': 12, fill: F.muted });
      });
      groups.forEach(([name, sub, k], gi) => {
        const gx = left + gi * gw, bw = Math.min(30, gw / 3.4);
        if (gi) svgEl('line', { x1: gx, x2: gx, y1: base0 - plotH - 10, y2: base0 + 54, stroke: F.grid, 'stroke-width': 1 }, svg);
        const xb = gx + gw / 2 - bw - 2, xo = gx + gw / 2 + 2, hb = plotH * base[k] / 100, ho = plotH * mine[k] / 100;
        svgEl('rect', { x: xb, y: base0 - hb, width: bw, height: hb, fill: F.toolFill }, svg);
        svgEl('rect', { x: xo, y: base0 - ho, width: bw, height: ho, fill: F.red }, svg);
        ftext(svg, xb + bw / 2, base0 - hb - 6, base[k].toFixed(1), { 'text-anchor': 'middle', 'font-size': 12.5, fill: F.muted });
        ftext(svg, xo + bw / 2, base0 - ho - 6, mine[k].toFixed(1), { 'text-anchor': 'middle', 'font-size': 14, fill: F.red, 'font-weight': 700 });
        ftext(svg, gx + gw / 2, base0 + 22, name, { 'text-anchor': 'middle', 'font-size': 14.5, 'font-weight': k === 5 ? 700 : 400 });
        ftext(svg, gx + gw / 2, base0 + 40, sub, { 'text-anchor': 'middle', 'font-size': 12.5, 'font-style': 'italic', fill: F.muted });
        hit(svg, gx, base0 - plotH - 10, gw, plotH + 64, [{ color: F.red, value: mine[k].toFixed(1) + '%', label: 'PyRUA-Lean Coder v0.1-4B' },
          { color: F.tool, value: base[k].toFixed(1) + '%', label: 'Qwen3.5-4B (base)' }, { color: C.third, value: cod.models[1].values[k].toFixed(1) + '%', label: 'Qwen3.5-27B (base)' }],
          name + ' · ' + sub);
      });
    }
    const lge = htmlEl('div', 'f1-legend', null, e);
    [['Qwen3.5-4B (base)', 'base sq'], ['PyRUA-Lean Coder v0.1-4B (after GRPO)', 'ours sq']].forEach(([t, c]) => { const x = htmlEl('span', 'f1-key ' + c, null, lge); htmlEl('i', null, null, x); htmlEl('span', null, t, x); });
  }

  // ---- the RoboDojo board: every entry, one dimension at a time --------------------------------------------------
  function boardChart(root, board) {
    const dims = [['average', 'Average'], ['generalization', 'Generalization'], ['precision', 'Precision'],
                  ['longHorizon', 'Long-Horizon'], ['memory', 'Memory'], ['open', 'Open']];
    let dim = 'average', metric = 'sr';
    const controls = htmlEl('div', 'chart-controls', null, root);
    segmented(controls, 'Dimension', dims.map(([v, l]) => ({ value: v, label: l })), dim, v => { dim = v; draw(); });
    segmented(controls, 'Metric', [{ value: 'sr', label: 'SR' }, { value: 'score', label: 'Score' }], metric,
              v => { metric = v; draw(); });
    legend(root, [{ label: 'PyRUA-Lean (ours, GPT-6 Astra)', color: C.ours },
                  { label: 'GPT-6 Astra, direct control', color: C.third },
                  { label: 'Other board entries', color: C.base }]);
    const body = htmlEl('div', 'chart-body', null, root);
    htmlEl('p', 'chart-note', `A snapshot of RoboDojo's public leaderboard taken on ${(([y, m, dd]) => ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'][m - 1] + ' ' + dd + ', ' + y)(board.stamp.split('.').map(Number))} (33 entries), and PyRUA-Lean on seed 0. SR: success rate (%). The board's average is the mean of its five dimensions.`, root);
    const colorOf = e => e.ours ? C.ours : e.name === 'GPT-6-Astra' ? C.third : C.base;
    const nameOf = e => e.name === 'GPT-6-Astra' ? 'GPT-6 Astra, direct control' : e.name;

    function draw() {
      body.replaceChildren();
      const label = dims.find(d => d[0] === dim)[1];
      const sorted = board.entries.slice().sort((p, q) => q[dim][metric] - p[dim][metric]);
      const W = Math.max(320, body.clientWidth), labelW = W < 560 ? 150 : 210, rightW = 56;
      const barH = 12, rowH = 18, top = 6, H = top + sorted.length * rowH + 22;
      const svg = svgEl('svg', { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: 'img',
                                  'aria-label': `RoboDojo board, ${label} ${metric === 'sr' ? 'success rate' : 'score'}` }, body);
      const x0 = labelW, plotW = W - labelW - rightW, max = 100;
      const sx = v => (v / max) * plotW;
      [0, 25, 50, 75, 100].forEach(t => {
        svgEl('line', { x1: x0 + sx(t), x2: x0 + sx(t), y1: top - 2, y2: H - 20, stroke: t ? C.grid : C.axis, 'stroke-width': 1 }, svg);
        text(svg, x0 + sx(t), H - 6, String(t), { 'text-anchor': 'middle', 'font-size': 11, fill: C.muted });
      });
      sorted.forEach((e, i) => {
        const y = top + i * rowH, v = e[dim][metric], special = e.ours || e.name === 'GPT-6-Astra';
        text(svg, 4, y + barH - 1, String(i + 1), { 'font-size': 11, fill: C.muted });
        text(svg, x0 - 8, y + barH - 1, nameOf(e), { 'text-anchor': 'end', 'font-size': 12, fill: special ? C.ink : C.ink2,
                                                     'font-weight': e.ours ? 700 : special ? 600 : 400 });
        const b = hbar(svg, x0, y, sx(v), barH, colorOf(e));
        b.classList.add('mark');
        if (special || i === 0 || (i === 1 && sorted[0].ours)) text(svg, x0 + sx(v) + 5, y + barH - 1, v.toFixed(2),
            { 'font-size': 11, fill: e.ours ? C.ink : C.ink2, 'font-weight': e.ours ? 700 : 400 });
        hit(svg, 0, y - 3, W, rowH, [{ color: colorOf(e), value: v.toFixed(2), label: (metric === 'sr' ? 'SR' : 'Score') + ', ' + label }],
            `#${i + 1} ${nameOf(e)}${e.team ? ' · ' + e.team : ''}`, [b]);
      });
    }
    draw();
    onResize(draw);
  }

  document.addEventListener('DOMContentLoaded', function () {
    const el = document.getElementById('site-data');
    if (!el) return;
    const data = JSON.parse(el.textContent);
    const mount = (id, fn, d) => { const r = document.getElementById(id); if (r) fn(r, d); };
    mount('figure-one', figureOne, data);
    const CT = document.getElementById('chart-cost');
    if (CT) {
      const I = data.interface;
      const fewer = g => (g.base / g.ours).toFixed(1) + '× fewer';
      columnsPanel(CT, [
        { name: 'LLM calls', baseLabel: 'Tool calling (RPent)', fmt: v => v.toFixed(1), gain: fewer, max: niceMax(Math.max(...I.map(r => r.calls[0]))),
          note: 'LLM calls per solved episode, on the task instances both agents solved.',
          groups: I.map(r => ({ label: r.bench.replace('RoboCasa365', 'RC365'), base: r.calls[0], ours: r.calls[1], pill: r.ratio.calls + '× fewer', who: 'Tool calling (RPent)' })) },
        { name: 'Input tokens', baseLabel: 'Tool calling (RPent)', fmt: fmtTokens, gain: fewer, max: niceMax(Math.max(...I.map(r => r.tokens[0]))),
          note: 'Input tokens per solved episode, on the task instances both agents solved.',
          groups: I.map(r => ({ label: r.bench.replace('RoboCasa365', 'RC365'), base: r.tokens[0], ours: r.tokens[1], pill: r.ratio.tokens + '× fewer', who: 'Tool calling (RPent)' })) },
        { name: 'Cost', baseLabel: 'Tool calling (RPent)', fmt: v => '$' + v.toFixed(2), gain: g => (g.base / g.ours).toFixed(1) + '× cheaper', max: niceMax(Math.max(...I.map(r => r.cost[0]))),
          note: 'GPT-6 Astra list-price dollars per solved episode, on the task instances both agents solved.',
          groups: I.map(r => ({ label: r.bench.replace('RoboCasa365', 'RC365'), base: r.cost[0], ours: r.cost[1], pill: r.ratio.cost + '× cheaper', who: 'Tool calling (RPent)' })) },
      ]);
    }
    mount('chart-board', boardChart, data.board);
    document.addEventListener('scroll', hideTip, { passive: true });
  });
})();
