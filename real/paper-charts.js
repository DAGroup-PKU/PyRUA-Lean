// Reveal only the original PDF data layers; axes and labels remain stationary.
export function mountPaperCharts(root) {
  if (!root) return () => {};
  const abort = new AbortController();
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const records = [...root.querySelectorAll('[data-chart-src]')].map(figure => ({figure, rects: [], inView: false, elapsed: 0, frame: null, last: null}));
  const drawDuration = 6800, holdDuration = 3600, duration = drawDuration + holdDuration;
  let destroyed = false;

  function paint(record, fraction) {
    for (const {rect, width} of record.rects) rect.setAttribute('width', String(Math.max(2, width * fraction)));
    record.figure.dataset.progress = fraction.toFixed(3);
  }
  function eligible(record) {
    return !destroyed && record.rects.length > 0 && record.inView && !document.hidden && !reduced.matches;
  }
  function stop(record) {
    cancelAnimationFrame(record.frame);
    record.frame = null;
    record.last = null;
  }
  function tick(record, now) {
    record.frame = null;
    if (!eligible(record)) return;
    const delta = record.last === null ? 0 : Math.max(0, Math.min(now - record.last, 100));
    record.last = now;
    record.elapsed = (record.elapsed + delta) % duration;
    paint(record, Math.min(1, record.elapsed / drawDuration));
    record.frame = requestAnimationFrame(time => tick(record, time));
  }
  function sync(record) {
    if (!eligible(record)) {
      stop(record);
      if (reduced.matches) paint(record, 1);
      return;
    }
    if (record.frame === null) {
      record.last = null;
      record.frame = requestAnimationFrame(time => tick(record, time));
    }
  }
  const observer = new IntersectionObserver(entries => {
    for (const entry of entries) {
      const record = records.find(item => item.figure === entry.target);
      record.inView = entry.isIntersecting && entry.intersectionRatio >= .1;
      sync(record);
    }
  }, {threshold: [0, .1]});
  records.forEach(record => observer.observe(record.figure));

  records.forEach(async record => {
    try {
      const response = await fetch(record.figure.dataset.chartSrc, {signal: abort.signal});
      if (!response.ok) throw new Error('Figure unavailable');
      const document = new DOMParser().parseFromString(await response.text(), 'image/svg+xml');
      const svg = document.documentElement;
      if (destroyed || svg.localName !== 'svg' || document.querySelector('parsererror')) return;
      const rects = [...svg.querySelectorAll('[data-reveal-rect]')].map(rect => ({rect, width: Number(rect.dataset.fullWidth || rect.getAttribute('width'))}));
      if (!rects.length || rects.some(item => !Number.isFinite(item.width) || item.width <= 0)) return;
      svg.setAttribute('role', 'img');
      svg.setAttribute('aria-label', record.figure.querySelector('img')?.alt || 'Program revision results');
      svg.setAttribute('focusable', 'false');
      record.rects = rects;
      paint(record, reduced.matches ? 1 : 0);
      record.figure.replaceChildren(svg);
      record.figure.dataset.ready = 'true';
      sync(record);
    } catch (error) {
      // The original paper image stays visible if the vector asset cannot load.
      if (error.name !== 'AbortError') record.figure.dataset.ready = 'fallback';
    }
  });
  const syncAll = () => records.forEach(sync);
  document.addEventListener('visibilitychange', syncAll, {signal: abort.signal});
  reduced.addEventListener('change', syncAll, {signal: abort.signal});
  return () => {
    destroyed = true;
    abort.abort();
    observer.disconnect();
    records.forEach(stop);
  };
}
