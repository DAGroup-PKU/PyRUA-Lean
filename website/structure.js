const SVG_NS = 'http://www.w3.org/2000/svg';
let diagramCount = 0;

const icons = {
  experience: '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 4 16 4 16 0V5M4 12c0 4 16 4 16 0"/>',
  researcher: '<circle cx="10" cy="10" r="6"/><path d="m15 15 6 6M7 10h6m-3-3v6"/>',
  engineer: '<path d="m8 7-5 5 5 5m8-10 5 5-5 5m-3-14-2 18"/>',
  program: '<rect x="3" y="3" width="18" height="18" rx="3"/><path d="m9 8 7 4-7 4z"/>',
  human: '<circle cx="12" cy="7" r="3"/><path d="M5 21v-4a7 7 0 0 1 14 0v4"/>',
  feedback: '<path d="M3 9a9 9 0 1 1 0 7M3 3v6h6M12 7v5l4 2"/>',
  environment: '<path d="m12 3 9 5v9l-9 5-9-5V8Zm0 10v9M3 8l9 5 9-5"/>',
};
const icon = name => `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${icons[name] || icons.program}</svg>`;

const revisionCycle = version => [
  { name: `Program v${version}`, version, duration: 1600, phases: [{ edges: ['program'], source: 'engineer', target: 'program', caption: version === 1 ? 'The engineer writes Program v1.' : `The engineer publishes revised Program v${version}.` }] },
  { name: `Execute v${version}`, version, duration: 2700, phases: ['perception', 'planning', 'control'].map(stage => ({ edges: ['actions', 'observations'], source: 'environment', target: 'program', stage, caption: `Program v${version} executes with live observations.` })) },
  { name: `Evidence v${version}`, version, duration: 1100, phases: [{ edges: ['evidence'], source: 'program', target: 'feedback', caption: `Program v${version} produces execution evidence.` }] },
  { name: `Diagnose v${version}`, version, duration: 1500, phases: [{ edges: ['diagnose'], source: 'feedback', target: 'engineer', caption: version < 3 ? `The engineer diagnoses v${version} before revising.` : 'The engineer reviews the latest execution.' }] },
];
const STEPS = [
  { name: 'Goal', version: 0, duration: 1800, phases: [{ edges: ['goal'], source: 'human', target: 'researcher', caption: 'Human input defines the goal and task variants.' }] },
  { name: 'Retrieve', version: 0, duration: 2000, phases: [{ edges: ['retrieve'], source: 'experience', target: 'researcher', caption: 'The researcher retrieves relevant experience.' }] },
  { name: 'Experiment', version: 0, duration: 1800, phases: [{ edges: ['experiment'], source: 'researcher', target: 'engineer', caption: 'The researcher specifies the next experiment.' }] },
  ...revisionCycle(1),
  ...revisionCycle(2),
  ...revisionCycle(3),
  { name: 'Report', version: 3, duration: 2000, phases: [{ edges: ['report'], source: 'engineer', target: 'researcher', caption: 'The engineer reports the experiment findings.' }] },
  { name: 'Retain', version: 3, duration: 2600, phases: [{ edges: ['retain'], source: 'researcher', target: 'experience', retain: true, caption: 'Memory and implementations are retained for reuse.' }] },
];
const STATIC_PHASE = { edges: ['program', 'evidence', 'diagnose'], source: 'engineer', target: 'program', involved: ['feedback'], caption: 'Execution feedback drives successive program revisions.' };

export function mountStructure(root) {
  if (!root) return () => {};
  const uid = `stx-${++diagramCount}`;
  const makeNode = (id, title, body, extra = '') => `<div class="stx-node stx-node-${id} ${extra}" data-stx-node="${id}" role="group" aria-label="${title}"><span class="stx-node-heading"><span class="stx-node-icon">${icon(id)}</span><strong>${title}</strong></span>${body}</div>`;
  root.classList.add('stx-root');
  root.innerHTML = `
    <div class="stx-map" role="group" aria-label="PyRUA-REAL system structure">
      ${makeNode('experience', 'Experience', `<span class="stx-experience-parts"><span data-stx-store="facts"><b>Facts</b><small>Workspace · Conditions</small></span><span data-stx-store="memory"><b>Memory</b><small>Attempts · Findings</small></span><span data-stx-store="implementations"><b>Implementations</b><small>Programs · Skills</small></span></span>`)}
      <div class="stx-core">
        ${makeNode('researcher', 'Researcher', '<span class="stx-role-items"><span>Objective</span><span>Stop criteria</span><span>Selected experience</span></span>')}
        ${makeNode('engineer', 'Engineer', '<span class="stx-role-items"><span>Code revisions</span><span>Diagnosis</span><span>Reports</span></span>')}
        ${makeNode('program', 'Program', '<span class="stx-revisions" aria-label="Successive program revisions"><b data-stx-version="1">v1</b><b data-stx-version="2">v2</b><b data-stx-version="3">v3</b></span><span class="stx-program-stages"><span data-stx-stage="perception"><i></i>Perception</span><span data-stx-stage="planning"><i></i>Planning</span><span data-stx-stage="control"><i></i>Control</span></span>')}
      </div>
      <div class="stx-bottom">
        ${makeNode('human', 'Human input', '<span class="stx-node-detail">Goal · Survey</span>')}
        ${makeNode('feedback', 'Feedback', '<span class="stx-node-detail">Frames · States · Logs</span>')}
        ${makeNode('environment', 'Environment', '<span class="stx-node-detail">Live observations</span>')}
      </div>
      <svg class="stx-links" aria-hidden="true"><defs><marker id="${uid}-arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="4" markerHeight="4" orient="auto-start-reverse"><path d="M0 0 10 5 0 10z" fill="currentColor"/></marker><marker id="${uid}-active-arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="4" markerHeight="4" orient="auto-start-reverse"><path d="M0 0 10 5 0 10z" fill="#8c0000"/></marker></defs><g class="stx-links-layer"></g></svg>
    </div>
    <div class="stx-player"><p class="stx-caption"></p></div>`;

  const map = root.querySelector('.stx-map');
  const svg = root.querySelector('.stx-links');
  const layer = root.querySelector('.stx-links-layer');
  const nodes = Object.fromEntries([...root.querySelectorAll('[data-stx-node]')].map(node => [node.dataset.stxNode, node]));
  const caption = root.querySelector('.stx-caption');
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  let step = 0;
  let phase = 0;
  let elapsed = 0;
  let running = false;
  let inView = false;
  let destroyed = false;
  let ticker = null;
  let redrawFrame = null;
  let lastTick = null;

  const port = (name, side, fraction = .5) => {
    const area = map.getBoundingClientRect();
    const bounds = nodes[name].getBoundingClientRect();
    const x = bounds.left - area.left, y = bounds.top - area.top;
    if (side === 'top') return [x + bounds.width * fraction, y];
    if (side === 'bottom') return [x + bounds.width * fraction, y + bounds.height];
    if (side === 'left') return [x, y + bounds.height * fraction];
    return [x + bounds.width, y + bounds.height * fraction];
  };
  const roundedPath = raw => {
    const points = raw.filter((p, i) => !i || p[0] !== raw[i - 1][0] || p[1] !== raw[i - 1][1]);
    let path = `M${points[0]}`;
    for (let i = 1; i < points.length - 1; i++) {
      const prev = points[i - 1], here = points[i], next = points[i + 1];
      const before = Math.hypot(prev[0] - here[0], prev[1] - here[1]);
      const after = Math.hypot(next[0] - here[0], next[1] - here[1]);
      const r = Math.min(9, before / 2, after / 2);
      const a = [here[0] + (prev[0] - here[0]) * r / before, here[1] + (prev[1] - here[1]) * r / before];
      const b = [here[0] + (next[0] - here[0]) * r / after, here[1] + (next[1] - here[1]) * r / after];
      path += ` L${a} Q${here} ${b}`;
    }
    return `${path} L${points.at(-1)}`;
  };
  const edge = (id, points, label = '', labelAt = null) => {
    const group = document.createElementNS(SVG_NS, 'g');
    group.setAttribute('class', 'stx-edge');
    group.dataset.stxEdge = id;
    const d = roundedPath(points);
    const line = document.createElementNS(SVG_NS, 'path');
    line.setAttribute('class', 'stx-edge-line'); line.setAttribute('d', d); line.setAttribute('marker-end', `url(#${uid}-arrow)`);
    const packet = document.createElementNS(SVG_NS, 'path');
    packet.setAttribute('class', 'stx-edge-packet'); packet.setAttribute('d', d);
    group.append(line, packet);
    if (label && labelAt) {
      const text = document.createElementNS(SVG_NS, 'text');
      text.textContent = label;
      text.setAttribute('class', 'stx-edge-label'); text.setAttribute('x', labelAt[0]); text.setAttribute('y', labelAt[1]);
      group.append(text);
    }
    layer.append(group);
  };
  const paint = () => {
    if (destroyed) return;
    const isStatic = reducedMotion.matches;
    const current = isStatic ? STATIC_PHASE : STEPS[step].phases[phase];
    const version = isStatic ? 3 : STEPS[step].version;
    root.classList.toggle('stx-playing', running);
    root.classList.toggle('stx-static', isStatic);
    root.dataset.stxState = isStatic ? 'static' : STEPS[step].name.toLowerCase().replace(/[^a-z0-9]+/g, '-');
    root.dataset.stxVersion = String(version);
    Object.entries(nodes).forEach(([name, node]) => {
      const source = name === current.source || current.involved?.includes(name);
      node.classList.toggle('stx-active', name === current.target);
      node.classList.toggle('stx-source', !!source);
      node.classList.toggle('stx-idle', name !== current.target && !source);
    });
    layer.querySelectorAll('[data-stx-edge]').forEach(item => {
      const active = current.edges.includes(item.dataset.stxEdge);
      item.classList.toggle('stx-current', active);
      item.querySelector('.stx-edge-line').setAttribute('marker-end', `url(#${uid}-${active ? 'active-arrow' : 'arrow'})`);
    });
    root.querySelectorAll('[data-stx-stage]').forEach(item => item.classList.toggle('stx-stage-active', item.dataset.stxStage === current.stage));
    root.querySelectorAll('[data-stx-store]').forEach(item => item.classList.toggle('stx-store-active', !!current.retain && item.dataset.stxStore !== 'facts'));
    root.querySelectorAll('[data-stx-version]').forEach(item => {
      const revision = Number(item.dataset.stxVersion);
      item.classList.toggle('stx-version-current', revision === version);
      item.classList.toggle('stx-version-past', revision < version);
      item.classList.toggle('stx-version-future', revision > version);
    });
    nodes.program.setAttribute('aria-label', version ? `Program v${version}` : 'Program');
    if (caption.textContent !== current.caption) caption.textContent = current.caption;
  };
  const draw = () => {
    if (destroyed || !map.clientWidth) return;
    layer.replaceChildren();
    svg.setAttribute('viewBox', `0 0 ${map.clientWidth} ${map.clientHeight}`);
    const mobile = map.clientWidth < 620;
    const human = port('human', 'top'), researcher = port('researcher', 'bottom');
    edge('goal', [human, researcher], mobile ? '' : 'Goal + survey', [human[0], (human[1] + researcher[1]) / 2 - 9]);
    const retrieve = port('researcher', 'top', .5);
    const experience = [retrieve[0], port('experience', 'bottom')[1]];
    edge('retrieve', [experience, retrieve], mobile ? '' : 'Retrieve', [retrieve[0] + 35, (retrieve[1] + experience[1]) / 2 + 4]);
    const rOut = port('researcher', 'right', .37), eIn = port('engineer', 'left', .37);
    edge('experiment', [rOut, eIn], mobile ? '' : 'Experiment', [(rOut[0] + eIn[0]) / 2, rOut[1] - 11]);
    const eOut = port('engineer', 'left', .76), rIn = port('researcher', 'right', .76);
    edge('report', [eOut, rIn], mobile ? '' : 'Report', [(eOut[0] + rIn[0]) / 2, eOut[1] + 21]);
    const eProgram = port('engineer', 'right', .37), pIn = port('program', 'left', .37);
    edge('program', [eProgram, pIn], mobile ? '' : 'Program', [(eProgram[0] + pIn[0]) / 2, eProgram[1] - 11]);
    const feedback = port('feedback', 'top'), engineer = port('engineer', 'bottom');
    edge('diagnose', [feedback, engineer], mobile ? '' : 'Diagnose', [feedback[0] - 34, (feedback[1] + engineer[1]) / 2 + 4]);
    const pEvidence = port('program', 'left', .83), fEvidence = port('feedback', 'top', .83);
    const laneX = (port('program', 'left')[0] + port('engineer', 'right')[0]) / 2;
    const laneY = (port('program', 'bottom')[1] + port('feedback', 'top')[1]) / 2;
    edge('evidence', [pEvidence, [laneX, pEvidence[1]], [laneX, laneY], [fEvidence[0], laneY], fEvidence], mobile ? '' : 'Evidence', [laneX + 35, laneY - 9]);
    const action = port('program', 'bottom', .72), envAction = port('environment', 'top', .72);
    edge('actions', [action, envAction], mobile ? '' : 'Action', [action[0] + 27, (action[1] + envAction[1]) / 2 + 4]);
    const envObs = port('environment', 'top', .32), observation = port('program', 'bottom', .32);
    edge('observations', [envObs, observation], mobile ? '' : 'Observe', [observation[0] - 34, (observation[1] + envObs[1]) / 2 + 4]);
    const retained = port('researcher', 'left', .3), store = port('experience', 'left', .5);
    const outer = Math.max(3, retained[0] - (mobile ? 12 : 21));
    edge('retain', [retained, [outer, retained[1]], [outer, store[1]], store]);
    paint();
  };
  const scheduleDraw = () => {
    if (destroyed) return;
    cancelAnimationFrame(redrawFrame);
    redrawFrame = requestAnimationFrame(draw);
  };
  const canPlay = () => !destroyed && inView && !document.hidden && !reducedMotion.matches;
  const tick = now => {
    ticker = null;
    if (!canPlay()) { running = false; lastTick = null; paint(); return; }
    const delta = lastTick === null ? 0 : Math.max(0, Math.min(now - lastTick, 100));
    lastTick = now;
    elapsed += delta;
    let changed = false;
    while (elapsed >= STEPS[step].duration) {
      elapsed -= STEPS[step].duration;
      step = (step + 1) % STEPS.length;
      phase = 0;
      changed = true;
    }
    const current = STEPS[step];
    const nextPhase = Math.max(0, Math.min(current.phases.length - 1, Math.floor(elapsed / current.duration * current.phases.length)));
    if (nextPhase !== phase) { phase = nextPhase; changed = true; }
    if (changed) paint();
    ticker = requestAnimationFrame(tick);
  };
  const syncPlayback = () => {
    if (destroyed) return;
    const shouldPlay = canPlay();
    if (shouldPlay && !running) {
      running = true;
      lastTick = null;
      ticker = requestAnimationFrame(tick);
    } else if (!shouldPlay) {
      running = false;
      cancelAnimationFrame(ticker);
      ticker = null;
      lastTick = null;
    }
    paint();
  };
  reducedMotion.addEventListener('change', syncPlayback);
  document.addEventListener('visibilitychange', syncPlayback);
  const resize = new ResizeObserver(scheduleDraw);
  resize.observe(map);
  const observer = new IntersectionObserver(entries => {
    inView = entries[0].isIntersecting && entries[0].intersectionRatio >= .15;
    syncPlayback();
  }, { threshold: [0, .15] });
  observer.observe(root);
  document.fonts?.ready.then(scheduleDraw);
  paint(); scheduleDraw();
  return () => {
    destroyed = true;
    cancelAnimationFrame(ticker);
    cancelAnimationFrame(redrawFrame);
    resize.disconnect();
    observer.disconnect();
    reducedMotion.removeEventListener('change', syncPlayback);
    document.removeEventListener('visibilitychange', syncPlayback);
    root.classList.remove('stx-playing');
  };
}
