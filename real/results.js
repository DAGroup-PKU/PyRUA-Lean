import {leaderboardMarkup, mountLeaderboard} from './leaderboard.js?v=18';
import {RESULT_DATA as DATA} from './results-data.js?v=20';
import {mountPaperCharts} from './paper-charts.js';

const format = (value, places = 2) => Number(value).toLocaleString('en-US', {minimumFractionDigits: places, maximumFractionDigits: places});
const heading = title => `<header class="rs-heading"><h3 class="rs-label">${title}</h3></header>`;
const shortNames={'tic-tac-toe':'Tic-tac-toe','five-block-stacking':'Five-block<br>stacking','tower-of-hanoi':'Tower of<br>Hanoi','towel-folding':'Towel<br>folding','push-t':'Push T','block-sorting':'Block<br>sorting','block-storage':'Block<br>storage','ethernet-insertion':'Ethernet<br>insertion<sup>*</sup>'};
function acquisition() {
  const metrics=[
    {key:'success',title:'Success rate',unit:'%',max:100,ticks:[0,25,50,75,100],color:'#8c0000',text:'All seven tasks with agent-free solutions reached <strong>91.67–100% recorded development success</strong>. These rates pool attempts across program revisions; they do not measure a single frozen policy.'},
    {key:'hours',title:'Exploration time (h)',unit:'hours',max:50,ticks:[0,10,20,30,40,50],color:'#5b7f9f',text:'Time to first complete success ranged from <strong>about 1.5 to 45 hours</strong>. Acquisition cost varied substantially across tasks, with Push T requiring the longest exploration.'},
    {key:'seconds',title:'Execution time (s)',unit:'seconds',max:1200,ticks:[0,300,600,900,1200],color:'#a67538',text:'The acquired programs execute with <strong>live perception and control, without runtime language-model calls</strong>. Bars show successful-run medians; task-specific endpoints include reset for tic-tac-toe, stacking and Hanoi.'}
  ];
  return `<article class="rs-story rs-acquisition" id="result-acquisition">
    ${heading('Real world performance')}
    <div data-performance-slot></div>
    <div class="rs-metrics" id="real-world-charts">${metrics.map(metric=>`<figure class="rs-bar-chart" style="--bar-color:${metric.color}" aria-label="${metric.title}">
      <figcaption><h4>${metric.title}</h4>${metric.key==='seconds'?'<p class="rs-agent-time-note">Methods involving online agents can take over 10× as long. <a href="assets/paper.pdf#page=16" target="_blank" rel="noopener">See paper, §4.8.</a></p>':''}</figcaption>
      <div class="rs-vertical-chart">${metric.key==='success'?'<span class="rs-axis-unit">%</span>':''}
        <div class="rs-gridlines" aria-hidden="true">${metric.ticks.map(value=>`<div style="bottom:${value/metric.max*100}%"><span>${value.toLocaleString('en-US')}</span></div>`).join('')}</div>
        <div class="rs-columns">${DATA.tasks.map((task,i)=>`<div class="rs-column" data-task="${task.id}" style="--bar-size:${task[metric.key]/metric.max*100}%;--delay:${i*75}ms" aria-label="${task.name}: ${task[metric.key]} ${metric.unit}"><div class="rs-column-plot"><strong>${metric.key==='hours'?'~':''}${metric.key==='hours'?task.hours:format(task[metric.key])}</strong><i></i></div><span class="rs-column-name">${shortNames[task.id]}</span></div>`).join('')}</div>
      </div>
      <p class="rs-result-copy">${metric.text}</p><p class="rs-task-footnote">* Ethernet insertion uses online agent assistance.${metric.key==='hours'?' The ~15 h are additional exploration after 240 h without agent-free success.':''}</p>
    </figure>`).join('')}</div>
  </article>`;
}

function improvement() {
  return `<article class="rs-story rs-improvement" id="result-improvement">
    ${heading('Continued improvement')}
    <div class="rs-paper-charts">
      <figure class="rs-animated-figure" data-chart-src="assets/improvement-stacking-animated.svg?v=2"><img src="assets/improvement-stacking.webp?v=2" width="846" height="1578" decoding="async" alt="Five-block stacking: original success-rate and execution-time charts across 33 program revisions." loading="lazy"></figure>
      <figure class="rs-animated-figure" data-chart-src="assets/improvement-tic-animated.svg?v=2"><img src="assets/improvement-tic-tac-toe.webp?v=2" width="846" height="1578" decoding="async" alt="Tic-tac-toe: original success-rate and execution-time charts across nine program revisions." loading="lazy"></figure>
      <figure class="rs-animated-figure" data-chart-src="assets/improvement-hanoi-animated.svg?v=2"><img src="assets/improvement-hanoi.webp?v=2" width="846" height="1578" decoding="async" alt="Tower of Hanoi: original success-rate and execution-time charts across seven program revisions." loading="lazy"></figure>
    </div>
    <p class="rs-result-copy">Improvement continues <strong>beyond the first successful rollout</strong>. Stacking cycle time falls from 285.3 to 156.0 seconds, while tic-tac-toe and Hanoi also become faster through revised motion strategies. Intermediate regressions show why speed must be checked alongside reliability.</p>
  </article>`;
}

function benchmark() {
  return `<article class="rs-story rs-benchmark" id="result-benchmark">
    ${heading('RoboDojo performance')}
    ${leaderboardMarkup()}
    <p class="rs-result-copy">Fixed programs reach <strong>84.42% success and 88.53 score</strong> across 42 tasks (54 configurations; 2,100 episodes), with no language-model calls at evaluation time. These are development-seed results; the same programs achieve <strong>80.73% / 85.36 on held-out seed 2</strong>. The comparison uses separately reported evaluations, with five capability groups weighted equally.</p>
  </article>`;
}

function findings(){
 const findings=[
  ['Extra reviewer agents substantially slow exploration without improving outcomes.', '4.4', 11],
  ['Assign research investigation and physical exploration to separate agents.', '4.4', 12],
  ['Image-based feedback enables most effective exploration and self-improvement.', '4.5', 13],
  ['Explicit numerical requirements deliver the greatest generalization gains.', '4.6', 13],
  ['Cross-task experience positively transfer to new tasks.', '4.7', 14]
 ];
 return `<article class="rs-story rs-findings" id="other-findings">${heading('Further findings')}<div class="rs-finding-grid">${findings.map(([copy,section,page])=>`<article class="rs-finding-card"><p>${copy} <a href="assets/paper.pdf#page=${page}" target="_blank" rel="noopener">See paper, §${section}.</a></p></article>`).join('')}</div></article>`;
}

export function mountResults(root) {
  if (!root) return () => {};
  root.resultsDestroy?.();
  root.className = 'rs-root';
  // Keep the initial HTML anchor node when enhancing the results.
  const performance = root.querySelector('#performance');
  root.innerHTML = acquisition() + improvement() + benchmark() + findings();
  if (performance) root.querySelector('[data-performance-slot]').append(performance);
  const abort = new AbortController();
  const destroyCharts = mountPaperCharts(root.querySelector('.rs-paper-charts'));
  const destroyLeaderboard = mountLeaderboard(root.querySelector('.lb-panel'));
  const observer = new IntersectionObserver(entries => entries.forEach(entry => {
    if (entry.isIntersecting) { entry.target.classList.add('rs-visible'); observer.unobserve(entry.target); }
  }), {threshold: .12});
  root.querySelectorAll('.rs-story').forEach(item => observer.observe(item));
  const metricCharts = root.querySelectorAll('.rs-bar-chart');
  const chartObserver = new IntersectionObserver(entries=>entries.forEach(entry=>{
    if(entry.isIntersecting){entry.target.classList.add('bars-visible');chartObserver.unobserve(entry.target);}
  }),{threshold:.35});
  metricCharts.forEach(chart=>chartObserver.observe(chart));
  const destroy = () => { abort.abort(); observer.disconnect(); chartObserver.disconnect(); destroyCharts(); destroyLeaderboard(); delete root.resultsDestroy; };
  root.resultsDestroy = destroy;
  return destroy;
}
