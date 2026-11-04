import {LEADERBOARD} from './leaderboard-data.js?v=19';
const dimensions=[['average','Average'],['generalization','Generalization'],['precision','Precision'],['longHorizon','Long-Horizon'],['memory','Memory'],['open','Open']];
const display=value=>value.toFixed(2);
export function leaderboardMarkup(){
  return `<div class="lb-panel">
    <div class="lb-controls"><div class="lb-tabs" role="group" aria-label="Capability dimension">${dimensions.map(([key,label],i)=>`<button type="button" data-dimension="${key}" aria-pressed="${i===0}">${label}</button>`).join('')}</div><div class="lb-tabs lb-metric" role="group" aria-label="Leaderboard metric"><button type="button" data-metric="sr" aria-pressed="true">SR</button><button type="button" data-metric="score" aria-pressed="false">Score</button></div></div>
    <div class="lb-legend"><span><i class="lb-ours"></i>PyRUA-REAL (ours)</span><span><i class="lb-astra"></i>GPT-6-Astra, direct control</span><span><i class="lb-baseline"></i>Selected board entries</span></div>
    <div class="lb-chart" role="img" aria-label="Average success rate comparison"><div class="lb-rows">${LEADERBOARD.models.map((m,i)=>`<div class="lb-row" data-model="${i}" data-kind="${m.kind}"><div class="lb-name"><strong>${m.name}</strong><span>${m.tag}</span></div><div class="lb-track"><div class="lb-bar"><span class="lb-value"></span></div></div></div>`).join('')}</div><div class="lb-axis" aria-hidden="true">${[0,25,50,75,100].map(n=>`<span style="left:${n}%">${n}</span>`).join('')}</div></div>
    <div class="lb-status" role="status" aria-live="polite"></div>
  </div>`;
}
export function mountLeaderboard(root){
  const abort=new AbortController(), reduced=matchMedia('(prefers-reduced-motion: reduce)');
  let dimension='average',metric='sr',visible=false;
  const rows=[...root.querySelectorAll('.lb-row')],container=root.querySelector('.lb-rows');
  function update(animate=false){
    const before=new Map(rows.map(row=>[row,row.getBoundingClientRect().top]));
    const sorted=LEADERBOARD.models.map((model,index)=>({model,index})).sort((a,b)=>b.model.values[dimension][metric]-a.model.values[dimension][metric]);
    sorted.forEach(({model,index})=>{
      const row=rows[index],value=model.values[dimension][metric];container.append(row);
      row.querySelector('.lb-bar').style.width=`${value}%`;
      row.querySelector('.lb-value').textContent=display(value);
      row.setAttribute('aria-label',`${model.name}: ${display(value)}${metric==='sr'?' percent':' score'}`);
    });
    if(animate&&!reduced.matches)rows.forEach(row=>{const delta=before.get(row)-row.getBoundingClientRect().top;if(delta)row.animate([{transform:`translateY(${delta}px)`},{transform:'translateY(0)'}],{duration:550,easing:'cubic-bezier(.2,.7,.2,1)'});});
    root.querySelectorAll('[data-dimension]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.dimension===dimension)));
    root.querySelectorAll('[data-metric]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.metric===metric)));
    const label=dimensions.find(([key])=>key===dimension)[1]+(metric==='sr'?' success rate (%)':' score');
    root.querySelector('.lb-chart').setAttribute('aria-label',label+'. '+sorted.map(({model})=>`${model.name}: ${display(model.values[dimension][metric])}`).join('; '));
    root.querySelector('.lb-status').textContent=label;
  }
  root.addEventListener('click',event=>{const button=event.target.closest('button');if(!button)return;if(button.dataset.dimension)dimension=button.dataset.dimension;else if(button.dataset.metric)metric=button.dataset.metric;else return;update(visible);},{signal:abort.signal});
  update();
  const observer=new IntersectionObserver(entries=>{if(entries.some(e=>e.isIntersecting)){visible=true;root.classList.add('lb-visible');observer.disconnect();}},{threshold:.2});observer.observe(root);
  return ()=>{abort.abort();observer.disconnect();};
}
