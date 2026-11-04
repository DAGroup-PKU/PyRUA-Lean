/** Deliberate scroll gestures: reel ↔ mosaic → paper. */
export function mountCinematic(root, items = [], reel = []) {
  const doc=root.ownerDocument, win=doc.defaultView;
  const reduced=win.matchMedia('(prefers-reduced-motion: reduce)'), abort=new AbortController();
  const on=(target,event,fn,options={})=>target.addEventListener(event,fn,{...options,signal:abort.signal});
  const make=(tag,cls)=>Object.assign(doc.createElement(tag),{className:cls});
  const playlist=reel.length?reel:items.slice(0,2);
  let paused=false,visible=(!win.location.hash||win.location.hash==='#top'),destroyed=false,progress=0,animation=0;
  let phase='reel',exiting=false,lastWheel=-Infinity,transitionEnded=-Infinity,touchY=null,shot=0,active=0,fading=false,fadeTimer=0;
  const openingVideo=root.querySelector('[data-opening-video]');
  const sticky=root.querySelector('.cin-sticky')||make('div','cin-sticky'),field=make('div','cin-field'),overlay=make('div','cin-overlay');
  const identity=make('div','cin-identity');
  identity.innerHTML='<div class="cin-wordmark">PyRUA-REAL:</div><div class="cin-subtitle"><span class="real-initial">R</span>obot <span class="real-initial">E</span>xploration and <span class="real-initial">A</span>daptive <span class="real-initial">L</span>earning through RSI in the Real World</div>';
  const down=make('button','cin-down');down.type='button';
  overlay.append(identity,down);sticky.append(field,overlay);root.classList.add('cin-stage');if(sticky.parentElement!==root)root.replaceChildren(sticky);
  function videoFor(item){
    const v=make('video','cin-video');v.muted=v.defaultMuted=true;v.playsInline=true;v.setAttribute('muted','');v.setAttribute('playsinline','');v.setAttribute('aria-hidden','true');v.tabIndex=-1;
    v.preload='none';v.poster=item.poster;v.src=item.src;return v;
  }
  const entries=items.slice(0,16).map(item=>{
    const tile=make('div','cin-tile'),video=videoFor(item);tile.dataset.task=item.taskId||item.id;video.loop=true;video.preload='none';tile.append(video);field.append(tile);return {tile,video,pending:false};
  });
  const hero=make('div','cin-hero');field.append(hero);
  const heroes=[0,1].map(i=>{const video=i===0&&openingVideo?openingVideo:videoFor(playlist[i%playlist.length]);video.autoplay=false;video.preload=i===0&&visible?'auto':'none';video.classList.add('cin-reel-layer');video.style.opacity=i?'0':'1';hero.append(video);return {video,pending:false};});
  // Preserve the initial element and its downloaded buffer; warm the next clip
  // only after the opening has started playing.
  let nextWarmed=false;
  function warmNext(){
    if(nextWarmed||destroyed||paused||!visible||doc.hidden)return;
    nextWarmed=true;heroes[1].video.preload='auto';heroes[1].video.load();
  }
  on(heroes[0].video,'playing',warmNext);
  if(!heroes[0].video.paused&&heroes[0].video.readyState>=3)warmNext();
  function play(entry,allowed){
    if(!allowed){entry.video.pause();entry.video.preload='none';return;}
    if(!entry.video.paused||entry.pending)return;
    entry.pending=true;Promise.resolve(entry.video.play()).catch(error=>{if(heroes.includes(entry)&&error.name==='NotAllowedError')playback.hidden=false;}).finally(()=>{entry.pending=false;if(destroyed||paused||!visible||doc.hidden)entry.video.pause();});
  }
  function sync(){
    const enabled=!destroyed&&!paused&&visible&&!doc.hidden;
    heroes.forEach((e,i)=>play(e,enabled&&(i===active||fading)));
    entries.forEach(e=>play(e,enabled&&progress>0));
    root.dataset.motion=paused?'paused':'playing';
  }
  function updatePoster(){hero.style.setProperty('--hero-poster',`url("${playlist[shot].poster}")`);root.dataset.shot=playlist[shot].id;}
  function crossfade(){
    if(fading||paused||!visible||doc.hidden)return;
    const next=1-active;if(heroes[next].video.readyState<3){const current=heroes[active].video;if(current.ended){current.currentTime=0;sync();}return;}
    const old=active;active=next;shot=(shot+1)%playlist.length;fading=true;
    heroes[active].video.currentTime=0;heroes[active].video.style.opacity='1';heroes[old].video.style.opacity='0';updatePoster();sync();
    fadeTimer=win.setTimeout(()=>{fading=false;const v=heroes[old].video,item=playlist[(shot+1)%playlist.length];v.pause();v.preload='auto';v.poster=item.poster;v.src=item.src;v.load();sync();},650);
  }
  heroes.forEach((entry,i)=>{
    on(entry.video,'timeupdate',()=>{if(i===active&&entry.video.duration-entry.video.currentTime<.55)crossfade();});
    on(entry.video,'ended',()=>{if(i===active)crossfade();});
    on(entry.video,'canplay',()=>{if(i!==active&&heroes[active].video.ended)crossfade();sync();});
  });
  function render(){
    const w=sticky.clientWidth,h=sticky.clientHeight,cw=w/4,ch=h/4,eased=progress*progress*(3-2*progress);
    hero.style.left=`${3*cw*eased}px`;hero.style.top=`${3*ch*eased}px`;hero.style.width=`${w+(cw-w)*eased}px`;hero.style.height=`${h+(ch-h)*eased}px`;
    entries.forEach(({tile},i)=>{tile.style.left=`${i%4*cw}px`;tile.style.top=`${Math.floor(i/4)*ch}px`;tile.style.width=`${cw}px`;tile.style.height=`${ch}px`;tile.style.opacity=i===15?'0':String(Math.min(1,progress*3));});
    const reveal=Math.min(1,progress*1.6);identity.style.opacity=String(reveal);identity.style.transform=`translate(-50%,calc(-50% + ${(1-reveal)*40}px))`;
    hero.dataset.expanded=String(progress<.85);root.dataset.cinematic=phase;
    down.innerHTML=phase==='mosaic'?'Scroll to read<span>↓</span>':'Scroll to explore<span>↓</span>';
    down.setAttribute('aria-label',phase==='mosaic'?'Continue to the paper':'Reveal the project');
  }
  function transition(target){
    if(animation || progress===target)return;
    phase=target?'shrink':'expand';
    const from=progress,start=performance.now(),duration=reduced.matches?0:950;
    function tick(now){
      const t=duration?Math.max(0,Math.min(1,(now-start)/duration)):1;
      progress=from+(target-from)*t;render();sync();
      if(t<1)animation=win.requestAnimationFrame(tick);
      else{animation=0;transitionEnded=performance.now();phase=target?'mosaic':'reel';render();}
    }
    animation=win.requestAnimationFrame(tick);
  }
  function leave(){exiting=true;doc.getElementById('paper').scrollIntoView({behavior:reduced.matches?'instant':'smooth',block:'start'});}
  function advance(){if(phase==='reel')transition(1);else if(phase==='mosaic')leave();}
  function atOpening(){return !exiting&&root.getBoundingClientRect().top>=-3&&win.scrollY<root.offsetTop+4;}
  on(win,'wheel',e=>{
    const now=performance.now(),fresh=now-lastWheel>300&&now-transitionEnded>500;lastWheel=now;
    if(e.ctrlKey||e.deltaY===0||Math.abs(e.deltaX)>Math.abs(e.deltaY)||!atOpening())return;
    e.preventDefault();
    if(e.deltaY<0){if(phase==='mosaic'&&fresh)transition(0);}
    else if(phase==='reel'||(phase==='mosaic'&&fresh))advance();
  },{passive:false});
  on(sticky,'touchstart',e=>{touchY=e.touches.length===1?e.touches[0].clientY:null;},{passive:true});
  on(sticky,'touchmove',e=>{if(touchY!==null&&atOpening()&&Math.abs(touchY-e.touches[0].clientY)>8)e.preventDefault();},{passive:false});
  on(sticky,'touchend',e=>{if(touchY!==null&&atOpening()){const delta=touchY-e.changedTouches[0].clientY;if(delta>28)advance();else if(delta< -28&&phase==='mosaic')transition(0);}touchY=null;},{passive:true});
  on(win,'keydown',e=>{if(e.repeat||!atOpening()||!['ArrowDown','PageDown','ArrowUp','PageUp',' '].includes(e.key)||e.target.closest('button,a,input,select,textarea'))return;e.preventDefault();if(['ArrowUp','PageUp'].includes(e.key)||(e.key===' '&&e.shiftKey)){if(phase==='mosaic')transition(0);}else advance();});
  on(win,'scroll',()=>{if(exiting&&win.scrollY<3)exiting=false;},{passive:true});
  function replayOpening(){
    if(animation)win.cancelAnimationFrame(animation);win.clearTimeout(fadeTimer);animation=0;fading=false;phase='reel';progress=0;shot=0;active=0;paused=false;exiting=false;nextWarmed=false;lastWheel=performance.now();
    heroes.forEach((e,i)=>{e.video.pause();e.video.style.transition='none';e.video.style.opacity=i?'0':'1';e.video.src=playlist[i%playlist.length].src;e.video.poster=playlist[i%playlist.length].poster;e.video.load();});
    win.requestAnimationFrame(()=>heroes.forEach(e=>e.video.style.transition=''));entries.forEach(e=>{e.video.currentTime=0;});root.scrollIntoView({behavior:'instant'});updatePoster();render();sync();
  }
  const playback=make('button','cin-play');playback.type='button';playback.textContent='Play video';playback.setAttribute('aria-label','Play introduction video');overlay.append(playback);on(playback,'click',()=>{paused=false;visible=true;sync();});heroes.forEach(({video})=>{on(video,'playing',()=>{playback.hidden=true;});on(video,'error',()=>{playback.hidden=false;});});playback.hidden=!paused;
  on(down,'click',advance);on(doc,'visibilitychange',sync);
  on(reduced,'change',()=>{paused=false;sync();});
  const resize=new ResizeObserver(render);resize.observe(sticky);
  const visibility=new IntersectionObserver(([e])=>{visible=e.isIntersecting;sync();},{threshold:0});visibility.observe(sticky);
  updatePoster();render();sync();
  return {replay:replayOpening,pause(){paused=true;sync();},resume(){paused=false;sync();},destroy(){destroyed=true;abort.abort();resize.disconnect();visibility.disconnect();win.cancelAnimationFrame(animation);win.clearTimeout(fadeTimer);[...entries,...heroes].forEach(({video})=>{video.pause();video.removeAttribute('src');video.load();});root.replaceChildren();}};
}
