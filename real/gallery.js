/** Full-frame recordings, visible-card playback, and evidence-backed code captions. */
export function mountGallery(root, media){
  const grid=root.querySelector('#task-video-grid'),reduced=matchMedia('(prefers-reduced-motion: reduce)');
  const codeToggle=root.querySelector('.gallery-code-toggle');
  const make=(tag,cls,text)=>Object.assign(document.createElement(tag),{className:cls,...(text===undefined?{}:{textContent:text})});
  const clock=t=>`${Math.floor((t||0)/60)}:${String(Math.floor((t||0)%60)).padStart(2,'0')}`;
  let domain='real',cards=[],paused=false,overlays={},expandedCard=null,selectedCard=null;
  const observer=new IntersectionObserver(entries=>{entries.forEach(entry=>{const card=cards.find(c=>c.frame===entry.target);if(card)card.visible=entry.isIntersecting&&entry.intersectionRatio>0;});updateMotion();},{threshold:0});
  function allowed(card){if(paused||document.hidden)return false;if(expandedCard)return expandedCard===card;return card.visible;}
  function sync(card){
    if(!card.video)return;
    const playing=allowed(card);
    if(!playing)card.video.pause();
    if(!playing&&!card.warm){card.video.preload='none';if(card.hlsReady&&card.hlsLoading)card.hls.stopLoad();card.hlsLoading=false;return;}
    card.video.preload='auto';
    if(card.hls)card.hls.autoLevelCapping=expandedCard===card?-1:0;
    if(card.hlsNeeded&&!card.hls){if(!card.hlsInitializing){card.hlsInitializing=true;card.prepareHls().catch(()=>{card.hlsInitializing=false;card.playButton.hidden=false;card.playButton.textContent='Retry playback';});}return;}
    if(!card.hls && !card.video.getAttribute('src')){card.video.src=card.source;card.video.preload='auto';}
    if(card.hls&&!card.hlsStarted){card.hlsStarted=true;card.hls.loadSource(card.source);}
    if(card.hlsReady&&!card.hlsLoading){card.hlsLoading=true;card.hls.startLoad(card.hlsEverLoaded?-1:(card.item.startAt||-1));card.hlsEverLoaded=true;}
    if(!playing)return;
    card.video.playbackRate=card.item.playbackRate||8;if(!card.pending&&card.video.paused){card.pending=true;card.video.play().catch(()=>{card.playButton.hidden=false;}).finally(()=>{card.pending=false;if(!allowed(card))card.video.pause();});}
  }
  function updateMotion(){
    cards.forEach(card=>{card.warm=false;sync(card);});
  }
  function attachCode(card){
    const trace=overlays[card.item.id];if(!trace?.lines?.length)return;
    const panel=make('aside','gallery-code');panel.setAttribute('aria-label',trace.hideLabel?`${card.item.title} code`:`${card.item.title} — ${trace.label||'Source excerpt'}`);
    panel.title=trace.hideLabel?trace.file:`${trace.file} · ${trace.label||'Source excerpt'}`;
    const label=make('div','gallery-code-label',trace.label||'Source excerpt'),viewport=make('div','gallery-code-viewport'),track=make('div','gallery-code-track');
    const lines=trace.lines.map((line,i)=>{const row=make('div','gallery-code-line');row.append(make('span','gallery-line-number',String(i+1).padStart(2,'0')),make('code','',line||' '));track.append(row);return row;});
    viewport.append(track);if(!trace.hideLabel)panel.append(label);panel.append(viewport);card.frame.append(panel);card.trace={...trace,track,lines,viewport,lastLine:-1};updateCode(card);
  }
  function updateCode(card){
    const trace=card.trace;if(!trace)return;const time=card.video.currentTime*(card.item.timelineScale||1);
    // A cue highlights a logged stage, never a guessed line-level execution trace.
    let line=0,hasCue=false;
    if(trace.phases?.length){
      const phase=trace.phases.findLast(p=>p.start<=time)||trace.phases[0];
      const progress=Math.max(0,Math.min(1,(time-phase.start)/(phase.end-phase.start)));
      line=phase.startLine+Math.floor(progress*(phase.endLine-phase.startLine));
    }
    else if(trace.cues?.length){const cue=trace.cues.findLast(c=>c.time<=time);line=cue?.line??0;hasCue=!!cue;}
    else if(card.video.duration)line=Math.floor(card.video.currentTime/card.video.duration*Math.max(0,trace.lines.length-4));
    line=Math.max(0,Math.min(trace.lines.length-1,line));
    if(trace.lastLine!==line||trace.hasCue!==hasCue){trace.lines.forEach((row,i)=>row.classList.toggle('is-current',hasCue&&i===line));trace.lastLine=line;trace.hasCue=hasCue;}
    const height=trace.lines[0].getBoundingClientRect().height||13;
    trace.track.style.transform=`translateY(${-Math.max(0,line-2)*height}px)`;
  }
  function render(){
    codeToggle.hidden=domain==='sim';observer.disconnect();cards.forEach(c=>{clearTimeout(c.retryTimer);c.video?.pause();c.hls?.destroy();if(c.video){c.video.removeAttribute('src');c.video.load();}});cards=[];grid.replaceChildren();grid.dataset.domain=domain;
    const items=domain==='real'?media.real:media.sim;
    items.forEach(item=>{
      const article=make('article','task-video-card'),header=make('header','gallery-card-header');article.dataset.task=item.id;
      if(item.poster){const img=make('img','gallery-task-image');img.src=item.poster;img.alt='';img.loading='lazy';header.append(img);}
      const title=make('h4','',item.title);if(item.onlineCoding)title.append(make('sup','gallery-asterisk','*'));header.append(title);article.append(header);
      const player=make('div','gallery-player'),frame=make('div','gallery-frame');player.append(frame);article.append(player);
      if(item.onlineCoding){const note=make('p','gallery-recording-note');note.innerHTML='* This execution includes <strong>agent online coding</strong>. ';article.append(note);}
      grid.append(article);
      const card={item,frame,visible:false,userPaused:false,pending:false,video:null};cards.push(card);
      const source=item.sources.find(Boolean);
      if(!source){frame.classList.add('gallery-pending');frame.append(make('span','','Recording forthcoming'));return;}
      const video=make('video','gallery-video');Object.assign(video,{muted:true,defaultMuted:true,playsInline:true,loop:!item.startAt,preload:'none',playbackRate:item.playbackRate||8});video.setAttribute('aria-label',domain==='sim'?`${item.title} recorded rollout, original playback rate`:`${item.title} ${item.trimmed?"recording with shortened leading idle":"complete recording"}, eight times speed`);video.setAttribute('muted','');video.setAttribute('playsinline','');card.video=video;if(item.videoPoster)video.poster=item.videoPoster;frame.append(video);
      card.source=source;card.hlsNeeded=source.includes('.m3u8');
      card.prepareHls=async()=>{const {default:Hls}=await import('./vendor/hls.light.mjs?v=22');if(!cards.includes(card))return;
      if(Hls.isSupported()){
        // Use the built-in loader: one request per HLS segment, with standard timeout/retry handling.
        card.hls=new Hls({startPosition:item.startAt||-1,autoStartLoad:false,enableWorker:true,startLevel:0,capLevelToPlayerSize:true,maxBufferLength:6,maxMaxBufferLength:12,maxBufferSize:32000000,backBufferLength:4});
        card.source=source;card.hls.attachMedia(video);
        for(const event of [Hls.Events.MANIFEST_LOADING,Hls.Events.MANIFEST_LOADED,Hls.Events.FRAG_LOADING,Hls.Events.FRAG_LOADED,Hls.Events.BUFFER_APPENDED])card.hls.on(event,(_,data)=>{frame.dataset.streamPhase=event;if(data.frag)frame.dataset.streamSegment=String(data.frag.sn);});
        card.hls.on(Hls.Events.BUFFER_APPENDED,updateMotion);
        card.hls.on(Hls.Events.MANIFEST_PARSED,()=>{card.hlsReady=true;sync(card);});
        card.hls.on(Hls.Events.ERROR,(_,data)=>{if(!data.fatal)return;frame.dataset.streamError=data.details;card.hlsLoading=false;card.playButton.hidden=false;card.playButton.textContent='Retry playback';if((card.retries||0)<3){card.retries=(card.retries||0)+1;clearTimeout(card.retryTimer);card.retryTimer=setTimeout(()=>{if(!cards.includes(card)||!allowed(card))return;if(data.type===Hls.ErrorTypes.MEDIA_ERROR)card.hls.recoverMediaError();sync(card);},1500*card.retries);}});
      sync(card);
      }else{card.hlsNeeded=false;sync(card);}
      };
      const playButton=make('button','gallery-play','Play');playButton.hidden=true;playButton.type='button';playButton.setAttribute('aria-label',`Play ${item.title}`);card.playButton=playButton;frame.append(playButton);playButton.addEventListener('click',()=>{paused=false;card.userPaused=false;selectedCard=card;card.retries=0;delete frame.dataset.streamError;if(card.hls&&video.error)card.hls.recoverMediaError();else if(!card.hls&&video.error)video.load();updateMotion();});
      video.addEventListener('progress',updateMotion);
      video.addEventListener('playing',()=>{card.retries=0;delete frame.dataset.streamError;playButton.hidden=true;});
      const expand=make('button','gallery-expand','⛶');expand.type='button';expand.setAttribute('aria-label',`Expand ${item.title}`);frame.append(expand);
      video.addEventListener('timeupdate',()=>updateCode(card));
      video.addEventListener('loadedmetadata',()=>{video.playbackRate=item.playbackRate||8;if(item.startAt&&video.currentTime<item.startAt)video.currentTime=item.startAt;sync(card);});
      if(item.startAt)video.addEventListener('ended',()=>{video.currentTime=item.startAt;sync(card);});
      video.addEventListener('error',()=>{playButton.hidden=false;playButton.textContent='Retry playback';});
      expand.addEventListener('click',()=>{
        if(player.classList.contains('is-expanded')){player.closest('dialog').close();return;}
        const placeholder=make('div','gallery-placeholder');placeholder.style.height=`${player.getBoundingClientRect().height}px`;player.before(placeholder);
        const dialog=make('dialog','gallery-dialog'),bar=make('div','gallery-dialog-header'),close=make('button','', '×');
        close.type='button';close.setAttribute('aria-label',`Close ${item.title}`);bar.append(make('span','',item.title),close);
        dialog.style.setProperty('--video-ratio',String(video.videoWidth/video.videoHeight||16/9));dialog.setAttribute('aria-label',`${item.title} expanded video`);dialog.append(bar,player);document.body.append(dialog);player.classList.add('is-expanded');document.body.classList.add('gallery-modal-open');
        close.addEventListener('click',()=>dialog.close());
        dialog.addEventListener('click',event=>{if(event.target===dialog)dialog.close();});
        dialog.addEventListener('close',()=>{player.classList.remove('is-expanded');placeholder.replaceWith(player);dialog.remove();document.body.classList.remove('gallery-modal-open');expandedCard=null;cards.forEach(sync);updateCode(card);},{once:true});
        dialog.showModal();expandedCard=card;cards.forEach(sync);updateCode(card);
      });
      observer.observe(frame);if(domain==='real')attachCode(card);
    });
    updateMotion();
  }
  root.querySelectorAll('[data-domain]').forEach(button=>button.addEventListener('click',()=>{
    domain=button.dataset.domain;root.querySelectorAll('[data-domain]').forEach(b=>{b.classList.toggle('active',b===button);b.setAttribute('aria-pressed',String(b===button));});render();
  }));
  codeToggle.addEventListener('click',()=>{const off=root.classList.toggle('gallery-hide-code');codeToggle.setAttribute('aria-checked',String(!off));});
  document.addEventListener('visibilitychange',updateMotion);
  document.addEventListener('fullscreenchange',()=>cards.forEach(updateCode));
  render();
  fetch('assets/code-overlays.json?v=17').then(r=>r.ok?r.json():{}).then(data=>{overlays=data;cards.filter(c=>c.video&&domain==='real').forEach(attachCode);}).catch(()=>{});
  return {pause(){paused=true;updateMotion();},resume(){paused=false;updateMotion();}};
}
