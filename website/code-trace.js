/** Render source without evaluating it; highlight timestamped lines from a supplied trace. */
export class CodeTrace {
  constructor(root, player) {
    this.root=root; this.player=player; this.trace=null; this.rows=[]; this.lastKey='';
    this.content=root.querySelector('#code-content');
    this.filename=root.querySelector('#code-filename');
    this.state=root.querySelector('#code-state');
    this.time=root.querySelector('#code-time');
    this.update=this.update.bind(this);
    this.events=['timeupdate','seeked','seeking','play','pause','ended','loadedmetadata','error'];
    this.videos=[...player.root.querySelectorAll('video')];
    this.videos.forEach(video=>this.events.forEach(event=>video.addEventListener(event,this.update)));
  }
  setTrace(trace) {
    this.trace=trace && typeof trace.source==='string' && trace.source.trim() ? trace : null;
    this.content.replaceChildren(); this.rows=[]; this.lastKey='';
    this.filename.textContent=this.trace?.filename || 'task.py';
    if(!this.trace){
      const empty=document.createElement('div');empty.className='code-empty';
      const icon=document.createElement('span');icon.textContent='{ }';icon.setAttribute('aria-hidden','true');
      const label=document.createElement('p');label.textContent='Code pending';empty.append(icon,label);this.content.append(empty);
      this.update();return;
    }
    const pre=document.createElement('pre');pre.className='code-lines';pre.setAttribute('aria-label',this.trace.filename || 'Task source');
    this.trace.source.replace(/\r\n/g,'\n').split('\n').forEach((line,index)=>{
      const row=document.createElement('span');row.className='code-line';
      const number=document.createElement('span');number.className='line-number';number.textContent=String(index+1);number.setAttribute('aria-hidden','true');
      const code=document.createElement('code');this.highlight(line,code);row.append(number,code);pre.append(row);this.rows.push(row);
    });
    this.content.append(pre);this.update();
  }
  highlight(line,host) {
    const tokens=/(#.*$|"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|\b(?:def|class|if|else|elif|for|while|return|import|from|as|with|try|except|finally|raise|assert|True|False|None|and|or|not|in|is|await|async|pass|break|continue)\b|\b\d+(?:\.\d+)?\b)/g;
    let last=0;
    for(const token of line.matchAll(tokens)){
      host.append(document.createTextNode(line.slice(last,token.index)));
      const span=document.createElement('span');span.textContent=token[0];
      span.className=token[0][0]==='#'?'token-comment':/^["']/.test(token[0])?'token-string':/^\d/.test(token[0])?'token-number':'token-keyword';
      host.append(span);last=token.index+token[0].length;
    }
    host.append(document.createTextNode(line.slice(last)||' '));
  }
  update() {
    const master=this.videos.find(video=>video.getAttribute('src') && !video.error);
    const seconds=master?.currentTime || 0;
    this.time.textContent=`${String(Math.floor(seconds/60)).padStart(2,'0')}:${String(Math.floor(seconds%60)).padStart(2,'0')}`;
    const playing=Boolean(master && !master.paused && !master.ended);
    this.state.dataset.state=playing?'playing':'paused';
    this.state.textContent=!this.trace?'No trace':!master?'Source ready':playing?'Running':'Paused';
    const cues=this.trace?.cues || [];
    const active=master ? cues.findLast(cue=>Number.isFinite(cue.start) && seconds>=cue.start && (cue.end==null || seconds<cue.end)) : null;
    const lines=Array.isArray(active?.lines)?active.lines:active?.line?[active.line]:[];
    const key=lines.join(',');
    if(key===this.lastKey)return;
    this.rows.forEach((row,index)=>{const current=lines.includes(index+1);row.classList.toggle('is-current',current);if(current)row.setAttribute('aria-current','true');else row.removeAttribute('aria-current');});
    this.lastKey=key;
    const row=this.rows[lines[0]-1];
    if(row){const y=row.offsetTop-this.content.offsetTop;const lower=this.content.scrollTop+this.content.clientHeight-50;if(y<this.content.scrollTop+25 || y>lower)this.content.scrollTo({top:Math.max(0,y-this.content.clientHeight/2),behavior:'auto'});}
  }
  destroy(){this.videos.forEach(video=>this.events.forEach(event=>video.removeEventListener(event,this.update)));}
}
