// Small, retryable byte ranges keep a stalled large response from holding the
// entire fragment. Reassembly preserves the original media bytes exactly.
export function rangeLoader(LoadStats){
 return class {
  constructor(){this.stats=new LoadStats();this.context=null;this.controller=null;this.response=null;}
  load(context,config,callbacks){
   this.context=context;this.controller=new AbortController();this.stats.loading.start=performance.now();
   const signal=this.controller.signal,stats=this.stats,chunkSize=256*1024;
   const read=async(start,end)=>{
    for(let attempt=0;;attempt++){
     try{
      const response=await fetch(context.url,{credentials:'same-origin',headers:{Range:`bytes=${start}-${end}`},signal:AbortSignal.any([signal,AbortSignal.timeout(15000)])});
      this.response=response;if(!stats.loading.first)stats.loading.first=performance.now();
      if(!response.ok)throw Error(`HTTP ${response.status}`);
      const data=new Uint8Array(await response.arrayBuffer());
      if(response.status===206){const range=/^bytes (\d+)-(\d+)\/(\d+)$/.exec(response.headers.get('Content-Range')||'');if(!range||Number(range[1])!==start||data.length!==Number(range[2])-start+1)throw Error('Incomplete video range');return {data,total:Number(range[3]),partial:true};}
      if(start!==0)throw Error('Video range ignored');return {data,total:data.length,partial:false};
     }catch(error){if(signal.aborted||attempt>=2)throw error;stats.retry++;}
    }
   };
   (async()=>{
    const first=await read(0,chunkSize-1),data=new Uint8Array(first.total);data.set(first.data);stats.total=first.total;stats.loaded=first.data.length;
    let next=first.data.length;
    await Promise.all(Array.from({length:3},async()=>{while(next<first.total){const start=next;next+=chunkSize;const part=await read(start,Math.min(start+chunkSize,first.total)-1);data.set(part.data,start);stats.loaded+=part.data.length;}}));
    if(signal.aborted)return;stats.loading.end=performance.now();callbacks.onSuccess({url:context.url,data:data.buffer,code:200},stats,context,this.response);
   })().catch(error=>{if(!signal.aborted)callbacks.onError({code:0,text:error.message},context,this.response,stats);});
  }
  abort(){this.stats.aborted=true;this.controller?.abort();}
  destroy(){this.abort();this.context=null;}
  getCacheAge(){return null;}
  getResponseHeader(name){return this.response?.headers.get(name)||null;}
 };
}
