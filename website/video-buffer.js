// Small web playback copies are fetched once, with bounded concurrency. Playing
// a completed Blob avoids network stalls and repeated downloads on every loop.
const pending=new Map(),queue=[];let active=0;
function pump(){
  while(active<2&&queue.length){
    const job=queue.shift();active++;
    (async()=>{
      for(let attempt=0;attempt<3;attempt++){
        try{
          const response=await fetch(job.url,{credentials:'same-origin'});
          if(!response.ok)throw new Error(`Video request failed (${response.status})`);
          return URL.createObjectURL(await response.blob());
        }catch(error){if(attempt===2)throw error;}
      }
    })().then(job.resolve,error=>{pending.delete(job.url);job.reject(error);}).finally(()=>{active--;pump();});
  }
}
export function bufferedVideo(url){
  if(!pending.has(url))pending.set(url,new Promise((resolve,reject)=>{queue.push({url,resolve,reject});pump();}));
  return pending.get(url);
}
