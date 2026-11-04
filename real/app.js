import {MEDIA,CINEMATIC,HERO_CLIPS} from './media.js?v=23';
import {mountCinematic} from './cinematic.js?v=26';
import {mountStructure} from './structure.js';
import {mountResults} from './results.js?v=27';
import {mountGallery} from './gallery.js?v=27';
const $=(selector,root=document)=>root.querySelector(selector);
const $$=(selector,root=document)=>[...root.querySelectorAll(selector)];
mountResults($('#results-visualization'));
const gallery=mountGallery($('#performance'),MEDIA);
const cinema=mountCinematic($('#cinematic'),CINEMATIC,HERO_CLIPS);
mountStructure($('#structure-diagram'));

const sectionIds=['abstract','structure','performance','real-world-charts','result-improvement','result-benchmark','other-findings'];
const contentsButton=$('.contents-toggle');
function closeContents(){document.body.classList.remove('contents-open');contentsButton.setAttribute('aria-expanded','false');contentsButton.setAttribute('aria-label','Open contents');}
contentsButton.addEventListener('click',()=>{const open=document.body.classList.toggle('contents-open');contentsButton.setAttribute('aria-expanded',String(open));contentsButton.setAttribute('aria-label',open?'Close contents':'Open contents');});
$$('.section-nav a').forEach(link=>link.addEventListener('click',closeContents));
document.addEventListener('keydown',event=>{if(event.key==='Escape')closeContents();});
let scrollTick=false;
function updateScroll(){
  const shown=$('#paper').getBoundingClientRect().top<innerHeight*.65;
  document.body.classList.toggle('article-in-view',shown);if(!shown)closeContents();
  const current=sectionIds.filter(id=>document.getElementById(id).getBoundingClientRect().top<innerHeight*.3).at(-1)||'abstract';
  $$('.section-nav>div a').forEach(link=>{const active=link.hash===`#${current}`;link.classList.toggle('active',active);if(active)link.setAttribute('aria-current','location');else link.removeAttribute('aria-current');});
  scrollTick=false;
}
addEventListener('scroll',()=>{if(!scrollTick){scrollTick=true;requestAnimationFrame(updateScroll);}},{passive:true});updateScroll();
const reveals=[$('#paper'),$('#abstract'),$('#structure'),$('#result>h2'),...$$('.rs-story'),$('#performance')];
const revealObserver=new IntersectionObserver(entries=>entries.forEach(entry=>{if(entry.isIntersecting){entry.target.classList.add('is-revealed');revealObserver.unobserve(entry.target);}}),{threshold:.06});
reveals.forEach(section=>{section.classList.add('reveal-section');revealObserver.observe(section);});
document.body.classList.add('motion-ready');
addEventListener('pagehide',()=>{gallery.pause();cinema.pause();});
addEventListener('pageshow',event=>{if(event.persisted){gallery.resume();cinema.resume();}});

// Signal after every layout-producing mount, without waiting for video downloads.
window.dispatchEvent(new Event('pyrua:ready'));
