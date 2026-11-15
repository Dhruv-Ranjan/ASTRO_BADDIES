import {spawn} from 'node:child_process';
import fs from 'node:fs/promises';
const chrome='C:/Program Files/Google/Chrome/Application/chrome.exe';
const profile='C:/Users/ranja/AppData/Local/Temp/aurora-sr-demo-render';
const out='G:/CubeSat/ASTRO_BADDIES/submission/screenshots/2026-laptop/rendered';
await fs.mkdir(out,{recursive:true});
const proc=spawn(chrome,['--headless','--no-sandbox','--disable-gpu','--hide-scrollbars','--remote-debugging-port=9341',`--user-data-dir=${profile}`,'--window-size=1280,800','file:///G:/CubeSat/ASTRO_BADDIES/submission/demo-slides.html'],{stdio:'ignore',windowsHide:true});
const sleep=ms=>new Promise(r=>setTimeout(r,ms));let targets=[];
for(let i=0;i<40;i++){try{targets=await(await fetch('http://127.0.0.1:9341/json/list')).json();if(targets.some(t=>t.type==='page'))break}catch{}await sleep(300)}
const page=targets.find(t=>t.type==='page');if(!page){proc.kill();throw Error('Chrome page target unavailable')}
const ws=new WebSocket(page.webSocketDebuggerUrl);await new Promise((resolve,reject)=>{ws.onopen=resolve;ws.onerror=reject});let id=0;const pending=new Map();ws.onmessage=e=>{const m=JSON.parse(e.data);if(m.id&&pending.has(m.id)){const q=pending.get(m.id);pending.delete(m.id);m.error?q.reject(m.error):q.resolve(m.result)}};
const call=(method,params={})=>new Promise((resolve,reject)=>{const n=++id;pending.set(n,{resolve,reject});ws.send(JSON.stringify({id:n,method,params}))});
await call('Page.enable');await call('Emulation.setDeviceMetricsOverride',{width:1280,height:800,deviceScaleFactor:1,mobile:false});await sleep(900);
for(let i=0;i<7;i++){await call('Runtime.evaluate',{expression:`window.scrollTo(0,${i*800})`});await sleep(250);const shot=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});await fs.writeFile(`${out}/slide-${String(i+1).padStart(2,'0')}.png`,Buffer.from(shot.data,'base64'))}
ws.close();proc.kill();console.log('Rendered 7 16:10 slide previews to '+out);
