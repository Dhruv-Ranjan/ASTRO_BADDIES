import {spawn} from 'node:child_process';
import fs from 'node:fs/promises';
const root='G:/CubeSat/ASTRO_BADDIES';
const chrome='C:/Program Files/Google/Chrome/Application/chrome.exe';
const out=root+'/submission/screenshots/2026-laptop';
await fs.mkdir(out,{recursive:true});
const server=spawn('python',['-m','http.server','8766','--directory',root+'/netlify'],{stdio:'ignore',windowsHide:true});
const proc=spawn(chrome,['--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check','--hide-scrollbars','--remote-debugging-port=9343','--user-data-dir=C:/Users/ranja/AppData/Local/Temp/aurora-laptop-capture','--window-size=1440,900','http://127.0.0.1:8766/'],{stdio:'ignore',windowsHide:true});
const sleep=ms=>new Promise(r=>setTimeout(r,ms)); let targets=[];
for(let i=0;i<80;i++){try{targets=await(await fetch('http://127.0.0.1:9343/json/list')).json();if(targets.some(t=>t.type==='page'))break}catch{}await sleep(250)}
const page=targets.find(t=>t.type==='page');if(!page){proc.kill();server.kill();throw Error('Chrome page target unavailable')}
const ws=new WebSocket(page.webSocketDebuggerUrl);await new Promise((resolve,reject)=>{ws.onopen=resolve;ws.onerror=reject});let id=0;const pending=new Map();ws.onmessage=e=>{const m=JSON.parse(e.data);if(m.id&&pending.has(m.id)){const q=pending.get(m.id);pending.delete(m.id);m.error?q.reject(m.error):q.resolve(m.result)}};
const call=(method,params={})=>new Promise((resolve,reject)=>{const n=++id;pending.set(n,{resolve,reject});ws.send(JSON.stringify({id:n,method,params}))});
await call('Page.enable');await call('Emulation.setDeviceMetricsOverride',{width:1440,height:900,deviceScaleFactor:1,mobile:false});
for(const [file,url] of [['trend-lab-1440x900.png','http://127.0.0.1:8766/'],['spectral-lab-1440x900.png','http://127.0.0.1:8766/spectral-lab.html'],['research-console-1440x900.png','http://127.0.0.1:8766/mission-control.html'],['research-figures-1440x900.png','http://127.0.0.1:8766/graphs.html']]){
 await call('Page.navigate',{url});await sleep(2400);const dimensions=await call('Runtime.evaluate',{expression:'JSON.stringify({w:innerWidth,h:innerHeight,dpr:devicePixelRatio})',returnByValue:true});const shot=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});await fs.writeFile(out+'/'+file,Buffer.from(shot.data,'base64'));console.log(file,dimensions.result.value);
}
ws.close();proc.kill();server.kill();
