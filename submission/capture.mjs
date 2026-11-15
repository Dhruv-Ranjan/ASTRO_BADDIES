import {spawn} from 'node:child_process';
import fs from 'node:fs/promises';
const chrome='C:/Program Files/Google/Chrome/Application/chrome.exe';
const profile='G:/CubeSat/ASTRO_BADDIES/submission/chrome-profile-cdp';
const proc=spawn(chrome,['--headless','--no-sandbox','--disable-gpu','--hide-scrollbars','--remote-debugging-port=9333',`--user-data-dir=${profile}`,'--window-size=1600,1000','https://astrobaddies.netlify.app/mission-control.html'],{stdio:'ignore',windowsHide:true});
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
let targets=[];for(let i=0;i<40;i++){try{targets=await(await fetch('http://127.0.0.1:9333/json/list')).json();if(targets.some(x=>x.type==='page'))break}catch{}await sleep(500)}
const page=targets.find(x=>x.type==='page');if(!page)throw Error('Chrome page target unavailable');
const ws=new WebSocket(page.webSocketDebuggerUrl);await new Promise((resolve,reject)=>{ws.onopen=resolve;ws.onerror=reject});let nextId=0;const pending=new Map();ws.onmessage=e=>{const m=JSON.parse(e.data);if(m.id&&pending.has(m.id)){const p=pending.get(m.id);pending.delete(m.id);m.error?p.reject(m.error):p.resolve(m.result)}};
const call=(method,params={})=>new Promise((resolve,reject)=>{const id=++nextId;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}))});
await call('Page.enable');await call('Runtime.enable');await sleep(7000);
await call('Runtime.evaluate',{expression:"document.querySelector('[data-view=\\\"mission\\\"]').click()"});await sleep(500);
await call('Runtime.evaluate',{expression:"document.getElementById('run-plan').click()"});await sleep(1200);
const shot=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});await fs.writeFile('G:/CubeSat/ASTRO_BADDIES/submission/screenshots/04-planner-run.png',Buffer.from(shot.data,'base64'));
ws.close();proc.kill();
