'use strict';
const $ = id => document.getElementById(id);
const {rate,hz,signal,features} = MelDSP;
const margin = 52, right = 25;
const db = (p,ref) => 10*Math.log10(Math.max(1e-12,p/ref));
const colors = [[21,25,46],[41,73,107],[22,135,128],[245,211,101]];
let audio, playing, x, data, cache;
const hints = {
  mix:'Start at the hum, then move the window into the hiss. Which frequencies light up?',
  tones:'Compare 10 ms and 60 ms: can you distinguish the two tones?',
  chirp:'Move right. Does the strongest frequency rise?',
  pulse:'Compare 10 ms and 60 ms: which window shows the click in more frames?'
};
function heatmap(rows,ref) {
  const c=document.createElement('canvas');
  c.width=rows.length; c.height=rows[0].length;
  const g=c.getContext('2d'), pixels=g.createImageData(c.width,c.height);
  rows.forEach((row,i)=>row.forEach((p,j)=>{
    const v=3*Math.max(0,Math.min(1,(db(p,ref)+65)/65));
    const k=Math.min(2,Math.floor(v)), f=v-k, a=4*((c.height-1-j)*c.width+i);
    for(let channel=0;channel<3;channel++)
      pixels.data[a+channel]=colors[k][channel]*(1-f)+colors[k+1][channel]*f;
    pixels.data[a+3]=255;
  }));
  g.putImageData(pixels,0,0);
  return c;
}
function rebuild(reset=false) {
  if(reset) {
    x=signal($('sound').value);
    $('hint').textContent=hints[$('sound').value];
    if(playing) playing.stop();
  }
  const oldTime=data ? +$('frame').value*data.hop/rate : 0;
  data=features(x,+$('size').value,+$('hop').value,+$('bands').value);
  cache=[heatmap(data.frames,data.linearPeak),heatmap(data.mel,data.melPeak)];
  $('frame').max=data.frames.length-1;
  $('frame').value=Math.min(data.frames.length-1,Math.round((reset ? ($('sound').value==='pulse'?1.2:.4) : oldTime)*rate/data.hop));
  $('band').max=data.bank.length;
  $('shape').textContent=`${data.frames.length} frames × ${data.bank.length} mel bands.`;
  draw();
}
function chart(id) {
  const c=$(id), w=c.clientWidth, h=c.clientHeight, d=window.devicePixelRatio||1;
  c.width=Math.round(w*d); c.height=Math.round(h*d);
  const g=c.getContext('2d'); g.scale(d,d);
  const s=getComputedStyle(c);
  g.fillStyle=g.strokeStyle=s.color;
  g.font='12px system-ui'; g.textBaseline='middle';
  return {g,w:w-margin-right,h:h-27,accent:s.getPropertyValue('--accent').trim()};
}
function axes(g,w,h,xTicks,yTicks) {
  const labelColor = g.fillStyle;
  g.textAlign='right';
  yTicks.forEach(([f,label])=>g.fillText(label,margin-6,Math.max(8,Math.min(h-4,(1-f)*h))));
  g.textAlign='center';
  xTicks.forEach(([f,label])=>g.fillText(label,margin+f*w,h+15));
  g.strokeStyle=labelColor;
  g.beginPath(); g.moveTo(margin,0); g.lineTo(margin,h); g.lineTo(margin+w,h); g.stroke();
}
const timeTicks=[[0,'0 s'],[.25,'0.6'],[.5,'1.2'],[.75,'1.8'],[1,'2.4 s']];
function cursor(g,at,h,color) {
  g.strokeStyle=color; g.beginPath(); g.moveTo(at,0); g.lineTo(at,h); g.stroke();
}
function draw() {
  const i=+$('frame').value, j=+$('band').value-1, center=i*data.hop;
  const {lo,mid,hi,bins}=data.bank[j], frame=data.frames[i];
  $('frameInfo').textContent=`Frame ${i+1} of ${data.frames.length} · ${(center/rate).toFixed(3)} s · ${Math.max(0,100*(1-data.hop/data.width)).toFixed(0)}% overlap between windows`;
  const p=data.mel[i][j], level=p>0?`${db(p,data.melPeak).toFixed(1)} dB`:'no energy';
  $('bandInfo').textContent=`Band ${j+1} · ${Math.round(lo)}–${Math.round(hi)} Hz (center ${Math.round(mid)} Hz) · ${bins.length} weighted bins → ${p.toPrecision(3)} power → ${level}`;
  const largest=bins.reduce((top,[k,w])=>frame[k]*w>top.value ? {k,w,value:frame[k]*w} : top,{value:0});
  $('contribution').textContent=largest.value>0
    ? `Largest contribution: ${hz(largest.k).toFixed(1)} Hz bin · ${frame[largest.k].toPrecision(3)} power × ${largest.w.toFixed(3)} weight = ${largest.value.toPrecision(3)} (${(100*largest.value/p).toFixed(0)}% of the band’s total).`
    : 'None of the bins in this band have energy in this frame.';
  {
    const {g,w,h,accent}=chart('wave');
    const l=Math.max(0,center-data.width/2), r=Math.min(x.length,center+data.width/2);
    g.save(); g.fillStyle=accent; g.globalAlpha=.22; g.fillRect(margin+l*w/x.length,0,(r-l)*w/x.length,h); g.restore();
    g.strokeStyle=accent; g.beginPath();
    for(let a=0;a<w;a++){
      const start=Math.floor(a*x.length/w), end=Math.floor((a+1)*x.length/w);
      let min=x[start],max=min;
      for(let q=start+1;q<end;q++){min=Math.min(min,x[q]); max=Math.max(max,x[q]);}
      g.moveTo(margin+a,h*(1-max)/2);g.lineTo(margin+a,h*(1-min)/2);
    }
    g.stroke(); cursor(g,margin+center*w/x.length,h,accent);
    axes(g,w,h,timeTicks,[[0,'−1'],[.5,'0'],[1,'+1']]);
  }
  ['linear','mel'].forEach((id,n)=>{
    const {g,w,h}=chart(id);
    const step=w*data.hop/x.length, at=margin+i*step;
    // Frame centers align with waveform centers; the first half-cell is clipped.
    g.save();g.beginPath();g.rect(margin,0,w,h);g.clip();g.imageSmoothingEnabled=false;
    g.drawImage(cache[n],margin-step/2,0,step*data.frames.length,h);
    cursor(g,at,h,'#fff');
    if(n){
      const bandHeight=h/data.bank.length;
      g.strokeStyle='#fff';g.lineWidth=1.5;
      g.strokeRect(at-Math.max(3,step)/2,h-(j+1)*bandHeight,Math.max(3,step),bandHeight);
    }
    g.restore();
    axes(g,w,h,timeTicks,n?[[0,'1'],[.5,String(Math.round(data.bank.length/2))],[1,String(data.bank.length)]]:[[0,'0 Hz'],[.5,'4k'],[1,'8k Hz']]);
  });
  {
    const {g,w,h,accent}=chart('detail');
    const span=Math.max(160,hi-lo), left=Math.max(0,lo-span*.35), end=Math.min(rate/2,hi+span*.35);
    const px=f=>margin+w*(f-left)/(end-left);
    g.save();g.beginPath();g.rect(margin,0,w,h);g.clip();
    g.strokeStyle=accent; g.beginPath();
    frame.forEach((p,k)=>{
      const y=h*(1-Math.max(0,(db(p,data.linearPeak)+65)/65));
      if(k) g.lineTo(px(hz(k)),y); else g.moveTo(px(0),y);
    });
    g.stroke();
    g.strokeStyle=g.fillStyle;g.setLineDash([5,3]);g.beginPath();
    g.moveTo(px(lo),h);g.lineTo(px(mid),4);g.lineTo(px(hi),h);g.stroke();g.setLineDash([]);
    g.restore();
    axes(g,w,h,[[0,`${Math.round(left)} Hz`],[1,`${Math.round(end)} Hz`]],[[0,'−65'],[.5,'−33'],[1,'0 dB']]);
  }
}
$('reset').onclick=()=>{
  for(const [id,value] of Object.entries({sound:'mix',size:25,hop:10,bands:80,band:12})) $(id).value=value;
  rebuild(true);
};
$('sound').onchange=()=>rebuild(true);
for(const id of ['size','hop','bands']) $(id).onchange=()=>rebuild();
for(const id of ['frame','band']) $(id).oninput=draw;
for(const id of ['wave','linear','mel']) $(id).onclick=e=>{
  const box=$(id).getBoundingClientRect();
  $('frame').value=Math.round((e.clientX-box.left-margin)/(box.width-margin-right)*x.length/data.hop);
  if(id==='mel') $('band').value=1+Math.floor((1-(e.clientY-box.top)/(box.height-27))*data.bank.length);
  draw();
};
$('play').onclick=async()=>{
  if(playing){playing.stop();return;}
  try {
    audio ??= new AudioContext();
    await audio.resume();
    if(playing)return;
    const b=audio.createBuffer(1,x.length,rate); b.copyToChannel(x,0);
    const s=audio.createBufferSource(); s.buffer=b; s.connect(audio.destination); playing=s;
    s.onended=()=>{if(playing===s){playing=null;$('play').textContent='Play sound';}};
    s.start();$('play').textContent='Stop sound';
  } catch {
    playing=null;$('play').textContent='Play sound';
    $('hint').textContent='Audio playback is unavailable in this browser. The plots and controls still work.';
  }
};
rebuild(true);
new ResizeObserver(draw).observe($('wave'));
