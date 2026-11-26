'use strict';
const $ = id => document.getElementById(id);
const {vocab, corpora, runs, rate} = WORD2VEC_DATA;
const id = w => vocab.indexOf(w);
const names = {cbow:'CBOW',skip:'Skip-gram'};
const defaults = {mode:'cbow',radius:1,variant:'base',sentence:0,center:1,step:9,query:'dog',stage:'lookup',pair:0,score:null};
const state = {...defaults};
const fmt = x => Number(x.toPrecision(9)).toString();
const dot = (a,b) => a.reduce((n,x,i)=>n+x*b[i],0);
const cosine = (a,b) => {
  const norm = Math.sqrt(dot(a,a)*dot(b,b));
  return norm ? dot(a,b)/norm : 0;
};
const vec = a => '['+a.map(fmt).join(', ')+']';
const opts = (el, pairs, selection) => {
  el.replaceChildren(...pairs.map(([value,label])=>new Option(label,value)));
  el.value=selection;
};
const table = (name,head,rows, active=-1) => '<div class="numtable" tabindex="0" role="region" aria-label="'+name+'"><table><caption>'+name+'</caption><thead><tr>'+head.map(x=>'<th scope="col">'+x+'</th>').join('')+'</tr></thead><tbody>'+rows.map((r,i)=>'<tr'+(i===active?' data-active':'')+'>'+r.map((x,j)=>j===0?'<th scope="row">'+x+'</th>':'<td>'+x+'</td>').join('')+'</tr>').join('')+'</tbody></table></div>';
const equation = txt => '<p class="formula">'+txt+'</p>';
const checkpoint = () => runs[state.variant][state.mode][state.radius][state.step];
$('step').max = String(runs.base.cbow[1].length - 1);
const words = () => corpora[state.variant][state.sentence].split(' ');
const context = (row,i) => row.slice(Math.max(0,i-state.radius),i).concat(row.slice(i+1,i+state.radius+1));
const fixed = Object.values(runs).flatMap(v=>Object.values(v)).flatMap(v=>Object.values(v)).flat();
const extent = Math.ceil(Math.max(...fixed.flatMap(c=>c.E.flatMap(row=>row.map(Math.abs)))));
let pairs = [];

function renderPairs(focus=false) {
  const row=words(), near=context(row,state.center);
  const center=row[state.center];
  pairs=state.mode==='cbow'?[{x:near,y:center}]:near.map(y=>({x:[center],y}));
  state.pair=Math.min(state.pair,pairs.length-1);
  $('tokens').replaceChildren(...row.map((w,i)=>{
    const b=document.createElement('button');
    b.type='button'; b.dataset.token=i; b.textContent=w;
    const chosen=i===state.center, beside=!chosen&&Math.abs(i-state.center)<=state.radius;
    b.className='control token '+(chosen?'center':beside?'context':'');
    b.setAttribute('aria-pressed',String(chosen));
    b.setAttribute('aria-label',w+' at position '+(i+1)+(chosen?', center':beside?', context':''));
    return b;
  }));
  if(focus) $('tokens').children[state.center].focus();
  $('pairs').innerHTML=pairs.map(({x,y})=>'<div class="pair"><span class="clue">Input: '+x.join(' + ')+'</span><span aria-hidden="true">→</span><span class="target">Target: '+y+'</span></div>').join('');
  $('rule').textContent=state.mode==='cbow'
    ? 'CBOW averages the input vectors. Swapping the context words gives the same average.'
    : 'Skip-gram uses one center vector. Each neighbor gets its own training pair, using the same distribution over words.';
  opts($('pair'),pairs.map(({x,y},i)=>[i,x.join(' + ')+' → '+y]),state.pair);
}

function countRows(w) {
  const counts=vocab.map(()=>0);
  for(const s of corpora[state.variant]){
    const row=s.split(' ');
    row.forEach((token,i)=>{
      if(token===w) context(row,i).forEach(x=>counts[id(x)]++);
    });
  }
  return counts;
}
function ranked(q, rows) {
  return vocab.map((w,i)=>[w,cosine(rows[q],rows[i])]).filter(([w])=>w!==vocab[q])
    .sort((a,b)=>b[1]-a[1]||a[0].localeCompare(b[0]));
}
function renderNeighbors() {
  const c=checkpoint(), q=id(state.query);
  const plot=$('plot');
  const at=x=>160+x*103/extent, up=y=>124-y*103/extent;
  plot.innerHTML='<title>Input embeddings after '+c.step+' updates; selected word '+state.query+'</title>'+
    '<line x1="57" y1="124" x2="263" y2="124"/><line x1="160" y1="21" x2="160" y2="227"/>'+
    '<g fill="#535d70"><text x="57" y="244">−'+extent+'</text><text x="250" y="244">+'+extent+'</text><text x="151" y="244">x₁</text><text x="166" y="18">x₂ · +'+extent+'</text><text x="166" y="225">−'+extent+'</text></g>'+
    c.E.map(([x,y],i)=>i===q?'':'<circle cx="'+at(x)+'" cy="'+up(y)+'" r="4" fill="#8094cb"><title>'+vocab[i]+' '+vec([x,y])+'</title></circle>').join('')+
    '<circle cx="'+at(c.E[q][0])+'" cy="'+up(c.E[q][1])+'" r="6" fill="#a85b11"/><text x="'+(at(c.E[q][0])+9)+'" y="'+(up(c.E[q][1])+4)+'" fill="#663c0d">'+state.query+'</text>';
  const counts=vocab.map(countRows);
  const hits={};
  const cue=['the',state.query];
  for(const s of corpora[state.variant]){
    const row=s.split(' ');
    for(let i=0;i<row.length-2;i++) if(row[i]===cue[0]&&row[i+1]===cue[1]) hits[row[i+2]]=(hits[row[i+2]]||0)+1;
  }
  const raw=vocab.map((w,i)=>counts[q][i]?w+' × '+counts[q][i]:'').filter(Boolean);
  $('counts').innerHTML='<p><strong>Exact trigram:</strong> after “'+cue.join(' ')+'”, '+(Object.entries(hits).map(([w,n])=>w+' × '+n).join(', ')||'no matches')+'.</p>'+
    '<p><strong>Context counts:</strong> neighbors of “'+state.query+'” are '+(raw.join(', ')||'unseen')+'.</p>';
  const counted=ranked(q,counts), learned=ranked(q,c.E);
  const scored=(rows,w)=>rows.find(r=>r[0]===w)[1].toFixed(3);
  $('neighbors').innerHTML=table('Cosine similarity to '+state.query,['Word','Context counts','Learned vectors'],
    [...new Set([...learned.slice(0,3).map(([w])=>w),...['cat','dog','bird'].filter(w=>w!==state.query)])].map(w=>[w,scored(counted,w),scored(learned,w)]))+
    '<p class="note">Closest by context counts: '+counted.slice(0,3).map(([w])=>w).join(', ')+
    '. Selected input vector: '+vec(c.E[q])+'. Zero-length count vectors are scored as 0.</p>';
  $('step-label').textContent=c.step+(c.step===1?' update':' updates');
  $('step').setAttribute('aria-valuetext',$('step-label').textContent);
  $('training').textContent=names[state.mode]+' · radius '+state.radius+' · mean training loss '+c.loss.toFixed(4)+' nats';
}

function mathTrace() {
  const c=checkpoint(), {E,U}=c, {x,y}=pairs[state.pair], ids=x.map(id), target=id(y);
  const isRecorded = JSON.stringify(c.next_pair)===JSON.stringify([ids,target]);
  $('recorded').disabled=!c.next_pair;
  const score=id(state.score||y);
  const h=[0,1].map(d=>ids.reduce((a,i)=>a+E[i][d],0)/ids.length);
  const z=U.map(row=>dot(row,h)), max=Math.max(...z);
  const terms=z.map(v=>Math.exp(v-max)), denom=terms.reduce((a,b)=>a+b,0);
  const p=terms.map(v=>v/denom), g=p.map((v,i)=>v-(i===target?1:0));
  $('score').value=vocab[score];
  document.querySelectorAll('[data-stage]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.stage===state.stage)));
  $('weight-controls').hidden=state.stage!=='update';
  $('update').textContent='';
  let markup='<p class="note">Inputs: '+x.join(' + ')+' · target: '+y+' · '+checkpoint().step+' updates. Values rounded to 9 significant digits. On narrow screens, scroll tables sideways.'+(isRecorded?' This pair was used for the next recorded update.':'')+'</p>';
  if(state.stage==='lookup'){
    markup+=equation('h = (sum of selected rows E) / '+ids.length+' = '+vec(h))+
      table('Input table E ('+vocab.length+' × 2)',['Word (ID)','Feature 1','Feature 2'],vocab.map((w,i)=>[w+' ('+i+')',...E[i].map(fmt)]))+
      '<p>Rows used, in order: '+ids.join(', ')+'. An input word contributes once per occurrence.</p>';
  }else if(state.stage==='score'){
    markup+=equation('z['+vocab[score]+'] = h · U['+vocab[score]+'] = ('+fmt(h[0])+' × '+fmt(U[score][0])+') + ('+fmt(h[1])+' × '+fmt(U[score][1])+') = '+fmt(z[score]))+
      table('Output table U ('+vocab.length+' × 2)',['Word','Feature 1','Feature 2','Logit'],vocab.map((w,i)=>[w,...U[i].map(fmt),fmt(z[i])]),score);
  }else if(state.stage==='loss'){
    markup+=equation('m = max(z) = '+fmt(max)+'; p[i] = exp(z[i] − m) / '+fmt(denom))+
      equation('Loss = −log p['+y+'] = −log('+fmt(p[target])+') = '+fmt(max+Math.log(denom)-z[target])+' nats')+
      table('All terms in the softmax denominator',['Word','Logit','exp(z − m)','Probability'],vocab.map((w,i)=>[w,fmt(z[i]),fmt(terms[i]),fmt(p[i])]),score);
  }else{
    const w=id($('weight').value), d=+$('feature').value, isU=$('table').value==='U';
    const old=(isU?U:E)[w][d], factor=ids.filter(i=>i===w).length/ids.length;
    const summands=g.map((v,i)=>v*U[i][d]);
    const dh=summands.reduce((a,b)=>a+b,0);
    const grad=isU?g[w]*h[d]:factor*dh;
    markup+=equation('g[i] = p[i] − 1[i = target]')+
      table('Backpropagation operands',['Word','p[i]','g[i]','U[i,feature]','g[i] × U[i,feature]'],vocab.map((v,i)=>[v,fmt(p[i]),fmt(g[i]),fmt(U[i][d]),fmt(summands[i])]),w);
    $('update').innerHTML=equation(isU
      ? '∂L/∂U['+vocab[w]+','+(d+1)+'] = g['+vocab[w]+'] × h['+(d+1)+'] = '+fmt(g[w])+' × '+fmt(h[d])+' = '+fmt(grad)
      : '∂L/∂E['+vocab[w]+','+(d+1)+'] = (occurrences / inputs) × Σᵢ g[i]U[i,'+(d+1)+'] = '+fmt(factor)+' × '+fmt(dh)+' = '+fmt(grad))+
      equation('New weight = old − rate × gradient = '+fmt(old)+' − '+rate+' × ('+fmt(grad)+') = '+fmt(old-rate*grad));
  }
  $('operands').innerHTML=markup;
}

function render(rebuild=false,focus=false) {
  if(rebuild) {
    opts($('sentence'),corpora[state.variant].map((s,i)=>[i,s]),state.sentence);
    $('corpus-list').replaceChildren(...corpora[state.variant].map(s=>Object.assign(document.createElement('li'),{textContent:s})));
  }
  $('cbow').setAttribute('aria-pressed',String(state.mode==='cbow'));
  $('skip').setAttribute('aria-pressed',String(state.mode==='skip'));
  renderPairs(focus); renderNeighbors(); mathTrace();
}
for(const field of ['query','weight','score']) opts($(field),vocab.map(w=>[w,w]),field==='query'?'dog':'cat');
for(const m of ['cbow','skip']) $(m).onclick=()=>{state.mode=m;state.pair=0;state.score=null;render();};
$('tokens').onclick=e=>{const b=e.target.closest('[data-token]');if(b){state.center=+b.dataset.token;state.pair=0;state.score=null;render(false,true);}};
for(const field of ['variant','sentence','radius','query','step','pair','score']){
  $(field).addEventListener(field==='step'?'input':'change',()=>{
    state[field]=['variant','query','score'].includes(field)?$(field).value:+$(field).value;
    if(['variant','sentence','radius'].includes(field)){state.pair=0;state.score=null;}
    if(field==='pair')state.score=null;
    if(['step','query'].includes(field)){renderNeighbors();mathTrace();}
    else if(['pair','score'].includes(field))mathTrace();
    else render(field==='variant');
  });
}
for(const field of ['table','weight','feature']) $(field).onchange=mathTrace;
$('stages').onclick=e=>{const b=e.target.closest('[data-stage]');if(b){state.stage=b.dataset.stage;mathTrace();}};
$('recorded').onclick=()=>{
  const pair=checkpoint().next_pair;
  if(!pair)return;
  for(const [sentence,line] of corpora[state.variant].entries()){
    const row=line.split(' ');
    for(const [center,word] of row.entries()){
      const near=context(row,center);
      const candidates=state.mode==='cbow'?[[near.map(id),id(word)]]:near.map(w=>[[id(word)],id(w)]);
      const index=candidates.findIndex(x=>JSON.stringify(x)===JSON.stringify(pair));
      if(index>=0){
        Object.assign(state,{sentence,center,pair:index,score:null});
        $('weight').value=$('table').value==='E'?vocab[pair[0][0]]:vocab[pair[1]];
        render(true);
        return;
      }
    }
  }
};
$('reset').onclick=()=>{
  Object.assign(state,defaults);
  for(const field of ['variant','radius','query','step']) $(field).value=state[field];
  $('table').value='U';$('weight').value='cat';$('feature').value='0';
  render(true);
};
render(true);

// The guided probe is fixed to the original initialization. Its optional SGD
// update uses a copy; it cannot mutate the saved training trajectory.
(() => {
  const {E,U}=runs.base.cbow[1][0], inputs=['the','eats'].map(id), target=id('cat');
  const forward=(e,u)=>{
    const h=[0,1].map(d=>inputs.reduce((sum,i)=>sum+e[i][d],0)/inputs.length);
    const z=u.map(row=>dot(row,h)), max=Math.max(...z);
    const terms=z.map(value=>Math.exp(value-max)), denom=terms.reduce((a,b)=>a+b);
    return {h,z,p:terms.map(value=>value/denom),max,terms,denom};
  };
  const before=forward(E,U);
  const errors=before.p.map((x,i)=>x-Number(i===target));
  const dh=[0,1].map(d=>dot(errors,U.map(row=>row[d])));
  const updatedE=E.map((row,i)=>row.map((v,d)=>v-rate*inputs.filter(x=>x===i).length/inputs.length*dh[d]));
  const updatedU=U.map((row,i)=>row.map((v,d)=>v-rate*errors[i]*before.h[d]));
  const after=forward(updatedE,updatedU);
  const short=x=>x.toFixed(6);
  const percent=p=>(p*100).toFixed(3)+'%';
  $('walk-average').innerHTML=table('The two input rows and their average',['Row','Feature 1','Feature 2'],[
    ...inputs.map(i=>[vocab[i],...E[i].map(short)]),['Average',...before.h.map(short)]
  ])+equation('Clue feature 2 = ('+inputs.map(i=>short(E[i][1])).join(' + ')+') / 2 = '+short(before.h[1]));
  const renderScore=()=>{
    const w=$('walk-word').value, i=id(w);
    $('walk-quick').textContent='Before training: '+percent(before.p[i])+' for '+w;
    $('walk-score').innerHTML=equation('Score for '+w+' = ('+short(before.h[0])+' × '+short(U[i][0])+') + ('+
      short(before.h[1])+' × '+short(U[i][1])+') ≈ '+short(before.z[i]))+
      '<p>Before training: '+percent(before.p[i])+' probability for <em>'+w+'</em>.</p>';
    $('walk-probs').innerHTML=equation('To keep the exponentials stable, subtract the same max score m = '+short(before.max)+
      ' from every score. Sum exp(score − m) ≈ '+short(before.denom)+'.')+
      table('Full vocabulary',['Word','Score','exp(score − m)','Probability'],vocab.map((word,j)=>[
        word,short(before.z[j]),short(before.terms[j]),short(before.p[j])
      ]),i);
  };
  const renderStep=applied=>{
    const old=U[target][1], grad=errors[target]*before.h[1];
    $('walk-gradient').innerHTML=equation('Before the update: cat’s output weight 2 = '+short(old)+
      '; clue feature 2 = '+short(before.h[1]))+(applied?
      '<p>The weight goes down. Multiplying a more negative weight by the negative clue feature raises that part of the score.</p>'+
      equation('Gradient = (p(cat) − 1) × clue feature 2 = ('+short(before.p[target])+
        ' − 1) × '+short(before.h[1])+' ≈ '+short(grad))+
      equation('New weight ≈ '+short(old)+' − '+rate+' × '+short(grad)+' = '+short(updatedU[target][1]))+
      '<p class="note">That’s one weight. SGD updates both clue rows and all scoring rows together, using the gradients from before the step.</p>'
      :'');
    $('walk-result').textContent=applied?'Run the same example again: p(cat) rises from '+percent(before.p[target])+
      ' to '+percent(after.p[target])+'. Loss falls from '+(-Math.log(before.p[target])).toFixed(4)+' to '+
      (-Math.log(after.p[target])).toFixed(4)+' nats.':'Before the update, p(cat) = '+percent(before.p[target])+'.';
    $('walk-step').disabled=applied;
    $('walk-reset').disabled=!applied;
  };
  $('walk-word').onchange=renderScore;
  $('walk-step').onclick=()=>{renderStep(true);$('walk-reset').focus();};
  $('walk-reset').onclick=()=>{renderStep(false);$('walk-step').focus();};
  $('walk-audit').onclick=()=>{
    Object.assign(state,defaults,{step:0,stage:'update'});
    for(const field of ['variant','radius','query','step']) $(field).value=state[field];
    $('table').value='U';$('weight').value='cat';$('feature').value='1';
    render(true);$('math').open=true;
    $('math').querySelector('summary').focus({preventScroll:true});
    $('math').scrollIntoView({block:'start'});
  };
  $('reset').addEventListener('click',()=>{
    $('walk-word').value='cat';renderScore();renderStep(false);
  });
  renderScore();renderStep(false);
})();
