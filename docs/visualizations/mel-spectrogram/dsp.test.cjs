const {test} = require('node:test');
const assert = require('node:assert/strict');
const {fftSize, signal, features, windowTerm, dftTerm} = require('./dsp.js');

test('paper example: an offset and a cosine yield two nonzero bins', () => {
  const x = [3,1,-1,1,3,1,-1,1];
  const powers = Array.from({length:5}, (_, k) => {
    const terms = x.map((v,n) => dftTerm(v,n,k,x.length));
    const sum = key => terms.reduce((a,b) => a + b[key],0);
    return Math.round(sum('re') ** 2 + sum('im') ** 2);
  });
  assert.deepEqual(powers, [64,0,64,0,0]);
});

test('direct operand sums agree with actual FFT and mel output', () => {
  const x = signal('mix'), {width,hop,frames,bank,mel} = features(x,25,10,80);
  const close = (a,b) => assert.ok(Math.abs(a-b) < 1e-10 * Math.max(1, Math.abs(b)), `${a} vs ${b}`);
  assert.equal(windowTerm(x,0,width,0).padded,true);
  assert.equal(windowTerm(x,6400,width,width).value,0);
  for (const t of [0,40,120,239]) {
    for (const k of [0,14,256,512]) {
      let re = 0, im = 0;
      for (let n=0; n<fftSize; n++) {
        const {value} = windowTerm(x,t*hop,width,n);
        const term = dftTerm(value,n,k,fftSize);
        re += term.re;
        im += term.im;
      }
      close(re*re+im*im,frames[t][k]);
    }
    for (const b of [0,11,79]) {
      const {lo,mid,hi,bins} = bank[b];
      const sum = bins.reduce((s,[k]) => {
        const hz = k * 16000 / 1024;
        return s + frames[t][k] * Math.max(0,Math.min((hz-lo)/(mid-lo),(hi-hz)/(hi-mid)));
      },0);
      close(sum,mel[t][b]);
    }
  }
});
