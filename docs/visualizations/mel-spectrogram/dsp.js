// Numerical core shared by the browser and the Node tests.
const MelDSP = (() => {
  const rate = 16000, length = 38400, fftSize = 1024;
  const hzToMel = hz => 2595 * Math.log10(1 + hz / 700);
  const melToHz = m => 700 * (10 ** (m / 2595) - 1);
  const hz = k => k * rate / fftSize;

  function signal(kind) {
    let rng = 7, previous = 0;
    const sine = cycles => Math.sin(2 * Math.PI * cycles);
    const fade = (t, duration) => Math.max(0, Math.min(1, t / .025, (duration - t) / .025));
    return Float32Array.from({length}, (_, i) => {
      const t = i / rate;
      if (kind === 'pulse') return .7 * Math.exp(-(((t - 1.2) / .00016) ** 2) / 2);
      if (kind === 'tones') return fade(t, 2.4) * (.26 * sine(440*t) + .26 * sine(510*t));
      if (kind === 'chirp') return fade(t, 2.4) * .45 * sine(220*t + 750*t*t);
      if (t < .8) return fade(t,.8) * (.38*sine(220*t)+.16*sine(440*t)+.11*sine(660*t)+.06*sine(880*t));
      if (t < 1.6) {
        rng = (Math.imul(rng,1664525) + 1013904223) >>> 0;
        const noise = rng / 2**31 - 1, high = noise - previous;
        previous = noise;
        return fade(t-.8,.8) * .23 * high;
      }
      const u = t - 1.6;
      return fade(u,.8) * .45 * sine(320*u + 1700*u*u);
    });
  }

  function power(x, center, width) {
    const re = new Float64Array(fftSize), im = new Float64Array(fftSize);
    for (let i=0; i<width; i++)
      re[i] = (x[center - Math.floor(width/2) + i] ?? 0) * (.5 - .5*Math.cos(2*Math.PI*i/(width-1)));
    for (let i=1,j=0; i<fftSize; i++) {
      let bit = fftSize >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i<j) [re[i],re[j]] = [re[j],re[i]];
    }
    for (let n=2; n<=fftSize; n*=2) {
      for (let j=0; j<n/2; j++) {
        const c=Math.cos(-2*Math.PI*j/n), s=Math.sin(-2*Math.PI*j/n);
        for (let i=j; i<fftSize; i+=n) {
          const k=i+n/2, a=re[k]*c-im[k]*s, b=re[k]*s+im[k]*c;
          re[k]=re[i]-a; im[k]=im[i]-b; re[i]+=a; im[i]+=b;
        }
      }
    }
    return re.slice(0,fftSize/2+1).map((v,k)=>v*v+im[k]*im[k]);
  }

  function filters(count) {
    const edges = Array.from({length:count+2}, (_, j)=>melToHz(j*hzToMel(rate/2)/(count+1)));
    return edges.slice(0,count).map((lo,j) => {
      const mid=edges[j+1], hi=edges[j+2], bins=[];
      for (let k=0;k<=fftSize/2;k++) {
        const f=hz(k), w=Math.min((f-lo)/(mid-lo),(hi-f)/(hi-mid));
        if (w>0) bins.push([k,w]);
      }
      return {lo,mid,hi,bins};
    });
  }
  function features(x, windowMs, hopMs, count) {
    const width=windowMs*rate/1000, hop=hopMs*rate/1000, bank=filters(count);
    const frames=Array.from({length:Math.ceil(x.length/hop)},(_,j)=>power(x,j*hop,width));
    const mel=frames.map(row=>bank.map(({bins})=>bins.reduce((sum,[k,w])=>sum+w*row[k],0)));
    const peak=rows=>Math.max(1e-12,...rows.map(row=>Math.max(...row)));
    return {width,hop,bank,frames,mel,linearPeak:peak(frames),melPeak:peak(mel)};
  }
  return {rate,length,fftSize,hz,signal,features};
})();
if (typeof module !== 'undefined') module.exports = MelDSP;
