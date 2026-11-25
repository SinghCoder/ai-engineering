'use strict';

(() => {
  const { fftSize, windowTerm, dftTerm } = MelDSP;
  const audit = byId('audit'), target = byId('audit-content');
  const stages = [...audit.querySelectorAll('[data-stage]')];
  const f = x => Number(x.toPrecision(10)).toString();
  let stage = 'log', source = 'mel', page = 0;

  function table(caption, headers, rows, selected) {
    const head = headers.map(text => '<th scope="col">' + text + '</th>').join('');
    const body = rows.map((row, i) => '<tr' + (i === selected ? ' data-selected' : '') + '>' + row.map((v, j) => {
      const tag = j === 0 ? 'th' : 'td';
      return '<' + tag + (j === 0 ? ' scope="row"' : '') + '>' + (typeof v === 'number' ? f(v) : v) + '</' + tag + '>';
    }).join('') + '</tr>').join('');
    return '<div class="audit-table-wrap" role="region" aria-label="' + caption + '" tabindex="0"><table class="audit-table"><caption>' +
      caption + '</caption><thead><tr>' + head + '</tr></thead><tbody>' + body + '</tbody></table></div>';
  }

  const eq = text => '<p class="audit-formula">' + text + '</p>';
  const go = (next, label) => '<button class="text-control" type="button" data-go="' + next + '">' + label + '</button>';
  const readInteger = (input, min, max, fallback) =>
    input.value === '' ? fallback : clamp(Math.round(+input.value), min, max);

  function selectStage(next) {
    const keepPage = ['window', 'dft'].includes(stage) && ['window', 'dft'].includes(next);
    stage = next;
    if (next === 'mel') source = 'mel';
    if (next === 'power' || next === 'dft' || next === 'window') source = 'linear';
    if (!keepPage) page = 0;
    render();
  }

  function render() {
    const t = +frameInput.value, b = +bandInput.value - 1, k = +binInput.value;
    byId('audit-frame').value = t;
    byId('audit-selection').textContent = (t * model.hop / rate).toFixed(2) + ' s · frame ' + t;
    byId('audit-bin').value = k;
    byId('audit-band').value = b + 1;
    if (!audit.open) return;
    stages.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.stage === stage)));
    const row = model.frames[t], {lo, mid, hi, bins} = model.bank[b];
    const linearPower = row[k], melPower = model.mel[t][b];
    let html = '<p>Frame and FFT bin indices start at 0; mel band labels start at 1. Numbers are rounded to 10 significant digits. On narrow screens, scroll the tables sideways to see every column.</p>';
    if (stage === 'window' || stage === 'dft' || stage === 'power') {
      const term = n => windowTerm(clip, t * model.hop, model.width, n);
      const items = Array.from({length: fftSize}, (_, n) => {
        const w = term(n);
        return { ...w, ...dftTerm(w.value, n, k, fftSize) };
      });
      const re = items.reduce((total, item) => total + item.re, 0);
      const im = items.reduce((total, item) => total + item.im, 0);
      const view = items.slice(page * 16, (page + 1) * 16).map((item, i) => {
        const n = page * 16 + i;
        return stage === 'window' ? [n, item.padded ? 'Padding' : item.index, item.sample, item.weight, item.value]
          : [n, item.value, item.cosine, item.sine, item.re, item.im];
      });
      if (stage === 'window') {
        html += '<h3>Apply the Hann window, then pad</h3><p>Take 400 samples around the window center and multiply them by Hann weights. Put those 400 products at the start of a 1,024-element vector and fill the rest with zeros. Before the start of the clip, missing waveform samples also become zeros.</p>' +
          eq('x[n] = clip[' + (t * model.hop - model.width / 2) + ' + n] × (0.5 − 0.5 cos(2πn / 399)), for n = 0…399; x[n] = 0 for n = 400…1023') +
          table('Selected frame: waveform × Hann weight', ['n', 'Clip index', 'Sample', 'Weight', 'x[n]'], view);
      } else if (stage === 'dft') {
        html += '<h3>One Fourier bin is two dot products</h3><p>For real audio, the 513 nonnegative bins can be written as two 513 × 1,024 matrices multiplied by the same 1,024 × 1 sample vector. Pick a bin to see one row of each matrix. The FFT used by the plot computes the same answer faster, up to floating-point rounding.</p>' +
          eq('θ[k,n] = −2πkn / 1024; Re[k] = Σ x[n]cos(θ[k,n]); Im[k] = Σ x[n]sin(θ[k,n])') +
          eq('Bin ' + k + ' (' + f(hz(k)) + ' Hz): Re = ' + f(re) + '; Im = ' + f(im)) +
          table('Selected DFT rows, column by column', ['n', 'x[n]', 'cos θ', 'sin θ', 'Real product', 'Imaginary product'], view) +
          go('window', 'Where did x[n] come from?');
      } else {
        html += '<h3>Square both Fourier components and add</h3>' +
          eq('Power[' + k + '] = (' + f(re) + ')² + (' + f(im) + ')² ≈ ' + f(linearPower)) +
          go('dft', 'Inspect both dot products') + go('log', 'Follow this power to its color');
      }
      if (stage !== 'power') {
        html += '<div class="audit-controls"><button class="control" type="button" data-page="-1"' + (!page ? ' disabled' : '') + '>Previous</button>' +
          '<span>Rows ' + (page * 16) + '–' + ((page + 1) * 16 - 1) + ' of 1,024</span>' +
          '<button class="control" type="button" data-page="1"' + (page === 63 ? ' disabled' : '') + '>Next</button><label for="audit-jump">Jump to sample</label><input id="audit-jump" type="number" min="0" max="1023" value="' + page * 16 + '"></div>';
      }
    } else if (stage === 'mel') {
      const weights = bins.map(([i,w]) => [
        '<button class="text-control" type="button" data-bin="' + i + '" aria-label="Trace Fourier bin ' + i + '">' + i + '</button>',
        hz(i), row[i], w, row[i] * w,
      ]);
      const edge = j => '700 × (10^(' + j + ' × log₁₀(1 + 8000/700) / 81) − 1)';
      html += '<h3>One row of the mel filter bank</h3><p>The full filter bank is 80 × 513. Multiply it by the 513 × 1 power vector. This table shows all nonzero weights for the selected band. Click a bin to trace its power.</p>' +
        eq('Edges (Hz): e[j] = 700 × (10^(j × log₁₀(1 + 8000/700) / 81) − 1)') +
        eq('e[' + b + '] = ' + edge(b) + ' = ' + f(lo) + '; e[' + (b + 1) + '] = ' + f(mid) + '; e[' + (b + 2) + '] = ' + f(hi)) +
        eq('w(f) = max(0, min((f − lo)/(mid − lo), (hi − f)/(hi − mid))); f[k] = k × 16000 / 1024 Hz') +
        table('Nonzero weights for mel band ' + (b + 1), ['FFT bin', 'Hz', 'Power', 'Weight', 'Product'], weights, bins.findIndex(([i]) => i === k)) +
        eq('Mel power = Σ products ≈ ' + f(melPower)) + go('log', 'Follow this mel power to its color');
    } else {
      const mel = source === 'mel', p = mel ? melPower : linearPower;
      const ref = mel ? model.melPeak : model.linearPeak, db = toDb(p, ref);
      let refFrame = 0, refBin = 0;
      const rows = mel ? model.mel : model.frames;
      rows.some((values, i) => {
        const j = values.indexOf(ref);
        if (j < 0) return false;
        refFrame = i;
        refBin = mel ? j + 1 : j;
        return true;
      });
      const position = heightAt(p, ref);
      const rgb = [...new Uint8ClampedArray(colorAt(p, ref))].join(', ');
      html += '<h3>From ' + (mel ? 'mel power' : 'FFT bin power') + ' to displayed color</h3><p>Each plot uses its own maximum power as its reference. For this plot it occurs at frame ' + refFrame + ', ' + (mel ? 'band ' : 'bin ') + refBin + '.</p>' +
        eq('dB = 10 log₁₀(max(10⁻¹², ' + f(p) + ' / ' + f(ref) + ')) ≈ ' + f(db) + ' dB') +
        eq('Color fraction = clamp((dB + 65) / 65, 0, 1) ≈ ' + f(position) + '; RGB(' + rgb + ')') +
        '<p>The five palette colors, evenly spaced from 0 to 1, are ' + ramp.map(rgb => '(' + rgb.join(', ') + ')').join(', ') + '. The plot interpolates each channel between the neighboring colors and rounds to a byte.</p>' +
        go(mel ? 'mel' : 'power', 'Where did the power come from?') +
        '<button class="text-control" type="button" data-source="' + (mel ? 'linear' : 'mel') + '">Show ' + (mel ? 'FFT bin' : 'mel band') + ' instead</button>';
    }
    target.innerHTML = html;
  }

  function openTrace(nextSource) {
    stopPlayback();
    source = nextSource;
    audit.open = true;
    selectStage('log');
    byId('audit-title').focus({ preventScroll: true });
    audit.scrollIntoView({ block: 'start' });
  }
  byId('trace-bin').onclick = () => openTrace('linear');
  byId('trace-mel').onclick = () => openTrace('mel');
  stages.forEach(button => { button.onclick = () => selectStage(button.dataset.stage); });
  audit.addEventListener('toggle', render);
  audit.addEventListener('tracechange', render);
  byId('audit-frame').oninput = e => { stopPlayback(); frameInput.value = e.target.value; drawSelection(); };
  byId('audit-bin').onchange = e => { stopPlayback(); binInput.value = readInteger(e.target, 0, 512, +binInput.value); drawSelection(); };
  byId('audit-band').onchange = e => { stopPlayback(); bandInput.value = readInteger(e.target, 1, 80, +bandInput.value); drawSelection(); };
  target.onchange = e => {
    if (e.target.id !== 'audit-jump') return;
    page = Math.floor(readInteger(e.target, 0, fftSize - 1, page * 16) / 16);
    render();
    byId('audit-jump').focus();
  };
  target.onclick = e => {
    const button = e.target.closest('button');
    if (!button) return;
    if (button.dataset.page) {
      page += +button.dataset.page;
      render();
      (target.querySelector('[data-page="' + button.dataset.page + '"]:enabled') || target.querySelector('[data-page]:enabled'))?.focus();
    } else if (button.dataset.source) {
      source = button.dataset.source;
      render();
      target.querySelector('[data-source]').focus();
    } else {
      if (button.dataset.bin !== undefined) {
        binInput.value = button.dataset.bin;
        drawSelection();
      }
      selectStage(button.dataset.go || 'power');
      stages.find(item => item.dataset.stage === stage).focus();
    }
  };

  const toy = [3, 1, -1, 1, 3, 1, -1, 1];
  const display = x => Math.abs(x) < 1e-12 ? 0 : x;
  const toyRows = Array.from({length: 5}, (_, k) => toy.map((x, n) => dftTerm(x, n, k, toy.length)));
  const matrix = (caption, key) => table(caption, ['k / n', ...toy.keys()], toyRows.map((r,k) => ['k = ' + k, ...r.map(v => display(v[key]))]));
  byId('paper-content').innerHTML =
    eq('x = [' + toy.join(', ') + ']; N = 8') +
    matrix('Cosine matrix (5 × 8)', 'cosine') + matrix('Negative-sine matrix (5 × 8)', 'sine') +
    table('Multiply each matrix by x; then calculate power', ['Bin k', 'Re', 'Im', 'Re² + Im²'], toyRows.map((r,k) => {
      const re = display(r.reduce((s,v) => s+v.re, 0)), im = display(r.reduce((s,v) => s+v.im, 0));
      return [k, re, im, display(re*re+im*im)];
    })) + '<p>Bin 1 cancels out. Try bin 2: its cosine repeats [1, 0, −1, 0] twice. The dot product is 8, the sine dot product is 0, and the power is 64.</p>';
  render();
})();
