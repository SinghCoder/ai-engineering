'use strict';

const byId = id => document.getElementById(id);
const { rate, hz, signal, features } = MelDSP;
const palette = { signal: '#3757ae', grid: '#dbe1ec', ink: '#535d70', weight: '#a85b11' };
const ramp = [[238,241,250], [159,174,227], [82,113,187], [32,43,104], [18,27,67]];
const clip = signal('mix');
const frameInput = byId('frame');
const bandInput = byId('band');
const binInput = byId('bin');
const model = features(clip, 25, 10, 80);
const toDb = (power, reference) => 10 * Math.log10(Math.max(1e-12, power / reference));
const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
const heightAt = (power, reference) => clamp((toDb(power, reference) + 65) / 65, 0, 1);
const timeTicks = [[0, '0 s'], [.25, '0.6'], [.5, '1.2'], [.75, '1.8'], [1, '2.4']];
const powerTicks = [[0, '−65'], [.5, '−33'], [1, '0 dB']];
frameInput.max = model.frames.length - 1;

function chart(id) {
  const canvas = byId(id);
  const style = getComputedStyle(canvas);
  const left = parseFloat(style.getPropertyValue('--plot-left'));
  const right = parseFloat(style.getPropertyValue('--plot-right'));
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  const scale = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * scale);
  canvas.height = Math.round(height * scale);
  const g = canvas.getContext('2d');
  g.scale(scale, scale);
  g.font = '500 11px "DM Sans", Arial, sans-serif';
  g.textBaseline = 'middle';
  return { g, left, width: width - left - right, height: height - 26 };
}

function axes(plot, xTicks, yTicks, grid = true) {
  const { g, left, width, height } = plot;
  g.fillStyle = palette.ink;
  g.strokeStyle = palette.grid;
  g.lineWidth = 1;
  g.textAlign = 'right';
  for (const [position, label] of yTicks) {
    const y = (1 - position) * height;
    g.fillText(label, left - 7, clamp(y, 7, height));
    if (grid) line(g, left, y, left + width, y);
  }
  line(g, left, height + .5, left + width, height + .5);
  g.textAlign = 'center';
  for (const [position, label] of xTicks) g.fillText(label, left + position * width, height + 15);
}

function line(g, x1, y1, x2, y2) {
  g.beginPath();
  g.moveTo(x1, y1);
  g.lineTo(x2, y2);
  g.stroke();
}

function boundPlot({ g, left, width, height }) {
  g.save();
  g.beginPath();
  g.rect(left, 0, width, height);
  g.clip();
}

function colorAt(power, reference) {
  const position = (ramp.length - 1) * heightAt(power, reference);
  const index = Math.min(ramp.length - 2, Math.floor(position));
  const balance = position - index;
  return ramp[index].map((value, c) => value * (1 - balance) + ramp[index + 1][c] * balance);
}

function makeHeatmap(rows, reference) {
  const canvas = document.createElement('canvas');
  canvas.width = rows.length;
  canvas.height = rows[0].length;
  const g = canvas.getContext('2d');
  const image = g.createImageData(canvas.width, canvas.height);
  rows.forEach((row, column) => row.forEach((power, bin) => {
    const offset = 4 * ((canvas.height - 1 - bin) * canvas.width + column);
    image.data.set(colorAt(power, reference), offset);
    image.data[offset + 3] = 255;
  }));
  g.putImageData(image, 0, 0);
  return canvas;
}

const linearMap = makeHeatmap(model.frames, model.linearPeak);
const melMap = makeHeatmap(model.mel, model.melPeak);

function drawWave(center) {
  const plot = chart('wave');
  const { g, left, width, height } = plot;
  axes(plot, timeTicks, [[0, '−1'], [.5, '0'], [1, '+1']]);
  boundPlot(plot);
  const factor = width / clip.length;
  const x = left + center * factor;
  const window = model.width * factor;
  g.fillStyle = '#dce5fd';
  g.fillRect(x - window / 2, 0, window, height);
  g.strokeStyle = '#7e97cf';
  g.beginPath();
  for (let pixel = 0; pixel < width; pixel++) {
    const start = Math.floor(pixel / factor);
    const end = Math.min(clip.length, Math.floor((pixel + 1) / factor));
    let low = clip[start], high = low;
    for (let i = start + 1; i < end; i++) {
      low = Math.min(low, clip[i]);
      high = Math.max(high, clip[i]);
    }
    g.moveTo(left + pixel + .5, (1 - high) * height / 2);
    g.lineTo(left + pixel + .5, (1 - low) * height / 2);
  }
  g.stroke();
  g.strokeStyle = palette.signal;
  g.lineWidth = 2;
  line(g, x, 0, x, height);
  g.restore();
}

function spectrumTrace(plot, row, reference, start, end) {
  const { g, left, width, height } = plot;
  const first = Math.ceil(start / hz(1)), last = Math.floor(end / hz(1));
  g.strokeStyle = palette.signal;
  g.lineWidth = 1.8;
  g.beginPath();
  for (let k = first; k <= last; k++) {
    const x = left + (hz(k) - start) * width / (end - start);
    const y = (1 - heightAt(row[k], reference)) * height;
    if (k === first) g.moveTo(x, y);
    else g.lineTo(x, y);
  }
  g.stroke();
}

function drawMaps(index, band) {
  for (const [id, image, ticks] of [
    ['linear', linearMap, [[0, '0 Hz'], [.5, '4 kHz'], [1, '8 kHz']]],
    ['mel', melMap, [[0, '1'], [.5, '40'], [1, '80']]],
  ]) {
    const plot = chart(id);
    const { g, left, width, height } = plot;
    boundPlot(plot);
    const step = width * model.hop / clip.length;
    const x = left + index * step;
    g.imageSmoothingEnabled = false;
    g.drawImage(image, left - step / 2, 0, step * image.width, height);
    g.strokeStyle = '#14204c';
    g.lineWidth = 3;
    line(g, x, 0, x, height);
    g.strokeStyle = '#fff';
    g.lineWidth = 1;
    line(g, x, 0, x, height);
    {
      const selected = id === 'mel' ? band : +binInput.value;
      const unit = height / image.height;
      g.strokeStyle = '#ffc46b';
      g.lineWidth = 2;
      g.strokeRect(x - Math.max(3, step) / 2, height - (selected + .5) * unit - Math.max(4, unit) / 2, Math.max(4, step), Math.max(4, unit));
    }
    g.restore();
    axes(plot, timeTicks, ticks, false);
  }
}

function drawColumn(row, selected) {
  const plot = chart('column');
  const {g, left, width, height} = plot;
  const x = k => left + heightAt(row[k], model.linearPeak) * width;
  const y = k => height * (1 - (k + .5) / row.length);
  axes(plot, [[0, '−65 dB'], [1, '0 dB']], [[0, '0 Hz'], [.5, '4 kHz'], [1, '8 kHz']]);
  g.strokeStyle = palette.signal;
  g.lineWidth = 1.5;
  g.beginPath();
  row.forEach((_, k) => k ? g.lineTo(x(k), y(k)) : g.moveTo(x(k), y(k)));
  g.stroke();
  g.fillStyle = palette.weight;
  g.beginPath();
  g.arc(x(selected), clamp(y(selected), 4, height), 4, 0, 2 * Math.PI);
  g.fill();
}

function drawBand(row, band) {
  const plot = chart('detail');
  const { g, left, width, height } = plot;
  const { lo, mid, hi, bins } = band;
  const padding = Math.max(60, (hi - lo) * .25);
  const start = Math.max(0, lo - padding), end = Math.min(rate / 2, hi + padding);
  const span = end - start;
  const binWidth = width * hz(1) / span;
  const x = frequency => left + width * (frequency - start) / span;
  axes(plot, [[0, Math.round(start)], [.5, Math.round((start + end) / 2)], [1, Math.round(end) + ' Hz']], powerTicks);
  boundPlot(plot);
  g.fillStyle = '#a6b7e5';
  for (let k = Math.ceil(start / hz(1)); hz(k) <= end; k++) {
    const bar = heightAt(row[k], model.linearPeak) * height;
    g.fillRect(x(hz(k)) - binWidth * .35, height - bar, Math.min(12, binWidth * .7), bar);
  }
  g.fillStyle = palette.signal;
  for (const [k] of bins) {
    const bar = heightAt(row[k], model.linearPeak) * height;
    g.fillRect(x(hz(k)) - binWidth * .35, height - bar, Math.min(12, binWidth * .7), bar);
  }
  g.strokeStyle = palette.weight;
  g.lineWidth = 2;
  g.beginPath();
  g.moveTo(x(lo), height);
  g.lineTo(x(mid), 3);
  g.lineTo(x(hi), height);
  g.stroke();
  g.restore();
}

function drawSelection() {
  const index = +frameInput.value, bandIndex = +bandInput.value - 1;
  const row = model.frames[index], band = model.bank[bandIndex];
  const center = index * model.hop, time = (center / rate).toFixed(2);
  const power = model.mel[index][bandIndex];
  const timeLabel = time + ' s';
  const region = center / rate < 0.8 ? 'Hum' : center / rate < 1.6 ? 'Hiss' : 'Rising tone';
  byId('frameInfo').textContent = timeLabel;
  byId('cellTime').textContent = timeLabel;
  byId('window-region').textContent = region;
  frameInput.setAttribute('aria-valuetext', region + ', ' + timeLabel);
  byId('bandInfo').textContent = Math.round(band.lo) + '–' + Math.round(band.hi) + ' Hz';
  bandInput.setAttribute('aria-valuetext', 'Band ' + (bandIndex + 1) + ', ' + byId('bandInfo').textContent);
  byId('power').textContent = power.toPrecision(3);
  byId('level').textContent = toDb(power, model.melPeak).toFixed(1) + ' dB';
  const top = band.bins.reduce((best, [k, weight]) => {
    const value = row[k] * weight;
    return value > best.value ? { k, weight, value } : best;
  }, { value: 0 });
  byId('contribution').textContent = top.value > 0
    ? 'Strongest contribution: the ' + hz(top.k).toFixed(1) + ' Hz bin supplies ' + (100 * top.value / power).toFixed(0) + '% of the weighted sum. The amber outline shows filter weights from zero to one.'
    : 'The selected frame contributes zero power to this band.';
  drawWave(center);
  const spectrum = chart('spectrum');
  const k = +binInput.value, frequency = hz(k), level = toDb(row[k], model.linearPeak);
  const span = frequency > 3000 ? 8000 : 3000;
  axes(spectrum, [[0, '0 Hz'], [1/3, (span / 3000).toFixed(1) + ' kHz'], [2/3, (span / 1500).toFixed(1) + ' kHz'], [1, span / 1000 + ' kHz']], powerTicks);
  spectrumTrace(spectrum, row, model.linearPeak, 0, span);
  const {g, left, width, height} = spectrum;
  g.beginPath();
  g.arc(left + width * frequency / span, height * (1 - heightAt(row[k], model.linearPeak)), 4, 0, 2 * Math.PI);
  g.fillStyle = palette.weight;
  g.fill();
  byId('bin-frequency').textContent = frequency.toFixed(2) + ' Hz';
  binInput.setAttribute('aria-valuetext', 'Bin ' + k + ', ' + byId('bin-frequency').textContent);
  byId('bin-result').textContent = timeLabel + ' · bin ' + k + ' · ' + level.toFixed(2) + ' dB in both plots' + (level < -65 ? ' (displayed at the −65 dB floor)' : '');
  byId('audit').dispatchEvent(new Event('tracechange'));
  drawMaps(index, bandIndex);
  drawColumn(row, k);
  drawBand(row, band);
}

let example = 'tones', comparison;
function updateComparison() {
  const hop = +byId('hop').value;
  const sample = signal(example);
  const windows = [10, 60].map(window => features(sample, window, hop, 24));
  const center = Math.round(1.2 * rate / windows[0].hop);
  const totalPower = row => row.reduce((sum, bin) => sum + bin, 0);
  const envelope = windows.map(item => item.frames.map(totalPower));
  comparison = { windows, center, envelope };
  byId('comparison-note').textContent = example === 'tones'
    ? 'At 1.20 s the 60 ms window separates the 440 Hz and 510 Hz tones. The short window blends them into one broad peak.'
    : 'The 60 ms window picks up the click over a longer time span. Each dot is one analysis frame.';
  byId('hop-note').textContent = 'A ' + hop + ' ms hop gives ' + windows[0].frames.length + ' frames over 2.4 s for either window.';
  for (const id of ['shortWindow', 'longWindow']) {
    const duration = id === 'shortWindow' ? 10 : 60;
    byId(id).setAttribute('aria-label', (example === 'tones' ? 'Frequency spectrum' : 'Total power around the click') + ' using a ' + duration + ' millisecond window.');
  }
  drawComparison();
}

function drawComparison() {
  const { windows, center, envelope } = comparison;
  const reference = example === 'tones'
    ? Math.max(...windows.map(item => Math.max(...item.frames[center])))
    : Math.max(...envelope.map(row => Math.max(...row)));
  ['shortWindow', 'longWindow'].forEach((id, index) => {
    const plot = chart(id);
    const { g, left, width, height } = plot;
    if (example === 'tones') {
      axes(plot, [[0, '300 Hz'], [.5, '500'], [1, '700']], powerTicks);
      spectrumTrace(plot, windows[index].frames[center], reference, 300, 700);
      return;
    }
    const start = 1.14, end = 1.26;
    axes(plot, [[0, '1.14 s'], [.5, '1.20'], [1, '1.26']], powerTicks);
    g.strokeStyle = palette.grid;
    line(g, left + width / 2, 0, left + width / 2, height);
    g.fillStyle = palette.signal;
    envelope[index].forEach((power, step) => {
      const time = step * windows[index].hop / rate;
      if (time < start - 1e-9 || time > end + 1e-9) return;
      const x = left + (time - start) * width / (end - start);
      const y = (1 - heightAt(power, reference)) * (height - 7) + 3.5;
      g.beginPath();
      g.arc(x, y, 3.3, 0, Math.PI * 2);
      g.fill();
    });
  });
}

let audio, playing;
const playButton = byId('play');
function playbackLabel(active) {
  byId('play-label').textContent = active ? 'Stop clip' : 'Play clip';
  byId('play-icon').textContent = active ? '■' : '▶';
}
function stopPlayback() {
  if (!playing) return;
  const source = playing;
  playing = undefined;
  source.stop();
  playbackLabel(false);
}
playButton.addEventListener('click', async () => {
  if (playing) return stopPlayback();
  playButton.disabled = true;
  try {
    audio ??= new AudioContext();
    await audio.resume();
    const buffer = audio.createBuffer(1, clip.length, rate);
    buffer.copyToChannel(clip, 0);
    const source = audio.createBufferSource();
    source.buffer = buffer;
    source.connect(audio.destination);
    const started = audio.currentTime;
    playing = source;
    source.onended = () => {
      if (playing === source) {
        playing = undefined;
        playbackLabel(false);
      }
    };
    source.start();
    playbackLabel(true);
    byId('audio-status').textContent = '';
    const advance = () => {
      if (playing !== source) return;
      const index = Math.min(+frameInput.max, Math.floor((audio.currentTime - started) * rate / model.hop));
      if (index !== +frameInput.value) {
        frameInput.value = index;
        drawSelection();
      }
      requestAnimationFrame(advance);
    };
    requestAnimationFrame(advance);
  } catch {
    byId('audio-status').textContent = 'Audio is unavailable. Move the window to explore the waveform.';
  } finally {
    playButton.disabled = false;
  }
});

for (const id of ['wave', 'spectrum', 'linear', 'mel', 'column']) {
  const canvas = byId(id);
  const select = event => {
    stopPlayback();
    const box = canvas.getBoundingClientRect(), style = getComputedStyle(canvas);
    const left = parseFloat(style.getPropertyValue('--plot-left')), right = parseFloat(style.getPropertyValue('--plot-right'));
    const position = clamp((event.clientX - box.left - left) / (box.width - left - right), 0, 1);
    if (id === 'spectrum') {
      const span = hz(+binInput.value) > 3000 ? 8000 : 3000;
      binInput.value = Math.round(position * span / hz(1));
    } else if (id === 'column') {
      binInput.value = Math.floor((1 - (event.clientY - box.top) / (box.height - 26)) * model.frames[0].length);
    } else {
      frameInput.value = Math.round(position * clip.length / model.hop);
      if (id === 'linear') binInput.value = Math.floor((1 - (event.clientY - box.top) / (box.height - 26)) * model.frames[0].length);
    }
    if (id === 'mel') bandInput.value = 1 + Math.floor((1 - (event.clientY - box.top) / (box.height - 26)) * model.bank.length);
    drawSelection();
  };
  canvas.addEventListener('pointerdown', event => {
    if (!event.isPrimary || event.button !== 0) return;
    if (event.pointerType === 'mouse' && id !== 'spectrum') canvas.setPointerCapture(event.pointerId);
    select(event);
  });
  canvas.addEventListener('pointermove', event => {
    if (canvas.hasPointerCapture(event.pointerId)) select(event);
  });
}
frameInput.addEventListener('input', () => { stopPlayback(); drawSelection(); });
bandInput.addEventListener('input', drawSelection);
binInput.addEventListener('input', () => { stopPlayback(); drawSelection(); });
byId('hop').addEventListener('change', updateComparison);
document.querySelectorAll('[data-example]').forEach(button => button.addEventListener('click', () => {
  example = button.dataset.example;
  document.querySelectorAll('[data-example]').forEach(item => item.setAttribute('aria-pressed', String(button === item)));
  updateComparison();
}));
drawSelection();
updateComparison();
new ResizeObserver(() => { drawSelection(); drawComparison(); }).observe(byId('wave'));
document.fonts.ready.then(() => { drawSelection(); drawComparison(); });
