// demo/lineup-core.js: the data side of the demo (no UI code). Works in any modern browser,
// served from the same site as data/ (GitHub Pages is fine; file:// pages can't fetch).

export async function loadDemo(dir = 'data') {
  const M = await (await fetch(`${dir}/manifest.json`)).json();
  return { dir, M, scores: {} };
}

export async function scoresFor(D, name) {        // Int16Array [P, T, K]
  if (!D.scores[name]) {
    const buf = await (await fetch(`${D.dir}/scores_${name}.i16`)).arrayBuffer();
    D.scores[name] = new Int16Array(buf);
  }
  return D.scores[name];
}

const range = (a, b) => Array.from({ length: Math.max(0, b - a) }, (_, i) => a + i);

// Ranked lineup for participant p and true clip t. kind: 'hard' or 'easy'; band: 0-9 or null.
// Slot 0 is the clip heard, slots 1-49 the hard distractors (most similar first), the rest
// random eligible distractors. A missing window (-32768) drops out; check items.some(c => c.heard).
export function lineup(D, q, name, p, t, kind, size, band = null) {
  const { M } = D;
  const { lo, hi } = M.scores[name];
  const base = (p * M.T + t) * M.K;
  let pick = kind === 'hard' ? range(1, 50) : range(50, M.K);
  if (band !== null) pick = range(1, M.K).filter((s) => M.bands[t][s] === band);
  return [0, ...pick.slice(0, size - 1)]
    .map((s) => ({ clip: M.slots[t][s], heard: s === 0, q: q[base + s] }))
    .filter((c) => c.q !== -32768)
    .map((c) => ({ ...c, score: lo + ((c.q + 32767) / 65534) * (hi - lo) }))
    .sort((a, b) => b.score - a.score);
}

// The model's probability that each lineup clip is the one heard (softmax of the scores, Eq. 2).
export function probabilities(items) {
  const m = Math.max(...items.map((c) => c.score));
  const e = items.map((c) => Math.exp(c.score - m));
  const z = e.reduce((a, b) => a + b, 0);
  return items.map((c, i) => ({ ...c, prob: e[i] / z }));
}

// Play one clip: seek inside its sound file and stop after 3 s. Encode the files as
// constant-bitrate MP3 so seeking lands where it should.
const players = {};
export function play(D, clipId, dur = 3) {
  const c = D.M.clips[clipId];
  Object.values(players).forEach((a) => a.pause());
  const a = (players[c.seg] ??= new Audio(`${D.dir}/audio/seg${c.seg}.mp3`));
  a.currentTime = c.start;
  a.ontimeupdate = () => { if (a.currentTime >= c.start + dur) a.pause(); };
  a.play();
}
