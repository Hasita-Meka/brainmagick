"""End-to-end check of the audio-only pipeline, anchors, metrics and bootstrap on synthetic data."""
import sys, tempfile, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.io import wavfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ext import pool as P, envelope as E, lineups as LU, anchors as A, metrics as M, stats as ST

rng = np.random.default_rng(1)
SEG_DUR = 70.0
stop = sorted(P.STOP)[:20]                       # function words: 40% of tokens
content = [f"w{k}" for k in range(1500)]
vocab = stop + content
flat = 1 / (np.arange(1, 1501) + 60.0)
zipf = np.r_[np.full(20, 0.4 / 20), 0.6 * flat / flat.sum()]

# --- synthetic word table and audio -------------------------------------------------------
rows, tmp = [], Path(tempfile.mkdtemp())
seg_dur = {}
for seg in range(1, 13):
    t, words = 0.4, []
    while t < SEG_DUR - 0.3:
        words.append((t, rng.choice(vocab, p=zipf)))
        t += rng.uniform(0.25, 0.5) + (rng.random() < 0.08) * rng.uniform(0.3, 0.9)  # pauses
    if seg in (3, 7, 11):                       # repeated phrase, as in the chapter
        k = len(words) // 2
        for j, w in enumerate(["down", "down", "down"]):
            words[k + j] = (words[k + j][0], w)
    rows += [dict(seg=seg, onset=o, word=w) for o, w in words]
    sr = 22050
    n = int(SEG_DUR * sr)
    env = np.zeros(n)
    for o, _ in words:                          # one syllable-like burst per word
        i0 = int(o * sr); L = int(rng.uniform(0.12, 0.3) * sr)
        env[i0:i0 + L] += np.hanning(L)[: len(env[i0:i0 + L])] * rng.uniform(0.3, 1.0)
    x = env * rng.standard_normal(n) * 0.3 + 1e-3 * rng.standard_normal(n)
    wavfile.write(tmp / f"seg{seg}.wav", sr, (np.clip(x, -1, 1) * 32767).astype(np.int16))
    seg_dur[seg] = n / sr
words = pd.DataFrame(rows)

# --- 4.1 pool and split tags --------------------------------------------------------------
pool = P.build_pool(words, seg_dur)
win = []
for seg in range(1, 13):                        # emulate repo windows: 70/20/10 by time
    for a in np.arange(0, 0.7 * SEG_DUR - 3, 0.5): win.append((seg, a, a + 3, "train"))
    g = pool[pool.seg == seg]
    for o in g.onset:
        s = "valid" if 0.7 * SEG_DUR + 0.5 <= o - 0.5 and o + 2.5 <= 0.9 * SEG_DUR else None
        s = "test" if o - 0.5 >= 0.9 * SEG_DUR + 0.5 else s
        if s: win.append((seg, o - 0.5, o + 2.5, s))
pool["split"] = P.tag_splits(pool, pd.DataFrame(win, columns=["seg", "start", "stop", "split"]))
print(pool.split.value_counts().to_dict(), "pool size", len(pool))
true_ids = pool.index[pool.split == "test"].to_numpy()

# --- 4.2 eligibility ---------------------------------------------------------------------
elig, overlap, repeat = P.eligibility(pool)
assert not elig[np.arange(len(pool)), np.arange(len(pool))].any()
n_rep = (repeat & ~overlap).sum()
print("eligible per true clip (min/median):", elig[true_ids].sum(1).min(), int(np.median(elig[true_ids].sum(1))),
      "| repeat-only exclusions:", int(n_rep))
assert n_rep > 0

# --- 4.3 envelopes -----------------------------------------------------------------------
t0 = time.time()
env_raw, env_slow = {}, {}
for seg in range(1, 13):
    x, sr = E.load_wav(tmp / f"seg{seg}.wav")
    e = E.log_envelope(x, sr); env_raw[seg] = e; env_slow[seg] = E.slow(e)
    assert abs(len(e) / E.SR_FEAT - seg_dur[seg]) < 0.05, (len(e), seg_dur[seg])
clips_raw, clips_slow = E.slice_clips(env_raw, pool), E.slice_clips(env_slow, pool)
print("envelopes ok", clips_raw.shape, f"{time.time()-t0:.1f}s")

# --- 4.4 similarity and lineups ----------------------------------------------------------
sim = LU.similarity(clips_slow)
sim_p = LU.similarity(clips_slow, center=True)
assert np.allclose(np.diag(sim), 1) and np.allclose(np.diag(sim_p), 1)
# window -> clip matching, with jitter below one sample and one bad window
tw = pool.loc[true_ids, ["seg", "start"]].copy()
tw["start"] += rng.uniform(-0.004, 0.004, len(tw))
assert (P.match_windows(tw, pool) == true_ids).all()
bad = pd.concat([tw, pd.DataFrame({"seg": [1], "start": [999.0]})])
try:
    P.match_windows(bad, pool); raise AssertionError("bad window not caught")
except ValueError as e:
    print("match_windows catches unmatched windows:", e)
hard, band, easy = LU.make_lineups(sim, elig, true_ids)
print("hardness:", LU.hardness_report(sim, elig, true_ids, hard))
assert all(elig[i, hard[t]].all() for t, i in enumerate(true_ids))
assert ((band >= 0) == elig[true_ids]).all()

# --- target spaces -----------------------------------------------------------------------
def standardize(Y):  # per channel over all clips and times
    return (Y - Y.mean(axis=(0, 2), keepdims=True)) / Y.std(axis=(0, 2), keepdims=True)
N, T = len(pool), clips_raw.shape[1]
Y_env = standardize(clips_raw[:, None, :].astype(np.float64))
Y_slow = standardize(clips_slow[:, None, :].astype(np.float64))
F = 24
W = rng.standard_normal((F, 1))
word_code = {w: rng.standard_normal(F) for w in vocab + ["down"]}
lex = np.zeros((N, F, T))                      # word identity at each word onset in the clip
for c, (s, a) in enumerate(zip(pool.seg, pool.start)):
    g = words[(words.seg == s) & (words.onset >= a) & (words.onset < a + 3)]
    for o, w in zip(g.onset, g.word):
        k = int((o - a) * 120); lex[c, :, k:k + 30] += word_code[w][:, None]
Y_w2v = standardize(W[None] * Y_env + 1.0 * lex)

# --- 4.5 anchors -------------------------------------------------------------------------
t0 = time.time()
G = {k: A.gram_scores(Y) for k, Y in dict(env=Y_env, slow=Y_slow, w2v=Y_w2v).items()}
L = {k: A.noise_factor(g) for k, g in G.items()}
curves = {}
for k in G:
    sig = A.sigma_grid(G[k])
    curves[k] = A.anchor_curves(G[k], L[k], true_ids, elig, hard, sig, n_draws=16)
    fe = curves[k][0].mean(1)
    assert fe[0] > 0.97 and abs(fe[-1] - 0.5) < 0.03, (k, fe[0], fe[-1])
print(f"anchor curves ok ({time.time()-t0:.1f}s)")
for target in (0.6, 0.7, 0.8):
    R = {k: A.retention(target, A.matched_hard(*[c.mean(1) for c in curves[k]], target)) for k in curves}
    print(f"  step zero @F_easy={target}: R_loud(slow)={R['slow']:.2f} R_loud(raw)={R['env']:.2f} R_even(w2v)={R['w2v']:.2f}")

# --- simulated trained models --------------------------------------------------------------
def scores(Z, Ycand):           # same rule as the anchors: <Z, Y_c> / ||Y_c||
    Zf, Yf = Z.reshape(len(Z), -1), Ycand.reshape(len(Ycand), -1)
    return Zf @ Yf.T / np.linalg.norm(Yf, axis=1)[None]

def calibrate(Ytrue_space, Ycand, target=0.72):
    lo, hi = 1e-2, 1e4                       # bisection on log noise for one participant
    for _ in range(25):
        s = np.sqrt(lo * hi)
        Z = Ytrue_space[true_ids] + s * np.random.default_rng(0).standard_normal(Ytrue_space[true_ids].shape)
        fe, _, _ = M.fractions(scores(Z, Ycand)[None], true_ids, elig, hard)
        lo, hi = (s, hi) if np.nanmean(fe) > target else (lo, s)
    return s

def simulate(Ytrue_space, Ycand, noise_scale=None, n_part=33):
    noise_scale = noise_scale or calibrate(Ytrue_space, Ycand)
    S = np.full((n_part, len(true_ids), N), np.nan, np.float32)
    for p in range(n_part):
        s = noise_scale * rng.uniform(0.8, 1.25)
        Z = Ytrue_space[true_ids] + s * rng.standard_normal(Ytrue_space[true_ids].shape)
        S[p] = scores(Z, Ycand)
    S[0, 3] = np.nan            # a missing window
    return S.astype(np.float16)

models = {
    "w2v_even": simulate(Y_w2v, Y_w2v),               # reads its whole target evenly -> p ~ 1
    "w2v_loud": simulate(W[None] * Y_slow, Y_w2v),    # only the slow-envelope part -> p ~ 0
    "envelope": simulate(Y_slow, Y_env),              # slow-envelope reader in envelope space
}
for name, S in models.items():
    fe, fh, fb = M.fractions(S, true_ids, elig, hard, band)
    space = "env" if name == "envelope" else "w2v"
    anchors = {"loud": curves["slow"], "even": curves[space]}
    point = ST.summarize(fe, fh, anchors)
    curve = np.round(M.group(fb), 3)
    print(f"{name:9s} F_easy={point['F_easy']:.3f} F_hard={point['F_hard']:.3f} R={point['R']:.2f} "
          f"R_loud={point['R_loud']:.2f} R_even={point['R_even']:.2f} p={point.get('p', np.nan):.2f} "
          f"| band curve {curve[[0, 4, 9]]}")
    assert 0.6 < point["F_easy"] < 0.85

# --- metrics consistency checks ----------------------------------------------------------
S = models["w2v_even"]
ex = M.topk_easy_exact(S, true_ids, elig)
mc = []                                          # Monte-Carlo random 50-clip lineups
for t, i in enumerate(true_ids[:40]):
    D = np.flatnonzero(elig[i]); s = S[1, t].astype(np.float32)
    hits = [(1 + (s[rng.choice(D, 49, replace=False)] > s[i]).sum()) <= 10 for _ in range(400)]
    mc.append(np.mean(hits))
print("top-10 easy exact vs Monte-Carlo:", round(float(np.mean(ex[10][1, :40])), 3), round(float(np.mean(mc)), 3))
assert abs(np.mean(ex[10][1, :40]) - np.mean(mc)) < 0.03
print("top-10 hard:", round(float(M.group(M.topk_hard(S, true_ids, hard)[10])), 3),
      "| paper-style top-10 over test clips:", round(float(M.group(M.topk_paper(S, true_ids)[10])), 3),
      "chance", round(10 / len(true_ids), 3))
print("split check:", {k: round(float(v), 3) for k, v in M.split_check(S, true_ids, elig, pool.split).items()})
ctrl = M.shuffled_control(S, true_ids, elig, overlap)
print("shuffled control:", round(ctrl, 3)); assert abs(ctrl - 0.5) < 0.03

# --- bootstrap ---------------------------------------------------------------------------
fe, fh, _ = M.fractions(S, true_ids, elig, hard)
t0 = time.time()
point, ci, boot = ST.two_way_bootstrap(fe, fh, {"loud": curves["slow"], "even": curves["w2v"]}, n_boot=300)
print(f"bootstrap ({time.time()-t0:.1f}s): p={point['p']:.2f} CI=[{ci.loc[0.025,'p']:.2f}, {ci.loc[0.975,'p']:.2f}]")
own = np.nanmean(fe, 1); swapped = own - rng.uniform(0, 0.05, len(own))
print("paired test:", ST.paired_test(own, swapped))
print("ALL CHECKS PASSED")

# --- step-zero power check with a simulated even reader -------------------------------------
sig_w2v = A.sigma_grid(G["w2v"])
targets = np.random.default_rng(0).uniform(0.6, 0.8, 33)
sig_p = [A.sigma_for(curves["w2v"][0].mean(1), sig_w2v, x) for x in targets]
S_sim = A.simulate_reader(G["w2v"], L["w2v"], true_ids, sig_p)
fe, fh, _ = M.fractions(S_sim, true_ids, elig, hard)
point, ci, _ = ST.two_way_bootstrap(fe, fh, {"loud": curves["slow"], "even": curves["w2v"]}, n_boot=500)
print(f"power check: F_easy={point['F_easy']:.3f} (targets mean {targets.mean():.3f}) "
      f"p={point['p']:.2f} CI=[{ci.loc[0.025,'p']:.2f}, {ci.loc[0.975,'p']:.2f}]")
assert abs(point["F_easy"] - targets.mean()) < 0.03
print("POWER CHECK OK")
