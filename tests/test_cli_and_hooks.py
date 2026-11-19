"""Checks for step_zero (three passes), report, probes helpers, repo-hook syntax and the demo core."""
import json, py_compile, subprocess, sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd
from scipy.io import wavfile
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ext import anchors as A, demo_export as DX, metrics as M, pool as P, probes as PR, report as RP, step_zero as SZ

rng = np.random.default_rng(7)
tmp = Path(tempfile.mkdtemp())
stop = sorted(P.STOP)[:20]
vocab = stop + [f"w{k}" for k in range(1500)]
flat = 1 / (np.arange(1, 1501) + 60.0)
prob = np.r_[np.full(20, 0.4 / 20), 0.6 * flat / flat.sum()]

# synthetic inputs in the CSV formats step_zero expects
words, sounds, windows, DUR = [], [], [], 60.0
for seg in range(1, 7):
    t, ws = 0.4, []
    while t < DUR - 0.3:
        ws.append((t, rng.choice(vocab, p=prob)))
        t += rng.uniform(0.25, 0.5) + (rng.random() < 0.08) * rng.uniform(0.3, 0.9)
    words += [dict(seg=seg, onset=o, word=w) for o, w in ws]
    sr, n = 16000, int(DUR * 16000)
    env = np.zeros(n)
    for o, _ in ws:
        i0, L = int(o * sr), int(rng.uniform(0.12, 0.3) * sr)
        env[i0:i0 + L] += np.hanning(L)[: len(env[i0:i0 + L])] * rng.uniform(0.3, 1.0)
    x = env * rng.standard_normal(n) * 0.3 + 1e-3 * rng.standard_normal(n)
    wavfile.write(tmp / f"seg{seg}.wav", sr, (np.clip(x, -1, 1) * 32767).astype(np.int16))
    sounds.append(dict(seg=seg, path=str(tmp / f"seg{seg}.wav")))
    for a in np.arange(0, 0.7 * DUR - 3, 0.5):
        windows.append(dict(seg=seg, start=a, stop=a + 3, split="train"))
    for o, _ in ws:
        if 0.7 * DUR + 0.5 <= o - 0.5 and o + 2.5 <= 0.9 * DUR:
            windows.append(dict(seg=seg, start=o - 0.5, stop=o + 2.5, split="valid"))
        elif o - 0.5 >= 0.9 * DUR + 0.5 and o + 2.5 <= DUR:
            windows.append(dict(seg=seg, start=o - 0.5, stop=o + 2.5, split="test"))
for name, rows in dict(words=words, sounds=sounds, windows=windows).items():
    pd.DataFrame(rows).to_csv(tmp / f"{name}.csv", index=False)
out = tmp / "results"
args = ["--words", str(tmp / "words.csv"), "--sounds", str(tmp / "sounds.csv"),
        "--windows", str(tmp / "windows.csv"), "--out", str(out)]

# pass 1: audio only, through the command-line entry point
SZ.main(args)
r1 = json.loads((out / "step_zero_report.json").read_text())
assert {"loud", "loud_raw"} == set(r1["retention_at_F_easy"]) and (out / "step_zero.png").exists()
pool = pd.read_csv(out / "pool.csv")

# pass 2: add a w2v-like target exported for this pool (envelope part + word identity)
env = np.load(out / "envelopes.npz")["raw"]
F, T = 24, env.shape[1]
code = {w: rng.standard_normal(F) for w in vocab}
wt = pd.DataFrame(words)
lex = np.zeros((len(pool), F, T))
for c, (s, a) in enumerate(zip(pool.seg, pool.start)):
    g = wt[(wt.seg == s) & (wt.onset >= a) & (wt.onset < a + 3)]
    for o, w in zip(g.onset, g.word):
        k = int((o - a) * 120); lex[c, :, k:k + 30] += code[w][:, None]
Y = SZ.standardize(rng.standard_normal((F, 1))[None] * SZ.standardize(env[:, None, :]) + lex)
np.save(out / "Y_w2v.npy", Y.astype(np.float16))
SZ.main(args + ["--y", f"w2v={out / 'Y_w2v.npy'}"])
r2 = json.loads((out / "step_zero_report.json").read_text())
pc = r2["power_check"]["w2v"]
print("pass 2 retention @0.7:", {k: v["0.7"] for k, v in r2["retention_at_F_easy"].items()}, "power:", pc)
assert pc["p_ci95"][0] > 0.3 and r2["retention_at_F_easy"]["w2v"]["0.7"] > r2["retention_at_F_easy"]["loud"]["0.7"]

# pass 3: exact G instead of Y, and an explicit true_ids file -> same anchors as pass 2
fe2 = np.load(out / "anchors_w2v.npz")["f_easy"]
np.save(out / "G_w2v.npy", A.gram_scores(np.load(out / "Y_w2v.npy")))
np.save(out / "true_ids.npy", np.load(out / "lineups.npz")["true_ids"])
SZ.main(args + ["--g", f"w2v={out / 'G_w2v.npy'}", "--true-ids", str(out / "true_ids.npy")])
assert np.allclose(np.load(out / "anchors_w2v.npz")["f_easy"], fe2), "G path differs from Y path"
np.save(tmp / "bad.npy", Y[:5])
try:
    SZ.main(args + ["--y", f"bad={tmp / 'bad.npy'}"])
    raise AssertionError("row mismatch not caught")
except ValueError as e:
    print("row-count guard:", str(e)[:60], "...")

# plain inner-product mode (if R4 says get_scores doesn't divide by the candidate norm)
z = np.load(out / "lineups.npz")
true_ids, hard, band = z["true_ids"], z["hard"], z["band"]
eligible, overlap = np.load(out / "eligible.npy"), np.load(out / "overlap.npy")
Gp = A.gram_scores(Y, normalize_candidates=False)
fe, fh = A.anchor_curves(Gp, A.noise_factor(Gp, False), true_ids, eligible, hard, A.sigma_grid(Gp, False), n_draws=8)
assert fe.mean(1)[0] > 0.97 and abs(fe.mean(1)[-1] - 0.5) < 0.04, fe.mean(1)[[0, -1]]

# report on a simulated even reader
G = np.load(out / "G_w2v.npy"); L = A.noise_factor(G); sig = A.sigma_grid(G)
curves = {k: (np.load(out / f"anchors_{k}.npz")["f_easy"], np.load(out / f"anchors_{k}.npz")["f_hard"]) for k in ("loud", "w2v")}
S = A.simulate_reader(G, L, true_ids, [A.sigma_for(curves["w2v"][0].mean(1), sig, x) for x in rng.uniform(0.65, 0.8, 33)])
S[2, 4] = np.nan
summary, per_part = RP.model_report(S.astype(np.float16), true_ids, eligible, overlap, hard, band, pool.split,
                                    {"loud": curves["loud"], "even": curves["w2v"]}, n_boot=300)
print("report:", {k: round(summary[k], 3) for k in ("F_easy", "F_hard", "R", "R_loud", "R_even", "p", "top10_paper", "shuffled_control")})
assert abs(summary["shuffled_control"] - 0.5) < 0.04 and len(per_part) == 33 and summary["p_ci95"][0] > 0.3

# probes helpers (the torch parts need the repo)
d = PR.derangement(np.arange(33), rng)
assert all(k != v for k, v in d.items()) and sorted(d.values()) == list(range(33))
assert PR.random_centers(20).shape == (20, 2)
PR.plot_layout(rng.uniform(0, 1, (60, 2)), weights=rng.random(60)).savefig(tmp / "layout.png")

# repo-hook skeletons compile
for f in (ROOT / "ext/repo_hooks").glob("*.py"):
    py_compile.compile(str(f), doraise=True)

# demo: export, then rank lineups with demo/lineup-core.js and compare with Python
slots, bands = DX.candidate_slots(true_ids, eligible, hard, band)
DX.export(tmp / "demo", {"w2v": S.astype(np.float16)}, slots, bands, pool, list(range(33)))
man = json.loads((tmp / "demo/manifest.json").read_text())
qq = np.fromfile(tmp / "demo/scores_w2v.i16", "<i2").reshape(man["P"], man["T"], man["K"])
expect = [[p, t, int(1 + (qq[p, t, 1:50] > qq[p, t, 0]).sum())] for p, t in ((0, 0), (3, 5), (10, 7))]
js = f"""
import {{ readFileSync }} from 'node:fs';
import {{ lineup, probabilities }} from '{(ROOT / 'demo/lineup-core.js').as_posix()}';
const dir = '{(tmp / 'demo').as_posix()}';
const M = JSON.parse(readFileSync(dir + '/manifest.json', 'utf8'));
const b = readFileSync(dir + '/scores_w2v.i16'); const q = new Int16Array(b.buffer, b.byteOffset, b.length / 2);
const D = {{ dir, M, scores: {{ w2v: q }} }};
const ranks = [[0,0],[3,5],[10,7]].map(([p,t]) => {{ const L = probabilities(lineup(D, q, 'w2v', p, t, 'hard', 50));
  return [p, t, L.findIndex(c => c.heard) + 1, Math.round(L.reduce((a, c) => a + c.prob, 0) * 1e6) / 1e6]; }});
console.log(JSON.stringify({{ ranks, easy: lineup(D, q, 'w2v', 1, 2, 'easy', 10).length,
  band0: lineup(D, q, 'w2v', 1, 2, 'easy', 20, 0).length, missing: lineup(D, q, 'w2v', 2, 4, 'hard', 50).length }}));
"""
(tmp / "check.mjs").write_text(js)
res = json.loads(subprocess.run(["node", str(tmp / "check.mjs")], capture_output=True, text=True, check=True).stdout)
print("demo core:", res, "python ranks:", expect)
assert [r[:3] for r in res["ranks"]] == expect and all(r[3] == 1 for r in res["ranks"])
assert res["easy"] == 10 and res["band0"] == 20 and res["missing"] == 0
print("ALL CLI/HOOK CHECKS PASSED")
