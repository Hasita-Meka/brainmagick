"""Step zero in one command: pool, envelopes, lineups, anchors and a power check, from audio alone.

First pass, before any targets exist:
    python -m ext.step_zero --words notes/word_table.csv --sounds notes/sound_files.csv \
        --windows notes/split_windows.csv
Second pass, once results/Y_w2v.npy exists for results/pool.csv (step 5.1); third pass with the
exact G and the exported test clips (step 5.2):
    python -m ext.step_zero ... --y w2v=results/Y_w2v.npy
    python -m ext.step_zero ... --g w2v=results/G_w2v.npy --true-ids results/true_ids.npy
The pool is deterministic, so every pass rebuilds the same clips in the same order.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import anchors as A
from . import envelope as E
from . import lineups as LU
from . import metrics as M
from . import pool as P
from . import stats as ST


def standardize(Y):
    """Per channel, over clips and time. Swap in the repo's normalization once R5 is answered."""
    Y = np.asarray(Y, np.float64)
    return (Y - Y.mean(axis=(0, 2), keepdims=True)) / Y.std(axis=(0, 2), keepdims=True)


def run(words, sounds, windows=None, true_ids=None, ys=None, gs=None, similarity="cosine",
        normalized=True, out="results", n_draws=32, n_boot=1000, seed=0):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    report = {}

    # 4.1 pool, split tags, eligibility
    raw, slow, seg_dur = {}, {}, {}
    for seg, path in zip(sounds.seg, sounds.path):
        x, sr = E.load_wav(path)
        seg_dur[seg] = len(x) / sr
        raw[seg] = E.log_envelope(x, sr)
        slow[seg] = E.slow(raw[seg])
    pool = P.build_pool(words, seg_dur)
    pool["split"] = P.tag_splits(pool, windows) if windows is not None else "unknown"
    eligible, overlap, repeat = P.eligibility(pool)
    true_ids = np.asarray(pool.index[pool.split == "test"] if true_ids is None else true_ids)
    pool.to_csv(out / "pool.csv", index=False)
    np.save(out / "eligible.npy", eligible)
    np.save(out / "overlap.npy", overlap)
    n_elig = eligible[true_ids].sum(axis=1)
    report["pool"] = {
        "clips": len(pool), "true_clips": len(true_ids),
        "splits": {k: int(v) for k, v in pool.split.value_counts().items()},
        "eligible_per_true_clip_min_median": [int(n_elig.min()), int(np.median(n_elig))],
        "share_removed_by_repeat_rule": round(float((repeat & ~overlap)[true_ids].mean()), 4),
    }

    # 4.2 envelopes; 4.3 similarity and lineups
    clips_raw, clips_slow = E.slice_clips(raw, pool), E.slice_clips(slow, pool)
    np.savez(out / "envelopes.npz", raw=clips_raw, slow=clips_slow)
    sim = LU.similarity(clips_slow, center=(similarity == "pearson"))
    hard, band, easy = LU.make_lineups(sim, eligible, true_ids, seed=seed)
    np.savez(out / "lineups.npz", true_ids=true_ids, hard=hard, band=band, easy=easy)
    report["hardness"] = {k: v.tolist()
                          for k, v in LU.hardness_report(sim, eligible, true_ids, hard).items()}

    # 4.4 anchors: two loudness readers from our envelopes, one even reader per target space
    spaces = {"loud": ("Y", standardize(clips_slow[:, None, :])),
              "loud_raw": ("Y", standardize(clips_raw[:, None, :]))}
    spaces.update({k: ("Y", np.load(v, mmap_mode="r")) for k, v in (ys or {}).items()})
    spaces.update({k: ("G", np.load(v)) for k, v in (gs or {}).items()})
    curves = {}
    for name, (kind, arr) in spaces.items():
        if arr.shape[0] != len(pool):
            raise ValueError(f"{name}: {arr.shape[0]} rows, but the pool has {len(pool)} clips; "
                             "export it again for results/pool.csv")
        G = A.gram_scores(arr, normalize_candidates=normalized) if kind == "Y" else np.asarray(arr, float)
        L, sig = A.noise_factor(G, normalized), A.sigma_grid(G, normalized)
        fe, fh = A.anchor_curves(G, L, true_ids, eligible, hard, sig, n_draws=n_draws, seed=seed)
        curves[name] = (G, L, sig, fe, fh)
        np.savez(out / f"anchors_{name}.npz", sigmas=sig, f_easy=fe, f_hard=fh)

    # 4.5 step-zero check and power check
    report["retention_at_F_easy"] = {
        name: {str(x): round(A.retention(x, A.matched_hard(fe.mean(1), fh.mean(1), x)), 3)
               for x in (0.6, 0.7, 0.8, 0.9)}
        for name, (_, _, _, fe, fh) in curves.items()}
    report["power_check"] = {}
    for name in sorted(set(curves) - {"loud", "loud_raw"}):
        G, L, sig, fe, fh = curves[name]
        targets = np.random.default_rng(seed).uniform(0.6, 0.85, 33)
        S = A.simulate_reader(G, L, true_ids, [A.sigma_for(fe.mean(1), sig, x) for x in targets], seed)
        f_e, f_h, _ = M.fractions(S, true_ids, eligible, hard)
        point, ci, _ = ST.two_way_bootstrap(f_e, f_h, {"loud": curves["loud"][3:], "even": (fe, fh)},
                                            n_boot=n_boot, seed=seed)
        report["power_check"][name] = {"p": round(point["p"], 3),
                                       "p_ci95": [round(ci.loc[0.025, "p"], 3), round(ci.loc[0.975, "p"], 3)]}
    _plot(curves, out / "step_zero.png")
    (out / "step_zero_report.json").write_text(json.dumps(report, indent=2))
    return report


def _plot(curves, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for name, (*_, fe, fh) in curves.items():
        FE, R = A.retention_curve(fe, fh)
        keep = (FE > 0.52) & (FE < 0.97)
        ax.plot(FE[keep], R[keep], label=name)
    ax.set_xlabel("easy-lineup fraction beaten")
    ax.set_ylabel("retention on the hard lineup")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Step zero: everything that needs only audio.")
    ap.add_argument("--words", required=True, help="CSV: seg, onset (s within its sound file), word")
    ap.add_argument("--sounds", required=True, help="CSV: seg, path (one wav per sound file)")
    ap.add_argument("--windows", help="CSV: seg, start, stop, split (the repo's windows, R9)")
    ap.add_argument("--true-ids", help=".npy of test clip ids written by export_scores.py")
    ap.add_argument("--y", action="append", default=[], metavar="NAME=PATH", help="[N, F, T] targets")
    ap.add_argument("--g", action="append", default=[], metavar="NAME=PATH", help="exact [N, N] G")
    ap.add_argument("--similarity", choices=["cosine", "pearson"], default="cosine")
    ap.add_argument("--plain-inner-product", action="store_true",
                    help="use if R4 shows get_scores does not divide by the candidate norm")
    ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)
    pairs = lambda items: dict(item.split("=", 1) for item in items)
    report = run(pd.read_csv(a.words), pd.read_csv(a.sounds),
                 pd.read_csv(a.windows) if a.windows else None,
                 np.load(a.true_ids) if a.true_ids else None,
                 pairs(a.y), pairs(a.g), a.similarity, not a.plain_inner_product, a.out)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
