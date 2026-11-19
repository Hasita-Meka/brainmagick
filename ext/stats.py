"""Step 8.5: two-way bootstrap (participants x clips) for R, the anchors and p; paired tests."""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .anchors import matched_hard, position, retention


def summarize(fe, fh, anchors, pi=None, ti=None):
    """One estimate on participants pi and clips ti (default: all).
    anchors: {'loud': (f_easy, f_hard), 'even': (f_easy, f_hard)}, curves [n_sigma, T];
    each anchor is matched to the model's mean F_easy (the across-participant choice)."""
    pi = np.arange(fe.shape[0]) if pi is None else pi
    ti = np.arange(fe.shape[1]) if ti is None else ti
    FE = np.nanmean(np.nanmean(fe[np.ix_(pi, ti)], axis=1))
    FH = np.nanmean(np.nanmean(fh[np.ix_(pi, ti)], axis=1))
    out = {"F_easy": FE, "F_hard": FH, "R": retention(FE, FH)}
    for name, (ae, ah) in anchors.items():
        out[f"R_{name}"] = retention(FE, matched_hard(ae[:, ti].mean(1), ah[:, ti].mean(1), FE))
    if {"loud", "even"} <= set(anchors):
        out["p"] = position(out["R"], out["R_loud"], out["R_even"])
    return out


def two_way_bootstrap(fe, fh, anchors, n_boot=2000, seed=0):
    """Resample participants and clips independently: all 33 people heard the same ~190 clips,
    so clips are a second source of variance. Returns point estimate, 95% CI table, replicates."""
    rng = np.random.default_rng(seed)
    P, T = fe.shape
    boot = pd.DataFrame([summarize(fe, fh, anchors, rng.integers(0, P, P), rng.integers(0, T, T))
                         for _ in range(n_boot)])
    return summarize(fe, fh, anchors), boot.quantile([0.025, 0.975]), boot


def paired_test(a, b):
    """Wilcoxon signed-rank across participants (a, b: one value per participant)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    return wilcoxon(a[ok], b[ok])


def bootstrap_diff(a, b, n_boot=2000, seed=0):
    """Mean of a - b (both [P, T], same windows, e.g. own vs swapped subject layer) with a
    two-way bootstrap 95% CI. Returns (point, low, high)."""
    rng = np.random.default_rng(seed)
    d = np.asarray(a, float) - np.asarray(b, float)
    P, T = d.shape
    est = lambda pi, ti: np.nanmean(np.nanmean(d[np.ix_(pi, ti)], axis=1))
    reps = [est(rng.integers(0, P, P), rng.integers(0, T, T)) for _ in range(n_boot)]
    return est(np.arange(P), np.arange(T)), *np.percentile(reps, [2.5, 97.5])
