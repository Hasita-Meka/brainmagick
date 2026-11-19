"""Step 8.1: lineup metrics from a model's score matrix.

S[p, t, c] = score of participant p's test window for true clip true_ids[t] against pool clip c,
computed with the repo's own scoring (step 5). A NaN row means that window is missing.
Average within each participant first, then across participants (the paper's convention).
"""
import numpy as np
from scipy.stats import hypergeom


def _beaten(dist, true):
    return (dist < true).mean(axis=1) + 0.5 * (dist == true).mean(axis=1)


def _rows(S, t, i):
    s = S[:, t, :].astype(np.float32)
    ok = ~np.isnan(s[:, i])
    return s[ok], ok


def fractions(S, true_ids, eligible, hard, band=None, n_bands=10):
    """Fraction beaten (primary metric, chance 0.5): easy = exact over all eligible distractors,
    hard = over the 49 envelope-matched ones, and optionally one value per similarity band."""
    P, T, _ = S.shape
    fe, fh = np.full((P, T), np.nan), np.full((P, T), np.nan)
    fb = None if band is None else np.full((P, T, n_bands), np.nan)
    for t, i in enumerate(true_ids):
        s, ok = _rows(S, t, i)
        if not ok.any():
            continue
        true = s[:, [i]]
        fe[ok, t] = _beaten(s[:, np.flatnonzero(eligible[i])], true)
        fh[ok, t] = _beaten(s[:, hard[t]], true)
        if band is not None:
            for b in range(n_bands):
                fb[ok, t, b] = _beaten(s[:, np.flatnonzero(band[t] == b)], true)
    return fe, fh, fb


def topk_easy_exact(S, true_ids, eligible, ks=(1, 10), lineup=50):
    """Expected top-k over every random lineup of `lineup` clips: the number of distractors that
    outscore the true clip among the lineup - 1 drawn is hypergeometric."""
    P, T, _ = S.shape
    out = {k: np.full((P, T), np.nan) for k in ks}
    for t, i in enumerate(true_ids):
        s, ok = _rows(S, t, i)
        D = np.flatnonzero(eligible[i])
        above = (s[:, D] > s[:, [i]]).sum(axis=1)
        for k in ks:
            out[k][ok, t] = hypergeom.cdf(k - 1, len(D), above, lineup - 1)
    return out


def topk_hard(S, true_ids, hard, ks=(1, 10)):
    P, T, _ = S.shape
    out = {k: np.full((P, T), np.nan) for k in ks}
    for t, i in enumerate(true_ids):
        s, ok = _rows(S, t, i)
        rank = 1 + (s[:, hard[t]] > s[:, [i]]).sum(axis=1)
        for k in ks:
            out[k][ok, t] = rank <= k
    return out


def topk_paper(S, true_ids, ks=(1, 10)):
    """The paper's metric: candidates are all test clips, overlapping neighbours included."""
    cols = np.asarray(true_ids)
    P, T, _ = S.shape
    out = {k: np.full((P, T), np.nan) for k in ks}
    for t, i in enumerate(true_ids):
        s, ok = _rows(S, t, i)
        rank = 1 + (s[:, cols] > s[:, [i]]).sum(axis=1)
        for k in ks:
            out[k][ok, t] = rank <= k
    return out


def group(x):
    """Mean within participant, then across participants (x: [P, T] or [P, T, B])."""
    return np.nanmean(np.nanmean(x, axis=1), axis=0)


def split_check(S, true_ids, eligible, split):
    """Change 5: F_easy over train-split distractors vs held-out ones. They should match."""
    split = np.asarray(split)
    out = {}
    for name, mask in {"train": split == "train",
                       "held_out": np.isin(split, ["valid", "test"])}.items():
        fe, _, _ = fractions(S, true_ids, eligible & mask[None, :],
                             hard=np.zeros((len(true_ids), 1), int))
        out[name] = group(fe)
    return out


def shuffled_control(S, true_ids, eligible, overlap, seed=0):
    """Pair each window with a random non-overlapping test clip; F should come out near 0.5."""
    rng = np.random.default_rng(seed)
    vals = []
    for t, i in enumerate(true_ids):
        j = rng.choice([c for c in true_ids if not overlap[i, c]])
        s, ok = _rows(S, t, j)
        D = np.flatnonzero(eligible[j] & ~overlap[i])
        vals.append(_beaten(s[:, D], s[:, [j]]))
    return float(np.mean(np.concatenate(vals)))
