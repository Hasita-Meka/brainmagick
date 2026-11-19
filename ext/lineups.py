"""Step 4.4: envelope similarity, hard lineups, difficulty bands and one seeded easy lineup."""
import numpy as np


def similarity(slow_clips: np.ndarray, center: bool = False) -> np.ndarray:
    """Similarity between the slow envelopes of every pair of clips (N x N).
    center=False: cosine on the standardized envelope, the geometry the loudness anchor and the
    models' scoring see (level and contour); center=True: Pearson r (contour only)."""
    z = (slow_clips - slow_clips.mean()) / slow_clips.std()
    if center:
        z = z - z.mean(axis=1, keepdims=True)
    z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
    return z @ z.T


def make_lineups(sim, eligible, true_ids, n_hard=49, n_bands=10, seed=0):
    """For each true clip (row t of true_ids):
    hard[t]    the n_hard eligible distractors with the most similar slow envelope
    band[t, c] similarity band of distractor c (0 = most similar decile), -1 if ineligible
    easy[t]    n_hard random eligible distractors (fixed seed), for the demo only;
               easy-lineup metrics are computed exactly over all eligible distractors."""
    rng = np.random.default_rng(seed)
    T, N = len(true_ids), sim.shape[0]
    hard = np.empty((T, n_hard), np.int64)
    easy = np.empty((T, n_hard), np.int64)
    band = np.full((T, N), -1, np.int8)
    for t, i in enumerate(true_ids):
        cand = np.flatnonzero(eligible[i])
        order = cand[np.argsort(-sim[i, cand], kind="stable")]
        hard[t] = order[:n_hard]
        for b, part in enumerate(np.array_split(order, n_bands)):
            band[t, part] = b
        easy[t] = rng.choice(cand, size=n_hard, replace=False)
    return hard, band, easy


def hardness_report(sim, eligible, true_ids, hard):
    """Similarity of the hard distractors vs all eligible ones: is the hard lineup actually hard?"""
    top = np.array([sim[i, hard[t]].mean() for t, i in enumerate(true_ids)])
    allr = np.array([sim[i, eligible[i]].mean() for i in true_ids])
    q = lambda a: np.round(np.percentile(a, [10, 50, 90]), 3)
    return {"hard_r_p10_p50_p90": q(top), "all_eligible_r_p10_p50_p90": q(allr)}
