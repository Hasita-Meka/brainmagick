"""Step 4.5: reference decoders (the anchors), retention and position p.

A reference decoder's estimate is the true clip's target plus isotropic Gaussian noise, scored
against every candidate with the models' own rule. When that rule divides by the candidate norm
(our reading of ClipLoss.get_scores, recon item R6), the noisy score is
    score(Y_i + s * eps -> c) = G[i, c] + s * n_c,   n ~ N(0, K),
with G = scores(Y_pool -> Y_pool) and K the cosine matrix between candidates. So an anchor needs
only G: no EEG, and no pass over the full feature tensor once G exists.
"""
import numpy as np


def gram_scores(Y, chunk=8192, normalize_candidates=True):
    """Step-zero stand-in for the repo's get_scores applied to (Y_pool, Y_pool):
    G[i, c] = <Y_i, Y_c> / ||Y_c|| over all of (F, T), or the plain inner product if R4 shows
    the repo doesn't divide by the candidate norm. Y: [N, F, T] array (a memmap is fine).
    Once R4 is answered, compute G with the repo's own function instead (step 5.2)."""
    N = Y.shape[0]
    flat = Y.reshape(N, -1)
    gram = np.zeros((N, N))
    for j in range(0, flat.shape[1], chunk):
        block = np.asarray(flat[:, j:j + chunk], np.float32)
        gram += block @ block.T
    return gram / np.sqrt(np.diag(gram))[None, :] if normalize_candidates else gram


def noise_factor(G, candidates_normalized=True, rel_tol=1e-7):
    """L with L @ L.T = K, the covariance of the noise term across candidates: the cosine
    matrix when the scoring divides by the candidate norm, the Gram matrix itself otherwise."""
    K = G / np.diag(G)[:, None] if candidates_normalized else np.array(G, float)
    K = 0.5 * (K + K.T)
    w, U = np.linalg.eigh(K)
    keep = w > rel_tol * w.max()
    return U[:, keep] * np.sqrt(w[keep])


def sigma_grid(G, candidates_normalized=True, n=80, lo=1e-3, hi=1e2):
    """Noise levels spanning 'near perfect' to 'chance', relative to the typical target norm."""
    norm = np.median(np.diag(G)) if candidates_normalized else np.sqrt(np.median(np.diag(G)))
    return norm * np.logspace(np.log10(lo), np.log10(hi), n)


def _beaten(dist, true):
    return (dist < true).mean(axis=1) + 0.5 * (dist == true).mean(axis=1)


def anchor_curves(G, L, true_ids, eligible, hard, sigmas, n_draws=32, seed=0):
    """Fraction beaten by a noisy reader of G's target space, per noise level and true clip.
    Returns f_easy, f_hard, each [len(sigmas), len(true_ids)]. The noise draws are fixed
    (common random numbers), so both curves change smoothly with sigma."""
    rng = np.random.default_rng(seed)
    sig = np.asarray(sigmas)[:, None, None]
    f_easy = np.zeros((len(sigmas), len(true_ids)))
    f_hard = np.zeros_like(f_easy)
    for t, i in enumerate(true_ids):
        noise = L @ rng.standard_normal((L.shape[1], n_draws))      # [N, draws], cov K
        sc = G[i][None, :, None] + sig * noise[None]                # [S, N, draws]
        true = sc[:, [i], :]
        f_easy[:, t] = _beaten(sc[:, np.flatnonzero(eligible[i]), :], true).mean(axis=-1)
        f_hard[:, t] = _beaten(sc[:, hard[t], :], true).mean(axis=-1)
    return f_easy, f_hard


def matched_hard(f_easy_curve, f_hard_curve, target_easy):
    """The anchor's F_hard at the noise level where its F_easy equals target_easy."""
    fe = np.minimum.accumulate(f_easy_curve)        # non-increasing in sigma
    fe, fh = fe[::-1], f_hard_curve[::-1]            # increasing, for np.interp
    if not fe[0] <= target_easy <= fe[-1]:
        return np.nan
    return float(np.interp(target_easy, fe, fh))


def retention(f_easy, f_hard):
    """Share of the above-chance margin kept on the hard lineup: 1 = no drop, 0 = chance."""
    return (f_hard - 0.5) / (f_easy - 0.5)


def position(R, R_loud, R_even):
    """0 = behaves like a pure loudness reader of the same accuracy; 1 = drops only as much
    as its own target's geometry predicts; above 1 = leans on what envelope matching leaves."""
    return (R - R_loud) / (R_even - R_loud)


def retention_curve(f_easy, f_hard):
    """Mean F_easy and the anchor's retention at each noise level (the step-zero plot)."""
    FE, FH = f_easy.mean(axis=1), f_hard.mean(axis=1)
    return FE, retention(FE, FH)


def sigma_for(f_easy_curve, sigmas, target_easy):
    """Noise level at which a reader with this curve reaches target_easy."""
    fe = np.minimum.accumulate(f_easy_curve)[::-1]
    return float(np.exp(np.interp(target_easy, fe, np.log(np.asarray(sigmas))[::-1])))


def simulate_reader(G, L, true_ids, sigma_per_participant, seed=0):
    """Score matrix [P, T, N] of a synthetic decoder that reads G's target space evenly, with
    one noise level per participant. Step zero uses it as a power check, before any EEG."""
    rng = np.random.default_rng(seed)
    S = np.empty((len(sigma_per_participant), len(true_ids), G.shape[0]), np.float32)
    for p, s in enumerate(sigma_per_participant):
        S[p] = G[true_ids] + s * (L @ rng.standard_normal((L.shape[1], len(true_ids)))).T
    return S
