"""Steps 4.1-4.2: candidate pool, split tags and distractor eligibility.

Everyone in Brennan & Hale (2019) heard the same sound files, so the pool is audio-only:
one 3-s clip per word, starting 0.5 s before its onset (the paper's test-window convention).
"""
from collections import defaultdict
import re

import numpy as np
import pandas as pd

TMIN, DUR = -0.5, 3.0   # clip = [onset - 0.5 s, onset + 2.5 s]
NGRAM = 3               # repeated-phrase rule: a shared run of >= 3 words ...
STOP = set("""a an the and or but of to in on at by for with from as into than then so
i you he she it we they me him her us them my your his its our their this that these those
is was were be been being am are had has have do did does not no""".split())
                        # ... that contains at least one word outside STOP


def _norm(word) -> str:
    return re.sub(r"[^a-z']", "", str(word).lower())


def build_pool(words: pd.DataFrame, seg_dur: dict) -> pd.DataFrame:
    """words: one row per word, columns seg (sound-file id), onset (s, within that file), word.
    seg_dur: {seg: duration of that sound file in s}. Returns one row per usable clip."""
    w = words.assign(word_n=words["word"].map(_norm))
    w = w[w.word_n != ""].sort_values(["seg", "onset"], kind="stable").reset_index(drop=True)
    w["start"] = w["onset"] + TMIN
    w["stop"] = w["start"] + DUR
    inside = (w["start"] >= 0) & (w["stop"] <= w["seg"].map(seg_dur))
    pool = w[inside].reset_index(drop=True)
    by_seg = {s: g for s, g in w.groupby("seg")}
    pool["clip_words"] = [
        " ".join(by_seg[s].word_n[(by_seg[s].onset >= a) & (by_seg[s].onset < b)])
        for s, a, b in zip(pool.seg, pool.start, pool.stop)
    ]
    pool.insert(0, "clip_id", np.arange(len(pool)))
    return pool


def _merge(iv: np.ndarray) -> np.ndarray:
    iv = iv[np.argsort(iv[:, 0])]
    out = [iv[0].copy()]
    for a, b in iv[1:]:
        if a <= out[-1][1] + 1e-6:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append(np.array([a, b], float))
    return np.array(out)


def tag_splits(pool: pd.DataFrame, windows: pd.DataFrame) -> pd.Series:
    """windows: one row per repo window (seg, start, stop, split) in sound-file time.
    Windows are merged into covered regions per split; a clip takes the split whose region
    contains it entirely, otherwise 'gap' (straddles two splits or falls in a dropped zone)."""
    regions = {
        (s, sp): _merge(g[["start", "stop"]].to_numpy(float))
        for (s, sp), g in windows.groupby(["seg", "split"])
    }
    tags = []
    for s, a, b in zip(pool.seg, pool.start, pool.stop):
        hits = {sp for (seg, sp), r in regions.items()
                if seg == s and np.any((r[:, 0] <= a + 1e-6) & (r[:, 1] >= b - 1e-6))}
        tags.append(hits.pop() if len(hits) == 1 else "gap")
    return pd.Series(tags, index=pool.index, name="split")


def eligibility(pool: pd.DataFrame, ngram: int = NGRAM, stop=STOP):
    """eligible[i, c]: may clip c be a distractor for true clip i?
    Excludes the clip itself, clips sharing audio with it (same file, onsets < 3 s apart)
    and clips repeating a run of >= `ngram` of its words (runs made only of `stop` words,
    such as "out of the", don't count)."""
    n = len(pool)
    seg, on = pool.seg.to_numpy(), pool.onset.to_numpy(float)
    overlap = (seg[:, None] == seg[None, :]) & (np.abs(on[:, None] - on[None, :]) < DUR)
    grams = []
    for text in pool.clip_words:
        ws = text.split()
        runs = (tuple(ws[k:k + ngram]) for k in range(len(ws) - ngram + 1))
        grams.append({g for g in runs if not set(g) <= stop})
    owners = defaultdict(list)
    for c, gs in enumerate(grams):
        for g in gs:
            owners[g].append(c)
    repeat = np.zeros((n, n), bool)
    for c_list in owners.values():
        idx = np.array(c_list)
        repeat[np.ix_(idx, idx)] = True
    eligible = ~(overlap | repeat)
    np.fill_diagonal(eligible, False)
    return eligible, overlap, repeat


def match_windows(win: pd.DataFrame, pool: pd.DataFrame, tol: float = 1 / 120) -> np.ndarray:
    """Pool clip id of each exported window (columns seg, start: the audio window's start in
    sound-file time, not the EEG's, which the model shifts by 150 ms). Raises if any window
    is further than `tol` from every clip start."""
    win = win.reset_index(drop=True)
    out = np.full(len(win), -1)
    for s, g in win.groupby("seg"):
        cand = pool[pool.seg == s].sort_values("start")
        starts, ids = cand.start.to_numpy(float), cand.clip_id.to_numpy()
        if len(starts) == 0:
            continue
        x = g.start.to_numpy(float)
        k = np.clip(np.searchsorted(starts, x), 0, len(starts) - 1)
        k0 = np.clip(k - 1, 0, len(starts) - 1)
        pick = np.where(np.abs(x - starts[k0]) < np.abs(x - starts[k]), k0, k)
        ok = np.abs(x - starts[pick]) <= tol
        out[g.index.to_numpy()[ok]] = ids[pick[ok]]
    if (out < 0).any():
        raise ValueError(f"{(out < 0).sum()} windows match no pool clip within {tol:.4f} s; "
                         "check the seg ids and the start-time convention")
    return out
