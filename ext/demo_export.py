"""Step 9.1: small static files for the demo, so it runs from any browser with no server code."""
import json
from pathlib import Path

import numpy as np


def candidate_slots(true_ids, eligible, hard, band, n_easy=200, seed=0):
    """Per true clip: slot 0 = the clip heard, slots 1-49 = hard distractors (most similar first),
    then n_easy random eligible distractors (any prefix of them is a random easy lineup)."""
    rng = np.random.default_rng(seed)
    rests = [np.setdiff1d(np.flatnonzero(eligible[i]), hard[t]) for t, i in enumerate(true_ids)]
    n_easy = min(n_easy, min(len(r) for r in rests))       # keep every row the same length
    rows, bands = [], []
    for t, (i, rest) in enumerate(zip(true_ids, rests)):
        row = np.r_[i, hard[t], rng.choice(rest, n_easy, replace=False)]
        rows.append(row)
        bands.append(np.r_[-1, band[t, row[1:]]])
    return np.stack(rows), np.stack(bands).astype(np.int8)


def export(out_dir, scores: dict, slots, bands, pool, participants):
    """scores: {name: S [P, T, N]} for each model or probe condition shown in the demo."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = {}
    for name, S in scores.items():
        sub = np.take_along_axis(S.astype(np.float32), slots[None], axis=2)      # [P, T, K]
        lo, hi = float(np.nanmin(sub)), float(np.nanmax(sub))
        q = np.round((sub - lo) / (hi - lo) * 65534 - 32767)
        np.where(np.isnan(sub), -32768, q).astype("<i2").tofile(out / f"scores_{name}.i16")
        meta[name] = {"lo": lo, "hi": hi}                                        # -32768 = missing
    manifest = {
        "P": len(participants), "T": int(slots.shape[0]), "K": int(slots.shape[1]),
        "participants": [str(p) for p in participants], "scores": meta,
        "slots": slots.tolist(), "bands": bands.tolist(),
        "clips": pool[["seg", "start", "clip_words", "split"]].round(3).to_dict("records"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest))
