"""Step 8: one model's read-out from its score matrix, the lineups and the anchors."""
import numpy as np
import pandas as pd

from . import metrics as M
from . import stats as ST


def load_anchor(path):
    z = np.load(path)
    return z["f_easy"], z["f_hard"]


def model_report(S, true_ids, eligible, overlap, hard, band, split, anchors, n_boot=2000, seed=0):
    """anchors: {'loud': (f_easy, f_hard), 'even': (f_easy, f_hard)} for this model's target
    space; pass only 'loud' for the envelope model, whose two anchors coincide.
    Returns (summary dict, one row per participant)."""
    fe, fh, fb = M.fractions(S, true_ids, eligible, hard, band)
    point, ci, _ = ST.two_way_bootstrap(fe, fh, anchors, n_boot=n_boot, seed=seed)
    paper, easy, hard50 = (M.topk_paper(S, true_ids), M.topk_easy_exact(S, true_ids, eligible),
                           M.topk_hard(S, true_ids, hard))
    summary = {k: float(v) for k, v in point.items()}
    summary.update({f"{k}_ci95": [float(ci.loc[0.025, k]), float(ci.loc[0.975, k])] for k in point})
    summary.update({
        "band_curve": M.group(fb).round(4).tolist(),
        "top1_paper": float(M.group(paper[1])), "top10_paper": float(M.group(paper[10])),
        "top10_easy50": float(M.group(easy[10])), "top10_hard50": float(M.group(hard50[10])),
        "split_check": {k: float(v) for k, v in M.split_check(S, true_ids, eligible, split).items()},
        "shuffled_control": M.shuffled_control(S, true_ids, eligible, overlap, seed=seed),
        "chance": {"F": 0.5, "top10_paper": 10 / len(true_ids), "top10_lineup50": 0.2},
    })
    per_participant = pd.DataFrame({"F_easy": np.nanmean(fe, 1), "F_hard": np.nanmean(fh, 1),
                                    "top10_paper": np.nanmean(paper[10], 1)})
    return summary, per_participant
