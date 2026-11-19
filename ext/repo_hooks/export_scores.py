"""Step 5.2 skeleton: score every test window against every pool clip with the repo's scoring.

Start from scripts/run_eval_probs.py (R8): fill the four ADAPT functions, then run
    python -m ext.repo_hooks.export_scores <signature> <model> <seed> [condition]
It writes results/S_<model>_s<seed>_<condition>.npy, true_ids.npy, participants.npy and, for
the 'own' condition, results/G_<model>.npy. Step 7 reuses score_test_windows under each probe.
"""
import sys

import numpy as np
import pandas as pd
import torch

from ext.pool import match_windows


def load(sig):
    """ADAPT (R8): load the trained XP the way run_eval_probs does. Return (model, clip_loss,
    test_loader), with model and clip_loss both in eval mode."""
    raise NotImplementedError


def forward(model, batch):
    """ADAPT (R8): the exact call run_eval_probs makes to get the brain estimate Z [B, F, T]."""
    raise NotImplementedError


def batch_targets(batch):
    """ADAPT (R6): the targets the loss compares Z with, [B, F, T] (batch.features after any
    speech module, for Deep Mel)."""
    raise NotImplementedError


def window_info(batch):
    """ADAPT (R8, R9): a DataFrame with one row per window: participant (subject index), seg,
    start. start is the audio window's start in sound-file time, not the EEG's (shifted 150 ms)."""
    raise NotImplementedError


def score_test_windows(model, clip, loader, Yp, pool, chunk=256):
    rows, scores, rel_err = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to("cuda")                    # ADAPT if batches move differently
            Z = forward(model, batch)
            info = window_info(batch)
            info["clip"] = match_windows(info, pool)
            Yb = batch_targets(batch).float()
            Yc = Yp[torch.as_tensor(info.clip.to_numpy(), device=Yp.device)].float()
            rel_err.append(((Yb - Yc).flatten(1).norm(dim=1) / Yb.flatten(1).norm(dim=1)).max().item())
            scores.append(torch.cat([clip.get_scores(Z, Yp[j:j + chunk].float())
                                     for j in range(0, len(Yp), chunk)], dim=1).half().cpu())
            rows.append(info)
    print(f"largest relative gap between batch and pool targets: {max(rel_err):.1e} (want < 1e-3)")
    return pd.concat(rows, ignore_index=True), torch.cat(scores).numpy()


def assemble(win, flat, n_pool):
    assert not win.duplicated(["participant", "clip"]).any(), "two windows map to one (participant, clip)"
    true_ids, parts = np.sort(win.clip.unique()), np.sort(win.participant.unique())
    S = np.full((len(parts), len(true_ids), n_pool), np.nan, np.float16)
    S[np.searchsorted(parts, win.participant), np.searchsorted(true_ids, win.clip)] = flat
    return S, true_ids, parts


def gram(clip, Yp, rows=128, cols=1024):
    """G[i, c] = get_scores(Y_i -> Y_c) over the whole pool, in chunks (the anchors' input)."""
    with torch.no_grad():
        return torch.cat([torch.cat([clip.get_scores(Yp[a:a + rows].float(), Yp[b:b + cols].float())
                                     for b in range(0, len(Yp), cols)], dim=1).cpu()
                          for a in range(0, len(Yp), rows)]).double().numpy()


def main(sig, model_name, seed, cond="own"):
    pool = pd.read_csv("results/pool.csv")
    Yp = torch.from_numpy(np.load(f"results/Y_{model_name}.npy")).cuda()
    model, clip, loader = load(sig)
    win, flat = score_test_windows(model, clip, loader, Yp, pool)
    S, true_ids, parts = assemble(win, flat, len(pool))
    np.save(f"results/S_{model_name}_s{seed}_{cond}.npy", S)
    np.save("results/true_ids.npy", true_ids)
    np.save("results/participants.npy", parts)
    if cond == "own":
        np.save(f"results/G_{model_name}.npy", gram(clip, Yp))
    print(S.shape, "test clips:", len(true_ids), "participants:", len(parts))


if __name__ == "__main__":
    main(*sys.argv[1:])
