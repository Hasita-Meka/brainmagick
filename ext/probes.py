"""Step 7: evaluation-only probes on a trained model.

region_removed needs the small patch to the spatial-attention module shown in step 7.2. The
attribute names (weights, eval_ban_center) follow our reading of the code: confirm them (R7).
"""
import contextlib

import numpy as np

REGIONS = {  # centres in the attention module's [0, 1] x [0, 1] layout; confirm on the plot
    "frontopolar": (0.50, 0.92),
    "fronto-central": (0.50, 0.62),
    "left temporal": (0.08, 0.50),
    "right temporal": (0.92, 0.50),
    "occipital": (0.50, 0.08),
}


@contextlib.contextmanager
def region_removed(attention, center, radius=0.2):
    """Drop every sensor within `radius` of `center` from the attention softmax, as training-time
    spatial dropout does around a random point (radius 0.2, Sec. 2.2.2)."""
    attention.eval_ban_center, attention.eval_ban_radius = tuple(center), radius
    try:
        yield
    finally:
        attention.eval_ban_center = None


@contextlib.contextmanager
def subject_layers_swapped(layer, trained_ids, donors=None, mean=False):
    """Temporarily replace the per-subject matrices (layer.weights: [n_subjects, C, C]).
    donors {subject: donor}: each listed participant gets the donor's matrix.
    mean=True: every trained participant gets the average of the trained rows. Rows beyond the
    participants were never trained, so they never enter the average."""
    import torch
    W = layer.weights
    backup = W.data.clone()
    idx = torch.as_tensor(np.asarray(trained_ids), device=W.device)
    try:
        if mean:
            W.data[idx] = backup[idx].mean(dim=0, keepdim=True)
        for subject, donor in (donors or {}).items():
            W.data[subject] = backup[donor]
        yield
    finally:
        W.data.copy_(backup)


def derangement(ids, rng):
    """{participant: another participant}: a shuffle with no fixed points."""
    ids = np.asarray(ids)
    while True:
        perm = rng.permutation(ids)
        if not np.any(perm == ids):
            return dict(zip(ids.tolist(), perm.tolist()))


def random_centers(n=20, seed=0):
    """Centres for the random-patch control: same radius as the regions, random places."""
    return np.random.default_rng(seed).uniform(0, 1, size=(n, 2))


def plot_layout(positions, names=None, weights=None, regions=REGIONS, radius=0.2):
    """The sensor layout as the attention module sees it, with the region discs drawn on top.
    Pass weights (one per sensor) to colour it as an attention map, like Fig. A.3."""
    import matplotlib.pyplot as plt
    pos = np.asarray(positions)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(pos[:, 0], pos[:, 1], c=weights, cmap="coolwarm", s=70, edgecolors="k", linewidths=0.3)
    for k, (x, y) in enumerate(pos):
        ax.annotate(str(names[k]) if names is not None else str(k), (x, y), fontsize=7,
                    ha="center", va="bottom", xytext=(0, 4), textcoords="offset points")
    for name, c in regions.items():
        ax.add_patch(plt.Circle(c, radius, fill=False, linestyle="--"))
        ax.text(c[0], c[1], name, fontsize=8, ha="center", va="center")
    ax.set_aspect("equal")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    return fig
