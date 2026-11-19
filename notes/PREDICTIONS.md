# Predictions and decision rule

Committed on ____, before any lineup result from a trained model. Tag: predictions-v1.

## Lineups
- Pool: one clip per word, [onset - 0.5 s, onset + 2.5 s], kept if it fits inside its sound file. N = ____.
- Distractors exclude the clip itself, clips sharing audio with it (same file, onsets < 3 s apart) and clips repeating a run of >= 3 of its words (runs of function words only don't count). Share removed by the repeat rule: ____.
- Hard lineup: the 49 eligible clips most similar to the true clip by [cosine | Pearson] on the slow log envelope (8 Hz zero-phase low-pass). Ten equal-size similarity bands for the difficulty curve.
- Easy lineup: exact expectation over all eligible distractors.

## Metrics
- Primary: fraction beaten, F (chance 0.5). Secondary: top-1 and top-10 in 50-clip lineups, and the paper's top-10 over the ~190 test clips.
- Average within each participant, then across participants.

## Anchors and read-out
- Loudness anchor in the [slow | raw] envelope space; even reader in each model's own target space; isotropic Gaussian noise with 32 fixed draws per clip; matched to the model's across-participant mean F_easy.
- R = (F_hard - 0.5) / (F_easy - 0.5); p = (R - R_loud) / (R_even - R_loud).
- Uncertainty: two-way bootstrap over participants and clips, 2,000 replicates, 95% percentile intervals. Wilcoxon signed-rank tests across participants for paired comparisons.

## Decision rule (copy the zones and thresholds from the plan's prediction table)
- p below ____ means ____
- p from ____ to ____ means ____
- p above ____ means ____
- The envelope model validates the anchors if its R falls inside the bootstrap interval of R_loud at its own F_easy.

## Aim 1 expectations
- wav2vec target, regression vs contrastive loss: ____
- Envelope vs Deep Mel vs wav2vec, paper-style top-10: ____
- No subject layer: ____ (paper: 25.7% to 20.2%)
- Own vs another participant's vs the mean subject layer: ____
- Removing frontopolar, fronto-central, left and right temporal, occipital sensors: ____
