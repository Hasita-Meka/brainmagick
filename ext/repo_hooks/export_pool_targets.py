"""Step 5.1 skeleton: the target Y of every pool clip, through the repo's own feature code.

Fill the ADAPT functions from recon items R5, R6, R9 (and R10 for Deep Mel), then run
    python -m ext.repo_hooks.export_pool_targets w2v
again for the envelope target (step 6.1) and, after training, for Deep Mel (step 6.4).
The wav2vec and envelope targets need no trained model, only the cached features.
"""
import sys

import numpy as np
import pandas as pd


def recording_and_offsets():
    """ADAPT (R9): one participant's recording object, and {seg: start of that sound file in the
    recording, in seconds}. Everyone heard the same audio, so one recording is enough."""
    raise NotImplementedError("fill from the brennan2019 study loader")


def window_features(recording, t0, t1):
    """ADAPT (R6, R5): the [F, T] array the dataset puts in batch.features for the window
    [t0, t1) of this recording, after the same normalization as during training."""
    raise NotImplementedError("copy the call made in the dataset's __getitem__")


def speech_module(y):
    """ADAPT (R10), Deep Mel only: run the mel features through the trained speech module."""
    raise NotImplementedError("load the Deep Mel XP and call its speech module")


def main(target):
    pool = pd.read_csv("results/pool.csv")
    rec, offsets = recording_and_offsets()
    Y = []
    for seg, start in zip(pool.seg, pool.start):
        t0 = offsets[seg] + start
        y = np.asarray(window_features(rec, t0, t0 + 3.0), np.float32)
        Y.append(np.asarray(speech_module(y), np.float32) if target == "deepmel" else y)
    Y = np.stack(Y).astype(np.float16)
    np.save(f"results/Y_{target}.npy", Y)
    print(target, Y.shape)


if __name__ == "__main__":
    main(sys.argv[1])
