
"""Write step zero's three inputs into notes/ (guide 4.1, R9).



word_table.csv    seg, onset, word   from Brennan's AliceChapterOne-EEG.csv

sound_files.csv   seg, path          one row per wav file

split_windows.csv seg, start, stop, split

    every window of one participant, from the repo's own train/valid/test datasets,

    in sound-file time. A window that crosses from one sound file into the next

    becomes one row per file, each clipped to that file, so every row lies inside

    a single file.

"""

import re

from pathlib import Path



import pandas as pd

import soundfile as sf



from bm import play



SIG = "4395629c"

DATA = Path("/courses/CS7150.202710/students/meka.ha/data/brennan2019/download")

OUT = Path("notes")





def seg_of(path):

    return int(re.search(r"SoundFile(\d+)\.wav", str(path)).group(1))





story = pd.read_csv(DATA / "AliceChapterOne-EEG.csv")

words = story.rename(columns={"Segment": "seg", "Word": "word"})[["seg", "onset", "word"]]

words.to_csv(OUT / "word_table.csv", index=False)

segs = sorted(int(s) for s in words.seg.unique())

paths = [str(DATA / "audio" / f"DownTheRabbitHoleFinal_SoundFile{s}.wav") for s in segs]

pd.DataFrame({"seg": segs, "path": paths}).to_csv(OUT / "sound_files.csv", index=False)

file_len = {s: sf.info(p).duration for s, p in zip(segs, paths)}

print(f"word_table.csv: {len(words)} words in {len(segs)} sound files")



solver = play.get_solver_from_sig(SIG, override_cfg={"device": "cpu"})

datasets = getattr(solver, "datasets", None)

if datasets is None:

    raise SystemExit(f"no 'datasets' on the solver; attributes: {sorted(vars(solver))}")



rows, uid = [], None

n_windows, n_crossing, n_silent = {}, 0, 0

for split in ["train", "valid", "test"]:

    for dset in getattr(datasets, split).datasets:

        if uid is None:

            uid = dset.recording.subject_uid

        if not dset.recording.subject_uid == uid:

            continue

        snd = dset.events[dset.events.kind == "sound"].sort_values("start").reset_index(drop=True)

        starts = list(snd.start)

        nxt = starts[1:] + [float("inf")]

        ends = [s0 + d if not pd.isna(d) else n for s0, d, n in zip(starts, snd.duration, nxt)]

        offs = list(snd.get("offset", pd.Series([0.0] * len(snd))).fillna(0.0))

        segs_k = [seg_of(fp) for fp in snd.filepath]

        n_windows[split] = len(dset)

        for idx in range(len(dset)):

            start, stop = dset._get_bounds_times(idx)

            parts = []

            for s0, e, off, seg in zip(starts, ends, offs, segs_k):

                a, b = max(start, s0), min(stop, e)

                if b > a:

                    shift = off - s0

                    parts.append(dict(seg=seg, start=round(a + shift, 4),

                                      stop=round(b + shift, 4), split=split))

            if not parts:

                n_silent += 1

            if len(set(p["seg"] for p in parts)) > 1:

                n_crossing += 1

            rows.extend(parts)



win = pd.DataFrame(rows)

win.to_csv(OUT / "split_windows.csv", index=False)

over = max(win.stop[win.seg == s].max() - file_len[s] for s in win.seg.unique())

print(f"participant {uid}: windows {n_windows}, rows {win.split.value_counts().to_dict()}")

print(f"windows crossing two files: {n_crossing}, windows with no audio: {n_silent}")

print(f"earliest start {win.start.min():.3f} s; latest stop past its file's end {over:.3f} s")

