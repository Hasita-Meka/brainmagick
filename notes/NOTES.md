# Repo facts (Part 2)

Fill every row with the answer and the file and line where you found it.

| # | Question | Answer | Where |
|---|---|---|---|
| R1 | Override that switches the speech target (mel, Deep Mel, wav2vec), and its value for each | | |
| R2 | Override that removes the subject layer; override that selects the regression loss | | |
| R3 | Keys for the seed and the batch size | | |
| R4 | get_scores: divides by the candidate norm? normalizes the estimate? trims time? centres? | | |
| R5 | Mel feature class and file; how features are normalized; where the statistics come from | | |
| R6 | The function that builds batch.features for a window | | |
| R7 | SubjectLayers: attribute and shape. Spatial attention: class, dropout block, softmax line | | |
| R8 | run_eval_probs: how it loads an XP, its test loop, what it saves per window | | |
| R9 | Brennan loader: word table, sound files, onsets, split assignment, test-window start | | |
| R10 | Deep Mel: speech-module attribute and output width F | | |

## Experiment signatures (step 1.6)

| Run | Signature | Notes |
|---|---|---|
| wav2vec (main) | | |
| Deep Mel | | |
| mel + CLIP | | |
| mel, regression | | |
| no subject layer | | |

## One-GPU batch rule (step 3.1)

Batch size for every run: ____ (why: ____)

## Run log (Parts 3 and 6)

| Model | Seed | Signature | Batch | Epochs | Wall time | Top-10, paper metric | F_easy |
|---|---|---|---|---|---|---|---|
