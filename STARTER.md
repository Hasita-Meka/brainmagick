# Starter code: EEG speech decoding, why it works and how much is loudness

Copy ext/, notes/, demo/ and tests/ into the root of your brainmagick fork.

- ext/pool.py, envelope.py, lineups.py, anchors.py, metrics.py, stats.py, report.py,
  demo_export.py, step_zero.py: the audio-only pipeline, metrics, anchors and statistics.
  Tested end to end on synthetic audio (tests/).
- ext/probes.py: Aim 1 probes. Needs torch and the attention patch from step 7.2; not run here.
- ext/repo_hooks/: skeletons that call into the repo. Fill the ADAPT functions from notes/NOTES.md.
- notes/NOTES.md: repo facts to record. notes/PREDICTIONS.md: commit and tag by Nov 9.
- demo/lineup-core.js: the data side of the demo.

Run the checks (numpy, scipy, pandas, matplotlib; Node 18+ for the demo check):

    python tests/test_synthetic.py
    python tests/test_cli_and_hooks.py
