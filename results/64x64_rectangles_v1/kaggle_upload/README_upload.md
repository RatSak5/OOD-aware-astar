# Initial Kaggle training bundle — 64x64_rectangles_v1

This bundle contains 1,140 manifest records: training (900), validation/calibration (180), and ID pilot (60). The filtered manifest includes only these splits and D1-D3.

The notebook requires CUDA and checks forward/backward/optimizer execution only for Neural A* and the independent OOD encoder. H1/H2/H3 remain optional secondary baselines. Each notebook startup allocates a unique output directory beneath `results/64x64_rectangles_v1/runs/`; outputs from distinct runs are isolated.

The OOD-pilot archive and per-map records are in a separate local bundle and are excluded here. ID holdout and sealed OOD test archives and per-map records are excluded. NPZs are copied and hashed as opaque bytes; this builder does not parse them.

Proposed Kaggle destination: title 64x64 Rectangles v1 Phase B; slug 64x64-rectangles-v1-phase-b. Not uploaded. Verify owner and private visibility before future upload.

No credentials, checkpoints, Prompt 8 artifacts, or unrelated personal files are packaged.
