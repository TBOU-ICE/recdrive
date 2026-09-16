# Student-to-teacher SFT → OPD controlled comparison

`student_to_teacher_opd_probe_4x32_seed0.json` is a deterministic held-out
probe built from `navtrain_full_val_index.json`:

- 32 samples from each of the four scene buckets (128 total);
- 3 samples per bucket are marked `draw_viz` (12 large figures total);
- every sample stores its absolute cache path and a token-stable diffusion
  noise seed;
- probe tokens are validation samples and are excluded from effective Phase-C
  training by the official `train_logs`/`val_logs` split.

Rebuild (normally unnecessary):

```bash
python scripts/data/build_student_to_teacher_opd_probe_index.py
```

Run the same Phase-C job twice, on separate 8-GPU machines if desired:

```bash
INIT_MODE=sft  MASTER_PORT=23631 \
  bash scripts/training/run_recogdrive_train_student_to_teacher_sft_opd_8gpu.sh

INIT_MODE=base MASTER_PORT=23631 \
  bash scripts/training/run_recogdrive_train_student_to_teacher_sft_opd_8gpu.sh
```

Each timestamped output directory contains:

- `run_manifest.json`: effective dataset/probe/source-index SHA256 and launch
  settings;
- `epoch_order_sha256.jsonl`: global token order hash for every epoch;
- `probe_outputs/step_*/`: all 128 per-token results and aggregate metrics.

After both runs have at least one completed epoch, compare them:

```bash
python scripts/evaluation/compare_student_to_teacher_opd_probe.py \
  --base-run /path/to/base/timestamp \
  --sft-run /path/to/sft/timestamp
```

The comparison exits with an error instead of drawing figures if the two jobs
did not use identical effective data, probe, seed, DDP layout, batch size,
source indexes, or token order.

The effective training union currently has 181,529 unique samples. On eight
GPUs the fixed sampler visits all of them and deterministically repeats the
first 7 entries of that epoch's permutation so every rank receives 22,692
samples (181,536 sampler entries total). These padding entries are included in
the recorded epoch-order SHA256.

## Reading the probe

All reported metrics are lower-is-better:

- `student_teacher_ade_m` / `fde_m`: average/final XY distance from the routed
  privileged teacher; these are the most direct teacher-gap measures.
- `student_gt_ade_m` / `fde_m`: average/final XY distance from logged GT.
- `teacher_gt_ade_m` / `fde_m`: teacher quality on the same sample. A student
  getting closer to a poor teacher is not automatically an improvement.
- `goal_fde_m`: student predicted-goal endpoint error against the GT endpoint.
- `reverse_kl_loss`: deterministic shared-state reverse-transition KL surrogate
  used for the probe; it is stable for comparison but is not the stochastic
  minibatch training loss.
- `x0_gap_m`: mean student/teacher denoised-trajectory gap across all DDIM
  states; `z1_x0_gap_m` and `z3_x0_gap_m` show early/intermediate states.

The top trajectory panel is on the real physical scale. The residual panel
plots `teacher + scale × (student − teacher)` solely to expose small overlap;
it is explicitly non-physical and must not be read as a driven trajectory.
