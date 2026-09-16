# GoalBridge teacher-guidance probe

Answers the question the in-training BEV viz cannot: when the routed teacher is
evaluated at the states the student actually visits, **is its guidance sane** —
and if not, is that the teacher being weak, the student's `z_t` having left the
teacher's training distribution, or the chain and the teacher disagreeing about
the goal?

Pure inference on one GPU. Trains nothing, writes no checkpoint, and is not
imported by the training path.

## What it computes

At every DDIM step, up to four x0 predictions, all denormalised to metres:

| key | `z_t` source | goal conditioning | colour |
|---|---|---|---|
| `student_pred` | student DDIM chain | student goal | blue |
| `teacher_gt` | student DDIM chain | GT goal | red |
| `teacher_pred` | student DDIM chain | student predicted goal | dark yellow `#B8860B` |
| `teacher_qgt` | forward-noised GT `sqrt(a)x0 + sqrt(1-a)eps` | GT goal | green |

`teacher_gt` is the condition already drawn during training. The others are the
controls, each differing from it in exactly one variable:

- `teacher_gt` vs `teacher_pred` — same `z_t`, goal swapped → **goal-conditioning
  conflict**. The chain is rolled out under the predicted goal while the teacher
  is asked to denoise towards the GT goal.
- `teacher_gt` vs `teacher_qgt` — same goal, `z_t` swapped for the teacher's own
  training marginal → **student `z_t` off-distribution** vs a teacher that is
  simply weak at this noise level. This is the decisive control.
- `student_pred` vs `teacher_gt` — the gap the KD term actually pulls on. Note it
  varies two things at once (model *and* goal), since the privileged teacher is
  meant to use the GT goal; do not read that gap as teacher weakness.

Plus two scalars per step that the BEV panels cannot give you:

- `clamp_frac` — per-sample fraction of x0 on the ±`denoised_clip_value` boundary.
  x0 is reconstructed as `(z - sqrt(1-a) eps_hat)/sqrt(a)`, so at early steps a
  small `eps_hat` error is amplified and then clamped. High value ⇒ the ragged
  trajectory is a truncation artefact, not bad guidance.
- `z_second_moment_ratio` — `E[z_t^2] / (a E[x0_norm^2] + 1-a)`. Away from 1 is
  direct evidence the student's chain leaves the teacher's noise scale, with no
  eyeballing. Worth watching because `_sample_chain` injects noise through
  `_safe_sigma(..., min_sigma)` while `ddim_eta = 0`.

## Running

```bash
STUDENT_CKPT=/path/to/goalbridge_student.ckpt \
scripts/diagnosis/run_goalbridge_guidance_diagnostic.sh
```

Two runs by default (`RUNS=both`; also `after` / `base`):

- `opd_after` — the GoalBridge student under test, `student_goal_source=pred`.
- `base_il` — the goal-free IL checkpoint OPD starts from,
  `student_goal_source=none`. Its goal head and goal encoder were never trained,
  and for `goal_mode=adaln` the goal encoder's last layer is zero-initialised, so
  an untrained goal embedding is added as zero — "goal-free" is what the model
  already does. `teacher_pred` is skipped for this run.

Together they separate **already broken at init** from **degraded during OPD**.

The default probe is balanced across all four routed scene buckets:
`SAMPLES_PER_BUCKET=64`, i.e. 256 samples per checkpoint. Useful env vars:
`SAMPLES_PER_BUCKET`, `BATCH_SIZE` (8), `SEED` (0), `AGENT`, and `OUT_DIR`.

## Outputs

Per run, under `$OUT_DIR`:

- `<tag>_summary.png` — **read this first.** Per-step ADE per condition, clamp
  saturation, and the `z_t` second-moment ratio.
- `<tag>_bev_by_error/` — all 256 samples, split into four equal error strata:
  `q1_low`, `q2_mid_low`, `q3_mid_high`, and `q4_high`. Each PNG contains one
  sample only, with four large panels showing z1 through z4 left-to-right.
  Panels share per-sample axis limits and use 2 m major / 1 m minor ticks.
  `opd_after` is stratified by goal error; goal-free `base_il` uses final
  student FDE.
- `<tag>_payload.pt` — raw tensors, so restyling needs no re-inference:

```bash
python -m navsim.planning.training.goalbridge_guidance_plot \
    $OUT_DIR/opd_after_payload.pt
```

## Reading the result

- `teacher_qgt` clean **and** `teacher_gt` ragged ⇒ the teacher is fine; the
  student's states are off-distribution.
- `teacher_qgt` also ragged ⇒ the teacher itself is weak at that noise level;
  nothing downstream of that is worth investigating first.
- `teacher_pred` clean but `teacher_gt` ragged ⇒ the goal-conditioning conflict
  dominates, not the noise distribution.
- High `clamp_frac` ⇒ discount the panel entirely; it is saturation.
