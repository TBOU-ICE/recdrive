# Privileged-OPD v2 — Final Plan & Runbook

This implementation is **additive**. Existing ReCogDrive/GoalBridge files and training scripts are not imported or modified by the new jobs. New code lives under:

- `navsim/agents/recogdrive/privileged_opd_v2/`
- `scripts/training/privileged_opd_v2/`
- `scripts/data/privileged_opd_v2/`
- two new training entrypoints under `navsim/planning/script/`
- two new Hydra agent configs under `navsim/planning/script/config/common/agent/`

## 0. Two decisions

**Teacher stages 1–2 data**

- **Stage 1 IL:** use the existing common **fullmix NAV + SimScale** IL checkpoint (`STAGE1_IL_CKPT`). Do not retrain it if already available.
- **Stage 2 RL:** do **not** use all NAV indiscriminately. Keep the existing scene-expert recipe: each expert starts from Stage-1 IL and trains on its matching scene data. The existing scripts default to `NAV full=0.0, NAV bucket=0.6, SimScale bucket=0.4`.
- **Stage 3 privilege:** no general-data mixing. Start from each old goal-free Stage-2 RL expert and train only the new goal branch on the same matching NAV/SimScale scene bucket. Default here is again 60/40 NAV-bucket/SimScale-bucket.

**OPD teacher feedback**

Keep **hard scene routing**: every training sample is supervised by exactly **one** of the four teachers. Do not average all four outputs in the main experiment; that reintroduces multi-modal/multi-expert averaging.

---

# 1. Teacher training

## Stage 1 — Common goal-free IL

Use your existing fullmix checkpoint:

```bash
export STAGE1_IL_CKPT=/.../training_dit_il_fullmix_simscale...ckpt
```

Conceptually:

```text
NAV full + SimScale -> goal-free IL -> common DiT checkpoint
```

## Stage 2 — Four goal-free RL scene experts

Reuse your old successful expert-RL recipe. All four start from the same Stage-1 IL checkpoint, then specialize by scene:

```bash
bash scripts/training/privileged_opd_v2/run_stage2_rl_progress.sh
bash scripts/training/privileged_opd_v2/run_stage2_rl_rule.sh
bash scripts/training/privileged_opd_v2/run_stage2_rl_safety.sh
bash scripts/training/privileged_opd_v2/run_stage2_rl_general.sh
```

Defaults inherited from the existing expert script:

```text
NAV full      0.0
NAV bucket    0.6
Sim bucket    0.4
```

If these four old goal-free IL+RL checkpoints already exist, **skip Stage 2** and point `RL_*_CKPT` to them.

## Stage 3 — Lightweight privileged goal adaptation

Main configuration (A):

```text
init            = corresponding old goal-free IL+RL expert
backbone        = frozen
privilege       = correct GT goal only (no noisy/masked goal)
goal points     = final point (A baseline)
injection       = zero-init gated residual cross-attention
trainable       = GoalEncoder + goal adapter + scalar gate
NAV/Sim ratio   = 0.6 / 0.4 within the SAME scene bucket
General mixing  = none
epochs          = 10 default; normally inspect 5–10
adapter LR      = 5e-5
last DiT blocks = 0 by default
```

Why: with `gated_cross`, gate starts at exactly 0, so the initialized Goal-ON teacher is exactly the old goal-free RL expert. Since the base is frozen, Goal-OFF remains the old expert throughout training; privilege is learned as a residual rather than replacing the policy.

Train four jobs:

```bash
bash scripts/training/privileged_opd_v2/run_teacher_stage3_progress.sh
bash scripts/training/privileged_opd_v2/run_teacher_stage3_rule.sh
bash scripts/training/privileged_opd_v2/run_teacher_stage3_safety.sh
bash scripts/training/privileged_opd_v2/run_teacher_stage3_general.sh
```

Fill the resulting checkpoints as `PRIV_*_CKPT`.

---

# 2. OPD distillation

## Student selection

Recommended main comparison:

```text
Student init = the same Stage-1 post-IL common checkpoint that was used to initialize Stage-2 RL experts.
```

This is deliberate: it preserves the common family/representation while leaving enough headroom, and it matches the setup in which old goal-free IL+RL teacher OPD already worked. Do not use a pre-IL random/initial DiT for the main run; its on-policy rollout gap is unnecessarily large. You also do not need to initialize from the already-89-PDMS student.

## Student architecture

Variant A deployment student stays **fully goal-free**:

```text
cached VLM hidden states + history + ego -> original goal-free DiT -> trajectory
```

No predicted goal is fed into the planner at train or test time.

## Same-student-rollout routed OPD

For each batch sample:

1. Route token to exactly one scene teacher.
2. Student samples its detached on-policy DDIM chain `z_t`.
3. Student and routed Teacher evaluate the **same `z_t`**.
4. Student sees `(O, z_t)`; Teacher sees `(O, z_t, GT goal)`.
5. Reconstruct both reverse means with the **same student DDIM sigma**.
6. Optimize shared-variance reverse-KL surrogate.

```text
                         same student z_t
                         /             \
              goal-free Student      routed Teacher
                    O,z_t             O,z_t + GT goal
                      |                    |
                     mu_S                 mu_T
                        \                  /
                     shared-sigma reverse KL
```

Main objective:

```text
L = lambda_KD * L_RKL + lambda_task * L_diffusion_IL
```

Defaults:

```text
KD_WEIGHT     = 1.0
TASK_WEIGHT   = 0.10
SMOOTH_WEIGHT = 0.0
LR            = 5e-5
MAX_EPOCHS    = 30
```

`L_task` is ordinary GT trajectory diffusion IL for capability preservation; OPD does **not** run another RL stage.

The reverse-KL precision is clipped and normalized so the mean timestep weight is ~1; it is not divided by an arbitrary constant as in the failed GoalBridge run.

### Gradient diagnostics

Every `GRAD_DIAG_INTERVAL` steps (default 100) the Lightning wrapper records:

```text
train/grad_opd_probe
train/grad_task_probe
train/grad_opd_task_ratio
train/gradient_norm_total
```

Use the ratio as a scale diagnostic. A reasonable initial target is roughly `0.3–1.0`; if OPD is orders of magnitude smaller, adjust `KD_WEIGHT` rather than comparing raw scalar loss values.

Run A:

```bash
bash scripts/training/privileged_opd_v2/run_opd_A.sh
```

OPD data defaults to full NAV cache plus available SimScale caches; every sample is still routed to one matching teacher.

---

# 3. Variants

## A — Main: Routed Same-Rollout RKL

- Teacher: final-point + `gated_cross`
- Student: goal-free
- Same student rollout
- Routed one-teacher reverse-KL + small task IL

```bash
bash scripts/training/privileged_opd_v2/run_opd_A.sh
```

## B — Auxiliary Goal Internalization

A + training-only auxiliary head:

```text
observable student features -> predicted final (x,y)
```

`predicted goal` is **never fed into the planner**. It only forces student representation to encode recoverable privileged information.

```bash
bash scripts/training/privileged_opd_v2/run_opd_B.sh
```

Default `GOAL_AUX_WEIGHT=0.2`.

## C — Goal Preference Distillation

A + a 2048-goal vocabulary. The student predicts a distribution over candidate goals. The target is **not** a single GT point.

For each sample, the routed privileged teacher:

1. produces its GT-goal privileged trajectory;
2. takes the nearest `M` vocabulary candidates (`M=8` default);
3. is re-evaluated under each candidate goal on the same final student rollout state;
4. ranks candidates by how close the candidate-conditioned teacher trajectory is to the GT-goal privileged teacher trajectory, with a small geometric endpoint term;
5. produces a soft preference distribution `q_T(G)`.

Student minimizes `KL(q_T || p_S)` over the full vocabulary. C currently uses **final-point teachers**; keep D multi-point as a separate ablation first.

Build vocabulary:

```bash
python scripts/data/privileged_opd_v2/build_goal_vocab.py \
  --cache-root "$NAV_CACHE" \
  --manifest "$NAV_MANIFEST" \
  --k 2048 \
  --max-samples 100000 \
  --out /path/to/goal_vocab_2048.npz
```

Then:

```bash
export GOAL_VOCAB=/path/to/goal_vocab_2048.npz
bash scripts/training/privileged_opd_v2/run_opd_C.sh
```

Useful knobs:

```text
GOAL_PREF_WEIGHT=0.2
GOAL_PREF_CANDIDATES=8
GOAL_PREF_TAU_M=2.0
GOAL_PREF_GEO_WEIGHT=0.25
```

## D — Goal representation / injection ablation

Train four Stage-3 teachers for the selected D case, then run OPD with the **same architecture setting**.

| Case | Goal representation | Injection |
|---|---|---|
| D1 | final | AdaLN |
| D2 | final | Cross-attention K/V token |
| D3 | final | Zero-init gated goal adapter |
| D4 | near+mid+far | AdaLN |
| D5 | near+mid+far | Cross-attention K/V tokens |
| D6 | near+mid+far | Zero-init gated goal adapter |

Expected safest/best: **D6**.

`multi3` defaults to GT trajectory indices `[1,4,7]` from the 8-point / 0.5s horizon, approximately 1.0s, 2.5s, 4.0s. Each point gets a separate token with shared GoalEncoder + type embedding. Cached VLM states remain unchanged.

Example teacher:

```bash
D_CASE=D6 \
BUCKET_NAME=progress_curbside_stopgo \
BASE_RL_CKPT="$RL_PROGRESS_CKPT" \
bash scripts/training/privileged_opd_v2/run_teacher_stage3_D_case.sh
```

Repeat for four buckets. After filling four matching teacher checkpoint paths:

```bash
D_CASE=D6 bash scripts/training/privileged_opd_v2/run_opd_D_case.sh
```

## F — Residual-Privilege OPD

A control designed to isolate privilege itself:

```text
target_mu = frozen_student_family_ref_mu + [teacher_goal_ON_mu - teacher_goal_OFF_mu]
```

This removes the teacher's base-policy difference and asks whether the **goal-induced residual alone** can be internalized.

```bash
bash scripts/training/privileged_opd_v2/run_opd_F.sh
```

Default `RESIDUAL_REF_CKPT=$STUDENT_CKPT`.

## G — Combined advanced variant

F + C + small B:

```text
Residual privilege OPD + Goal Preference Distillation + Auxiliary Goal Internalization
```

```bash
export GOAL_VOCAB=/path/to/goal_vocab_2048.npz
bash scripts/training/privileged_opd_v2/run_opd_G.sh
```

Use only after A/B/C/F establish which component is helpful; otherwise attribution becomes unclear.

---

# 4. Paths to fill

Copy:

```bash
cp scripts/training/privileged_opd_v2/paths.example.sh \
   scripts/training/privileged_opd_v2/paths.local.sh
```

Fill at least:

```text
CONDA_BIN
VLM_PATH
STAGE1_IL_CKPT
STUDENT_CKPT

RL_PROGRESS_CKPT
RL_RULE_CKPT
RL_SAFETY_CKPT
RL_GENERAL_CKPT

PRIV_PROGRESS_CKPT
PRIV_RULE_CKPT
PRIV_SAFETY_CKPT
PRIV_GENERAL_CKPT

NAV_CACHE
NAV_MANIFEST
NAV_BUCKET_ROOT
TOKEN_TO_BUCKET_NAV

SIM_CACHE_R0 / R1
SIM_MANIFEST_R0 / R1
SIM_BUCKET_R0_ROOT / R1_ROOT

NUPLAN_MAPS_ROOT
OPENSCENE_DATA_ROOT
NAVSIM_EXP_ROOT

GOAL_VOCAB   # C/G only
```

---

# 5. Important checks before long runs

1. **Stage-3 checkpoint identity:** `BASE_RL_CKPT` must be a goal-free old IL+RL scene expert, not the old 94-point privileged teacher and not the failed v4 teacher.
2. **A teacher architecture match:** default A expects `final + gated_cross`; OPD loader intentionally errors if checkpoint goal parameters do not match.
3. **D architecture match:** the four teacher checkpoints and OPD `GOAL_POINT_MODE/GOAL_INJECTION` must be identical.
4. **No noisy/masked goal in Stage 3.** The new branch uses correct GT privilege only.
5. **No general mixing in Stage 3.** Use the corresponding NAV bucket and corresponding SimScale bucket only.
6. **Student main init:** use post-IL common fullmix checkpoint for the cleanest comparison; do not use pre-IL random DiT for the first experiment.
7. **Cached VLM states remain valid.** Goal encoding happens after the cache boundary inside the planner.
8. **Routing is mandatory for A/B/C/D/F/G main jobs:** exactly one scene teacher per sample.
9. **C cost:** candidate ranking adds teacher forwards on the final rollout step. Start with `GOAL_PREF_CANDIDATES=8`; increase only after profiling memory/time.
10. **C + multi3:** not enabled in this first implementation; validate C and D separately before composing them.
11. **Gradient probes:** if DDP/autograd diagnostics cause an environment-specific reducer issue, set `GRAD_DIAG_INTERVAL=0` for the run and first debug the ratio on 1 GPU. The actual training loss does not depend on the probe.
12. **First-batch check:** confirm `opd_loss`, `task_loss`, `x0_gap_m`, `sigma_mean`, total gradient norm are finite before launching long jobs.
13. **Teacher success criterion:** with adapter-only training, Goal-OFF should be numerically the old RL expert because base parameters are frozen. Judge Stage 3 by Goal-ON gain, not by forcing the teacher to work without goal.
14. **Do not use all-four-teacher averaging as the default.** If desired, add it later as a negative/ablation control against hard routing.

