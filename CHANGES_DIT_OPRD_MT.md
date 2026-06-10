# Pure OPRD Multi-Teacher DiT Distillation Additions

This patch only adds new files and does not delete or edit the original OPD / temporal OPD implementation.

## New files

- `navsim/agents/recogdrive/recogdrive_dit_oprd_multi_teacher_trainer.py`
- `navsim/agents/recogdrive/recogdrive_oprd_multi_teacher_dit_agent.py`
- `navsim/planning/script/config/common/agent/recogdrive_agent_oprd_multi_teacher_dit.yaml`
- `scripts/training/run_recogdrive_train_oprd_multi_teacher_dit_8gpu.sh`

## Method

The new trainer is a pure hidden-state OPRD variant:

- Keeps student on-policy DDIM chain sampling.
- Removes trajectory-level KL / `mu` distillation.
- Removes temporal consistency loss.
- Distills only DiT representations returned by `LightningDiT.forward(return_hidden_states=True)`.
- Middle denoising steps + middle DiT layers learn from the IL teacher only.
- Late denoising steps + last DiT layers learn from an IL-heavy/RL-light mixture.
- The optional final representation before `action_decoder` is also distilled in late steps.

Default loss:

```text
L = L_mid_IL + L_late_last + L_late_final

L_mid_IL    = 1.00 * MSE(norm(h_student_mid),  norm(h_IL_mid))
L_late_last = 0.85 * MSE(norm(h_student_last), norm(h_IL_last))
            + 0.15 * MSE(norm(h_student_last), norm(h_RL_last))
L_late_final follows the same 0.85 / 0.15 teacher weighting.
```

Teacher DiTs and the VLM remain frozen. Only the student DiT receives gradients.
