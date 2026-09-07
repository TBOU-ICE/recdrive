"""Pre-flight check for goal-teacher scene-router OPD distillation.

Run on CPU before queueing GPUs:

    TORCHDYNAMO_DISABLE=1 CUDA_VISIBLE_DEVICES= \
        python scripts/diagnosis/check_goal_opd_trainer.py

Verifies that:
  * the goal trainer with a no-op goal branch reproduces the base trainer's
    loss bit-for-bit (same RNG) -- the loss path itself is unchanged;
  * with a live goal branch, the teacher targets actually move (goal effect
    metric > 0) and the loss stays finite;
  * every diagnostic key is present, finite, and the per-bucket key set is
    complete (DDP rank symmetry);
  * the checkpoint guard rejects goal_mode mismatches and dead goal branches
    but accepts a matching live checkpoint.
"""

from types import SimpleNamespace

import torch
from torch import nn
from transformers.feature_extraction_utils import BatchFeature

from navsim.agents.recogdrive.recogdrive_agent import make_recogdrive_config
from navsim.agents.recogdrive.recogdrive_diffusion_planner import ReCogDriveDiffusionPlanner
from navsim.agents.recogdrive.recogdrive_goal_planner import GoalCondDiffusionPlanner
from navsim.agents.recogdrive.recogdrive_dit_scene_router_distill_trainer import (
    ReCogDriveDiTSceneRouterDistillTrainer,
)
from navsim.agents.recogdrive.recogdrive_dit_scene_router_goal_distill_trainer import (
    ReCogDriveDiTSceneRouterGoalDistillTrainer,
)
from navsim.agents.recogdrive.recogdrive_scene_router_goal_agent import (
    ReCogDriveSceneRouterGoalAgent,
)

BUCKETS = [
    "progress_curbside_stopgo",
    "rule_intersection",
    "safety_dynamics_interaction",
    "general_or_no_tag",
]
HORIZON, ACTION_DIM, BATCH = 8, 3, 4
VL_RAW = 1536


def _ok(msg):
    print(f"  PASS  {msg}")


def _cfg():
    cfg = make_recogdrive_config(
        "small", action_dim=ACTION_DIM, action_horizon=HORIZON,
        input_embedding_dim=384, sampling_method="ddim", grpo=False,
    )
    cfg.vlm_size = "small"
    return cfg


def _open_gates(planner, seed=7):
    g = torch.Generator().manual_seed(seed)
    mods = [b.adaLN_modulation for b in planner.model.transformer_blocks]
    mods.append(planner.model.final_layer.modulation_proj)
    for m in mods:
        for layer in m:
            if isinstance(layer, nn.Linear):
                layer.weight.data.normal_(0.0, 0.05, generator=g)
                layer.bias.data.normal_(0.0, 0.05, generator=g)
    return planner


def _build_planners(goal_live: bool):
    torch.manual_seed(11)
    student = _open_gates(ReCogDriveDiffusionPlanner(_cfg()).float().eval())
    teachers = {}
    for i, b in enumerate(BUCKETS):
        torch.manual_seed(100 + i)
        t = _open_gates(GoalCondDiffusionPlanner(_cfg(), goal_mode="adaln").float().eval())
        if goal_live:
            nn.init.normal_(t.goal_encoder.net[-1].weight, std=0.05)
            nn.init.normal_(t.goal_encoder.net[-1].bias, std=0.05)
        for p in t.parameters():
            p.requires_grad = False
        teachers[b] = t
    return student, teachers


def _batch():
    torch.manual_seed(3)
    return {
        "vl": torch.randn(BATCH, 11, VL_RAW),
        "input": BatchFeature(data={
            "his_traj": torch.randn(BATCH, 12),
            "status_feature": torch.randn(BATCH, 8),
            "action": torch.stack([
                torch.linspace(0, 30, HORIZON),
                torch.linspace(0, 2, HORIZON),
                torch.linspace(0, 0.2, HORIZON),
            ], dim=-1).unsqueeze(0).repeat(BATCH, 1, 1)
            + 0.5 * torch.randn(BATCH, HORIZON, ACTION_DIM),
        }),
        "buckets": [BUCKETS[i % len(BUCKETS)] for i in range(BATCH)],
    }


def main():
    kw = dict(bucket_names=BUCKETS, fallback_bucket="general_or_no_tag",
              min_sigma=0.04, smooth_weight=0.02, match_target="x0", exopd_lambda=1.0)
    batch = _batch()

    print("=== no-op goal branch: goal trainer == base trainer bit-for-bit ===")
    student, teachers = _build_planners(goal_live=False)  # adaln zero-init => no-op
    base_tr = ReCogDriveDiTSceneRouterDistillTrainer(**kw)
    goal_tr = ReCogDriveDiTSceneRouterGoalDistillTrainer(**kw, collect_viz=True)
    torch.manual_seed(42)
    out_base = base_tr.compute_loss(student, teachers, None, batch["vl"], batch["input"], batch["buckets"])
    torch.manual_seed(42)
    out_goal = goal_tr.compute_loss(student, teachers, None, batch["vl"], batch["input"], batch["buckets"])
    delta = (out_base["loss"] - out_goal["loss"]).abs().item()
    assert delta == 0.0, f"loss path changed: delta={delta}"
    _ok(f"loss identical with dormant goal branch (delta={delta:.1e})")
    assert out_goal["teacher_goal_effect_m"].item() == 0.0
    _ok("teacher_goal_effect_m == 0 flags the dormant branch")

    print("\n=== live goal branch: goal moves the target, metrics are sane ===")
    student, teachers = _build_planners(goal_live=True)
    goal_tr = ReCogDriveDiTSceneRouterGoalDistillTrainer(**kw, collect_viz=True)
    torch.manual_seed(42)
    out = goal_tr.compute_loss(student, teachers, None, batch["vl"], batch["input"], batch["buckets"])
    assert torch.isfinite(out["loss"]), "non-finite loss"
    effect = out["teacher_goal_effect_m"].item()
    assert effect > 1e-5, f"goal had no effect on the teacher target: {effect}"
    _ok(f"loss={out['loss'].item():.4f}, teacher_goal_effect_m={effect:.4f} m")

    expected = ["x0_gap_m", "x0_gap_final_m", "gauss_kl_mean", "entropy_mean",
                "student_fde_gt_m", "student_ade_gt_m", "teacher_goal_effect_m"]
    expected += [f"entropy_step_{i}" for i in range(int(out["denoising_steps"].item()))]
    for b in BUCKETS:
        expected += [f"kl_{b}_mean", f"n_samples_{b}", f"fde_gt_{b}", f"goal_effect_{b}"]
    missing = [k for k in expected if k not in out]
    bad = [k for k in expected if k in out and not torch.isfinite(out[k]).all()]
    assert not missing, f"missing metric keys: {missing}"
    assert not bad, f"non-finite metrics: {bad}"
    _ok(f"all {len(expected)} diagnostic keys present and finite (rank-symmetric bucket set)")

    viz = goal_tr.last_viz
    assert viz is not None and viz["student_traj"].shape == (BATCH, HORIZON, ACTION_DIM)
    assert len(viz["buckets"]) == BATCH
    _ok("last_viz payload populated for the lightning BEV hook")

    print("\n=== per-sample goal pairing: swapping goals changes the target ===")
    tname = batch["buckets"][0]
    teacher = teachers[tname]
    sel = torch.tensor([i for i, b in enumerate(batch["buckets"]) if b == tname])
    enc = goal_tr._encode(teacher, batch["vl"], batch["input"].his_traj,
                          batch["input"].status_feature, torch.float32)
    z = torch.randn(len(sel), HORIZON, ACTION_DIM)
    t_b = teacher.make_timesteps(len(sel), int(teacher.ddim_t[0].item()), z.device)
    i_b = teacher.make_timesteps(len(sel), 0, z.device)
    goal = batch["input"].action[:, -1, :].float()
    with torch.no_grad():
        with teacher.goal_context(goal[sel]):
            _, _, a = teacher.p_mean_variance(z, t_b, i_b, enc[0][sel], enc[1][sel], enc[2][sel], deterministic=True)
        with teacher.goal_context(goal[sel] + torch.tensor([20.0, -8.0, 0.0])):
            _, _, b = teacher.p_mean_variance(z, t_b, i_b, enc[0][sel], enc[1][sel], enc[2][sel], deterministic=True)
    assert (a - b).abs().max() > 1e-5, "teacher ignores which goal is bound"
    _ok(f"different goal moves teacher x0 by max {(a - b).abs().max():.2e}")

    print("\n=== checkpoint guard: mismatch / dead branch rejected, live accepted ===")
    ns = SimpleNamespace(teacher_goal_mode="adaln")
    planner = teachers[BUCKETS[0]]
    good = {k: v.clone() for k, v in planner.state_dict().items()}
    ReCogDriveSceneRouterGoalAgent._verify_goal_weights(ns, planner, good, "t[good]", "good.ckpt")
    _ok("matching live goal checkpoint accepted")
    for label, state in (
        ("goal-free ckpt", {k: v for k, v in good.items() if "goal_" not in k}),
        ("channel ckpt", {**{k: v for k, v in good.items() if "goal_" not in k},
                          **{k: v for k, v in good.items() if "goal_encoder" in k},
                          "goal_channel_proj.weight": torch.zeros(384, 384)}),
        ("dead branch", {**good, **{k: torch.zeros_like(v) for k, v in good.items() if "goal_" in k}}),
    ):
        try:
            ReCogDriveSceneRouterGoalAgent._verify_goal_weights(ns, planner, state, f"t[{label}]", "bad.ckpt")
        except RuntimeError:
            _ok(f"{label} rejected")
        else:
            raise AssertionError(f"{label} was NOT rejected")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
