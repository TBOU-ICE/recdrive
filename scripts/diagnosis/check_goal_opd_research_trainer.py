"""Pre-flight check for the THREE goal-OPD research variants (kl / anchor / phf).

Run on CPU before queueing GPUs:

    TORCHDYNAMO_DISABLE=1 CUDA_VISIBLE_DEVICES= \
        python scripts/diagnosis/check_goal_opd_research_trainer.py

Verifies, without touching any existing file, that:
  * variant='kl' forces match_target='mu' and reproduces the baseline goal
    trainer (match_target='mu') bit-for-bit under the same RNG -- the KL variant
    only reparameterises the regression target, nothing else changes;
  * variant='anchor' adds a strictly-positive, finite anchor term and keeps the
    total loss finite and differentiable w.r.t. the student;
  * variant='phf' adds strictly-positive, finite hidden + flow terms, keeps the
    loss finite/differentiable, and removes every forward hook afterwards;
  * every variant emits a complete, finite, rank-symmetric diagnostic key set,
    including the three new scalars anchor_loss / phf_hidden_loss / phf_flow_loss.
"""

import torch
from torch import nn
from transformers.feature_extraction_utils import BatchFeature

from navsim.agents.recogdrive.recogdrive_agent import make_recogdrive_config
from navsim.agents.recogdrive.recogdrive_diffusion_planner import ReCogDriveDiffusionPlanner
from navsim.agents.recogdrive.recogdrive_goal_planner import GoalCondDiffusionPlanner
from navsim.agents.recogdrive.recogdrive_dit_scene_router_goal_distill_trainer import (
    ReCogDriveDiTSceneRouterGoalDistillTrainer,
)
from navsim.agents.recogdrive.recogdrive_goal_distill_trainer_research import (
    ReCogDriveGoalDistillResearchTrainer,
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


def _build_planners():
    torch.manual_seed(11)
    student = _open_gates(ReCogDriveDiffusionPlanner(_cfg()).float().train())
    teachers = {}
    for i, b in enumerate(BUCKETS):
        torch.manual_seed(100 + i)
        t = _open_gates(GoalCondDiffusionPlanner(_cfg(), goal_mode="adaln").float().eval())
        nn.init.normal_(t.goal_encoder.net[-1].weight, std=0.05)
        nn.init.normal_(t.goal_encoder.net[-1].bias, std=0.05)
        for p in t.parameters():
            p.requires_grad = False
        teachers[b] = t
    torch.manual_seed(5)
    anchor = _open_gates(ReCogDriveDiffusionPlanner(_cfg()).float().eval())
    for p in anchor.parameters():
        p.requires_grad = False
    return student, teachers, anchor


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


def _expected_keys(out):
    keys = ["x0_gap_m", "x0_gap_final_m", "gauss_kl_mean", "entropy_mean",
            "student_fde_gt_m", "student_ade_gt_m", "teacher_goal_effect_m",
            "anchor_loss", "phf_hidden_loss", "phf_flow_loss"]
    keys += [f"entropy_step_{i}" for i in range(int(out["denoising_steps"].item()))]
    for b in BUCKETS:
        keys += [f"kl_{b}_mean", f"n_samples_{b}", f"fde_gt_{b}", f"goal_effect_{b}"]
    return keys


def _check_keys(out, tag):
    expected = _expected_keys(out)
    missing = [k for k in expected if k not in out]
    bad = [k for k in expected if k in out and not torch.isfinite(out[k]).all()]
    assert not missing, f"[{tag}] missing metric keys: {missing}"
    assert not bad, f"[{tag}] non-finite metrics: {bad}"
    _ok(f"[{tag}] all {len(expected)} diagnostic keys present and finite")


def _kw():
    return dict(bucket_names=BUCKETS, fallback_bucket="general_or_no_tag",
                min_sigma=0.04, smooth_weight=0.02, exopd_lambda=1.0)


def main():
    batch = _batch()

    print("=== variant='kl' == baseline goal trainer with match_target='mu' ===")
    student, teachers, _ = _build_planners()
    base = ReCogDriveDiTSceneRouterGoalDistillTrainer(**_kw(), match_target="mu", collect_viz=False)
    res = ReCogDriveGoalDistillResearchTrainer(**_kw(), match_target="x0", collect_viz=False, variant="kl")
    assert res.match_target == "mu", "variant=kl must force match_target='mu'"
    _ok("variant=kl forced match_target -> 'mu'")
    torch.manual_seed(42)
    out_b = base.compute_loss(student, teachers, None, batch["vl"], batch["input"], batch["buckets"])
    torch.manual_seed(42)
    out_k = res.compute_loss(student, teachers, None, batch["vl"], batch["input"], batch["buckets"])
    delta = (out_b["loss"] - out_k["loss"]).abs().item()
    assert delta < 1e-6, f"kl variant diverged from mu-baseline: delta={delta}"
    _ok(f"kl loss matches mu-baseline (delta={delta:.1e})")
    assert out_k["anchor_loss"].item() == 0.0 and out_k["phf_hidden_loss"].item() == 0.0
    _ok("kl leaves anchor/phf terms at 0")
    _check_keys(out_k, "kl")

    print("\n=== variant='anchor': positive finite anchor term, grad flows ===")
    student, teachers, anchor = _build_planners()
    res = ReCogDriveGoalDistillResearchTrainer(
        **_kw(), match_target="x0", collect_viz=False, variant="anchor", anchor_weight=0.3
    )
    res.anchor_planner = anchor
    torch.manual_seed(42)
    out_a = res.compute_loss(student, teachers, None, batch["vl"], batch["input"], batch["buckets"])
    assert torch.isfinite(out_a["loss"]) and out_a["loss"].requires_grad
    assert out_a["anchor_loss"].item() > 0.0, "anchor term should be > 0 vs a different base"
    out_a["loss"].backward()
    grad_norm = sum(p.grad.abs().sum().item() for p in student.parameters() if p.grad is not None)
    assert grad_norm > 0.0, "no gradient reached the student under anchor"
    _ok(f"anchor_loss={out_a['anchor_loss'].item():.4f}, loss={out_a['loss'].item():.4f}, grad_norm>0")
    _check_keys(out_a, "anchor")

    print("\n=== variant='phf': positive finite hidden+flow terms, hooks removed ===")
    student, teachers, _ = _build_planners()
    res = ReCogDriveGoalDistillResearchTrainer(
        **_kw(), match_target="x0", collect_viz=False, variant="phf",
        phf_weight=0.1, phf_flow_weight=0.1,
    )
    n_hooks_before = len(student.model.transformer_blocks[-1]._forward_hooks)
    torch.manual_seed(42)
    out_p = res.compute_loss(student, teachers, None, batch["vl"], batch["input"], batch["buckets"])
    assert torch.isfinite(out_p["loss"]) and out_p["loss"].requires_grad
    assert out_p["phf_hidden_loss"].item() > 0.0, "phf hidden term should be > 0"
    assert out_p["phf_flow_loss"].item() > 0.0, "phf flow term should be > 0"
    out_p["loss"].backward()
    grad_norm = sum(p.grad.abs().sum().item() for p in student.parameters() if p.grad is not None)
    assert grad_norm > 0.0, "no gradient reached the student under phf"
    n_hooks_after = len(student.model.transformer_blocks[-1]._forward_hooks)
    assert n_hooks_after == n_hooks_before, f"phf leaked forward hooks: {n_hooks_before}->{n_hooks_after}"
    for t in teachers.values():
        assert len(t.model.transformer_blocks[-1]._forward_hooks) == 0, "teacher hook leaked"
    _ok(f"phf_hidden={out_p['phf_hidden_loss'].item():.4f}, phf_flow={out_p['phf_flow_loss'].item():.4f}, hooks removed")
    _check_keys(out_p, "phf")

    print("\nAll research-variant checks passed.")


if __name__ == "__main__":
    main()
