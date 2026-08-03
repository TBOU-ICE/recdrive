"""Pre-flight check for the goal-conditioned planner.

Run this on the cluster (where the real deps exist) before launching a training
or scoring job, so wiring mistakes surface in seconds instead of after the
scheduler finally grants you GPUs.

    python scripts/diagnosis/check_goal_planner.py
    python scripts/diagnosis/check_goal_planner.py --dit-type large --vlm-size large

It builds a planner in each goal mode and verifies that:
  * goal_mode="none" is byte-identical to the untouched ReCogDriveDiffusionPlanner
  * every mode runs forward() and get_action() and produces finite output
  * the zero-initialised goal branches are an exact no-op, so warm-starting a
    teacher from a goal-free checkpoint changes nothing before the first step
  * a bound goal actually moves the prediction, and swapping goals between
    samples moves it differently (i.e. the goal/sample pairing is not shuffled)
  * the goal is replicated correctly when the batch is expanded, which is what
    GRPO's repeat_interleave(G, 0) and get_logprobs' per-step repeat rely on
"""

from __future__ import annotations

import argparse

import torch
from torch import nn
from transformers.feature_extraction_utils import BatchFeature

from navsim.agents.recogdrive.recogdrive_agent import make_recogdrive_config
from navsim.agents.recogdrive.recogdrive_diffusion_planner import ReCogDriveDiffusionPlanner
from navsim.agents.recogdrive.recogdrive_goal_planner import GoalCondDiffusionPlanner

HORIZON = 8
ACTION_DIM = 3


def _ok(message: str) -> None:
    print(f"  PASS  {message}")


def _simulate_trained_dit(planner: GoalCondDiffusionPlanner, seed: int = 7) -> GoalCondDiffusionPlanner:
    """Open the adaLN-Zero gates so the model behaves like a trained checkpoint.

    LightningDiT._initialize_weights zeroes every adaLN_modulation, which makes
    all residual gates zero on a freshly built model. In that state each block is
    an identity on the residual stream, so AdaLN and cross-attention conditioning
    of any kind is provably inert and the "does the goal matter" checks below
    would trivially fail. A warm-started teacher always begins from a trained
    checkpoint, so that is the regime we test.
    """
    generator = torch.Generator().manual_seed(seed)
    modules = [b.adaLN_modulation for b in planner.model.transformer_blocks]
    modules.append(planner.model.final_layer.modulation_proj)
    for module in modules:
        for layer in module:
            if isinstance(layer, nn.Linear):
                layer.weight.data.normal_(0.0, 0.05, generator=generator)
                layer.bias.data.normal_(0.0, 0.05, generator=generator)
    return planner


def _build(goal_mode: str, args, **kwargs) -> GoalCondDiffusionPlanner:
    embed_dim = 1536 if args.dit_type == "large" else 384
    cfg = make_recogdrive_config(
        args.dit_type,
        action_dim=ACTION_DIM,
        action_horizon=HORIZON,
        input_embedding_dim=embed_dim,
        sampling_method=args.sampling_method,
        grpo=False,
    )
    cfg.vlm_size = args.vlm_size
    return GoalCondDiffusionPlanner(cfg, goal_mode=goal_mode, **kwargs).float().eval()


def _batch(args):
    vl_raw = 3584 if args.vlm_size == "large" else 1536
    return {
        "vl": torch.randn(args.batch, 11, vl_raw),
        "input": BatchFeature(data={
            "his_traj": torch.randn(args.batch, 12),
            "status_feature": torch.randn(args.batch, 8),
            "action": torch.stack([
                torch.linspace(0, 30, HORIZON),
                torch.linspace(0, 2, HORIZON),
                torch.linspace(0, 0.2, HORIZON),
            ], dim=-1).unsqueeze(0).repeat(args.batch, 1, 1),
        }),
        "goal": torch.stack([
            torch.tensor([28.0, 1.9, 0.18]), torch.tensor([12.0, -3.0, -0.1]),
        ])[: args.batch].repeat(max(1, args.batch // 2), 1)[: args.batch],
    }


def _encode(planner, batch):
    return (
        planner.feature_encoder(batch["vl"]),
        planner.his_traj_encoder(batch["input"].his_traj.unsqueeze(1)).repeat(1, HORIZON, 1),
        planner.ego_status_encoder(batch["input"].status_feature),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dit-type", default="small", choices=["small", "large"])
    parser.add_argument("--vlm-size", default="small", choices=["small", "large"])
    parser.add_argument("--sampling-method", default="ddim", choices=["ddim", "ddpm"])
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=4, help="GRPO G, for the pairing check")
    args = parser.parse_args()

    torch.manual_seed(0)
    batch = _batch(args)

    print("=== goal_mode='none' is identical to the base planner ===")
    embed_dim = 1536 if args.dit_type == "large" else 384
    torch.manual_seed(1)
    base_cfg = make_recogdrive_config(
        args.dit_type, action_dim=ACTION_DIM, action_horizon=HORIZON,
        input_embedding_dim=embed_dim, sampling_method=args.sampling_method, grpo=False,
    )
    base_cfg.vlm_size = args.vlm_size
    base = ReCogDriveDiffusionPlanner(base_cfg).float().eval()
    torch.manual_seed(1)
    plain = _build("none", args)
    plain.load_state_dict(base.state_dict(), strict=True)
    _ok("state_dict keys and shapes match the base planner exactly")

    call = (torch.randn(args.batch, HORIZON, ACTION_DIM),
            torch.full((args.batch,), 40, dtype=torch.long),
            torch.full((args.batch,), 2, dtype=torch.long)) + _encode(base, batch)
    with torch.no_grad():
        delta = (base.p_mean_variance(*call, deterministic=True)[0]
                 - plain.p_mean_variance(*call, deterministic=True)[0]).abs().max()
    assert delta == 0, delta
    _ok(f"p_mean_variance output is bit-identical (max diff {delta:.2e})")

    print("\n=== all goal modes run ===")
    for mode in ("inpaint", "adaln", "channel", "cross"):
        planner = _build(mode, args)
        with planner.goal_context(batch["goal"]):
            loss = planner.forward(batch["vl"], batch["input"])["loss"]
        assert torch.isfinite(loss), f"{mode}: non-finite loss"
        with torch.no_grad(), planner.goal_context(batch["goal"]):
            traj = planner.get_action(batch["vl"], batch["input"], deterministic=True)["pred_traj"]
        assert traj.shape == (args.batch, HORIZON, ACTION_DIM), traj.shape
        assert torch.isfinite(traj).all(), f"{mode}: non-finite trajectory"
        _ok(f"{mode:8s} loss={loss.item():.4f}, trajectory {tuple(traj.shape)}")

    print("\n=== goal branches receive gradient (no stacked-zero deadlock) ===")
    # Catches the failure mode where a zero-init encoder feeds a zero-init
    # projection: with y = P.e, dL/dP = g.e^T = 0 (e = 0) and dL/de = P^T.g = 0
    # (P = 0), so the branch trains to nothing.  The 2026.08.02 channel/cross
    # teachers hit exactly this; their goal weights were still zero after ~200
    # epochs.  Unlike the "bound goal changes the prediction" check below, this
    # one does NOT re-initialise any goal weights, so it tests the real
    # from-scratch training regime.
    grad_targets = {
        "adaln": "goal_encoder.net.2.weight",
        "channel": "goal_channel_proj.weight",
        "cross": "goal_cross_proj.weight",
    }
    for mode, param_name in grad_targets.items():
        planner = _simulate_trained_dit(_build(mode, args)).train()
        with planner.goal_context(batch["goal"]):
            loss = planner.forward(batch["vl"], batch["input"])["loss"]
        loss.backward()
        param = dict(planner.named_parameters())[param_name]
        grad_max = 0.0 if param.grad is None else param.grad.abs().max().item()
        assert grad_max > 0, (
            f"{mode}: {param_name} received zero gradient -- goal branch is dead "
            "(two stacked zero-init layers?)"
        )
        _ok(f"{mode:8s} max |d(loss)/d({param_name})| = {grad_max:.2e}")

    print("\n=== zero-init goal branches are a no-op on a trained checkpoint ===")
    for mode in ("adaln", "channel", "cross"):
        planner = _simulate_trained_dit(_build(mode, args))
        call = (torch.randn(args.batch, HORIZON, ACTION_DIM),
                torch.full((args.batch,), 40, dtype=torch.long),
                torch.full((args.batch,), 2, dtype=torch.long)) + _encode(planner, batch)
        with torch.no_grad():
            off = planner.p_mean_variance(*call, deterministic=True)[0]
            with planner.goal_context(batch["goal"]):
                on = planner.p_mean_variance(*call, deterministic=True)[0]
        diff = (off - on).abs().max().item()
        # "cross" appends a zero KV token, but Attention.to_v carries a bias, so
        # v(0) != 0 and the extra token perturbs the softmax by a small amount.
        limit = 5e-2 if mode == "cross" else 0.0
        assert diff <= limit, f"{mode}: warm start is not a no-op, diff={diff}"
        _ok(f"{mode:8s} max diff {diff:.2e} (limit {limit:.0e})")

    print("\n=== a bound goal changes the prediction ===")
    for mode in ("adaln", "channel", "cross"):
        planner = _simulate_trained_dit(_build(mode, args))
        nn.init.normal_(planner.goal_encoder.net[-1].weight, std=0.05)
        for name in ("goal_channel_proj", "goal_cross_proj"):
            if hasattr(planner, name):
                nn.init.normal_(getattr(planner, name).weight, std=0.05)
        call = (torch.randn(args.batch, HORIZON, ACTION_DIM),
                torch.full((args.batch,), 40, dtype=torch.long),
                torch.full((args.batch,), 2, dtype=torch.long)) + _encode(planner, batch)
        with torch.no_grad():
            off = planner.p_mean_variance(*call, deterministic=True)[0]
            with planner.goal_context(batch["goal"]):
                on = planner.p_mean_variance(*call, deterministic=True)[0]
            with planner.goal_context(batch["goal"].flip(0)):
                swapped = planner.p_mean_variance(*call, deterministic=True)[0]
        assert (off - on).abs().max() > 1e-5, f"{mode}: goal had no effect"
        assert (on - swapped).abs().max() > 1e-5, f"{mode}: goals are interchangeable"
        _ok(f"{mode:8s} goal moves mu by {(off-on).abs().max():.2e}, "
            f"swapping moves it by {(on-swapped).abs().max():.2e}")

    print("\n=== inpaint pins the endpoint ===")
    planner = _build("inpaint", args, goal_inpaint_weight=1.0)
    with torch.no_grad(), planner.goal_context(batch["goal"]):
        traj = planner.get_action(batch["vl"], batch["input"], deterministic=True)["pred_traj"]
    error = (traj[:, -1, :2] - batch["goal"][:, :2]).abs().max().item()
    assert error < 1e-2, f"endpoint is {error} m from the goal"
    _ok(f"sampled endpoint is {error:.4f} m from the goal after {planner.ddim_steps} steps")

    print("\n=== goal replication under batch expansion (GRPO pairing) ===")
    group = args.group_size
    planner = _simulate_trained_dit(_build("adaln", args))
    nn.init.normal_(planner.goal_encoder.net[-1].weight, std=0.05)
    encoded = _encode(planner, batch)
    repeated = tuple(e.repeat_interleave(group, 0) for e in encoded)
    x = torch.randn(args.batch * group, HORIZON, ACTION_DIM)
    with torch.no_grad(), planner.goal_context(batch["goal"]):
        grouped = planner.p_mean_variance(
            x, torch.full((args.batch * group,), 40, dtype=torch.long),
            torch.full((args.batch * group,), 2, dtype=torch.long), *repeated, deterministic=True,
        )[0]
    with torch.no_grad(), planner.goal_context(batch["goal"][:1]):
        single = planner.p_mean_variance(
            x[:1], torch.full((1,), 40, dtype=torch.long), torch.full((1,), 2, dtype=torch.long),
            *(e[:1] for e in repeated), deterministic=True,
        )[0]
    assert torch.allclose(grouped[0], single[0], atol=1e-6), "goal/sample pairing is shifted"
    _ok(f"goal (batch {args.batch}) expands to {args.batch * group} with correct pairing")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
