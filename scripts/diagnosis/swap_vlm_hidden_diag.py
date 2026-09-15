#!/usr/bin/env python3
"""Causal VLM vs DiT check: swap / zero last_hidden_state, keep fail ego cond, score PDM.

For each fail token, with a FIXED DiT checkpoint:
  (a) self   : own VLM hidden
  (b) swap   : good-token VLM hidden (paired by index)
  (c) zero   : zeros_like(self hidden)

If (b) rescues score while (a) stays bad -> VLM-limited.
If (a) and (b) both bad -> more DiT / scene limited.
"""
from __future__ import annotations

import argparse
import csv
import json
import lzma
import os
import pickle
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers.feature_extraction_utils import BatchFeature

# repo root
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hydra.utils import instantiate
from omegaconf import OmegaConf
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.recogdrive.recogdrive_agent import ReCogDriveAgent
from navsim.agents.recogdrive.recogdrive_features import (
    decode_navigation_command,
    format_number,
)
from navsim.agents.recogdrive.utils.internvl_preprocess import load_image
from navsim.common.dataclasses import Trajectory
from navsim.common.dataloader import MetricCacheLoader, SceneFilter, SceneLoader
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator


def read_tokens(path: Path, limit: int = 0) -> List[str]:
    toks = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            toks.append(line)
    if limit > 0:
        toks = toks[:limit]
    return toks


def build_scene_loader(openscene: Path, scene_filter: SceneFilter, agent, load_image_path: bool = True) -> SceneLoader:
    return SceneLoader(
        sensor_blobs_path=openscene / "sensor_blobs/test",
        data_path=openscene / "navsim_logs/test",
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
        load_image_path=load_image_path,
    )


def load_scene_filter(repo: Path, name: str = "navtest_rule_intersection") -> SceneFilter:
    import hydra
    config_dir = str(repo / "navsim/planning/script/config/common/train_test_split/scene_filter")
    with hydra.initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = hydra.compose(config_name=name)
    return instantiate(cfg)


def extract_vlm_hidden(agent: ReCogDriveAgent, agent_input) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Run VLM backbone (and ego tensors) for one AgentInput. Returns (hidden[T,H], cond_cpu)."""
    features: Dict[str, torch.Tensor] = {}
    for builder in agent.get_feature_builders():
        features.update(builder.compute_features(agent_input))

    history_trajectory = features["history_trajectory"]
    high_command_one_hot = features["high_command_one_hot"]
    status_feature = features["status_feature"]
    if history_trajectory.ndim == 2:
        # already [4,3]
        pass
    image_path_tensor = features["image_path_tensor"]
    if image_path_tensor.ndim == 1:
        image_path_tensor = image_path_tensor.unsqueeze(0)
    image_paths = agent._decode_paths_from_tensor(image_path_tensor)
    pixel_values_list = [load_image(path) for path in image_paths]
    num_patches_list = [p.shape[0] for p in pixel_values_list]
    pixel_values_cat = torch.cat(pixel_values_list, dim=0).cuda()

    if high_command_one_hot.ndim == 1:
        high_command_one_hot_b = high_command_one_hot.unsqueeze(0)
    else:
        high_command_one_hot_b = high_command_one_hot
    if history_trajectory.ndim == 2:
        history_b = history_trajectory.unsqueeze(0)
    else:
        history_b = history_trajectory

    command_str_list = [decode_navigation_command(command) for command in high_command_one_hot_b]
    questions = []
    for i in range(high_command_one_hot_b.shape[0]):
        history_trajectory_sample = history_b[i]
        command_str_sample = command_str_list[i]
        history_str = " ".join(
            [
                f"   - t-{3-j}: ({format_number(history_trajectory_sample[j, 0].item())}, "
                f"{format_number(history_trajectory_sample[j, 1].item())}, "
                f"{format_number(history_trajectory_sample[j, 2].item())})"
                for j in range(history_trajectory_sample.shape[0])
            ]
        )
        prompt = (
            "<image>\nAs an autonomous driving system, predict the vehicle's trajectory based on:\n"
            "1. Visual perception from front camera view\n"
            f"2. Historical motion context (last 4 timesteps):{history_str}\n"
            f"3. Active navigation command: [{command_str_sample.upper()}]"
        )
        output_requirements = (
            "\nOutput requirements:\n- Predict 8 future trajectory points\n"
            "- Each point format: (x:float, y:float, heading:float)\n"
            "- Use [PT, ...] to encapsulate the trajectory\n"
            "- Maintain numerical precision to 2 decimal places"
        )
        questions.append(f"{prompt}{output_requirements}")

    with torch.no_grad():
        outputs = agent.backbone(pixel_values_cat, questions, num_patches_list=num_patches_list)
        last_hidden_state = outputs.hidden_states[-1]
    if last_hidden_state.ndim == 3:
        last_hidden_state = last_hidden_state[0]

    cond = {
        "history_trajectory": history_trajectory.cpu().float(),
        "status_feature": status_feature.cpu().float(),
        "high_command_one_hot": high_command_one_hot.cpu().float(),
    }
    return last_hidden_state.detach().float().cpu(), cond


def predict_traj(
    agent: ReCogDriveAgent,
    hidden: torch.Tensor,
    cond: Dict[str, torch.Tensor],
) -> Trajectory:
    model_dtype = next(agent.action_head.parameters()).dtype
    history = cond["history_trajectory"].cuda()
    status = cond["status_feature"].cuda()
    if history.ndim == 2:
        history_b = history.unsqueeze(0)
    else:
        history_b = history
    if status.ndim == 1:
        status_b = status.unsqueeze(0)
    else:
        status_b = status

    h = hidden.cuda().to(model_dtype)
    if h.ndim == 2:
        h = h.unsqueeze(0)

    history_reshaped = history_b.view(history_b.size(0), -1)
    action_inputs = BatchFeature(
        {
            "state": torch.cat([status_b, history_reshaped], dim=1).to(model_dtype),
            "his_traj": history_reshaped.to(model_dtype),
            "status_feature": status_b.to(model_dtype),
        }
    )
    with torch.no_grad():
        out = agent.action_head.get_action(h, action_inputs)
    poses = out["pred_traj"].float().cpu().squeeze(0)
    return Trajectory(poses.numpy())


def score_traj(metric_cache, traj, simulator, scorer) -> dict:
    result = pdm_score(
        metric_cache=metric_cache,
        model_trajectory=traj,
        future_sampling=simulator.proposal_sampling,
        simulator=simulator,
        scorer=scorer,
    )
    return asdict(result)


def mean_pool(h: torch.Tensor) -> torch.Tensor:
    if h.ndim == 1:
        return h
    return h.mean(dim=0)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    return float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fail-tokens", type=Path, required=True)
    ap.add_argument("--good-tokens", type=Path, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--vlm-path", type=str, required=True)
    ap.add_argument("--metric-cache-path", type=str, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--openscene-root", type=str, default=os.environ.get("OPENSCENE_DATA_ROOT", ""))
    ap.add_argument("--scene-filter", type=str, default="navtest_rule_intersection")
    ap.add_argument("--limit", type=int, default=20, help="Number of fail tokens to evaluate")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--tag", type=str, default="rl_45_45")
    args = ap.parse_args()

    assert args.openscene_root, "OPENSCENE_DATA_ROOT / --openscene-root required"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fail_tokens = read_tokens(args.fail_tokens, args.limit)
    good_tokens = read_tokens(args.good_tokens, 0)
    if len(good_tokens) < len(fail_tokens):
        raise SystemExit(f"Need >= {len(fail_tokens)} good tokens, got {len(good_tokens)}")

    device = torch.device(args.device)
    agent = ReCogDriveAgent(
        TrajectorySampling(time_horizon=4, interval_length=0.5),
        checkpoint_path=args.checkpoint,
        vlm_path=args.vlm_path,
        cam_type="single",
        vlm_type="internvl",
        dit_type="small",
        sampling_method="ddim",
        cache_mode=False,
        cache_hidden_state=False,
        vlm_size="small",
        grpo=False,
    ).to(device)
    agent.initialize()
    agent.eval()

    scene_filter = load_scene_filter(REPO, args.scene_filter)
    # Restrict loader to needed tokens for speed
    needed = sorted(set(fail_tokens) | set(good_tokens[: len(fail_tokens)]))
    scene_filter.tokens = needed
    openscene = Path(args.openscene_root)
    scene_loader = build_scene_loader(openscene, scene_filter, agent, load_image_path=True)

    proposal_sampling = TrajectorySampling(time_horizon=4, interval_length=0.1)
    simulator = PDMSimulator(proposal_sampling)
    # default scorer weights
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorerConfig

    scorer = PDMScorer(proposal_sampling, PDMScorerConfig())
    metric_loader = MetricCacheLoader(Path(args.metric_cache_path))

    available = set(scene_loader.tokens) & set(metric_loader.tokens)
    rows = []
    rescued = 0
    evaluated = 0

    for i, fail_tok in enumerate(fail_tokens):
        good_tok = good_tokens[i]
        if fail_tok not in available or good_tok not in available:
            print(f"[skip] missing data fail={fail_tok} good={good_tok}")
            continue
        print(f"[{i+1}/{len(fail_tokens)}] fail={fail_tok} good={good_tok}")
        try:
            fail_input = scene_loader.get_agent_input_from_token(fail_tok)
            good_input = scene_loader.get_agent_input_from_token(good_tok)
            fail_h, fail_cond = extract_vlm_hidden(agent, fail_input)
            good_h, _ = extract_vlm_hidden(agent, good_input)

            # align sequence length for swap/zero by truncate/pad to fail length
            Tf, H = fail_h.shape
            Tg = good_h.shape[0]
            if Tg >= Tf:
                good_h_use = good_h[:Tf]
            else:
                pad = torch.zeros(Tf - Tg, H, dtype=good_h.dtype)
                good_h_use = torch.cat([good_h, pad], dim=0)
            zero_h = torch.zeros_like(fail_h)

            with lzma.open(metric_loader.metric_cache_paths[fail_tok], "rb") as f:
                metric_cache = pickle.load(f)

            variants = {
                "self": fail_h,
                "swap_good": good_h_use,
                "zero": zero_h,
            }
            scores = {}
            trajs = {}
            for name, h in variants.items():
                traj = predict_traj(agent, h, fail_cond)
                sc = score_traj(metric_cache, traj, simulator, scorer)
                scores[name] = sc
                trajs[name] = traj.poses.tolist()

            cos = cosine(mean_pool(fail_h), mean_pool(good_h_use))
            delta = scores["swap_good"]["score"] - scores["self"]["score"]
            is_rescue = scores["self"]["score"] < 0.5 and scores["swap_good"]["score"] >= 0.7
            if is_rescue:
                rescued += 1
            evaluated += 1

            row = {
                "fail_token": fail_tok,
                "good_token": good_tok,
                "ckpt_tag": args.tag,
                "hidden_cosine": cos,
                "score_self": scores["self"]["score"],
                "score_swap": scores["swap_good"]["score"],
                "score_zero": scores["zero"]["score"],
                "delta_swap_minus_self": delta,
                "rescue": is_rescue,
                "nc_self": scores["self"]["no_at_fault_collisions"],
                "dac_self": scores["self"]["drivable_area_compliance"],
                "nc_swap": scores["swap_good"]["no_at_fault_collisions"],
                "dac_swap": scores["swap_good"]["drivable_area_compliance"],
            }
            rows.append(row)
            # dump per-case
            case_dir = args.output_dir / "cases" / fail_tok
            case_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "fail_hidden": fail_h,
                    "good_hidden": good_h_use,
                    "fail_cond": fail_cond,
                    "trajs": trajs,
                    "scores": scores,
                    "meta": row,
                },
                case_dir / "dump.pt",
            )
            print(
                f"  self={row['score_self']:.3f} swap={row['score_swap']:.3f} "
                f"zero={row['score_zero']:.3f} cos={cos:.3f} rescue={is_rescue}"
            )
        except Exception as e:
            print(f"[error] {fail_tok}: {e}")
            import traceback

            traceback.print_exc()

    out_csv = args.output_dir / f"swap_results_{args.tag}.csv"
    if rows:
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    summary = {
        "tag": args.tag,
        "checkpoint": args.checkpoint,
        "evaluated": evaluated,
        "rescued": rescued,
        "rescue_rate": (rescued / evaluated) if evaluated else None,
        "mean_delta_swap": (sum(r["delta_swap_minus_self"] for r in rows) / len(rows)) if rows else None,
        "mean_score_self": (sum(r["score_self"] for r in rows) / len(rows)) if rows else None,
        "mean_score_swap": (sum(r["score_swap"] for r in rows) / len(rows)) if rows else None,
        "mean_score_zero": (sum(r["score_zero"] for r in rows) / len(rows)) if rows else None,
    }
    (args.output_dir / f"swap_summary_{args.tag}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
