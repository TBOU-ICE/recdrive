#!/usr/bin/env python3
"""Plot BEV+front-camera for a token list with a ReCogDrive checkpoint."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import torch
from hydra.utils import instantiate
from tqdm import tqdm

from navsim.agents.recogdrive.recogdrive_agent import ReCogDriveAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.visualization.plots import plot_bev_and_camera_with_agent
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tokens-file", type=Path, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--vlm-path", type=str, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--tag", type=str, default="agent")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--scene-filter", type=str, default="navtest")
    p.add_argument("--device", type=str, default="cuda:0")
    return p.parse_args()


def load_tokens(path: Path) -> list[str]:
    tokens: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tokens.append(line)
    return tokens


def main() -> None:
    args = parse_args()
    openscene = Path(os.environ["OPENSCENE_DATA_ROOT"])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokens = load_tokens(args.tokens_file)
    if not tokens:
        raise SystemExit(f"No tokens in {args.tokens_file}")

    # Resolve hydra config relative to repo root
    repo = Path(__file__).resolve().parents[2]
    config_dir = str(repo / "navsim/planning/script/config/common/train_test_split/scene_filter")
    with hydra.initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = hydra.compose(config_name=args.scene_filter)
    scene_filter: SceneFilter = instantiate(cfg)

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

    sensor_config = agent.get_sensor_config()
    scene_loader = SceneLoader(
        openscene / f"navsim_logs/{args.split}",
        openscene / f"sensor_blobs/{args.split}",
        scene_filter,
        sensor_config=sensor_config,
    )
    scene_loader_traj = SceneLoader(
        openscene / f"navsim_logs/{args.split}",
        openscene / f"sensor_blobs/{args.split}",
        scene_filter,
        sensor_config=sensor_config,
        load_image_path=True,
    )

    available = set(scene_loader.tokens)
    missing = [t for t in tokens if t not in available]
    if missing:
        print(f"[warn] {len(missing)} tokens not in scene_loader, skip: {missing[:8]}")

    ok = 0
    for token in tqdm([t for t in tokens if t in available], desc=f"BEV[{args.tag}]"):
        scene = scene_loader.get_scene_from_token(token)
        scene_traj = scene_loader_traj.get_scene_from_token(token)
        frame_idx = scene.scene_metadata.num_history_frames - 1
        fig, _, _ = plot_bev_and_camera_with_agent(scene, scene_traj, frame_idx, agent)
        out = args.output_dir / f"{token}_{args.tag}_bev.png"
        fig.savefig(out, bbox_inches="tight", dpi=160)
        plt.close(fig)
        ok += 1
    print(f"Saved {ok} figures under {args.output_dir}")


if __name__ == "__main__":
    main()
