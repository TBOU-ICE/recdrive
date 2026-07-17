#!/usr/bin/env python3
"""Render short low-freq BEV video clips around selected fail tokens."""
from __future__ import annotations

import argparse
import os
import subprocess
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens-file", type=Path, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--vlm-path", type=str, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--tag", type=str, default="rule_rl")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--scene-filter", type=str, default="navtest")
    p.add_argument("--window", type=int, default=7, help="frames before/after center token")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--fps", type=int, default=2)
    return p.parse_args()


def load_tokens(path: Path) -> list[str]:
    tokens = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            tokens.append(line)
    return tokens


def main():
    args = parse_args()
    openscene = Path(os.environ["OPENSCENE_DATA_ROOT"])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokens = load_tokens(args.tokens_file)
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

    # Build per-log sorted token lists
    log_to_frames = {}
    for token, scene_frame_list in scene_loader.scene_frames_dicts.items():
        center_idx = scene_loader._scene_filter.num_history_frames - 1
        center = scene_frame_list[center_idx]
        log_name = scene_loader.token_to_log_file[token]
        log_to_frames.setdefault(log_name, []).append(center)
    for log_name, frames in log_to_frames.items():
        frames.sort(key=lambda x: x["timestamp"])

    for token in tokens:
        if token not in scene_loader.tokens:
            print(f"[skip] {token} not in loader")
            continue
        log_name = scene_loader.token_to_log_file[token]
        frames = log_to_frames[log_name]
        toks = [f["token"] for f in frames]
        try:
            center_i = toks.index(token)
        except ValueError:
            print(f"[skip] {token} not in sorted log frames")
            continue
        lo = max(0, center_i - args.window)
        hi = min(len(toks), center_i + args.window + 1)
        clip_tokens = toks[lo:hi]

        clip_dir = args.output_dir / f"{token}_{args.tag}_frames"
        clip_dir.mkdir(parents=True, exist_ok=True)
        for i, t in enumerate(tqdm(clip_tokens, desc=f"video[{token[:8]}]")):
            scene = scene_loader.get_scene_from_token(t)
            scene_traj = scene_loader_traj.get_scene_from_token(t)
            frame_idx = scene.scene_metadata.num_history_frames - 1
            fig, _, _ = plot_bev_and_camera_with_agent(scene, scene_traj, frame_idx, agent)
            fig.savefig(clip_dir / f"frame_{i:04d}.png", bbox_inches="tight", dpi=120)
            plt.close(fig)

        mp4 = args.output_dir / f"{token}_{args.tag}.mp4"
        cmd = [
            "ffmpeg", "-y", "-framerate", str(args.fps),
            "-i", str(clip_dir / "frame_%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(mp4),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"wrote {mp4}")
        except FileNotFoundError:
            print(f"[warn] ffmpeg not found; frames left in {clip_dir}")
        except subprocess.CalledProcessError as e:
            print(f"[warn] ffmpeg failed for {token}: {e.stderr[-400:] if e.stderr else e}")


if __name__ == "__main__":
    main()
