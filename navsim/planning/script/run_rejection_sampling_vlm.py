"""
Rejection sampling on the SFT-trained InternVL3 VLM.

Handles three verifiable question types:
  traj     – trajectory prediction  → PDM Score reward
  plan     – driving plan           → exact-match reward   ("KEEP, STRAIGHT")
  mot_pred – per-agent motion pred  → fraction-correct reward

Optimisations vs. naïve implementation
  1. Vision encoder runs ONCE per scene; vit_embeds are injected into every
     model.generate() call via the existing visual_features= parameter.
     (Old: 24× vision encoder / scene → New: 1× vision encoder / scene)
  2. Tokenisation runs ONCE per (scene, question-type), not once per sample.
     (Old: 24× tokenise / scene → New: 3× tokenise / scene)
  3. Per-type max_new_tokens prevent wasted decode steps on short answers.
     plan:     32 tokens  (answer ≈ "KEEP, STRAIGHT")
     mot_pred: 16 × N_obj tokens  (one line per object)
     traj:    128 tokens  (8 coordinate triples)

Output: 9 JSONL files per rank  {traj|plan|mot_pred}_{accepted|rejected|failed}_rank{N}.jsonl

Multi-GPU: torchrun; merge afterwards:
    for qt in traj plan mot_pred; do
        for s in accepted rejected failed; do
            cat ${OUT}/${qt}_${s}_rank*.jsonl > ${OUT}/${qt}_${s}_all.jsonl
        done
    done
"""

import json
import logging
import math
import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from navsim.agents.recogdrive.utils.conversation import get_conv_template
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from navsim.common.dataclasses import SceneFilter, SensorConfig, Trajectory
from navsim.common.dataloader import MetricCacheLoader, SceneLoader
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
    PDMScorer,
    PDMScorerConfig,
)
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
    PDMSimulator,
)
from navsim.planning.training.dataset import Dataset_For_Pipeline
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_rejection_sampling"

_SYSTEM_MESSAGE = (
    "You are a vehicle trajectory prediction model for autonomous driving. "
    "Your task is to predict the ego vehicle's 4-second trajectory based on "
    "multi-view images, ego vehicle states, and discrete navigation commands. "
    "The input provides a 2-second history; output a safe 4-second trajectory."
)

_PEDAL_STATUS = {
    "const": "KEEP",
    "accelerate": "ACCELERATE",
    "decelerate": "DECELERATE",
    "stop": "STOP",
}
_PATH_STATUS = {
    "right turn": "RIGHT_TURN",
    "right lane change": "RIGHT_CHANGE",
    "left turn": "LEFT_TURN",
    "left lane change": "LEFT_CHANGE",
    "straight": "STRAIGHT",
}

QTYPES   = ("traj", "plan", "mot_pred")
STATUSES = ("accepted", "rejected", "failed")

_DIS_THRESH = 40.0          # metres – matches pipeline get_mot_pred_qa default
_IMG_START  = "<img>"
_IMG_END    = "</img>"
_IMG_CTX    = "<IMG_CONTEXT>"


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helper
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(n: float, dp: int = 2) -> str:
    return f"{n:+.{dp}f}" if abs(round(n, dp)) > 1e-2 else "0.0"


# ─────────────────────────────────────────────────────────────────────────────
# GT helpers  (mirror run_generate_dataset_pipeline.py)
# ─────────────────────────────────────────────────────────────────────────────

def _pedal_from_traj(traj: np.ndarray, thresh: float = 3.0) -> str:
    vel = np.linalg.norm(traj[1:, :2] - traj[:-1, :2], axis=-1) / 0.5
    if np.max(vel) < 2.0:
        return "stop"
    dv = vel[-1] - vel[0]
    return "accelerate" if dv >= thresh else ("decelerate" if dv <= -thresh else "const")


def _path_from_traj(traj: np.ndarray, lat: float = 4.0, ang: float = 5.0) -> str:
    x, y = traj[:, 0], traj[:, 1]
    angle_diff = math.degrees(math.atan2(x[-1], y[-1])) - 90.0
    if y[-1] > lat and angle_diff <= -ang:
        return "left turn"
    if y[-1] > lat and abs(angle_diff) < ang:
        return "left lane change"
    if y[-1] <= -lat and angle_diff >= ang:
        return "right turn"
    if y[-1] <= -lat and abs(angle_diff) < ang:
        return "right lane change"
    return "straight"


def _pedal_from_vel(vel: np.ndarray, thresh: float = 3.0) -> str:
    speed = np.linalg.norm(vel[:, :2], axis=-1)
    if np.max(speed) < 2.0:
        return "stop"
    dv = speed[-1] - speed[0]
    return "accelerate" if dv >= thresh else ("decelerate" if dv <= -thresh else "const")


def _path_from_vel(vel: np.ndarray, lat: float = 4.0, ang: float = 5.0) -> str:
    x_diff = float(vel[-1, 0] - vel[-2, 0])
    y_diff = float(vel[-1, 1] - vel[-2, 1])
    angle_diff = math.degrees(math.atan2(x_diff, y_diff)) - math.degrees(float(vel[-2, 2]))
    if y_diff > lat and angle_diff <= -ang:
        return "left turn"
    if y_diff > lat and abs(angle_diff) < ang:
        return "left lane change"
    if y_diff <= -lat and angle_diff >= ang:
        return "right turn"
    if y_diff <= -lat and abs(angle_diff) < ang:
        return "right lane change"
    return "straight"


# ─────────────────────────────────────────────────────────────────────────────
# GT computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_plan_gt(future_traj: np.ndarray) -> str:
    return f"{_PEDAL_STATUS[_pedal_from_traj(future_traj)]}, {_PATH_STATUS[_path_from_traj(future_traj)]}"


def compute_mot_pred_data(
    agent_states,
    agent_labels,
    agent_names: List[str],
    future_velocities,
    dis_thresh: float = _DIS_THRESH,
) -> Tuple[List[str], List[np.ndarray], List[str], List[np.ndarray]]:
    states_np = np.array(agent_states)
    labels_np = np.array(agent_labels, dtype=bool)
    vels_np   = np.array(future_velocities)

    per_obj_gt, filt_boxes, filt_names, filt_vels = [], [], [], []
    for i in range(len(states_np)):
        if not labels_np[i] or np.linalg.norm(states_np[i, :2]) >= dis_thresh:
            continue
        vel = vels_np[i]
        per_obj_gt.append(
            f"{_PEDAL_STATUS[_pedal_from_vel(vel)]}, {_PATH_STATUS[_path_from_vel(vel)]}"
        )
        filt_boxes.append(states_np[i])
        filt_names.append(agent_names[i])
        filt_vels.append(vel)

    return per_obj_gt, filt_boxes, filt_names, filt_vels


# ─────────────────────────────────────────────────────────────────────────────
# Question builders
# ─────────────────────────────────────────────────────────────────────────────

def build_traj_question(ego_statuses) -> str:
    history = [
        {"x": _fmt(float(e.ego_pose[0])), "y": _fmt(float(e.ego_pose[1])), "heading": _fmt(float(e.ego_pose[2]))}
        for e in ego_statuses[:4]
    ]
    cmd = next(
        (c for c, v in zip(["turn left", "go straight", "turn right"], ego_statuses[-1].driving_command) if v == 1),
        "unknown",
    )
    hist_str = chr(2).join(
        f'   - t-{3-i}: ({t["x"]}, {t["y"]}, {t["heading"]})' for i, t in enumerate(history)
    )
    return (
        "<FRONT VIEW>:\n<image>\n\n"
        "As an autonomous driving system, predict the vehicle's trajectory based on:\n"
        "1. Visual perception from front camera view\n"
        f"2. Historical motion context (last 4 timesteps):{hist_str}\n"
        f"3. Active navigation command: [{cmd.upper()}]"
        "\nOutput requirements:\n- Predict 8 future trajectory points\n"
        "- Each point format: (x:float, y:float, heading:float)\n"
        "- Use [PT, ...] to encapsulate the trajectory\n"
        "- Maintain numerical precision to 2 decimal places"
    )


def build_plan_question(ego_statuses) -> str:
    history = [
        {"x": _fmt(float(e.ego_pose[0])), "y": _fmt(float(e.ego_pose[1])), "heading": _fmt(float(e.ego_pose[2]))}
        for e in ego_statuses[:4]
    ]
    cmd = next(
        (c for c, v in zip(["turn left", "go straight", "turn right"], ego_statuses[-1].driving_command) if v == 1),
        "unknown",
    )
    traj_str = chr(2).join(
        f'  - t-{3-i}: ({t["x"]}, {t["y"]}, {t["heading"]})' for i, t in enumerate(history)
    )
    return (
        "<image>\n"
        f"Your historical trajectories are {traj_str},"
        f"the navigation command is '{cmd}', "
        "based on the understanding of the driving scene and the navigation information, "
        "what is your plan for the next three seconds? "
        "Please answer your SPEED plan and your PATH plan. "
        "SPEED includes KEEP, ACCELERATE and DECELERATE, and STOP, "
        "PATH includes STRAIGHT, RIGHT_CHANGE, LEFT_CHANGE, RIGHT_TURN, LEFT_TURN. "
        "For example, a correct answer format is like 'KEEP, LEFT_CHANGE'."
    )


def build_mot_pred_question(
    filt_boxes: List[np.ndarray],
    filt_names: List[str],
    filt_vels: List[np.ndarray],
    img_type: str = "front",
) -> str:
    header = (
        "You are driving, I will now provide you with the location "
        f"and velocity information of dynamic objects in the {img_type} view image. "
        "Please predict their future driving behaviors, "
        "which can be divided into SPEED decisions and PATH decisions. "
        "SPEED includes KEEP, ACCELERATE, DECELERATE, and STOP, "
        "while PATH includes STRAIGHT, RIGHT_CHANGE, LEFT_CHANGE, RIGHT_TURN, and LEFT_TURN."
        "I will now provide you with the position and velocity information of the dynamic objects: \n"
    )
    obj_lines = []
    for k, (box, name, vel) in enumerate(zip(filt_boxes, filt_names, filt_vels), 1):
        x, y = float(box[0]), float(box[1])
        spd  = float(np.linalg.norm(vel[0, :2]))
        log_d = f"{int(x)} meters ahead" if x >= 0 else f"{abs(int(x))} meters behind"
        lat_d = f"{int(y)} meters to the left" if y >= 0 else f"{abs(int(y))} meters to the right"
        obj_lines.append(f"Object {k}: {name}, {log_d}, {lat_d}, speed of {int(spd)} m/s.")
    footer = (
        f"Please predict the future driving behaviors of these objects "
        f"based on the {img_type} view image. "
        "For example, a well-formatted answer should be like:\n"
        "Object 1: KEEP, STRAIGHT\n"
        "Object 2: DECELERATE, RIGHT_TURN\n"
        "Object 3: ACCELERATE, LEFT_CHANGE\n"
    )
    return "<image>\n" + header + "\n".join(obj_lines) + "\n" + footer


# ─────────────────────────────────────────────────────────────────────────────
# Parsers
# ─────────────────────────────────────────────────────────────────────────────

def parse_trajectory(text: str) -> Optional[np.ndarray]:
    m = re.search(r"\[PT,\s*(.*?)\]", text, re.DOTALL)
    if not m:
        return None
    pts = re.findall(
        r"\(\s*([+-]?\d+\.?\d*)\s*,\s*([+-]?\d+\.?\d*)\s*,\s*([+-]?\d+\.?\d*)\s*\)",
        m.group(1),
    )
    if len(pts) != 8:
        return None
    try:
        return np.array([[float(x), float(y), float(h)] for x, y, h in pts], dtype=np.float32)
    except ValueError:
        return None


def parse_plan(text: str) -> Optional[str]:
    m = re.search(
        r"\b(KEEP|ACCELERATE|DECELERATE|STOP)\s*,\s*"
        r"(STRAIGHT|RIGHT_CHANGE|LEFT_CHANGE|RIGHT_TURN|LEFT_TURN)\b",
        text,
    )
    return f"{m.group(1)}, {m.group(2)}" if m else None


def parse_mot_pred(text: str, num_objects: int) -> Optional[List[Optional[str]]]:
    pattern = re.compile(
        r"Object\s+(\d+)\s*:\s*(KEEP|ACCELERATE|DECELERATE|STOP)\s*,\s*"
        r"(STRAIGHT|RIGHT_CHANGE|LEFT_CHANGE|RIGHT_TURN|LEFT_TURN)"
    )
    found = {int(m.group(1)): f"{m.group(2)}, {m.group(3)}" for m in pattern.finditer(text)}
    if not found:
        return None
    return [found.get(i + 1) for i in range(num_objects)]


# ─────────────────────────────────────────────────────────────────────────────
# Image + feature loading  (optimised: extract once per scene)
# ─────────────────────────────────────────────────────────────────────────────

def _load_image(image_path: str, max_num: int = 12) -> Tuple[torch.Tensor, int]:
    from navsim.agents.recogdrive.utils.internvl_preprocess import load_image as _li
    pv = _li(image_path, max_num=max_num)
    return pv, pv.shape[0]


@torch.no_grad()
def _extract_features(model, pixel_values: torch.Tensor) -> torch.Tensor:
    """Run vision encoder + MLP projection once per scene. Returns (num_patches, C)."""
    return model.extract_feature(pixel_values)


# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers  (tokenise once, decode N times with cached vit_embeds)
# ─────────────────────────────────────────────────────────────────────────────

def _build_inputs(
    model,
    tokenizer,
    question: str,
    num_patches_list: List[int],
) -> Tuple[torch.Tensor, torch.Tensor, int, str]:
    """
    Replicate the tokenisation step of model.chat().
    Returns (input_ids, attention_mask, eos_token_id, sep_str) on the model device.
    Tokenise ONCE per (scene, question-type), not once per sample.
    sep_str is cached here so _sample_responses never re-creates the template.
    """
    if "<image>" not in question:
        question = "<image>\n" + question

    img_ctx_id = tokenizer.convert_tokens_to_ids(_IMG_CTX)
    model.img_context_token_id = img_ctx_id

    template = get_conv_template(model.template)
    template.system_message = model.system_message
    sep_str      = template.sep.strip()
    eos_token_id = tokenizer.convert_tokens_to_ids(sep_str)

    template.append_message(template.roles[0], question)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()

    for n in num_patches_list:
        image_tokens = _IMG_START + _IMG_CTX * model.num_image_token * n + _IMG_END
        query = query.replace("<image>", image_tokens, 1)

    enc = tokenizer(query, return_tensors="pt")
    device = next(model.parameters()).device
    return enc["input_ids"].to(device), enc["attention_mask"].to(device), eos_token_id, sep_str


def _sample_responses(
    model,
    pixel_values: torch.Tensor,
    vit_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    eos_token_id: int,
    sep_str: str,
    tokenizer,
    gen_cfg: dict,
    num_samples: int,
) -> Tuple[List[str], int]:
    """
    Generate num_samples responses.  Returns (responses, n_errors).
    Vision encoder is NOT invoked; vit_embeds is injected via visual_features=.
    input_ids / attention_mask are reused across all samples (same question).
    sep_str decoded once in _build_inputs – no template re-creation per call.
    """
    responses: List[str] = []
    n_errors = 0
    cfg = dict(gen_cfg, eos_token_id=eos_token_id)
    for _ in range(num_samples):
        try:
            with torch.no_grad():
                out = model.generate(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    visual_features=vit_embeds,
                    **cfg,
                )
            text = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
            responses.append(text.split(sep_str)[0].strip())
        except Exception as e:
            logger.debug(f"gen error: {e}")
            n_errors += 1
    return responses, n_errors


# ─────────────────────────────────────────────────────────────────────────────
# Output writer
# ─────────────────────────────────────────────────────────────────────────────

class _TypedWriters:
    """Context manager: 9 JSONL handles (3 qtypes × 3 statuses) per rank."""

    def __init__(self, out_dir: Path, rank: int) -> None:
        self._paths: Dict[Tuple[str, str], Path] = {
            (qt, st): out_dir / f"{qt}_{st}_rank{rank}.jsonl"
            for qt in QTYPES for st in STATUSES
        }
        self._handles: Dict = {}

    def __enter__(self) -> "_TypedWriters":
        for key, path in self._paths.items():
            self._handles[key] = open(path, "w", encoding="utf-8")
        return self

    def write(self, qtype: str, status: str, record: dict) -> None:
        self._handles[(qtype, status)].write(json.dumps(record, ensure_ascii=False) + "\n")

    def __exit__(self, *_) -> None:
        for fh in self._handles.values():
            fh.close()

    @property
    def paths(self) -> Dict:
        return self._paths


# ─────────────────────────────────────────────────────────────────────────────
# Record building
# ─────────────────────────────────────────────────────────────────────────────

def _make_record(
    token: str,
    qtype: str,
    image_path: str,
    question: str,
    gt_answer: str,
    raw_responses: List[str],
    scored: List[Tuple[str, object, float]],
    threshold: float,
) -> Tuple[dict, str]:
    if not scored:
        status    = "failed"
        best_resp  = raw_responses[-1] if raw_responses else ""
        best_reward = None
    else:
        best_reward = max(s[2] for s in scored)
        best_resp   = max(scored, key=lambda x: x[2])[0]
        status      = "accepted" if best_reward >= threshold else "rejected"

    return {
        "id":            f"{token}_{qtype}",
        "token":         token,
        "question_type": qtype,
        "status":        status,
        "image":         [image_path],
        "conversations": [
            {"from": "human", "value": question},
            {"from": "gpt",   "value": best_resp},
        ],
        "gt_answer":     gt_answer,
        "all_samples":   [{"response": r, "reward": s} for r, _, s in scored],
        "best_reward":   best_reward,
        "num_generated": len(raw_responses),
        "num_valid":     len(scored),
    }, status


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    rank       = int(os.getenv("RANK", 0))
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"

    rs = cfg.rs

    # ── 1. VLM ──────────────────────────────────────────────────────────────
    logger.info(f"[Rank {rank}] Loading VLM from {rs.vlm_path}")
    model = AutoModel.from_pretrained(
        rs.vlm_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        use_flash_attn=True,
        device_map=device,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        rs.vlm_path, trust_remote_code=True, use_fast=False
    )
    model.system_message = _SYSTEM_MESSAGE

    # ── 2. PDM scorer (traj only) ────────────────────────────────────────────
    proposal_sampling = TrajectorySampling(time_horizon=4, interval_length=0.1)
    simulator = PDMSimulator(proposal_sampling)
    scorer    = PDMScorer(
        proposal_sampling,
        PDMScorerConfig(progress_weight=10.0, ttc_weight=5.0, comfortable_weight=2.0),
    )
    metric_cache_loader = MetricCacheLoader(Path(rs.metric_cache_path))

    # ── 3. Dataset ───────────────────────────────────────────────────────────
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    scene_filter.log_names = list(cfg.train_logs)

    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_all_sensors(include=[0, 1, 2, 3]),
        load_image_path=True,
    )
    dataset = Dataset_For_Pipeline(
        scene_loader=scene_loader,
        feature_builders=[],
        target_builders=[],
        cache_path=None,
    )

    my_indices = list(range(rank, len(dataset), world_size))
    logger.info(f"[Rank {rank}] {len(my_indices)} / {len(dataset)} scenes assigned")

    # ── 4. Config ────────────────────────────────────────────────────────────
    num_samples    = OmegaConf.select(rs, "num_samples",        default=8)
    pdm_threshold  = OmegaConf.select(rs, "pdm_threshold",      default=0.4)
    plan_threshold = OmegaConf.select(rs, "plan_threshold",     default=1.0)
    mot_threshold  = OmegaConf.select(rs, "mot_pred_threshold", default=0.5)
    temperature    = OmegaConf.select(rs, "temperature",        default=0.7)
    top_p          = OmegaConf.select(rs, "top_p",              default=0.9)
    max_num_tiles  = OmegaConf.select(rs, "max_num_tiles",      default=12)
    out_dir        = Path(OmegaConf.select(rs, "output_dir",    default="rs_output"))
    log_every      = OmegaConf.select(rs, "log_every",          default=500)

    max_scenes = OmegaConf.select(rs, "max_scenes", default=None)   # None = all scenes
    if max_scenes is not None:
        my_indices = my_indices[: int(max_scenes)]
        logger.info(f"[Rank {rank}] DEBUG: capped to {len(my_indices)} scenes")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-type generation configs  (shorter outputs → smaller max_new_tokens)
    _base = dict(do_sample=True, temperature=temperature, top_p=top_p)
    gen_cfg_traj = dict(_base, max_new_tokens=128)   # [PT, (x,y,h)×8] ≈ 70 tokens
    gen_cfg_plan = dict(_base, max_new_tokens=32)    # "KEEP, STRAIGHT"  ≈  5 tokens
    # mot_pred max_new_tokens computed per-scene based on object count (below)

    # ── 5. Stats ─────────────────────────────────────────────────────────────
    stats: Dict = {qt: {st: 0 for st in STATUSES} for qt in QTYPES}
    stats["skip"] = {"no_cache": 0, "img_err": 0, "no_agents": 0, "gen_err": 0}

    # ── 6. Rejection-sampling loop ───────────────────────────────────────────
    with _TypedWriters(out_dir, rank) as writers:
        for step, i in enumerate(
            tqdm(my_indices, desc=f"Rank {rank} – RS", dynamic_ncols=True)
        ):
            (
                ego_statuses, cameras, future_trajectory,
                agent_states, agent_labels, agent_names,
                token, future_velocities, box_2d,
            ) = dataset[i]

            image_path = str(cameras[-1].cam_f0.image)

            # ── load image + extract vision features ONCE per scene ──────────
            try:
                pixel_values, num_patches = _load_image(image_path, max_num_tiles)
                pixel_values = pixel_values.to(device=device, dtype=torch.bfloat16)
                vit_embeds   = _extract_features(model, pixel_values)
            except Exception as e:
                logger.warning(f"[{token}] Image/feature error: {e}")
                stats["skip"]["img_err"] += 1
                continue

            num_patches_list = [num_patches]
            future_traj_np = (
                future_trajectory.numpy()
                if hasattr(future_trajectory, "numpy") else np.array(future_trajectory)
            )

            # ── TRAJ ─────────────────────────────────────────────────────────
            if token in metric_cache_loader.metric_cache_paths:
                metric_cache = metric_cache_loader.get_from_token(token)
                traj_q = build_traj_question(ego_statuses)
                gt_traj_str = (
                    "Here is the planning trajectory [PT, "
                    + ", ".join(
                        f"({_fmt(float(p[0]))}, {_fmt(float(p[1]))}, {_fmt(float(p[2]))})"
                        for p in future_traj_np
                    )
                    + "]."
                )

                # tokenise once, decode num_samples times
                t_ids, t_mask, t_eos, t_sep = _build_inputs(
                    model, tokenizer, traj_q, num_patches_list
                )
                traj_raw, n_err = _sample_responses(
                    model, pixel_values, vit_embeds,
                    t_ids, t_mask, t_eos, t_sep, tokenizer, gen_cfg_traj, num_samples,
                )
                stats["skip"]["gen_err"] += n_err

                traj_scored: List[Tuple[str, object, float]] = []
                for resp in traj_raw:
                    traj_arr = parse_trajectory(resp)
                    if traj_arr is None:
                        continue
                    try:
                        result = pdm_score(
                            metric_cache=metric_cache,
                            model_trajectory=Trajectory(traj_arr),
                            future_sampling=proposal_sampling,
                            simulator=simulator,
                            scorer=scorer,
                        )
                        traj_scored.append((resp, traj_arr.tolist(), float(asdict(result)["score"])))
                    except Exception:
                        pass

                rec, status = _make_record(
                    token, "traj", image_path, traj_q, gt_traj_str,
                    traj_raw, traj_scored, pdm_threshold,
                )
                writers.write("traj", status, rec)
                stats["traj"][status] += 1
            else:
                stats["skip"]["no_cache"] += 1

            # ── PLAN ─────────────────────────────────────────────────────────
            gt_plan = compute_plan_gt(future_traj_np)
            plan_q  = build_plan_question(ego_statuses)

            p_ids, p_mask, p_eos, p_sep = _build_inputs(
                model, tokenizer, plan_q, num_patches_list
            )
            plan_raw, n_err = _sample_responses(
                model, pixel_values, vit_embeds,
                p_ids, p_mask, p_eos, p_sep, tokenizer, gen_cfg_plan, num_samples,
            )
            stats["skip"]["gen_err"] += n_err

            plan_scored: List[Tuple[str, object, float]] = []
            for resp in plan_raw:
                parsed = parse_plan(resp)
                if parsed is not None:
                    reward = 1.0 if parsed == gt_plan else 0.0
                    plan_scored.append((resp, parsed, reward))

            rec, status = _make_record(
                token, "plan", image_path, plan_q, gt_plan,
                plan_raw, plan_scored, plan_threshold,
            )
            writers.write("plan", status, rec)
            stats["plan"][status] += 1

            # ── MOT_PRED ─────────────────────────────────────────────────────
            per_obj_gt, filt_boxes, filt_names, filt_vels = compute_mot_pred_data(
                agent_states, agent_labels, agent_names, future_velocities
            )

            if not per_obj_gt:
                stats["skip"]["no_agents"] += 1
            else:
                n_obj = len(per_obj_gt)
                mot_q = build_mot_pred_question(filt_boxes, filt_names, filt_vels)
                gt_mot_str = (
                    "\n".join(f"Object {j+1}: {gt}" for j, gt in enumerate(per_obj_gt)) + "\n"
                )
                # max_new_tokens scales with object count (each line ≈ 16 tokens)
                gen_cfg_mot = dict(_base, max_new_tokens=max(64, n_obj * 16))

                m_ids, m_mask, m_eos, m_sep = _build_inputs(
                    model, tokenizer, mot_q, num_patches_list
                )
                mot_raw, n_err = _sample_responses(
                    model, pixel_values, vit_embeds,
                    m_ids, m_mask, m_eos, m_sep, tokenizer, gen_cfg_mot, num_samples,
                )
                stats["skip"]["gen_err"] += n_err

                mot_scored: List[Tuple[str, object, float]] = []
                for resp in mot_raw:
                    parsed_list = parse_mot_pred(resp, n_obj)
                    if parsed_list is not None:
                        correct = sum(1 for pred, gt in zip(parsed_list, per_obj_gt) if pred == gt)
                        mot_scored.append((resp, parsed_list, correct / n_obj))

                rec, status = _make_record(
                    token, "mot_pred", image_path, mot_q, gt_mot_str,
                    mot_raw, mot_scored, mot_threshold,
                )
                writers.write("mot_pred", status, rec)
                stats["mot_pred"][status] += 1

            # ── periodic log ─────────────────────────────────────────────────
            if (step + 1) % log_every == 0:
                logger.info(
                    f"[Rank {rank}] step {step+1}/{len(my_indices)} | "
                    + " | ".join(
                        f"{qt}: acc={stats[qt]['accepted']} "
                        f"rej={stats[qt]['rejected']} "
                        f"fail={stats[qt]['failed']}"
                        for qt in QTYPES
                    )
                )

    # ── 7. Final summary ─────────────────────────────────────────────────────
    logger.info(
        f"\n{'='*70}\n"
        f"[Rank {rank}] Rejection Sampling Complete\n"
        + "\n".join(
            f"  {qt:10s}: accepted={stats[qt]['accepted']:6d}  "
            f"rejected={stats[qt]['rejected']:6d}  "
            f"failed={stats[qt]['failed']:6d}"
            for qt in QTYPES
        )
        + f"\n  skipped : no_cache={stats['skip']['no_cache']}  "
        f"img_err={stats['skip']['img_err']}  "
        f"no_agents={stats['skip']['no_agents']}  "
        f"gen_err={stats['skip']['gen_err']}\n"
        f"  output dir : {out_dir}\n"
        f"{'='*70}"
    )


if __name__ == "__main__":
    main()
