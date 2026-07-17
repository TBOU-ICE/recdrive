"""
Generate InternVL SFT data from SimScale scenes, following the ReCogDrive data
pipeline (run_generate_dataset.py + run_generate_dataset_pipeline.py) but adapted
for SimScale:

  * Works on the SimScale split (train_test_split=simscale_pdm_round0 / round1)
    and bypasses the navtrain `train_logs` filter (loads all logs in the split).
  * Shards work by LOG FILE across RANK/WORLD_SIZE (or SHARD_INDEX/SHARD_COUNT)
    so many CPU processes can run in parallel, each loading only its own logs.
  * Phase-1 default emits only rule-based (GT-derived) QA — NO external VLM
    needed. The subjective Qwen-authored turns (scene description, traffic
    light, road sign, driving influence, plan explanation, driving behavior)
    are gated behind USE_VLM=1 for a later phase.

Two JSONL products per shard (mirroring the two meta entries Navsim / Navsim_QA):
  * <out>/simscale_<dataset>_traj_shard{i}of{n}.jsonl   -> trajectory QA
  * <out>/simscale_<dataset>_qa_shard{i}of{n}.jsonl     -> rule-based multi-turn QA

Rule-based QA logic (plan / motion / vru / distance / 3d_info / 3d_det) is copied
verbatim from run_generate_dataset_pipeline.py to keep the output distribution
identical to the reference.

Run (single process):
  OPENSCENE_DATA_ROOT=/workspace/datasets/simscale/20260709 \
  python navsim/planning/script/run_generate_dataset_simscale.py \
      train_test_split=simscale_pdm_round0

Environment knobs:
  SHARD_INDEX / SHARD_COUNT  explicit shard (fallback to RANK / WORLD_SIZE, then 0/1)
  SIMSCALE_QA_OUT_DIR        output dir (default: $NAVSIM_EXP_ROOT/simscale_vlm_qa/<dataset>)
  SIMSCALE_IMAGE_ROOT        image paths stored relative to this (default OPENSCENE_DATA_ROOT)
  SIMSCALE_MAX_SCENES        cap number of scenes (for smoke tests; default unlimited)
  SIMSCALE_EMIT              which products: 'both' (default) | 'traj' | 'qa'
  VRU_KEEP_EMPTY_PROB        keep probability for empty-VRU answers (default 0.1)
  USE_VLM                    '1' to add Qwen-authored subjective QA (default '0')
  QWEN_BASE_URL / QWEN_API_KEY / QWEN_MODEL  OpenAI-compatible endpoint (USE_VLM=1)
  SIMSCALE_VQA_CACHE_DIR     cache dir for VLM answers (default: <out>/vqa_cache)
  RESUME                     '1' to skip tokens already present in output shard
"""

from typing import List, Optional, Tuple
from pathlib import Path
import logging
import os
import math
import json
import random

import numpy as np
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl

from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset import Dataset_For_Pipeline

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"

pedal_status = {'const': 'KEEP', 'accelerate': 'ACCELERATE', 'decelerate': 'DECELERATE', 'stop': 'STOP'}
path_status = {
    'right turn': 'RIGHT_TURN', 'right lane change': 'RIGHT_CHANGE',
    'left turn': 'LEFT_TURN', 'left lane change': 'LEFT_CHANGE', 'straight': 'STRAIGHT',
}

system_message = """
You are a vehicle trajectory prediction model for autonomous driving. Your task is to predict the ego vehicle's 4-second trajectory based on the following inputs: multi-view images from 8 cameras, ego vehicle states (position), and discrete navigation commands. The input provides a 2-second history, and your output should ensure a safe trajectory for the next 4 seconds. Your predictions must adhere to the following metrics:
1. **No at-fault Collisions (NC)**: Avoid collisions with other objects/vehicles.
2. **Drivable Area Compliance (DAC)**: Stay within the drivable area.
3. **Time to Collision (TTC)**: Maintain a safe distance from other vehicles.
4. **Ego Progress (EP)**: Ensure the ego vehicle moves forward without being stuck.
5. **Comfort (C)**: Avoid sharp turns and sudden decelerations.
6. **Driving Direction Compliance (DDC)**: Align with the intended driving direction.
For evaluation, use the **PDM Score**, which combines these metrics: **PDM Score** = NC * DAC * (5*TTC + 5*EP + 2*C + 0*DDC) / 12.
Your predictions will be evaluated through a non-reactive 4-second simulation with an LQR controller and background actors following their recorded trajectories. The better your predictions, the higher your score.
"""


# --------------------------------------------------------------------------------------
# Rule-based helpers (verbatim from run_generate_dataset_pipeline.py)
# --------------------------------------------------------------------------------------
def format_number(n, decimal_places=2):
    try:
        import torch
        if isinstance(n, torch.Tensor):
            n = n.item()
    except Exception:
        pass
    if abs(round(float(n), decimal_places)) <= 1e-2:
        return 0.0
    return f"{float(n):+.{decimal_places}f}"


def get_obj_acc_or_dec(trajectory, vel_diff_thresh=3.0):
    velocity = np.linalg.norm(trajectory[1:, :2] - trajectory[:-1, :2], axis=-1) / 0.5
    if np.max(velocity) < 2.0:
        return "stop"
    vel_diff = velocity[-1] - velocity[0]
    if vel_diff >= vel_diff_thresh:
        return "accelerate"
    elif vel_diff <= -vel_diff_thresh:
        return "decelerate"
    return "const"


def get_obj_turn_or_lane_change(trajectory, lat_thresh=4.0, angle_thresh=5.0):
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    endpoint_angle = math.degrees(math.atan2(x[-1], y[-1]))
    angle_diff = endpoint_angle - 90.0
    if y[-1] > lat_thresh and angle_diff <= -angle_thresh:
        return "left turn"
    elif y[-1] > lat_thresh and abs(angle_diff) < angle_thresh:
        return "left lane change"
    elif y[-1] <= -lat_thresh and angle_diff >= angle_thresh:
        return "right turn"
    elif y[-1] <= -lat_thresh and abs(angle_diff) < angle_thresh:
        return "right lane change"
    return "straight"


def get_obj_acc_or_dec_from_vel(velocity, vel_diff_thresh=3.0):
    speed = np.linalg.norm(velocity[:, :2], axis=-1)
    vel_diff = speed[-1] - speed[0]
    if np.max(speed) < 2.0:
        return "stop"
    elif vel_diff >= vel_diff_thresh:
        return "accelerate"
    elif vel_diff <= -vel_diff_thresh:
        return "decelerate"
    return "const"


def get_obj_turn_or_lane_change_from_vel(velocity, lat_thresh=4.0, angle_thresh=5.0):
    x_diff = velocity[-1, 0] - velocity[-2, 0]
    y_diff = velocity[-1, 1] - velocity[-2, 1]
    endpoint_angle = math.degrees(math.atan2(x_diff, y_diff))
    angle_diff = endpoint_angle - math.degrees(velocity[-2, 2])
    if y_diff > lat_thresh and angle_diff <= -angle_thresh:
        return "left turn"
    elif y_diff > lat_thresh and abs(angle_diff) < angle_thresh:
        return "left lane change"
    elif y_diff <= -lat_thresh and angle_diff >= angle_thresh:
        return "right turn"
    elif y_diff <= -lat_thresh and abs(angle_diff) < angle_thresh:
        return "right lane change"
    return "straight"


def get_decision(ego_speed_plan, ego_path_plan):
    pedal_decision = {'KEEP': 'maintain the current speed', 'ACCELERATE': 'accelerate',
                      'DECELERATE': 'decelerate', 'STOP': 'stop the car'}
    path_decision = {'RIGHT_TURN': 'turn right', 'RIGHT_CHANGE': 'change to the right lane',
                     'LEFT_TURN': 'turn left', 'LEFT_CHANGE': 'change to the left lane',
                     'STRAIGHT': 'go straight'}
    if ego_speed_plan == 'STOP':
        return pedal_decision[ego_speed_plan]
    return pedal_decision[ego_speed_plan] + ' and ' + path_decision[ego_path_plan]


def normalize_coordinates(box, image_width, image_height):
    x1, y1, x2, y2 = box.tolist() if hasattr(box, "tolist") else list(box)
    return [
        round((x1 / image_width) * 1000),
        round((y1 / image_height) * 1000),
        round((x2 / image_width) * 1000),
        round((y2 / image_height) * 1000),
    ]


def get_image_size(path: str, default=(1920, 1080)):
    """Return (width, height).

    The reference pipeline uses cv2.imread (full JPEG decode) just to read the
    image dimensions. That is wasteful, so we prefer:
      1. SIMSCALE_IMAGE_SIZE env override (e.g. "1920x1080") -> zero IO
      2. PIL lazy header read (Image.open(...).size) -> reads header only
      3. cv2 fallback (full decode)
    """
    env = os.environ.get("SIMSCALE_IMAGE_SIZE", "")
    if env:
        try:
            w, h = env.lower().split("x")
            return int(w), int(h)
        except Exception:
            pass
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size  # (width, height)
    except Exception:
        try:
            import cv2
            cv2.setNumThreads(0)
            img = cv2.imread(path)
            if img is not None:
                return img.shape[1], img.shape[0]
        except Exception:
            pass
    return default


# --------------------------------------------------------------------------------------
# Optional VLM (Qwen / local ReCogDrive-VLM) for subjective QA (phase 2)
# --------------------------------------------------------------------------------------
class VLMClient:
    """Lazy OpenAI-compatible client; only constructed when USE_VLM=1."""

    def __init__(self):
        from openai import OpenAI
        import io, base64  # noqa: F401
        self._io = io
        self._base64 = base64
        self._Image = __import__("PIL.Image", fromlist=["Image"])
        base_url = os.environ.get("QWEN_BASE_URL", "")
        api_key = os.environ.get("QWEN_API_KEY", "EMPTY")
        if not base_url:
            raise RuntimeError("USE_VLM=1 but QWEN_BASE_URL is not set")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        model_env = os.environ.get("QWEN_MODEL", "")
        self.model_id = model_env or self.client.models.list().data[0].id

    def infer(self, query: str, img_file: str, max_retries: int = 6) -> str:
        import time
        for attempt in range(max_retries):
            try:
                with open(img_file, "rb") as f:
                    img = self._Image.Image.open(f) if hasattr(self._Image, "Image") else __import__("PIL.Image", fromlist=["open"]).open(f)
                    img = img.resize((960, 540))
                    buf = self._io.BytesIO()
                    img.convert("RGB").save(buf, format="JPEG")
                    buf.seek(0)
                    enc = self._base64.b64encode(buf.read()).decode("utf-8")
                resp = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[{"role": "user", "content": [
                        {"type": "text", "text": query},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{enc}"}},
                    ]}],
                    timeout=600,
                )
                return resp.choices[0].message.content
            except Exception as e:  # noqa: BLE001
                if attempt < max_retries - 1:
                    time.sleep(5)
                else:
                    logger.warning(f"VLM infer failed: {e}")
                    return ""
        return ""


# --------------------------------------------------------------------------------------
# QA builders
# --------------------------------------------------------------------------------------
def build_history_and_command(ego_statuses):
    history_trajectory = []
    for i in range(4):
        ego = ego_statuses[i]
        history_trajectory.append({
            "x": format_number(ego.ego_pose[0]),
            "y": format_number(ego.ego_pose[1]),
            "heading": format_number(ego.ego_pose[2]),
        })
    high_command_one_hot = ego_statuses[-1].driving_command
    navigation_commands = ['turn left', 'go straight', 'turn right']
    command_str = [navigation_commands[i] for i in range(len(high_command_one_hot)) if high_command_one_hot[i] == 1]
    command_str = command_str[0] if command_str else "unknown"
    return history_trajectory, command_str


def build_trajectory_qa(ego_statuses, image_rel_path, future_trajectory, idx, token, prompt_type="base"):
    """Faithful port of run_generate_dataset.py process_data_and_create_qa_pair (single cam)."""
    history_trajectory, command_str = build_history_and_command(ego_statuses)

    future_points = [
        f"({format_number(p[0])}, {format_number(p[1])}, {format_number(p[2])})" for p in future_trajectory
    ]
    future_trajectory_str = f"Here is the planning trajectory [PT, {', '.join(future_points)}]."

    image_prompt = "1. Visual perception from front camera view\n"
    hist_join = " ".join(
        ["-t-{}: ({}, {}, {})".format(3 - i, t["x"], t["y"], t["heading"])
         for i, t in enumerate(history_trajectory)]
    )
    common_prompt = (
        f"As an autonomous driving system, predict the vehicle's trajectory based on:\n{image_prompt}"
        f"2. Historical motion context (last 4 timesteps):{hist_join}\n"
        f"3. Active navigation command: [{command_str.upper()}]"
    )
    output_requirements = (
        "\nOutput requirements:\n- Predict 8 future trajectory points\n"
        "- Each point format: (x:float, y:float, heading:float)\n"
        "- Use [PT, ...] to encapsulate the trajectory\n"
        "- Maintain numerical precision to 2 decimal places"
    )
    question = f"<image>\n\n{common_prompt}{output_requirements}"

    return {
        "id": idx,
        "image": [image_rel_path],
        "token": token,
        "conversations": [
            {"from": "system", "value": system_message},
            {"from": "human", "value": question},
            {"from": "gpt", "value": future_trajectory_str},
        ],
    }, future_points, history_trajectory, command_str


def build_plan_qa(future_points, history_trajectory, command_str):
    fut_np = np.array([[float(p.split(',')[0][1:]), float(p.split(',')[1]), float(p.split(',')[2][:-1])] for p in future_points])
    ego_speed_plan = pedal_status[get_obj_acc_or_dec(fut_np)]
    ego_path_plan = path_status[get_obj_turn_or_lane_change(fut_np)]
    trajectory_str = chr(2).join([f'  - t-{3-i}: ({t["x"]}, {t["y"]}, {t["heading"]})' for i, t in enumerate(history_trajectory)])
    question = (
        f"Your historical trajectories are {trajectory_str},"
        f"the navigation command is '{command_str}', "
        "based on the understanding of the driving scene and the navigation information, "
        "what is your plan for the next three seconds? "
        "Please answer your SPEED plan and your PATH plan. "
        "SPEED includes KEEP, ACCELERATE and DECELERATE, and STOP, "
        "PATH includes STRAIGHT, RIGHT_CHANGE, LEFT_CHANGE, RIGHT_TURN, LEFT_TURN. "
        "For example, a correct answer format is like 'KEEP, LEFT_CHANGE'."
    )
    answer = ego_speed_plan + ', ' + ego_path_plan + '\n'
    return question, answer


def build_mot_pred_qa(agent_boxes, agent_names, agent_vel, dis_thresh=40.0):
    img_type = 'front'
    question = (
        "You are driving, I will now provide you with the location "
        f"and velocity information of dynamic objects in the {img_type} view image. "
        "Please predict their future driving behaviors, "
        "which can be divided into SPEED decisions and PATH decisions. "
        "SPEED includes KEEP, ACCELERATE, DECELERATE, and STOP, "
        "while PATH includes STRAIGHT, RIGHT_CHANGE, LEFT_CHANGE, RIGHT_TURN, and LEFT_TURN."
        "I will now provide you with the position and velocity information of the dynamic objects: \n"
    )
    obj_cnt = 0
    answer = ""
    for i in range(len(agent_boxes)):
        box = agent_boxes[i]
        if np.linalg.norm(box[:2], axis=-1) >= dis_thresh:
            continue
        x_dis, y_dis = box[0], box[1]
        obj_vel = agent_vel[i]
        obj_pedal_status = get_obj_acc_or_dec_from_vel(obj_vel)
        obj_wheel_status = get_obj_turn_or_lane_change_from_vel(obj_vel)
        obj_speed_plan = pedal_status[obj_pedal_status]
        obj_path_plan = path_status[obj_wheel_status]
        obj_cls = agent_names[i]
        obj_speed = np.linalg.norm(obj_vel[0, :2], axis=-1)
        obj_cnt += 1
        log_describe = f"{int(x_dis)} meters ahead" if x_dis >= 0 else f"{abs(int(x_dis))} meters behind"
        lat_describe = f"{int(y_dis)} meters to the left" if y_dis >= 0 else f"{abs(int(y_dis))} meters to the right"
        question += f'Object {obj_cnt}: {obj_cls}, {log_describe}, {lat_describe}, speed of {int(obj_speed)} m/s.\n'
        answer += f"Object {obj_cnt}: {obj_speed_plan}, {obj_path_plan}\n"
    if obj_cnt == 0:
        return None, None
    question += (
        "Please predict the future driving behaviors of these objects "
        f"based on the {img_type} view image. "
        "For example, a well-formatted answer should be like:\n"
        "Object 1: KEEP, STRAIGHT\n"
        "Object 2: DECELERATE, RIGHT_TURN\n"
        "Object 3: ACCELERATE, LEFT_CHANGE\n"
    )
    return question, answer


def build_vru_qa(agent_boxes, agent_names, vru_dis_thresh=40.0):
    question = (
        f"Do you see any vulnerable road users within {int(vru_dis_thresh)} meters ahead of you, "
        "such as cyclists, motorcyclists, or pedestrians?"
    )
    vru_list = []
    vru_classes = ['motorcycle', 'pedestrian', 'bicycle']
    for i in range(len(agent_boxes)):
        box = agent_boxes[i]
        obj_loc = box[:2]
        obj_cls = agent_names[i]
        x_dis, y_dis = box[0], box[1]
        if obj_cls in vru_classes and np.linalg.norm(obj_loc) < vru_dis_thresh:
            if y_dis <= -2.0:
                lat_pos = f" and {float(abs(y_dis)):.2f} meters to the right"
            elif y_dis >= 2.0:
                lat_pos = f" and {float(abs(y_dis)):.2f} meters to the left"
            else:
                lat_pos = ""
            vru_list.append(f"a {obj_cls} located {float(abs(x_dis)):.2f} meters ahead of me{lat_pos}")
    if vru_list:
        answer = "Yes, I see " + ", and ".join(vru_list) + "."
        return question, answer, True
    answer = ("No, I don't see any vulnerable road users ahead of me, "
              "such as bicycles, motorcycles, or pedestrians.")
    return question, answer, False


def build_3d_det_qa(agent_boxes, agent_names):
    if len(agent_boxes) == 0:
        return None, None
    question = (
        "Detect every bicycle, pedestrian, and vehicle in 3D sorted from nearest to farthest and respond "
        "in the format: Object N: (x, y, z, l, w, h, heading, name). Where x, y, z are the center coordinates "
        "of the object in ego-coordinate system, l, w, h are length, width, height of the bounding box, "
        "heading is the object heading, and name is the object class name, sorted by distance from nearest to farthest."
    )
    lines = []
    for i in range(len(agent_boxes)):
        b = agent_boxes[i]
        lines.append(f"Object {i+1}: ({b[0]:.2f}, {b[1]:.2f}, {b[2]:.2f}, {b[3]:.2f}, {b[4]:.2f}, {b[5]:.2f}, {b[6]:.2f}, {agent_names[i]})")
    return question, " ".join(lines)


def _boxes_overlap(box1, box2):
    x1a, y1a, x2a, y2a = box1
    x1b, y1b, x2b, y2b = box2
    x_overlap = max(0, min(x2a, x2b) - max(x1a, x1b))
    y_overlap = max(0, min(y2a, y2b) - max(y1a, y1b))
    return x_overlap > 0 and y_overlap > 0


def build_distance_qa(agent_boxes, agent_names, box_2d, image_width, image_height):
    if len(box_2d) < 2:
        return None, None
    filtered_boxes, filtered_agent_boxes, filtered_names = [], [], []
    for i, box in enumerate(box_2d):
        x1, y1, x2, y2 = box.tolist() if hasattr(box, "tolist") else list(box)
        if x1 >= 0 and y1 >= 0 and x2 <= image_width and y2 <= image_height:
            filtered_boxes.append(box)
            filtered_agent_boxes.append(agent_boxes[i])
            filtered_names.append(agent_names[i])
    if len(filtered_boxes) < 2:
        return None, None
    valid_pairs = []
    n = len(filtered_boxes)
    for i in range(n):
        for j in range(i + 1, n):
            bi = filtered_boxes[i].tolist() if hasattr(filtered_boxes[i], "tolist") else list(filtered_boxes[i])
            bj = filtered_boxes[j].tolist() if hasattr(filtered_boxes[j], "tolist") else list(filtered_boxes[j])
            if not _boxes_overlap(bi, bj):
                valid_pairs.append((i, j))
    if not valid_pairs:
        return None, None
    i, j = random.choice(valid_pairs)
    nb1 = normalize_coordinates(filtered_boxes[i], image_width, image_height)
    nb2 = normalize_coordinates(filtered_boxes[j], image_width, image_height)
    obj1 = f"<{filtered_names[i]}><FRONT VIEW><box>{nb1}</box>"
    obj2 = f"<{filtered_names[j]}><FRONT VIEW><box>{nb2}</box>"
    templates = [
        f"How far apart are the {obj1} and the {obj2}?",
        f"What is the distance between the {obj1} and the {obj2}?",
        f"Calculate the separation between the {obj1} and the {obj2}.",
        f"Can you tell me the distance between the {obj1} and the {obj2}?",
        f"What's the gap between the {obj1} and the {obj2}?",
    ]
    dist_3d = np.linalg.norm(np.array(filtered_agent_boxes[i][:2]) - np.array(filtered_agent_boxes[j][:2]))
    return random.choice(templates), f"The {obj1} and the {obj2} are approximately {dist_3d:.2f} meters apart."


def build_3d_info_qa(agent_boxes, agent_names, box_2d, image_width, image_height, max_pairs=5):
    if len(box_2d) == 0:
        return []
    filtered_boxes, filtered_agent_boxes, filtered_names = [], [], []
    for i, box in enumerate(box_2d):
        x1, y1, x2, y2 = box.tolist() if hasattr(box, "tolist") else list(box)
        if x1 >= 0 and y1 >= 0 and x2 <= image_width and y2 <= image_height:
            filtered_boxes.append(box)
            filtered_agent_boxes.append(agent_boxes[i])
            filtered_names.append(agent_names[i])
    if len(filtered_boxes) == 0:
        return []
    indices = list(range(len(filtered_boxes)))
    if len(indices) > max_pairs:
        indices = random.sample(indices, max_pairs)
    qa = []
    for i in indices:
        nb = normalize_coordinates(filtered_boxes[i], image_width, image_height)
        templates = [
            f"What is the 3D information of the object with the 2D box {nb}?",
            f"Please provide the 3D details for the object whose 2D bounding box is {nb}.",
            f"Can you tell me the 3D information for the object located at 2D box {nb}?",
        ]
        b = filtered_agent_boxes[i]
        x, y, l, w, h, heading = b[:6][0], b[:6][1], b[3], b[4], b[5], b[6]
        qa.append((random.choice(templates),
                   f"The object is a {filtered_names[i]} location: ({x:.2f}, {y:.2f}), "
                   f"length: {l:.2f}, width: {w:.2f}, height: {h:.2f}, heading: {heading:.2f}."))
    return qa


# --------------------------------------------------------------------------------------
# Subjective (VLM) QA — only used when USE_VLM=1 (phase 2)
# --------------------------------------------------------------------------------------
def build_subjective_qa(vlm: "VLMClient", abs_img_path: str, cache_dir: Path, token: str):
    """Returns a list of (question, answer). Cached per (token, key) to .txt."""
    tasks = {
        "img_desc": (
            "Suppose you are driving, and I'm providing you with the image captured by the car's front, "
            "generate a description of the driving scene which includes the key factors for driving planning, "
            "including the positions and movements of vehicles and pedestrians; prevailing weather conditions; "
            "time of day; road conditions; and the status of traffic lights. The description should be concise."
        ),
        "traf_cong": (
            "Based on the provided forward-facing image, analyze the current traffic congestion level "
            "(heavily congested, moderately congested, or clear) and advise whether driving should be cautious or normal."
        ),
        "traf_light": (
            "Given the provided forward-facing image, identify if there is a traffic light that affects the car's behavior. "
            "Respond with a complete sentence such as 'The traffic light is red/green/yellow' or 'There is no traffic light visible'."
        ),
        "road_sign": (
            "Based on the provided forward-facing image, identify and describe the road markings and signs "
            "(traffic lines, road signs, pedestrian crossings, speed bumps) and advise the appropriate driving action "
            "in a concise single paragraph."
        ),
        "driving_influence": (
            "Based on the provided forward-facing image, identify the most influential object affecting the current "
            "driving situation, describe why it is influential, and advise the appropriate driving action in a concise paragraph."
        ),
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for key, question in tasks.items():
        cache_file = cache_dir / key / f"{token}.txt"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        if cache_file.is_file():
            answer = cache_file.read_text()
        else:
            answer = vlm.infer(question, abs_img_path)
            if answer:
                cache_file.write_text(answer)
        if not answer:
            continue
        if key in ("traf_light",) and "no traffic light" in answer.lower():
            continue
        out.append((question, answer))
    return out


# --------------------------------------------------------------------------------------
# Sharding
# --------------------------------------------------------------------------------------
def resolve_shard() -> Tuple[int, int]:
    if "SHARD_INDEX" in os.environ or "SHARD_COUNT" in os.environ:
        return int(os.environ.get("SHARD_INDEX", "0")), int(os.environ.get("SHARD_COUNT", "1"))
    return int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))


def prepend_image_token(qa_pair: dict) -> None:
    first = qa_pair["conversations"][0]
    # find first human turn (skip system)
    for turn in qa_pair["conversations"]:
        if turn["from"] == "human":
            first = turn
            break
    if "<image>" not in first["value"]:
        first["value"] = "<image>\n" + first["value"]


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed, workers=True)

    data_split = cfg.train_test_split.data_split
    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)

    image_root = Path(os.environ.get("SIMSCALE_IMAGE_ROOT", os.environ.get("OPENSCENE_DATA_ROOT", str(data_path.parent.parent))))
    out_dir = Path(os.environ.get(
        "SIMSCALE_QA_OUT_DIR",
        os.path.join(os.environ.get("NAVSIM_EXP_ROOT", str(image_root)), "simscale_vlm_qa", data_split),
    ))
    out_dir.mkdir(parents=True, exist_ok=True)

    emit = os.environ.get("SIMSCALE_EMIT", "both").lower()
    max_scenes = int(os.environ.get("SIMSCALE_MAX_SCENES", "0"))
    vru_keep_empty_prob = float(os.environ.get("VRU_KEEP_EMPTY_PROB", "0.1"))
    use_vlm = os.environ.get("USE_VLM", "0") == "1"
    resume = os.environ.get("RESUME", "0") == "1"
    shard_idx, shard_cnt = resolve_shard()

    logger.info(f"[simscale-gen] data_split={data_split} shard={shard_idx}/{shard_cnt}")
    logger.info(f"[simscale-gen] navsim_log_path={data_path}")
    logger.info(f"[simscale-gen] image_root={image_root}")
    logger.info(f"[simscale-gen] out_dir={out_dir} emit={emit} use_vlm={use_vlm}")

    # Shard by log file (each shard loads only its own logs).
    all_stems = sorted(p.stem for p in data_path.iterdir() if p.suffix == ".pkl")
    max_logs = int(os.environ.get("SIMSCALE_MAX_LOGS", "0"))
    if max_logs:
        all_stems = all_stems[:max_logs]
    shard_stems = all_stems[shard_idx::shard_cnt] if shard_cnt > 1 else all_stems
    logger.info(f"[simscale-gen] logs total={len(all_stems)} this_shard={len(shard_stems)}")
    if not shard_stems:
        logger.warning("[simscale-gen] no logs assigned to this shard; exiting")
        return

    scene_filter = SceneFilter(
        num_history_frames=cfg.train_test_split.scene_filter.num_history_frames,
        num_future_frames=cfg.train_test_split.scene_filter.num_future_frames,
        frame_interval=cfg.train_test_split.scene_filter.frame_interval,
        has_route=cfg.train_test_split.scene_filter.has_route,
        max_scenes=None,
        log_names=shard_stems,
        tokens=None,
    )
    sensor_config = SensorConfig(
        cam_f0=True, cam_l0=False, cam_l1=False, cam_l2=False,
        cam_r0=False, cam_r1=False, cam_r2=False, cam_b0=False, lidar_pc=False,
    )
    scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=scene_filter,
        sensor_config=sensor_config,
        load_image_path=True,
    )
    dataset = Dataset_For_Pipeline(
        scene_loader=scene_loader, feature_builders=[], target_builders=[],
        cache_path=None, force_cache_computation=False,
    )
    n = len(dataset)
    logger.info(f"[simscale-gen] scenes in shard={n}")

    vlm = VLMClient() if use_vlm else None
    vqa_cache_dir = Path(os.environ.get("SIMSCALE_VQA_CACHE_DIR", str(out_dir / "vqa_cache")))

    traj_path = out_dir / f"simscale_{data_split}_traj_shard{shard_idx}of{shard_cnt}.jsonl"
    qa_path = out_dir / f"simscale_{data_split}_qa_shard{shard_idx}of{shard_cnt}.jsonl"

    done_tokens = set()
    if resume:
        for p in (traj_path, qa_path):
            if p.is_file():
                with open(p) as f:
                    for line in f:
                        try:
                            done_tokens.add(json.loads(line).get("token"))
                        except Exception:
                            pass
        logger.info(f"[simscale-gen] resume: {len(done_tokens)} tokens already present")

    emit_traj = emit in ("both", "traj")
    emit_qa = emit in ("both", "qa")
    traj_f = open(traj_path, "a" if resume else "w", encoding="utf-8") if emit_traj else None
    qa_f = open(qa_path, "a" if resume else "w", encoding="utf-8") if emit_qa else None

    n_traj = n_qa = 0
    processed = 0
    for idx in range(n):
        if max_scenes and processed >= max_scenes:
            break
        (ego_statuses, cameras, future_trajectory, agent_states, agent_labels,
         agent_names, token, future_velocities, box_2d) = dataset[idx]

        if resume and token in done_tokens:
            continue
        processed += 1

        abs_img = str(cameras[-1].cam_f0.image)
        try:
            rel_img = os.path.relpath(abs_img, image_root)
        except ValueError:
            rel_img = abs_img

        # keep only valid agents (sorted nearest->farthest already)
        boxes, boxes2d, names, vels = [], [], [], []
        for j, lab in enumerate(agent_labels):
            if lab:
                boxes.append(np.array(agent_states[j]))
                boxes2d.append(box_2d[j])
                names.append(str(agent_names[j]))
                vels.append(np.array(future_velocities[j]))

        # ---- trajectory QA ----
        traj_pair, future_points, history_trajectory, command_str = build_trajectory_qa(
            ego_statuses, rel_img, future_trajectory, idx, token
        )
        if emit_traj:
            json.dump(traj_pair, traj_f, ensure_ascii=False)
            traj_f.write("\n")
            traj_f.flush()
            n_traj += 1

        if not emit_qa:
            continue

        # image dims for normalization / projection filtering (only needed when
        # there are agents to describe; lazy header read instead of full decode)
        if boxes:
            iw, ih = get_image_size(abs_img)
        else:
            iw, ih = (1920, 1080)

        conversations: List[dict] = []

        # planning
        plan_q, plan_a = build_plan_qa(future_points, history_trajectory, command_str)
        conversations += [{"from": "human", "value": plan_q}, {"from": "gpt", "value": plan_a}]

        # prediction (motion)
        mot_q, mot_a = build_mot_pred_qa(boxes, names, vels)
        if mot_q is not None:
            conversations += [{"from": "human", "value": mot_q}, {"from": "gpt", "value": mot_a}]

        # perception: VRU (subsample the frequent empty answers)
        vru_q, vru_a, vru_has = build_vru_qa(boxes, names)
        if vru_has or random.random() < vru_keep_empty_prob:
            conversations += [{"from": "human", "value": vru_q}, {"from": "gpt", "value": vru_a}]

        # perception: 3D detection
        det_q, det_a = build_3d_det_qa(boxes, names)
        if det_q is not None:
            conversations += [{"from": "human", "value": det_q}, {"from": "gpt", "value": det_a}]

        # perception: 3D info
        for q, a in build_3d_info_qa(boxes, names, boxes2d, iw, ih):
            conversations += [{"from": "human", "value": q}, {"from": "gpt", "value": a}]

        # perception: distance
        dis_q, dis_a = build_distance_qa(boxes, names, boxes2d, iw, ih)
        if dis_q is not None:
            conversations += [{"from": "human", "value": dis_q}, {"from": "gpt", "value": dis_a}]

        # optional subjective QA (phase 2, needs VLM)
        if vlm is not None:
            for q, a in build_subjective_qa(vlm, abs_img, vqa_cache_dir, token):
                conversations += [{"from": "human", "value": q}, {"from": "gpt", "value": a}]

        if not conversations:
            continue

        qa_pair = {"id": idx, "image": [rel_img], "token": token, "conversations": conversations}
        prepend_image_token(qa_pair)
        json.dump(qa_pair, qa_f, ensure_ascii=False)
        qa_f.write("\n")
        qa_f.flush()
        n_qa += 1

    if traj_f:
        traj_f.close()
    if qa_f:
        qa_f.close()
    logger.info(f"[simscale-gen] DONE shard={shard_idx}/{shard_cnt} processed={processed} traj={n_traj} qa={n_qa}")
    if emit_traj:
        logger.info(f"[simscale-gen] traj -> {traj_path}")
    if emit_qa:
        logger.info(f"[simscale-gen] qa   -> {qa_path}")


if __name__ == "__main__":
    main()
