#!/usr/bin/env python3
"""Diagnose whether v1 Goal-OPD recovered the teacher's privileged policy shift.

For each cached scene s, decode three deterministic DDIM trajectories:

    tau_base ~ P_base(tau | s)          goal-free IL init
    tau_T    ~ P_T(tau | s, g)          routed goal-conditioned teacher
    tau_S    ~ P_S(tau | s)             goal-free OPD student

Distance D is ADE (mean waypoint xy L2). Deterministic DDIM is a near-delta,
so this is the 2-Wasserstein distance between those deltas. FDE is also logged.

    Delta_T   = D(P_base, P_T)
    Recovery  = 1 - D(P_S, P_T) / D(P_base, P_T)

Recovery=1 means the student sits on the teacher; 0 means it is still the base;
negative means it moved away from the teacher.

The script does not load the VLM. It reads precomputed internvl hidden states
from the same new-VLM cache the v4 / resume22 runs used.

Does not modify recdrive-multi-opd-v1-gpt. Import the training-code planners
by setting PYTHONPATH / NAVSIM_DEVKIT_ROOT to that tree (see the launcher).
"""

from __future__ import annotations

import argparse
import gzip
import inspect
import json
import math
import os
import pickle
import random
import signal
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from transformers.feature_extraction_utils import BatchFeature

BUCKETS = (
    "progress_curbside_stopgo",
    "rule_intersection",
    "safety_dynamics_interaction",
    "general_or_no_tag",
)
FALLBACK_BUCKET = "general_or_no_tag"
FEATURE_NAME = "internvl_feature.gz"
TARGET_NAME = "trajectory_target.gz"
LOW_SHIFT_M = 0.25


def _normalize_token(token: object) -> Optional[str]:
    if token is None:
        return None
    if isinstance(token, (bytes, bytearray)):
        token = token.hex()
    if not isinstance(token, str):
        return None
    token = token.strip().lower()
    base, sep, suffix = token.rpartition("-")
    if not (sep and suffix.isdigit() and len(suffix) == 3 and base):
        token = token.replace("-", "")
    return token or None


def _strip_action_head_prefixes(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    stripped = {}
    for key, value in state.items():
        if key.startswith("agent.action_head."):
            stripped[key[len("agent.action_head.") :]] = value
        elif key.startswith("action_head."):
            stripped[key[len("action_head.") :]] = value
        elif key.startswith("module."):
            stripped[key[len("module.") :]] = value
        else:
            stripped[key] = value
    return stripped


def _torch_load(path: str) -> Dict[str, Any]:
    load_kw: Dict[str, Any] = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kw["weights_only"] = False
    return torch.load(path, **load_kw)


def _load_planner_weights(planner: torch.nn.Module, ckpt_path: str, name: str) -> None:
    checkpoint = _torch_load(ckpt_path)
    state = checkpoint.get("state_dict", checkpoint)
    stripped = _strip_action_head_prefixes(state)
    missing, unexpected = planner.load_state_dict(stripped, strict=False)
    print(
        f"[policy-shift] loaded {name} from {ckpt_path} "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )
    planner.eval()
    for p in planner.parameters():
        p.requires_grad_(False)


def _build_goal_free_planner(device: torch.device, dtype: torch.dtype):
    from navsim.agents.recogdrive.recogdrive_agent import make_recogdrive_config
    from navsim.agents.recogdrive.recogdrive_diffusion_planner import ReCogDriveDiffusionPlanner

    cfg = make_recogdrive_config(
        "small",
        action_dim=3,
        action_horizon=8,
        grpo=False,
        input_embedding_dim=384,
        sampling_method="ddim",
    )
    cfg.vlm_size = "small"
    planner = ReCogDriveDiffusionPlanner(cfg).to(device=device, dtype=dtype)
    planner.eval()
    return planner


def _build_goal_teacher(device: torch.device, dtype: torch.dtype, goal_mode: str):
    from navsim.agents.recogdrive.recogdrive_agent import make_recogdrive_config
    from navsim.agents.recogdrive.recogdrive_goal_planner import GoalCondDiffusionPlanner

    cfg = make_recogdrive_config(
        "small",
        action_dim=3,
        action_horizon=8,
        grpo=False,
        input_embedding_dim=384,
        sampling_method="ddim",
    )
    cfg.vlm_size = "small"
    planner = GoalCondDiffusionPlanner(
        cfg,
        goal_mode=goal_mode,
        goal_sincos_dim=128,
        goal_hidden_dim=1024,
        goal_use_heading=False,
    ).to(device=device, dtype=dtype)
    planner.eval()
    return planner


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_bad_tokens(path: Optional[str]) -> set:
    if not path or not os.path.isfile(path):
        return set()
    bad = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tok = _normalize_token(Path(line).name if "/" in line else line)
            if tok:
                bad.add(tok)
            # also keep raw relative shard ids so we can skip by path suffix
            bad.add(line)
    return bad


def _read_gz(path: Path) -> Dict[str, torch.Tensor]:
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def _load_sample(token_dir: Path, timeout_s: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    def _on_timeout(signum, frame):
        raise TimeoutError(f"cache load exceeded {timeout_s}s: {token_dir}")

    previous = None
    if timeout_s > 0:
        try:
            previous = signal.signal(signal.SIGALRM, _on_timeout)
            signal.alarm(timeout_s)
        except (ValueError, OSError):
            previous = None
    try:
        features = _read_gz(token_dir / FEATURE_NAME)
        targets = _read_gz(token_dir / TARGET_NAME)
        return features, targets
    finally:
        if previous is not None:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)


def _ade_fde(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """a,b: (B, H, >=2) metres. Returns (B,) ADE and (B,) FDE on xy."""
    err = (a[..., :2] - b[..., :2]).norm(dim=-1)
    return err.mean(dim=-1), err[:, -1]


def _collate(samples: Sequence[Tuple[str, str, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]):
    tokens = [s[0] for s in samples]
    buckets = [s[1] for s in samples]
    hidden = torch.stack([s[2]["last_hidden_state"].float() for s in samples], dim=0)
    his = torch.stack([s[2]["history_trajectory"].float().reshape(-1) for s in samples], dim=0)
    status = torch.stack([s[2]["status_feature"].float() for s in samples], dim=0)
    gt = torch.stack([s[3]["trajectory"].float() for s in samples], dim=0)
    return tokens, buckets, hidden, his, status, gt


@torch.no_grad()
def _decode(planner, hidden, his, status, goal: Optional[torch.Tensor], deterministic: bool) -> torch.Tensor:
    data = {
        "his_traj": his,
        "status_feature": status,
        "state": torch.cat([status, his], dim=1),
    }
    if goal is not None:
        data["goal"] = goal
    out = planner.get_action(
        hidden,
        BatchFeature(data=data),
        deterministic=deterministic,
    )
    return out["pred_traj"].float()


def _mean(xs: Sequence[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


def _median(xs: Sequence[float]) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    mid = len(ys) // 2
    if len(ys) % 2:
        return float(ys[mid])
    return float(0.5 * (ys[mid - 1] + ys[mid]))


def _recovery(d_st: float, d_bt: float) -> float:
    if not math.isfinite(d_bt) or d_bt < 1e-8:
        return float("nan")
    return 1.0 - d_st / d_bt


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def _agg(subset: List[Dict[str, Any]]) -> Dict[str, Any]:
        d_bt = [r["ade_base_teacher"] for r in subset]
        d_st = [r["ade_student_teacher"] for r in subset]
        d_bs = [r["ade_base_student"] for r in subset]
        d_to = [r["ade_teacher_on_off"] for r in subset]
        recs = [
            r["recovery_ade"]
            for r in subset
            if math.isfinite(r["recovery_ade"]) and r["ade_base_teacher"] >= LOW_SHIFT_M
        ]
        fde_bt = [r["fde_base_teacher"] for r in subset]
        fde_st = [r["fde_student_teacher"] for r in subset]
        recs_fde = [
            r["recovery_fde"]
            for r in subset
            if math.isfinite(r["recovery_fde"]) and r["fde_base_teacher"] >= LOW_SHIFT_M
        ]
        mean_bt = _mean(d_bt)
        mean_st = _mean(d_st)
        return {
            "n": len(subset),
            "n_shift": len(recs),
            "delta_T_ade": mean_bt,
            "D_student_teacher_ade": mean_st,
            "D_base_student_ade": _mean(d_bs),
            "teacher_goal_effect_ade": _mean(d_to),
            "recovery_ade_from_means": _recovery(mean_st, mean_bt),
            "recovery_ade_median": _median(recs),
            "recovery_ade_frac_positive": (
                float(sum(1 for x in recs if x > 0.0) / len(recs)) if recs else float("nan")
            ),
            "delta_T_fde": _mean(fde_bt),
            "D_student_teacher_fde": _mean(fde_st),
            "recovery_fde_from_means": _recovery(_mean(fde_st), _mean(fde_bt)),
            "recovery_fde_median": _median(recs_fde),
            "ade_base_gt": _mean([r["ade_base_gt"] for r in subset]),
            "ade_teacher_gt": _mean([r["ade_teacher_gt"] for r in subset]),
            "ade_student_gt": _mean([r["ade_student_gt"] for r in subset]),
        }

    out = {"all": _agg(rows), "by_bucket": {}}
    for bucket in BUCKETS:
        sub = [r for r in rows if r["bucket"] == bucket]
        if sub:
            out["by_bucket"][bucket] = _agg(sub)
    return out


def _print_summary(summary: Dict[str, Any]) -> None:
    def _row(name: str, s: Dict[str, Any]) -> str:
        return (
            f"{name:32s}  n={s['n']:4d}  "
            f"ΔT={s['delta_T_ade']:.3f}  "
            f"D(S,T)={s['D_student_teacher_ade']:.3f}  "
            f"Rec={s['recovery_ade_from_means']:.3f}  "
            f"goal_fx={s['teacher_goal_effect_ade']:.3f}  "
            f"ADE(S,GT)={s['ade_student_gt']:.3f}"
        )

    print("\n[policy-shift] ADE metres; Recovery = 1 - D(S,T) / D(base,T)")
    print(_row("ALL", summary["all"]))
    for bucket, s in summary["by_bucket"].items():
        print(_row(bucket, s))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-path", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--token-to-bucket-json", required=True)
    p.add_argument("--base-ckpt", required=True, help="P_base: goal-free IL init")
    p.add_argument("--student-ckpt", required=True, help="P_S: OPD student")
    p.add_argument("--teacher-progress-ckpt", required=True)
    p.add_argument("--teacher-rule-ckpt", required=True)
    p.add_argument("--teacher-safety-ckpt", required=True)
    p.add_argument("--teacher-general-ckpt", required=True)
    p.add_argument("--teacher-goal-mode", default="adaln")
    p.add_argument("--student-adaln-bound", type=float, default=8.0)
    p.add_argument("--n-per-bucket", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float32", choices=("float32", "bfloat16", "float16"))
    p.add_argument("--bad-cache-list", default="")
    p.add_argument("--load-timeout-s", type=int, default=60)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--deterministic", action="store_true", default=True)
    p.add_argument("--stochastic", action="store_true", help="override: sample DDIM with noise")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    deterministic = not args.stochastic
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = getattr(torch, args.dtype)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = _load_json(args.manifest)
    if manifest.get("cache_path") != os.path.abspath(args.cache_path) and manifest.get("cache_path") != args.cache_path:
        print(
            f"[policy-shift] warning: manifest cache_path={manifest.get('cache_path')} "
            f"vs --cache-path={args.cache_path}"
        )
    token_to_rel: Dict[str, str] = manifest["tokens"]
    raw_buckets: Dict[str, str] = _load_json(args.token_to_bucket_json)
    bucket_of = {_normalize_token(k): v for k, v in raw_buckets.items() if _normalize_token(k)}
    bad = _load_bad_tokens(args.bad_cache_list)

    by_bucket: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for token, rel in token_to_rel.items():
        ntok = _normalize_token(token)
        if ntok is None or ntok in bad or rel in bad:
            continue
        bucket = bucket_of.get(ntok, FALLBACK_BUCKET)
        if bucket not in BUCKETS:
            bucket = FALLBACK_BUCKET
        by_bucket[bucket].append((token, rel))

    rng = random.Random(args.seed)
    selected: List[Tuple[str, str, str]] = []
    for bucket in BUCKETS:
        pool = by_bucket[bucket]
        rng.shuffle(pool)
        take = pool[: min(args.n_per_bucket, len(pool))]
        print(f"[policy-shift] {bucket}: pool={len(pool)} take={len(take)}")
        selected.extend((tok, rel, bucket) for tok, rel in take)
    rng.shuffle(selected)
    if not selected:
        raise RuntimeError("no tokens selected; check manifest / token-to-bucket overlap")

    print(f"[policy-shift] device={device} dtype={dtype} n={len(selected)} deterministic={deterministic}")
    print("[policy-shift] building planners (no VLM)")
    base = _build_goal_free_planner(device, dtype)
    student = _build_goal_free_planner(device, dtype)
    _load_planner_weights(base, args.base_ckpt, "P_base")
    _load_planner_weights(student, args.student_ckpt, "P_S")
    if hasattr(student, "model") and hasattr(student.model, "set_adaln_bound"):
        student.model.set_adaln_bound(args.student_adaln_bound)
        print(f"[policy-shift] student adaLN bound={args.student_adaln_bound}")

    teacher_ckpts = {
        "progress_curbside_stopgo": args.teacher_progress_ckpt,
        "rule_intersection": args.teacher_rule_ckpt,
        "safety_dynamics_interaction": args.teacher_safety_ckpt,
        "general_or_no_tag": args.teacher_general_ckpt,
    }
    teachers = {}
    for name, ckpt in teacher_ckpts.items():
        planner = _build_goal_teacher(device, dtype, args.teacher_goal_mode)
        _load_planner_weights(planner, ckpt, f"P_T[{name}]")
        teachers[name] = planner

    cache_root = Path(args.cache_path)
    rows: List[Dict[str, Any]] = []
    skipped = 0
    i = 0
    while i < len(selected):
        batch_meta = selected[i : i + args.batch_size]
        loaded = []
        for token, rel, bucket in batch_meta:
            token_dir = cache_root / rel
            try:
                features, targets = _load_sample(token_dir, args.load_timeout_s)
                if "last_hidden_state" not in features or "trajectory" not in targets:
                    raise KeyError(f"missing keys in {token_dir}")
                loaded.append((token, bucket, features, targets))
            except Exception as exc:
                skipped += 1
                print(f"[policy-shift] skip {token}: {exc}")
        i += args.batch_size
        if not loaded:
            continue

        tokens, buckets, hidden, his, status, gt = _collate(loaded)
        hidden = hidden.to(device=device, dtype=dtype)
        his = his.to(device=device, dtype=dtype)
        status = status.to(device=device, dtype=dtype)
        gt = gt.to(device=device, dtype=torch.float32)
        goal = gt[:, -1, :].contiguous()

        tau_base = _decode(base, hidden, his, status, goal=None, deterministic=deterministic)
        tau_s = _decode(student, hidden, his, status, goal=None, deterministic=deterministic)

        tau_t = torch.zeros_like(tau_base)
        tau_t_off = torch.zeros_like(tau_base)
        for bucket in set(buckets):
            idx = [j for j, b in enumerate(buckets) if b == bucket]
            sel = torch.as_tensor(idx, device=device)
            teacher = teachers[bucket]
            tau_t[sel] = _decode(
                teacher, hidden[sel], his[sel], status[sel], goal=goal[sel], deterministic=deterministic
            )
            tau_t_off[sel] = _decode(
                teacher, hidden[sel], his[sel], status[sel], goal=None, deterministic=deterministic
            )

        ade_bt, fde_bt = _ade_fde(tau_base, tau_t)
        ade_st, fde_st = _ade_fde(tau_s, tau_t)
        ade_bs, fde_bs = _ade_fde(tau_base, tau_s)
        ade_to, fde_to = _ade_fde(tau_t, tau_t_off)
        ade_bg, fde_bg = _ade_fde(tau_base, gt)
        ade_tg, fde_tg = _ade_fde(tau_t, gt)
        ade_sg, fde_sg = _ade_fde(tau_s, gt)

        for j, token in enumerate(tokens):
            d_bt = float(ade_bt[j])
            d_st = float(ade_st[j])
            d_bt_f = float(fde_bt[j])
            d_st_f = float(fde_st[j])
            rows.append(
                {
                    "token": token,
                    "bucket": buckets[j],
                    "ade_base_teacher": d_bt,
                    "fde_base_teacher": d_bt_f,
                    "ade_student_teacher": d_st,
                    "fde_student_teacher": d_st_f,
                    "ade_base_student": float(ade_bs[j]),
                    "fde_base_student": float(fde_bs[j]),
                    "ade_teacher_on_off": float(ade_to[j]),
                    "fde_teacher_on_off": float(fde_to[j]),
                    "recovery_ade": _recovery(d_st, d_bt),
                    "recovery_fde": _recovery(d_st_f, d_bt_f),
                    "ade_base_gt": float(ade_bg[j]),
                    "ade_teacher_gt": float(ade_tg[j]),
                    "ade_student_gt": float(ade_sg[j]),
                    "fde_base_gt": float(fde_bg[j]),
                    "fde_teacher_gt": float(fde_tg[j]),
                    "fde_student_gt": float(fde_sg[j]),
                }
            )
        print(f"[policy-shift] decoded {len(rows)}/{len(selected)}  skipped={skipped}")

    if not rows:
        raise RuntimeError("no successful samples")

    summary = _summarize(rows)
    summary["config"] = {
        "base_ckpt": args.base_ckpt,
        "student_ckpt": args.student_ckpt,
        "n_per_bucket": args.n_per_bucket,
        "n_ok": len(rows),
        "n_skipped": skipped,
        "deterministic": deterministic,
        "distance": "ade_xy_metres (deterministic DDIM ~ W2 between deltas)",
        "low_shift_m": LOW_SHIFT_M,
    }
    csv_path = out_dir / "per_sample.csv"
    json_path = out_dir / "summary.json"
    keys = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _print_summary(summary)
    print(f"[policy-shift] wrote {json_path}")
    print(f"[policy-shift] wrote {csv_path}")


if __name__ == "__main__":
    main()
