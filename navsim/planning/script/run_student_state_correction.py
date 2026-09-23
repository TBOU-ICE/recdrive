"""collect / finetune / validate / opd for one teacher; see README_STATE_CORRECTION.md."""
from __future__ import annotations
import argparse
import json
import os
import random
import time
import traceback
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from navsim.agents.recogdrive.student_state_correction import (
    load_pt, digest, seed_for, read_indexes, assert_disjoint, raw_sample, inputs,
    build_planner, encode, rollout, teacher_prediction, errors, grad_norm,
    summary, CorrectionObjective, OPDObjective, PDMScoring,
)


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))


def write_tensorboard_scalars(writer, prefix, payload, step):
    """Recursively write numeric JSON metrics using stable slash-separated tags."""
    for key, value in payload.items():
        tag = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            write_tensorboard_scalars(writer, tag, value, step)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            writer.add_scalar(tag, value, step)


def seed_all(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup():
    world, rank = int(os.getenv("WORLD_SIZE", "1")), int(os.getenv("RANK", "0"))
    local = int(os.getenv("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    else:
        device = torch.device("cpu")
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    return device, rank, world


class Samples(Dataset):
    def __init__(self, rows, states=False):
        self.rows, self.states = rows, states

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        if not self.states:
            return raw_sample(row)
        state = load_pt(row["file"])
        # Schema 2 stores only generated student state. Frozen observation/GT
        # tensors stay in the canonical feature cache and are reloaded here.
        if state.get("schema") == 2:
            if state.get("token") != row["token"] or state.get("seed") != row["seed"]:
                raise ValueError(f"State file metadata mismatch: {row['file']}")
            sample = raw_sample(row)
            sample.update({
                "seed": state["seed"],
                "states": state["states"],
                "student_final": state["student_final"],
            })
            return sample
        # Read-only compatibility with schema-1 state files, which embedded the
        # full frozen feature/target tensors in every .pt file.
        return state


def collate(samples):
    # Avoid variable-length VLM padding changing encodings between stages.
    return samples


def state_manifest(cfg, split):
    path = Path(cfg["state_dir"]) / split / "manifest.json"
    manifest = json.loads(path.read_text())
    if manifest["schema"] not in (1, 2) or manifest["split"] != split:
        raise ValueError(f"Bad manifest: {path}")
    for row in manifest["samples"]:
        row["file"] = str(path.parent / row["file"])
    return manifest


def check_state_manifests(cfg, planner):
    train, val = state_manifest(cfg, "train"), state_manifest(cfg, "val")
    assert_disjoint(train["samples"], val["samples"])
    for m in (train, val):
        if m["collect"]["steps"] != cfg["collect"]["steps"]:
            raise ValueError("Cached steps differ from configured collection steps")
        if m["sampler"] != cfg["sampler"]:
            raise ValueError("State sampler differs from configured OPD sampler")
        if m["ddim_t"] != planner.ddim_t.cpu().tolist():
            raise ValueError("Teacher/student timestep schedule mismatch")
        if m["ddim_alphas"] != planner.ddim_alphas.cpu().tolist():
            raise ValueError("Teacher/student alpha schedule mismatch")
    if train["student_sha256"] != val["student_sha256"]:
        raise ValueError("Train and val states came from different students")
    return train, val


@torch.no_grad()
def collect(cfg, device, rank, world):
    if world != 1:
        raise ValueError("Collection is a deterministic single-process stage; launch with python")
    student = build_planner(cfg["collector_student"], device).eval().requires_grad_(False)
    train, val = read_indexes(cfg["train_indexes"]), read_indexes(cfg["val_indexes"])
    assert_disjoint(train, val)
    ckpt_hash = digest(cfg["collector_student"]["checkpoint"])
    steps = cfg["collect"]["steps"]
    if not steps or len(set(steps)) != len(steps) or any(s < 0 or s >= student.ddim_steps for s in steps):
        raise ValueError("Invalid collection step indices")
    for split, entries in (("train", train), ("val", val)):
        output = Path(cfg["state_dir"]) / split
        output.mkdir(parents=True, exist_ok=True)
        if any(output.iterdir()):
            raise FileExistsError(f"Refuse to mix state caches: choose an empty directory: {output}")
        # Seeded selection avoids chronological prefix bias when max_samples > 0.
        entries = sorted(entries, key=lambda r: seed_for(r["token"], cfg["seed"]))
        limit = cfg["collect"].get(f"max_{split}_samples", 0)
        if limit:
            entries = entries[:limit]
        manifest = {"schema": 2, "split": split, "student_sha256": ckpt_hash,
                    "student_spec": cfg["collector_student"], "sampler": cfg["sampler"], "collect": cfg["collect"],
                    "ddim_t": student.ddim_t.cpu().tolist(),
                    "ddim_alphas": student.ddim_alphas.cpu().tolist(), "samples": []}
        for n, row in enumerate(entries):
            sample = raw_sample(row)
            for repeat in range(cfg["collect"]["seeds_per_scene"]):
                sample_seed = seed_for(row["token"], cfg["seed"], repeat)
                seed_all(sample_seed)
                vl, action = inputs(sample, device)
                states, final, _ = rollout(student, vl, action, cfg["sampler"])
                state_payload = {
                    "schema": 2,
                    "token": row["token"],
                    "seed": sample_seed,
                    "states": [
                        {"step": s, "timestep": int(student.ddim_t[s]), "z": states[s][0].cpu()}
                        for s in steps
                    ],
                    "student_final": final[0].cpu(),
                }
                name = f"{n:08d}_{repeat}.pt"
                torch.save(state_payload, output / name)
                manifest["samples"].append({**row, "file": name, "repeat": repeat, "seed": sample_seed})
            if n % 100 == 0:
                print(f"collect {split}: {n + 1}/{len(entries)}", flush=True)
        write_json(output / "manifest.json", manifest)


@torch.no_grad()
def evaluate_teacher(teacher, samples, cfg, device, baseline=None, score=False, full=False, export=False):
    teacher.eval()
    if baseline is not None:
        baseline.eval()
    pdm = PDMScoring(cfg["metric_cache_path"]) if score else None
    rows = []
    for sample in samples:
        seed_all(sample["seed"] + 1)
        vl, action = inputs(sample, device)
        enc = encode(teacher, vl, action)
        enc_base = encode(baseline, vl, action) if baseline is not None else None
        for entry in sample["states"]:
            step, z = entry["step"], entry["z"].to(device).unsqueeze(0)
            if entry["timestep"] != int(teacher.ddim_t[step]):
                raise ValueError("Cached state has wrong timestep")
            x0, raw, eps = teacher_prediction(teacher, z, step, enc, action.goal)
            pred = teacher.denorm_odo(x0)
            before = teacher.denorm_odo(z)
            row = {"token": sample["token"], "seed": sample["seed"], "step": step,
                   **errors(pred, action.action),
                   "state_ade_m": errors(before, action.action)["ade_m"],
                   "correction_m": float((pred[..., :2] - before[..., :2]).norm(dim=-1).mean()),
                   "eps_abs_mean": float(eps.abs().mean()),
                   "clip_fraction": float((raw.abs() >= getattr(teacher, "denoised_clip_value", 1.0)).float().mean())}
            row["ade_gain_from_state_m"] = row["state_ade_m"] - row["ade_m"]
            # Fixed q(GT) counterfactual at the SAME timestep and same observation.
            idx = teacher.make_timesteps(1, step, device)
            alpha = teacher.extract(teacher.ddim_alphas, idx, z.shape)
            qgt = alpha.sqrt() * teacher.norm_odo(action.action) + (1-alpha).sqrt() * torch.randn_like(z)
            qx, _, _ = teacher_prediction(teacher, qgt, step, enc, action.goal)
            row["qgt_ade_m"] = errors(teacher.denorm_odo(qx), action.action)["ade_m"]
            if baseline is not None:
                bx, _, _ = teacher_prediction(baseline, z, step, enc_base, action.goal)
                bpred = baseline.denorm_odo(bx)
                row.update({f"baseline_{k}": v for k, v in errors(bpred, action.action).items()})
                bqx, _, _ = teacher_prediction(baseline, qgt, step, enc_base, action.goal)
                row["baseline_qgt_ade_m"] = errors(baseline.denorm_odo(bqx), action.action)["ade_m"]
                row["paired_ade_gain_m"] = row["baseline_ade_m"] - row["ade_m"]
                row["ade_improved"] = int(row["paired_ade_gain_m"] > 0)
            if pdm:
                # Strict failures: missing metric caches never become zero scores.
                row.update({f"x0_pdm_{k}": float(v) for k, v in pdm(sample["token"], pred[0]).items()})
                if baseline is not None:
                    row.update({f"baseline_x0_pdm_{k}": float(v) for k, v in pdm(sample["token"], bpred[0]).items()})
                    row["paired_pdms_gain"] = row["x0_pdm_score"] - row["baseline_x0_pdm_score"]
            if export:
                row["teacher_x0_trajectory"] = pred[0].cpu().tolist()
                row["student_state_trajectory"] = before[0].cpu().tolist()
                row["gt_trajectory"] = action.action[0].cpu().tolist()
                row["student_final_trajectory"] = sample["student_final"].tolist()
                if baseline is not None:
                    row["baseline_teacher_x0_trajectory"] = bpred[0].cpu().tolist()
            rows.append(row)
        if full:
            # Kept separate: original planning ability, not state correction.
            seed_all(sample["seed"])
            _, final, _ = rollout(teacher, vl, action, cfg.get("evaluation_sampler", cfg["sampler"]), privileged=True)
            row = {"token": sample["token"], "seed": sample["seed"], "step": -1,
                   **{f"full_{k}": v for k, v in errors(final, action.action).items()}}
            if pdm:
                row["full_pdms"] = float(pdm(sample["token"], final[0])["score"])
            if baseline is not None:
                seed_all(sample["seed"])
                _, bfinal, _ = rollout(baseline, vl, action, cfg.get("evaluation_sampler", cfg["sampler"]), privileged=True)
                row.update({f"baseline_full_{k}": v for k, v in errors(bfinal, action.action).items()})
                if pdm:
                    row["baseline_full_pdms"] = float(pdm(sample["token"], bfinal[0])["score"])
            rows.append(row)
    return {str(s): summary([r for r in rows if r["step"] == s]) for s in sorted({r["step"] for r in rows})}, rows


def save_model(path, planner, spec, cfg, epoch, global_step):
    state = {"agent.action_head." + k: v.detach().cpu() for k, v in planner.state_dict().items()}
    torch.save({"state_dict": state, "correction_metadata": {"spec": spec, "config": cfg,
               "epoch": epoch, "global_step": global_step}}, path)


@contextmanager
def validation_rng(device):
    """Evaluation seeds must not change any rank's subsequent training RNG."""
    import numpy as np
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[device.index] if device.type == "cuda" else []):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


@torch.no_grad()
def validate_epoch(planner, val_data, cfg, stage, device, rank, world, epoch,
                   chunk_size=32):
    """Exact unpadded validation, shared by all ranks in bounded chunks.

    Do not leave nonzero ranks in a NCCL barrier while rank 0 evaluates the
    entire dataset. Gather raw metric rows per chunk so quantiles and sample
    counts match single-process evaluation, including an uneven final shard.
    """
    count = min(cfg[stage].get("validation_samples", 0) or len(val_data), len(val_data))
    if count < 1 or chunk_size < 1:
        raise ValueError("Validation requires samples and a positive chunk_size")
    planner.eval()
    collected = []
    started = time.monotonic()
    if rank == 0:
        print(f"{stage} validation epoch={epoch} start samples={count} ranks={world}", flush=True)
    with validation_rng(device):
        for start in range(0, count, chunk_size * world):
            first = start + rank * chunk_size
            stop = min(first + chunk_size, count)
            payload = {"rows": [], "error": None}
            try:
                samples = (val_data[i] for i in range(first, stop))
                if stage == "finetune":
                    _, payload["rows"] = evaluate_teacher(planner, samples, cfg, device)
                else:
                    for sample in samples:
                        torch.manual_seed(seed_for(sample["token"], cfg["seed"]))
                        vl, action = inputs(sample, device)
                        _, pred, _ = rollout(planner, vl, action, cfg["sampler"])
                        payload["rows"].append(errors(pred, action.action))
            except Exception:
                # Report the actual failing shard on every rank instead of
                # letting its peers wait for a generic collective timeout.
                payload["error"] = f"rank={rank} samples=[{first},{stop})\n{traceback.format_exc()}"
            gathered = [None] * world
            if world > 1:
                dist.all_gather_object(gathered, payload)
            else:
                gathered[0] = payload
            failures = [item["error"] for item in gathered if item["error"]]
            if failures:
                raise RuntimeError("Epoch validation failed:\n" + "\n".join(failures))
            if rank == 0:
                for item in gathered:
                    collected.extend(item["rows"])
                done = min(start + chunk_size * world, count)
                print(f"{stage} validation epoch={epoch} samples={done}/{count} "
                      f"elapsed_s={time.monotonic() - started:.1f}", flush=True)
    if rank != 0:
        return None, None
    if stage == "finetune":
        report = {str(s): summary([row for row in collected if row["step"] == s])
                  for s in sorted({row["step"] for row in collected})}
        criterion = report[str(max(cfg["collect"]["steps"]))]["ade_m"]["mean"]
    else:
        report = summary(collected)
        criterion = report["ade_m"]["mean"]
    return report, criterion


def train(cfg, stage, device, rank, world, arm):
    if stage == "finetune":
        spec = cfg["teacher"]
        planner = build_planner(spec, device)
        train_m, val_m = check_state_manifests(cfg, planner)
        train_data, val_data = Samples(train_m["samples"], True), Samples(val_m["samples"], True)
        objective = CorrectionObjective(planner, cfg["finetune"])
    else:
        spec = cfg["opd_student"]
        if spec["kind"] != "goal_free":
            raise ValueError("This controlled OPD experiment requires goal_free student")
        planner = build_planner(spec, device)
        teacher_spec = dict(cfg["teacher"])
        if arm == "adapted":
            teacher_spec["checkpoint"] = cfg.get(
                "finetuned_teacher_checkpoint",
                str(Path(cfg["output_dir"]) / "finetune" / "best.ckpt"),
            )
        teacher = build_planner(teacher_spec, device)
        for key in ("ddim_t", "ddim_alphas", "ddim_alphas_prev"):
            if not torch.equal(getattr(planner, key), getattr(teacher, key)):
                raise ValueError(f"OPD schedules differ: {key}")
        tr, va = read_indexes(cfg["train_indexes"]), read_indexes(cfg["val_indexes"])
        assert_disjoint(tr, va)
        train_data, val_data = Samples(tr), Samples(va)
        objective = OPDObjective(planner, teacher, cfg["opd"], cfg["sampler"])
    options = cfg[stage]
    folder = Path(cfg["output_dir"]) / (stage if stage == "finetune" else f"opd_{arm}")
    if rank == 0:
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / "last.ckpt").exists():
            raise FileExistsError(f"Use a new output_dir; existing run: {folder}")
        write_json(folder / "config.json", cfg)
        provenance = {"initial_checkpoint_sha256": digest(spec["checkpoint"]), "arm": arm,
                      "world_size": world, "train_samples": len(train_data), "val_samples": len(val_data)}
        if stage == "opd":
            provenance["teacher_sha256"] = digest(teacher_spec["checkpoint"])
        else:
            provenance["train_manifest_sha256"] = digest(Path(cfg["state_dir"]) / "train/manifest.json")
            provenance["val_manifest_sha256"] = digest(Path(cfg["state_dir"]) / "val/manifest.json")
        write_json(folder / "provenance.json", provenance)
    if world > 1:
        dist.barrier()
    writer = None
    if rank == 0:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(folder / "tensorboard"))
    objective.to(device)
    model = DDP(objective, device_ids=[device.index] if device.type == "cuda" else None,
                find_unused_parameters=True) if world > 1 else objective
    sampler = DistributedSampler(train_data, shuffle=True, seed=cfg["seed"]) if world > 1 else None
    loader = DataLoader(train_data, batch_size=options["batch_size"], sampler=sampler,
                        shuffle=sampler is None, num_workers=options["workers"], collate_fn=collate)
    optimizer = torch.optim.AdamW([p for p in planner.parameters() if p.requires_grad],
                                 lr=options["lr"], weight_decay=options["weight_decay"])
    best, global_step = float("inf"), 0
    for epoch in range(options["epochs"]):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        for batch in loader:
            diagnostic = global_step % options["diagnostic_interval"] == 0
            optimizer.zero_grad(set_to_none=True)

            # Keep optional autograd.grad diagnostics separate from the DDP
            # update. This is not the cause of the epoch-end barrier timeout;
            # that is addressed by all-rank chunked validation below.
            diagnostic_rows = None
            if diagnostic and rank == 0:
                no_sync = model.no_sync() if world > 1 else nullcontext()
                with no_sync:
                    rng_devices = [device.index] if device.type == "cuda" else []
                    with torch.random.fork_rng(devices=rng_devices):
                        diagnostic_loss, diagnostic_rows = objective(batch, diagnostic=True)
                del diagnostic_loss

            loss, rows = model(batch, diagnostic=False)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            decoder_grad = grad_norm(planner.action_decoder.parameters())
            total_grad = torch.nn.utils.clip_grad_norm_(planner.parameters(), options["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            if diagnostic and rank == 0:
                metric_summary = summary(diagnostic_rows)
                row = {"epoch": epoch, "global_step": global_step, "loss": float(loss.detach()),
                       "decoder_total_grad_norm": decoder_grad, "total_grad_norm_before_clip": float(total_grad),
                       "lr": float(optimizer.param_groups[0]["lr"]),
                       "metrics": metric_summary,
                       "metrics_scope": (
                           "loss/metrics and branch-conflict gradients are rank0 local-batch; "
                           "total backward parameter gradients are DDP-averaged"
                       )}
                with open(folder / "train.jsonl", "a") as f:
                    f.write(json.dumps(row, allow_nan=False) + "\n")
                writer.add_scalar("train/total_loss", row["loss"], global_step)
                writer.add_scalar("train/decoder_total_grad_norm", decoder_grad, global_step)
                writer.add_scalar("train/total_grad_norm_before_clip", float(total_grad), global_step)
                writer.add_scalar("train/lr", row["lr"], global_step)
                write_tensorboard_scalars(writer, "train/metrics", metric_summary, global_step)
                writer.flush()
                print(f"{stage} {arm} epoch={epoch} step={global_step} loss={float(loss.detach()):.6g}", flush=True)
            global_step += 1
        # Do not insert a standalone NCCL barrier after the final DDP backward.
        # The final backward collective may still be queued on the CUDA stream;
        # on this cluster that barrier was the actual SeqNum=21037 timeout.
        # validate_epoch starts with the rank-symmetric all_gather_object and
        # therefore provides the required synchronization without a second
        # barrier collective.
        report, criterion = validate_epoch(planner, val_data, cfg, stage, device,
                                           rank, world, epoch)
        if rank == 0:
            write_json(folder / f"val_epoch_{epoch:03d}.json", report)
            write_tensorboard_scalars(writer, "validation", report, epoch)
            writer.flush()
            save_model(folder / "last.ckpt", planner, spec, cfg, epoch, global_step)
            if criterion < best:
                best = criterion
                save_model(folder / "best.ckpt", planner, spec, cfg, epoch, global_step)
            print(f"{stage} epoch={epoch} validation complete ade_m={criterion:.6g}; "
                  f"checkpoints saved in {folder}", flush=True)
        # The last validation all_gather already synchronized all ranks.  Rank
        # 0 may save checkpoints while other ranks prepare the next epoch; the
        # next DDP forward will synchronize parameters again.
    if writer is not None:
        writer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["collect", "finetune", "validate", "opd", "evaluate-student"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--arm", choices=["original", "adapted"], default="original")
    parser.add_argument("--checkpoint", help="Override checkpoint for validate or evaluate-student")
    parser.add_argument("--pdms", action="store_true", help="Score fixed-state x0; metric_cache_path required")
    parser.add_argument("--full-rollout", action="store_true", help="Also evaluate separate full teacher rollouts")
    parser.add_argument("--export-trajectories", action="store_true", help="Write physical-coordinate trajectories to per_sample.json")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if cfg["collect"]["seeds_per_scene"] < 1:
        raise ValueError("seeds_per_scene must be >= 1")
    for name in ("finetune", "opd"):
        opts = cfg[name]
        if any(opts[k] <= 0 for k in ("epochs", "batch_size", "lr", "grad_clip", "diagnostic_interval")):
            raise ValueError(f"Invalid positive training setting: {name}")
        if opts["workers"] < 0 or opts.get("validation_samples", 0) < 0:
            raise ValueError(f"Invalid loader/validation setting: {name}")
    if cfg["finetune"]["correction_weight"] <= 0 or cfg["finetune"]["retention_weight"] < 0:
        raise ValueError("Invalid correction/retention weights")
    if any(cfg["collect"].get(k, 0) < 0 for k in ("max_train_samples", "max_val_samples")):
        raise ValueError("Collection sample limits must be nonnegative")
    device, rank, world = setup()
    seed_all(cfg["seed"] + rank)
    if cfg["teacher"]["kind"] != "privileged":
        raise ValueError("Teacher must be privileged")
    if cfg["sampler"]["min_sigma"] < 0 or cfg["sampler"]["noise_clip"] <= 0:
        raise ValueError("Invalid sampler")
    if args.pdms and not cfg.get("metric_cache_path"):
        raise ValueError("--pdms requires metric_cache_path")
    if args.stage == "collect":
        collect(cfg, device, rank, world)
    elif args.stage == "validate":
        if world != 1:
            raise ValueError("Paired validation is single-process; use python")
        teacher_spec = dict(cfg["teacher"])
        teacher_spec["checkpoint"] = args.checkpoint or str(Path(cfg["output_dir"]) / "finetune/best.ckpt")
        teacher = build_planner(teacher_spec, device)
        baseline = build_planner(cfg["teacher"], device)
        _, val = check_state_manifests(cfg, teacher)
        if args.pdms and not cfg.get("metric_cache_path"):
            raise ValueError("--pdms requires metric_cache_path")
        report, rows = evaluate_teacher(teacher, Samples(val["samples"], True), cfg, device,
                                        baseline, args.pdms, args.full_rollout, args.export_trajectories)
        folder = Path(cfg["output_dir"]) / "paired_validation"
        write_json(folder / "summary.json", report)
        write_json(folder / "per_sample.json", rows)
        write_json(folder / "provenance.json", {"teacher_sha256": digest(teacher_spec["checkpoint"]),
                   "baseline_sha256": digest(cfg["teacher"]["checkpoint"]), "config": cfg,
                   "val_manifest_sha256": digest(Path(cfg["state_dir"]) / "val/manifest.json"),
                   "pdms_enabled": args.pdms, "full_rollout_enabled": args.full_rollout})
    elif args.stage == "evaluate-student":
        if world != 1:
            raise ValueError("Student evaluation is single-process; use python")
        spec = dict(cfg["opd_student"])
        spec["checkpoint"] = args.checkpoint or str(Path(cfg["output_dir"]) / f"opd_{args.arm}/best.ckpt")
        student = build_planner(spec, device).eval()
        pdm = PDMScoring(cfg["metric_cache_path"]) if args.pdms else None
        samples = Samples(read_indexes(cfg.get("evaluation_indexes") or cfg["val_indexes"]))
        rows = []
        with torch.no_grad():
            for sample in samples:
                seed_all(seed_for(sample["token"], cfg["seed"]))
                vl, action = inputs(sample, device)
                _, pred, _ = rollout(student, vl, action, cfg.get("evaluation_sampler", cfg["sampler"]))
                row = {"token": sample["token"], **errors(pred, action.action)}
                if pdm:
                    row.update({f"pdm_{k}": float(v) for k, v in pdm(sample["token"], pred[0]).items()})
                if args.export_trajectories:
                    row["trajectory"] = pred[0].cpu().tolist()
                rows.append(row)
        folder = Path(cfg["output_dir"]) / f"evaluate_student_{args.arm}"
        write_json(folder / "summary.json", summary(rows))
        write_json(folder / "per_sample.json", rows)
        write_json(folder / "provenance.json", {"checkpoint_sha256": digest(spec["checkpoint"]), "config": cfg})
    else:
        train(cfg, args.stage, device, rank, world, args.arm)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
