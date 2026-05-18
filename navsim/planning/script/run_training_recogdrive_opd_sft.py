import argparse
import ast
import csv
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter
from functools import partial

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset

from navsim.agents.recogdrive.recogdrive_backbone import RecogDriveBackbone
from navsim.agents.recogdrive.recogdrive_opd_trainer import ReCogDriveOPDTrainer
from navsim.agents.recogdrive.utils.internvl_preprocess import load_image


def _safe_json_loads(line: str):
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return ast.literal_eval(line)


def _normalize_image_placeholder(question: str) -> str:
    """Keep a single <image> placeholder that maps to all packed visual patches."""
    if '<image>' not in question:
        return '<image>\n' + question
    parts = question.split('<image>')
    # keep only the first marker; drop trailing duplicate markers
    return parts[0] + '<image>' + ''.join(parts[1:])


class SFTMetaDataset(Dataset):
    def __init__(self, meta_path: str, max_samples: int = 0):
        # sample format: (image_paths, question), where image_paths is List[str]
        self.samples: List[Tuple[List[str], str]] = []
        meta = json.loads(Path(meta_path).read_text(encoding='utf-8'))
        self.dataset_stats: Dict[str, Dict[str, int]] = {}
        mixed_pool: List[Tuple[List[str], str]] = []
        mixed_pool_sources: List[str] = []

        for ds_name, cfg in meta.items():
            root = cfg.get('root', '')
            ann = cfg.get('annotation')
            repeat_time = int(cfg.get('repeat_time', 1))
            if repeat_time < 1:
                repeat_time = 1
            if not ann:
                continue
            ann_path = Path(ann)
            if not ann_path.exists():
                continue

            ds_samples: List[Tuple[List[str], str]] = []

            with ann_path.open('r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = _safe_json_loads(line)
                    except Exception:
                        continue

                    image_field = row.get('image')
                    if image_field is None:
                        continue
                    if isinstance(image_field, list):
                        image_rels = [p for p in image_field if isinstance(p, str) and len(p) > 0]
                    elif isinstance(image_field, str):
                        image_rels = [image_field]
                    else:
                        image_rels = []
                    if len(image_rels) == 0:
                        continue

                    conv = row.get('conversations') or row.get('messages')
                    if not isinstance(conv, list):
                        continue

                    question = None
                    for turn in conv:
                        if not isinstance(turn, dict):
                            continue
                        role = str(turn.get('from', turn.get('role', ''))).lower()
                        if role in {'human', 'user'}:
                            question = turn.get('value', turn.get('content', turn.get('text', '')))
                            break
                    if not isinstance(question, str) or len(question.strip()) == 0:
                        continue
                    question = _normalize_image_placeholder(question)

                    image_paths: List[str] = []
                    ok = True
                    for rel in image_rels:
                        if rel.startswith('s3://'):
                            ok = False
                            break
                        p = os.path.join(root, rel)
                        if not os.path.exists(p):
                            ok = False
                            break
                        image_paths.append(p)
                    if not ok or len(image_paths) == 0:
                        continue

                    ds_samples.append((image_paths, question))

            if len(ds_samples) == 0:
                continue

            raw_count = len(ds_samples)

            # Mixed sampling approximation by repeat_time: replicate dataset slice.
            for _ in range(repeat_time):
                mixed_pool.extend(ds_samples)
                mixed_pool_sources.extend([ds_name] * len(ds_samples))

            self.dataset_stats[ds_name] = {
                'raw': raw_count,
                'repeat_time': repeat_time,
                'effective': raw_count * repeat_time,
            }

        # Global shuffle then optional truncation keeps per-dataset mixing ratio
        # aligned with repeat_time, instead of introducing dataset-order bias.
        perm = list(range(len(mixed_pool)))
        random.shuffle(perm)
        mixed_pool = [mixed_pool[i] for i in perm]
        mixed_pool_sources = [mixed_pool_sources[i] for i in perm]
        if max_samples > 0:
            self.samples = mixed_pool[:max_samples]
            selected_sources = mixed_pool_sources[:max_samples]
        else:
            self.samples = mixed_pool
            selected_sources = mixed_pool_sources

        source_counter = Counter(selected_sources)
        total_selected = len(selected_sources)
        for ds_name, st in self.dataset_stats.items():
            selected = int(source_counter.get(ds_name, 0))
            st['selected'] = selected
            st['selected_ratio'] = (selected / total_selected) if total_selected > 0 else 0.0

        if len(self.samples) == 0:
            raise RuntimeError(f'No valid SFT OPD samples found from meta_path={meta_path}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch, image_max_num: int = 12):
    image_paths_list, questions = zip(*batch)
    pixel_values_list = []
    num_patches_list = []
    merged_image_paths = []
    for image_paths in image_paths_list:
        per_sample_tensors = [load_image(p, max_num=image_max_num) for p in image_paths]
        merged_image_paths.append(image_paths)
        sample_tensor = torch.cat(per_sample_tensors, dim=0)
        pixel_values_list.append(sample_tensor)
        num_patches_list.append(sample_tensor.shape[0])
    pixel_values = torch.cat(pixel_values_list, dim=0)
    return {
        'pixel_values': pixel_values,
        'num_patches_list': num_patches_list,
        'questions': list(questions),
        'image_paths': list(merged_image_paths),
    }


def setup_dist():
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    world_size = int(os.getenv('WORLD_SIZE', '1'))
    rank = int(os.getenv('RANK', '0'))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)
    torch.cuda.set_device(local_rank)
    return local_rank, world_size, rank


def maybe_all_reduce_mean(x: torch.Tensor, world_size: int) -> float:
    if world_size > 1:
        y = x.detach().clone()
        dist.all_reduce(y, op=dist.ReduceOp.SUM)
        y = y / world_size
        return float(y.item())
    return float(x.item())


class SFTOPDMetricsLogger:
    """Persist OPD training metrics and sample outputs under output_dir (rank 0 only)."""

    METRIC_FIELDS = (
        'step',
        'loss',
        'policy_entropy',
        'grad_norm',
        'response_length_mean',
        'lr',
    )

    def __init__(self, output_dir: str, rank: int):
        self.enabled = rank == 0
        if not self.enabled:
            return
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.metrics_csv = out / 'metrics.csv'
        self.training_log = out / 'training_log.txt'
        self.sample_log = out / 'sample_outputs.log'
        self._csv_initialized = self.metrics_csv.is_file() and self.metrics_csv.stat().st_size > 0
        with self.training_log.open('a', encoding='utf-8') as f:
            f.write(f'\n=== SFT-OPD run started {datetime.now().isoformat()} ===\n')

    def log_step_metrics(
        self,
        step: int,
        loss: float,
        policy_entropy: float,
        grad_norm: float,
        response_length_mean: float,
        lr: float,
    ) -> None:
        if not self.enabled:
            return
        row = {
            'step': step,
            'loss': loss,
            'policy_entropy': policy_entropy,
            'grad_norm': grad_norm,
            'response_length_mean': response_length_mean,
            'lr': lr,
        }
        with self.metrics_csv.open('a', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.METRIC_FIELDS)
            if not self._csv_initialized:
                writer.writeheader()
                self._csv_initialized = True
            writer.writerow(row)

    def log_training_line(
        self,
        step: int,
        loss: float,
        policy_entropy: float,
        grad_norm: float,
        response_length_mean: float,
        lr: float,
    ) -> None:
        if not self.enabled:
            return
        line = (
            f'[train] step={step} loss={loss:.6f} '
            f'Policy Entropy={policy_entropy:.6f} '
            f'Gradient Norm={grad_norm:.6f} '
            f'Response Length(mean)={response_length_mean:.2f} '
            f'lr={lr:.3e}\n'
        )
        with self.training_log.open('a', encoding='utf-8') as f:
            f.write(line)

    def log_sample(self, step: int, sample_text: str) -> None:
        if not self.enabled:
            return
        with self.sample_log.open('a', encoding='utf-8') as f:
            f.write(f'[sample][step={step}] {sample_text}\n')


def decode_first_completion(
    student: RecogDriveBackbone,
    prompt_input_ids: torch.Tensor,
    generated_ids: torch.Tensor,
    max_chars: int = 300,
) -> str:
    prompt_len = prompt_input_ids.shape[1]
    sample_ids = generated_ids[0]
    if (
        generated_ids.shape[1] >= prompt_len
        and torch.equal(generated_ids[0, :prompt_len], prompt_input_ids[0])
    ):
        sample_ids = generated_ids[0, prompt_len:]
    sample_text = student.tokenizer.decode(sample_ids, skip_special_tokens=True).replace('\n', ' ').strip()
    if len(sample_text) > max_chars:
        sample_text = sample_text[:max_chars] + ' ...'
    return sample_text


def main():
    parser = argparse.ArgumentParser(description='SFT JSONL OPD training for ReCogDrive VLM')
    parser.add_argument('--student_model_path', required=True)
    parser.add_argument('--teacher_model_path', required=True)
    parser.add_argument('--meta_path', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=2e-6)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--max_steps', type=int, default=20000)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--opd_topk', type=int, default=32)
    parser.add_argument('--opd_group_size', type=int, default=4)
    parser.add_argument('--opd_max_new_tokens', type=int, default=320)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--log_every',
        type=int,
        default=20,
        help='Print metrics to stdout and append training_log.txt every N steps.',
    )
    parser.add_argument(
        '--print_every',
        type=int,
        default=100,
        help='Decode and log model completion to stdout/sample_outputs.log every N steps.',
    )
    parser.add_argument(
        '--save_every',
        type=int,
        default=1000,
        help='Save student checkpoint step{N:08d}.pt every N steps (rank 0).',
    )
    parser.add_argument(
        '--metrics_log_every',
        type=int,
        default=1,
        help='Append Policy Entropy / Gradient Norm / Response Length to metrics.csv every N steps.',
    )
    parser.add_argument(
        '--image_max_num',
        type=int,
        default=4,
        help='Max dynamic tiles per image in InternVL preprocessing. Lower value reduces visual tokens and GPU memory.',
    )
    parser.add_argument('--max_samples', type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    local_rank, world_size, rank = setup_dist()
    device = torch.device(f'cuda:{local_rank}')

    student = RecogDriveBackbone('internvl', args.student_model_path, device=f'cuda:{local_rank}')
    teacher = RecogDriveBackbone('internvl', args.teacher_model_path, device=f'cuda:{local_rank}')
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    student.train()
    for name, p in student.model.named_parameters():
        if 'vision_model' in name or 'mlp1' in name:
            p.requires_grad = False
        else:
            p.requires_grad = True

    trainable = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))

    def lr_lambda(step):
        if step < args.warmup_steps:
            return float(step + 1) / float(max(1, args.warmup_steps))
        progress = (step - args.warmup_steps) / float(max(1, args.max_steps - args.warmup_steps))
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535))).item()

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    opd_trainer = ReCogDriveOPDTrainer(topk=args.opd_topk)

    dataset = SFTMetaDataset(meta_path=args.meta_path, max_samples=args.max_samples)
    sampler = None
    if world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=partial(collate_fn, image_max_num=args.image_max_num),
    )

    if len(loader) == 0:
        raise RuntimeError(
            'Dataloader is empty. This usually happens when dataset is too small under '
            'DDP with drop_last=True. Increase data/max_samples or reduce world_size.'
        )

    metrics_logger = SFTOPDMetricsLogger(args.output_dir, rank)

    if rank == 0:
        print(f'[SFT-OPD] dataset_size={len(dataset)} world_size={world_size}')
        print(
            f'[SFT-OPD] output_dir={args.output_dir} '
            f'metrics_csv={args.output_dir}/metrics.csv '
            f'training_log={args.output_dir}/training_log.txt '
            f'sample_log={args.output_dir}/sample_outputs.log'
        )
        print(
            f'[SFT-OPD] log_every={args.log_every} metrics_log_every={args.metrics_log_every} '
            f'print_every={args.print_every} save_every={args.save_every} image_max_num={args.image_max_num}'
        )
        if hasattr(dataset, 'dataset_stats'):
            for ds_name, st in dataset.dataset_stats.items():
                print(
                    f"[SFT-OPD][mix] {ds_name}: raw={st['raw']} repeat_time={st['repeat_time']} "
                    f"effective={st['effective']} selected={st.get('selected', 0)} "
                    f"selected_ratio={st.get('selected_ratio', 0.0):.4f}"
                )

    step = 0
    epoch = 0

    while step < args.max_steps:
        epoch += 1
        if sampler is not None:
            sampler.set_epoch(epoch)

        for batch in loader:
            if step >= args.max_steps:
                break

            step += 1
            pixel_values = batch['pixel_values'].to(device, non_blocking=True)
            questions = batch['questions']
            num_patches_list = batch['num_patches_list']
            G = args.opd_group_size

            pv_chunks = list(torch.split(pixel_values, num_patches_list, dim=0))
            pv_expanded = torch.cat([chunk for chunk in pv_chunks for _ in range(G)], dim=0)
            questions_expanded = [q for q in questions for _ in range(G)]
            num_patches_expanded = [n for n in num_patches_list for _ in range(G)]

            prompt_input_ids, prompt_attention_mask, _, _, _ = student._build_model_inputs(
                pv_expanded, questions_expanded, num_patches_expanded
            )

            with torch.no_grad():
                vit_embeds = student.model.extract_feature(pv_expanded.to(torch.bfloat16))
                input_embeds = student.model.language_model.get_input_embeddings()(prompt_input_ids)
                BG, N, C = input_embeds.shape
                embeds_flat = input_embeds.reshape(BG * N, C)
                ids_flat = prompt_input_ids.reshape(BG * N)
                selected = (ids_flat == student.img_context_token_id)
                n_img_slots = selected.sum().item()
                n_vit_tokens = vit_embeds.reshape(-1, C).shape[0]
                if n_img_slots != n_vit_tokens:
                    if rank == 0:
                        print(
                            f'[warn] skip step={step} due to image token mismatch: '
                            f'img_slots={n_img_slots} vit_tokens={n_vit_tokens}. '
                            f'Try increasing max_length or reducing image_max_num.',
                            flush=True,
                        )
                    step -= 1
                    continue
                embeds_flat[selected] = vit_embeds.reshape(-1, C).to(device=embeds_flat.device, dtype=embeds_flat.dtype)
                input_embeds = embeds_flat.reshape(BG, N, C)

                generated_ids = student.model.language_model.generate(
                    inputs_embeds=input_embeds,
                    attention_mask=prompt_attention_mask,
                    max_new_tokens=args.opd_max_new_tokens,
                    do_sample=True,
                    top_p=0.9,
                    temperature=1.0,
                    pad_token_id=student.tokenizer.pad_token_id,
                    eos_token_id=student.tokenizer.eos_token_id,
                )

            _, student_logits, response_mask = student.forward_with_logits(
                pixel_values=pv_expanded,
                questions=questions_expanded,
                num_patches_list=num_patches_expanded,
                generated_input_ids=generated_ids,
            )

            with torch.no_grad():
                _, teacher_logits, _ = teacher.forward_with_logits(
                    pixel_values=pv_expanded,
                    questions=questions_expanded,
                    num_patches_list=num_patches_expanded,
                    generated_input_ids=generated_ids,
                )

            out = opd_trainer.compute_loss(
                student_logits=student_logits.float(),
                teacher_logits=teacher_logits.float(),
                response_mask=response_mask.float(),
            )
            loss = out['loss']

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # Synchronize gradients across GPUs. forward_with_logits is called directly
            # rather than through nn.Module.__call__, so DDP hooks don't fire; we
            # all-reduce and average manually instead.
            if world_size > 1:
                for p in trainable:
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                        p.grad.div_(world_size)

            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)

            optimizer.step()
            scheduler.step()

            loss_m = maybe_all_reduce_mean(loss, world_size)
            ent_m = maybe_all_reduce_mean(out['policy_entropy'], world_size)
            rlen_m = maybe_all_reduce_mean(out['response_length_mean'], world_size)
            gnorm_m = maybe_all_reduce_mean(grad_norm, world_size)
            lr_m = scheduler.get_last_lr()[0]

            if step % args.metrics_log_every == 0:
                metrics_logger.log_step_metrics(
                    step=step,
                    loss=loss_m,
                    policy_entropy=ent_m,
                    grad_norm=gnorm_m,
                    response_length_mean=rlen_m,
                    lr=lr_m,
                )

            if step % args.log_every == 0:
                metrics_logger.log_training_line(
                    step=step,
                    loss=loss_m,
                    policy_entropy=ent_m,
                    grad_norm=gnorm_m,
                    response_length_mean=rlen_m,
                    lr=lr_m,
                )
                if rank == 0:
                    print(
                        f'[train] step={step} loss={loss_m:.6f} '
                        f'Policy Entropy={ent_m:.6f} '
                        f'Gradient Norm={gnorm_m:.6f} '
                        f'Response Length(mean)={rlen_m:.2f} '
                        f'lr={lr_m:.3e}',
                        flush=True,
                    )

            if step % args.print_every == 0 and rank == 0:
                sample_text = decode_first_completion(student, prompt_input_ids, generated_ids)
                print(f'[sample][step={step}] {sample_text}', flush=True)
                metrics_logger.log_sample(step, sample_text)

            if step % args.save_every == 0 and rank == 0:
                ckpt_path = os.path.join(args.output_dir, f'step{step:08d}.pt')
                torch.save(
                    {
                        'step': step,
                        'student_state_dict': student.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'args': vars(args),
                    },
                    ckpt_path,
                )
                print(f'[ckpt] saved: {ckpt_path}')

    if rank == 0:
        final_path = os.path.join(args.output_dir, 'final.pt')
        torch.save({'step': step, 'student_state_dict': student.state_dict(), 'args': vars(args)}, final_path)
        print(f'[done] saved final checkpoint: {final_path}')


if __name__ == '__main__':
    main()
