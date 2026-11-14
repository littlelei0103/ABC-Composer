#!/usr/bin/env python3
"""
Mandopop-Studio V3 — Phase 1 (Base Pre-Training) & Phase 2 (Golden CoT SFT) Trainer
===================================================================================
Supports both single-GPU and multi-GPU DDP (`torchrun`) training with:
  - Automatic 3D Musical Coordinate Tensor computation (`bar_ids, grid_ids, track_ids, chord_ids`)
  - Token-level loss masking for Phase 2 SFT (masks prompt before `<think>`, trains on `<think>...</think><score>...</score>`)
  - Cosine warmup LR scheduler, bfloat16 mixed precision, gradient clipping
  - HuggingFace PreTrainedModel saving (`save_pretrained` / `from_pretrained`)
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v3.model.modeling_mandopop_v3 import (
    MandopopV3Config,
    MandopopV3ForCausalLM,
)
from v3.tokenizer.v3_unified_tokenizer import MandopopV3Tokenizer


class MandopopV3SequenceDataset(Dataset):
    """Dataset of pre-tokenized V3 sequence dicts (`{"input_ids": Tensor, "labels": Tensor, ...}`)."""

    def __init__(
        self,
        samples: List[Dict],
        tokenizer: MandopopV3Tokenizer,
        max_seq_len: int = 1280,
        use_masked_labels: bool = False,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.use_masked_labels = use_masked_labels

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]
        input_ids = item["input_ids"][: self.max_seq_len].to(torch.long)
        if self.use_masked_labels and "labels" in item:
            labels = item["labels"][: self.max_seq_len].to(torch.long)
        else:
            labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "labels": labels,
            "length": int(input_ids.shape[0]),
        }


def make_collate_fn(tokenizer: MandopopV3Tokenizer, max_seq_len: int = 1280):
    pad_id = tokenizer.pad_id

    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = min(max(item["length"] for item in batch), max_seq_len)
        max_len = min(((max_len + 15) // 16) * 16, max_seq_len)

        bsz = len(batch)
        input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
        labels = torch.full((bsz, max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)

        for i, item in enumerate(batch):
            L = min(item["length"], max_len)
            input_ids[i, :L] = item["input_ids"][:L]
            labels[i, :L] = item["labels"][:L]
            attention_mask[i, :L] = 1

        bar_ids, grid_ids, track_ids, chord_ids = tokenizer.compute_3d_musical_coordinates(input_ids)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "bar_ids": bar_ids,
            "grid_ids": grid_ids,
            "track_ids": track_ids,
            "chord_ids": chord_ids,
        }

    return collate_fn


def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return rank, world_size, local_rank, True
    else:
        return 0, 1, 0, False


def cleanup_ddp(is_ddp: bool):
    if is_ddp and dist.is_initialized():
        dist.destroy_process_group()


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int, min_lr_ratio: float = 0.08):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_phase(
    phase_name: str,
    data_path: Path,
    output_dir: Path,
    init_checkpoint: Optional[Path],
    tokenizer: MandopopV3Tokenizer,
    epochs: int,
    batch_size_per_gpu: int,
    grad_accum_steps: int,
    lr: float,
    weight_decay: float,
    max_seq_len: int,
    use_masked_labels: bool,
    rank: int,
    world_size: int,
    local_rank: int,
    is_ddp: bool,
):
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if rank == 0:
        print(f"\n==================================================================")
        print(f"🚀 Starting Mandopop-Studio V3 [{phase_name}]")
        print(f"   Data Path      : {data_path}")
        print(f"   Output Dir     : {output_dir}")
        print(f"   Init Checkpoint: {init_checkpoint}")
        print(f"   World Size     : {world_size} GPUs | Batch/GPU: {batch_size_per_gpu} | Accum: {grad_accum_steps}")
        print(f"   Effective Batch: {world_size * batch_size_per_gpu * grad_accum_steps} sequences/step")
        print(f"   Epochs         : {epochs} | Peak LR: {lr} | Max Seq Len: {max_seq_len}")
        print(f"==================================================================", flush=True)

    samples: List[Dict] = torch.load(data_path, weights_only=False)
    rng = random.Random(42)
    rng.shuffle(samples)

    val_count = max(32, min(400, int(len(samples) * 0.03)))
    val_samples = samples[:val_count]
    train_samples = samples[val_count:]

    if rank == 0:
        print(f"   Loaded {len(samples):,} total sequences -> {len(train_samples):,} train / {len(val_samples):,} val", flush=True)

    train_dataset = MandopopV3SequenceDataset(
        train_samples,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        use_masked_labels=use_masked_labels,
    )
    val_dataset = MandopopV3SequenceDataset(
        val_samples,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        use_masked_labels=use_masked_labels,
    )

    collate_fn = make_collate_fn(tokenizer, max_seq_len=max_seq_len)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if is_ddp else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_per_gpu,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size_per_gpu,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    if init_checkpoint is not None and init_checkpoint.exists():
        if rank == 0:
            print(f"   Loading pretrained weights from {init_checkpoint} ...", flush=True)
        model = MandopopV3ForCausalLM.from_pretrained(str(init_checkpoint), torch_dtype=torch.bfloat16)
    else:
        cfg = MandopopV3Config(vocab_size=tokenizer.vocab_size)
        model = MandopopV3ForCausalLM(cfg)

    model.to(device=device, dtype=torch.bfloat16)

    if rank == 0:
        pinfo = model.count_parameters()
        print(f"   Model parameter count: {pinfo['total_params']:,} ({pinfo['total_millions']:.2f}M)", flush=True)

    if is_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    raw_model: MandopopV3ForCausalLM = model.module if is_ddp else model

    decay_params = []
    no_decay_params = []
    for name, param in raw_model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "norm" in name or "bias" in name or "embed" in name or "scale" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=torch.cuda.is_available(),
    )

    steps_per_epoch = max(1, len(train_loader) // grad_accum_steps)
    total_steps = steps_per_epoch * epochs
    warmup_steps = max(15, int(0.06 * total_steps))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    if rank == 0:
        print(f"   Steps/epoch: {steps_per_epoch} | Total optimizer steps: {total_steps} | Warmup: {warmup_steps}", flush=True)

    history = []
    global_step = 0
    best_val_loss = float("inf")
    start_time = time.time()

    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        running_loss = 0.0
        accum_count = 0
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            bar_ids = batch["bar_ids"].to(device, non_blocking=True)
            grid_ids = batch["grid_ids"].to(device, non_blocking=True)
            track_ids = batch["track_ids"].to(device, non_blocking=True)
            chord_ids = batch["chord_ids"].to(device, non_blocking=True)

            is_accum_step = ((batch_idx + 1) % grad_accum_steps != 0) and ((batch_idx + 1) != len(train_loader))

            sync_ctx = model.no_sync() if (is_ddp and is_accum_step) else torch.enable_grad()
            with sync_ctx:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=None,
                        bar_ids=bar_ids,
                        grid_ids=grid_ids,
                        track_ids=track_ids,
                        chord_ids=chord_ids,
                        use_cache=False,
                    )
                    # V3.5 Melody-First Hierarchical Loss Weighting:
                    # Emphasize Lead_Hook (`track_id == 1`, weight 1.60) and Piano_Arp (`track_id == 2`, weight 1.15)
                    # while downweighting rule-synthesized Strings (`track_id == 3`, weight 0.65) and Bass (`track_id == 4`, weight 0.55)
                    shift_logits = out.logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    shift_tracks = track_ids[..., 1:].contiguous()

                    per_tok_loss = torch.nn.functional.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100,
                        reduction="none",
                    ).view(shift_labels.shape)

                    valid_mask = (shift_labels != -100).float()
                    trk_weights = torch.ones_like(valid_mask)
                    trk_weights = torch.where(shift_tracks == 1, 1.60, trk_weights)
                    trk_weights = torch.where(shift_tracks == 2, 1.15, trk_weights)
                    trk_weights = torch.where(shift_tracks == 3, 0.65, trk_weights)
                    trk_weights = torch.where(shift_tracks == 4, 0.55, trk_weights)

                    weighted_mask = valid_mask * trk_weights
                    raw_loss = (per_tok_loss * weighted_mask).sum() / weighted_mask.sum().clamp(min=1.0)
                    loss = raw_loss / grad_accum_steps
                loss.backward()

            running_loss += raw_loss.detach().item()
            accum_count += 1

            if not is_accum_step:
                grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0).item()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                avg_step_loss = running_loss / max(1, accum_count)
                running_loss = 0.0
                accum_count = 0

                if rank == 0 and (global_step % 25 == 0 or global_step == 1 or global_step == total_steps):
                    elapsed = time.time() - start_time
                    cur_lr = scheduler.get_last_lr()[0]
                    ppl = math.exp(min(15.0, avg_step_loss))
                    print(
                        f"   [{phase_name}] Ep {epoch}/{epochs} | Step {global_step:4d}/{total_steps} | "
                        f"Loss: {avg_step_loss:.4f} (PPL {ppl:.2f}) | GradNorm: {grad_norm:.3f} | "
                        f"LR: {cur_lr:.2e} | Time: {elapsed:.1f}s",
                        flush=True,
                    )
                    history.append(
                        {
                            "step": global_step,
                            "epoch": epoch,
                            "train_loss": round(avg_step_loss, 4),
                            "ppl": round(ppl, 2),
                            "grad_norm": round(grad_norm, 4),
                            "lr": cur_lr,
                            "elapsed_sec": round(elapsed, 1),
                        }
                    )

        if rank == 0:
            raw_model.eval()
            val_losses = []
            with torch.no_grad():
                for vbatch in val_loader:
                    v_input_ids = vbatch["input_ids"].to(device, non_blocking=True)
                    v_labels = vbatch["labels"].to(device, non_blocking=True)
                    v_att = vbatch["attention_mask"].to(device, non_blocking=True)
                    v_bar_ids = vbatch["bar_ids"].to(device, non_blocking=True)
                    v_grid_ids = vbatch["grid_ids"].to(device, non_blocking=True)
                    v_track_ids = vbatch["track_ids"].to(device, non_blocking=True)
                    v_chord_ids = vbatch["chord_ids"].to(device, non_blocking=True)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        vout = raw_model(
                            input_ids=v_input_ids,
                            attention_mask=v_att,
                            labels=v_labels,
                            bar_ids=v_bar_ids,
                            grid_ids=v_grid_ids,
                            track_ids=v_track_ids,
                            chord_ids=v_chord_ids,
                            use_cache=False,
                        )
                    val_losses.append(vout.loss.item())
            val_loss = sum(val_losses) / max(1, len(val_losses))
            val_ppl = math.exp(min(15.0, val_loss))
            print(
                f"✨ [{phase_name}] Epoch {epoch}/{epochs} Validation -> Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}",
                flush=True,
            )
            history.append(
                {
                    "step": global_step,
                    "epoch": epoch,
                    "val_loss": round(val_loss, 4),
                    "val_ppl": round(val_ppl, 2),
                }
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                raw_model.save_pretrained(str(output_dir / "best_model"))

        if is_ddp:
            dist.barrier()

    if rank == 0:
        final_dir = output_dir / "final_model"
        raw_model.save_pretrained(str(final_dir))
        tokenizer.save_vocab(str(final_dir / "v3_vocab_4096.json"))
        with open(output_dir / "training_history.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "phase": phase_name,
                    "total_steps": global_step,
                    "epochs": epochs,
                    "best_val_loss": round(best_val_loss, 4),
                    "history": history,
                },
                f,
                indent=2,
            )
        print(f"✅ [{phase_name}] Finished! Saved final checkpoint to {final_dir}", flush=True)

    if is_ddp:
        dist.barrier()


def main():
    parser = argparse.ArgumentParser(description="Mandopop-Studio V3 Pretrain & SFT Trainer")
    parser.add_argument("--mode", type=str, choices=["pretrain", "sft", "both"], default="both")
    parser.add_argument("--data_dir", type=str, default=str(PROJECT_ROOT / "data/processed/v3_dataset"))
    parser.add_argument("--ckpt_root", type=str, default=str(PROJECT_ROOT / "checkpoints"))
    parser.add_argument("--pretrain_epochs", type=int, default=4)
    parser.add_argument("--sft_epochs", type=int, default=4)
    parser.add_argument("--batch_size_per_gpu", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=2)
    parser.add_argument("--pretrain_lr", type=float, default=4e-4)
    parser.add_argument("--sft_lr", type=float, default=1.5e-4)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    args = parser.parse_args()

    rank, world_size, local_rank, is_ddp = setup_ddp()
    tokenizer = MandopopV3Tokenizer()
    data_dir = Path(args.data_dir)
    ckpt_root = Path(args.ckpt_root)

    try:
        if args.mode in ("pretrain", "both"):
            train_phase(
                phase_name="Phase1_Pretrain",
                data_path=data_dir / "pretrain_samples.pt",
                output_dir=ckpt_root / "v3_pretrain",
                init_checkpoint=None,
                tokenizer=tokenizer,
                epochs=args.pretrain_epochs,
                batch_size_per_gpu=args.batch_size_per_gpu,
                grad_accum_steps=args.grad_accum_steps,
                lr=args.pretrain_lr,
                weight_decay=0.05,
                max_seq_len=args.max_seq_len,
                use_masked_labels=False,
                rank=rank,
                world_size=world_size,
                local_rank=local_rank,
                is_ddp=is_ddp,
            )

        if args.mode in ("sft", "both"):
            pretrain_ckpt = ckpt_root / "v3_pretrain" / "final_model"
            train_phase(
                phase_name="Phase2_Golden_CoT_SFT",
                data_path=data_dir / "sft_samples.pt",
                output_dir=ckpt_root / "v3_sft",
                init_checkpoint=pretrain_ckpt,
                tokenizer=tokenizer,
                epochs=args.sft_epochs,
                batch_size_per_gpu=args.batch_size_per_gpu,
                grad_accum_steps=args.grad_accum_steps,
                lr=args.sft_lr,
                weight_decay=0.02,
                max_seq_len=args.max_seq_len,
                use_masked_labels=True,
                rank=rank,
                world_size=world_size,
                local_rank=local_rank,
                is_ddp=is_ddp,
            )
    finally:
        cleanup_ddp(is_ddp)


if __name__ == "__main__":
    main()
