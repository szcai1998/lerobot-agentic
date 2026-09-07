#!/usr/bin/env python3
"""
scripts/train_policy.py
Production training script for Action Chunking with Transformers (ACT).
Trains ACT-B (unconditioned baseline) or ACT-G (goal-conditioned policy)
on nominal demonstration datasets targeting the NVIDIA RTX 3070 8GB VRAM envelope.
"""

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Train LeRobot ACT Policy (ACT-B or ACT-G).")
    parser.add_argument(
        "--policy-type",
        type=str,
        required=True,
        choices=["act-b", "act-g"],
        help="Policy variant: 'act-b' (unconditioned baseline) or 'act-g' (goal-conditioned).",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="data/nominal_train_v1",
        help="Path to training LeRobotDataset v3.0 directory.",
    )
    parser.add_argument(
        "--val-dataset-dir",
        type=str,
        default="data/nominal_val_v1",
        help="Path to validation LeRobotDataset v3.0 directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for checkpoints and telemetry.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10000,
        help="Total number of optimizer updates (default: 10,000).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Micro-batch size per forward step (default: 8).",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=2,
        help="Gradient accumulation steps (default: 2, yielding effective batch size 16).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for Transformer and CVAE layers (default: 1e-4).",
    )
    parser.add_argument(
        "--backbone-lr",
        type=float,
        default=1e-5,
        help="Learning rate for shared ResNet-18 visual backbone (default: 1e-5).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for AdamW (default: 1e-4).",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=500,
        help="Linear warmup steps for learning rate schedule (default: 500).",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=1000,
        help="Validation evaluation frequency in optimizer updates (default: 1000).",
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=2000,
        help="Checkpoint save frequency in optimizer updates (default: 2000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device (default: cuda if available).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of DataLoader workers (default: 2).",
    )
    return parser.parse_args()


def compute_file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit() -> str:
    # 1. Check baked commit file (useful for remote workers without .git)
    commit_file = Path(".git_commit")
    if commit_file.is_file():
        commit = commit_file.read_text().strip()
        if commit:
            return commit
    # 2. Check environment variable
    env_commit = os.environ.get("GIT_COMMIT")
    if env_commit:
        return env_commit.strip()
    # 3. Query git repository directly
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def save_checkpoint(
    policy: ACTPolicy,
    preprocessor,
    postprocessor,
    save_dir: Path,
    manifest: dict | None = None,
):
    save_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(save_dir)
    preprocessor.save_pretrained(save_dir)
    postprocessor.save_pretrained(save_dir)
    if manifest is not None:
        manifest_file = save_dir / "run_manifest.json"
        with open(manifest_file, "w") as f:
            json.dump(manifest, f, indent=2, default=str)


def evaluate_offline(policy: ACTPolicy, val_loader: DataLoader, preprocessor, device: torch.device) -> dict[str, float]:
    policy.eval()
    total_val_l1 = 0.0
    val_batches = 0

    with torch.inference_mode():
        for batch in val_loader:
            batch = preprocessor(batch)
            loss, _loss_dict = policy(batch)
            total_val_l1 += loss.item()
            val_batches += 1

    policy.train()
    mean_val_l1 = total_val_l1 / max(1, val_batches)
    return {"val_prior_l1": mean_val_l1}


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    train_dir = Path(args.dataset_dir)
    val_dir = Path(args.val_dataset_dir)

    if args.output_dir is None:
        args.output_dir = f"outputs/checkpoints/{args.policy_type}_nominal_v1"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    telemetry_path = output_dir / "training_telemetry.jsonl"
    print("=" * 80)
    print(f"🚀 Launching ACT Training: {args.policy_type.upper()}")
    print("=" * 80)
    print(f"• Policy Type:        {args.policy_type}")
    print(f"• Training Dataset:   {train_dir}")
    print(f"• Validation Dataset: {val_dir}")
    print(f"• Output Directory:   {output_dir}")
    print(f"• Device:             {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"• Total Updates:      {args.steps:,} (Micro-batch: {args.batch_size}, Grad Accum: {args.grad_accum} -> Effective BS: {args.batch_size * args.grad_accum})")
    print(f"• Learning Rates:     Policy={args.lr}, Backbone={args.backbone_lr}")

    # 1. Action Chunking Timestamp Configuration
    chunk_size = 50
    fps = 50
    delta_timestamps = {
        "action": [t / fps for t in range(chunk_size)],
    }

    print("\n📂 Ingesting Datasets...")
    train_dataset = LeRobotDataset(
        repo_id=train_dir.name,
        root=train_dir,
        delta_timestamps=delta_timestamps,
        video_backend="pyav",
    )
    val_dataset = LeRobotDataset(
        repo_id=val_dir.name,
        root=val_dir,
        delta_timestamps=delta_timestamps,
        video_backend="pyav",
    )
    print(f"  ✓ Train frames: {train_dataset.num_frames:,} across {train_dataset.num_episodes} episodes")
    print(f"  ✓ Val frames:   {val_dataset.num_frames:,} across {val_dataset.num_episodes} episodes")

    dl_generator = torch.Generator()
    dl_generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=dl_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # 2. Policy Feature Definition
    input_features = {
        "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
        "observation.images.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
    }
    normalization_mapping = {
        "VISUAL": NormalizationMode.MEAN_STD,
        "STATE": NormalizationMode.MEAN_STD,
        "ACTION": NormalizationMode.MEAN_STD,
    }

    if args.policy_type == "act-g":
        input_features["observation.environment_state"] = PolicyFeature(type=FeatureType.ENV, shape=(13,))
        normalization_mapping["ENV"] = NormalizationMode.IDENTITY

    output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
    }

    policy_cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=chunk_size,
        n_action_steps=10,
        n_decoder_layers=1,
        n_encoder_layers=4,
        n_vae_encoder_layers=4,
        dim_model=512,
        n_heads=8,
        dim_feedforward=3200,
        latent_dim=32,
        kl_weight=10.0,
        temporal_ensemble_coeff=None,
        vision_backbone="resnet18",
        pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
        optimizer_lr=args.lr,
        optimizer_lr_backbone=args.backbone_lr,
        optimizer_weight_decay=args.weight_decay,
        normalization_mapping=normalization_mapping,
        device=str(device),
    )

    print("\n🧠 Instantiating Policy and Pre/Post Processors...")
    policy = ACTPolicy(policy_cfg).to(device)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg,
        dataset_stats=train_dataset.meta.stats,
    )

    total_params = sum(p.numel() for p in policy.parameters())
    print(f"  ✓ Model initialized: {total_params:,} parameters (Backbone: shared ResNet-18)")

    # Construct and save run_manifest.json
    run_manifest = {
        "policy_type": args.policy_type,
        "training_seed": args.seed,
        "git_commit": get_git_commit(),
        "uv_lock_sha256": compute_file_sha256(Path("uv.lock")),
        "dataset_manifest": {
            "train": {
                "dir": str(train_dir),
                "num_episodes": train_dataset.num_episodes,
                "num_frames": train_dataset.num_frames,
            },
            "val": {
                "dir": str(val_dir),
                "num_episodes": val_dataset.num_episodes,
                "num_frames": val_dataset.num_frames,
            },
        },
        "environment": {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        },
        "act_config": asdict(policy_cfg),
        "optimizer": {
            "type": "AdamW",
            "lr": args.lr,
            "backbone_lr": args.backbone_lr,
            "weight_decay": args.weight_decay,
        },
        "scheduler": {
            "type": "CosineAnnealingWithLinearWarmup",
            "warmup_steps": args.warmup_steps,
            "total_steps": args.steps,
            "min_lr_ratio": 0.01,
        },
        "training_parameters": {
            "micro_batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum,
            "effective_batch_size": args.batch_size * args.grad_accum,
            "total_updates": args.steps,
            "amp_dtype": "float16",
        },
    }
    with open(output_dir / "run_manifest.json", "w") as f:
        json.dump(run_manifest, f, indent=2, default=str)
    print(f"  ✓ Persisted run manifest to {output_dir / 'run_manifest.json'}")

    # 3. Optimizer & Learning Rate Schedule
    optimizer = torch.optim.AdamW(
        policy.get_optim_params(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    def lr_lambda(current_step: int):
        if current_step < args.warmup_steps:
            return float(current_step) / float(max(1, args.warmup_steps))
        progress = float(current_step - args.warmup_steps) / float(max(1, args.steps - args.warmup_steps))
        return max(1e-2, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"), init_scale=1024.0)

    # 4. Training Loop
    print("\n🏋️  Starting Training Loop...")
    step_count = 0
    epoch = 0
    best_val_l1 = float("inf")
    start_time = time.perf_counter()

    train_iter = iter(train_loader)
    policy.train()
    optimizer.zero_grad()

    accum_loss = 0.0
    accum_l1 = 0.0
    accum_kld = 0.0

    while step_count < args.steps:
        # Micro-batch loop
        for _micro_step in range(args.grad_accum):
            try:
                batch = next(train_iter)
            except StopIteration:
                epoch += 1
                train_iter = iter(train_loader)
                batch = next(train_iter)

            batch = preprocessor(batch)

            with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
                loss, loss_dict = policy(batch)
                loss_scaled = loss / args.grad_accum

            scaler.scale(loss_scaled).backward()
            accum_loss += loss.item() / args.grad_accum
            accum_l1 += loss_dict.get("l1_loss", 0.0) / args.grad_accum
            accum_kld += loss_dict.get("kld_loss", 0.0) / args.grad_accum

        # Optimizer update
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()
        step_count += 1

        # Step logging every 100 steps
        if step_count % 100 == 0 or step_count == 1:
            elapsed = time.perf_counter() - start_time
            steps_per_sec = step_count / max(1e-5, elapsed)
            current_lr = optimizer.param_groups[0]["lr"]
            peak_vram_mb = torch.cuda.max_memory_allocated() / 1e6 if device.type == "cuda" else 0.0

            log_entry = {
                "step": step_count,
                "epoch": epoch,
                "train_total_loss": round(accum_loss, 4),
                "train_l1": round(accum_l1, 4),
                "train_kld": round(accum_kld, 4),
                "grad_norm": round(float(grad_norm.item()) if torch.is_tensor(grad_norm) else float(grad_norm), 4),
                "lr": round(current_lr, 7),
                "peak_vram_mb": round(peak_vram_mb, 1),
                "steps_per_sec": round(steps_per_sec, 2),
                "timestamp": round(time.time(), 2),
            }

            with open(telemetry_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            print(
                f"Step {step_count:6d}/{args.steps} | Epoch {epoch:2d} | "
                f"Loss: {accum_loss:.4f} (L1: {accum_l1:.4f}, KL: {accum_kld:.4f}) | "
                f"Norm: {log_entry['grad_norm']:.2f} | LR: {current_lr:.2e} | "
                f"VRAM: {peak_vram_mb:.0f} MB | {steps_per_sec:.1f} steps/s"
            )

        accum_loss = 0.0
        accum_l1 = 0.0
        accum_kld = 0.0

        # Offline Validation (Deployment prior z=0)
        if step_count % args.eval_freq == 0:
            print(f"\n📊 Running Offline Validation at Step {step_count}...")
            val_metrics = evaluate_offline(policy, val_loader, preprocessor, device)
            val_prior_l1 = val_metrics["val_prior_l1"]
            print(f"  ✓ Val Prior-Mean L1 (z=0): {val_prior_l1:.4f}")

            val_log = {
                "step": step_count,
                "val_prior_l1": round(val_prior_l1, 4),
                "timestamp": round(time.time(), 2),
            }
            with open(telemetry_path, "a") as f:
                f.write(json.dumps(val_log) + "\n")

            if val_prior_l1 < best_val_l1:
                best_val_l1 = val_prior_l1
                best_dir = output_dir / "best_offline"
                save_checkpoint(policy, preprocessor, postprocessor, best_dir, manifest=run_manifest)
                print(f"  ⭐ New best validation prior L1 ({best_val_l1:.4f})! Saved to {best_dir}")

        # Periodic Checkpoint Saving
        if step_count % args.save_freq == 0:
            step_dir = output_dir / f"checkpoint_step_{step_count}"
            save_checkpoint(policy, preprocessor, postprocessor, step_dir, manifest=run_manifest)
            print(f"  💾 Saved periodic checkpoint to {step_dir}")

    # Final Checkpoint
    final_dir = output_dir / "final"
    save_checkpoint(policy, preprocessor, postprocessor, final_dir, manifest=run_manifest)
    print("\n" + "=" * 80)
    print(f"✅ Training Complete! Checkpoints saved to: {output_dir}")
    print(f"  • Final checkpoint:        {final_dir}")
    print(f"  • Best offline checkpoint: {output_dir / 'best_offline'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
