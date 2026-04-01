"""EEG-JEPA Pretraining Script.

Usage:
  uv run python main.py --config configs/default.yaml --hf-repo username/reve-preprocessed

For local testing with dummy data:
  uv run python main.py --config configs/default.yaml --dummy
"""

import argparse
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import torch
import torch.nn as nn
import wandb
import yaml
from torch.utils.data import DataLoader

from src.model.eeg_jepa import EEGJEPA
from src.preprocessing.dataset import make_dummy_dataset
from src.preprocessing.streaming_dataset import (
    StreamingEEGDataset,
    eeg_collate_fn,
)


def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer, cfg: dict, steps_per_epoch: int
) -> torch.optim.lr_scheduler.LRScheduler:
    total_steps = cfg["training"]["epochs"] * steps_per_epoch
    warmup_epochs = cfg["training"].get("warmup_epochs", 10)
    warmup_steps = warmup_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + __import__("math").cos(__import__("math").pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_dataloader(
    cfg: dict, hf_repo: str | None, dummy: bool
) -> DataLoader:
    if dummy:
        dataset = make_dummy_dataset(
            num_samples=500,
            num_channels=19,
            window_samples=cfg["preprocessing"]["window_samples"],
        )
        return DataLoader(
            dataset,
            batch_size=cfg["training"]["batch_size"],
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )

    dataset = StreamingEEGDataset(
        hf_repo=hf_repo,
        split="train",
        shuffle_buffer=10000,
    )
    return DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        collate_fn=eeg_collate_fn,
        num_workers=2,
        prefetch_factor=4,
        pin_memory=True,
    )


def train_one_epoch(
    model: EEGJEPA,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    epoch: int,
    global_step: int,
    grad_accum_steps: int = 1,
    log_interval: int = 50,
) -> tuple[dict, int]:
    model.train()
    total_loss = 0.0
    total_pred = 0.0
    total_sigreg = 0.0
    total_query = 0.0
    num_steps = 0
    optimizer.zero_grad()

    t_start = time.time()

    for step, batch in enumerate(loader):
        # Move to device
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # Forward with AMP (bf16)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = model(batch)
            loss = output["loss"] / grad_accum_steps

        # Backward (no GradScaler needed for bf16)
        loss.backward()

        if (step + 1) % grad_accum_steps == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            # Log to wandb every step
            wandb.log({
                "train/loss": output["loss"].item(),
                "train/pred_loss": output["pred_loss"].item(),
                "train/sigreg_loss": output["sigreg_loss"].item(),
                "train/query_loss": output["query_loss"].item(),
                "train/grad_norm": grad_norm.item(),
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/epoch": epoch,
                "train/batch_max_channels": batch["num_channels"].max().item(),
                "train/batch_mean_channels": batch["num_channels"].float().mean().item(),
            }, step=global_step)

        total_loss += output["loss"].item()
        total_pred += output["pred_loss"].item()
        total_sigreg += output["sigreg_loss"].item()
        total_query += output["query_loss"].item()
        num_steps += 1

        if (step + 1) % log_interval == 0:
            elapsed = time.time() - t_start
            avg_loss = total_loss / num_steps
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  [E{epoch}|S{step+1}] "
                f"loss={avg_loss:.4f} "
                f"pred={total_pred/num_steps:.4f} "
                f"sigreg={total_sigreg/num_steps:.4f} "
                f"query={total_query/num_steps:.6f} "
                f"lr={lr:.2e} "
                f"({elapsed:.0f}s)"
            )

    if num_steps == 0:
        return {"loss": 0, "pred_loss": 0, "sigreg_loss": 0, "query_loss": 0}, global_step

    return {
        "loss": total_loss / num_steps,
        "pred_loss": total_pred / num_steps,
        "sigreg_loss": total_sigreg / num_steps,
        "query_loss": total_query / num_steps,
    }, global_step


def save_checkpoint(
    model: EEGJEPA,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    metrics: dict,
    path: Path,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics": metrics,
        },
        path,
    )
    print(f"  Checkpoint saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="EEG-JEPA Pretraining")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--hf-repo", type=str, default=None,
        help="HuggingFace dataset repo for streaming",
    )
    parser.add_argument(
        "--dummy", action="store_true",
        help="Use dummy data for testing",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints",
        help="Directory for saving checkpoints",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--grad-accum", type=int, default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--wandb-run-name", type=str, default=None,
        help="Custom wandb run name",
    )
    parser.add_argument(
        "--no-wandb", action="store_true",
        help="Disable wandb logging",
    )
    args = parser.parse_args()

    if not args.dummy and not args.hf_repo:
        parser.error("Either --hf-repo or --dummy is required")

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    log_cfg = cfg.get("logging", {})
    log_interval = log_cfg.get("log_interval", 50)
    save_interval = log_cfg.get("save_interval", 5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    # Build model
    model = EEGJEPA(cfg).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.1f}M)")

    # Component breakdown
    comps = {}
    for name, param in model.named_parameters():
        comp = name.split(".")[0]
        comps[comp] = comps.get(comp, 0) + param.numel()
    for comp, count in sorted(comps.items(), key=lambda x: -x[1]):
        print(f"  {comp:20s}: {count:>10,} ({count/num_params*100:.1f}%)")

    # Init wandb
    if not args.no_wandb:
        wandb.init(
            project=log_cfg.get("wandb_project", "eeg-wm-jepa"),
            name=args.wandb_run_name,
            config={
                "model": cfg["model"],
                "training": cfg["training"],
                "preprocessing": cfg["preprocessing"],
                "grad_accum": args.grad_accum,
                "effective_batch_size": cfg["training"]["batch_size"] * args.grad_accum,
                "num_params": num_params,
                "trainable_params": trainable_params,
                "components": comps,
                "device": str(device),
                "gpu": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
            },
        )
        wandb.watch(model, log="gradients", log_freq=log_interval * 5)
    else:
        wandb.init(mode="disabled")

    # Build training infrastructure
    loader = build_dataloader(cfg, args.hf_repo, args.dummy)
    steps_per_epoch = 500 if not args.dummy else len(loader)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch)

    start_epoch = 0
    global_step = 0

    # Resume from checkpoint
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        global_step = start_epoch * steps_per_epoch
        print(f"Resumed from epoch {start_epoch}")

    # Training loop
    ckpt_dir = Path(args.checkpoint_dir)
    epochs = cfg["training"]["epochs"]
    effective_bs = cfg["training"]["batch_size"] * args.grad_accum
    print(f"\nStarting training: {epochs} epochs")
    print(f"Batch size: {cfg['training']['batch_size']} × {args.grad_accum} = {effective_bs} effective")
    print(f"Patch size: {cfg['model']['patch_size']} → {cfg['preprocessing']['window_samples'] // cfg['model']['patch_size']} patches")
    print(f"Seq length: {cfg['model']['num_queries']} queries × {cfg['preprocessing']['window_samples'] // cfg['model']['patch_size']} patches = {cfg['model']['num_queries'] * (cfg['preprocessing']['window_samples'] // cfg['model']['patch_size'])}")
    print()

    for epoch in range(start_epoch, epochs):
        t_epoch = time.time()
        print(f"Epoch {epoch}/{epochs-1}")

        metrics, global_step = train_one_epoch(
            model, loader, optimizer, scheduler,
            device, epoch, global_step,
            grad_accum_steps=args.grad_accum,
            log_interval=log_interval,
        )

        elapsed = time.time() - t_epoch

        # Epoch-level logging
        wandb.log({
            "epoch/loss": metrics["loss"],
            "epoch/pred_loss": metrics["pred_loss"],
            "epoch/sigreg_loss": metrics["sigreg_loss"],
            "epoch/query_loss": metrics["query_loss"],
            "epoch/duration_sec": elapsed,
        }, step=global_step)

        print(
            f"  Epoch {epoch} done in {elapsed:.0f}s — "
            f"loss={metrics['loss']:.4f} "
            f"pred={metrics['pred_loss']:.4f} "
            f"sigreg={metrics['sigreg_loss']:.4f}"
        )

        # Save checkpoint
        if (epoch + 1) % save_interval == 0 or epoch == epochs - 1:
            save_checkpoint(
                model, optimizer, scheduler, epoch, metrics,
                ckpt_dir / f"epoch_{epoch:04d}.pt",
            )
            # Log checkpoint as wandb artifact
            if not args.no_wandb:
                artifact = wandb.Artifact(
                    f"model-epoch-{epoch:04d}",
                    type="model",
                    metadata=metrics,
                )
                artifact.add_file(str(ckpt_dir / f"epoch_{epoch:04d}.pt"))
                wandb.log_artifact(artifact)

    print("\nTraining complete!")
    wandb.finish()


if __name__ == "__main__":
    main()
