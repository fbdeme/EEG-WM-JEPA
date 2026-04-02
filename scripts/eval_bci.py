"""BCI Competition IV 2a Downstream Evaluation.

Evaluates pretrained EEG-JEPA encoder on 4-class motor imagery.
Supports both linear probing (frozen encoder) and fine-tuning.

Usage:
  # Linear probing with pretrained model
  uv run python scripts/eval_bci.py --checkpoint checkpoints/epoch_0099.pt

  # Fine-tuning
  uv run python scripts/eval_bci.py --checkpoint checkpoints/epoch_0099.pt --finetune

  # Random init baseline (no pretrain)
  uv run python scripts/eval_bci.py --random-init

  # Cross-subject evaluation
  uv run python scripts/eval_bci.py --checkpoint checkpoints/epoch_0099.pt --cross-subject
"""

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.evaluation.bci_dataset import load_bci_subject, load_bci_cross_subject
from src.evaluation.downstream import DownstreamModel
from src.model.eeg_jepa import EEGJEPA


NUM_CLASSES = 4
SUBJECTS = list(range(1, 10))


def collate_fn(batch):
    """Simple collate for fixed-channel BCI data."""
    return {
        "eeg": torch.stack([s["eeg"] for s in batch]),
        "coords": batch[0]["coords"].unsqueeze(0).expand(len(batch), -1, -1),
        "label": torch.stack([s["label"] for s in batch]),
        "num_channels": torch.tensor([s["num_channels"] for s in batch]),
    }


def train_and_evaluate(
    model: DownstreamModel,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 30,
    lr: float = 1e-3,
) -> dict:
    """Train classification head and evaluate."""
    # Only optimize trainable parameters
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(epochs):
        # Train
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for batch in train_loader:
            batch_gpu = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(batch_gpu)
                loss = criterion(logits, batch_gpu["label"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_gpu["label"].size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == batch_gpu["label"]).sum().item()
            total += batch_gpu["label"].size(0)

        scheduler.step()
        train_acc = correct / total

        # Eval
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_loader:
                batch_gpu = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(batch_gpu)
                preds = logits.argmax(dim=1)
                correct += (preds == batch_gpu["label"]).sum().item()
                total += batch_gpu["label"].size(0)

        test_acc = correct / total
        best_acc = max(best_acc, test_acc)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"    Epoch {epoch+1:>3d}: "
                f"train_acc={train_acc:.3f} test_acc={test_acc:.3f} "
                f"loss={total_loss/total:.4f}"
            )

    return {"best_acc": best_acc, "final_acc": test_acc}


def build_model(cfg, checkpoint_path, finetune, device):
    """Build DownstreamModel from config and optional checkpoint."""
    eeg_jepa = EEGJEPA(cfg)

    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        eeg_jepa.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint: {checkpoint_path}")

    model = DownstreamModel(
        eeg_jepa,
        num_classes=NUM_CLASSES,
        freeze_encoder=not finetune,
    )
    return model.to(device)


def main():
    parser = argparse.ArgumentParser(description="BCI IV 2a Evaluation")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--random-init", action="store_true",
                        help="Use random init (no pretrain baseline)")
    parser.add_argument("--finetune", action="store_true",
                        help="Fine-tune encoder (default: linear probing)")
    parser.add_argument("--cross-subject", action="store_true",
                        help="LOSO cross-subject evaluation")
    parser.add_argument("--subjects", type=int, nargs="+", default=SUBJECTS)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    if not args.checkpoint and not args.random_init:
        parser.error("Provide --checkpoint or --random-init")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = "finetune" if args.finetune else "linear_probe"
    method = "cross-subject" if args.cross_subject else "within-subject"
    print(f"BCI IV 2a | {mode} | {method}")
    print(f"Subjects: {args.subjects}")
    print()

    results = {}

    for subj in args.subjects:
        print(f"Subject {subj}:")

        # Load data
        if args.cross_subject:
            train_ds, test_ds, _ = load_bci_cross_subject(subj)
        else:
            train_ds, test_ds, _ = load_bci_subject(subj)

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            collate_fn=collate_fn, drop_last=False,
        )
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_fn,
        )

        # Build fresh model for each subject
        model = build_model(cfg, args.checkpoint, args.finetune, device)

        # Train & eval
        result = train_and_evaluate(
            model, train_loader, test_loader, device,
            epochs=args.epochs, lr=args.lr,
        )
        results[subj] = result
        print(f"  → Best: {result['best_acc']:.3f}")
        print()

    # Summary
    accs = [r["best_acc"] for r in results.values()]
    mean_acc = sum(accs) / len(accs)
    print("=" * 40)
    print(f"Mean accuracy: {mean_acc:.3f}")
    for subj, r in results.items():
        print(f"  S{subj}: {r['best_acc']:.3f}")


if __name__ == "__main__":
    main()
