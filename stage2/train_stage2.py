"""Train MS-CAM-Net Stage 2 with strict subject-exclusive development folds."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from dataset_stage2 import Stage2NPYDataset, discover_subjects
from losses_stage2 import Stage2Loss
from stage2_model import Stage2PhysiologicalModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-dir", required=True, type=Path)
    parser.add_argument("--stage1-dir", required=True, type=Path,
                        help="Directory containing model_stage1.py.")
    parser.add_argument("--stage1-weights", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-folds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--final-train", action="store_true",
                        help="Train one final model on all development subjects for the median selected fold epoch.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def import_stage1_encoder(stage1_dir: Path):
    sys.path.insert(0, str(stage1_dir.resolve()))
    from model_stage1 import Encoder  # pylint: disable=import-outside-toplevel
    return Encoder


def load_encoder(encoder_class, weights: Path, device: torch.device):
    encoder = encoder_class().to(device)
    state = torch.load(weights, map_location=device, weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    encoder.load_state_dict(state, strict=True)
    return encoder


def make_optimizer(model: Stage2PhysiologicalModel, lr: float, weight_decay: float):
    return torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": 0.05 * lr},
            {"params": model.attention_stacks.parameters(), "lr": lr},
            {"params": model.projectors.parameters(), "lr": lr},
            {"params": model.feature_fusion.parameters(), "lr": lr},
            {"params": model.long_range_refinement.parameters(), "lr": lr},
            {"params": model.rppg_head.parameters(), "lr": 1.2 * lr},
            {"params": model.aux_rppg_heads.parameters(), "lr": 1.2 * lr},
            {"params": model.vitals_head.parameters(), "lr": 1.5 * lr},
        ],
        lr=lr,
        weight_decay=weight_decay,
    )


def train_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total = 0.0
    count = 0
    for batch in loader:
        clip = batch["clip"].to(device, non_blocking=True)
        targets = {
            "rppg": batch["rppg"].to(device, non_blocking=True),
            "hr": batch["hr"].to(device, non_blocking=True),
            "hrv": batch["hrv"].to(device, non_blocking=True),
        }
        optimizer.zero_grad(set_to_none=True)
        outputs = model(clip)
        loss, _ = criterion(outputs, targets, epoch=epoch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += float(loss.detach()) * clip.shape[0]
        count += clip.shape[0]
    return total / max(count, 1)


@torch.no_grad()
def validate(model, loader, criterion, device, epoch):
    model.eval()
    losses = []
    pred_hr, true_hr, pred_hrv, true_hrv = [], [], [], []
    for batch in loader:
        clip = batch["clip"].to(device, non_blocking=True)
        targets = {
            "rppg": batch["rppg"].to(device, non_blocking=True),
            "hr": batch["hr"].to(device, non_blocking=True),
            "hrv": batch["hrv"].to(device, non_blocking=True),
        }
        outputs = model(clip)
        loss, _ = criterion(outputs, targets, epoch=epoch)
        losses.append(float(loss.detach()))
        pred_hr.extend(outputs["hr"].cpu().numpy().tolist())
        true_hr.extend(batch["hr"].numpy().tolist())
        pred_hrv.extend(outputs["hrv"].cpu().numpy().tolist())
        true_hrv.extend(batch["hrv"].numpy().tolist())

    pred_hr = np.asarray(pred_hr, dtype=float)
    true_hr = np.asarray(true_hr, dtype=float)
    pred_hrv = np.asarray(pred_hrv, dtype=float)
    true_hrv = np.asarray(true_hrv, dtype=float)
    hr_mae = float(np.mean(np.abs(pred_hr - true_hr)))
    hrv_mae = float(np.mean(np.abs(pred_hrv - true_hrv)))
    return float(np.mean(losses)), hr_mae, hrv_mae


def train_fold(args, fold_idx, train_subjects, val_subjects, encoder_class, device):
    fold_dir = args.output_dir / f"fold_{fold_idx:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    train_set = Stage2NPYDataset(args.development_dir, train_subjects)
    val_set = Stage2NPYDataset(args.development_dir, val_subjects)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda"
    )
    val_loader = DataLoader(
        val_set, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda"
    )

    encoder = load_encoder(encoder_class, args.stage1_weights, device)
    model = Stage2PhysiologicalModel(encoder).to(device)
    criterion = Stage2Loss().to(device)
    optimizer = make_optimizer(model, args.learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_hr_mae = float("inf")
    best_epoch = 0
    no_improvement = 0
    history_path = fold_dir / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_hr_mae_bpm", "val_hrv_mae_ms"])

        for epoch in range(1, args.epochs + 1):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
            val_loss, hr_mae, hrv_mae = validate(model, val_loader, criterion, device, epoch)
            writer.writerow([epoch, train_loss, val_loss, hr_mae, hrv_mae])
            handle.flush()

            print(
                f"Fold {fold_idx:02d} | Epoch {epoch:03d} | "
                f"train={train_loss:.6f} | val={val_loss:.6f} | "
                f"HR MAE={hr_mae:.3f} bpm | HRV MAE={hrv_mae:.3f} ms"
            )

            if hr_mae < best_hr_mae:
                best_hr_mae = hr_mae
                best_epoch = epoch
                no_improvement = 0
                torch.save(model.state_dict(), fold_dir / "best_stage2_model.pth")
            else:
                no_improvement += 1

            scheduler.step(hr_mae)
            if no_improvement >= args.patience:
                break

    metadata = {
        "fold": fold_idx,
        "train_subjects": list(train_subjects),
        "validation_subjects": list(val_subjects),
        "best_epoch": best_epoch,
        "best_validation_hr_mae_bpm": best_hr_mae,
    }
    (fold_dir / "split_and_selection.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def train_final_model(args, subjects, encoder_class, device, epochs):
    final_dir = args.output_dir / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    dataset = Stage2NPYDataset(args.development_dir, subjects)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda"
    )
    model = Stage2PhysiologicalModel(load_encoder(encoder_class, args.stage1_weights, device)).to(device)
    criterion = Stage2Loss().to(device)
    optimizer = make_optimizer(model, args.learning_rate, weight_decay=0.05)

    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, loader, optimizer, criterion, device, epoch)
        print(f"Final model | Epoch {epoch:03d}/{epochs:03d} | train={loss:.6f}")

    torch.save(model.state_dict(), final_dir / "final_stage2_model.pth")
    (final_dir / "training_metadata.json").write_text(
        json.dumps({"subjects": list(subjects), "epochs": epochs, "weight_decay": 0.05}, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_class = import_stage1_encoder(args.stage1_dir)

    subjects = discover_subjects(args.development_dir)
    if len(subjects) != 48:
        raise RuntimeError(
            f"Expected exactly 48 development subjects after fixed-test exclusion, found {len(subjects)}."
        )
    if not 2 <= args.num_folds <= len(subjects):
        raise ValueError("num-folds must be between 2 and the number of development subjects.")

    splitter = KFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
    fold_metadata = []
    subject_array = np.asarray(subjects)
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(subject_array), start=1):
        fold_metadata.append(
            train_fold(
                args,
                fold_idx,
                subject_array[train_idx].tolist(),
                subject_array[val_idx].tolist(),
                encoder_class,
                device,
            )
        )

    summary = {"num_folds": args.num_folds, "seed": args.seed, "folds": fold_metadata}
    (args.output_dir / "cross_validation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    if args.final_train:
        selected_epochs = [item["best_epoch"] for item in fold_metadata if item["best_epoch"] > 0]
        final_epochs = int(round(float(np.median(selected_epochs))))
        train_final_model(args, subjects, encoder_class, device, final_epochs)


if __name__ == "__main__":
    main()
