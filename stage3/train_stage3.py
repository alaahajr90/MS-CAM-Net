"""Train MS-CAM-Net Stage 3 using subject-exclusive development folds."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from dataset_stage3 import Stage3NPYDataset, discover_subjects
from losses_stage3 import Stage3Loss
from model_stage3 import StressRecognitionModel
from trainer_stage3 import evaluate_epoch, train_epoch

FIXED_TEST_SUBJECTS = ("s1", "s10", "s11", "s12", "s13", "s14", "s15", "s16")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-dir", type=Path, required=True)
    parser.add_argument("--stage1-dir", type=Path, required=True)
    parser.add_argument("--stage2-dir", type=Path, required=True)
    parser.add_argument("--stage2-weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40,
                        help="Maximum epochs per development fold.")
    parser.add_argument("--final-epochs", type=int, default=100,
                        help="Epochs used when --final-train is requested.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--final-train", action="store_true")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def import_class(module_path: Path, module_name: str, class_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {class_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def load_stage2(args, device):
    if str(args.stage1_dir) not in sys.path:
        sys.path.insert(0, str(args.stage1_dir))
    if str(args.stage2_dir) not in sys.path:
        sys.path.insert(0, str(args.stage2_dir))

    Encoder = import_class(args.stage1_dir / "model_stage1.py", "repo_stage1_model", "Encoder")
    Stage2Model = import_class(args.stage2_dir / "stage2_model.py", "repo_stage2_model", "Stage2PhysiologicalModel")
    encoder = Encoder()
    stage2 = Stage2Model(encoder)
    state = torch.load(args.stage2_weights, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    stage2.load_state_dict(state, strict=True)
    return stage2.to(device)


def make_model(args, device):
    return StressRecognitionModel(load_stage2(args, device)).to(device)


def make_loader(data_dir, subjects, batch_size, shuffle, workers, device):
    return DataLoader(
        Stage3NPYDataset(data_dir, subjects),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )


def warmup_factor(epoch: int, warmup_epochs: int) -> float:
    if warmup_epochs <= 0:
        return 1.0
    return min(1.0, epoch / float(warmup_epochs))


def train_fold(args, fold_idx, train_subjects, val_subjects, device):
    fold_dir = args.output_dir / f"fold_{fold_idx:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    model = make_model(args, device)
    criterion = Stage3Loss().to(device)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        amsgrad=True,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    train_loader = make_loader(
        args.development_dir, train_subjects, args.batch_size, True, args.num_workers, device
    )
    val_loader = make_loader(
        args.development_dir, val_subjects, 1, False, args.num_workers, device
    )

    base_lrs = [group["lr"] for group in optimizer.param_groups]
    best_loss = float("inf")
    best_epoch = 0
    no_improvement = 0

    history_path = fold_dir / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "epoch", "train_loss", "val_loss",
            "state_accuracy", "state_f1", "state_auc",
            "level_accuracy", "level_f1", "level_auc",
        ])

        for epoch in range(1, args.epochs + 1):
            if epoch <= args.warmup_epochs:
                factor = warmup_factor(epoch, args.warmup_epochs)
                for group, base_lr in zip(optimizer.param_groups, base_lrs):
                    group["lr"] = base_lr * factor

            train_result = train_epoch(
                model, train_loader, optimizer, criterion, device, args.accumulation_steps
            )
            val_result = evaluate_epoch(model, val_loader, criterion, device)

            if epoch > args.warmup_epochs:
                scheduler.step(val_result["loss"])

            sm = val_result["metrics"]["state"]
            lm = val_result["metrics"]["level"]
            writer.writerow([
                epoch, train_result["loss"], val_result["loss"],
                sm["accuracy"], sm["f1"], sm["auc"],
                lm["accuracy"], lm["f1"], lm["auc"],
            ])
            handle.flush()

            print(
                f"Fold {fold_idx:02d} | Epoch {epoch:03d} | "
                f"train={train_result['loss']:.5f} | val={val_result['loss']:.5f} | "
                f"State Acc={sm['accuracy']:.2f}% | Level Acc={lm['accuracy']:.2f}%"
            )

            if val_result["loss"] < best_loss:
                best_loss = val_result["loss"]
                best_epoch = epoch
                no_improvement = 0
                torch.save(model.state_dict(), fold_dir / "best_stage3_model.pth")
                best_metrics = val_result["metrics"]
            else:
                no_improvement += 1

            if no_improvement >= args.patience:
                break

    metadata = {
        "fold": fold_idx,
        "train_subjects": list(train_subjects),
        "validation_subjects": list(val_subjects),
        "selected_epoch": best_epoch,
        "joint_validation_loss": best_loss,
        "validation_metrics": best_metrics,
        "trainable_parameters": model.trainable_parameter_count(),
    }
    (fold_dir / "split_and_selection.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=True), encoding="utf-8"
    )
    return metadata


def train_final(args, subjects, device):
    final_dir = args.output_dir / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    model = make_model(args, device)
    criterion = Stage3Loss().to(device)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        amsgrad=True,
    )
    loader = make_loader(
        args.development_dir, subjects, args.batch_size, True, args.num_workers, device
    )

    base_lrs = [group["lr"] for group in optimizer.param_groups]
    history = []
    for epoch in range(1, args.final_epochs + 1):
        if epoch <= args.warmup_epochs:
            factor = warmup_factor(epoch, args.warmup_epochs)
            for group, base_lr in zip(optimizer.param_groups, base_lrs):
                group["lr"] = base_lr * factor
        result = train_epoch(
            model, loader, optimizer, criterion, device, args.accumulation_steps
        )
        history.append({"epoch": epoch, "train_loss": result["loss"], "metrics": result["metrics"]})
        print(f"Final Stage 3 | Epoch {epoch:03d}/{args.final_epochs:03d} | loss={result['loss']:.5f}")

    torch.save(model.state_dict(), final_dir / "final_stage3_model.pth")
    (final_dir / "training_metadata.json").write_text(
        json.dumps({
            "subjects": list(subjects),
            "epochs": args.final_epochs,
            "test_subjects_not_used": list(FIXED_TEST_SUBJECTS),
            "history": history,
        }, indent=2, allow_nan=True),
        encoding="utf-8",
    )


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    subjects = discover_subjects(args.development_dir)
    if any(subject in FIXED_TEST_SUBJECTS for subject in subjects):
        raise RuntimeError("The development directory contains one or more fixed test subjects.")
    if len(subjects) != 48:
        raise RuntimeError(f"Expected 48 development subjects, found {len(subjects)}.")

    splitter = KFold(n_splits=10, shuffle=True, random_state=args.seed)
    subject_array = np.asarray(subjects)
    fold_metadata = []
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(subject_array), start=1):
        fold_metadata.append(
            train_fold(
                args,
                fold_idx,
                subject_array[train_idx].tolist(),
                subject_array[val_idx].tolist(),
                device,
            )
        )

    (args.output_dir / "cross_validation_summary.json").write_text(
        json.dumps({"seed": args.seed, "folds": fold_metadata}, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    if args.final_train:
        train_final(args, subjects, device)


if __name__ == "__main__":
    main()
