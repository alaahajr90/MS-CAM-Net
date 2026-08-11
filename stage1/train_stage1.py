"""Train MS-CAM-Net Stage 1 with strict subject-independent test exclusion.

The fixed test subjects are never loaded, evaluated, or used for checkpoint
selection in this script. Ten subject-exclusive folds are created only from the
48-subject development pool. A final development-only encoder can optionally be
trained for a validation-selected number of epochs for transfer to Stage 2.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_stage1 import Stage1NPYDataset
from model_stage1 import Encoder, ProjectionHead, contrastive_temporal_loss
from visualization_utils import plot_loss_curves


FIXED_TEST_SUBJECTS = {
    "s1", "s10", "s11", "s12", "s13", "s14", "s15", "s16"
}


@dataclass(frozen=True)
class Stage1Config:
    seed: int = 42
    batch_size: int = 16
    max_epochs: int = 100
    n_folds: int = 10
    learning_rate: float = 3e-4
    patience: int = 10
    temporal_crop_frames: int = 128
    temperature: float = 0.07
    margin: float = 0.3
    temporal_weight: float = 0.25
    num_workers: int = 4
    projection_input_dim: int = 64
    projection_hidden_dim: int = 64
    projection_output_dim: int = 64
    projection_dropout: float = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MS-CAM-Net Stage 1.")
    parser.add_argument(
        "--development-dir",
        type=Path,
        required=True,
        help="Directory containing only development-partition .npy clips.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory used for checkpoints, logs, plots, and split metadata.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--skip-final-training",
        action="store_true",
        help="Run only the ten development folds and do not create a final transfer encoder.",
    )
    return parser.parse_args()


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def log(message: str, log_file: Path) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    text = f"[{timestamp}] {message}"
    print(text)
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(text + "\n")


def load_subject_inventory(development_dir: Path) -> Dict[str, List[Path]]:
    subject_to_files: Dict[str, List[Path]] = {}

    for path in sorted(development_dir.glob("*.npy")):
        data = np.load(path, allow_pickle=True).item()
        meta = data.get("meta", {})
        subject = str(meta.get("sub", "")).strip()
        if not subject:
            raise ValueError(f"Missing meta['sub'] in {path}")
        if subject in FIXED_TEST_SUBJECTS:
            raise RuntimeError(
                f"Test subject {subject} was found in the Stage 1 development directory. "
                "Stage 1 must exclude all fixed test subjects."
            )
        subject_to_files.setdefault(subject, []).append(path)

    for files in subject_to_files.values():
        files.sort()

    if not subject_to_files:
        raise RuntimeError(f"No Stage 1 clips were found in {development_dir}")

    return dict(sorted(subject_to_files.items()))


def files_for_subjects(
    subject_to_files: Dict[str, List[Path]],
    subjects: Sequence[str],
) -> List[Path]:
    return [path for subject in subjects for path in subject_to_files[subject]]


def make_loader(
    data_dir: Path,
    samples: Sequence[Path],
    config: Stage1Config,
    *,
    shuffle: bool,
    random_crop: bool,
    seed_offset: int,
) -> DataLoader:
    dataset = Stage1NPYDataset(
        data_dir=data_dir,
        clip_len=config.temporal_crop_frames,
        samples=samples,
        random_crop=random_crop,
    )

    generator = torch.Generator()
    generator.manual_seed(config.seed + seed_offset)

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        worker_init_fn=worker_init_fn,
        generator=generator,
        persistent_workers=config.num_workers > 0,
    )


def build_models(config: Stage1Config, device: torch.device):
    encoder = Encoder().to(device)
    projector = ProjectionHead(
        input_dim=config.projection_input_dim,
        hidden_dim=config.projection_hidden_dim,
        output_dim=config.projection_output_dim,
        dropout_rate=config.projection_dropout,
    ).to(device)
    return encoder, projector


def run_epoch(
    encoder: Encoder,
    projector: ProjectionHead,
    loader: DataLoader,
    device: torch.device,
    config: Stage1Config,
    optimizer: torch.optim.Optimizer | None,
) -> Dict[str, float]:
    training = optimizer is not None
    encoder.train(training)
    projector.train(training)

    totals = {
        "total": 0.0,
        "contrastive": 0.0,
        "temporal_unweighted": 0.0,
    }
    batches = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        progress = tqdm(loader, leave=False, desc="Train" if training else "Validation")
        for view1, view2 in progress:
            view1 = view1.to(device, non_blocking=True)
            view2 = view2.to(device, non_blocking=True)

            z1 = projector(encoder(view1))
            z2 = projector(encoder(view2))
            loss, components = contrastive_temporal_loss(
                z1,
                z2,
                temperature=config.temperature,
                margin=config.margin,
                temporal_weight=config.temporal_weight,
            )

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            totals["total"] += float(loss.detach().cpu())
            totals["contrastive"] += float(components["contrastive"].cpu())
            totals["temporal_unweighted"] += float(
                components["temporal_unweighted"].cpu()
            )
            batches += 1

            progress.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}")

    if batches == 0:
        raise RuntimeError("The DataLoader produced zero batches.")

    return {key: value / batches for key, value in totals.items()}


def save_history(path: Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def train_fold(
    fold_number: int,
    train_subjects: Sequence[str],
    val_subjects: Sequence[str],
    subject_to_files: Dict[str, List[Path]],
    development_dir: Path,
    output_dir: Path,
    config: Stage1Config,
    device: torch.device,
    log_file: Path,
) -> Tuple[int, float]:
    fold_dir = output_dir / "folds" / f"fold_{fold_number:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    with (fold_dir / "subjects.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "train_subjects": list(train_subjects),
                "validation_subjects": list(val_subjects),
                "fixed_test_subjects_excluded": sorted(FIXED_TEST_SUBJECTS),
            },
            handle,
            indent=2,
        )

    train_samples = files_for_subjects(subject_to_files, train_subjects)
    val_samples = files_for_subjects(subject_to_files, val_subjects)

    train_loader = make_loader(
        development_dir,
        train_samples,
        config,
        shuffle=True,
        random_crop=True,
        seed_offset=fold_number * 100,
    )
    val_loader = make_loader(
        development_dir,
        val_samples,
        config,
        shuffle=False,
        random_crop=False,
        seed_offset=fold_number * 100 + 1,
    )

    set_reproducibility(config.seed + fold_number)
    encoder, projector = build_models(config, device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(projector.parameters()),
        lr=config.learning_rate,
    )

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    history: List[Dict[str, float]] = []

    log(
        f"Fold {fold_number}: {len(train_subjects)} training subjects, "
        f"{len(val_subjects)} validation subjects, "
        f"{len(train_samples)} training clips, {len(val_samples)} validation clips.",
        log_file,
    )

    for epoch in range(1, config.max_epochs + 1):
        train_metrics = run_epoch(
            encoder,
            projector,
            train_loader,
            device,
            config,
            optimizer,
        )
        val_metrics = run_epoch(
            encoder,
            projector,
            val_loader,
            device,
            config,
            optimizer=None,
        )

        row = {
            "epoch": epoch,
            "train_total": train_metrics["total"],
            "train_contrastive": train_metrics["contrastive"],
            "train_temporal_unweighted": train_metrics["temporal_unweighted"],
            "val_total": val_metrics["total"],
            "val_contrastive": val_metrics["contrastive"],
            "val_temporal_unweighted": val_metrics["temporal_unweighted"],
        }
        history.append(row)
        save_history(fold_dir / "history.csv", history)

        log(
            f"Fold {fold_number}, epoch {epoch}: "
            f"train={train_metrics['total']:.6f}, "
            f"val={val_metrics['total']:.6f}, "
            f"val_contrastive={val_metrics['contrastive']:.6f}, "
            f"val_temporal={val_metrics['temporal_unweighted']:.6f}.",
            log_file,
        )

        if val_metrics["total"] < best_val_loss:
            best_val_loss = val_metrics["total"]
            best_epoch = epoch
            patience_counter = 0
            torch.save(encoder.state_dict(), fold_dir / "encoder_best.pth")
            torch.save(projector.state_dict(), fold_dir / "projector_best.pth")
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            log(
                f"Fold {fold_number}: early stopping at epoch {epoch}; "
                f"best epoch={best_epoch}, best validation loss={best_val_loss:.6f}.",
                log_file,
            )
            break

    plot_loss_curves(
        [row["train_total"] for row in history],
        [row["val_total"] for row in history],
        fold_dir / "loss_curve.png",
        title=f"Stage 1 - Fold {fold_number}",
    )

    summary = {
        "fold": fold_number,
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "train_subject_count": len(train_subjects),
        "validation_subject_count": len(val_subjects),
        "train_clip_count": len(train_samples),
        "validation_clip_count": len(val_samples),
    }
    with (fold_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return best_epoch, best_val_loss


def train_final_development_encoder(
    subjects: Sequence[str],
    subject_to_files: Dict[str, List[Path]],
    development_dir: Path,
    output_dir: Path,
    config: Stage1Config,
    device: torch.device,
    epochs: int,
    log_file: Path,
) -> None:
    """Train a transfer encoder on all development subjects without using test data."""
    final_dir = output_dir / "final_development_training"
    final_dir.mkdir(parents=True, exist_ok=True)

    samples = files_for_subjects(subject_to_files, subjects)
    loader = make_loader(
        development_dir,
        samples,
        config,
        shuffle=True,
        random_crop=True,
        seed_offset=9000,
    )

    set_reproducibility(config.seed)
    encoder, projector = build_models(config, device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(projector.parameters()),
        lr=config.learning_rate,
    )

    history: List[Dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            encoder,
            projector,
            loader,
            device,
            config,
            optimizer,
        )
        history.append(
            {
                "epoch": epoch,
                "train_total": train_metrics["total"],
                "train_contrastive": train_metrics["contrastive"],
                "train_temporal_unweighted": train_metrics["temporal_unweighted"],
            }
        )
        save_history(final_dir / "history.csv", history)
        log(
            f"Final development training, epoch {epoch}/{epochs}: "
            f"loss={train_metrics['total']:.6f}.",
            log_file,
        )

    torch.save(encoder.state_dict(), final_dir / "final_stage1_encoder.pth")
    torch.save(projector.state_dict(), final_dir / "final_stage1_projector.pth")

    # Compatibility filename for downstream code that expects the previous name.
    torch.save(encoder.state_dict(), output_dir / "final_stage1_encoder_last.pth")

    with (final_dir / "training_protocol.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "subjects": list(subjects),
                "fixed_test_subjects_excluded": sorted(FIXED_TEST_SUBJECTS),
                "epochs": int(epochs),
                "epoch_selection_rule": "rounded median of the ten validation-selected fold epochs",
            },
            handle,
            indent=2,
        )


def main() -> None:
    args = parse_args()
    development_dir = args.development_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = Stage1Config(num_workers=args.num_workers)
    log_file = output_dir / "training_log.txt"
    set_reproducibility(config.seed)

    if not development_dir.is_dir():
        raise FileNotFoundError(f"Development directory does not exist: {development_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subject_to_files = load_subject_inventory(development_dir)
    subjects = np.asarray(sorted(subject_to_files.keys()))

    if len(subjects) != 48:
        raise RuntimeError(
            f"Expected 48 development subjects after fixed test exclusion, found {len(subjects)}."
        )

    with (output_dir / "stage1_config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2)

    with (output_dir / "dataset_inventory.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "development_subjects": subjects.tolist(),
                "fixed_test_subjects_excluded": sorted(FIXED_TEST_SUBJECTS),
                "clips_per_subject": {
                    subject: len(subject_to_files[subject]) for subject in subjects
                },
            },
            handle,
            indent=2,
        )

    log(
        f"Stage 1 started on {device}. Found {len(subjects)} development subjects and "
        f"{sum(len(v) for v in subject_to_files.values())} clips. Fixed test subjects "
        "are excluded before cross-validation.",
        log_file,
    )

    # With 48 subjects, KFold produces eight 5-subject validation folds and two
    # 4-subject validation folds. The complementary training sets contain 43 or 44 subjects.
    kfold = KFold(
        n_splits=config.n_folds,
        shuffle=True,
        random_state=config.seed,
    )

    split_manifest: List[Dict[str, object]] = []
    best_epochs: List[int] = []
    fold_summaries: List[Dict[str, object]] = []

    for fold_number, (train_idx, val_idx) in enumerate(kfold.split(subjects), start=1):
        train_subjects = subjects[train_idx].tolist()
        val_subjects = subjects[val_idx].tolist()

        split_manifest.append(
            {
                "fold": fold_number,
                "train_subjects": train_subjects,
                "validation_subjects": val_subjects,
                "fixed_test_subjects_excluded": sorted(FIXED_TEST_SUBJECTS),
            }
        )

        best_epoch, best_loss = train_fold(
            fold_number,
            train_subjects,
            val_subjects,
            subject_to_files,
            development_dir,
            output_dir,
            config,
            device,
            log_file,
        )
        best_epochs.append(best_epoch)
        fold_summaries.append(
            {
                "fold": fold_number,
                "best_epoch": best_epoch,
                "best_validation_loss": best_loss,
            }
        )

    with (output_dir / "original_10fold_subject_splits.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(split_manifest, handle, indent=2)

    with (output_dir / "fold_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(fold_summaries, handle, indent=2)

    if not args.skip_final_training:
        selected_final_epochs = int(np.rint(np.median(best_epochs)))
        selected_final_epochs = max(1, min(selected_final_epochs, config.max_epochs))
        log(
            f"Final development-only Stage 1 training will use {selected_final_epochs} epochs, "
            "selected as the rounded median of the ten fold-specific best epochs. The fixed "
            "test subjects remain completely unused.",
            log_file,
        )
        train_final_development_encoder(
            subjects.tolist(),
            subject_to_files,
            development_dir,
            output_dir,
            config,
            device,
            selected_final_epochs,
            log_file,
        )

    log("Stage 1 completed successfully without loading the fixed test partition.", log_file)


if __name__ == "__main__":
    main()
