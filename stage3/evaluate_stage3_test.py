"""Evaluate a trained MS-CAM-Net Stage 3 model once on the fixed test cohort."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset_stage3 import Stage3NPYDataset, discover_subjects
from metrics_stage3 import classification_metrics, confusion
from model_stage3 import StressRecognitionModel

FIXED_TEST_SUBJECTS = ("s1", "s10", "s11", "s12", "s13", "s14", "s15", "s16")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--stage1-dir", type=Path, required=True)
    parser.add_argument("--stage2-dir", type=Path, required=True)
    parser.add_argument("--stage2-weights", type=Path, required=True)
    parser.add_argument("--stage3-weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def import_class(module_path: Path, module_name: str, class_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {class_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def build_model(args, device):
    if str(args.stage1_dir) not in sys.path:
        sys.path.insert(0, str(args.stage1_dir))
    if str(args.stage2_dir) not in sys.path:
        sys.path.insert(0, str(args.stage2_dir))
    Encoder = import_class(args.stage1_dir / "model_stage1.py", "eval_stage1_model", "Encoder")
    Stage2Model = import_class(args.stage2_dir / "stage2_model.py", "eval_stage2_model", "Stage2PhysiologicalModel")
    stage2 = Stage2Model(Encoder())
    stage2_state = torch.load(args.stage2_weights, map_location="cpu")
    if isinstance(stage2_state, dict) and "state_dict" in stage2_state:
        stage2_state = stage2_state["state_dict"]
    stage2.load_state_dict(stage2_state, strict=True)
    model = StressRecognitionModel(stage2)
    stage3_state = torch.load(args.stage3_weights, map_location="cpu")
    if isinstance(stage3_state, dict) and "state_dict" in stage3_state:
        stage3_state = stage3_state["state_dict"]
    model.load_state_dict(stage3_state, strict=True)
    return model.to(device)


@torch.no_grad()
def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    subjects = discover_subjects(args.test_dir)
    if tuple(subjects) != FIXED_TEST_SUBJECTS:
        raise RuntimeError(
            "The test directory must contain exactly the fixed subject-independent test cohort: "
            + ", ".join(FIXED_TEST_SUBJECTS)
        )

    loader = DataLoader(
        Stage3NPYDataset(args.test_dir, FIXED_TEST_SUBJECTS),
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
    )
    model = build_model(args, device)
    model.eval()

    state_true, state_pred, state_prob = [], [], []
    level_true, level_pred, level_prob = [], [], []
    shared_features = []
    sample_names = []

    for batch in loader:
        outputs = model(batch["clip"].to(device), return_features=True)
        state_p = torch.softmax(outputs["state"], dim=1)
        level_p = torch.softmax(outputs["level"], dim=1)
        state_true.extend(batch["state"].numpy())
        state_pred.extend(state_p.argmax(dim=1).cpu().numpy())
        state_prob.extend(state_p.cpu().numpy())
        level_true.extend(batch["level"].numpy())
        level_pred.extend(level_p.argmax(dim=1).cpu().numpy())
        level_prob.extend(level_p.cpu().numpy())
        shared_features.extend(outputs["shared_features"].cpu().numpy())
        sample_names.extend(batch["name"])

    results = {
        "subjects": subjects,
        "n_clips": len(state_true),
        "stress_state": classification_metrics(state_true, state_pred, state_prob),
        "stress_level": classification_metrics(level_true, level_pred, level_prob),
        "confusion_state": confusion(state_true, state_pred, labels=[0, 1]),
        "confusion_level": confusion(level_true, level_pred, labels=[0, 1, 2]),
    }
    (args.output_dir / "test_metrics.json").write_text(
        json.dumps(results, indent=2, allow_nan=True), encoding="utf-8"
    )
    np.savez_compressed(
        args.output_dir / "test_predictions_and_features.npz",
        names=np.asarray(sample_names),
        state_true=np.asarray(state_true),
        state_pred=np.asarray(state_pred),
        state_prob=np.asarray(state_prob),
        level_true=np.asarray(level_true),
        level_pred=np.asarray(level_pred),
        level_prob=np.asarray(level_prob),
        shared_features=np.asarray(shared_features),
    )

    print(json.dumps(results, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
