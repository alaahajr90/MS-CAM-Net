"""Five-fold complete-session physiological validation for MS-CAM-Net Stage 2."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from dataset_stage2 import Stage2NPYDataset, discover_subjects
from stage2_model import Stage2PhysiologicalModel
from utils.physiology import (
    artifact_masked_nrmse,
    cardiac_band_coherence,
    cardiac_band_snr_db,
    hr_from_waveform,
    lag_aware_pearson,
    polarity_corrected_pearson,
    rmssd_from_waveform,
    safe_pearson,
    summarize,
    weighted_overlap_add,
)

FS = 35.0
WINDOW_SECONDS = 20
STEP_SECONDS = 10


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-dir", required=True, type=Path)
    parser.add_argument("--stage1-dir", required=True, type=Path)
    parser.add_argument("--stage1-weights", required=True, type=Path)
    parser.add_argument("--fold-root", required=True, type=Path,
                        help="Directory produced by train_stage2.py --num-folds 5.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def import_encoder(stage1_dir: Path):
    sys.path.insert(0, str(stage1_dir.resolve()))
    from model_stage1 import Encoder  # pylint: disable=import-outside-toplevel
    return Encoder


def parse_start_frame(name: str) -> int:
    match = re.search(r"_(\d+)$", name)
    return int(match.group(1)) if match else 0


def session_id(name: str) -> str:
    parts = name.split("_")
    return "_".join(parts[:2])


def twenty_second_predictions(model, clip, device):
    total_frames = clip.shape[2]
    win = int(WINDOW_SECONDS * FS)
    step = int(STEP_SECONDS * FS)
    starts = list(range(0, max(total_frames - win + 1, 1), step))
    if not starts or starts[-1] + win < total_frames:
        starts.append(max(0, total_frames - win))

    pred_windows = []
    direct_hr = []
    direct_hrv = []
    for start in sorted(set(starts)):
        end = min(start + win, total_frames)
        segment = clip[:, :, start:end, :, :].to(device)
        output = model(segment)
        pred_windows.append((start, output["rppg"][0].detach().cpu().numpy()))
        direct_hr.append(float(output["hr"][0].detach().cpu()))
        direct_hrv.append(float(output["hrv"][0].detach().cpu()))

    reconstructed = weighted_overlap_add(pred_windows, total_frames)
    return reconstructed, direct_hr, direct_hrv


def evaluate_fold(model, loader, device):
    sessions = defaultdict(lambda: {
        "pred_windows": [], "gt_windows": [], "direct_hr": [], "direct_hrv": [],
        "gt_hr": [], "gt_hrv": [], "max_end": 0,
    })

    model.eval()
    with torch.no_grad():
        for batch in loader:
            name = batch["name"][0]
            sid = session_id(name)
            start = parse_start_frame(name)
            clip = batch["clip"]
            pred_clip, hr_values, hrv_values = twenty_second_predictions(model, clip, device)
            gt_clip = batch["rppg"][0].numpy()
            end = start + len(pred_clip)

            record = sessions[sid]
            record["pred_windows"].append((start, pred_clip))
            record["gt_windows"].append((start, gt_clip))
            record["direct_hr"].extend(hr_values)
            record["direct_hrv"].extend(hrv_values)
            record["gt_hr"].append(float(batch["hr"][0]))
            record["gt_hrv"].append(float(batch["hrv"][0]))
            record["max_end"] = max(record["max_end"], end)

    rows = []
    for sid, record in sorted(sessions.items()):
        pred = weighted_overlap_add(record["pred_windows"], record["max_end"])
        ref = weighted_overlap_add(record["gt_windows"], record["max_end"])
        direct_hr = float(np.median(record["direct_hr"]))
        direct_hrv = float(np.median(record["direct_hrv"]))
        gt_hr = float(np.median(record["gt_hr"]))
        gt_hrv = float(np.median(record["gt_hrv"]))

        rows.append({
            "session": sid,
            "direct_hr_pred": direct_hr,
            "direct_hr_gt": gt_hr,
            "direct_hrv_pred": direct_hrv,
            "direct_hrv_gt": gt_hrv,
            "raw_pearson": safe_pearson(pred, ref),
            "polarity_corrected_pearson": polarity_corrected_pearson(pred, ref),
            "lag_aware_pearson": lag_aware_pearson(pred, ref),
            "artifact_masked_nrmse": artifact_masked_nrmse(pred, ref),
            "spectral_snr_db": cardiac_band_snr_db(pred),
            "spectral_coherence": cardiac_band_coherence(pred, ref),
            "hr_from_rppg": hr_from_waveform(pred),
            "hr_from_ref": hr_from_waveform(ref),
            "rmssd_from_rppg": rmssd_from_waveform(pred),
            "rmssd_from_ref": rmssd_from_waveform(ref),
        })
    return rows


def mae(values_a, values_b):
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    return float(np.mean(np.abs(a[mask] - b[mask]))) if np.any(mask) else float("nan")


def rmse(values_a, values_b):
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[mask] - b[mask]) ** 2))) if np.any(mask) else float("nan")


def correlation(values_a, values_b):
    return safe_pearson(np.asarray(values_a, dtype=float), np.asarray(values_b, dtype=float))


def fold_metrics(rows):
    hr_p = [r["direct_hr_pred"] for r in rows]
    hr_t = [r["direct_hr_gt"] for r in rows]
    hrv_p = [r["direct_hrv_pred"] for r in rows]
    hrv_t = [r["direct_hrv_gt"] for r in rows]
    whr_p = [r["hr_from_rppg"] for r in rows]
    whr_t = [r["hr_from_ref"] for r in rows]
    wrm_p = [r["rmssd_from_rppg"] for r in rows]
    wrm_t = [r["rmssd_from_ref"] for r in rows]
    return {
        "direct_hr_mae": mae(hr_p, hr_t),
        "direct_hr_rmse": rmse(hr_p, hr_t),
        "direct_hr_pearson": correlation(hr_p, hr_t),
        "direct_hrv_mae": mae(hrv_p, hrv_t),
        "direct_hrv_rmse": rmse(hrv_p, hrv_t),
        "direct_hrv_pearson": correlation(hrv_p, hrv_t),
        "lag_aware_pearson": float(np.nanmean([r["lag_aware_pearson"] for r in rows])),
        "artifact_masked_nrmse": float(np.nanmean([r["artifact_masked_nrmse"] for r in rows])),
        "spectral_snr_db": float(np.nanmean([r["spectral_snr_db"] for r in rows])),
        "spectral_coherence": float(np.nanmean([r["spectral_coherence"] for r in rows])),
        "hr_from_rppg_mae": mae(whr_p, whr_t),
        "hr_from_rppg_pearson": correlation(whr_p, whr_t),
        "rmssd_from_rppg_mae": mae(wrm_p, wrm_t),
        "rmssd_from_rppg_pearson": correlation(wrm_p, wrm_t),
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Encoder = import_encoder(args.stage1_dir)
    subjects = np.asarray(discover_subjects(args.development_dir))
    if len(subjects) != 48:
        raise RuntimeError(f"Expected 48 development subjects, found {len(subjects)}")

    splitter = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    all_fold_metrics = []
    all_session_rows = []
    for fold_idx, (_, val_idx) in enumerate(splitter.split(subjects), start=1):
        checkpoint = args.fold_root / f"fold_{fold_idx:02d}" / "best_stage2_model.pth"
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Missing {checkpoint}. Train a five-fold Stage 2 run first with --num-folds 5."
            )
        encoder = Encoder().to(device)
        encoder.load_state_dict(torch.load(args.stage1_weights, map_location=device, weights_only=True))
        model = Stage2PhysiologicalModel(encoder).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))

        val_subjects = subjects[val_idx].tolist()
        loader = DataLoader(
            Stage2NPYDataset(args.development_dir, val_subjects),
            batch_size=1, shuffle=False, num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        rows = evaluate_fold(model, loader, device)
        metrics = fold_metrics(rows)
        metrics["fold"] = fold_idx
        all_fold_metrics.append(metrics)
        for row in rows:
            row["fold"] = fold_idx
            all_session_rows.append(row)

    with (args.output_dir / "session_level_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_session_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_session_rows)

    metric_names = [key for key in all_fold_metrics[0] if key != "fold"]
    summary = {}
    for key in metric_names:
        mean, sd = summarize([fold[key] for fold in all_fold_metrics])
        summary[key] = {"mean": mean, "sample_sd": sd}

    (args.output_dir / "five_fold_summary.json").write_text(
        json.dumps({"folds": all_fold_metrics, "summary": summary}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
