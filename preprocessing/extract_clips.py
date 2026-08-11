"""Preprocess UBFC-Phys recordings into subject-partitioned facial-video clips.

This script implements the clip-generation protocol reported for MS-CAM-Net:
- 35 fps input video
- 60 s clips (2100 frames)
- 30 s step (1050 frames; 50% overlap)
- dlib frontal-face detection with 68 landmarks
- forehead extension of 0.7 times the interocular distance
- RGB facial ROI resized to 64 x 64 pixels
- synchronized BVP extraction at 64 Hz

The fixed subject-independent test subjects are separated before clip generation.
Stage 1 must use only the generated ``development`` partition.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import dlib
import neurokit2 as nk
import numpy as np
import pandas as pd
from tqdm import tqdm


FPS_VIDEO = 35
FS_BVP = 64
WINDOW_SIZE = 2100
STEP_SIZE = 1050
IMG_SIZE = 64
FOREHEAD_SCALE = 0.7

FIXED_TEST_SUBJECTS = {
    "s1", "s10", "s11", "s12", "s13", "s14", "s15", "s16"
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate MS-CAM-Net clips from the UBFC-Phys dataset."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root directory containing the UBFC-Phys subject folders.",
    )
    parser.add_argument(
        "--predictor-path",
        type=Path,
        required=True,
        help="Path to shape_predictor_68_face_landmarks.dat.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Destination root. Separate development/ and test/ folders are created.",
    )
    parser.add_argument(
        "--partition",
        choices=("development", "test", "both"),
        default="both",
        help="Which subject partition to preprocess.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing clip files.",
    )
    return parser.parse_args()


def lexicographic_subjects(dataset_root: Path) -> List[str]:
    return sorted(
        [p.name for p in dataset_root.iterdir() if p.is_dir()]
    )


def find_task_file(
    subject_dir: Path,
    task: str,
    *,
    suffix: Optional[str] = None,
    required_token: Optional[str] = None,
) -> Optional[Path]:
    task_upper = task.upper()
    for path in sorted(subject_dir.iterdir()):
        if not path.is_file():
            continue
        name_upper = path.name.upper()
        if task_upper not in name_upper:
            continue
        if suffix is not None and path.suffix.lower() != suffix.lower():
            continue
        if required_token is not None and required_token.upper() not in name_upper:
            continue
        return path
    return None


def read_signal(path: Path) -> np.ndarray:
    return pd.read_csv(path, header=None).values.flatten().astype(np.float32)


def compute_hr_and_hrv_rmssd(bvp_window: np.ndarray) -> Tuple[float, float]:
    """Compute clip-level HR and RMSSD from the reference BVP.

    These targets are retained for the supervised physiological stage. They are
    not used by Stage 1 self-supervised pre-training.
    """
    if bvp_window.size < FS_BVP * 5:
        return float("nan"), float("nan")

    try:
        peaks = nk.ppg_findpeaks(
            bvp_window,
            sampling_rate=FS_BVP,
        )["PPG_Peaks"]

        if len(peaks) < 2:
            return float("nan"), float("nan")

        rate = nk.signal_rate(
            peaks,
            sampling_rate=FS_BVP,
            desired_length=len(bvp_window),
        )
        hr = float(np.nanmean(rate))

        hrv_table = nk.hrv_time(peaks, sampling_rate=FS_BVP, show=False)
        hrv = float(hrv_table["HRV_RMSSD"].iloc[0])
        return hr, hrv
    except Exception:
        return float("nan"), float("nan")


class FacialROIExtractor:
    """Track and crop the facial ROI while processing one recording sequentially."""

    def __init__(self, predictor_path: Path) -> None:
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(str(predictor_path))
        self.last_valid_box: Optional[Tuple[int, int, int, int]] = None

    def reset(self) -> None:
        self.last_valid_box = None

    def _detect_box(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)
        if len(rects) == 0:
            return None

        shape = self.predictor(gray, rects[0])
        points = np.asarray(
            [[shape.part(i).x, shape.part(i).y] for i in range(68)],
            dtype=np.float32,
        )

        left_eye = points[36:42].mean(axis=0)
        right_eye = points[42:48].mean(axis=0)
        interocular_distance = float(np.linalg.norm(right_eye - left_eye))

        x_min = int(np.floor(points[:, 0].min()))
        x_max = int(np.ceil(points[:, 0].max()))
        y_min = int(np.floor(points[:, 1].min()))
        y_max = int(np.ceil(points[:, 1].max()))

        height, width = frame.shape[:2]
        x_min = max(0, min(x_min, width - 1))
        x_max = max(x_min + 1, min(x_max, width))
        top_y = max(0, int(round(y_min - FOREHEAD_SCALE * interocular_distance)))
        y_max = max(top_y + 1, min(y_max, height))

        return x_min, top_y, x_max, y_max

    @staticmethod
    def _crop_resize(
        frame: np.ndarray,
        box: Tuple[int, int, int, int],
    ) -> Optional[np.ndarray]:
        x_min, y_min, x_max, y_max = box
        roi = frame[y_min:y_max, x_min:x_max]
        if roi.size == 0:
            return None
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        return cv2.resize(roi, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        detected_box = self._detect_box(frame)
        if detected_box is not None:
            self.last_valid_box = detected_box

        if self.last_valid_box is None:
            return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

        roi = self._crop_resize(frame, self.last_valid_box)
        if roi is None:
            return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        return roi.astype(np.uint8, copy=False)


def preprocess_video(video_path: Path, roi_extractor: FacialROIExtractor) -> np.ndarray:
    """Process a complete recording sequentially so failed detections can reuse the last ROI."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    roi_extractor.reset()
    frames: List[np.ndarray] = []

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(roi_extractor.process_frame(frame))
    finally:
        capture.release()

    if not frames:
        raise RuntimeError(f"No frames were decoded from: {video_path}")

    return np.stack(frames, axis=0)


def bvp_slice_for_video_window(
    bvp: np.ndarray,
    start_frame: int,
    end_frame: int,
) -> np.ndarray:
    start_idx = int(start_frame * FS_BVP / FPS_VIDEO)
    end_idx = int(end_frame * FS_BVP / FPS_VIDEO)
    return bvp[start_idx:end_idx].astype(np.float32, copy=False)


def subject_partition(subject: str) -> str:
    return "test" if subject in FIXED_TEST_SUBJECTS else "development"


def should_process(partition: str, requested: str) -> bool:
    return requested == "both" or requested == partition


def save_clip(
    destination: Path,
    video_clip: np.ndarray,
    bvp_window: np.ndarray,
    subject: str,
    task: str,
    start_frame: int,
    overwrite: bool,
) -> bool:
    if destination.exists() and not overwrite:
        return False

    hr, hrv = compute_hr_and_hrv_rmssd(bvp_window)

    clip_data: Dict[str, object] = {
        "video": video_clip.astype(np.uint8, copy=False),
        "signals": {
            "bvp": bvp_window.astype(np.float32, copy=False),
        },
        "labels": {
            "hr": hr,
            "hrv": hrv,
        },
        "meta": {
            "sub": subject,
            "task": task,
            "start_frame": int(start_frame),
            "fps_video": FPS_VIDEO,
            "fs_bvp": FS_BVP,
        },
    }

    np.save(destination, clip_data, allow_pickle=True)
    return True


def write_manifest(output_root: Path, subjects: Sequence[str]) -> None:
    manifest = {
        "fps_video": FPS_VIDEO,
        "fs_bvp": FS_BVP,
        "window_size_frames": WINDOW_SIZE,
        "step_size_frames": STEP_SIZE,
        "window_seconds": WINDOW_SIZE / FPS_VIDEO,
        "step_seconds": STEP_SIZE / FPS_VIDEO,
        "overlap_fraction": 1.0 - STEP_SIZE / WINDOW_SIZE,
        "roi_size": [IMG_SIZE, IMG_SIZE],
        "forehead_extension_interocular": FOREHEAD_SCALE,
        "fixed_test_subjects": sorted(FIXED_TEST_SUBJECTS),
        "all_detected_subjects": list(subjects),
        "development_subjects": [s for s in subjects if s not in FIXED_TEST_SUBJECTS],
        "test_subjects": [s for s in subjects if s in FIXED_TEST_SUBJECTS],
    }
    with (output_root / "preprocessing_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def main() -> None:
    args = parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    predictor_path = args.predictor_path.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    if not predictor_path.is_file():
        raise FileNotFoundError(f"Landmark predictor does not exist: {predictor_path}")

    development_dir = output_root / "development"
    test_dir = output_root / "test"
    development_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    subjects = lexicographic_subjects(dataset_root)
    write_manifest(output_root, subjects)

    roi_extractor = FacialROIExtractor(predictor_path)
    written = 0
    skipped_existing = 0
    skipped_recordings = 0

    for subject in subjects:
        partition = subject_partition(subject)
        if not should_process(partition, args.partition):
            continue

        subject_dir = dataset_root / subject
        destination_dir = test_dir if partition == "test" else development_dir

        for task in ("T1", "T2", "T3"):
            video_path = find_task_file(subject_dir, task, suffix=".avi")
            bvp_path = find_task_file(subject_dir, task, required_token="BVP")

            if video_path is None or bvp_path is None:
                skipped_recordings += 1
                print(
                    f"Skipping {subject}/{task}: required video or BVP file was not found."
                )
                continue

            try:
                processed_video = preprocess_video(video_path, roi_extractor)
                bvp = read_signal(bvp_path)
            except Exception as exc:
                skipped_recordings += 1
                print(f"Skipping {subject}/{task}: {exc}")
                continue

            total_frames = processed_video.shape[0]
            starts = range(0, total_frames - WINDOW_SIZE + 1, STEP_SIZE)
            for start_frame in tqdm(
                starts,
                desc=f"{partition}: {subject}/{task}",
                leave=False,
            ):
                end_frame = start_frame + WINDOW_SIZE
                clip = processed_video[start_frame:end_frame]
                bvp_window = bvp_slice_for_video_window(bvp, start_frame, end_frame)

                filename = f"{subject}_{task}_{start_frame}.npy"
                destination = destination_dir / filename
                if save_clip(
                    destination,
                    clip,
                    bvp_window,
                    subject,
                    task,
                    start_frame,
                    args.overwrite,
                ):
                    written += 1
                else:
                    skipped_existing += 1

    print("Preprocessing completed.")
    print(f"Written clips: {written}")
    print(f"Existing clips skipped: {skipped_existing}")
    print(f"Recordings skipped because of missing/invalid input: {skipped_recordings}")
    print(f"Development clips: {development_dir}")
    print(f"Test clips: {test_dir}")


if __name__ == "__main__":
    main()
