"""Dataset utilities for MS-CAM-Net Stage 2 physiological refinement."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

PathLike = Union[str, Path]

VIDEO_FS = 35.0
BVP_FS = 64.0


class Stage2NPYDataset(Dataset):
    """Load preprocessed clips with synchronized BVP, HR, and HRV targets.

    The dataset expects files produced by the repository preprocessing script.
    Each file contains a 60-s RGB facial clip, the synchronized BVP signal,
    physiological targets, and subject/task metadata.
    """

    def __init__(
        self,
        data_dir: PathLike,
        subject_ids: Optional[Sequence[str]] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory does not exist: {self.data_dir}")

        allowed = set(subject_ids) if subject_ids is not None else None
        self.samples = []
        for path in sorted(self.data_dir.glob("*.npy")):
            try:
                item = np.load(path, allow_pickle=True).item()
                subject = str(item["meta"]["sub"])
                if allowed is None or subject in allowed:
                    self.samples.append(path)
            except Exception as exc:
                raise RuntimeError(f"Failed to inspect {path}: {exc}") from exc

        if not self.samples:
            raise RuntimeError(f"No Stage 2 samples were found in {self.data_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _resample_bvp_to_video_frames(bvp: np.ndarray, n_frames: int) -> torch.Tensor:
        signal = torch.as_tensor(np.asarray(bvp, dtype=np.float32).reshape(-1))
        if signal.numel() < 2:
            raise ValueError("BVP target must contain at least two samples.")
        signal = F.interpolate(
            signal.view(1, 1, -1),
            size=n_frames,
            mode="linear",
            align_corners=False,
        ).view(-1)
        signal = (signal - signal.mean()) / (signal.std(unbiased=False) + 1e-8)
        return signal

    def __getitem__(self, index: int):
        path = self.samples[index]
        item = np.load(path, allow_pickle=True).item()

        video = np.asarray(item["video"], dtype=np.float32)
        if video.ndim != 4 or video.shape[-1] != 3:
            raise ValueError(f"Expected [T,H,W,3] video in {path}, got {video.shape}")
        video = torch.from_numpy(np.ascontiguousarray(video.transpose(3, 0, 1, 2)))

        signals = item.get("signals", {})
        if "bvp" not in signals:
            raise KeyError(f"Missing BVP target in {path}")
        rppg_target = self._resample_bvp_to_video_frames(signals["bvp"], video.shape[1])

        labels = item.get("labels", {})
        hr = float(labels.get("hr", np.nan))
        hrv = float(labels.get("hrv", np.nan))
        if not np.isfinite(hr) or not np.isfinite(hrv):
            raise ValueError(f"Invalid HR/HRV target in {path}")

        meta = item.get("meta", {})
        return {
            "clip": video,
            "rppg": rppg_target,
            "hr": torch.tensor(hr, dtype=torch.float32),
            "hrv": torch.tensor(hrv, dtype=torch.float32),
            "name": path.stem,
            "subject": str(meta.get("sub", "")),
            "task": str(meta.get("task", "")),
        }


def discover_subjects(data_dir: PathLike) -> list[str]:
    """Return lexicographically sorted subject IDs contained in a clip directory."""
    subjects = set()
    for path in sorted(Path(data_dir).glob("*.npy")):
        item = np.load(path, allow_pickle=True).item()
        subjects.add(str(item["meta"]["sub"]))
    if not subjects:
        raise RuntimeError(f"No subjects were found in {data_dir}")
    return sorted(subjects)
