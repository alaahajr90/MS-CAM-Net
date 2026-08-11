"""Dataset utilities for MS-CAM-Net Stage 3 stress classification."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

PathLike = Union[str, Path]


class Stage3NPYDataset(Dataset):
    """Load facial clips, stress labels, and synchronized BVP targets.

    Stress labels follow the UBFC-Phys task mapping used in the manuscript:
    T1 -> Rest, T2 -> Low Stress, and T3 -> High Stress. The binary stress-state
    target is Rest versus Stress, where T2 and T3 are merged into Stress.
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
        self.samples: list[Path] = []
        for path in sorted(self.data_dir.glob("*.npy")):
            item = np.load(path, allow_pickle=True).item()
            subject = str(item["meta"]["sub"])
            if allowed is None or subject in allowed:
                self.samples.append(path)

        if not self.samples:
            raise RuntimeError(f"No Stage 3 samples were found in {self.data_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _labels_from_task(task: str) -> tuple[int, int]:
        task = task.upper()
        if task == "T1":
            return 0, 0
        if task == "T2":
            return 1, 1
        if task == "T3":
            return 1, 2
        raise ValueError(f"Unsupported UBFC-Phys task label: {task}")

    @staticmethod
    def _resample_bvp(bvp: np.ndarray, n_frames: int) -> torch.Tensor:
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

    def __getitem__(self, index: int) -> dict:
        path = self.samples[index]
        item = np.load(path, allow_pickle=True).item()

        video = np.asarray(item["video"], dtype=np.float32)
        if video.ndim != 4 or video.shape[-1] != 3:
            raise ValueError(f"Expected [T,H,W,3] video in {path}, got {video.shape}")
        video = torch.from_numpy(np.ascontiguousarray(video.transpose(3, 0, 1, 2)))

        meta = item.get("meta", {})
        subject = str(meta.get("sub", ""))
        task = str(meta.get("task", "")).upper()
        stress_state, stress_level = self._labels_from_task(task)

        signals = item.get("signals", {})
        if "bvp" not in signals:
            raise KeyError(f"Missing synchronized BVP target in {path}")
        rppg_target = self._resample_bvp(signals["bvp"], video.shape[1])

        return {
            "clip": video,
            "state": torch.tensor(stress_state, dtype=torch.long),
            "level": torch.tensor(stress_level, dtype=torch.long),
            "rppg": rppg_target,
            "name": path.stem,
            "subject": subject,
            "task": task,
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
