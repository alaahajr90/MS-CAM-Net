"""Dataset utilities for MS-CAM-Net Stage 1 self-supervised pre-training."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset


PathLike = Union[str, Path]


class Stage1NPYDataset(Dataset):
    """Load preprocessed facial-video clips for self-supervised Stage 1.

    Stage 1 intentionally ignores stress labels and physiological targets. Each
    sample returns two copies of the same 128-frame temporal crop. The projection
    head remains stochastic during training through dropout, while no color,
    brightness, hue, saturation, or frame-order perturbation is applied.
    """

    def __init__(
        self,
        data_dir: PathLike,
        clip_len: int = 128,
        samples: Optional[Sequence[PathLike]] = None,
        random_crop: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.clip_len = int(clip_len)
        self.random_crop = bool(random_crop)

        if self.clip_len <= 0:
            raise ValueError("clip_len must be positive.")

        if samples is None:
            self.samples = sorted(self.data_dir.glob("*.npy"))
        else:
            self.samples = sorted(Path(path) for path in samples)

        if not self.samples:
            raise RuntimeError(f"No .npy clips were found in {self.data_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def _temporal_crop(self, video: np.ndarray) -> np.ndarray:
        if video.ndim != 4 or video.shape[-1] != 3:
            raise ValueError(
                f"Expected video with shape [T, H, W, 3], got {video.shape}."
            )

        total_frames = int(video.shape[0])
        if total_frames < self.clip_len:
            raise ValueError(
                f"Clip contains {total_frames} frames, fewer than the required "
                f"{self.clip_len} frames."
            )

        max_start = total_frames - self.clip_len
        if self.random_crop and max_start > 0:
            start = random.randint(0, max_start)
        else:
            start = max_start // 2

        return video[start : start + self.clip_len]

    @staticmethod
    def _to_tensor_layout(clip: np.ndarray) -> np.ndarray:
        # [T, H, W, C] -> [C, T, H, W]
        return np.ascontiguousarray(
            clip.transpose(3, 0, 1, 2),
            dtype=np.float32,
        )

    @staticmethod
    def _make_view(clip: np.ndarray) -> np.ndarray:
        """Return an unchanged view to preserve temporal chromatic information."""
        return clip.copy()

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path = self.samples[index]
        data = np.load(path, allow_pickle=True).item()

        if "video" not in data:
            raise KeyError(f"Missing 'video' entry in {path}")

        crop = self._temporal_crop(np.asarray(data["video"]))
        crop = self._to_tensor_layout(crop)

        view1 = self._make_view(crop)
        view2 = self._make_view(crop)

        return torch.from_numpy(view1), torch.from_numpy(view2)
