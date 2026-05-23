"""Video decoding helpers (torchvision 0.27+ removed read_video)."""

from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import torch


def read_video(path: str | Path, target_fps: int = 10) -> tuple[torch.Tensor, float]:
    """Return frames as float tensor [T, C, H, W] in 0..1 and native fps."""
    path = Path(path)
    plugin = "pyav"
    try:
        meta = iio.immeta(path, plugin=plugin)
    except Exception:
        plugin = "ffmpeg"
        meta = iio.immeta(path, plugin=plugin)
    native_fps = float(meta.get("fps") or target_fps)
    frames = iio.imread(path, plugin=plugin)
    if frames.ndim == 3:
        frames = frames[None, ...]
    video = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    return video, native_fps
