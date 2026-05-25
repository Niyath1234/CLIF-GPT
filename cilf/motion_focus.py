"""Subject-focused priors for object slot tracking.

Static backgrounds (walls, floor) stay still; the person (face, hands) moves and
usually sits near the center. This module builds a per-patch weight map that
downweights static regions so slot attention binds to the subject, not the
whole scene.
"""

from __future__ import annotations

import cv2
import numpy as np


class SubjectFocusPrior:
    """Motion + center + detail prior over the vision patch grid."""

    def __init__(
        self,
        model_size: int = 160,
        center_sigma: float = 0.42,
        motion_gain: float = 3.0,
        detail_gain: float = 1.5,
        static_floor: float = 0.08,
        warmup_frames: int = 8,
    ) -> None:
        self.model_size = int(model_size)
        self.center_sigma = float(center_sigma)
        self.motion_gain = float(motion_gain)
        self.detail_gain = float(detail_gain)
        self.static_floor = float(static_floor)
        self.warmup_frames = int(warmup_frames)

        self._prev_gray: np.ndarray | None = None
        self._bg_median: np.ndarray | None = None
        self._warmup_buffer: list[np.ndarray] = []
        self._center_grid: np.ndarray | None = None

    def _gray_at_model_size(self, bgr: np.ndarray) -> np.ndarray:
        small = cv2.resize(bgr, (self.model_size, self.model_size), interpolation=cv2.INTER_LINEAR)
        return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    def _center_prior(self, grid_hw: tuple[int, int]) -> np.ndarray:
        if self._center_grid is not None and self._center_grid.shape == grid_hw:
            return self._center_grid
        grid_h, grid_w = grid_hw
        yy, xx = np.mgrid[0:grid_h, 0:grid_w].astype(np.float32)
        cy = (grid_h - 1) * 0.5
        cx = (grid_w - 1) * 0.5
        dist_sq = ((yy - cy) / max(grid_h, 1)) ** 2 + ((xx - cx) / max(grid_w, 1)) ** 2
        sigma = self.center_sigma
        center = np.exp(-dist_sq / (2.0 * sigma * sigma))
        self._center_grid = center
        return center

    def update(self, bgr: np.ndarray, grid_hw: tuple[int, int]) -> np.ndarray:
        """Return subject weights [grid_h, grid_w] in [0, 1]."""

        gray = self._gray_at_model_size(bgr)
        grid_h, grid_w = grid_hw

        self._warmup_buffer.append(gray)
        if len(self._warmup_buffer) > self.warmup_frames:
            self._warmup_buffer.pop(0)
        if self._bg_median is None and len(self._warmup_buffer) >= self.warmup_frames:
            stack = np.stack(self._warmup_buffer, axis=0)
            self._bg_median = np.median(stack, axis=0)

        if self._prev_gray is None:
            self._prev_gray = gray
            return self._center_prior(grid_hw)

        frame_diff = np.abs(gray - self._prev_gray)
        if self._bg_median is not None:
            bg_diff = np.abs(gray - self._bg_median)
            motion_raw = 0.65 * frame_diff + 0.35 * bg_diff
        else:
            motion_raw = frame_diff
        self._prev_gray = gray

        laplacian = cv2.Laplacian((gray * 255.0).astype(np.uint8), cv2.CV_32F)
        detail_raw = np.abs(laplacian).astype(np.float32)
        detail_raw = detail_raw / (detail_raw.max() + 1e-6)

        motion_grid = cv2.resize(motion_raw, (grid_w, grid_h), interpolation=cv2.INTER_AREA)
        detail_grid = cv2.resize(detail_raw, (grid_w, grid_h), interpolation=cv2.INTER_AREA)
        center_grid = self._center_prior(grid_hw)

        motion_norm = motion_grid / (motion_grid.max() + 1e-6)
        detail_norm = detail_grid / (detail_grid.max() + 1e-6)

        subject = (
            (motion_norm ** 0.7) ** self.motion_gain
            * (0.55 + 0.45 * detail_norm) ** self.detail_gain
            * (0.35 + 0.65 * center_grid)
        )
        subject = subject / (subject.max() + 1e-6)
        subject = np.clip(subject, self.static_floor, 1.0)
        return subject.astype(np.float32)

    def patch_weights(self, bgr: np.ndarray, grid_hw: tuple[int, int]) -> np.ndarray:
        """Flattened weights [num_patches] aligned with vision patch order."""

        grid = self.update(bgr, grid_hw)
        return grid.reshape(-1)

    @staticmethod
    def mean_weight_in_box(
        subject_grid: np.ndarray,
        box: tuple[int, int, int, int],
        image_hw: tuple[int, int],
    ) -> float:
        height, width = image_hw
        grid_h, grid_w = subject_grid.shape
        x1, y1, x2, y2 = box
        patch_h = height / grid_h
        patch_w = width / grid_w
        gi0 = int(np.floor(y1 / patch_h))
        gi1 = int(np.ceil(y2 / patch_h))
        gj0 = int(np.floor(x1 / patch_w))
        gj1 = int(np.ceil(x2 / patch_w))
        gi0 = max(0, min(grid_h - 1, gi0))
        gi1 = max(gi0 + 1, min(grid_h, gi1))
        gj0 = max(0, min(grid_w - 1, gj0))
        gj1 = max(gj0 + 1, min(grid_w, gj1))
        region = subject_grid[gi0:gi1, gj0:gj1]
        return float(region.mean()) if region.size else 0.0
