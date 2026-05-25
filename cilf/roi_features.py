"""ROI pooling from vision-backbone patch features.

Bridges detector-produced bounding boxes onto the same SigLIP/ViT patch grid
that the rest of the model consumes. For every (sample, frame, track) triple we
collect a per-track feature vector by area-weighted averaging the patch tokens
that intersect that box.

This is intentionally minimal -- no learned interpolation, no sampling grid --
because the resulting embedding gets projected into ``state_dim`` immediately
and then handed off to the per-object dynamics / relation modules. The point is
to give those modules the *right object*, not to win at fine-grained
recognition.

Conventions
-----------
* Boxes are stored in normalized image coordinates ``[x1, y1, x2, y2]`` with
  ``0 <= x1 < x2 <= 1`` and ``0 <= y1 < y2 <= 1`` and origin at the top-left.
* Invalid tracks (object not visible on a frame) carry ``mask = 0`` and are
  zeroed out after pooling.
"""

from __future__ import annotations

import math

import torch


def _patch_grid_from_count(num_patches: int) -> tuple[int, int]:
    """Best-effort recovery of (H, W) when the backbone returns a flat sequence.

    SigLIP / ViT backbones return either ``num_patches`` purely (square grid),
    or include a leading CLS token (``num_patches + 1``). We strip the CLS token
    in the caller; here we just take the closest square factorization.
    """

    side = int(round(math.sqrt(num_patches)))
    if side * side == num_patches:
        return side, side
    for h in range(side, 0, -1):
        if num_patches % h == 0:
            return h, num_patches // h
    return 1, num_patches


def _normalise_patches(patches: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
    """Strip the CLS token if present and infer the patch grid shape."""

    batch_size, frames, num_patches, feature_dim = patches.shape
    grid_hw = _patch_grid_from_count(num_patches)
    if grid_hw[0] * grid_hw[1] == num_patches:
        return patches, grid_hw

    if num_patches > 1:
        candidate_hw = _patch_grid_from_count(num_patches - 1)
        if candidate_hw[0] * candidate_hw[1] == num_patches - 1:
            return patches[..., 1:, :], candidate_hw

    raise ValueError(
        f"Cannot reshape {num_patches} patch tokens into a 2D grid (feature_dim={feature_dim})."
    )


def roi_pool_patches(
    patch_features: torch.Tensor,
    boxes: torch.Tensor,
    mask: torch.Tensor | None = None,
    grid_hw: tuple[int, int] | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Area-weighted patch pooling over boxes.

    Args:
        patch_features: ``[B, T, P, D]`` per-frame patch tokens (CLS token will
            be stripped if it is present).
        boxes: ``[B, T, K, 4]`` normalised boxes ``(x1, y1, x2, y2)``.
        mask: optional ``[B, T, K]`` 0/1 mask. Missing tracks contribute 0 and
            are *not* normalised away (so per-track pooling stays differentiable
            with respect to other frames where the track is visible).
        grid_hw: explicit patch grid override. If ``None`` we infer it from the
            patch count (assumes square or near-square grids; works for SigLIP
            base/14 which returns 196 patches over a 14x14 grid).

    Returns:
        ``[B, T, K, D]`` per-track features. Invalid tracks are zero-filled.
    """

    if patch_features.dim() != 4:
        raise ValueError("patch_features must be [B, T, P, D].")
    if boxes.dim() != 4 or boxes.shape[-1] != 4:
        raise ValueError("boxes must be [B, T, K, 4].")

    patches, inferred_hw = _normalise_patches(patch_features)
    if grid_hw is None:
        grid_hw = inferred_hw
    grid_h, grid_w = int(grid_hw[0]), int(grid_hw[1])
    batch_size, frames, num_patches, feature_dim = patches.shape
    expected = grid_h * grid_w
    if expected != num_patches:
        raise ValueError(
            f"grid_hw={grid_hw} does not match {num_patches} patch tokens; "
            "explicit grid_hw must equal H*W."
        )

    if boxes.shape[:2] != (batch_size, frames):
        raise ValueError(
            f"boxes leading shape {tuple(boxes.shape[:2])} != patches {tuple(patches.shape[:2])}."
        )
    num_tracks = boxes.shape[2]

    if mask is None:
        mask = torch.ones(batch_size, frames, num_tracks, device=patches.device, dtype=patches.dtype)
    mask = mask.to(dtype=patches.dtype, device=patches.device)

    boxes_pix = boxes.detach().clone()
    boxes_pix[..., 0] = boxes_pix[..., 0].clamp(0.0, 1.0) * grid_w
    boxes_pix[..., 2] = boxes_pix[..., 2].clamp(0.0, 1.0) * grid_w
    boxes_pix[..., 1] = boxes_pix[..., 1].clamp(0.0, 1.0) * grid_h
    boxes_pix[..., 3] = boxes_pix[..., 3].clamp(0.0, 1.0) * grid_h

    px = torch.arange(grid_w, device=patches.device, dtype=patches.dtype)
    py = torch.arange(grid_h, device=patches.device, dtype=patches.dtype)

    x1 = boxes_pix[..., 0].unsqueeze(-1)  # [B,T,K,1]
    x2 = boxes_pix[..., 2].unsqueeze(-1)
    y1 = boxes_pix[..., 1].unsqueeze(-1)
    y2 = boxes_pix[..., 3].unsqueeze(-1)

    x_overlap = (torch.minimum(x2, px + 1.0) - torch.maximum(x1, px)).clamp_min(0.0)  # [B,T,K,W]
    y_overlap = (torch.minimum(y2, py + 1.0) - torch.maximum(y1, py)).clamp_min(0.0)  # [B,T,K,H]

    area = y_overlap.unsqueeze(-1) * x_overlap.unsqueeze(-2)  # [B,T,K,H,W]
    area = area.reshape(batch_size, frames, num_tracks, num_patches)

    box_area = area.sum(dim=-1, keepdim=True).clamp_min(eps)
    weights = area / box_area  # [B,T,K,P], sums to ~1 over patches

    # weighted average over patches
    pooled = torch.einsum("btkp,btpd->btkd", weights, patches)
    pooled = pooled * mask.unsqueeze(-1)
    return pooled


def pool_track_states_to_object(
    track_features: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Average a track over the frames where it is visible.

    track_features: ``[B, T, K, D]``; mask: ``[B, T, K]``.
    Returns ``[B, K, D]``.
    """

    weights = mask.to(track_features.dtype).unsqueeze(-1)
    weighted = track_features * weights
    denom = weights.sum(dim=1).clamp_min(eps)
    return weighted.sum(dim=1) / denom


__all__ = ["roi_pool_patches", "pool_track_states_to_object"]
