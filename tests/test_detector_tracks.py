"""Smoke test: detector-tracks path through ROI pooling + per-object dynamics.

Verifies that:
  * ``roi_pool_patches`` selects the right patches for a planted box.
  * ``CILFModel(use_detector_tracks=True)`` consumes ``track_boxes`` and
    produces per-object dynamics outputs.
  * Gradients reach the ROI projection and the per-object dynamics module.
"""

from __future__ import annotations

import torch

from cilf.losses import (
    object_dynamics_energy_loss,
    object_temporal_consistency_loss,
)
from cilf.model import CILFModel
from cilf.roi_features import roi_pool_patches


def _make_planted_patches(batch: int, frames: int, hp: int, wp: int, dim: int) -> torch.Tensor:
    """Patch grid where one corner is "hot" so we can verify ROI pooling."""

    patches = torch.zeros(batch, frames, hp * wp, dim)
    # patch index 0 = top-left; we plant a strong signal there
    patches[..., 0, :] = 1.0
    return patches


def test_roi_pool_recovers_planted_box() -> None:
    hp, wp, dim = 4, 4, 8
    patches = _make_planted_patches(batch=1, frames=1, hp=hp, wp=wp, dim=dim)
    # box covering exactly the top-left patch
    box = torch.tensor([[[[0.0, 0.0, 1.0 / wp, 1.0 / hp]]]])
    mask = torch.ones(1, 1, 1)
    pooled = roi_pool_patches(patches, box, mask, grid_hw=(hp, wp))
    assert pooled.shape == (1, 1, 1, dim)
    assert torch.allclose(pooled.squeeze(), torch.ones(dim))


def test_roi_pool_zero_for_invalid_mask() -> None:
    hp, wp, dim = 4, 4, 8
    patches = _make_planted_patches(1, 1, hp, wp, dim)
    box = torch.tensor([[[[0.0, 0.0, 0.5, 0.5]]]])
    mask = torch.zeros(1, 1, 1)
    pooled = roi_pool_patches(patches, box, mask, grid_hw=(hp, wp))
    assert pooled.abs().sum().item() == 0.0


def test_detector_tracks_forward_backward() -> None:
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    model = CILFModel(
        llm_name="gpt2",
        state_dim=128,
        jepa_hidden_dim=256,
        vision_pretrained=False,
        dynamics_type="mlp",
        ode_steps=2,
        use_sheaf_alignment=False,
        use_tropical_fusion=False,
        use_object_tracking=False,
        use_detector_tracks=True,
        max_detector_tracks=3,
        use_relation_tokens=True,
    ).to(device)

    batch_size, num_frames = 2, 4
    seq_len = 8
    frames = torch.randn(batch_size, num_frames, 3, 64, 64, device=device)
    input_ids = torch.randint(0, 50257, (batch_size, seq_len), device=device)
    attention_mask = torch.ones(batch_size, seq_len, device=device)

    # Two boxes drifting from left to right across the frame -> the dynamics
    # module should see motion in the per-object state.
    base = torch.tensor(
        [
            [0.10, 0.10, 0.40, 0.40],
            [0.55, 0.40, 0.95, 0.80],
            [0.30, 0.50, 0.70, 0.90],
        ],
        device=device,
    )
    track_boxes = base.unsqueeze(0).unsqueeze(0).expand(batch_size, num_frames, -1, -1).clone()
    drift = torch.linspace(0.0, 0.15, num_frames, device=device).view(1, num_frames, 1, 1)
    track_boxes[..., 0::2] += drift  # shift x1, x2
    track_boxes = track_boxes.clamp(0.0, 1.0)
    track_mask = torch.ones(batch_size, num_frames, 3, device=device)

    model.train()
    outputs = model(
        frames=frames,
        input_ids=input_ids,
        attention_mask=attention_mask,
        alpha_scale=1.0,
        track_boxes=track_boxes,
        track_mask=track_mask,
    )

    for key in ("predicted_object_next", "object_observed_next", "object_trajectory"):
        assert key in outputs, f"missing key {key}"
    assert outputs["object_source"] == "detector"
    assert outputs["object_trajectory"].shape == (batch_size, num_frames, 3, 128)

    consistency = object_temporal_consistency_loss(outputs["object_trajectory"])
    dynamics_loss, pos_energy, neg_energy = object_dynamics_energy_loss(
        outputs["predicted_object_next"],
        outputs["object_observed_next"].detach(),
        margin=0.2,
    )

    loss = outputs["fused_logits"].sum() + consistency + dynamics_loss
    loss.backward()

    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.roi_projection.parameters()
    ), "roi_projection received no gradient"
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.object_dynamics.parameters()
    ), "object_dynamics received no gradient"

    print(
        f"detector-tracks forward+backward OK | "
        f"obj_consistency={consistency.item():.4f} obj_dyn={dynamics_loss.item():.4f} "
        f"posE={pos_energy.item():.4f} negE={neg_energy.item():.4f}"
    )


def main() -> None:
    test_roi_pool_recovers_planted_box()
    test_roi_pool_zero_for_invalid_mask()
    test_detector_tracks_forward_backward()
    print("All detector-tracks checks passed.")


if __name__ == "__main__":
    main()
