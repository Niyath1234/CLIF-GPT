"""Smoke test: object-centric CILF forward + backward pass.

Validates that the slot-attention object tracker, per-object dynamics, and
relation tokens are wired into the model, that the new outputs are present in
the forward dict, and that gradients reach the new modules.
"""

from __future__ import annotations

import torch

from cilf.losses import (
    object_dynamics_energy_loss,
    object_temporal_consistency_loss,
)
from cilf.model import CILFModel


def main() -> None:
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
        use_object_tracking=True,
        num_object_slots=4,
        slot_iters=2,
        use_relation_tokens=True,
    ).to(device)
    print("model created with use_object_tracking=True")

    batch_size, num_frames, channels, height, width = 2, 4, 3, 64, 64
    seq_len = 8
    frames = torch.randn(batch_size, num_frames, channels, height, width, device=device)
    input_ids = torch.randint(0, 50257, (batch_size, seq_len), device=device)
    attention_mask = torch.ones(batch_size, seq_len, device=device)

    model.train()
    outputs = model(
        frames=frames,
        input_ids=input_ids,
        attention_mask=attention_mask,
        alpha_scale=1.0,
    )
    print("forward succeeded")

    required_keys = [
        "slot_trajectory",
        "object_current",
        "object_observed_next",
        "predicted_object_next",
        "slot_dynamics_trajectory",
        "fused_logits",
    ]
    for key in required_keys:
        assert key in outputs, f"missing output key: {key}"
        tensor = outputs[key]
        assert torch.is_tensor(tensor), f"{key} is not a tensor"
        assert not torch.isnan(tensor).any(), f"{key} contains NaN"
        assert not torch.isinf(tensor).any(), f"{key} contains Inf"

    assert outputs["slot_trajectory"].shape == (
        batch_size,
        num_frames,
        model.num_object_slots,
        model.state_dim,
    ), f"unexpected slot_trajectory shape: {outputs['slot_trajectory'].shape}"

    expected_video_token_count = num_frames + 1 + model.num_object_slots + model.num_object_slots
    if model.relation_head is not None:
        expected_video_token_count += model.num_object_slots * (model.num_object_slots - 1)
    assert outputs["video_tokens"].shape[1] == expected_video_token_count, (
        f"unexpected video token count: got {outputs['video_tokens'].shape[1]}, "
        f"expected {expected_video_token_count}"
    )
    print(
        f"slot_trajectory: {tuple(outputs['slot_trajectory'].shape)}, "
        f"predicted_object_next: {tuple(outputs['predicted_object_next'].shape)}, "
        f"video_tokens: {tuple(outputs['video_tokens'].shape)}"
    )

    consistency = object_temporal_consistency_loss(outputs["slot_trajectory"])
    dynamics_loss, pos_energy, neg_energy = object_dynamics_energy_loss(
        outputs["predicted_object_next"],
        outputs["object_observed_next"].detach(),
        margin=0.2,
    )
    print(
        f"object_consistency={consistency.item():.4f} "
        f"object_dynamics={dynamics_loss.item():.4f} "
        f"posE={pos_energy.item():.4f} negE={neg_energy.item():.4f}"
    )

    loss = (
        outputs["fused_logits"].sum()
        + consistency
        + dynamics_loss
    )
    loss.backward()
    print("backward succeeded")

    def _has_grad(module: torch.nn.Module) -> bool:
        for parameter in module.parameters():
            if parameter.grad is not None and parameter.grad.abs().sum().item() > 0.0:
                return True
        return False

    assert _has_grad(model.object_tracker), "object_tracker received no gradient"
    assert _has_grad(model.object_dynamics), "object_dynamics received no gradient"
    assert _has_grad(model.object_token_projector), "object_token_projector received no gradient"
    assert _has_grad(model.object_delta_projector), "object_delta_projector received no gradient"
    if model.relation_head is not None:
        assert _has_grad(model.relation_head), "relation_head received no gradient"
    print("gradients verified for object tracker, dynamics, projectors, and relation head")

    print("All object-tracking checks passed.")


if __name__ == "__main__":
    main()
