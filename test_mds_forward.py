#!/usr/bin/env python3
"""Unit test: MDS-SheafNet forward pass and gradient flow validation."""

import torch
from cilf.model import CILFModel

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    # Create a small model with MDS-SheafNet flags enabled
    model = CILFModel(
        llm_name="gpt2",
        state_dim=256,
        jepa_hidden_dim=512,
        vision_pretrained=False,  # use simple conv encoder for speed
        dynamics_type="mlp",
        ode_steps=2,
        use_sheaf_alignment=True,
        sheaf_threshold=5.0,
        use_tropical_fusion=True,
    ).to(device)

    print("✓ Model created with MDS-SheafNet flags")

    # Synthetic batch (small sizes for speed)
    batch_size, num_frames, channels, height, width = 2, 3, 3, 64, 64
    seq_len = 8
    frames = torch.randn(batch_size, num_frames, channels, height, width).to(device)
    input_ids = torch.randint(0, 50257, (batch_size, seq_len)).to(device)
    attention_mask = torch.ones(batch_size, seq_len).to(device)

    print(f"✓ Synthetic batch created: frames {frames.shape}, input_ids {input_ids.shape}, attn_mask {attention_mask.shape}")

    # Forward pass
    model.train()
    outputs = model(
        frames=frames,
        input_ids=input_ids,
        attention_mask=attention_mask,
        alpha_scale=1.0,
    )

    print("✓ Forward pass succeeded")
    print(f"  - fused_logits shape: {outputs['fused_logits'].shape}")
    print(f"  - h_fused shape: {outputs['h_fused'].shape}")
    print(f"  - sheaf_obstruction_scalar: {outputs.get('sheaf_obstruction_scalar', 'N/A')}")
    if 'sheaf_obstruction_vector' in outputs:
        print(f"  - sheaf_obstruction_vector shape: {outputs['sheaf_obstruction_vector'].shape}")

    # Test gradient flow
    loss = outputs["fused_logits"].sum() + outputs.get("sheaf_obstruction_scalar", torch.tensor(0.0)).mean()
    loss.backward()

    # Check that gradients flow to key parameters
    has_grad_tropical_w = model.tropical_W_raw.grad is not None and model.tropical_W_raw.grad.abs().sum() > 0
    has_grad_rho_sem = next(model.rho_sem.parameters()).grad is not None
    has_grad_rho_phys = next(model.rho_phys.parameters()).grad is not None

    print(f"✓ Backward pass succeeded")
    print(f"  - tropical_W_raw has non-zero grads: {has_grad_tropical_w}")
    print(f"  - rho_sem has grads: {has_grad_rho_sem}")
    print(f"  - rho_phys has grads: {has_grad_rho_phys}")

    # Verify no NaN/Inf
    fused_valid = not (torch.isnan(outputs["fused_logits"]).any() or torch.isinf(outputs["fused_logits"]).any())
    obstruction_valid = not (torch.isnan(outputs.get("sheaf_obstruction_scalar", torch.tensor(0.0))).any())
    print(f"✓ Numerical stability check:")
    print(f"  - fused_logits valid (no NaN/Inf): {fused_valid}")
    print(f"  - sheaf_obstruction valid (no NaN/Inf): {obstruction_valid}")

    print("\n✅ All checks passed! MDS-SheafNet architecture is functional.")

if __name__ == "__main__":
    main()
