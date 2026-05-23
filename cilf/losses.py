"""Loss functions for the CILF energy surface."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def informational_surprise_loss(
    predicted_state: torch.Tensor,
    observed_state: torch.Tensor,
) -> torch.Tensor:
    """Cosine-distance surprise in latent space."""

    return 1.0 - F.cosine_similarity(predicted_state, observed_state, dim=-1)


def energy_loss(
    predicted_next_state: torch.Tensor,
    observed_next_state: torch.Tensor,
    corrupted_next_state: torch.Tensor,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    positive_energy = informational_surprise_loss(predicted_next_state, observed_next_state)
    negative_energy = informational_surprise_loss(predicted_next_state, corrupted_next_state)
    hinge = F.relu(margin + positive_energy - negative_energy)
    return (positive_energy + hinge).mean(), positive_energy.mean(), negative_energy.mean()


def trajectory_energy_loss(
    trajectory: torch.Tensor,
    visual_states: torch.Tensor,
    corrupted_next_state: torch.Tensor | None,
    margin: float,
    *,
    integrate_time: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Integrated energy E = int L(s,a) dt approximated along the ODE discretization.

    Args:
        trajectory: [batch, steps, state_dim] predicted states along integration.
        visual_states: [batch, frames, state_dim] observed frame encodings.
        corrupted_next_state: optional hard-negative terminal state. If omitted, the
            next frame in sequence is used as temporal mismatch negative.
    """
    batch_size, num_steps, _ = trajectory.shape
    frame_count = visual_states.shape[1]
    target_indices = torch.linspace(
        0,
        frame_count - 1,
        num_steps,
        device=trajectory.device,
    ).long().clamp(0, frame_count - 1)
    observed_along = visual_states[:, target_indices]

    if integrate_time:
        step_loss = informational_surprise_loss(trajectory, observed_along)
        positive_energy = step_loss.mean(dim=-1)
    else:
        positive_energy = informational_surprise_loss(
            trajectory[:, -1],
            visual_states[:, -1],
        )

    if corrupted_next_state is None:
        shifted_indices = (target_indices + 1).clamp(0, frame_count - 1)
        mismatch_state = visual_states[:, shifted_indices][:, -1]
    else:
        mismatch_state = corrupted_next_state
    negative_energy = informational_surprise_loss(trajectory[:, -1], mismatch_state)
    hinge = F.relu(margin + positive_energy - negative_energy)
    return (positive_energy + hinge).mean(), positive_energy.mean(), negative_energy.mean()


def vicreg_variance_loss(states: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Penalize collapsed latent dimensions with the VICReg variance term."""

    flat_states = states.reshape(-1, states.shape[-1])
    std = torch.sqrt(flat_states.var(dim=0) + eps)
    return F.relu(1.0 - std).mean()


def causal_hidden_l2_regularization(h_causal: torch.Tensor) -> torch.Tensor:
    """L2 penalty on projected causal vectors in LLM hidden space (per-sample mean)."""

    return torch.norm(h_causal, p=2, dim=-1).mean()


def text_video_contrastive_loss(
    text_features: torch.Tensor,
    video_features: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Symmetric InfoNCE loss aligning matching text/video representations."""

    if text_features.shape[0] <= 1:
        return text_features.new_tensor(0.0)
    text_norm = F.normalize(text_features, dim=-1)
    video_norm = F.normalize(video_features, dim=-1)
    logits = text_norm @ video_norm.T / max(float(temperature), 1e-6)
    labels = torch.arange(text_features.shape[0], device=text_features.device)
    text_to_video = F.cross_entropy(logits, labels)
    video_to_text = F.cross_entropy(logits.T, labels)
    return 0.5 * (text_to_video + video_to_text)
