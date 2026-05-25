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


def object_temporal_consistency_loss(
    slot_trajectory: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """InfoNCE pulling slot k at time t+1 close to slot k at time t.

    Same-slot transitions are positives; other-slot states at the same next
    frame are negatives. This encourages persistent object identity across
    frames instead of slots re-binding to whichever object is most salient.

    slot_trajectory: [batch, num_frames, num_slots, dim].
    """

    batch_size, num_frames, num_slots, _ = slot_trajectory.shape
    if num_frames < 2 or num_slots < 2:
        return slot_trajectory.new_tensor(0.0)

    current = F.normalize(slot_trajectory[:, :-1], dim=-1)
    nxt = F.normalize(slot_trajectory[:, 1:], dim=-1)

    logits = torch.einsum("btkd,btld->btkl", current, nxt) / max(float(temperature), 1e-6)
    targets = torch.arange(num_slots, device=slot_trajectory.device)
    targets = targets.view(1, 1, num_slots).expand(batch_size, num_frames - 1, num_slots)

    return F.cross_entropy(
        logits.reshape(-1, num_slots),
        targets.reshape(-1),
    )


def object_dynamics_energy_loss(
    predicted_next_object_state: torch.Tensor,
    observed_next_object_state: torch.Tensor,
    margin: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hinge energy on per-object next-state prediction.

    For each slot, the positive is its own observed next state; the negative is
    that slot's next state from another sample in the batch (paired with a
    different physical situation). This pushes the dynamics module to predict
    the right *object* future, not just any plausible future.

    Both tensors: [batch, num_slots, dim].
    """

    batch_size = predicted_next_object_state.shape[0]
    positive = informational_surprise_loss(predicted_next_object_state, observed_next_object_state)

    if batch_size > 1:
        rolled = torch.roll(observed_next_object_state, shifts=1, dims=0).detach()
    else:
        rolled = observed_next_object_state.detach() + 0.1 * torch.randn_like(observed_next_object_state)
    negative = informational_surprise_loss(predicted_next_object_state, rolled)

    hinge = F.relu(margin + positive - negative)
    return (positive + hinge).mean(), positive.mean(), negative.mean()


def abstract_dynamics_contrastive_loss(
    features: torch.Tensor,
    group_ids: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Supervised contrastive loss over an abstract-dynamics label.

    Clips that share an abstract-dynamics label (e.g. ``support_loss_freefall``)
    are pulled together in the causal-bias embedding space; clips with
    different abstract dynamics are pushed apart. This is the transition-level
    invariant: the model should learn that a falling vase and a falling
    domino share a hidden-meaning embedding, even though their objects and
    captions are unrelated.

    Args:
        features: [batch, dim] embeddings (typically the per-sample causal
            bias ``h_bias`` produced by the fusion block).
        group_ids: [batch] integer ids; rows with the same id are positives.
            Use -1 for samples whose abstract dynamics is unknown; those rows
            are skipped as anchors but can still serve as negatives.
        temperature: softmax temperature.
    """

    if features.shape[0] <= 1:
        return features.new_tensor(0.0)

    normalized = F.normalize(features, dim=-1)
    similarity = normalized @ normalized.T / max(float(temperature), 1e-6)

    batch_size = features.shape[0]
    diagonal_mask = torch.eye(batch_size, dtype=torch.bool, device=features.device)
    similarity = similarity.masked_fill(diagonal_mask, float("-inf"))

    group_ids = group_ids.view(-1, 1)
    positive_mask = (group_ids == group_ids.T) & (~diagonal_mask) & (group_ids >= 0)

    anchor_valid = positive_mask.any(dim=1)
    if not anchor_valid.any():
        return features.new_tensor(0.0)

    log_prob = similarity - torch.logsumexp(similarity, dim=1, keepdim=True)
    log_prob_masked = torch.where(
        positive_mask,
        log_prob,
        torch.zeros_like(log_prob),
    )
    positives_per_anchor = positive_mask.float()
    positives_sum = positives_per_anchor.sum(dim=1).clamp_min(1.0)

    mean_log_prob_pos = log_prob_masked.sum(dim=1) / positives_sum
    loss = -mean_log_prob_pos[anchor_valid]
    return loss.mean()
