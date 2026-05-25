"""Object-centric tracking: slot attention with temporal binding.

Given per-frame patch features [batch, num_frames, num_patches, feature_dim], an
``ObjectTracker`` produces per-frame, per-slot object states
[batch, num_frames, num_slots, state_dim]. Slot identity is preserved across
frames by feeding the previous-frame slot embeddings as the initial query for
the next frame.

The point of this module in CLIF is to give the model a handle on *which object
changed*, instead of pooling the whole scene into a single state vector. The
per-slot trajectories feed:
  * a per-object dynamics path (delta-state per slot)
  * a relation/interaction head (pairwise slot tokens)
  * the language fusion block, as additional video tokens
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SlotAttention(nn.Module):
    """Locatello-et-al. slot attention with iterative refinement.

    Softmax is taken over slots (not inputs), so slots compete to bind to
    different parts of the scene. A GRU update lets slot state evolve across
    iterations while preserving identity.
    """

    def __init__(
        self,
        num_slots: int,
        dim: int,
        iters: int = 3,
        hidden_dim: int | None = None,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.num_slots = int(num_slots)
        self.iters = int(iters)
        self.eps = float(eps)
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, 1, dim))

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        self.gru = nn.GRUCell(dim, dim)

        ff_dim = int(hidden_dim or dim * 2)
        self.mlp = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, dim),
        )

        self.norm_inputs = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def initial_slots(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        mu = self.slots_mu.expand(batch_size, self.num_slots, -1).to(device=device, dtype=dtype)
        sigma = self.slots_log_sigma.exp().expand(batch_size, self.num_slots, -1).to(device=device, dtype=dtype)
        noise = torch.randn_like(mu)
        return mu + sigma * noise

    def forward(self, inputs: torch.Tensor, prev_slots: torch.Tensor | None = None) -> torch.Tensor:
        """inputs: [batch, num_inputs, dim]; prev_slots: [batch, num_slots, dim] or None."""
        batch_size, _, dim = inputs.shape
        if prev_slots is None:
            slots = self.initial_slots(batch_size, inputs.device, inputs.dtype)
        else:
            slots = prev_slots

        inputs_n = self.norm_inputs(inputs)
        keys = self.to_k(inputs_n)
        values = self.to_v(inputs_n)

        for _ in range(self.iters):
            slots_prev = slots
            slots_n = self.norm_slots(slots)
            queries = self.to_q(slots_n)

            dots = torch.einsum("bkd,bnd->bkn", queries, keys) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum("bnd,bkn->bkd", values, attn)

            slots = self.gru(
                updates.reshape(-1, dim),
                slots_prev.reshape(-1, dim),
            ).reshape(batch_size, self.num_slots, dim)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots


class ObjectTracker(nn.Module):
    """Track ``num_slots`` object slots across frames using slot attention."""

    def __init__(
        self,
        num_slots: int,
        patch_feature_dim: int,
        state_dim: int,
        iters: int = 2,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.num_slots = int(num_slots)
        self.state_dim = int(state_dim)
        self.input_projection = nn.Sequential(
            nn.Linear(int(patch_feature_dim), int(state_dim)),
            nn.LayerNorm(int(state_dim)),
        )
        self.slot_attention = SlotAttention(
            num_slots=num_slots,
            dim=state_dim,
            iters=iters,
            hidden_dim=hidden_dim,
        )

    def forward(self, patch_features: torch.Tensor) -> torch.Tensor:
        """patch_features: [batch, num_frames, num_patches, feature_dim].

        Returns object trajectory [batch, num_frames, num_slots, state_dim].
        """

        batch_size, num_frames, _, _ = patch_features.shape
        projected = self.input_projection(patch_features)
        slots: torch.Tensor | None = None
        slot_trajectory: list[torch.Tensor] = []
        for frame_index in range(num_frames):
            slots = self.slot_attention(projected[:, frame_index], prev_slots=slots)
            slot_trajectory.append(slots)
        return torch.stack(slot_trajectory, dim=1)


class RelationHead(nn.Module):
    """Compose pairwise object-interaction tokens from slot trajectories.

    For every ordered pair (i, j) of slots, the relation token encodes:
      * pooled state of slot i
      * pooled state of slot j
      * delta of slot i (last - first)
      * delta of slot j (last - first)
    These are intended to express "object i acts on object j" features such as
    contact + transfer of motion.
    """

    def __init__(self, state_dim: int, output_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        ff_dim = int(hidden_dim or output_dim)
        self.mlp = nn.Sequential(
            nn.Linear(int(state_dim) * 4, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, int(output_dim)),
            nn.LayerNorm(int(output_dim)),
        )

    def forward(self, slot_trajectory: torch.Tensor) -> torch.Tensor:
        """slot_trajectory: [batch, num_frames, num_slots, state_dim].

        Returns relation tokens [batch, num_pairs, output_dim] for all ordered
        (i, j), i != j pairs.
        """

        batch_size, num_frames, num_slots, state_dim = slot_trajectory.shape
        if num_slots < 2:
            return slot_trajectory.new_zeros(batch_size, 0, self.mlp[-2].out_features)

        pooled = slot_trajectory.mean(dim=1)
        deltas = slot_trajectory[:, -1] - slot_trajectory[:, 0]

        pair_features: list[torch.Tensor] = []
        for i in range(num_slots):
            for j in range(num_slots):
                if i == j:
                    continue
                feature = torch.cat(
                    [pooled[:, i], pooled[:, j], deltas[:, i], deltas[:, j]],
                    dim=-1,
                )
                pair_features.append(feature)
        stacked = torch.stack(pair_features, dim=1)
        return self.mlp(stacked)
