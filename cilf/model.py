"""CILF model: frozen LLM with cross-attentive video-language causal fusion."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM

from cilf.dynamics.ode import create_dynamics
from cilf.objects import ObjectTracker, RelationHead
from cilf.roi_features import roi_pool_patches


class VisionFoundationEncoder(nn.Module):
    """Visual encoder with trainable state projection and optional patch features.

    The encoder always returns a pooled per-frame state. When ``return_patches``
    is True in ``forward`` it additionally returns spatial patch features used
    by the object tracker.
    """

    def __init__(
        self,
        state_dim: int,
        model_name: str = "google/siglip-base-patch16-224",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        if not pretrained:
            self.freeze_foundation = False
            self._conv_stack = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
                nn.GELU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
            )
            self.foundation = nn.Sequential(
                self._conv_stack,
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )
            feature_dim = 64
        else:
            self.freeze_foundation = True
            self.foundation = AutoModel.from_pretrained(model_name)
            self.foundation.eval()
            for parameter in self.foundation.parameters():
                parameter.requires_grad = False

            feature_dim = getattr(self.foundation.config, "projection_dim", None)
            if feature_dim is None:
                feature_dim = getattr(self.foundation.config, "hidden_size", None)
            if feature_dim is None:
                vision_config = getattr(self.foundation.config, "vision_config", None)
                if vision_config is not None:
                    feature_dim = getattr(vision_config, "projection_dim", None) or getattr(
                        vision_config, "hidden_size", None
                    )
            if feature_dim is None:
                raise ValueError(f"Unable to infer feature_dim from vision model config for {model_name}.")

            vision_config = getattr(self.foundation.config, "vision_config", None)
            patch_dim_value = None
            if vision_config is not None:
                patch_dim_value = getattr(vision_config, "hidden_size", None)
            if patch_dim_value is None:
                patch_dim_value = getattr(self.foundation.config, "hidden_size", None)
            self.patch_feature_dim = int(patch_dim_value or feature_dim)

        if not pretrained:
            self.patch_feature_dim = 64

        self.feature_dim = int(feature_dim)
        self.project = nn.Sequential(
            nn.Linear(self.feature_dim, state_dim),
            nn.LayerNorm(state_dim),
        )

    def forward(
        self,
        frames: torch.Tensor,
        return_patches: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch_size, frame_count, channels, height, width = frames.shape
        flat_frames = frames.reshape(batch_size * frame_count, channels, height, width)
        patch_features_flat: torch.Tensor | None = None

        if not self.freeze_foundation:
            features = self.foundation(flat_frames)
            if return_patches:
                spatial = self._conv_stack(flat_frames)
                patch_features_flat = spatial.flatten(2).transpose(1, 2)
        else:
            with torch.no_grad():
                if return_patches:
                    vision_model = getattr(self.foundation, "vision_model", None)
                    if vision_model is None:
                        vision_model = self.foundation
                    raw = vision_model(pixel_values=flat_frames)
                    last_hidden = getattr(raw, "last_hidden_state", None)
                    if last_hidden is None:
                        last_hidden = raw if isinstance(raw, torch.Tensor) else None
                    if last_hidden is None:
                        raise TypeError("Vision backbone does not expose last_hidden_state for patch features.")
                    patch_features_flat = last_hidden
                    features = self._extract_features(raw)
                else:
                    if hasattr(self.foundation, "get_image_features"):
                        raw = self.foundation.get_image_features(pixel_values=flat_frames)
                    else:
                        raw = self.foundation(pixel_values=flat_frames)
                    features = self._extract_features(raw)
        states = self.project(features)
        states = states.reshape(batch_size, frame_count, -1)
        if return_patches:
            if patch_features_flat is None:
                raise RuntimeError("Failed to obtain patch features from vision backbone.")
            num_patches = patch_features_flat.shape[1]
            patch_dim = patch_features_flat.shape[2]
            patch_features = patch_features_flat.reshape(batch_size, frame_count, num_patches, patch_dim)
            return states, patch_features
        return states

    @staticmethod
    def _extract_features(raw: object) -> torch.Tensor:
        if isinstance(raw, torch.Tensor):
            return raw
        pooler = getattr(raw, "pooler_output", None)
        if pooler is not None:
            return pooler
        last_hidden = getattr(raw, "last_hidden_state", None)
        if last_hidden is not None:
            return last_hidden.mean(dim=1)
        image_embeds = getattr(raw, "image_embeds", None)
        if image_embeds is not None:
            return image_embeds
        raise TypeError(f"Unable to extract image features from output type {type(raw)!r}")


class LowRankHiddenProjector(nn.Module):
    """Low-rank residual bottleneck in LLM hidden space."""

    def __init__(self, input_dim: int, llm_dim: int, bottleneck_dim: int) -> None:
        super().__init__()
        self.down_proj = nn.Linear(input_dim, bottleneck_dim)
        self.layer_norm = nn.LayerNorm(bottleneck_dim)
        self.up_proj = nn.Linear(bottleneck_dim, llm_dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        bottleneck = self.layer_norm(self.down_proj(hidden))
        return self.up_proj(bottleneck)


class CrossAttentionFusionBlock(nn.Module):
    """Bidirectional text/video attention before producing an LM hidden-space bias."""

    def __init__(
        self,
        llm_dim: int,
        num_heads: int,
        dropout: float,
        bottleneck_dim: int,
    ) -> None:
        super().__init__()
        self.text_to_video = nn.MultiheadAttention(
            embed_dim=llm_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.video_to_text = nn.MultiheadAttention(
            embed_dim=llm_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.text_norm = nn.LayerNorm(llm_dim)
        self.video_norm = nn.LayerNorm(llm_dim)
        self.joint_norm = nn.LayerNorm(llm_dim)
        self.bias_projector = LowRankHiddenProjector(
            input_dim=llm_dim * 2,
            llm_dim=llm_dim,
            bottleneck_dim=bottleneck_dim,
        )

    def forward(
        self,
        text_tokens: torch.Tensor,
        video_tokens: torch.Tensor,
        text_key_padding_mask: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        text_from_video, text_attn = self.text_to_video(
            query=text_tokens,
            key=video_tokens,
            value=video_tokens,
            need_weights=True,
        )
        video_from_text, video_attn = self.video_to_text(
            query=video_tokens,
            key=text_tokens,
            value=text_tokens,
            key_padding_mask=text_key_padding_mask,
            need_weights=True,
        )
        text_joint = self.text_norm(text_tokens + text_from_video)
        video_joint = self.video_norm(video_tokens + video_from_text)
        pooled_video = video_joint.mean(dim=1, keepdim=True).expand(-1, text_joint.shape[1], -1)
        joint_tokens = self.joint_norm(text_joint + pooled_video)
        bias_tokens = self.bias_projector(torch.cat([joint_tokens, pooled_video], dim=-1))
        return {
            "joint_tokens": joint_tokens,
            "bias_tokens": bias_tokens,
            "text_from_video": text_from_video,
            "video_from_text": video_from_text,
            "video_joint": video_joint,
            "text_attn": text_attn,
            "video_attn": video_attn,
        }


class CILFModel(nn.Module):
    def __init__(
        self,
        llm_name: str,
        state_dim: int,
        jepa_hidden_dim: int,
        vision_pretrained: bool = True,
        vision_model_name: str = "google/siglip-base-patch16-224",
        dynamics_type: str = "mlp",
        ode_horizon: float = 1.0,
        ode_steps: int = 4,
        ode_method: str = "rk4",
        ode_use_adjoint: bool = True,
        fixed_alpha: float | None = None,
        bottleneck_dim: int = 64,
        cross_attention_heads: int = 4,
        cross_attention_dropout: float = 0.0,
        use_sheaf_alignment: bool = False,
        sheaf_threshold: float = 5.0,
        use_tropical_fusion: bool = False,
        use_object_tracking: bool = False,
        num_object_slots: int = 4,
        slot_iters: int = 2,
        use_relation_tokens: bool = True,
        use_detector_tracks: bool = False,
        max_detector_tracks: int = 6,
    ) -> None:
        super().__init__()
        self.llm = AutoModelForCausalLM.from_pretrained(llm_name)
        self.llm.eval()
        for parameter in self.llm.parameters():
            parameter.requires_grad = False

        self.state_dim = state_dim
        self.dynamics_type = dynamics_type
        self.fixed_alpha = fixed_alpha
        self.bottleneck_dim = int(bottleneck_dim)

        llm_dim = self.llm.config.hidden_size
        self.llm_dim = llm_dim
        self.visual_encoder = VisionFoundationEncoder(
            state_dim=state_dim,
            model_name=vision_model_name,
            pretrained=vision_pretrained,
        )
        self.temporal_aggregator = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.GELU(),
            nn.LayerNorm(state_dim),
        )
        self.text_to_action = nn.Sequential(
            nn.Linear(llm_dim, state_dim),
            nn.GELU(),
            nn.LayerNorm(state_dim),
        )
        self.dynamics = create_dynamics(
            dynamics_type,
            state_dim=state_dim,
            hidden_dim=jepa_hidden_dim,
            horizon=ode_horizon,
            num_steps=ode_steps,
            method=ode_method,
            use_adjoint=ode_use_adjoint,
        )
        self.use_sheaf_alignment = use_sheaf_alignment
        self.sheaf_threshold = float(sheaf_threshold)
        self.use_tropical_fusion = use_tropical_fusion
        # Tropical fusion parameters (stable, bounded offset per-dimension)
        self.tropical_projector = nn.Linear(llm_dim, llm_dim)
        self.tropical_W_raw = nn.Parameter(torch.zeros(llm_dim))
        self.tropical_clip = float(10.0)
        # Sheaf projection operators: semantic (text) and physical (state->hidden)
        self.rho_sem = nn.Linear(llm_dim, llm_dim)
        self.rho_phys = nn.Linear(state_dim, llm_dim)
        self.video_token_projector = nn.Sequential(
            nn.Linear(state_dim, llm_dim),
            nn.LayerNorm(llm_dim),
        )
        self.delta_token_projector = nn.Sequential(
            nn.Linear(state_dim, llm_dim),
            nn.LayerNorm(llm_dim),
        )
        self.fusion_block = CrossAttentionFusionBlock(
            llm_dim=llm_dim,
            num_heads=int(cross_attention_heads),
            dropout=float(cross_attention_dropout),
            bottleneck_dim=self.bottleneck_dim,
        )
        self.alpha_gate = nn.Sequential(
            nn.Linear(llm_dim * 2, llm_dim // 2),
            nn.GELU(),
            nn.Linear(llm_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Token-type embeddings so cross-attention can tell "what kind of video
        # token am I attending to" instead of having to learn the structure
        # purely from concatenation order.
        # 0 = frame, 1 = scene_delta, 2 = object, 3 = object_delta, 4 = relation
        self.video_token_type_embedding = nn.Embedding(5, llm_dim)

        self.use_object_tracking = bool(use_object_tracking)
        self.num_object_slots = int(num_object_slots)
        self.use_relation_tokens = bool(use_relation_tokens)
        self.use_detector_tracks = bool(use_detector_tracks)
        self.max_detector_tracks = int(max_detector_tracks)

        # The slot-attention path and the detector-track path share the same
        # downstream modules (dynamics, projectors, relation head). When
        # detector tracks are enabled they take precedence at runtime; slot
        # attention serves as the fallback for samples that come without
        # precomputed boxes.
        any_object_path = self.use_object_tracking or self.use_detector_tracks

        if self.use_object_tracking:
            self.object_tracker = ObjectTracker(
                num_slots=self.num_object_slots,
                patch_feature_dim=self.visual_encoder.patch_feature_dim,
                state_dim=state_dim,
                iters=int(slot_iters),
            )
        else:
            self.object_tracker = None

        if self.use_detector_tracks:
            self.roi_projection = nn.Sequential(
                nn.Linear(self.visual_encoder.patch_feature_dim, state_dim),
                nn.LayerNorm(state_dim),
            )
        else:
            self.roi_projection = None

        if any_object_path:
            self.object_dynamics = create_dynamics(
                dynamics_type,
                state_dim=state_dim,
                hidden_dim=jepa_hidden_dim,
                horizon=ode_horizon,
                num_steps=ode_steps,
                method=ode_method,
                use_adjoint=ode_use_adjoint,
            )
            self.object_token_projector = nn.Sequential(
                nn.Linear(state_dim, llm_dim),
                nn.LayerNorm(llm_dim),
            )
            self.object_delta_projector = nn.Sequential(
                nn.Linear(state_dim, llm_dim),
                nn.LayerNorm(llm_dim),
            )
            min_slots = max(
                self.num_object_slots if self.use_object_tracking else 0,
                self.max_detector_tracks if self.use_detector_tracks else 0,
            )
            if self.use_relation_tokens and min_slots >= 2:
                self.relation_head = RelationHead(state_dim=state_dim, output_dim=llm_dim)
            else:
                self.relation_head = None
        else:
            self.object_dynamics = None
            self.object_token_projector = None
            self.object_delta_projector = None
            self.relation_head = None

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_hidden, lingual_logits, _ = self.encode_text_sequence(input_ids, attention_mask)
        return text_hidden, lingual_logits

    def encode_text_sequence(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        llm_outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        last_indices = attention_mask.sum(dim=1).clamp_min(1) - 1
        last_indices = last_indices.long()
        batch_indices = torch.arange(input_ids.shape[0], device=input_ids.device)
        text_tokens = llm_outputs.hidden_states[-1]
        text_hidden = text_tokens[batch_indices, last_indices]
        lingual_logits = llm_outputs.logits[batch_indices, last_indices]
        return text_hidden, lingual_logits, text_tokens

    def _run_dynamics(
        self,
        current_state: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        result = self.dynamics(current_state, action, return_trajectory=True)
        predicted_next_state, trajectory = result
        return predicted_next_state, trajectory

    def _align_text_to_physical(
        self,
        text_tokens: torch.Tensor,
        trajectory: torch.Tensor,
    ) -> torch.Tensor:
        """Map discrete text positions onto the continuous physical trajectory."""
        batch_size, seq_len, _ = text_tokens.shape
        num_states = trajectory.shape[1]
        if num_states == 1:
            return trajectory.expand(batch_size, seq_len, -1)

        positions = torch.linspace(0.0, float(num_states - 1), steps=seq_len, device=trajectory.device)
        lower = positions.floor().long().clamp(max=num_states - 2)
        upper = (lower + 1).clamp(max=num_states - 1)
        fraction = (positions - lower.float()).unsqueeze(0).unsqueeze(-1)

        lower_states = trajectory[:, lower, :]
        upper_states = trajectory[:, upper, :]
        return lower_states * (1.0 - fraction) + upper_states * fraction

    def _cell_complex_incidence(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Simple incidence matrix for a 1D cell complex over token positions.

        Returns an edge incidence tensor of shape [edges, 2] with indices for lower/upper nodes.
        """
        if seq_len <= 1:
            return torch.empty((0, 2), dtype=torch.long, device=device)
        lower = torch.arange(0, seq_len - 1, device=device, dtype=torch.long)
        upper = lower + 1
        return torch.stack([lower, upper], dim=1)

    def _compute_sheaf_obstruction(
        self,
        text_tokens: torch.Tensor,
        trajectory: torch.Tensor,
    ) -> torch.Tensor:
        """Compute a sheaf-style discrepancy score between text and physical state sequences.

        Implements rho_sem(H_text) and rho_phys(hat_s_T) per Section 4.6/4.9 and
        returns both a scalar obstruction per-sample and a per-dimension obstruction vector.
        """
        # semantic projection: summarize text hidden-space per sample
        # use per-token hidden states mean as H_text summary
        h_text_summary = text_tokens.mean(dim=1)
        rho_sem_h = self.rho_sem(h_text_summary)

        # physical projection: use terminal predicted state from trajectory (last step)
        phys_terminal = trajectory[:, -1]
        rho_phys_s = self.rho_phys(phys_terminal)

        # per-dimension discrepancy vector and scalar L2 obstruction
        obstruction_vector = rho_sem_h - rho_phys_s
        obstruction_scalar = torch.norm(obstruction_vector, p=2, dim=-1)
        return obstruction_scalar, obstruction_vector

    def _tropical_fuse(
        self,
        text_hidden: torch.Tensor,
        h_bias: torch.Tensor,
    ) -> torch.Tensor:
        """Stable min-plus fusion of the hidden bias and the LLM hidden state."""
        # bounded per-dimension offset to avoid numerical explosion
        W = torch.tanh(self.tropical_W_raw) * self.tropical_clip
        bias_transformed = self.tropical_projector(h_bias)
        # compute min(text_hidden, W + h_bias) per-dimension in a vectorized manner
        rhs = h_bias + W
        return torch.minimum(text_hidden, rhs)

    def predict_next_state(
        self,
        frames: torch.Tensor,
        action: torch.Tensor | None = None,
        track_boxes: torch.Tensor | None = None,
        track_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        need_patches = self.use_object_tracking or (
            self.use_detector_tracks and track_boxes is not None and track_boxes.numel() > 0
        )
        if need_patches:
            visual_states, patch_features = self.visual_encoder(frames, return_patches=True)
        else:
            visual_states = self.visual_encoder(frames)
            patch_features = None
        if visual_states.shape[1] > 1:
            temporal_context = visual_states[:, :-1].mean(dim=1)
        else:
            temporal_context = visual_states[:, 0]
        current_state = self.temporal_aggregator(temporal_context)
        observed_next_state = visual_states[:, -1]

        if action is None:
            action = torch.zeros_like(current_state)

        predicted_next_state, trajectory = self._run_dynamics(current_state, action)
        outputs: dict[str, torch.Tensor] = {
            "visual_states": visual_states,
            "current_state": current_state,
            "observed_next_state": observed_next_state,
            "predicted_next_state": predicted_next_state,
            "trajectory": trajectory,
            "action": action,
        }

        per_object_trajectory: torch.Tensor | None = None
        per_object_mask: torch.Tensor | None = None
        source_label: str | None = None

        # Detector tracks take precedence when boxes are available -- they
        # carry real object identity, slot attention has to discover it.
        if (
            self.use_detector_tracks
            and self.roi_projection is not None
            and patch_features is not None
            and track_boxes is not None
            and track_boxes.numel() > 0
        ):
            if track_mask is None:
                track_mask = torch.ones(
                    track_boxes.shape[0], track_boxes.shape[1], track_boxes.shape[2],
                    device=track_boxes.device,
                    dtype=track_boxes.dtype,
                )
            pooled = roi_pool_patches(patch_features, track_boxes, track_mask)
            per_object_trajectory = self.roi_projection(pooled)
            per_object_mask = track_mask
            outputs["track_boxes"] = track_boxes
            outputs["track_mask"] = track_mask
            source_label = "detector"
        elif self.use_object_tracking and patch_features is not None:
            slot_trajectory = self.object_tracker(patch_features)
            per_object_trajectory = slot_trajectory
            outputs["slot_trajectory"] = slot_trajectory
            source_label = "slot"

        if per_object_trajectory is not None and self.object_dynamics is not None:
            if per_object_trajectory.shape[1] > 1:
                object_current = per_object_trajectory[:, :-1].mean(dim=1)
            else:
                object_current = per_object_trajectory[:, 0]
            object_observed_next = per_object_trajectory[:, -1]

            batch_size, num_slots, slot_dim = object_current.shape
            slot_action = (
                action.unsqueeze(1).expand(-1, num_slots, -1).reshape(batch_size * num_slots, -1)
            )
            slot_state_flat = object_current.reshape(batch_size * num_slots, slot_dim)
            predicted_flat, slot_trajectory_flat = self.object_dynamics(
                slot_state_flat,
                slot_action,
                return_trajectory=True,
            )
            predicted_object_next = predicted_flat.reshape(batch_size, num_slots, slot_dim)
            slot_dynamics_trajectory = slot_trajectory_flat.reshape(
                batch_size, num_slots, slot_trajectory_flat.shape[1], slot_dim,
            )

            outputs.update(
                {
                    "patch_features": patch_features,
                    "object_trajectory": per_object_trajectory,
                    "object_current": object_current,
                    "object_observed_next": object_observed_next,
                    "predicted_object_next": predicted_object_next,
                    "slot_dynamics_trajectory": slot_dynamics_trajectory,
                    "object_source": source_label or "",
                }
            )
            # Slot trajectory key preserved for backwards-compatible loss code.
            if "slot_trajectory" not in outputs:
                outputs["slot_trajectory"] = per_object_trajectory
            if per_object_mask is not None:
                outputs["object_mask"] = per_object_mask
        return outputs

    def _build_video_tokens(
        self,
        visual_states: torch.Tensor,
        state_delta: torch.Tensor,
        slot_trajectory: torch.Tensor | None = None,
        predicted_object_next: torch.Tensor | None = None,
        object_current: torch.Tensor | None = None,
        object_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build the H_video sequence consumed by cross-attention.

        Layout:
          - frame tokens [B, T, D] tagged with type=0
          - scene delta token [B, 1, D] tagged with type=1 (existing baseline)
          - object pooled tokens [B, K, D] tagged with type=2
          - per-object delta tokens [B, K, D] tagged with type=3
          - relation tokens [B, K*(K-1), D] tagged with type=4

        Dedicated token-type embeddings let the fusion block tell *what* each
        token represents without having to recover the structure from
        concatenation order alone, and let scene-delta and per-object deltas
        carry distinct semantics rather than being averaged into one "change"
        signal.
        """

        def _add_type(tokens: torch.Tensor, type_id: int) -> torch.Tensor:
            embedding = self.video_token_type_embedding.weight[type_id]
            return tokens + embedding

        video_tokens = _add_type(self.video_token_projector(visual_states), 0)
        delta_token = _add_type(self.delta_token_projector(state_delta).unsqueeze(1), 1)
        tokens = [video_tokens, delta_token]

        if (
            slot_trajectory is not None
            and self.object_token_projector is not None
            and self.object_delta_projector is not None
        ):
            if object_mask is not None:
                weights = object_mask.to(slot_trajectory.dtype).unsqueeze(-1)
                weighted = slot_trajectory * weights
                denom = weights.sum(dim=1).clamp_min(1e-6)
                pooled_objects = weighted.sum(dim=1) / denom
            else:
                pooled_objects = slot_trajectory.mean(dim=1)
            object_tokens = _add_type(self.object_token_projector(pooled_objects), 2)
            tokens.append(object_tokens)

            if predicted_object_next is not None and object_current is not None:
                object_delta = predicted_object_next - object_current
            else:
                object_delta = slot_trajectory[:, -1] - slot_trajectory[:, 0]
            object_delta_tokens = _add_type(self.object_delta_projector(object_delta), 3)
            tokens.append(object_delta_tokens)

            if self.relation_head is not None:
                relation_tokens = self.relation_head(slot_trajectory)
                if relation_tokens.shape[1] > 0:
                    tokens.append(_add_type(relation_tokens, 4))

        return torch.cat(tokens, dim=1)

    def _fuse_joint(
        self,
        text_tokens: torch.Tensor,
        text_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        video_tokens: torch.Tensor,
        alpha_scale: float,
        trajectory: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        lm_head = self.llm.get_output_embeddings()
        text_key_padding_mask = attention_mask == 0
        joint = self.fusion_block(
            text_tokens=text_tokens,
            video_tokens=video_tokens,
            text_key_padding_mask=text_key_padding_mask,
        )
        last_indices = attention_mask.sum(dim=1).clamp_min(1) - 1
        last_indices = last_indices.long()
        batch_indices = torch.arange(attention_mask.shape[0], device=attention_mask.device)
        h_bias = joint["bias_tokens"][batch_indices, last_indices]
        video_pooled = joint["video_joint"].mean(dim=1)
        sheaf_obstruction = None

        if self.use_sheaf_alignment and trajectory is not None:
            sheaf_scalar, sheaf_vector = self._compute_sheaf_obstruction(text_tokens, trajectory)
            # mask components of h_bias whose per-dimension obstruction exceeds threshold
            mask = (sheaf_vector.abs() <= self.sheaf_threshold).float()

            # recompute joint with original video tokens, then apply mask to bias
            joint = self.fusion_block(
                text_tokens=text_tokens,
                video_tokens=video_tokens,
                text_key_padding_mask=text_key_padding_mask,
            )
            h_bias = joint["bias_tokens"][batch_indices, last_indices]
            # apply per-dimension mask to bias (broadcast over batch)
            h_bias = h_bias * mask
            video_pooled = joint["video_joint"].mean(dim=1)

        if self.fixed_alpha is not None:
            alpha = torch.full(
                (text_hidden.shape[0], 1),
                float(self.fixed_alpha) * alpha_scale,
                device=text_hidden.device,
                dtype=text_hidden.dtype,
            )
        else:
            alpha = self.alpha_gate(torch.cat([text_hidden, h_bias], dim=-1)) * alpha_scale

        if self.use_tropical_fusion:
            h_fused = self._tropical_fuse(text_hidden, h_bias)
        else:
            h_fused = text_hidden + alpha * h_bias

        fused_logits = lm_head(h_fused)
        causal_logits = lm_head(h_bias)
        result = {
            "h_causal": h_bias,
            "h_bias": h_bias,
            "h_fused": h_fused,
            "text_pool": text_hidden,
            "video_pool": video_pooled,
            "causal_logits": causal_logits,
            "fused_logits": fused_logits,
            "alpha": alpha.squeeze(-1),
            **joint,
        }
        if self.use_sheaf_alignment and trajectory is not None:
            # provide both scalar and vector obstruction for downstream losses/diagnostics
            result["sheaf_obstruction_scalar"] = sheaf_scalar
            result["sheaf_obstruction_vector"] = sheaf_vector
        return result

    def forward(
        self,
        frames: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        alpha_scale: float = 1.0,
        action: torch.Tensor | None = None,
        track_boxes: torch.Tensor | None = None,
        track_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        text_hidden, lingual_logits, text_tokens = self.encode_text_sequence(input_ids, attention_mask)
        if action is None:
            action = self.text_to_action(text_hidden)
        dynamics = self.predict_next_state(
            frames,
            action=action,
            track_boxes=track_boxes,
            track_mask=track_mask,
        )
        state_delta = dynamics["predicted_next_state"] - dynamics["current_state"]
        video_tokens = self._build_video_tokens(
            dynamics["visual_states"],
            state_delta,
            slot_trajectory=dynamics.get("slot_trajectory"),
            predicted_object_next=dynamics.get("predicted_object_next"),
            object_current=dynamics.get("object_current"),
            object_mask=dynamics.get("object_mask"),
        )
        fusion = self._fuse_joint(
            text_tokens,
            text_hidden,
            attention_mask,
            video_tokens,
            alpha_scale,
            trajectory=dynamics["trajectory"],
        )
        return {
            "lingual_logits": lingual_logits,
            "text_hidden": text_hidden,
            "text_tokens": text_tokens,
            "video_tokens": video_tokens,
            **dynamics,
            **fusion,
        }
