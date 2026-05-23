"""CILF model: frozen LLM with cross-attentive video-language causal fusion."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM

from cilf.dynamics.ode import create_dynamics


class VisionFoundationEncoder(nn.Module):
    """Visual encoder followed by trainable state projection."""

    def __init__(
        self,
        state_dim: int,
        model_name: str = "google/siglip-base-patch16-224",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        if not pretrained:
            self.freeze_foundation = False
            self.foundation = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
                nn.GELU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
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

        self.feature_dim = int(feature_dim)
        self.project = nn.Sequential(
            nn.Linear(self.feature_dim, state_dim),
            nn.LayerNorm(state_dim),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        batch_size, frame_count, channels, height, width = frames.shape
        flat_frames = frames.reshape(batch_size * frame_count, channels, height, width)
        if not self.freeze_foundation:
            features = self.foundation(flat_frames)
        else:
            with torch.no_grad():
                if hasattr(self.foundation, "get_image_features"):
                    raw = self.foundation.get_image_features(pixel_values=flat_frames)
                else:
                    raw = self.foundation(pixel_values=flat_frames)
                features = self._extract_features(raw)
        states = self.project(features)
        return states.reshape(batch_size, frame_count, -1)

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

    def predict_next_state(
        self,
        frames: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        visual_states = self.visual_encoder(frames)
        if visual_states.shape[1] > 1:
            temporal_context = visual_states[:, :-1].mean(dim=1)
        else:
            temporal_context = visual_states[:, 0]
        current_state = self.temporal_aggregator(temporal_context)
        observed_next_state = visual_states[:, -1]

        if action is None:
            action = torch.zeros_like(current_state)

        predicted_next_state, trajectory = self._run_dynamics(current_state, action)
        return {
            "visual_states": visual_states,
            "current_state": current_state,
            "observed_next_state": observed_next_state,
            "predicted_next_state": predicted_next_state,
            "trajectory": trajectory,
            "action": action,
        }

    def _build_video_tokens(
        self,
        visual_states: torch.Tensor,
        state_delta: torch.Tensor,
    ) -> torch.Tensor:
        video_tokens = self.video_token_projector(visual_states)
        delta_token = self.delta_token_projector(state_delta).unsqueeze(1)
        return torch.cat([video_tokens, delta_token], dim=1)

    def _fuse_joint(
        self,
        text_tokens: torch.Tensor,
        text_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        video_tokens: torch.Tensor,
        alpha_scale: float,
    ) -> dict[str, torch.Tensor]:
        lm_head = self.llm.get_output_embeddings()
        text_key_padding_mask = attention_mask == 0
        joint = self.fusion_block(
            text_tokens=text_tokens,
            video_tokens=video_tokens,
            text_key_padding_mask=text_key_padding_mask,
        )
        last_indices = attention_mask.sum(dim=1).clamp_min(1) - 1
        batch_indices = torch.arange(attention_mask.shape[0], device=attention_mask.device)
        h_bias = joint["bias_tokens"][batch_indices, last_indices]
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

        h_fused = text_hidden + alpha * h_bias
        fused_logits = lm_head(h_fused)
        causal_logits = lm_head(h_bias)
        return {
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

    def forward(
        self,
        frames: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        alpha_scale: float = 1.0,
        action: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        text_hidden, lingual_logits, text_tokens = self.encode_text_sequence(input_ids, attention_mask)
        if action is None:
            action = self.text_to_action(text_hidden)
        dynamics = self.predict_next_state(frames, action=action)
        state_delta = dynamics["predicted_next_state"] - dynamics["current_state"]
        video_tokens = self._build_video_tokens(dynamics["visual_states"], state_delta)
        fusion = self._fuse_joint(text_tokens, text_hidden, attention_mask, video_tokens, alpha_scale)
        return {
            "lingual_logits": lingual_logits,
            "text_hidden": text_hidden,
            "text_tokens": text_tokens,
            "video_tokens": video_tokens,
            **dynamics,
            **fusion,
        }
