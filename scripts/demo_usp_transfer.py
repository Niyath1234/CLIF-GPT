"""Gradio USP demo: zero-shot cross-domain transfer with real CILF logits.

This app demonstrates the CILF claim that the same learned causal latent dynamic
can transfer across domains while language context disambiguates the final token.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml
from transformers import AutoTokenizer

from cilf.train import choose_device, make_model
from cilf.video_io import read_video


DISPLAY_WORDS = [
    "fell",
    "dropped",
    "crashed",
    "shattered",
    "rolled",
    "caught",
    "settled",
    "hit",
    "stopped",
    "waited",
    "yelled",
    "left",
]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOADED_ENGINE: USPTransferEngine | None = None


def _format_token(token: str) -> str:
    cleaned = token.strip()
    if cleaned and cleaned.isprintable() and cleaned.isascii():
        return cleaned
    return repr(token)


def _best_readable_token(logits: torch.Tensor, tokenizer, k: int = 32) -> tuple[str, int]:
    """Pick the highest-probability fused token that renders legibly in the UI."""
    top_ids = torch.topk(logits, k=min(k, logits.numel())).indices.tolist()
    for token_id in top_ids:
        text = tokenizer.decode([int(token_id)]).strip()
        if text and text.isprintable() and text.isascii() and not text.isspace():
            return text, int(token_id)
    token_id = int(logits.argmax().item())
    return _format_token(tokenizer.decode([token_id])), token_id


def _empty_barplot_df() -> pd.DataFrame:
    return pd.DataFrame({"token": [], "value": []})


def _empty_plot(title: str = ""):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    return fig


def _bar_figure(df: pd.DataFrame, title: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    if df.empty:
        ax.set_title(title)
        ax.axis("off")
        fig.tight_layout()
        return fig
    plot_df = df.copy()
    plot_df["value"] = pd.to_numeric(plot_df["value"], errors="coerce").fillna(0.0)
    plot_df = plot_df.sort_values("value", ascending=False)
    colors = ["#4f46e5" if value >= 0 else "#ef4444" for value in plot_df["value"]]
    ax.bar(plot_df["token"], plot_df["value"], color=colors)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def _latest_kinetic_checkpoint() -> Path | None:
    run_dir = PROJECT_ROOT / "runs/cilf"
    if not run_dir.exists():
        return None
    cilf_ckpts = sorted(
        run_dir.glob("cilf_checkpoint_step_*.pt"),
        key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
    )
    if cilf_ckpts:
        return cilf_ckpts[-1]
    return None


def discover_default_config(explicit: str | None) -> str:
    if explicit:
        return str(Path(explicit).expanduser().resolve())
    default = PROJECT_ROOT / "configs/cilf.yaml"
    return str(default)


def discover_default_checkpoint(explicit: str | None) -> str:
    if explicit:
        return str(Path(explicit).expanduser().resolve())
    kinetic = _latest_kinetic_checkpoint()
    return str(kinetic) if kinetic is not None else ""


def discover_default_impact_video(explicit: str | None) -> str:
    if explicit:
        return str(Path(explicit).expanduser().resolve())

    manifest_candidates = [
        PROJECT_ROOT / "data/kinetic_transfer/manifest_kinetic_train.jsonl",
        PROJECT_ROOT / "data/kinetic_transfer/manifest_kinetic_transfer.jsonl",
        PROJECT_ROOT / "data/physion/manifest_train.jsonl",
    ]
    for manifest_path in manifest_candidates:
        if not manifest_path.exists():
            continue
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = line.strip()
                if not row or row.startswith("#"):
                    continue
                try:
                    payload = json.loads(row)
                except json.JSONDecodeError:
                    continue
                raw_video = payload.get("video_path") or payload.get("video") or payload.get("path")
                if not raw_video:
                    continue
                video_path = Path(raw_video)
                if not video_path.is_absolute():
                    video_path = (manifest_path.parent / video_path).resolve()
                if video_path.exists():
                    return str(video_path)
    return ""


class USPTransferEngine:
    """Loads trained CILF components and emits real logits for demo words."""

    def __init__(self, config_path: str | Path, checkpoint_path: str | Path) -> None:
        self.config_path = Path(config_path)
        self.checkpoint_path = Path(checkpoint_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        with self.config_path.open("r", encoding="utf-8") as handle:
            self.config = yaml.safe_load(handle)

        self.device = choose_device(str(self.config.get("device", "auto")))
        self.model = make_model(self.config, self.device)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"], strict=False)
        self.model.eval()

        model_cfg = self.config["model"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_cfg["llm_name"])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        data_cfg = self.config.get("data", {})
        self.num_frames = int(data_cfg.get("num_frames", 4))
        self.image_size = int(data_cfg.get("image_size", 96))
        self.fps = int(data_cfg.get("fps", 6))
        self.max_prompt_length = int(data_cfg.get("max_prompt_length", 48))

        self.token_lookup = self._build_token_lookup(DISPLAY_WORDS)
        missing = [word for word in DISPLAY_WORDS if word not in self.token_lookup]
        if missing:
            raise ValueError(
                "Tokenizer could not map demo words to single-token IDs: "
                + ", ".join(missing)
            )


    def _build_token_lookup(self, words: list[str]) -> dict[str, int]:
        mapping: dict[str, int] = {}
        for word in words:
            token_id = self._resolve_word_token(word)
            if token_id is not None:
                mapping[word] = token_id
        return mapping

    def _resolve_word_token(self, word: str) -> int | None:
        # GPT-style tokenizers usually encode lexical words with a leading space.
        candidates = [f" {word}", word, f" {word.capitalize()}", word.capitalize()]
        for cand in candidates:
            ids = self.tokenizer.encode(cand, add_special_tokens=False)
            if len(ids) == 1:
                return int(ids[0])
        return None

    def _tokenize_prompt(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        # For GPT-style tokenizers, trailing space biases next-token prediction toward whole-word starts.
        normalized_prompt = prompt.rstrip() + " "
        encoded = self.tokenizer(
            normalized_prompt,
            truncation=True,
            max_length=self.max_prompt_length,
            padding="max_length",
            return_tensors="pt",
        )
        return encoded["input_ids"].to(self.device), encoded["attention_mask"].to(self.device)

    def _load_frames(self, video_path: Path) -> torch.Tensor:
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        video, native_fps = read_video(video_path, target_fps=self.fps)
        if video.numel() == 0:
            raise ValueError(f"No frames decoded from {video_path}")

        stride = max(1, round(native_fps / self.fps))
        video = video[::stride].float()
        if video.shape[0] < self.num_frames:
            pad = video[-1:].repeat(self.num_frames - video.shape[0], 1, 1, 1)
            video = torch.cat([video, pad], dim=0)
        else:
            idx = torch.linspace(0, video.shape[0] - 1, steps=self.num_frames).long()
            video = video[idx]

        video = torch.nn.functional.interpolate(
            video,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        video = (video - mean) / std
        return video.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def run(
        self,
        prompt: str,
        impact_video_path: str,
        alpha_gate: float,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, str]:
        video_path = Path(impact_video_path).expanduser().resolve()
        frames = self._load_frames(video_path)
        input_ids, attention_mask = self._tokenize_prompt(prompt)
        outputs = self.model(
            frames=frames,
            input_ids=input_ids,
            attention_mask=attention_mask,
            alpha_scale=float(alpha_gate),
        )
        z_l = outputs["lingual_logits"][0]
        z_j = outputs["causal_logits"][0]
        fused_logits = outputs["fused_logits"][0]
        fused_probs = torch.softmax(fused_logits, dim=-1)

        z_l_rows: list[dict[str, Any]] = []
        z_j_rows: list[dict[str, Any]] = []
        p_f_rows: list[dict[str, Any]] = []
        for word in DISPLAY_WORDS:
            token_id = self.token_lookup[word]
            z_l_rows.append({"token": word, "value": float(z_l[token_id].item())})
            z_j_rows.append({"token": word, "value": float(z_j[token_id].item())})
            p_f_rows.append({"token": word, "value": float(fused_probs[token_id].item())})

        z_l_df = pd.DataFrame(z_l_rows)
        z_j_df = pd.DataFrame(z_j_rows)
        p_f_df = pd.DataFrame(p_f_rows).sort_values("value", ascending=False)

        llm_scores = torch.tensor(
            [float(z_l[self.token_lookup[word]].item()) for word in DISPLAY_WORDS],
            dtype=torch.float32,
        )
        causal_scores = torch.tensor(
            [float(z_j[self.token_lookup[word]].item()) for word in DISPLAY_WORDS],
            dtype=torch.float32,
        )
        fused_scores = torch.tensor(
            [float(fused_logits[self.token_lookup[word]].item()) for word in DISPLAY_WORDS],
            dtype=torch.float32,
        )
        llm_probs = torch.softmax(llm_scores, dim=-1)
        causal_probs = torch.softmax(causal_scores, dim=-1)
        fused_probs_demo = torch.softmax(fused_scores, dim=-1)
        fusion_lift = fused_probs_demo - llm_probs

        def _top_word(probs: torch.Tensor) -> tuple[str, float]:
            idx = int(probs.argmax().item())
            return DISPLAY_WORDS[idx], float(probs[idx].item())

        llm_kinetic_top, _ = _top_word(llm_probs)
        causal_kinetic_top, _ = _top_word(causal_probs)
        fused_kinetic_top, _ = _top_word(fused_probs_demo)
        lift_idx = int(fusion_lift.argmax().item())
        lift_word = DISPLAY_WORDS[lift_idx]
        lift_value = float(fusion_lift[lift_idx].item())

        p_f_display_df = pd.DataFrame(
            [
                {"token": word, "value": float(fusion_lift[i].item())}
                for i, word in enumerate(DISPLAY_WORDS)
            ]
        ).sort_values("value", ascending=False)

        llm_top, _ = _best_readable_token(z_l, self.tokenizer)
        causal_top, _ = _best_readable_token(z_j, self.tokenizer)
        fused_top, _ = _best_readable_token(fused_logits, self.tokenizer)
        raw_fused_top = self.tokenizer.decode([int(fused_logits.argmax().item())])
        alpha_note = (
            f"Kinetic set — LLM: `{llm_kinetic_top}` | Video (z_J): `{causal_kinetic_top}` | "
            f"Fused: `{fused_kinetic_top}` | **Fusion lift pick**: `{lift_word}` (+{lift_value:.1%}) | "
            f"Open-vocab fused: `{fused_top}` | Raw argmax: `{_format_token(raw_fused_top)}` | "
            f"alpha_gate={alpha_gate:.2f}. "
            "If fused always shows `shattered`, the **video** is dominating (fixed clip → fixed z_J). "
            "Try a Collide/Roll clip or lower alpha_gate."
        )
        completion = (
            f"Prompt (LLM) kinetic: {prompt.strip()} {llm_kinetic_top}\n"
            f"Video (z_J) kinetic: {prompt.strip()} {causal_kinetic_top}\n"
            f"Fusion lift pick: {prompt.strip()} {lift_word}\n"
            f"Open-vocab fused: {prompt.strip()} {fused_top}"
        )
        return z_l_df, z_j_df, p_f_display_df, alpha_note, completion


def build_app(
    default_config_path: str,
    default_checkpoint_path: str,
    default_impact_video_path: str,
    auto_load_model: bool,
) -> gr.Blocks:
    def load_engine(config_path: str, checkpoint_path: str) -> str:
        global LOADED_ENGINE
        try:
            LOADED_ENGINE = USPTransferEngine(config_path=config_path, checkpoint_path=checkpoint_path)
            msg = (
                f"Loaded model on `{LOADED_ENGINE.device}` from `{Path(checkpoint_path).name}`.\n\n"
                "Running inference on the default prompt/video..."
            )
            return msg
        except Exception as exc:
            LOADED_ENGINE = None
            return f"Model load failed: `{exc}`"

    def load_and_run(
        config_path: str,
        checkpoint_path: str,
        prompt: str,
        alpha_gate: float,
        uploaded_video: str | None,
        impact_video_path: str,
    ) -> tuple[str, Any, Any, Any, str, str]:
        status = load_engine(config_path, checkpoint_path)
        plots_and_outputs = run_demo(prompt, alpha_gate, uploaded_video, impact_video_path)
        return (status, *plots_and_outputs)

    def resolve_video_path(uploaded_video: str | None, impact_video_path: str) -> str:
        if uploaded_video and str(uploaded_video).strip():
            return str(uploaded_video)
        return impact_video_path

    def preview_video(uploaded_video: str | None, impact_video_path: str) -> str | None:
        chosen = resolve_video_path(uploaded_video, impact_video_path).strip()
        if not chosen:
            return None
        resolved = Path(chosen).expanduser().resolve()
        return str(resolved) if resolved.exists() else None

    def run_demo(
        prompt: str,
        alpha_gate: float,
        uploaded_video: str | None,
        impact_video_path: str,
    ) -> tuple[Any, Any, Any, str, str]:
        if LOADED_ENGINE is None:
            return (
                _empty_plot("System 1 (LLM Context, z_L)"),
                _empty_plot("System 2 (Causal Intuition, z_J)"),
                _empty_plot("Fused Probability (P_F)"),
                "Load a trained checkpoint first (real inference only; no mocked logits).",
                "",
            )
        chosen_video = resolve_video_path(uploaded_video, impact_video_path)
        if not chosen_video.strip():
            return (
                _empty_plot("System 1 (LLM Context, z_L)"),
                _empty_plot("System 2 (Causal Intuition, z_J)"),
                _empty_plot("Fused Probability (P_F)"),
                "Provide a video (upload or path) for real causal-logit inference.",
                "",
            )
        if not prompt.strip():
            return (
                _empty_plot("System 1 (LLM Context, z_L)"),
                _empty_plot("System 2 (Causal Intuition, z_J)"),
                _empty_plot("Fused Probability (P_F)"),
                "Enter a prompt to complete.",
                "",
            )

        try:
            z_l_df, z_j_df, p_f_df, top_note, completion = LOADED_ENGINE.run(
                prompt=prompt,
                impact_video_path=chosen_video,
                alpha_gate=alpha_gate,
            )
            return (
                _bar_figure(z_l_df, "System 1 (LLM Context, z_L)", "logit"),
                _bar_figure(z_j_df, "System 2 (Causal Intuition, z_J)", "logit"),
                _bar_figure(p_f_df, "Fusion lift over LLM (ΔP)", "Δ probability"),
                top_note,
                completion,
            )
        except Exception as exc:
            return (
                _empty_plot("System 1 (LLM Context, z_L)"),
                _empty_plot("System 2 (Causal Intuition, z_J)"),
                _empty_plot("Fused Probability (P_F)"),
                f"Inference failed: `{exc}`",
                "",
            )

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# CILF: Zero-Shot Cross-Domain Transfer (Simple Demo)")
        gr.Markdown(
            "Upload any impact-like video (or keep default dominoes), type any prompt, and run fusion. "
            "System 2 projects ODE state change into LLM hidden space (`h_causal`). "
            "Fusion injects `h_fused = h_t + alpha_gate * h_causal` and decodes once through the frozen LM head."
        )

        with gr.Accordion("Model Setup (Real Inference)", open=True):
            config_path = gr.Textbox(label="config_path", value=default_config_path)
            checkpoint_path = gr.Textbox(
                label="checkpoint_path",
                value=default_checkpoint_path,
                placeholder="runs/.../cilf_checkpoint_step_XXX.pt",
            )
            load_btn = gr.Button("Load Model")
            model_status = gr.Markdown("Load a trained model checkpoint to start.")

        prompt_box = gr.Textbox(
            label="Prompt",
            value="I was walking, hit my leg and",
            lines=2,
            placeholder="Type any prompt ending right before the next token...",
        )
        with gr.Row():
            uploaded_video = gr.File(label="Upload video (.mp4)", file_types=[".mp4"], type="filepath")
            impact_video_path = gr.Textbox(
                label="or use video path",
                value=default_impact_video_path,
                placeholder="Absolute or workspace-relative path to an impact clip (.mp4)",
            )
        alpha_gate = gr.Slider(
            label="alpha_gate",
            minimum=0.0,
            maximum=1.0,
            value=0.8,
            step=0.01,
        )
        run_btn = gr.Button("Run Zero-Shot Transfer", variant="primary")
        completion_box = gr.Textbox(label="Model outputs", lines=4, interactive=False)
        video_preview = gr.Video(
            label="Video used by System 2",
            value=default_impact_video_path if default_impact_video_path else None,
            interactive=False,
        )

        with gr.Row():
            system1_plot = gr.Plot(
                label="System 1 (LLM Context, z_L)",
                value=_empty_plot("System 1 (LLM Context, z_L)"),
            )
            system2_plot = gr.Plot(
                label="System 2 (Causal Intuition, z_J)",
                value=_empty_plot("System 2 (Causal Intuition, z_J)"),
            )
            fused_plot = gr.Plot(
                label="Fusion lift over LLM (ΔP on kinetic set)",
                value=_empty_plot("Fusion lift over LLM (ΔP on kinetic set)"),
            )

        inference_note = gr.Markdown()

        load_btn.click(
            fn=load_engine,
            inputs=[config_path, checkpoint_path],
            outputs=[model_status],
        )

        run_btn.click(
            fn=run_demo,
            inputs=[prompt_box, alpha_gate, uploaded_video, impact_video_path],
            outputs=[system1_plot, system2_plot, fused_plot, inference_note, completion_box],
        )
        alpha_gate.change(
            fn=run_demo,
            inputs=[prompt_box, alpha_gate, uploaded_video, impact_video_path],
            outputs=[system1_plot, system2_plot, fused_plot, inference_note, completion_box],
        )
        uploaded_video.change(
            fn=preview_video,
            inputs=[uploaded_video, impact_video_path],
            outputs=[video_preview],
        )
        impact_video_path.change(
            fn=preview_video,
            inputs=[uploaded_video, impact_video_path],
            outputs=[video_preview],
        )
        if auto_load_model:
            demo.load(
                fn=load_and_run,
                inputs=[config_path, checkpoint_path, prompt_box, alpha_gate, uploaded_video, impact_video_path],
                outputs=[model_status, system1_plot, system2_plot, fused_plot, inference_note, completion_box],
            )
        demo.load(
            fn=preview_video,
            inputs=[uploaded_video, impact_video_path],
            outputs=[video_preview],
        )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--impact-video", default=None)
    parser.add_argument("--no-auto-load", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = discover_default_config(args.config)
    checkpoint_path = discover_default_checkpoint(args.checkpoint)
    impact_video_path = discover_default_impact_video(args.impact_video)
    app = build_app(
        default_config_path=config_path,
        default_checkpoint_path=checkpoint_path,
        default_impact_video_path=impact_video_path,
        auto_load_model=not args.no_auto_load,
    )
    app.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
