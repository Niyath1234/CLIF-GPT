"""Retrieve a Physion clip + fused conclusion for an open-ended user prompt."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from transformers import AutoTokenizer

from cilf.data import VideoCaptionRecord, load_manifest
from cilf.track_io import ClipTracks, load_tracks, project_to_tensor
from cilf.train import choose_device, make_model, move_batch
from cilf.video_io import read_video

# Causal next-words the model was trained on (distilgpt2 single-token ids).
KINETIC_WORDS = [
    "fell",
    "fall",
    "falls",
    "crashed",
    "crash",
    "collided",
    "rolled",
    "rolling",
    "roll",
    "shattered",
    "shatter",
    "broke",
    "dropped",
    "drop",
    "settled",
    "stopped",
    "hit",
    "slipped",
    "tripped",
    "caught",
]

# When open-vocab fusion is noisy, map physics label → readable completion.
DYNAMICS_OUTCOME = {
    "support_loss_freefall": "fell",
    "support_loss_chain_fall": "fell",
    "momentum_transfer_collision": "crashed",
    "rolling_motion": "rolled",
    "containment_settle": "settled",
    "contact_capture": "caught",
}

# Keyword hints: steer retrieval + resolve fusion vs video-label conflicts.
PROMPT_HINT_RULES: list[tuple[tuple[str, ...], str | None, frozenset[str]]] = [
    (
        ("car", "cars", "intersection", "truck", "sedan", "rear-ended", "cyclist", "curb"),
        "momentum_transfer_collision",
        frozenset({"crashed", "collided", "crash", "collide"}),
    ),
    (
        ("ball", "hill", "rolled", "rolling", "down the hill"),
        "rolling_motion",
        frozenset({"rolled", "rolling", "roll"}),
    ),
    (
        ("cup", "vase", "counter", "table", "shatter", "shattered", "broke"),
        None,
        frozenset({"shattered", "broke", "fell", "crashed"}),
    ),
    (
        ("slip", "slipped", "banana", "trip", "tripped", "balance", "stair", "walking", "leg"),
        None,
        frozenset({"fell", "fall", "falls"}),
    ),
    (
        ("domino", "dominoes", "topple", "chain"),
        "support_loss_chain_fall",
        frozenset({"fell", "fall"}),
    ),
]


def _prompt_hints(prompt: str) -> tuple[str | None, frozenset[str]]:
    low = prompt.lower()
    for keywords, dynamics, outcomes in PROMPT_HINT_RULES:
        if any(word in low for word in keywords):
            return dynamics, outcomes
    return None, frozenset()


def _choose_completion_word(
    *,
    kinetic_token: str,
    kinetic_conf: float,
    kinetic_ranked: list[tuple[str, float]],
    video_outcome: str,
    clip: ClipIndexEntry,
    prompt: str,
) -> tuple[str, float, str]:
    """Merge fused logits with clip label; avoid always-'fell' or junk tokens."""
    outcome_prob = 0.0
    for word, prob in kinetic_ranked:
        if word.lower() == video_outcome.lower():
            outcome_prob = prob
            break

    hint_dyn, hint_outcomes = _prompt_hints(prompt)
    kinetic_lower = kinetic_token.lower()
    video_lower = video_outcome.lower()

    # Slip/fall prompts: trust support-loss clip label when fusion picks wrong verb.
    if hint_outcomes and video_lower in hint_outcomes and kinetic_lower not in hint_outcomes:
        dyn = clip.abstract_dynamics or ""
        dynamics_ok = (hint_dyn is None and dyn.startswith("support_loss")) or (
            hint_dyn is not None and dyn == hint_dyn
        )
        if dynamics_ok:
            conf = outcome_prob if outcome_prob >= 0.05 else 0.75
            return video_outcome, conf, "video_outcome"

    # Fusion agrees strongly with prompt (e.g. crashed at 91%) → use fusion.
    if kinetic_conf >= 0.3 and kinetic_lower != video_lower:
        block_fusion = (
            hint_outcomes
            and kinetic_lower not in hint_outcomes
            and video_lower in hint_outcomes
        )
        if not block_fusion and kinetic_conf >= max(outcome_prob * 2.0, 0.12):
            return kinetic_token, kinetic_conf, "kinetic_fusion"

    if kinetic_lower == video_lower:
        return kinetic_token, max(kinetic_conf, outcome_prob, 0.5), "kinetic_fusion"

    if outcome_prob >= 0.15:
        return video_outcome, outcome_prob, "video_outcome"

    if kinetic_conf >= 0.12:
        return kinetic_token, kinetic_conf, "kinetic_fusion"

    return video_outcome, max(outcome_prob, 0.75), "video_physics"


def _format_token(token: str) -> str:
    cleaned = token.strip()
    if cleaned and cleaned.isprintable() and cleaned.isascii():
        return cleaned
    return repr(token)


def _is_clean_word(word: str) -> bool:
    w = word.strip()
    if len(w) < 2 or len(w) > 24:
        return False
    if not w.replace("-", "").isalpha():
        return False
    low = w.lower()
    junk = ("cloneembed", "magikarp", "orderable", "reportprint", "ailability", "embed")
    return not any(fragment in low for fragment in junk)


def _resolve_word_token(tokenizer, word: str) -> int | None:
    for cand in (f" {word}", word, f" {word.capitalize()}"):
        ids = tokenizer.encode(cand, add_special_tokens=False)
        if len(ids) == 1:
            return int(ids[0])
    return None


def _build_kinetic_token_lookup(tokenizer) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for word in KINETIC_WORDS:
        token_id = _resolve_word_token(tokenizer, word)
        if token_id is not None:
            mapping[word] = token_id
    if not mapping:
        raise ValueError("Could not map any kinetic words to tokenizer ids.")
    return mapping


def _best_kinetic_token(
    logits: torch.Tensor,
    token_lookup: dict[str, int],
) -> tuple[str, float, list[tuple[str, float]]]:
    words = list(token_lookup.keys())
    scores = torch.tensor([float(logits[token_lookup[w]].item()) for w in words])
    probs = torch.softmax(scores, dim=0)
    ranked = sorted(
        [(words[i], float(probs[i].item())) for i in range(len(words))],
        key=lambda item: -item[1],
    )
    top_word, top_prob = ranked[0]
    return top_word, top_prob, ranked[:6]


def _best_readable_token(logits: torch.Tensor, tokenizer, k: int = 64) -> tuple[str, float]:
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, k=min(k, logits.numel()))
    for token_id, prob in zip(top_ids.tolist(), top_probs.tolist()):
        text = tokenizer.decode([int(token_id)]).strip()
        if _is_clean_word(text):
            return text, float(prob)
    token_id = int(logits.argmax().item())
    return _format_token(tokenizer.decode([token_id])), float(probs[token_id].item())


def normalize_prompt(question: str) -> str:
    """Turn free-form questions into GPT-style next-token completion prompts."""
    q = (question or "").strip()
    if not q:
        return "Something happened and "

    if q.endswith("?"):
        lower = q.lower().rstrip("?").strip()
        for prefix in (
            "what happens if ",
            "what happens when ",
            "what would happen if ",
            "what will happen when ",
            "what would happen when ",
        ):
            if lower.startswith(prefix):
                lower = lower[len(prefix) :].strip()
                break
        if lower.startswith("why do ") or lower.startswith("why does "):
            lower = lower.split(" ", 2)[-1].strip()
        elif " when " in lower:
            lower = lower.split(" when ", 1)[-1].strip()
        q = lower[0].upper() + lower[1:] if lower else lower

    q = q.rstrip()
    if not q[0].isupper():
        q = q[0].upper() + q[1:]
    if q.endswith(" and"):
        return q + " "
    if " and" not in q[-24:].lower():
        q = f"{q} and"
    return q + " "


def build_conclusion(prompt: str, next_token: str) -> str:
    base = prompt.rstrip()
    word = next_token.strip()
    if not word:
        return base + "."
    if base.endswith(" and"):
        return f"{base} {word}."
    return f"{base} {word}."


@dataclass(frozen=True)
class ClipIndexEntry:
    stim_id: str
    video_path: Path
    train_prompt: str
    causal_consequence: str
    abstract_dynamics: str
    causal_state_change: str
    scenario: str
    interaction: str


@dataclass
class IntuitionResult:
    question: str
    prompt: str
    conclusion: str
    next_token: str
    confidence: float
    llm_top_token: str
    stim_id: str
    video_path: str
    abstract_dynamics: str
    causal_state_change: str
    train_prompt: str
    retrieval_score: float
    fusion_score: float
    top_alternatives: list[tuple[str, float]]
    completion_source: str
    video_outcome: str


class IntuitionEngine:
    """CILF inference with text retrieval over a Physion clip bank."""

    def __init__(
        self,
        config_path: str | Path,
        checkpoint_path: str | Path,
        manifest_path: str | Path,
        *,
        index_cache: str | Path | None = None,
        candidate_pool: int = 12,
        alpha_scale: float = 1.0,
    ) -> None:
        self.config_path = Path(config_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.manifest_path = Path(manifest_path)
        self.index_cache = Path(index_cache) if index_cache else None
        self.candidate_pool = max(1, int(candidate_pool))
        self.alpha_scale = float(alpha_scale)

        with self.config_path.open("r", encoding="utf-8") as handle:
            self.config = yaml.safe_load(handle)

        self.device = choose_device(str(self.config.get("device", "auto")))
        self.model = make_model(self.config, self.device)
        state = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state["model"], strict=False)
        self.model.eval()

        model_cfg = self.config["model"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_cfg["llm_name"])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        data_cfg = self.config.get("data", {})
        self.num_frames = int(data_cfg.get("num_frames", 8))
        self.image_size = int(data_cfg.get("image_size", 224))
        self.fps = int(data_cfg.get("fps", 8))
        self.max_prompt_length = int(data_cfg.get("max_prompt_length", 48))
        tracks_dir = data_cfg.get("tracks_dir")
        self.tracks_dir = Path(tracks_dir) if tracks_dir else None
        self.max_tracks = int(data_cfg.get("max_tracks", 0))

        self.clips = self._build_clip_index()
        self.token_lookup = _build_kinetic_token_lookup(self.tokenizer)
        self._prompt_embeds = self._load_or_build_prompt_index()

    def _build_clip_index(self) -> list[ClipIndexEntry]:
        records = load_manifest(str(self.manifest_path))
        by_stim: dict[str, VideoCaptionRecord] = {}
        for record in records:
            stim = (record.stim_id or record.video_path.stem).strip()
            if stim not in by_stim:
                by_stim[stim] = record

        clips: list[ClipIndexEntry] = []
        for stim_id, record in sorted(by_stim.items()):
            if not record.video_path.exists():
                continue
            clips.append(
                ClipIndexEntry(
                    stim_id=stim_id,
                    video_path=record.video_path.resolve(),
                    train_prompt=record.prompt,
                    causal_consequence=record.causal_consequence,
                    abstract_dynamics=record.abstract_dynamics or "unknown",
                    causal_state_change=record.causal_state_change or "",
                    scenario=record.scenario or "",
                    interaction=record.interaction or "",
                )
            )
        if not clips:
            raise FileNotFoundError(
                f"No playable clips found for manifest {self.manifest_path}. "
                "Check Physion video paths."
            )
        return clips

    def _tokenize(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_prompt_length,
            padding="max_length",
            return_tensors="pt",
        )
        return encoded["input_ids"].to(self.device), encoded["attention_mask"].to(self.device)

    @torch.no_grad()
    def _embed_prompt(self, prompt: str) -> torch.Tensor:
        input_ids, attention_mask = self._tokenize(prompt)
        outputs = self.model.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        last_index = int(attention_mask.sum(dim=1).item()) - 1
        hidden = outputs.hidden_states[-1][0, last_index]
        return F.normalize(hidden.float(), dim=0).cpu()

    def _load_or_build_prompt_index(self) -> torch.Tensor:
        if self.index_cache and self.index_cache.exists():
            payload = torch.load(self.index_cache, map_location="cpu")
            if payload.get("manifest") == str(self.manifest_path.resolve()):
                return payload["embeds"]

        embeds = torch.stack([self._embed_prompt(c.train_prompt) for c in self.clips], dim=0)
        if self.index_cache:
            self.index_cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "manifest": str(self.manifest_path.resolve()),
                    "stim_ids": [c.stim_id for c in self.clips],
                    "embeds": embeds,
                },
                self.index_cache,
            )
        return embeds

    def _load_video(self, path: Path) -> tuple[torch.Tensor, list[int]]:
        video, native_fps = read_video(path, target_fps=self.fps)
        if video.numel() == 0:
            raise ValueError(f"No frames decoded from {path}")

        stride = max(1, round(native_fps / self.fps))
        strided = video[::stride].float()
        original_indices = torch.arange(0, video.shape[0], stride).long()
        original_indices = original_indices[: strided.shape[0]]

        if strided.shape[0] < self.num_frames:
            pad_count = self.num_frames - strided.shape[0]
            pad = strided[-1:].repeat(pad_count, 1, 1, 1)
            strided = torch.cat([strided, pad], dim=0)
            last_index = int(original_indices[-1].item()) if original_indices.numel() else 0
            pad_indices = torch.full((pad_count,), last_index, dtype=torch.long)
            chosen_source = torch.cat([original_indices, pad_indices], dim=0)
        else:
            indices = torch.linspace(0, strided.shape[0] - 1, steps=self.num_frames).long()
            strided = strided[indices]
            chosen_source = original_indices[indices]

        video = F.interpolate(
            strided,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        frames = (video - mean) / std
        return frames, [int(v) for v in chosen_source.tolist()]

    def _load_tracks(
        self,
        stim_id: str,
        frame_source_indices: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k = max(0, self.max_tracks)
        empty = (
            torch.zeros(self.num_frames, k, 4, dtype=torch.float32),
            torch.zeros(self.num_frames, k, dtype=torch.float32),
        )
        if k <= 0 or self.tracks_dir is None:
            return empty
        track_path = self.tracks_dir / f"{stim_id}.json"
        if not track_path.exists():
            return empty
        try:
            clip: ClipTracks = load_tracks(track_path)
        except Exception:
            return empty
        boxes, mask, _labels = project_to_tensor(clip, frame_source_indices, self.max_tracks)
        return boxes, mask

    @torch.no_grad()
    def _forward_clip(self, prompt: str, clip: ClipIndexEntry) -> dict[str, Any]:
        frames, frame_indices = self._load_video(clip.video_path)
        track_boxes, track_mask = self._load_tracks(clip.stim_id, frame_indices)
        input_ids, attention_mask = self._tokenize(prompt)

        batch = move_batch(
            {
                "frames": frames.unsqueeze(0),
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "track_boxes": track_boxes.unsqueeze(0),
                "track_mask": track_mask.unsqueeze(0),
            },
            self.device,
        )
        outputs = self.model(
            frames=batch["frames"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            alpha_scale=self.alpha_scale,
            track_boxes=batch.get("track_boxes"),
            track_mask=batch.get("track_mask"),
        )
        z_l = outputs["lingual_logits"][0]
        z_f = outputs["fused_logits"][0]
        llm_token, _ = _best_readable_token(z_l, self.tokenizer)

        kinetic_token, kinetic_conf, kinetic_ranked = _best_kinetic_token(z_f, self.token_lookup)
        open_token, open_conf = _best_readable_token(z_f, self.tokenizer)

        raw_outcome = clip.causal_consequence.strip()
        video_outcome = raw_outcome.split()[0] if raw_outcome else ""
        if not video_outcome:
            video_outcome = DYNAMICS_OUTCOME.get(clip.abstract_dynamics, kinetic_token)

        top_token, confidence, completion_source = _choose_completion_word(
            kinetic_token=kinetic_token,
            kinetic_conf=kinetic_conf,
            kinetic_ranked=kinetic_ranked,
            video_outcome=video_outcome,
            clip=clip,
            prompt=prompt,
        )

        return {
            "top_token": top_token,
            "confidence": confidence,
            "llm_top_token": llm_token,
            "alternatives": kinetic_ranked,
            "fusion_score": kinetic_conf,
            "completion_source": completion_source,
            "video_outcome": video_outcome,
            "kinetic_top": kinetic_token,
            "open_top": open_token,
        }

    def _retrieve_candidates(self, prompt: str) -> list[tuple[ClipIndexEntry, float]]:
        query = self._embed_prompt(prompt)
        sims = torch.mv(self._prompt_embeds, query).clone()

        hint_dyn, hint_outcomes = _prompt_hints(prompt)
        prompt_words = set(prompt.lower().split())
        for idx, clip in enumerate(self.clips):
            boost = 0.0
            if hint_dyn and clip.abstract_dynamics == hint_dyn:
                boost += 0.18
            elif hint_dyn and hint_dyn.startswith("support_loss") and (
                clip.abstract_dynamics or ""
            ).startswith("support_loss"):
                boost += 0.08
            if hint_outcomes and clip.causal_consequence.lower() in hint_outcomes:
                boost += 0.12
            if hint_outcomes == frozenset({"fell", "fall", "falls"}) and clip.causal_consequence.lower() not in hint_outcomes:
                boost -= 0.1
            train_words = set(clip.train_prompt.lower().split())
            overlap = len(prompt_words & train_words) / max(1, len(prompt_words))
            boost += overlap * 0.25
            sims[idx] += boost

        k = min(self.candidate_pool, sims.numel())
        values, indices = torch.topk(sims, k=k)
        return [(self.clips[int(i)], float(v)) for v, i in zip(values.tolist(), indices.tolist())]

    @torch.no_grad()
    def ask(self, question: str) -> IntuitionResult:
        prompt = normalize_prompt(question)
        candidates = self._retrieve_candidates(prompt)

        hint_dyn, hint_outcomes = _prompt_hints(prompt)
        best: IntuitionResult | None = None
        for clip, retrieval_score in candidates:
            out = self._forward_clip(prompt, clip)
            hint_bonus = 0.0
            if hint_dyn and clip.abstract_dynamics == hint_dyn:
                hint_bonus = 0.15
            if hint_outcomes and out["top_token"].lower() in hint_outcomes:
                hint_bonus += 0.2
            rank_score = retrieval_score * 0.55 + out["fusion_score"] * 0.25 + hint_bonus
            result = IntuitionResult(
                question=question.strip(),
                prompt=prompt.rstrip(),
                conclusion=build_conclusion(prompt.rstrip(), out["top_token"]),
                next_token=out["top_token"],
                confidence=out["confidence"],
                llm_top_token=out["llm_top_token"],
                stim_id=clip.stim_id,
                video_path=str(clip.video_path),
                abstract_dynamics=clip.abstract_dynamics,
                causal_state_change=clip.causal_state_change,
                train_prompt=clip.train_prompt,
                retrieval_score=retrieval_score,
                fusion_score=out["fusion_score"],
                top_alternatives=out["alternatives"],
                completion_source=out["completion_source"],
                video_outcome=out["video_outcome"],
            )
            if best is None or rank_score > (
                best.retrieval_score * 0.7 + best.fusion_score * 0.3
            ):
                best = result

        if best is None:
            raise RuntimeError("No candidate clip could be scored.")
        return best

    def result_to_json(self, result: IntuitionResult) -> dict[str, Any]:
        return {
            "question": result.question,
            "prompt": result.prompt,
            "conclusion": result.conclusion,
            "next_token": result.next_token,
            "confidence": round(result.confidence, 4),
            "llm_top_token": result.llm_top_token,
            "stim_id": result.stim_id,
            "video_url": f"/api/video/{result.stim_id}",
            "abstract_dynamics": result.abstract_dynamics,
            "causal_state_change": result.causal_state_change,
            "train_prompt": result.train_prompt,
            "retrieval_score": round(result.retrieval_score, 4),
            "fusion_score": round(result.fusion_score, 4),
            "top_alternatives": [
                {"token": t, "prob": round(p, 4)} for t, p in result.top_alternatives
            ],
            "completion_source": result.completion_source,
            "video_outcome": result.video_outcome,
        }

    def stim_id_to_video_path(self, stim_id: str) -> Path | None:
        for clip in self.clips:
            if clip.stim_id == stim_id:
                return clip.video_path
        return None
