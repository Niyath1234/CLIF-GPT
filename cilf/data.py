"""General causal video-caption dataset loader."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from cilf.video_io import read_video
from cilf.vocab import DomainVocab, find_target_token_id


@dataclass(frozen=True)
class VideoCaptionRecord:
    video_path: Path
    prompt: str
    causal_consequence: str
    scenario: str | None
    stim_id: str | None
    split: str
    causal_trigger: bool | None
    metadata: dict[str, Any]


def _as_record(raw: dict[str, Any], manifest_dir: Path) -> VideoCaptionRecord:
    video_key = raw.get("video_path") or raw.get("video") or raw.get("path")
    if video_key is None:
        raise ValueError("Manifest row is missing video_path/video/path")

    caption = raw.get("caption") or raw.get("text") or ""
    prompt = raw.get("prompt") or raw.get("context") or caption
    causal_consequence = (
        raw.get("causal_consequence")
        or raw.get("target_text")
        or raw.get("outcome")
        or raw.get("label")
        or caption
    )
    video_path = Path(video_key)
    if not video_path.is_absolute():
        video_path = manifest_dir / video_path
    trigger = raw.get("causal_trigger")
    if not isinstance(trigger, bool):
        trigger = raw.get("contact") if isinstance(raw.get("contact"), bool) else None
    return VideoCaptionRecord(
        video_path=video_path,
        prompt=prompt,
        causal_consequence=str(causal_consequence),
        scenario=str(raw.get("scenario")) if raw.get("scenario") is not None else None,
        stim_id=str(raw.get("stim_id")) if raw.get("stim_id") is not None else None,
        split=str(raw.get("split") or "unknown"),
        causal_trigger=trigger,
        metadata={key: value for key, value in raw.items() if key not in {"video_path", "video", "path"}},
    )


def load_manifest(path: str) -> list[VideoCaptionRecord]:
    manifest_path = Path(path)
    records = []
    for line_no, raw_line in enumerate(manifest_path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            records.append(_as_record(json.loads(line), manifest_path.parent))
        except Exception as exc:
            raise ValueError(f"Invalid manifest row {line_no} in {manifest_path}: {exc}") from exc
    if not records:
        raise ValueError(f"No records found in {manifest_path}")
    return records


class GeneralCausalVideoDataset(Dataset):
    """Loads generic narrative-causal video rows and returns tokenized supervision."""

    def __init__(
        self,
        manifest_path: str,
        tokenizer,
        domain_vocab: DomainVocab | None = None,
        allowed_token_ids: list[int] | None = None,
        num_frames: int = 8,
        image_size: int = 224,
        fps: int = 10,
        max_prompt_length: int = 64,
    ) -> None:
        self.records = load_manifest(manifest_path)
        self.tokenizer = tokenizer
        self.domain_vocab = domain_vocab
        if self.domain_vocab is None:
            if allowed_token_ids is None:
                self.domain_vocab = DomainVocab(tokenizer=tokenizer, mode="full", terms=None)
            else:
                self.domain_vocab = DomainVocab(
                    tokenizer=tokenizer,
                    mode="bounded",
                    terms=None,
                    token_ids=[int(token_id) for token_id in allowed_token_ids],
                )
        self.allowed_token_ids = set(allowed_token_ids or self.domain_vocab.allowed_token_ids())
        self.num_frames = num_frames
        self.image_size = image_size
        self.fps = fps
        self.max_prompt_length = max_prompt_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        record = self.records[idx]
        frames = self._load_video(record.video_path)

        tokenized = self.tokenizer(
            record.prompt,
            truncation=True,
            max_length=self.max_prompt_length,
            padding="max_length",
            return_tensors="pt",
        )
        target_token_id = find_target_token_id(self.tokenizer, record.causal_consequence, self.allowed_token_ids)
        if target_token_id is None:
            raise ValueError(
                f"Target text '{record.causal_consequence}' contains no token in the active domain vocabulary."
            )

        return {
            "frames": frames,
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "target_token_id": torch.tensor(target_token_id, dtype=torch.long),
            "causal_trigger_label": torch.tensor(
                -1 if record.causal_trigger is None else int(record.causal_trigger),
                dtype=torch.long,
            ),
            "prompt": record.prompt,
            "causal_consequence": record.causal_consequence,
            "scenario": record.scenario or "unknown",
            "stim_id": record.stim_id or record.video_path.stem,
            "metadata": record.metadata,
        }

    def _load_video(self, path: Path) -> torch.Tensor:
        if not path.exists():
            raise FileNotFoundError(path)

        video, native_fps = read_video(path, target_fps=self.fps)
        if video.numel() == 0:
            raise ValueError(f"No frames decoded from {path}")
        stride = max(1, round(native_fps / self.fps))
        video = video[::stride].float()

        if video.shape[0] < self.num_frames:
            pad_count = self.num_frames - video.shape[0]
            pad = video[-1:].repeat(pad_count, 1, 1, 1)
            video = torch.cat([video, pad], dim=0)
        else:
            indices = torch.linspace(0, video.shape[0] - 1, steps=self.num_frames).long()
            video = video[indices]

        video = F.interpolate(
            video,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return (video - mean) / std
