"""General causal video-caption dataset loader."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from cilf.track_io import ClipTracks, load_tracks, project_to_tensor
from cilf.video_io import read_video
from cilf.vocab import DomainVocab, find_target_token_id


@dataclass(frozen=True)
class VideoCaptionRecord:
    """One manifest row.

    Required fields are ``video_path``, ``prompt`` and ``causal_consequence``.
    Everything else is optional and exposed for downstream losses and eval.
    """

    video_path: Path
    prompt: str
    causal_consequence: str
    scenario: str | None
    stim_id: str | None
    split: str
    causal_trigger: bool | None
    abstract_dynamics: str | None
    objects: list[str] | None
    subject_object: str | None
    affected_object: str | None
    interaction: str | None
    precondition: str | None
    postcondition: str | None
    causal_state_change: str | None
    counterfactual_prompt: str | None
    tracks: Any | None
    metadata: dict[str, Any]


def _opt_str(raw: dict[str, Any], key: str) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    return str(value)


def _opt_object_list(raw: dict[str, Any], key: str) -> list[str] | None:
    value = raw.get(key)
    if value is None:
        return None
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


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

    reserved = {"video_path", "video", "path"}
    return VideoCaptionRecord(
        video_path=video_path,
        prompt=prompt,
        causal_consequence=str(causal_consequence),
        scenario=_opt_str(raw, "scenario"),
        stim_id=_opt_str(raw, "stim_id"),
        split=str(raw.get("split") or "unknown"),
        causal_trigger=trigger,
        abstract_dynamics=_opt_str(raw, "abstract_dynamics"),
        objects=_opt_object_list(raw, "objects"),
        subject_object=_opt_str(raw, "subject_object"),
        affected_object=_opt_str(raw, "affected_object"),
        interaction=_opt_str(raw, "interaction"),
        precondition=_opt_str(raw, "precondition"),
        postcondition=_opt_str(raw, "postcondition"),
        causal_state_change=_opt_str(raw, "causal_state_change"),
        counterfactual_prompt=_opt_str(raw, "counterfactual_prompt"),
        tracks=raw.get("tracks"),
        metadata={key: value for key, value in raw.items() if key not in reserved},
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
        tracks_dir: str | Path | None = None,
        max_tracks: int = 0,
    ) -> None:
        self.manifest_path = Path(manifest_path)
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
        self.tracks_dir = Path(tracks_dir) if tracks_dir else None
        self.max_tracks = int(max_tracks)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        record = self.records[idx]
        frames, frame_source_indices = self._load_video(record.video_path)

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

        track_boxes, track_mask, track_labels = self._load_tracks(record, frame_source_indices)

        item: dict[str, torch.Tensor | str | list[str]] = {
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
            "abstract_dynamics": record.abstract_dynamics or "unknown_dynamics",
            "objects": ",".join(record.objects) if record.objects else "",
            "subject_object": record.subject_object or "",
            "affected_object": record.affected_object or "",
            "interaction": record.interaction or "",
            "precondition": record.precondition or "",
            "postcondition": record.postcondition or "",
            "causal_state_change": record.causal_state_change or "",
            "counterfactual_prompt": record.counterfactual_prompt or "",
            "track_boxes": track_boxes,
            "track_mask": track_mask,
            "track_labels": ",".join(track_labels),
        }
        # Metadata is intentionally omitted from __getitem__ because rows can
        # carry different keys (causal_trigger, contact, ...), which crashes
        # the default DataLoader collator. The original record is still on
        # ``self.records[idx]`` for any code that needs it.
        return item

    def _load_video(self, path: Path) -> tuple[torch.Tensor, list[int]]:
        if not path.exists():
            raise FileNotFoundError(path)

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
            original_indices = torch.cat([original_indices, pad_indices], dim=0)
            chosen_source = original_indices
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
        return (video - mean) / std, [int(value) for value in chosen_source.tolist()]

    def _track_file_for(self, record: "VideoCaptionRecord") -> Path | None:
        if self.tracks_dir is None:
            return None
        stim_id = (record.stim_id or "").strip()
        name = f"{stim_id}.json" if stim_id else f"{record.video_path.stem}.json"
        return self.tracks_dir / name

    def _empty_tracks(self) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        K = max(0, self.max_tracks)
        return (
            torch.zeros(self.num_frames, K, 4, dtype=torch.float32),
            torch.zeros(self.num_frames, K, dtype=torch.float32),
            ["" for _ in range(K)],
        )

    def _load_tracks(
        self,
        record: VideoCaptionRecord,
        frame_source_indices: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        if self.max_tracks <= 0 or self.tracks_dir is None:
            return self._empty_tracks()
        track_path = self._track_file_for(record)
        if track_path is None or not track_path.exists():
            return self._empty_tracks()
        try:
            clip: ClipTracks = load_tracks(track_path)
        except Exception:
            return self._empty_tracks()
        return project_to_tensor(clip, frame_source_indices, self.max_tracks)
