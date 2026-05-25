"""Read/write per-clip object track files.

A track file is a JSON document with the schema below. Boxes are stored in
*normalised* coordinates so the same file can be consumed at any image size.

```
{
  "video_path": "PhysionTrain-Dynamics/.../foo.mp4",
  "fps": 10,
  "num_frames": 32,
  "frame_size": [W, H],
  "tracks": [
    {
      "track_id": 7,
      "label": "ball",
      "frames": [
        {"frame_index": 0, "box": [x1, y1, x2, y2], "conf": 0.91},
        {"frame_index": 1, "box": [x1, y1, x2, y2], "conf": 0.94},
        ...
      ]
    },
    ...
  ]
}
```

For training we project this into tensors indexed by ``[T, K, ...]`` where
``T`` is the number of subsampled frames the dataset returns and ``K`` is the
fixed track slot count (padded; ``mask`` distinguishes valid entries).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class TrackFrame:
    frame_index: int
    box_norm: tuple[float, float, float, float]
    conf: float


@dataclass(frozen=True)
class Track:
    track_id: int
    label: str
    frames: list[TrackFrame]


@dataclass(frozen=True)
class ClipTracks:
    video_path: str
    fps: float
    num_frames: int
    frame_size: tuple[int, int]
    tracks: list[Track]


def load_tracks(path: str | Path) -> ClipTracks:
    with Path(path).open("r") as handle:
        raw = json.load(handle)
    tracks: list[Track] = []
    for entry in raw.get("tracks", []):
        frames = [
            TrackFrame(
                frame_index=int(frame["frame_index"]),
                box_norm=tuple(float(value) for value in frame["box"]),
                conf=float(frame.get("conf", 1.0)),
            )
            for frame in entry.get("frames", [])
        ]
        tracks.append(
            Track(
                track_id=int(entry.get("track_id", -1)),
                label=str(entry.get("label", "object")),
                frames=frames,
            )
        )
    frame_size = raw.get("frame_size") or [0, 0]
    return ClipTracks(
        video_path=str(raw.get("video_path", "")),
        fps=float(raw.get("fps", 0.0)),
        num_frames=int(raw.get("num_frames", 0)),
        frame_size=(int(frame_size[0]), int(frame_size[1])),
        tracks=tracks,
    )


def save_tracks(path: str | Path, clip: ClipTracks) -> None:
    body: dict[str, Any] = {
        "video_path": clip.video_path,
        "fps": clip.fps,
        "num_frames": clip.num_frames,
        "frame_size": list(clip.frame_size),
        "tracks": [
            {
                "track_id": track.track_id,
                "label": track.label,
                "frames": [
                    {
                        "frame_index": frame.frame_index,
                        "box": list(frame.box_norm),
                        "conf": frame.conf,
                    }
                    for frame in track.frames
                ],
            }
            for track in clip.tracks
        ],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w") as handle:
        json.dump(body, handle)


def _interpolate(prev: TrackFrame | None, nxt: TrackFrame | None, index: int) -> tuple[list[float], float]:
    if prev is None and nxt is None:
        return [0.0, 0.0, 0.0, 0.0], 0.0
    if prev is None:
        return list(nxt.box_norm), 0.0  # only future observation; treat as invalid
    if nxt is None:
        return list(prev.box_norm), 0.0
    span = max(1, nxt.frame_index - prev.frame_index)
    alpha = (index - prev.frame_index) / span
    box = [prev.box_norm[i] * (1.0 - alpha) + nxt.box_norm[i] * alpha for i in range(4)]
    conf = float(min(prev.conf, nxt.conf) * 0.5)
    return box, conf


def project_to_tensor(
    clip: ClipTracks,
    selected_frame_indices: list[int],
    max_tracks: int,
    min_track_frames: int = 2,
    score_floor: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Return ``(boxes [T, K, 4], mask [T, K], labels [K])``.

    For every track we look up the box at each *selected source frame index*.
    Missing observations are linearly interpolated between the two nearest
    detections of the same track; if no neighbouring detection exists, the box
    is filled with zeros and ``mask`` is 0.

    Tracks are ranked by ``mean_conf * sqrt(num_frames_visible)`` and the top
    ``max_tracks`` are kept. Tracks with fewer than ``min_track_frames``
    detections are dropped.
    """

    num_frames = len(selected_frame_indices)
    if num_frames == 0:
        return (
            torch.zeros(0, max_tracks, 4, dtype=torch.float32),
            torch.zeros(0, max_tracks, dtype=torch.float32),
            ["" for _ in range(max_tracks)],
        )

    scored: list[tuple[float, Track]] = []
    for track in clip.tracks:
        if len(track.frames) < min_track_frames:
            continue
        mean_conf = sum(f.conf for f in track.frames) / max(1, len(track.frames))
        if mean_conf < score_floor:
            continue
        score = mean_conf * (len(track.frames) ** 0.5)
        scored.append((score, track))
    scored.sort(key=lambda item: item[0], reverse=True)
    selected_tracks = [t for _, t in scored[:max_tracks]]

    boxes = torch.zeros(num_frames, max_tracks, 4, dtype=torch.float32)
    mask = torch.zeros(num_frames, max_tracks, dtype=torch.float32)
    labels: list[str] = ["" for _ in range(max_tracks)]

    for slot, track in enumerate(selected_tracks):
        labels[slot] = track.label
        sorted_frames = sorted(track.frames, key=lambda f: f.frame_index)
        indices = [f.frame_index for f in sorted_frames]
        for ti, frame_index in enumerate(selected_frame_indices):
            exact: TrackFrame | None = None
            prev: TrackFrame | None = None
            nxt: TrackFrame | None = None
            for tracked_frame, src_index in zip(sorted_frames, indices):
                if src_index == frame_index:
                    exact = tracked_frame
                    break
                if src_index < frame_index:
                    prev = tracked_frame
                else:
                    nxt = tracked_frame
                    break
            if exact is not None:
                box = list(exact.box_norm)
                conf = exact.conf
                valid = 1.0
            else:
                box, conf = _interpolate(prev, nxt, frame_index)
                valid = 1.0 if (prev is not None and nxt is not None and conf > 0.0) else 0.0
            boxes[ti, slot, :] = torch.tensor(box, dtype=torch.float32)
            mask[ti, slot] = float(valid * (1.0 if conf >= score_floor else 0.0))

    return boxes, mask, labels


__all__ = [
    "ClipTracks",
    "Track",
    "TrackFrame",
    "load_tracks",
    "save_tracks",
    "project_to_tensor",
]
