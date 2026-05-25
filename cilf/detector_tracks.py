"""YOLO + ByteTrack adapter that turns an mp4 into a :class:`ClipTracks` object.

The detector is intentionally swappable: any function returning, per frame, a
list of ``(label, conf, box_xyxy_pixels, track_id_or_None)`` plus the original
frame size can be plugged in. The default implementation uses Ultralytics YOLO
because it ships with ByteTrack out of the box and gives us robust temporal IDs
for free.

We keep the heavy import (`ultralytics`) lazy so the module imports cleanly even
when the optional ``[detect]`` extra is not installed.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from cilf.track_io import ClipTracks, Track, TrackFrame


def _build_yolo(model_name: str, classes: list[str] | None, device: str | None):
    from ultralytics import YOLO  # type: ignore

    model = YOLO(model_name)
    if classes:
        # YOLO-World models support runtime class-vocabulary setting.
        if hasattr(model, "set_classes"):
            try:
                model.set_classes(classes)
            except Exception:
                pass
    if device:
        try:
            model.to(device)
        except Exception:
            pass
    return model


def _iter_video_frames(path: Path):
    import cv2  # lazy

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    try:
        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame_index, frame
            frame_index += 1
    finally:
        cap.release()
    return  # for static analysers


def detect_tracks(
    video_path: str | Path,
    model_name: str = "yolov8s-worldv2.pt",
    classes: list[str] | None = None,
    conf: float = 0.25,
    imgsz: int = 640,
    device: str | None = None,
    max_frames: int | None = None,
    every_n: int = 1,
) -> ClipTracks:
    """Run YOLO + ByteTrack over a video and return a :class:`ClipTracks`.

    Args:
        video_path: source clip.
        model_name: YOLO checkpoint (``yolov8s-worldv2.pt``, ``yolo11n.pt``,
            ``yolo11n-pose.pt`` etc.). World checkpoints support open-vocab via
            ``classes``.
        classes: optional list of strings that constrains the open-vocabulary
            detector to those labels. COCO models ignore this.
        conf: detection confidence threshold.
        imgsz: detector inference size.
        device: torch device override (``"mps"``, ``"cuda"``, ``"cpu"``).
        max_frames: optional cap on the number of frames processed.
        every_n: keep every Nth frame (still tracked across the gap because we
            call ``model.track(..., persist=True)`` on the kept frame stream).
    """

    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(path)

    model = _build_yolo(model_name, classes, device)
    predict_kwargs = {"conf": float(conf), "imgsz": int(imgsz), "verbose": False, "persist": True}
    if device:
        predict_kwargs["device"] = device

    width = 0
    height = 0
    frames_kept = 0
    per_track: dict[int, dict[str, object]] = defaultdict(lambda: {"label": "object", "frames": []})

    for frame_index, frame in _iter_video_frames(path):
        if every_n > 1 and frame_index % every_n != 0:
            continue
        if max_frames is not None and frames_kept >= max_frames:
            break
        if width == 0:
            height, width = int(frame.shape[0]), int(frame.shape[1])

        results = model.track(frame, **predict_kwargs)
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            frames_kept += 1
            continue

        names = getattr(result, "names", {}) or {}
        for box in boxes:
            cls_id = int(box.cls[0])
            label = str(names.get(cls_id, str(cls_id)))
            conf_value = float(box.conf[0])
            track_id_raw = getattr(box, "id", None)
            if track_id_raw is None:
                continue  # only keep boxes that ByteTrack assigned an id to
            track_id = int(track_id_raw[0])
            xyxy = box.xyxy[0].cpu().numpy().astype(float)
            x1 = max(0.0, float(xyxy[0])) / max(1.0, float(width))
            y1 = max(0.0, float(xyxy[1])) / max(1.0, float(height))
            x2 = min(1.0, float(xyxy[2]) / max(1.0, float(width)))
            y2 = min(1.0, float(xyxy[3]) / max(1.0, float(height)))
            if x2 <= x1 or y2 <= y1:
                continue
            per_track[track_id]["label"] = label
            per_track[track_id]["frames"].append(
                TrackFrame(frame_index=frame_index, box_norm=(x1, y1, x2, y2), conf=conf_value)
            )
        frames_kept += 1

    tracks = [
        Track(track_id=tid, label=str(payload["label"]), frames=list(payload["frames"]))  # type: ignore[arg-type]
        for tid, payload in per_track.items()
    ]

    fps_meta = 0.0
    try:
        import cv2

        cap = cv2.VideoCapture(str(path))
        fps_meta = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        cap.release()
    except Exception:
        fps_meta = 0.0

    return ClipTracks(
        video_path=str(path),
        fps=fps_meta,
        num_frames=frame_index + 1 if frames_kept else 0,
        frame_size=(width, height),
        tracks=tracks,
    )


def iter_world_class_palette() -> Iterable[str]:
    """Convenience label set covering the common Physion / household scenarios."""

    return [
        "ball",
        "block",
        "cube",
        "box",
        "cylinder",
        "ramp",
        "table",
        "platform",
        "shelf",
        "wall",
        "rope",
        "string",
        "cloth",
        "bottle",
        "cup",
        "can",
        "bag",
        "object",
    ]


__all__ = ["detect_tracks", "iter_world_class_palette"]
