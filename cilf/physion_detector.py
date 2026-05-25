"""Detectors tailored for Physion-style rendered clips.

Physion objects are saturated colored blocks on neutral gray/blue rooms.
YOLO-World does not transfer here; pure motion blobs merge touching dominoes
into one huge box (the failure mode you see in early visualizations).

**Default method ``color``** (recommended):

1. Segment high-saturation pixels and bucket by hue (red / yellow / green / …).
2. Run connected-components **per hue bucket** so each colored block gets its
   own box even when dominoes touch.
3. Apply per-frame NMS and a strict max box area (no “whole chain” boxes).
4. Track with IoU + center matching across frames.

**Legacy method ``motion``** keeps the old median-background diff only — still
available via ``method="motion"`` for debugging.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from cilf.track_io import ClipTracks, Track, TrackFrame


# OpenCV HSV hue ranges (H in 0..179)
HUE_RANGES: list[tuple[str, int, int]] = [
    ("red", 0, 10),
    ("red2", 170, 180),
    ("orange", 10, 22),
    ("yellow", 22, 38),
    ("green", 38, 85),
    ("cyan", 85, 100),
    ("blue", 100, 130),
    ("purple", 130, 155),
    ("pink", 155, 170),
]

# Dark blocks (black domino, shadows) — low saturation, mid value
DARK_V_MAX = 90
DARK_S_MAX = 80


def _hue_label(hue: float, sat: float, val: float) -> str:
    if sat < 50 and val < DARK_V_MAX:
        return "black"
    if sat < 45:
        return "gray"
    h = float(hue) % 180.0
    for name, lo, hi in HUE_RANGES:
        key = name.replace("2", "")
        if lo <= h < hi:
            return key
    return "other"


def _read_all_frames(video_path: Path, every_n: int = 1, max_frames: int | None = None):
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames: list[tuple[int, object]] = []
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if every_n <= 1 or idx % every_n == 0:
                frames.append((idx, frame))
                if max_frames is not None and len(frames) >= max_frames:
                    break
            idx += 1
    finally:
        cap.release()
    return frames, fps, idx


def _iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = float(inter_w * inter_h)
    area_a = float(max(0, xa2 - xa1) * max(0, ya2 - ya1))
    area_b = float(max(0, xb2 - xb1) * max(0, yb2 - yb1))
    return inter_area / (area_a + area_b - inter_area + 1e-6)


def _center_distance(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ca = ((box_a[0] + box_a[2]) * 0.5, (box_a[1] + box_a[3]) * 0.5)
    cb = ((box_b[0] + box_b[2]) * 0.5, (box_b[1] + box_b[3]) * 0.5)
    return ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5


def _box_area(box: tuple[int, int, int, int]) -> float:
    return float(max(0, box[2] - box[0]) * max(0, box[3] - box[1]))


def _containment_ratio(inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]) -> float:
    xa1, ya1, xa2, ya2 = inner
    xb1, yb1, xb2, yb2 = outer
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    return inter / max(_box_area(inner), 1.0)


def _nms_boxes(
    boxes: list[tuple[str, float, tuple[int, int, int, int]]],
    iou_threshold: float = 0.35,
) -> list[tuple[str, float, tuple[int, int, int, int]]]:
    """Greedy NMS + drop boxes mostly inside a larger box."""

    if len(boxes) <= 1:
        return boxes
    order = sorted(
        range(len(boxes)),
        key=lambda i: (boxes[i][1], _box_area(boxes[i][2])),
        reverse=True,
    )
    kept: list[tuple[str, float, tuple[int, int, int, int]]] = []
    for idx in order:
        candidate = boxes[idx]
        suppress = False
        for kept_box in kept:
            if _iou(candidate[2], kept_box[2]) > iou_threshold:
                suppress = True
                break
            if _containment_ratio(candidate[2], kept_box[2]) > 0.72:
                suppress = True
                break
        if not suppress:
            kept.append(candidate)
    return kept


def _assign_hue_bucket(hue: "object", sat: "object", val: "object", object_pixels: "object") -> "object":
    """Each pixel belongs to exactly one hue bucket (no double-counting in CC)."""

    import numpy as np

    height, width = hue.shape
    assignment = np.full((height, width), -1, dtype=np.int32)
    buckets: list[tuple[str, int, int, float]] = []
    for idx, (name, lo, hi) in enumerate(HUE_RANGES):
        center = (lo + hi) * 0.5 if lo < hi else 0.0
        buckets.append((name.replace("2", ""), lo, hi, center))

    ys, xs = np.where(object_pixels)
    for y, x in zip(ys, xs):
        h = float(hue[y, x])
        s = float(sat[y, x])
        v = float(val[y, x])
        if s < DARK_S_MAX and v < DARK_V_MAX:
            assignment[y, x] = len(HUE_RANGES)
            continue
        best_idx = -1
        best_dist = 1e9
        for idx, (_name, lo, hi, center) in enumerate(buckets):
            if lo < hi:
                if not (lo <= h < hi):
                    continue
                dist = abs(h - center)
            else:
                dist = min(abs(h - lo), abs(h - hi))
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        assignment[y, x] = best_idx
    return assignment


def _wall_mask(hsv, median_bg) -> "object":
    """Pixels that look like the static room (close to median background)."""

    import numpy as np

    diff_s = np.abs(hsv[:, :, 1].astype(np.int32) - median_bg[:, :, 1].astype(np.int32))
    diff_v = np.abs(hsv[:, :, 2].astype(np.int32) - median_bg[:, :, 2].astype(np.int32))
    return (diff_s < 30) & (diff_v < 35)


def _boxes_from_color_instances(
    frame_bgr,
    median_bg,
    *,
    min_area_ratio: float,
    max_area_ratio: float,
    min_sat: int,
    motion_mask=None,
) -> list[tuple[str, float, tuple[int, int, int, int]]]:
    """One box per saturated hue blob (Physion block)."""

    import cv2
    import numpy as np

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    height, width = hsv.shape[:2]
    total = float(height * width)
    wall = _wall_mask(hsv, median_bg)

    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    hue = hsv[:, :, 0]

    # Colored blocks: saturated and not wallpaper
    object_pixels = (sat >= min_sat) & (~wall) & (val > 35) & (val < 252)
    if motion_mask is not None:
        # Prefer moving pixels but keep saturated static blocks (standing dominoes)
        object_pixels = object_pixels & (
            (motion_mask > 0) | (sat >= min_sat + 25)
        )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    candidates: list[tuple[str, float, tuple[int, int, int, int]]] = []

    def _add_components(mask_bool, default_label: str) -> None:
        mask_u8 = (mask_bool.astype(np.uint8)) * 255
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        for label_id in range(1, num_labels):
            area = float(stats[label_id, cv2.CC_STAT_AREA])
            ratio = area / total
            if ratio < min_area_ratio or ratio > max_area_ratio:
                continue
            x = int(stats[label_id, cv2.CC_STAT_LEFT])
            y = int(stats[label_id, cv2.CC_STAT_TOP])
            w = int(stats[label_id, cv2.CC_STAT_WIDTH])
            h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
            if w < 8 or h < 8:
                continue
            aspect = w / max(1, h)
            if aspect > 4.5 or aspect < 0.15:
                continue
            extent = area / max(1.0, float(w * h))
            if extent < 0.42:
                continue
            comp = labels == label_id
            mean_h = float(np.mean(hue[comp]))
            mean_s = float(np.mean(sat[comp]))
            mean_v = float(np.mean(val[comp]))
            label = _hue_label(mean_h, mean_s, mean_v)
            if label == "gray":
                label = default_label
            # Confidence from saturation contrast vs background
            conf = float(min(1.0, 0.35 + mean_s / 255.0 * 0.5 + min(ratio * 8.0, 0.15)))
            candidates.append((label, conf, (x, y, x + w, y + h)))

    assignment = _assign_hue_bucket(hue, sat, val, object_pixels)
    num_buckets = len(HUE_RANGES) + 1  # +1 for dark
    for bucket_id in range(num_buckets):
        bucket_mask = assignment == bucket_id
        if not bucket_mask.any():
            continue
        default = "black" if bucket_id == len(HUE_RANGES) else HUE_RANGES[bucket_id][0].replace("2", "")
        _add_components(bucket_mask, default)

    return _nms_boxes(candidates, iou_threshold=0.35)


def _motion_mask_frame(hsv, median_bg, diff_threshold: int):
    import cv2
    import numpy as np

    diff_v = np.abs(hsv[:, :, 2].astype(np.int32) - median_bg[:, :, 2].astype(np.int32))
    diff_s = np.abs(hsv[:, :, 1].astype(np.int32) - median_bg[:, :, 1].astype(np.int32))
    diff = np.maximum(diff_v, diff_s).astype(np.uint8)
    _, mask = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def _track_boxes_over_frames(
    frames: list[tuple[int, object]],
    per_frame_boxes: dict[int, list[tuple[str, float, tuple[int, int, int, int]]]],
    width: int,
    height: int,
    *,
    iou_match_threshold: float,
    center_match_pixels: float,
    max_age: int,
) -> list[Track]:
    next_track_id = 1
    active: dict[int, dict[str, object]] = {}
    per_track_frames: dict[int, list[TrackFrame]] = defaultdict(list)
    per_track_label: dict[int, str] = {}

    for frame_index, _frame in frames:
        current_boxes = per_frame_boxes.get(frame_index, [])
        used_track_ids: set[int] = set()
        assignments: dict[int, int] = {}

        order = sorted(range(len(current_boxes)), key=lambda i: current_boxes[i][1], reverse=True)
        for idx in order:
            _, _, box = current_boxes[idx]
            best_track = None
            best_score = iou_match_threshold
            for tid, payload in active.items():
                if tid in used_track_ids:
                    continue
                iou_score = _iou(payload["box"], box)  # type: ignore[arg-type]
                if iou_score > best_score:
                    best_score = iou_score
                    best_track = tid
            if best_track is None:
                best_dist = center_match_pixels
                for tid, payload in active.items():
                    if tid in used_track_ids:
                        continue
                    dist = _center_distance(payload["box"], box)  # type: ignore[arg-type]
                    if dist < best_dist:
                        best_dist = dist
                        best_track = tid
            if best_track is None:
                best_track = next_track_id
                next_track_id += 1
            assignments[idx] = best_track
            used_track_ids.add(best_track)

        new_active: dict[int, dict[str, object]] = {}
        for idx, (label, conf, box) in enumerate(current_boxes):
            tid = assignments[idx]
            new_active[tid] = {"box": box, "label": label, "age": 0}
            if conf >= 0.5:
                per_track_label[tid] = label
            else:
                per_track_label.setdefault(tid, label)
            x1, y1, x2, y2 = box
            per_track_frames[tid].append(
                TrackFrame(
                    frame_index=frame_index,
                    box_norm=(
                        x1 / max(1.0, float(width)),
                        y1 / max(1.0, float(height)),
                        x2 / max(1.0, float(width)),
                        y2 / max(1.0, float(height)),
                    ),
                    conf=conf,
                )
            )
        for tid, payload in active.items():
            if tid in used_track_ids:
                continue
            age = int(payload.get("age", 0)) + 1
            if age <= max_age:
                new_active[tid] = {"box": payload["box"], "label": payload["label"], "age": age}
        active = new_active

    return [
        Track(track_id=tid, label=per_track_label.get(tid, "object"), frames=list(items))
        for tid, items in per_track_frames.items()
        if len(items) >= 2
    ]


def detect_tracks_physion(
    video_path: str | Path,
    method: str = "color",
    diff_threshold: int = 18,
    min_area_ratio: float = 0.0008,
    max_area_ratio: float = 0.06,
    min_sat: int = 55,
    iou_match_threshold: float = 0.25,
    center_match_pixels: float = 45.0,
    max_age: int = 8,
    every_n: int = 1,
    max_frames: int | None = None,
    bg_samples: int = 24,
) -> ClipTracks:
    """Track Physion objects.

    Args:
        method: ``color`` (default, per-hue instances + NMS) or ``motion`` (legacy).
        max_area_ratio: Cap per-box area; 0.06 ≈ one domino, not the whole chain.
    """

    import cv2
    import numpy as np

    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(path)

    frames, fps, total_frames = _read_all_frames(path, every_n=every_n, max_frames=max_frames)
    if not frames:
        return ClipTracks(video_path=str(path), fps=fps, num_frames=0, frame_size=(0, 0), tracks=[])

    height, width = frames[0][1].shape[:2]
    bg_indices = np.linspace(0, len(frames) - 1, min(bg_samples, len(frames))).astype(int)
    bg_stack = np.stack(
        [cv2.cvtColor(frames[i][1], cv2.COLOR_BGR2HSV) for i in bg_indices],
        axis=0,
    )
    median_bg = np.median(bg_stack, axis=0).astype(np.uint8)

    per_frame_boxes: dict[int, list[tuple[str, float, tuple[int, int, int, int]]]] = {}

    if method == "motion":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        for frame_index, frame in frames:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            motion = _motion_mask_frame(hsv, median_bg, diff_threshold)
            _, mask = cv2.threshold(motion, 127, 255, cv2.THRESH_BINARY)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            total = float(height * width)
            boxes: list[tuple[str, float, tuple[int, int, int, int]]] = []
            for label_id in range(1, num_labels):
                area = float(stats[label_id, cv2.CC_STAT_AREA])
                ratio = area / total
                if ratio < min_area_ratio or ratio > max_area_ratio * 4:
                    continue
                x, y, w, h = (
                    int(stats[label_id, cv2.CC_STAT_LEFT]),
                    int(stats[label_id, cv2.CC_STAT_TOP]),
                    int(stats[label_id, cv2.CC_STAT_WIDTH]),
                    int(stats[label_id, cv2.CC_STAT_HEIGHT]),
                )
                if w < 4 or h < 4:
                    continue
                comp = labels == label_id
                label = _hue_label(
                    float(np.mean(hsv[comp, 0])),
                    float(np.mean(hsv[comp, 1])),
                    float(np.mean(hsv[comp, 2])),
                )
                boxes.append((label, 0.5, (x, y, x + w, y + h)))
            per_frame_boxes[frame_index] = _nms_boxes(boxes)
    else:
        for frame_index, frame in frames:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            motion = _motion_mask_frame(hsv, median_bg, diff_threshold)
            per_frame_boxes[frame_index] = _boxes_from_color_instances(
                frame,
                median_bg,
                min_area_ratio=min_area_ratio,
                max_area_ratio=max_area_ratio,
                min_sat=min_sat,
                motion_mask=motion,
            )

    tracks = _track_boxes_over_frames(
        frames,
        per_frame_boxes,
        width,
        height,
        iou_match_threshold=iou_match_threshold,
        center_match_pixels=center_match_pixels,
        max_age=max_age,
    )

    return ClipTracks(
        video_path=str(path),
        fps=fps,
        num_frames=total_frames or (frames[-1][0] + 1),
        frame_size=(width, height),
        tracks=tracks,
    )


__all__ = ["detect_tracks_physion"]
