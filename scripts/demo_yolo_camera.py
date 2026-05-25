#!/usr/bin/env python3
"""Live webcam demo using Ultralytics YOLO (online SOTA object detection).

Uses models from https://github.com/ultralytics/ultralytics — downloads weights
on first run.

Modes:
  world  — YOLO-World open-vocabulary: person, face, hand, head (best for you)
  pose   — YOLO pose: person skeleton + tight face/hand boxes from keypoints
  coco   — YOLO11 COCO: person + common objects (80 classes)

Tracking uses ByteTrack built into Ultralytics (``model.track(..., persist=True)``).

Usage:
  pip install ultralytics   # or: pip install -e \".[detect]\"
  python scripts/demo_yolo_camera.py
  python scripts/demo_yolo_camera.py --mode world
  python scripts/demo_yolo_camera.py --mode pose --model yolo11n-pose.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "demo_outputs"

# BGR colors per label (case-insensitive match on first token)
LABEL_COLORS: dict[str, tuple[int, int, int]] = {
    "person": (60, 180, 60),
    "face": (60, 120, 255),
    "hand": (255, 160, 60),
    "head": (200, 60, 200),
    "default": (220, 220, 60),
}

WORLD_CLASSES = ["person", "face", "hand", "head"]
COCO_SUBJECT_CLASSES = {0: "person"}  # extend via --show-all


def label_color(name: str) -> tuple[int, int, int]:
    key = name.lower().split()[0]
    return LABEL_COLORS.get(key, LABEL_COLORS["default"])


def load_model(mode: str, model_path: str | None, device: str):
    from ultralytics import YOLO

    if mode == "world":
        path = model_path or "yolov8s-worldv2.pt"
        model = YOLO(path)
        model.set_classes(WORLD_CLASSES)
        print(f"YOLO-World loaded: {path}")
        print(f"  classes: {WORLD_CLASSES}")
        return model, None

    if mode == "pose":
        path = model_path or "yolo11n-pose.pt"
        model = YOLO(path)
        print(f"YOLO-pose loaded: {path}")
        return model, "pose"

    path = model_path or "yolo11n.pt"
    model = YOLO(path)
    print(f"YOLO detect loaded: {path}")
    return model, "coco"


def keypoint_boxes(result, frame_shape: tuple[int, int, int]) -> list[tuple[str, float, tuple[int, int, int, int]]]:
    """Derive tight face / hand boxes from pose keypoints (COCO skeleton)."""

    if result.keypoints is None:
        return []
    height, width = frame_shape[:2]
    kpts = result.keypoints.data
    if kpts is None or kpts.numel() == 0:
        return []
    kpts = kpts[0].cpu().numpy()

    # COCO pose indices: 0 nose, 1-2 eyes, 3-4 ears, 5-6 shoulders, 7-8 elbows, 9-10 wrists
    groups = {
        "face": [0, 1, 2, 3, 4],
        "hand_l": [9],
        "hand_r": [10],
    }
    boxes: list[tuple[str, float, tuple[int, int, int, int]]] = []
    for label, indices in groups.items():
        pts = []
        confs = []
        for idx in indices:
            if idx >= len(kpts):
                continue
            x, y, c = kpts[idx]
            if c > 0.3:
                pts.append((x, y))
                confs.append(c)
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        margin = 25 if label == "face" else 40
        x1 = int(max(0, min(xs) - margin))
        y1 = int(max(0, min(ys) - margin))
        x2 = int(min(width - 1, max(xs) + margin))
        y2 = int(min(height - 1, max(ys) + margin))
        conf = float(np.mean(confs))
        display = "hand" if label.startswith("hand") else label
        boxes.append((display, conf, (x1, y1, x2, y2)))
    return boxes


def draw_detections(
    frame: np.ndarray,
    boxes: list[tuple[str, float, tuple[int, int, int, int], int | None]],
) -> np.ndarray:
    out = frame.copy()
    for label, conf, (x1, y1, x2, y2), track_id in boxes:
        color = label_color(label)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        tag = f"{label} {conf:.2f}"
        if track_id is not None:
            tag += f" #{track_id}"
        (tw, th), base = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(out, (x1, y1 - th - base - 4), (x1 + tw + 6, y1), color, -1)
        cv2.putText(out, tag, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        out,
        f"tracking {len(boxes)} box(es)",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )
    return out


def parse_yolo_results(result, frame_shape, mode_tag: str | None) -> list[tuple[str, float, tuple[int, int, int, int], int | None]]:
    parsed: list[tuple[str, float, tuple[int, int, int, int], int | None]] = []

    if mode_tag == "pose":
        for label, conf, box in keypoint_boxes(result, frame_shape):
            parsed.append((label, conf, box, None))
        if result.boxes is not None and len(result.boxes):
            names = result.names
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if names.get(cls_id, "") != "person":
                    continue
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy
                track_id = int(box.id[0]) if box.id is not None else None
                parsed.append(("person", conf, (x1, y1, x2, y2), track_id))
        return parsed

    if result.boxes is None or len(result.boxes) == 0:
        return parsed

    names = result.names
    for box in result.boxes:
        cls_id = int(box.cls[0])
        name = names.get(cls_id, str(cls_id))
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = xyxy
        track_id = int(box.id[0]) if box.id is not None else None
        parsed.append((name, conf, (x1, y1, x2, y2), track_id))
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("world", "pose", "coco"),
        default="world",
        help="world=face/hand/person open vocab; pose=keypoint face/hands; coco=COCO80",
    )
    parser.add_argument("--model", default="", help="Override checkpoint path.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold.")
    parser.add_argument("--mirror", action="store_true", default=True)
    parser.add_argument("--no-mirror", action="store_false", dest="mirror")
    parser.add_argument("--device", default="", help="cpu, mps, cuda, or empty=auto")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference size.")
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="coco mode only: show all COCO classes, not just person.",
    )
    args = parser.parse_args()

    device = args.device.strip() or None
    model, mode_tag = load_model(args.mode, args.model or None, device or "auto")

    cap = cv2.VideoCapture(int(args.camera))
    if not cap.isOpened():
        raise SystemExit("Could not open camera. Grant Terminal camera access in System Settings.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Controls: q=quit  s=snapshot  r=toggle record")
    print("Repo: https://github.com/ultralytics/ultralytics")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    recording = False
    writer: cv2.VideoWriter | None = None
    frame_count = 0
    fps_smooth = 0.0

    predict_kwargs: dict = {
        "conf": float(args.conf),
        "imgsz": int(args.imgsz),
        "verbose": False,
    }
    if device:
        predict_kwargs["device"] = device
    if args.mode == "coco" and not args.show_all:
        predict_kwargs["classes"] = [0]

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed.")
                break
            if args.mirror:
                frame = cv2.flip(frame, 1)

            t0 = time.perf_counter()
            results = model.track(frame, persist=True, **predict_kwargs)
            elapsed = time.perf_counter() - t0
            fps_smooth = 0.9 * fps_smooth + 0.1 / max(elapsed, 1e-6)

            parsed = parse_yolo_results(results[0], frame.shape, mode_tag)
            display = draw_detections(frame, parsed)
            cv2.putText(
                display,
                f"YOLO-{args.mode}  fps~{fps_smooth:.1f}",
                (10, display.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

            if recording and writer is not None:
                writer.write(display)

            cv2.imshow("Ultralytics YOLO tracking", display)
            frame_count += 1
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                path = OUTPUT_DIR / "yolo_snapshot.png"
                cv2.imwrite(str(path), display)
                print(f"Saved {path}")
            if key == ord("r"):
                if not recording:
                    path = str(OUTPUT_DIR / "yolo_camera.mp4")
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(path, fourcc, 20.0, (display.shape[1], display.shape[0]))
                    recording = True
                    print(f"Recording {path}")
                else:
                    recording = False
                    if writer:
                        writer.release()
                        writer = None
                    print("Recording stopped.")

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print(f"Done. frames={frame_count}")


if __name__ == "__main__":
    main()
