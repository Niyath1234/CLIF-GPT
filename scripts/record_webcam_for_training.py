#!/usr/bin/env python3
"""Record short webcam clips + YOLO tracks + manifest for CILF training.

Workflow:
  1. Live preview with YOLO boxes (same as demo_yolo_camera.py).
  2. Press **SPACE** to start/stop a clip (saved as mp4).
  3. Press **n** to enter the next-token target word for that clip (e.g. fell).
  4. Press **q** to finish → writes manifest + precomputed tracks.

Output layout::

  data/webcam/
    raw/<clip_id>.mp4
    tracks/<clip_id>.json
    manifest_webcam.jsonl

Then train::

  python -m cilf.train --config configs/cilf_webcam.yaml

Usage::

  python scripts/record_webcam_for_training.py
  python scripts/record_webcam_for_training.py --mode pose --max-clips 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_ROOT = PROJECT_ROOT / "data" / "webcam"
RAW_DIR = OUTPUT_ROOT / "raw"
TRACKS_DIR = OUTPUT_ROOT / "tracks"
MANIFEST_PATH = OUTPUT_ROOT / "manifest_webcam.jsonl"

DEFAULT_PROMPTS = [
    ("I moved my hand toward the object and", "touched"),
    ("I pulled back and the cup", "fell"),
    ("My arm crossed the frame and then", "stopped"),
]


def parse_yolo_frame(result, mode_tag: str | None, frame_shape) -> list:
    from scripts.demo_yolo_camera import parse_yolo_results

    return parse_yolo_results(result, frame_shape, mode_tag)


def yolo_track_video(
    video_path: Path,
    mode: str,
    device: str | None,
) -> None:
    """Run YOLO+ByteTrack on a saved mp4 and write track JSON."""

    from cilf.detector_tracks import detect_tracks
    from cilf.track_io import save_tracks

    model_names = {
        "world": "yolov8s-worldv2.pt",
        "pose": "yolo11n-pose.pt",
        "coco": "yolo11n.pt",
    }
    classes = ["person", "face", "hand", "head"] if mode == "world" else None
    clip = detect_tracks(
        video_path,
        model_name=model_names.get(mode, "yolov8s-worldv2.pt"),
        classes=classes,
        conf=0.25,
        imgsz=640,
        device=device,
    )
    save_tracks(TRACKS_DIR / f"{video_path.stem}.json", clip)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("world", "pose", "coco"), default="world")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--max-clips", type=int, default=10)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--clip-seconds", type=float, default=4.0)
    args = parser.parse_args()

    import cv2
    from scripts.demo_yolo_camera import load_model

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    TRACKS_DIR.mkdir(parents=True, exist_ok=True)

    device = args.device.strip() or None
    model, mode_tag = load_model(args.mode, None, device or "auto")
    predict_kwargs: dict = {"conf": 0.25, "imgsz": 640, "verbose": False, "persist": True}
    if device:
        predict_kwargs["device"] = device
    if args.mode == "coco":
        predict_kwargs["classes"] = [0]

    cap = cv2.VideoCapture(int(args.camera))
    if not cap.isOpened():
        raise SystemExit(
            "Could not open camera. Grant Terminal/Cursor camera access in System Settings."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)

    manifest_rows: list[dict] = []
    clip_index = 0
    recording = False
    writer: cv2.VideoWriter | None = None
    clip_path: Path | None = None
    frames_written = 0
    target_frames = max(1, int(args.clip_seconds * fps))

    print("Controls: SPACE=start/stop clip | n=set target word after clip | q=done")
    print(f"Saving to {OUTPUT_ROOT}")

    try:
        while clip_index < args.max_clips:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            results = model.track(frame, **predict_kwargs)
            from scripts.demo_yolo_camera import draw_detections, parse_yolo_results

            parsed = parse_yolo_results(results[0], frame.shape, mode_tag)
            display = draw_detections(frame, parsed)
            status = "REC" if recording else "idle"
            cv2.putText(
                display,
                f"{status} clips={clip_index}/{args.max_clips} SPACE=record q=quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255) if recording else (200, 200, 200),
                2,
            )
            cv2.imshow("Record for CILF training", display)

            if recording and writer is not None:
                writer.write(display)
                frames_written += 1
                if frames_written >= target_frames:
                    recording = False
                    writer.release()
                    writer = None
                    if clip_path is not None:
                        print(f"Saved {clip_path} ({frames_written} frames)")
                        default_prompt, default_target = DEFAULT_PROMPTS[
                            clip_index % len(DEFAULT_PROMPTS)
                        ]
                        prompt = input(
                            f"Prompt [{default_prompt}]: "
                        ).strip() or default_prompt
                        target = input(
                            f"Target word [{default_target}]: "
                        ).strip() or default_target
                        yolo_track_video(clip_path, args.mode, device)
                        manifest_rows.append(
                            {
                                "video_path": str(Path("raw") / clip_path.name),
                                "prompt": prompt,
                                "causal_consequence": target,
                                "causal_trigger": True,
                                "scenario": "webcam_user",
                                "stim_id": clip_path.stem,
                                "split": "train",
                                "abstract_dynamics": "user_interaction",
                                "causal_state_change": "motion -> contact -> outcome",
                            }
                        )
                        clip_index += 1
                        frames_written = 0

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                if not recording:
                    clip_id = f"webcam_clip_{clip_index:02d}"
                    clip_path = RAW_DIR / f"{clip_id}.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(
                        str(clip_path),
                        fourcc,
                        fps,
                        (display.shape[1], display.shape[0]),
                    )
                    recording = True
                    frames_written = 0
                    print(f"Recording {clip_path} ...")
                elif writer is not None:
                    writer.release()
                    writer = None
                    recording = False

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    if manifest_rows:
        with MANIFEST_PATH.open("w") as handle:
            for row in manifest_rows:
                handle.write(json.dumps(row) + "\n")
        print(f"Wrote {len(manifest_rows)} rows -> {MANIFEST_PATH}")
        print("Train with: python -m cilf.train --config configs/cilf_webcam.yaml")
    else:
        print("No clips recorded.")


if __name__ == "__main__":
    main()
