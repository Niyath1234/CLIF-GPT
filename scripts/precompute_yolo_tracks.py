#!/usr/bin/env python3
"""Precompute YOLO + ByteTrack tracks for every clip in a CLIF manifest.

Outputs one JSON file per video under ``--tracks-dir`` (mirroring the relative
clip path with ``.json`` suffix). The training dataset will pick these up
automatically when ``tracks_dir`` is set in the data config.

Example:

```
python scripts/precompute_yolo_tracks.py \
    --manifest data/physion/manifest_train.jsonl \
    --tracks-dir data/physion/tracks \
    --model yolov8s-worldv2.pt \
    --classes ball block cube ramp box object \
    --imgsz 480
```

Re-runs are idempotent (existing track files are skipped unless ``--overwrite``
is passed).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cilf.detector_tracks import detect_tracks, iter_world_class_palette  # noqa: E402
from cilf.physion_detector import detect_tracks_physion  # noqa: E402
from cilf.track_io import save_tracks  # noqa: E402


def iter_manifest(path: Path) -> Iterable[dict]:
    for line in path.read_text().splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        yield json.loads(text)


def resolve_video(manifest_dir: Path, row: dict) -> Path:
    key = row.get("video_path") or row.get("video") or row.get("path")
    if not key:
        raise ValueError("manifest row missing video_path")
    candidate = Path(key)
    if not candidate.is_absolute():
        candidate = (manifest_dir / candidate).resolve()
    return candidate


def track_filename(row: dict, video_path: Path) -> str:
    """Stable JSON filename for a manifest row.

    Prefers ``stim_id`` (Physion has unique stem-style identifiers) but falls
    back to the video file's stem. The same logic is used by the dataset
    loader so tracks land where it expects them.
    """

    stim_id = row.get("stim_id")
    if isinstance(stim_id, str) and stim_id.strip():
        return f"{stim_id}.json"
    return f"{video_path.stem}.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Path to JSONL manifest.")
    parser.add_argument("--tracks-dir", required=True, help="Output directory for track JSON files.")
    parser.add_argument(
        "--detector",
        choices=("yolo", "physion"),
        default="physion",
        help=(
            "yolo  = YOLO-World + ByteTrack (recommended for real-world video). "
            "physion = color-blob + IoU tracker, ideal for Physion's rendered "
            "primitives where YOLO transfers poorly."
        ),
    )
    parser.add_argument(
        "--model",
        default="yolov8s-worldv2.pt",
        help="YOLO checkpoint (used only with --detector yolo).",
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="Open-vocab class names to look for (YOLO-World only).",
    )
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--imgsz", type=int, default=480)
    parser.add_argument("--device", default="")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--every-n", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N rows.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-missing", action="store_true", help="Skip rows whose video file is absent.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")
    manifest_dir = manifest_path.parent
    tracks_dir = Path(args.tracks_dir)
    tracks_dir.mkdir(parents=True, exist_ok=True)

    classes = args.classes if args.classes else list(iter_world_class_palette())
    device = args.device or None

    rows = list(iter_manifest(manifest_path))
    if args.limit is not None:
        rows = rows[: args.limit]
    print(f"Precomputing tracks for {len(rows)} clip(s) -> {tracks_dir}", flush=True)

    processed = 0
    skipped = 0
    failed: list[tuple[Path, str]] = []
    start = time.perf_counter()
    for idx, row in enumerate(rows):
        try:
            video_path = resolve_video(manifest_dir, row)
        except Exception as exc:
            print(f"[{idx}] BAD ROW: {exc}", flush=True)
            failed.append((Path(str(row)), str(exc)))
            continue

        track_path = tracks_dir / track_filename(row, video_path)
        if track_path.exists() and not args.overwrite:
            skipped += 1
            continue
        if not video_path.exists():
            message = f"missing video: {video_path}"
            if args.skip_missing:
                print(f"[{idx}] SKIP {message}", flush=True)
                skipped += 1
                continue
            failed.append((video_path, message))
            print(f"[{idx}] FAIL {message}", flush=True)
            continue

        try:
            if args.detector == "yolo":
                clip = detect_tracks(
                    video_path,
                    model_name=args.model,
                    classes=classes,
                    conf=args.conf,
                    imgsz=args.imgsz,
                    device=device,
                    max_frames=args.max_frames,
                    every_n=args.every_n,
                )
            else:
                clip = detect_tracks_physion(
                    video_path,
                    max_frames=args.max_frames,
                    every_n=args.every_n,
                )
        except Exception as exc:
            failed.append((video_path, str(exc)))
            print(f"[{idx}] FAIL {video_path}: {exc}", flush=True)
            continue

        save_tracks(track_path, clip)
        processed += 1
        if processed % 20 == 0 or processed == 1:
            elapsed = time.perf_counter() - start
            rate = processed / max(elapsed, 1e-6)
            print(
                f"[{idx + 1}/{len(rows)}] processed={processed} skipped={skipped} "
                f"failed={len(failed)} rate={rate:.2f} clips/s",
                flush=True,
            )

    print(
        f"Done. processed={processed} skipped={skipped} failed={len(failed)} "
        f"output={tracks_dir}",
        flush=True,
    )
    if failed:
        print("First failures:")
        for path, reason in failed[:5]:
            print(f"  {path}: {reason}")


if __name__ == "__main__":
    main()
