#!/usr/bin/env python3
"""Visualize Physion clips with precomputed object tracks and manifest labels.

Outputs under ``demo_outputs/vis/``:

* ``<stim_id>_tracks.mp4`` — annotated video with per-track bounding boxes
* ``<stim_id>_grid.png`` — contact sheet of evenly spaced frames + boxes
* ``dataset_overview.png`` — one row per scenario (Collide, Drop, Dominoes, …)
* ``training_sample.png`` — what the dataloader sees (subsampled frames + boxes)

Examples:

```
python scripts/visualize_physion_dataset.py --stim-id pilot_dominoes_4mid_boxroom_2_0002_img
python scripts/visualize_physion_dataset.py --manifest data/kinetic_transfer/manifest_kinetic_val.jsonl --limit 8
python scripts/visualize_physion_dataset.py --overview
```
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cilf.track_io import ClipTracks, Track, load_tracks  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "demo_outputs" / "vis"

# Distinct BGR colors per track (cycle if many tracks)
TRACK_COLORS_BGR: list[tuple[int, int, int]] = [
    (60, 220, 60),
    (60, 140, 255),
    (255, 160, 60),
    (200, 60, 200),
    (255, 255, 60),
    (60, 255, 255),
    (180, 60, 60),
    (220, 220, 220),
]


def _require_cv2():
    import cv2

    return cv2


def _color_for_track(track_id: int) -> tuple[int, int, int]:
    return TRACK_COLORS_BGR[track_id % len(TRACK_COLORS_BGR)]


def _box_at_frame(track: Track, frame_index: int) -> tuple[tuple[float, float, float, float], float] | None:
    """Return normalized box + conf for ``frame_index``, or None."""

    exact = None
    prev = None
    nxt = None
    for tf in track.frames:
        if tf.frame_index == frame_index:
            exact = tf
            break
        if tf.frame_index < frame_index:
            prev = tf
        elif tf.frame_index > frame_index and nxt is None:
            nxt = tf
            break
    if exact is not None:
        return exact.box_norm, exact.conf
    if prev is not None and nxt is not None:
        span = max(1, nxt.frame_index - prev.frame_index)
        alpha = (frame_index - prev.frame_index) / span
        box = tuple(
            prev.box_norm[i] * (1.0 - alpha) + nxt.box_norm[i] * alpha for i in range(4)
        )
        return box, float(min(prev.conf, nxt.conf) * 0.5)
    if prev is not None:
        return prev.box_norm, prev.conf * 0.3
    if nxt is not None:
        return nxt.box_norm, nxt.conf * 0.3
    return None


def _norm_to_pixels(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1 = int(box[0] * width)
    y1 = int(box[1] * height)
    x2 = int(box[2] * width)
    y2 = int(box[3] * height)
    return x1, y1, x2, y2


def draw_tracks_on_frame(
    frame_bgr: np.ndarray,
    clip: ClipTracks,
    frame_index: int,
    *,
    min_conf: float = 0.05,
    title_lines: list[str] | None = None,
    exact_frame_only: bool = False,
) -> np.ndarray:
    cv2 = _require_cv2()
    out = frame_bgr.copy()
    height, width = out.shape[:2]

    for track in clip.tracks:
        if exact_frame_only:
            hit = None
            for tf in track.frames:
                if tf.frame_index == frame_index:
                    hit = (tf.box_norm, tf.conf)
                    break
        else:
            hit = _box_at_frame(track, frame_index)
        if hit is None:
            continue
        box_norm, conf = hit
        if conf < min_conf:
            continue
        x1, y1, x2, y2 = _norm_to_pixels(box_norm, width, height)
        if x2 <= x1 or y2 <= y1:
            continue
        color = _color_for_track(track.track_id)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        tag = f"#{track.track_id} {track.label} {conf:.2f}"
        (tw, th), base = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - base - 4)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            out,
            tag,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    if title_lines:
        y = 22
        for line in title_lines:
            cv2.putText(
                out,
                line,
                (8, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (240, 240, 240),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                out,
                line,
                (8, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
            y += 20

    cv2.putText(
        out,
        f"frame {frame_index}",
        (8, height - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    return out


def resolve_video_path(manifest_dir: Path, video_key: str) -> Path:
    candidate = Path(video_key)
    if not candidate.is_absolute():
        candidate = (manifest_dir / candidate).resolve()
    return candidate


def resolve_track_path(tracks_dir: Path, stim_id: str) -> Path | None:
    path = tracks_dir / f"{stim_id}.json"
    return path if path.exists() else None


def iter_manifest(path: Path, limit: int | None) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text().splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        rows.append(json.loads(text))
        if limit is not None and len(rows) >= limit:
            break
    return rows


def render_tracked_video(
    video_path: Path,
    clip: ClipTracks,
    row: dict,
    output_path: Path,
    *,
    max_frames: int | None = None,
    fps: float | None = None,
) -> None:
    cv2 = _require_cv2()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_fps = fps or max(8.0, min(native_fps, 20.0))

    title = [
        str(row.get("scenario", ""))[:40],
        f"dynamics: {row.get('abstract_dynamics', '')}"[:50],
        f"target: {row.get('causal_consequence', '')} | tracks: {len(clip.tracks)}",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        out_fps,
        (width, height),
    )

    frame_index = 0
    written = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames is not None and written >= max_frames:
                break
            annotated = draw_tracks_on_frame(
                frame,
                clip,
                frame_index,
                title_lines=title,
                exact_frame_only=True,
            )
            writer.write(annotated)
            written += 1
            frame_index += 1
    finally:
        cap.release()
        writer.release()

    print(f"Wrote video {output_path} ({written} frames)")


def render_contact_sheet(
    video_path: Path,
    clip: ClipTracks,
    row: dict,
    output_path: Path,
    *,
    num_panels: int = 8,
) -> None:
    import matplotlib.pyplot as plt

    cv2 = _require_cv2()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
    indices = np.linspace(0, max(0, total - 1), num_panels).astype(int)

    cols = min(4, num_panels)
    rows_n = (num_panels + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(4 * cols, 3.5 * rows_n))
    axes_flat = np.atleast_1d(axes).flatten()

    for panel, frame_index in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        if not ok:
            continue
        annotated = draw_tracks_on_frame(frame, clip, int(frame_index))
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        axes_flat[panel].imshow(rgb)
        axes_flat[panel].set_title(f"t={frame_index}", fontsize=9)
        axes_flat[panel].axis("off")

    for panel in range(len(indices), len(axes_flat)):
        axes_flat[panel].axis("off")

    stim = row.get("stim_id", video_path.stem)
    prompt = row.get("prompt", "")
    fig.suptitle(
        f"{stim}\n{prompt} → {row.get('causal_consequence', '')} | {row.get('abstract_dynamics', '')}",
        fontsize=10,
        y=1.02,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    cap.release()
    print(f"Wrote grid {output_path}")


def render_dataset_overview(
    manifest_path: Path,
    tracks_dir: Path,
    output_path: Path,
    *,
    per_scenario: int = 1,
) -> None:
    """One panel per Physion scenario with boxes on a mid-frame."""

    import matplotlib.pyplot as plt

    cv2 = _require_cv2()
    manifest_dir = manifest_path.parent
    by_scenario: dict[str, list[dict]] = defaultdict(list)
    for row in iter_manifest(manifest_path, limit=None):
        scenario = str(row.get("scenario", "unknown"))
        # kinetic_dominoes -> dominoes
        short = scenario.replace("kinetic_", "").split("_")[0].capitalize()
        if len(by_scenario[short]) < per_scenario:
            by_scenario[short].append(row)

    scenarios = sorted(by_scenario.keys())
    if not scenarios:
        raise SystemExit("No manifest rows found.")

    n = len(scenarios)
    cols = min(4, n)
    rows_n = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(4.2 * cols, 3.8 * rows_n))
    axes_flat = np.atleast_1d(axes).flatten()

    for idx, scenario in enumerate(scenarios):
        row = by_scenario[scenario][0]
        stim_id = str(row.get("stim_id", ""))
        video_path = resolve_video_path(manifest_dir, row["video_path"])
        track_path = resolve_track_path(tracks_dir, stim_id)
        ax = axes_flat[idx]
        if not video_path.exists():
            ax.text(0.5, 0.5, "missing video", ha="center", va="center")
            ax.set_title(scenario)
            ax.axis("off")
            continue

        clip = load_tracks(track_path) if track_path else ClipTracks(
            video_path=str(video_path), fps=0, num_frames=0, frame_size=(0, 0), tracks=[]
        )
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
        mid = total // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            ax.axis("off")
            continue
        annotated = draw_tracks_on_frame(frame, clip, mid)
        ax.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{scenario}\n{len(clip.tracks)} tracks", fontsize=9)
        ax.axis("off")

    for idx in range(len(scenarios), len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.suptitle("Physion dataset — motion-tracked objects (mid-frame)", fontsize=12, y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote overview {output_path}")


def render_training_view(
    manifest_path: Path,
    tracks_dir: Path,
    stim_id: str,
    output_path: Path,
    *,
    num_frames: int = 8,
    image_size: int = 224,
) -> None:
    """Show subsampled + normalized frames exactly as the training loader."""

    import matplotlib.pyplot as plt
    import torch
    from transformers import AutoTokenizer

    from cilf.data import GeneralCausalVideoDataset

    manifest_dir = manifest_path.parent
    mini_manifest = OUTPUT_DIR / "_mini_train_vis.jsonl"
    rows = [r for r in iter_manifest(manifest_path, limit=None) if r.get("stim_id") == stim_id]
    if not rows:
        raise SystemExit(f"stim_id {stim_id!r} not in {manifest_path}")
    row = dict(rows[0])
    row["video_path"] = str(resolve_video_path(manifest_dir, row["video_path"]))
    mini_manifest.write_text(json.dumps(row) + "\n")

    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = GeneralCausalVideoDataset(
        manifest_path=str(mini_manifest),
        tokenizer=tokenizer,
        num_frames=num_frames,
        image_size=image_size,
        fps=8,
        max_prompt_length=48,
        tracks_dir=str(tracks_dir),
        max_tracks=6,
    )
    sample = dataset[0]
    frames = sample["frames"]  # [T, 3, H, W] normalized
    boxes = sample["track_boxes"]
    mask = sample["track_mask"]

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    fig, axes = plt.subplots(2, num_frames, figsize=(2.2 * num_frames, 5))
    for t in range(num_frames):
        img = (frames[t] * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        axes[0, t].imshow(img)
        axes[0, t].set_title(f"t={t}", fontsize=8)
        axes[0, t].axis("off")

        import matplotlib.patches as patches

        h, w = image_size, image_size
        axes[1, t].imshow(img)
        for k in range(boxes.shape[1]):
            if mask[t, k].item() < 0.5:
                continue
            x1, y1, x2, y2 = boxes[t, k].tolist()
            px1, py1 = int(x1 * w), int(y1 * h)
            px2, py2 = int(x2 * w), int(y2 * h)
            bgr = _color_for_track(k)
            color = (bgr[2] / 255.0, bgr[1] / 255.0, bgr[0] / 255.0)
            axes[1, t].add_patch(
                patches.Rectangle(
                    (px1, py1),
                    max(1, px2 - px1),
                    max(1, py2 - py1),
                    linewidth=2,
                    edgecolor=color,
                    facecolor="none",
                )
            )
        axes[1, t].set_title(f"slot k (mask)", fontsize=8)
        axes[1, t].axis("off")

    record = dataset.records[0]
    fig.suptitle(
        f"Training view: {stim_id}\n"
        f"prompt: {record.prompt!r} → {record.causal_consequence!r}\n"
        f"dynamics: {record.abstract_dynamics} | causal_state_change: {record.causal_state_change}",
        fontsize=9,
        y=1.05,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote training view {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="data/kinetic_transfer/manifest_kinetic_val.jsonl",
    )
    parser.add_argument("--tracks-dir", default="data/kinetic_transfer/tracks")
    parser.add_argument("--stim-id", default="", help="Visualize one clip by stim_id.")
    parser.add_argument("--limit", type=int, default=4, help="Max clips when scanning manifest.")
    parser.add_argument("--overview", action="store_true", help="Write dataset_overview.png only.")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--video-max-frames", type=int, default=None)
    parser.add_argument("--training-view", action="store_true", help="Also write dataloader subsample panel.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = PROJECT_ROOT / args.manifest
    tracks_dir = PROJECT_ROOT / args.tracks_dir
    manifest_dir = manifest_path.parent

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.overview:
        render_dataset_overview(
            manifest_path,
            tracks_dir,
            OUTPUT_DIR / "dataset_overview.png",
        )
        return

    if args.stim_id:
        rows = [r for r in iter_manifest(manifest_path, limit=None) if r.get("stim_id") == args.stim_id]
        if not rows:
            raise SystemExit(f"stim_id not found: {args.stim_id}")
    else:
        rows = iter_manifest(manifest_path, args.limit)

    for row in rows:
        stim_id = str(row.get("stim_id", ""))
        video_path = resolve_video_path(manifest_dir, row["video_path"])
        track_path = resolve_track_path(tracks_dir, stim_id)
        if not video_path.exists():
            print(f"SKIP {stim_id}: missing video {video_path}")
            continue

        clip = load_tracks(track_path) if track_path else ClipTracks(
            video_path=str(video_path), fps=0, num_frames=0, frame_size=(0, 0), tracks=[]
        )
        if not clip.tracks:
            print(f"WARN {stim_id}: no tracks ({track_path})")

        if args.training_view:
            render_training_view(
                manifest_path, tracks_dir, stim_id, OUTPUT_DIR / f"{stim_id}_training.png"
            )

        if not args.no_video:
            render_tracked_video(
                video_path,
                clip,
                row,
                OUTPUT_DIR / f"{stim_id}.mp4",
                max_frames=args.video_max_frames,
            )
        render_contact_sheet(video_path, clip, row, OUTPUT_DIR / f"{stim_id}_grid.png")

    render_dataset_overview(manifest_path, tracks_dir, OUTPUT_DIR / "dataset_overview.png")
    print(f"\nAll outputs in {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
