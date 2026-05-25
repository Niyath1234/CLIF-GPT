#!/usr/bin/env python3
"""Live webcam demo: track you (and other regions) with CLIF object slots.

Opens your default camera, runs slot attention frame-by-frame with temporal
binding (slot identity persists across frames), and draws a **bounding box**
per slot around the image region that slot is tracking.

Controls:
  q / ESC  — quit
  s        — save a snapshot to demo_outputs/camera_snapshot.png
  r        — start/stop recording to demo_outputs/camera_tracking.mp4

Usage:
  source .venv/bin/activate
  python scripts/demo_object_tracking_camera.py
  python scripts/demo_object_tracking_camera.py --camera 0 --num-slots 4
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from cilf.model import VisionFoundationEncoder
from cilf.motion_focus import SubjectFocusPrior
from cilf.objects import ObjectTracker

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "demo_outputs"

SLOT_COLORS_BGR = [
    (60, 60, 220),
    (60, 200, 60),
    (220, 120, 60),
    (200, 60, 200),
    (60, 200, 200),
    (180, 180, 60),
]


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def slot_attention_step(
    module,
    inputs: torch.Tensor,
    prev_slots: torch.Tensor | None,
    patch_log_prior: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One frame of slot attention; returns slots [1,K,D] and attn [1,K,P].

    ``patch_log_prior`` [1, num_patches] downweights static background patches
    before the slot softmax (motion + center + detail prior).
    """

    batch_size, _, dim = inputs.shape
    slots = module.initial_slots(batch_size, inputs.device, inputs.dtype) if prev_slots is None else prev_slots

    inputs_n = module.norm_inputs(inputs)
    keys = module.to_k(inputs_n)
    values = module.to_v(inputs_n)
    last_attn = None

    for _ in range(module.iters):
        slots_prev = slots
        slots_n = module.norm_slots(slots)
        queries = module.to_q(slots_n)
        dots = torch.einsum("bkd,bnd->bkn", queries, keys) * module.scale
        if patch_log_prior is not None:
            dots = dots + patch_log_prior.unsqueeze(1)
        attn = dots.softmax(dim=1) + module.eps
        attn = attn / attn.sum(dim=-1, keepdim=True)
        last_attn = attn
        updates = torch.einsum("bnd,bkn->bkd", values, attn)
        slots = module.gru(
            updates.reshape(-1, dim),
            slots_prev.reshape(-1, dim),
        ).reshape(batch_size, module.num_slots, dim)
        slots = slots + module.mlp(module.norm_pre_ff(slots))

    return slots, last_attn


def infer_grid_hw(num_patches: int) -> tuple[int, int]:
    grid_size = int(round(num_patches**0.5))
    if grid_size * grid_size == num_patches:
        return grid_size, grid_size
    return 1, num_patches


def frame_to_tensor(bgr: np.ndarray, image_size: int, device: torch.device) -> torch.Tensor:
    """Return [batch=1, frames=1, C, H, W] for VisionFoundationEncoder."""

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0).unsqueeze(0).to(device)


def slot_attention_bboxes(
    attn: torch.Tensor,
    grid_hw: tuple[int, int],
    image_hw: tuple[int, int],
    subject_grid: np.ndarray | None = None,
    threshold_ratio: float = 0.5,
    min_area_ratio: float = 0.003,
    max_area_ratio: float = 0.22,
    min_subject_score: float = 0.12,
) -> list[tuple[int, int, int, int] | None]:
    """Map each slot's patch attention grid to an image-space bounding box.

    Returns one box per slot as (x1, y1, x2, y2) in pixel coords, or None if the
    slot is not attending strongly enough to any region.
    """

    height, width = image_hw
    grid_h, grid_w = grid_hw
    num_slots = attn.shape[0]
    attn_np = attn.detach().float().cpu().numpy().reshape(num_slots, grid_h, grid_w)
    attn_np = attn_np - attn_np.min(axis=(1, 2), keepdims=True)
    peak = attn_np.max(axis=(1, 2), keepdims=True)
    attn_np = attn_np / np.maximum(peak, 1e-6)

    patch_h = height / grid_h
    patch_w = width / grid_w
    min_area = height * width * min_area_ratio
    max_area = height * width * max_area_ratio
    boxes: list[tuple[int, int, int, int] | None] = []

    if subject_grid is not None:
        subject_grid = subject_grid.astype(np.float32)
        subject_grid = subject_grid / (subject_grid.max() + 1e-6)

    for slot_index in range(num_slots):
        slot_map = attn_np[slot_index]
        slot_peak = float(slot_map.max())
        if slot_peak < 0.2:
            boxes.append(None)
            continue

        threshold = max(threshold_ratio * slot_peak, float(np.quantile(slot_map, 0.72)))
        mask = slot_map >= threshold

        if subject_grid is not None:
            motion_mask = subject_grid >= np.quantile(subject_grid, 0.55)
            mask = mask & motion_mask

        if not mask.any():
            boxes.append(None)
            continue

        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        if rows.size == 0 or cols.size == 0:
            boxes.append(None)
            continue

        y1 = int(rows[0] * patch_h)
        y2 = int((rows[-1] + 1) * patch_h)
        x1 = int(cols[0] * patch_w)
        x2 = int((cols[-1] + 1) * patch_w)

        pad_x = int(0.04 * (x2 - x1))
        pad_y = int(0.04 * (y2 - y1))
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(width - 1, x2 + pad_x)
        y2 = min(height - 1, y2 + pad_y)

        area = (x2 - x1) * (y2 - y1)
        if area < min_area or area > max_area:
            boxes.append(None)
            continue

        if subject_grid is not None:
            motion_score = SubjectFocusPrior.mean_weight_in_box(
                subject_grid, (x1, y1, x2, y2), (height, width)
            )
            if motion_score < min_subject_score:
                boxes.append(None)
                continue

        boxes.append((x1, y1, x2, y2))

    return boxes


def smooth_boxes(
    current: list[tuple[int, int, int, int] | None],
    previous: list[tuple[int, int, int, int] | None],
    momentum: float = 0.65,
) -> list[tuple[int, int, int, int] | None]:
    """EMA smoothing so boxes do not flicker frame-to-frame."""

    if not previous:
        return current

    smoothed: list[tuple[int, int, int, int] | None] = []
    for slot_index, box in enumerate(current):
        prev = previous[slot_index] if slot_index < len(previous) else None
        if box is None:
            smoothed.append(prev)
            continue
        if prev is None:
            smoothed.append(box)
            continue
        x1 = int(momentum * prev[0] + (1.0 - momentum) * box[0])
        y1 = int(momentum * prev[1] + (1.0 - momentum) * box[1])
        x2 = int(momentum * prev[2] + (1.0 - momentum) * box[2])
        y2 = int(momentum * prev[3] + (1.0 - momentum) * box[3])
        smoothed.append((x1, y1, x2, y2))
    return smoothed


def draw_tracking_boxes(
    bgr: np.ndarray,
    boxes: list[tuple[int, int, int, int] | None],
    dominant_slot: int | None = None,
    show_fill: bool = True,
    attn: torch.Tensor | None = None,
    grid_hw: tuple[int, int] | None = None,
) -> np.ndarray:
    """Draw bounding boxes and optional light attention fill per slot."""

    result = bgr.copy()
    height, width = result.shape[:2]

    if show_fill and attn is not None and grid_hw is not None:
        grid_h, grid_w = grid_hw
        attn_np = attn.detach().float().cpu().numpy().reshape(attn.shape[0], grid_h, grid_w)
        attn_np = attn_np - attn_np.min(axis=(1, 2), keepdims=True)
        attn_np = attn_np / np.maximum(attn_np.max(axis=(1, 2), keepdims=True), 1e-6)
        overlay = result.astype(np.float32)
        for slot_index, box in enumerate(boxes):
            if box is None:
                continue
            weight = torch.from_numpy(attn_np[slot_index]).float()
            weight_up = F.interpolate(
                weight.unsqueeze(0).unsqueeze(0),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )[0, 0].numpy()
            color = np.array(SLOT_COLORS_BGR[slot_index % len(SLOT_COLORS_BGR)], dtype=np.float32)
            overlay = overlay * (1.0 - 0.2 * weight_up[..., None]) + color * (0.2 * weight_up[..., None])
        result = np.clip(overlay, 0, 255).astype(np.uint8)

    active_count = 0
    for slot_index, box in enumerate(boxes):
        if box is None:
            continue
        active_count += 1
        x1, y1, x2, y2 = box
        color = SLOT_COLORS_BGR[slot_index % len(SLOT_COLORS_BGR)]
        thickness = 3 if slot_index == dominant_slot else 2
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

        label = f"obj {slot_index}"
        if slot_index == dominant_slot:
            label += " *"
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        label_y1 = max(0, y1 - text_h - baseline - 6)
        label_y2 = label_y1 + text_h + baseline + 6
        label_x2 = min(width - 1, x1 + text_w + 8)
        cv2.rectangle(result, (x1, label_y1), (label_x2, label_y2), color, -1, cv2.LINE_AA)
        cv2.putText(
            result,
            label,
            (x1 + 4, label_y2 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        result,
        f"tracking {active_count} region(s)",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )
    return result


def dominant_slot_at_center(
    attn: torch.Tensor,
    grid_hw: tuple[int, int],
    subject_grid: np.ndarray | None = None,
) -> int:
    """Slot with strongest subject-weighted attention near the center."""

    grid_h, grid_w = grid_hw
    attn_np = attn.detach().float().cpu().numpy().reshape(attn.shape[0], grid_h, grid_w)
    if subject_grid is not None:
        scores = (attn_np * subject_grid).sum(axis=(1, 2))
    else:
        cy, cx = grid_h // 2, grid_w // 2
        scores = attn_np[:, cy, cx]
        if grid_h > 3 and grid_w > 3:
            patch = attn_np[:, cy - 1 : cy + 2, cx - 1 : cx + 2]
            scores = patch.reshape(attn.shape[0], -1).max(axis=1)
    return int(scores.argmax())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (0 = default).")
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--num-slots", type=int, default=4)
    parser.add_argument("--slot-iters", type=int, default=2)
    parser.add_argument("--state-dim", type=int, default=64)
    parser.add_argument("--mirror", action="store_true", default=True, help="Mirror preview (selfie view).")
    parser.add_argument("--no-mirror", action="store_false", dest="mirror")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after N frames (0 = run until q). Useful with --no-display.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Record to demo_outputs/camera_tracking.mp4 without opening a window.",
    )
    parser.add_argument(
        "--bbox-threshold",
        type=float,
        default=0.5,
        help="Attention fraction of peak required inside a slot box (higher = tighter).",
    )
    parser.add_argument(
        "--no-fill",
        action="store_true",
        help="Draw bounding boxes only (no colored attention wash).",
    )
    parser.add_argument(
        "--no-subject-focus",
        action="store_true",
        help="Disable motion/center prior (tracks large static regions too).",
    )
    parser.add_argument(
        "--max-box-area",
        type=float,
        default=0.22,
        help="Reject boxes covering more than this fraction of the frame.",
    )
    args = parser.parse_args()

    device = choose_device()
    print(f"device={device}")
    print("Starting camera... (grant camera permission if macOS prompts you)")
    print("Controls: q=quit  s=snapshot  r=toggle record")

    cap = cv2.VideoCapture(int(args.camera))
    if not cap.isOpened():
        raise SystemExit(
            f"Could not open camera {args.camera}. "
            "Check System Settings → Privacy → Camera, then retry."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    encoder = VisionFoundationEncoder(state_dim=int(args.state_dim), pretrained=False).to(device)
    tracker = ObjectTracker(
        num_slots=int(args.num_slots),
        patch_feature_dim=encoder.patch_feature_dim,
        state_dim=int(args.state_dim),
        iters=int(args.slot_iters),
    ).to(device)
    encoder.eval()
    tracker.eval()

    prev_slots: torch.Tensor | None = None
    prev_boxes: list[tuple[int, int, int, int] | None] = []
    slot_history: list[torch.Tensor] = []
    focus_prior = SubjectFocusPrior(model_size=int(args.image_size))
    use_subject_focus = not args.no_subject_focus
    if use_subject_focus:
        print("Subject focus ON: motion + center + detail (face/hands, not static background)")
    recording = bool(args.no_display)
    writer: cv2.VideoWriter | None = None
    frame_count = 0
    fps_smooth = 0.0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.no_display:
        out_path = str(OUTPUT_DIR / "camera_tracking.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, 15.0, (640, 480))
        print(f"Headless mode: recording -> {out_path}")

    try:
        while True:
            if args.max_frames > 0 and frame_count >= args.max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from camera.")
                break

            if args.mirror:
                frame = cv2.flip(frame, 1)

            frame_tensor = frame_to_tensor(frame, int(args.image_size), device)

            t0 = time.perf_counter()
            with torch.no_grad():
                _, patch_features = encoder(frame_tensor, return_patches=True)
                projected = tracker.input_projection(patch_features[:, 0])

                subject_grid: np.ndarray | None = None
                patch_log_prior: torch.Tensor | None = None
                grid_hw_frame = infer_grid_hw(int(patch_features.shape[2]))
                if use_subject_focus:
                    subject_grid = focus_prior.update(frame, grid_hw_frame)
                    patch_weights = torch.from_numpy(
                        focus_prior.patch_weights(frame, grid_hw_frame)
                    ).float().to(device)
                    patch_log_prior = torch.log(patch_weights.clamp(min=1e-4)).unsqueeze(0)

                prev_slots, attn = slot_attention_step(
                    tracker.slot_attention,
                    projected,
                    prev_slots=prev_slots,
                    patch_log_prior=patch_log_prior,
                )
                slot_history.append(prev_slots[0].detach().cpu())

            elapsed = time.perf_counter() - t0
            fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / max(elapsed, 1e-6))

            grid_hw = infer_grid_hw(int(patch_features.shape[2]))

            dom = dominant_slot_at_center(attn[0], grid_hw, subject_grid=subject_grid)
            height, width = frame.shape[:2]
            raw_boxes = slot_attention_bboxes(
                attn[0],
                grid_hw,
                (height, width),
                subject_grid=subject_grid,
                threshold_ratio=float(args.bbox_threshold),
                max_area_ratio=float(args.max_box_area),
            )
            boxes = smooth_boxes(raw_boxes, prev_boxes, momentum=0.65)
            prev_boxes = boxes
            display = draw_tracking_boxes(
                frame,
                boxes,
                dominant_slot=dom,
                show_fill=not args.no_fill,
                attn=attn[0],
                grid_hw=grid_hw,
            )

            if len(slot_history) >= 2:
                delta = torch.norm(slot_history[-1] - slot_history[-2], dim=-1)
                cv2.putText(
                    display,
                    f"slot motion: {', '.join(f'{v:.2f}' for v in delta.tolist())}",
                    (10, display.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (240, 240, 240),
                    1,
                    cv2.LINE_AA,
                )

            cv2.putText(
                display,
                f"fps~{fps_smooth:.1f}  frames={frame_count}",
                (10, display.shape[0] - 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )

            if recording and writer is not None:
                writer.write(display)

            frame_count += 1

            if not args.no_display:
                cv2.imshow("CLIF object tracking (camera)", display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("s"):
                    path = OUTPUT_DIR / "camera_snapshot.png"
                    cv2.imwrite(str(path), display)
                    print(f"Saved snapshot -> {path}")
                if key == ord("r"):
                    if not recording:
                        out_path = str(OUTPUT_DIR / "camera_tracking.mp4")
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(
                            out_path,
                            fourcc,
                            15.0,
                            (display.shape[1], display.shape[0]),
                        )
                        recording = True
                        print(f"Recording -> {out_path}")
                    else:
                        recording = False
                        if writer is not None:
                            writer.release()
                            writer = None
                        print("Recording stopped.")

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

        if len(slot_history) >= 2:
            traj = torch.stack(slot_history, dim=0)
            print("\n=== Session summary ===")
            print(f"tracked_frames={traj.shape[0]}  slots={traj.shape[1]}  state_dim={traj.shape[2]}")
            total_change = torch.norm(traj[-1] - traj[0], dim=-1)
            for slot_index, value in enumerate(total_change.tolist()):
                print(f"  slot {slot_index}: total_state_change={value:.4f}")


if __name__ == "__main__":
    main()
