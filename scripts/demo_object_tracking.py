#!/usr/bin/env python3
"""Visual demo: object-centric slot tracking on a video clip.

Runs the CLIF object tracker on real or synthetic video and writes:
  - ``demo_outputs/object_tracking_overlay.mp4`` — frames with slot-colored regions
  - ``demo_outputs/slot_trajectory.png`` — per-slot state norm over time + deltas

Usage:
  python scripts/demo_object_tracking.py
  python scripts/demo_object_tracking.py --video path/to/clip.mp4
  python scripts/demo_object_tracking.py --manifest data/kinetic_transfer/manifest_kinetic_train.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from cilf.model import VisionFoundationEncoder
from cilf.objects import ObjectTracker
from cilf.video_io import read_video

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "demo_outputs"

SLOT_COLORS_BGR = [
    (220, 60, 60),    # red
    (60, 200, 60),    # green
    (60, 120, 255),   # blue
    (60, 220, 220),   # yellow
    (220, 60, 220),   # magenta
    (220, 180, 60),   # cyan-ish
]


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def slot_attention_with_maps(
    module,
    inputs: torch.Tensor,
    prev_slots: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run SlotAttention and return (slots, attn) with attn [batch, slots, patches]."""

    batch_size, _, dim = inputs.shape
    if prev_slots is None:
        slots = module.initial_slots(batch_size, inputs.device, inputs.dtype)
    else:
        slots = prev_slots

    inputs_n = module.norm_inputs(inputs)
    keys = module.to_k(inputs_n)
    values = module.to_v(inputs_n)
    last_attn: torch.Tensor | None = None

    for _ in range(module.iters):
        slots_prev = slots
        slots_n = module.norm_slots(slots)
        queries = module.to_q(slots_n)
        dots = torch.einsum("bkd,bnd->bkn", queries, keys) * module.scale
        attn = dots.softmax(dim=1) + module.eps
        attn = attn / attn.sum(dim=-1, keepdim=True)
        last_attn = attn
        updates = torch.einsum("bnd,bkn->bkd", values, attn)
        slots = module.gru(
            updates.reshape(-1, dim),
            slots_prev.reshape(-1, dim),
        ).reshape(batch_size, module.num_slots, dim)
        slots = slots + module.mlp(module.norm_pre_ff(slots))

    assert last_attn is not None
    return slots, last_attn


def track_with_attention_maps(
    tracker: ObjectTracker,
    patch_features: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Returns slot_trajectory [1,T,K,D] and per-frame attention [K, P]."""

    projected = tracker.input_projection(patch_features)
    batch_size, num_frames, _, _ = projected.shape
    slots: torch.Tensor | None = None
    trajectory: list[torch.Tensor] = []
    attention_maps: list[torch.Tensor] = []

    for frame_index in range(num_frames):
        slots, attn = slot_attention_with_maps(
            tracker.slot_attention,
            projected[:, frame_index],
            prev_slots=slots,
        )
        trajectory.append(slots)
        attention_maps.append(attn[0])

    return torch.stack(trajectory, dim=1), attention_maps


def make_synthetic_video(path: Path, num_frames: int = 24, size: int = 128) -> None:
    """Three colored discs moving on independent paths (good slot-attention test)."""

    try:
        import imageio.v3 as iio
    except ImportError as exc:
        raise RuntimeError("imageio required for synthetic video export") from exc

    frames: list[np.ndarray] = []
    t_axis = np.linspace(0.0, 2.0 * np.pi, num_frames)
    centers = [
        (32 + 28 * np.sin(t_axis), 32 + 18 * np.cos(t_axis * 0.7)),
        (96 + 20 * np.cos(t_axis * 1.1), 64 + 24 * np.sin(t_axis)),
        (64 + 30 * np.sin(t_axis * 0.5 + 1.0), 96 + 16 * np.cos(t_axis * 1.3)),
    ]
    colors = [(220, 80, 80), (80, 200, 80), (80, 120, 240)]

    for frame_idx in range(num_frames):
        canvas = np.full((size, size, 3), 40, dtype=np.uint8)
        yy, xx = np.mgrid[0:size, 0:size]
        for (cy, cx), color in zip(
            [(centers[i][0][frame_idx], centers[i][1][frame_idx]) for i in range(3)],
            colors,
        ):
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 < 12**2
            canvas[mask] = color
        frames.append(canvas)

    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, np.stack(frames, axis=0), fps=8, plugin="ffmpeg")
    print(f"Wrote synthetic demo video -> {path}")


def resolve_video_path(args: argparse.Namespace) -> Path:
    if args.video:
        path = Path(args.video)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    if args.manifest:
        manifest = Path(args.manifest)
        if not manifest.is_absolute():
            manifest = PROJECT_ROOT / manifest
        for line in manifest.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            row = json.loads(line)
            rel = row.get("video_path") or row.get("video")
            if rel is None:
                continue
            candidate = (manifest.parent / rel).resolve()
            if candidate.exists():
                print(f"Using manifest video: {candidate}")
                return candidate
        print(f"No existing video found via {manifest}; falling back to synthetic clip.")

    synthetic = PROJECT_ROOT / "data/demo/synthetic_objects.mp4"
    if not synthetic.exists():
        make_synthetic_video(synthetic)
    return synthetic


def load_frames_tensor(path: Path, num_frames: int, image_size: int, fps: int, device: torch.device) -> torch.Tensor:
    video, native_fps = read_video(path, target_fps=fps)
    if video.numel() == 0:
        raise ValueError(f"No frames in {path}")

    stride = max(1, round(native_fps / fps))
    video = video[::stride].float()
    if video.shape[0] < num_frames:
        pad = video[-1:].repeat(num_frames - video.shape[0], 1, 1, 1)
        video = torch.cat([video, pad], dim=0)
    else:
        indices = torch.linspace(0, video.shape[0] - 1, steps=num_frames).long()
        video = video[indices]

    video = F.interpolate(video, size=(image_size, image_size), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    video = (video - mean) / std
    return video.unsqueeze(0).to(device)


def denormalize_frame(frame_chw: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = (frame_chw.cpu() * std + mean).clamp(0.0, 1.0)
    rgb = (img.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return rgb[:, :, ::-1].copy()


def render_overlay_video(
    frames_chw: torch.Tensor,
    attention_maps: list[torch.Tensor],
    grid_hw: tuple[int, int],
    output_path: Path,
    fps: int,
) -> None:
    import imageio.v3 as iio

    height, width = int(frames_chw.shape[-2]), int(frames_chw.shape[-1])
    grid_h, grid_w = grid_hw
    num_slots = attention_maps[0].shape[0]

    out_frames: list[np.ndarray] = []
    for frame_index, attn in enumerate(attention_maps):
        base = denormalize_frame(frames_chw[frame_index])
        attn_np = attn.detach().float().cpu().numpy()
        attn_2d = attn_np.reshape(num_slots, grid_h, grid_w)
        attn_2d = (attn_2d - attn_2d.min(axis=(1, 2), keepdims=True))
        denom = attn_2d.max(axis=(1, 2), keepdims=True)
        attn_2d = attn_2d / np.maximum(denom, 1e-6)

        overlay = base.astype(np.float32)
        for slot_index in range(num_slots):
            weight = torch.from_numpy(attn_2d[slot_index]).float()
            weight_up = F.interpolate(
                weight.unsqueeze(0).unsqueeze(0),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )[0, 0].numpy()
            color = np.array(SLOT_COLORS_BGR[slot_index % len(SLOT_COLORS_BGR)], dtype=np.float32)
            overlay = overlay * (1.0 - 0.55 * weight_up[..., None]) + color * (0.55 * weight_up[..., None])

        out_frames.append(np.clip(overlay, 0, 255).astype(np.uint8))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(output_path, np.stack(out_frames, axis=0), fps=fps, plugin="ffmpeg")
    print(f"Wrote overlay video -> {output_path}")


def plot_slot_dynamics(
    slot_trajectory: torch.Tensor,
    output_path: Path,
) -> None:
    """Plot slot L2 norm over time and frame-to-frame delta magnitude."""

    traj = slot_trajectory[0].detach().float().cpu()
    num_frames, num_slots, _ = traj.shape
    norms = torch.norm(traj, dim=-1).numpy()
    deltas = torch.norm(traj[1:] - traj[:-1], dim=-1).numpy()
    final_delta = torch.norm(traj[-1] - traj[0], dim=-1).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    for slot_index in range(num_slots):
        axes[0].plot(range(num_frames), norms[:, slot_index], marker="o", label=f"slot {slot_index}")
    axes[0].set_title("Slot state norm over frames")
    axes[0].set_xlabel("frame")
    axes[0].set_ylabel("||state||")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    for slot_index in range(num_slots):
        axes[1].plot(range(1, num_frames), deltas[:, slot_index], marker="o", label=f"slot {slot_index}")
    axes[1].set_title("Per-step state change")
    axes[1].set_xlabel("frame transition")
    axes[1].set_ylabel("||s_t - s_{t-1}||")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].bar(range(num_slots), final_delta, color=[f"C{i}" for i in range(num_slots)])
    axes[2].set_title("Total change (last - first frame)")
    axes[2].set_xlabel("slot")
    axes[2].set_ylabel("||delta||")
    axes[2].grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Wrote trajectory plot -> {output_path}")


def print_slot_report(slot_trajectory: torch.Tensor, num_slots: int) -> None:
    traj = slot_trajectory[0].detach().float().cpu()
    num_frames = traj.shape[0]

    print("\n=== Object tracking report ===")
    print(f"frames={num_frames}  slots={num_slots}  state_dim={traj.shape[-1]}")

    for slot_index in range(num_slots):
        step_changes = torch.norm(traj[1:, slot_index] - traj[:-1, slot_index], dim=-1)
        total_change = torch.norm(traj[-1, slot_index] - traj[0, slot_index]).item()
        peak_step = step_changes.max().item() if step_changes.numel() else 0.0
        print(
            f"  slot {slot_index}: total_change={total_change:.4f}  "
            f"peak_step_change={peak_step:.4f}  mean_norm={traj[:, slot_index].norm(dim=-1).mean():.4f}"
        )

    slot_ids = torch.arange(num_slots)
    consistency = []
    for frame_index in range(1, num_frames):
        prev_n = F.normalize(traj[frame_index - 1], dim=-1)
        curr_n = F.normalize(traj[frame_index], dim=-1)
        sim = (prev_n * curr_n).sum(dim=-1)
        consistency.append(sim)
    consistency_tensor = torch.stack(consistency, dim=0)
    mean_consistency = consistency_tensor.mean(dim=0)
    print("\n  Temporal identity (cosine sim slot_t vs slot_{t-1}):")
    for slot_index in range(num_slots):
        print(f"    slot {slot_index}: mean={mean_consistency[slot_index]:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="", help="Path to an mp4 clip.")
    parser.add_argument(
        "--manifest",
        default="",
        help="JSONL manifest; uses the first row whose video_path exists.",
    )
    parser.add_argument("--num-frames", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--num-slots", type=int, default=4)
    parser.add_argument("--slot-iters", type=int, default=3)
    parser.add_argument("--state-dim", type=int, default=64)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    device = choose_device()
    print(f"device={device}")

    video_path = resolve_video_path(args)
    print(f"video={video_path}")

    frames = load_frames_tensor(
        video_path,
        num_frames=int(args.num_frames),
        image_size=int(args.image_size),
        fps=int(args.fps),
        device=device,
    )

    encoder = VisionFoundationEncoder(
        state_dim=int(args.state_dim),
        pretrained=False,
    ).to(device)
    tracker = ObjectTracker(
        num_slots=int(args.num_slots),
        patch_feature_dim=encoder.patch_feature_dim,
        state_dim=int(args.state_dim),
        iters=int(args.slot_iters),
    ).to(device)

    encoder.eval()
    tracker.eval()

    with torch.no_grad():
        _, patch_features = encoder(frames, return_patches=True)
        slot_trajectory, attention_maps = track_with_attention_maps(tracker, patch_features)

    batch_size, num_frames, num_patches, _ = patch_features.shape
    grid_size = int(round(num_patches**0.5))
    if grid_size * grid_size != num_patches:
        grid_h = max(1, num_patches // grid_size)
        grid_w = num_patches // grid_h
        if grid_h * grid_w != num_patches:
            grid_h, grid_w = 1, num_patches
    else:
        grid_h = grid_w = grid_size

    output_dir = Path(args.output_dir)
    render_overlay_video(
        frames[0],
        attention_maps,
        grid_hw=(grid_h, grid_w),
        output_path=output_dir / "object_tracking_overlay.mp4",
        fps=int(args.fps),
    )
    plot_slot_dynamics(slot_trajectory, output_dir / "slot_trajectory.png")
    print_slot_report(slot_trajectory, int(args.num_slots))

    print("\nDone. Open these files:")
    print(f"  {output_dir / 'object_tracking_overlay.mp4'}")
    print(f"  {output_dir / 'slot_trajectory.png'}")


if __name__ == "__main__":
    main()
