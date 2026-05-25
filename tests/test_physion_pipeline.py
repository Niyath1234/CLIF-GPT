"""End-to-end smoke test: real Physion clips + precomputed tracks + model fwd/bwd.

Run after ``scripts/precompute_yolo_tracks.py`` has generated at least a
handful of track files. The test:

* Constructs a tiny ``GeneralCausalVideoDataset`` over the precomputed clips.
* Wraps it in a DataLoader.
* Builds a ``CILFModel(use_detector_tracks=True)`` and steps a few batches
  through the full training-style forward + loss + backward.
* Asserts the per-object signals (``object_trajectory``, dynamics, relation
  tokens) are present and finite.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cilf.data import GeneralCausalVideoDataset  # noqa: E402
from cilf.losses import (  # noqa: E402
    object_dynamics_energy_loss,
    object_temporal_consistency_loss,
)
from cilf.model import CILFModel  # noqa: E402


MANIFEST_PATH = PROJECT_ROOT / "data/kinetic_transfer/manifest_kinetic_val.jsonl"
TRACKS_DIR = PROJECT_ROOT / "data/kinetic_transfer/tracks"


def _build_mini_manifest(limit: int = 6) -> Path:
    """Take only rows that have track files (so we exercise the detector path)."""

    out = PROJECT_ROOT / "data/kinetic_transfer/manifest_smoke.jsonl"
    rows: list[str] = []
    available = {p.stem for p in TRACKS_DIR.glob("*.json")}
    if not available:
        raise SystemExit(
            "No precomputed tracks found. Run scripts/precompute_yolo_tracks.py first."
        )
    for line in MANIFEST_PATH.read_text().splitlines():
        text = line.strip()
        if not text:
            continue
        row = json.loads(text)
        if row.get("stim_id") in available:
            rows.append(text)
            if len(rows) >= limit:
                break
    out.write_text("\n".join(rows) + "\n")
    return out


def main() -> None:
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    mini = _build_mini_manifest(limit=4)
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = GeneralCausalVideoDataset(
        manifest_path=str(mini),
        tokenizer=tokenizer,
        num_frames=6,
        image_size=224,
        fps=8,
        max_prompt_length=32,
        tracks_dir=str(TRACKS_DIR),
        max_tracks=4,
    )
    print(f"dataset_size={len(dataset)}")
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    # Vision is left unpretrained here so the smoke test runs in seconds
    # without pulling SigLIP. The full configs in configs/cilf_physion.yaml
    # use the SigLIP backbone end-to-end.
    model = CILFModel(
        llm_name="distilgpt2",
        state_dim=128,
        jepa_hidden_dim=256,
        vision_pretrained=False,
        dynamics_type="mlp",
        ode_steps=2,
        use_sheaf_alignment=False,
        use_tropical_fusion=False,
        use_object_tracking=False,
        use_detector_tracks=True,
        max_detector_tracks=4,
        use_relation_tokens=True,
    ).to(device)
    model.train()
    model.llm.eval()
    if model.visual_encoder.freeze_foundation:
        model.visual_encoder.foundation.eval()

    optimiser = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=1e-3
    )

    seen = 0
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= 2:
            break
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        outputs = model(
            frames=batch["frames"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            alpha_scale=0.5,
            track_boxes=batch.get("track_boxes"),
            track_mask=batch.get("track_mask"),
        )

        for key in ("predicted_object_next", "object_observed_next", "object_trajectory"):
            assert key in outputs, f"missing per-object output: {key}"
            tensor = outputs[key]
            assert torch.is_tensor(tensor)
            assert torch.isfinite(tensor).all(), f"{key} non-finite"

        consistency = object_temporal_consistency_loss(outputs["object_trajectory"])
        dyn_loss, pos, neg = object_dynamics_energy_loss(
            outputs["predicted_object_next"],
            outputs["object_observed_next"].detach(),
            margin=0.2,
        )
        ce = torch.nn.functional.cross_entropy(outputs["fused_logits"], batch["target_token_id"])
        loss = ce + 0.2 * consistency + 0.3 * dyn_loss

        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        seen += batch["frames"].shape[0]

        print(
            f"batch {batch_idx} loss={loss.item():.4f} ce={ce.item():.4f} "
            f"consistency={consistency.item():.4f} dyn={dyn_loss.item():.4f} "
            f"posE={pos.item():.4f} negE={neg.item():.4f} "
            f"track_mask_sum={batch['track_mask'].sum().item():.0f}"
        )

    assert seen >= 2, "no batches processed"
    print(f"\nEnd-to-end Physion pipeline OK ({seen} clips through fwd+bwd).")


if __name__ == "__main__":
    main()
