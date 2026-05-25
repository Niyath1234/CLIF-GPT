"""Causal-transfer evaluation: next-token + transition + embedding metrics.

The plan separates three measurement axes that the existing ``prove_transfer.py``
script blurs into one:

1. **Next-token accuracy / rank** — does the fused distribution put the
   expected causal-consequence token at rank 1, and does it improve over the
   language-only baseline?
2. **Cross-prompt causal transfer** — when the same clip is paired with
   prompts that were never used in training, does the model still produce the
   right hidden meaning?
3. **Embedding generalization** — do the causal-bias embeddings ``h_bias``
   cluster by their hidden ``abstract_dynamics`` label, so that unseen
   prompt-video pairs land near other clips with the same underlying
   transition?

The script reports all three from a single forward pass over the manifest.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from transformers import AutoTokenizer

from cilf.data import GeneralCausalVideoDataset
from cilf.train import choose_device, make_model, move_batch
from cilf.vocab import DomainVocab


def topk_tokens(logits: torch.Tensor, tokenizer, k: int = 5) -> list[tuple[str, float]]:
    probs = F.softmax(logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, k=k)
    items: list[tuple[str, float]] = []
    for token_id, prob in zip(top_ids.tolist(), top_probs.tolist()):
        items.append((tokenizer.decode([int(token_id)]).strip(), float(prob)))
    return items


def expected_rank(logits: torch.Tensor, tokenizer, expected_word: str) -> int:
    candidates = tokenizer.encode(" " + expected_word, add_special_tokens=False)
    if not candidates:
        return -1
    target_id = candidates[0]
    rank = int((logits > logits[target_id]).sum().item()) + 1
    return rank


def _load_config(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _build_dataset(config: dict[str, Any], tokenizer, manifest_path: str) -> GeneralCausalVideoDataset:
    data_cfg = dict(config["data"])
    data_cfg["manifest_path"] = manifest_path
    domain_vocab = DomainVocab.from_config(tokenizer, data_cfg)
    return GeneralCausalVideoDataset(
        manifest_path=manifest_path,
        tokenizer=tokenizer,
        domain_vocab=domain_vocab,
        num_frames=int(data_cfg.get("num_frames", 6)),
        image_size=int(data_cfg.get("image_size", 224)),
        fps=int(data_cfg.get("fps", 8)),
        max_prompt_length=int(data_cfg.get("max_prompt_length", 48)),
        tracks_dir=data_cfg.get("tracks_dir"),
        max_tracks=int(data_cfg.get("max_tracks", 0)),
    )


@torch.no_grad()
def _forward_sample(
    model,
    sample: dict[str, Any],
    device: torch.device,
    alpha_scale: float,
) -> dict[str, torch.Tensor]:
    batch: dict[str, Any] = {
        "frames": sample["frames"].unsqueeze(0),
        "input_ids": sample["input_ids"].unsqueeze(0),
        "attention_mask": sample["attention_mask"].unsqueeze(0),
    }
    for key in ("track_boxes", "track_mask"):
        value = sample.get(key)
        if torch.is_tensor(value):
            batch[key] = value.unsqueeze(0)
    batch = move_batch(batch, device)
    return model(
        frames=batch["frames"],
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        alpha_scale=alpha_scale,
        track_boxes=batch.get("track_boxes"),
        track_mask=batch.get("track_mask"),
    )


def _retrieval_metrics(
    embeddings: torch.Tensor,
    group_ids: list[int],
) -> dict[str, float]:
    """Embedding-space retrieval: for each anchor with a valid group id, look
    at the top-K neighbours (excluding itself) and report whether at least one
    shares the same group id (Recall@K) and the precision of the top-K set."""

    valid_indices = [i for i, gid in enumerate(group_ids) if gid >= 0]
    if len(valid_indices) < 2:
        return {"recall_at_1": 0.0, "recall_at_5": 0.0, "precision_at_5": 0.0}

    feats = F.normalize(embeddings[valid_indices], dim=-1)
    sims = feats @ feats.T
    sims.fill_diagonal_(float("-inf"))

    labels = torch.tensor([group_ids[i] for i in valid_indices])
    label_matches = labels.unsqueeze(1) == labels.unsqueeze(0)

    k1 = sims.argmax(dim=1)
    recall_at_1 = label_matches[torch.arange(len(valid_indices)), k1].float().mean().item()

    top_k = min(5, sims.shape[1] - 1)
    top_indices = sims.topk(k=top_k, dim=1).indices
    matches_topk = torch.gather(label_matches, 1, top_indices)
    recall_at_5 = (matches_topk.any(dim=1).float().mean().item())
    precision_at_5 = matches_topk.float().mean().item()

    return {
        "recall_at_1": recall_at_1,
        "recall_at_5": recall_at_5,
        "precision_at_5": precision_at_5,
    }


def evaluate(
    config_path: str,
    checkpoint_path: str,
    manifest_path: str,
    alpha_scale: float,
    output_path: str | None,
) -> dict[str, Any]:
    config = _load_config(config_path)
    device = choose_device(str(config.get("device", "auto")))

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["llm_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = _build_dataset(config, tokenizer, manifest_path)
    model = make_model(config, device)
    state = torch.load(Path(checkpoint_path), map_location=device)
    model.load_state_dict(state["model"], strict=False)
    model.eval()

    print(f"checkpoint={checkpoint_path}")
    print(f"manifest={manifest_path}  rows={len(dataset)}  alpha_scale={alpha_scale}")
    print("=" * 100)

    # ---- Pass 1: per-sample next-token + transition rank ----
    per_sample_rows: list[dict[str, Any]] = []
    per_dynamics_correct: dict[str, list[int]] = defaultdict(list)
    fused_better_than_llm = 0
    fused_correct = 0
    fused_top5_hits = 0
    llm_correct = 0
    total = 0

    embeddings: list[torch.Tensor] = []
    dynamics_label_to_id: dict[str, int] = {}
    embedding_group_ids: list[int] = []

    for idx in range(len(dataset)):
        sample = dataset[idx]
        record = dataset.records[idx]
        expected = record.causal_consequence

        outputs = _forward_sample(model, sample, device, alpha_scale)
        z_l = outputs["lingual_logits"][0]
        z_fused = outputs["fused_logits"][0]
        h_bias = outputs["h_bias"][0].detach().to("cpu").float()

        llm_top1 = topk_tokens(z_l, tokenizer, k=1)[0][0]
        fused_top5 = topk_tokens(z_fused, tokenizer, k=5)
        rank_llm = expected_rank(z_l, tokenizer, expected)
        rank_fused = expected_rank(z_fused, tokenizer, expected)

        is_fused_correct = fused_top5[0][0] == expected
        is_llm_correct = llm_top1 == expected
        in_top5 = any(token == expected for token, _ in fused_top5)
        improved = rank_fused != -1 and rank_llm != -1 and rank_fused < rank_llm

        fused_correct += int(is_fused_correct)
        llm_correct += int(is_llm_correct)
        fused_top5_hits += int(in_top5)
        fused_better_than_llm += int(improved)
        total += 1

        dyn_label = record.abstract_dynamics or "unknown_dynamics"
        per_dynamics_correct[dyn_label].append(int(is_fused_correct))

        if dyn_label != "unknown_dynamics":
            if dyn_label not in dynamics_label_to_id:
                dynamics_label_to_id[dyn_label] = len(dynamics_label_to_id)
            embedding_group_ids.append(dynamics_label_to_id[dyn_label])
        else:
            embedding_group_ids.append(-1)
        embeddings.append(h_bias)

        per_sample_rows.append(
            {
                "idx": idx,
                "scenario": record.scenario,
                "stim_id": record.stim_id,
                "abstract_dynamics": dyn_label,
                "prompt": record.prompt,
                "expected": expected,
                "llm_top1": llm_top1,
                "fused_top1": fused_top5[0][0],
                "fused_top5": [(w, round(p, 4)) for w, p in fused_top5],
                "rank_llm": rank_llm,
                "rank_fused": rank_fused,
                "fused_correct": is_fused_correct,
                "rank_improved": improved,
            }
        )

        print(
            f"[{idx:02d}] dynamics={dyn_label} scenario={record.scenario}\n"
            f"     prompt: {record.prompt!r}\n"
            f"     expected: {expected!r}\n"
            f"     LLM top1: {llm_top1!r:>14}  rank_LLM={rank_llm}\n"
            f"     FUSED   : {fused_top5[0][0]!r:>14}  rank_FUSED={rank_fused}\n"
            f"     fused top5: {[(w, round(p, 4)) for w, p in fused_top5]}\n"
        )

    # ---- Aggregate next-token metrics ----
    next_token = {
        "fused_top1_acc": fused_correct / max(1, total),
        "fused_top5_acc": fused_top5_hits / max(1, total),
        "llm_top1_acc": llm_correct / max(1, total),
        "fused_rank_improves_over_llm": fused_better_than_llm / max(1, total),
        "samples": total,
    }

    # ---- Per-abstract-dynamics breakdown (causal transfer) ----
    per_dynamics = {
        label: {
            "samples": len(scores),
            "fused_top1_acc": (sum(scores) / max(1, len(scores))),
        }
        for label, scores in sorted(per_dynamics_correct.items())
    }

    # ---- Embedding generalization (abstract-dynamics retrieval) ----
    embedding_matrix = torch.stack(embeddings, dim=0)
    retrieval = _retrieval_metrics(embedding_matrix, embedding_group_ids)

    print("=" * 100)
    print("[ Next-token metrics ]")
    for key, value in next_token.items():
        print(f"  {key}: {value}")
    print("\n[ Per-abstract-dynamics fused top-1 accuracy ]")
    for label, metrics in per_dynamics.items():
        print(f"  {label:<35s}  n={metrics['samples']:>3d}  acc={metrics['fused_top1_acc']:.3f}")
    print("\n[ Embedding generalization (h_bias retrieval by abstract_dynamics) ]")
    for key, value in retrieval.items():
        print(f"  {key}: {value:.3f}")

    summary = {
        "checkpoint": checkpoint_path,
        "manifest": manifest_path,
        "alpha_scale": alpha_scale,
        "next_token": next_token,
        "per_abstract_dynamics": per_dynamics,
        "embedding_retrieval": retrieval,
        "per_sample": per_sample_rows,
    }

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with Path(output_path).open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(f"\nwrote summary to {output_path}")

    return summary


def _latest_checkpoint(run_dir: Path) -> Path | None:
    if not run_dir.exists():
        return None
    ckpts = sorted(
        run_dir.glob("cilf_checkpoint_step_*.pt"),
        key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
    )
    return ckpts[-1] if ckpts else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/cilf.yaml")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument(
        "--manifest",
        default="data/kinetic_transfer/manifest_kinetic_transfer.jsonl",
    )
    parser.add_argument("--alpha-scale", type=float, default=1.0)
    parser.add_argument(
        "--output",
        default="",
        help="Optional path to write the full JSON summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.strip()
    if not checkpoint:
        latest = _latest_checkpoint(Path("runs/cilf"))
        if latest is None:
            raise SystemExit("No checkpoint found under runs/cilf — pass --checkpoint or train first.")
        checkpoint = str(latest)
    evaluate(
        config_path=args.config,
        checkpoint_path=checkpoint,
        manifest_path=args.manifest,
        alpha_scale=float(args.alpha_scale),
        output_path=args.output or None,
    )


if __name__ == "__main__":
    main()
