"""Print a clean before/after proof sheet for cross-domain causal transfer.

For each (prompt, video) pair the script reports:
  - LLM-only top-1 token  (System 1 alone)
  - CILF fused top-1 token (hidden-space fusion + LM head)
  - Rank of the *expected* kinetic word in the fused distribution
  - Top-5 fused tokens (open vocabulary)
"""

from __future__ import annotations

import argparse
from pathlib import Path

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


def evaluate(
    config_path: str,
    checkpoint_path: str,
    manifest_path: str,
    alpha_scale: float,
) -> None:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    import os

    forced = os.environ.get("CILF_FORCE_DEVICE")
    device = choose_device(forced if forced else str(config.get("device", "auto")))

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["llm_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data_cfg = dict(config["data"])
    data_cfg["manifest_path"] = manifest_path
    domain_vocab = DomainVocab.from_config(tokenizer, data_cfg)
    dataset = GeneralCausalVideoDataset(
        manifest_path=manifest_path,
        tokenizer=tokenizer,
        domain_vocab=domain_vocab,
        num_frames=int(data_cfg.get("num_frames", 6)),
        image_size=int(data_cfg.get("image_size", 224)),
        fps=int(data_cfg.get("fps", 8)),
        max_prompt_length=int(data_cfg.get("max_prompt_length", 48)),
    )

    model = make_model(config, device)
    state = torch.load(Path(checkpoint_path), map_location=device)
    model.load_state_dict(state["model"], strict=False)
    model.eval()

    print(f"checkpoint={checkpoint_path}")
    print(f"manifest={manifest_path}  rows={len(dataset)}  alpha_scale={alpha_scale}")
    print("=" * 100)

    matched = 0
    total = 0
    fused_rank_better = 0
    for idx in range(len(dataset)):
        sample = dataset[idx]
        record = dataset.records[idx]
        expected = record.causal_consequence

        batch = {
            "frames": sample["frames"].unsqueeze(0),
            "input_ids": sample["input_ids"].unsqueeze(0),
            "attention_mask": sample["attention_mask"].unsqueeze(0),
        }
        batch = move_batch(batch, device)

        with torch.no_grad():
            outputs = model(
                frames=batch["frames"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                alpha_scale=alpha_scale,
            )
            z_l = outputs["lingual_logits"][0]
            z_fused = outputs["fused_logits"][0]

        llm_top = topk_tokens(z_l, tokenizer, k=1)[0]
        fused_top5 = topk_tokens(z_fused, tokenizer, k=5)
        rank_llm = expected_rank(z_l, tokenizer, expected)
        rank_fused = expected_rank(z_fused, tokenizer, expected)

        total += 1
        if fused_top5[0][0] == expected:
            matched += 1
        if rank_fused < rank_llm:
            fused_rank_better += 1

        print(
            f"[{idx:02d}] {record.scenario}\n"
            f"     prompt: {record.prompt!r}\n"
            f"     expected: {expected!r}\n"
            f"     LLM top1: {llm_top[0]!r:>14}  rank_LLM={rank_llm}\n"
            f"     FUSED   : {fused_top5[0][0]!r:>14}  rank_FUSED={rank_fused}\n"
            f"     fused top5: {[(w, round(p, 4)) for w, p in fused_top5]}\n"
        )

    print("=" * 100)
    print(
        f"Summary: fused_top1 matches expected = {matched}/{total} "
        f"({matched / max(1, total) * 100:.1f}%); "
        f"fused improves rank over LLM in {fused_rank_better}/{total} cases."
    )


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
    parser.add_argument("--manifest", default="data/kinetic_transfer/manifest_kinetic_transfer.jsonl")
    parser.add_argument("--alpha-scale", type=float, default=1.0)
    parser.add_argument("--device", default=None, help="Override device (cpu/mps/cuda).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.strip()
    if not checkpoint:
        latest = _latest_checkpoint(Path("runs/cilf"))
        if latest is None:
            raise SystemExit("No checkpoint found under runs/cilf — pass --checkpoint or train first.")
        checkpoint = str(latest)
    if args.device is not None:
        import os

        os.environ["CILF_FORCE_DEVICE"] = args.device
    evaluate(
        config_path=args.config,
        checkpoint_path=checkpoint,
        manifest_path=args.manifest,
        alpha_scale=float(args.alpha_scale),
    )


if __name__ == "__main__":
    main()
