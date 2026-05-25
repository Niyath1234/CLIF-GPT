"""Two-stage training entrypoint for the CILF narrative-causal engine."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from cilf.data import GeneralCausalVideoDataset
from cilf.losses import (
    abstract_dynamics_contrastive_loss,
    causal_hidden_l2_regularization,
    energy_loss,
    object_dynamics_energy_loss,
    object_temporal_consistency_loss,
    text_video_contrastive_loss,
    trajectory_energy_loss,
    vicreg_variance_loss,
)
from cilf.model import CILFModel
from cilf.vocab import DomainVocab


def load_config(path: str) -> dict[str, Any]:
    with Path(path).open("r") as handle:
        return yaml.safe_load(handle)


_SHORTHAND_KEYS = {
    "stage": ["training", "stage"],
    "max_steps": ["training", "max_steps"],
    "checkpoint_path": ["training", "checkpoint_path"],
    "batch_size": ["training", "batch_size"],
    "learning_rate": ["training", "learning_rate"],
}


def apply_overrides(config: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply simple key=value CLI overrides to a loaded YAML config."""

    expanded: list[str] = []
    for item in overrides or []:
        expanded.extend(part.strip() for part in item.split(",") if part.strip())

    for override in expanded:
        if "=" not in override:
            raise ValueError(f"Invalid override {override!r}; expected key=value.")
        raw_key, raw_value = override.split("=", 1)
        key = raw_key.strip()
        if not key:
            raise ValueError(f"Invalid override {override!r}; key cannot be empty.")

        path = _SHORTHAND_KEYS.get(key, key.split("."))
        value = yaml.safe_load(raw_value)
        cursor: dict[str, Any] = config
        for part in path[:-1]:
            if part not in cursor or not isinstance(cursor[part], dict):
                cursor[part] = {}
            cursor = cursor[part]
        cursor[path[-1]] = value
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device(device_name: str) -> torch.device:
    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_energy(
    outputs: dict[str, Any],
    corrupted_state: torch.Tensor,
    margin: float,
    use_trajectory: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if use_trajectory and "trajectory" in outputs:
        return trajectory_energy_loss(
            outputs["trajectory"],
            outputs["visual_states"],
            corrupted_state,
            margin=margin,
        )
    return energy_loss(
        outputs["predicted_next_state"],
        outputs["observed_next_state"].detach(),
        corrupted_state,
        margin=margin,
    )


def corrupt_next_state(
    observed_next_state: torch.Tensor,
    causal_trigger_label: torch.Tensor | None = None,
) -> torch.Tensor:
    """Prefer hard negatives from opposite trigger labels; fall back to batch roll."""

    batch_size = observed_next_state.shape[0]
    if batch_size <= 1:
        return (observed_next_state + torch.randn_like(observed_next_state) * 0.1).detach()

    if causal_trigger_label is not None and (causal_trigger_label >= 0).all():
        indices = []
        for idx in range(batch_size):
            opposite = torch.nonzero(
                causal_trigger_label != causal_trigger_label[idx], as_tuple=False
            ).flatten()
            if opposite.numel() > 0:
                indices.append(opposite[0])
            else:
                indices.append(torch.tensor((idx + 1) % batch_size, device=observed_next_state.device))
        return observed_next_state[torch.stack(indices)].detach()

    return torch.roll(observed_next_state, shifts=1, dims=0).detach()


def _log(verbose: bool, message: str) -> None:
    if verbose:
        print(message, flush=True)


def build_dataloader(
    data_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    tokenizer,
    domain_vocab: DomainVocab,
    shuffle: bool,
) -> DataLoader:
    dataset = GeneralCausalVideoDataset(
        manifest_path=data_cfg["manifest_path"],
        tokenizer=tokenizer,
        domain_vocab=domain_vocab,
        num_frames=int(data_cfg.get("num_frames", 8)),
        image_size=int(data_cfg.get("image_size", 224)),
        fps=int(data_cfg.get("fps", 10)),
        max_prompt_length=int(data_cfg.get("max_prompt_length", 64)),
        tracks_dir=data_cfg.get("tracks_dir"),
        max_tracks=int(data_cfg.get("max_tracks", 0)),
    )
    return DataLoader(
        dataset,
        batch_size=int(train_cfg.get("batch_size", 4)),
        shuffle=shuffle,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=False,
    )


def make_model(config: dict[str, Any], device: torch.device) -> CILFModel:
    model_cfg = config["model"]
    fixed_alpha = model_cfg.get("fixed_alpha")
    model = CILFModel(
        llm_name=model_cfg["llm_name"],
        state_dim=int(model_cfg.get("state_dim", 256)),
        jepa_hidden_dim=int(model_cfg.get("jepa_hidden_dim", 512)),
        vision_pretrained=bool(model_cfg.get("vision_pretrained", True)),
        vision_model_name=str(model_cfg.get("vision_model_name", "google/siglip-base-patch16-224")),
        dynamics_type=str(model_cfg.get("dynamics_type", "mlp")),
        ode_horizon=float(model_cfg.get("ode_horizon", 1.0)),
        ode_steps=int(model_cfg.get("ode_steps", 4)),
        ode_method=str(model_cfg.get("ode_method", "rk4")),
        ode_use_adjoint=bool(model_cfg.get("ode_use_adjoint", True)),
        fixed_alpha=float(fixed_alpha) if fixed_alpha is not None else None,
        bottleneck_dim=int(model_cfg.get("bottleneck_dim", 64)),
        cross_attention_heads=int(model_cfg.get("cross_attention_heads", 4)),
        cross_attention_dropout=float(model_cfg.get("cross_attention_dropout", 0.0)),
        use_sheaf_alignment=bool(model_cfg.get("use_sheaf_alignment", False)),
        sheaf_threshold=float(model_cfg.get("sheaf_threshold", 5.0)),
        use_tropical_fusion=bool(model_cfg.get("use_tropical_fusion", False)),
        use_object_tracking=bool(model_cfg.get("use_object_tracking", False)),
        num_object_slots=int(model_cfg.get("num_object_slots", 4)),
        slot_iters=int(model_cfg.get("slot_iters", 2)),
        use_relation_tokens=bool(model_cfg.get("use_relation_tokens", True)),
        use_detector_tracks=bool(model_cfg.get("use_detector_tracks", False)),
        max_detector_tracks=int(model_cfg.get("max_detector_tracks", 6)),
    ).to(device)
    model.llm.eval()
    if model.visual_encoder.freeze_foundation:
        model.visual_encoder.foundation.eval()
    return model


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def abstract_dynamics_group_ids(
    labels: list[str] | tuple[str, ...],
    device: torch.device,
) -> torch.Tensor:
    """Map string labels to integer group ids. ``unknown_dynamics`` -> -1."""

    name_to_id: dict[str, int] = {}
    ids: list[int] = []
    for label in labels:
        text = str(label).strip()
        if not text or text == "unknown_dynamics":
            ids.append(-1)
            continue
        if text not in name_to_id:
            name_to_id[text] = len(name_to_id)
        ids.append(name_to_id[text])
    return torch.tensor(ids, dtype=torch.long, device=device)


def semantic_soft_ce(
    logits: torch.Tensor,
    target_token_id: torch.Tensor,
    embedding_weight: torch.Tensor,
    epsilon: float,
    neighbor_count: int,
) -> torch.Tensor:
    """Cross-entropy with probability mass spread to unembedding-nearest neighbors."""

    if epsilon <= 0.0 or neighbor_count <= 0:
        return F.cross_entropy(logits, target_token_id)

    log_probs = F.log_softmax(logits, dim=-1)
    hard_loss = F.nll_loss(log_probs, target_token_id, reduction="none")

    with torch.no_grad():
        normalized_weight = F.normalize(embedding_weight.detach(), dim=-1)
        target_vectors = normalized_weight[target_token_id]
        similarities = target_vectors @ normalized_weight.T
        similarities.scatter_(1, target_token_id[:, None], float("-inf"))
        neighbor_count = min(neighbor_count, similarities.shape[-1] - 1)
        neighbor_scores, neighbor_ids = similarities.topk(k=neighbor_count, dim=-1)
        neighbor_weights = F.softmax(neighbor_scores, dim=-1)

    neighbor_loss = -(log_probs.gather(dim=-1, index=neighbor_ids) * neighbor_weights).sum(dim=-1)
    return ((1.0 - epsilon) * hard_loss + epsilon * neighbor_loss).mean()


def train_jepa(
    model: CILFModel,
    dataloader: DataLoader,
    config: dict[str, Any],
    device: torch.device,
) -> None:
    verbose = bool(config.get("verbose", False))
    train_cfg = config["training"]
    loss_cfg = config["loss"]
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(train_cfg.get("learning_rate", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )

    max_steps = int(train_cfg.get("max_steps", 1000))
    log_every = int(train_cfg.get("log_every", 10))
    save_every = int(train_cfg.get("save_every", 250))
    global_step = 0
    model.train()
    model.llm.eval()
    if model.visual_encoder.freeze_foundation:
        model.visual_encoder.foundation.eval()

    use_trajectory = model.dynamics_type == "ode"
    progress = tqdm(total=max_steps, desc="JEPA pretrain")
    while global_step < max_steps:
        for batch in dataloader:
            batch = move_batch(batch, device)
            outputs = model.predict_next_state(batch["frames"])
            corrupted_state = corrupt_next_state(outputs["observed_next_state"])
            causal_loss, positive_energy, negative_energy = compute_energy(
                outputs,
                corrupted_state,
                margin=float(loss_cfg.get("margin", 0.2)),
                use_trajectory=use_trajectory and bool(loss_cfg.get("use_trajectory_energy", True)),
            )
            variance_loss = vicreg_variance_loss(outputs["visual_states"])
            total_loss = causal_loss + float(loss_cfg.get("lambda_variance", 0.1)) * variance_loss

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("grad_clip", 1.0)))
            optimizer.step()

            global_step += 1
            progress.update(1)
            if global_step == 1 or global_step % log_every == 0:
                margin = (negative_energy - positive_energy).item()
                progress.write(
                    f"stage=jepa step={global_step} total={total_loss.item():.4f} "
                    f"energy={causal_loss.item():.4f} posE={positive_energy.item():.4f} "
                    f"negE={negative_energy.item():.4f} margin={margin:.4f} var={variance_loss.item():.4f}"
                )
            if global_step % save_every == 0:
                save_checkpoint(model, optimizer, global_step, train_cfg["output_dir"], stage="jepa")
            if global_step >= max_steps:
                break
    progress.close()
    save_checkpoint(model, optimizer, global_step, train_cfg["output_dir"], stage="jepa")


def train_cilf(
    model: CILFModel,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    config: dict[str, Any],
    device: torch.device,
) -> None:
    train_cfg = config["training"]
    loss_cfg = config["loss"]
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(train_cfg.get("learning_rate", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )

    max_steps = int(train_cfg.get("max_steps", 1000))
    log_every = int(train_cfg.get("log_every", 10))
    save_every = int(train_cfg.get("save_every", 250))
    val_every = int(train_cfg.get("val_every", 100))
    alpha_warmup_steps = max(1, int(train_cfg.get("alpha_warmup_steps", 200)))
    semantic_smoothing = float(loss_cfg.get("semantic_smoothing_epsilon", 0.0))
    semantic_neighbor_count = int(loss_cfg.get("semantic_neighbor_count", 0))
    global_step = 0
    model.train()
    model.llm.eval()
    if model.visual_encoder.freeze_foundation:
        model.visual_encoder.foundation.eval()

    use_trajectory = model.dynamics_type == "ode"
    progress = tqdm(total=max_steps, desc="CILF fusion")
    while global_step < max_steps:
        for batch in train_loader:
            batch = move_batch(batch, device)
            alpha_scale = min(1.0, (global_step + 1) / alpha_warmup_steps)
            outputs = model(
                frames=batch["frames"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                alpha_scale=alpha_scale,
                track_boxes=batch.get("track_boxes"),
                track_mask=batch.get("track_mask"),
            )
            corrupted_state = corrupt_next_state(outputs["observed_next_state"], batch["causal_trigger_label"])

            llm_loss = semantic_soft_ce(
                outputs["fused_logits"],
                batch["target_token_id"],
                model.llm.get_output_embeddings().weight,
                epsilon=semantic_smoothing,
                neighbor_count=semantic_neighbor_count,
            )
            causal_loss, positive_energy, negative_energy = compute_energy(
                outputs,
                corrupted_state,
                margin=float(loss_cfg.get("margin", 0.2)),
                use_trajectory=use_trajectory and bool(loss_cfg.get("use_trajectory_energy", True)),
            )
            variance_loss = vicreg_variance_loss(outputs["visual_states"])
            reg_loss = causal_hidden_l2_regularization(outputs["h_causal"])
            align_loss = text_video_contrastive_loss(
                outputs["text_pool"],
                outputs["video_pool"],
                temperature=float(loss_cfg.get("alignment_temperature", 0.07)),
            )
            alpha_regularizer = outputs["alpha"].mean() * (1.0 - alpha_scale)
            obstruction_penalty = 0.0
            if model.use_sheaf_alignment and "sheaf_obstruction_scalar" in outputs:
                obstruction_penalty = outputs["sheaf_obstruction_scalar"].mean()

            object_consistency_loss = torch.tensor(0.0, device=device)
            object_dynamics_loss = torch.tensor(0.0, device=device)
            object_pos_energy = torch.tensor(0.0, device=device)
            object_neg_energy = torch.tensor(0.0, device=device)
            has_object_signal = (
                "predicted_object_next" in outputs
                and "object_observed_next" in outputs
                and "slot_trajectory" in outputs
            )
            if has_object_signal:
                object_consistency_loss = object_temporal_consistency_loss(
                    outputs["slot_trajectory"],
                    temperature=float(loss_cfg.get("object_consistency_temperature", 0.1)),
                )
                object_dynamics_loss, object_pos_energy, object_neg_energy = object_dynamics_energy_loss(
                    outputs["predicted_object_next"],
                    outputs["object_observed_next"].detach(),
                    margin=float(loss_cfg.get("object_margin", 0.2)),
                )

            transition_loss = torch.tensor(0.0, device=device)
            dynamics_labels = batch.get("abstract_dynamics")
            if dynamics_labels is not None and float(loss_cfg.get("lambda_transition", 0.0)) > 0.0:
                group_ids = abstract_dynamics_group_ids(list(dynamics_labels), device)
                transition_loss = abstract_dynamics_contrastive_loss(
                    outputs["h_bias"],
                    group_ids,
                    temperature=float(loss_cfg.get("transition_temperature", 0.1)),
                )

            total_loss = (
                float(loss_cfg.get("lambda_ce", 1.0)) * llm_loss
                + float(loss_cfg.get("lambda_energy", 1.0)) * causal_loss
                + float(loss_cfg.get("lambda_variance", 0.1)) * variance_loss
                + float(loss_cfg.get("lambda_h_causal", 0.01)) * reg_loss
                + float(loss_cfg.get("lambda_alignment", 0.1)) * align_loss
                + float(loss_cfg.get("lambda_alpha_warmup", 0.1)) * alpha_regularizer
                + float(loss_cfg.get("lambda_obstruction", 0.1)) * obstruction_penalty
                + float(loss_cfg.get("lambda_object_consistency", 0.1)) * object_consistency_loss
                + float(loss_cfg.get("lambda_object_dynamics", 0.2)) * object_dynamics_loss
                + float(loss_cfg.get("lambda_transition", 0.0)) * transition_loss
            )

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("grad_clip", 1.0)))
            optimizer.step()

            global_step += 1
            progress.update(1)

            if global_step == 1 or global_step % log_every == 0:
                accuracy = (outputs["fused_logits"].argmax(dim=-1) == batch["target_token_id"]).float().mean()
                margin = (negative_energy - positive_energy).item()
                obj_line = ""
                if has_object_signal:
                    obj_line = (
                        " obj_consist={oc:.4f} obj_dyn={od:.4f} obj_posE={op:.4f} obj_negE={on:.4f}".format(
                            oc=float(object_consistency_loss.item()),
                            od=float(object_dynamics_loss.item()),
                            op=float(object_pos_energy.item()),
                            on=float(object_neg_energy.item()),
                        )
                    )
                if float(loss_cfg.get("lambda_transition", 0.0)) > 0.0:
                    obj_line += " transition={tr:.4f}".format(tr=float(transition_loss.item()))
                line = (
                    "stage=cilf step={step} total={total:.4f} ce={ce:.4f} acc={acc:.3f} "
                    "energy={energy:.4f} posE={pos:.4f} negE={neg:.4f} margin={margin:.4f} "
                    "align={align:.4f} reg={reg:.4f} var={var:.4f} "
                    "alpha={alpha:.3f} alpha_scale={scale:.2f}{obj}".format(
                        step=global_step,
                        total=total_loss.item(),
                        ce=llm_loss.item(),
                        acc=accuracy.item(),
                        energy=causal_loss.item(),
                        pos=positive_energy.item(),
                        neg=negative_energy.item(),
                        margin=margin,
                        align=align_loss.item(),
                        reg=reg_loss.item(),
                        var=variance_loss.item(),
                        alpha=outputs["alpha"].mean().item(),
                        scale=alpha_scale,
                        obj=obj_line,
                    )
                )
                progress.write(line)

            if global_step % save_every == 0:
                save_checkpoint(model, optimizer, global_step, train_cfg["output_dir"], stage="cilf")
            if val_loader is not None and global_step % val_every == 0:
                metrics = evaluate(model, val_loader, device)
                progress.write(
                    "val step={step} acc={acc:.3f} ce={ce:.4f} posE={pos:.4f} "
                    "negE={neg:.4f} margin={margin:.4f} alpha={alpha:.3f}".format(
                        step=global_step,
                        **metrics,
                    )
                )
                model.train()
                model.llm.eval()
                if model.visual_encoder.freeze_foundation:
                    model.visual_encoder.foundation.eval()

            if global_step >= max_steps:
                break

    progress.close()
    save_checkpoint(model, optimizer, global_step, train_cfg["output_dir"], stage="cilf")


@torch.no_grad()
def evaluate(model: CILFModel, dataloader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total = 0
    correct = 0.0
    ce_total = 0.0
    pos_total = 0.0
    neg_total = 0.0
    alpha_total = 0.0
    for batch in dataloader:
        batch = move_batch(batch, device)
        outputs = model(
            frames=batch["frames"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            alpha_scale=1.0,
            track_boxes=batch.get("track_boxes"),
            track_mask=batch.get("track_mask"),
        )
        corrupted_state = corrupt_next_state(outputs["observed_next_state"], batch["causal_trigger_label"])
        _, positive_energy, negative_energy = compute_energy(
            outputs,
            corrupted_state,
            margin=0.2,
            use_trajectory=model.dynamics_type == "ode",
        )
        batch_size = batch["frames"].shape[0]
        total += batch_size
        correct += (outputs["fused_logits"].argmax(dim=-1) == batch["target_token_id"]).float().sum().item()
        ce_total += F.cross_entropy(outputs["fused_logits"], batch["target_token_id"], reduction="sum").item()
        pos_total += positive_energy.item() * batch_size
        neg_total += negative_energy.item() * batch_size
        alpha_total += outputs["alpha"].mean().item() * batch_size
    pos = pos_total / max(1, total)
    neg = neg_total / max(1, total)
    return {
        "acc": correct / max(1, total),
        "ce": ce_total / max(1, total),
        "pos": pos,
        "neg": neg,
        "margin": neg - pos,
        "alpha": alpha_total / max(1, total),
    }


def save_checkpoint(
    model: CILFModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    output_dir: str,
    stage: str,
) -> None:
    path = Path(output_dir) / f"{stage}_checkpoint_step_{step}.pt"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "stage": stage,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CILF on generic causal video-caption data.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help=(
            "Override YAML key=value (repeat flag or comma-separate). "
            "Examples: --override stage=jepa_pretrain --override max_steps=400 "
            'or --override "stage=jepa_pretrain,max_steps=400".'
        ),
    )
    args = parser.parse_args()
    config = apply_overrides(load_config(args.config), args.override)
    verbose = bool(config.get("verbose", False))
    set_seed(int(config.get("seed", 13)))
    device = choose_device(config.get("device", "auto"))
    _log(verbose, f"device={device}")

    model_cfg = config["model"]
    data_cfg = config["data"]
    train_cfg = config["training"]
    stage = train_cfg.get("stage", "jepa_pretrain")

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["llm_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    domain_vocab = DomainVocab.from_config(tokenizer, data_cfg)
    allowed_token_ids = domain_vocab.allowed_token_ids()
    train_loader = build_dataloader(data_cfg, train_cfg, tokenizer, domain_vocab, shuffle=True)
    _log(verbose, f"stage={stage} dataset_size={len(train_loader.dataset)} manifest={data_cfg['manifest_path']}")
    _log(verbose, f"active_vocab_size={len(allowed_token_ids)}")

    sample = train_loader.dataset[0]
    _log(
        verbose,
        "sample0 "
        f"frames={tuple(sample['frames'].shape)} "
        f"prompt={sample['prompt']!r} "
        f"target={sample['causal_consequence']!r} "
        f"trigger={sample['causal_trigger_label'].item()}",
    )

    _log(verbose, f"loading_llm={model_cfg['llm_name']}")
    model = make_model(config, device)
    checkpoint_path = train_cfg.get("checkpoint_path")
    if checkpoint_path and (stage != "jepa_pretrain" or bool(train_cfg.get("resume_from_checkpoint", False))):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"], strict=False)
        _log(verbose, f"loaded_checkpoint={checkpoint_path}")
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    _log(verbose, f"trainable_params={trainable:,}")

    if stage == "jepa_pretrain":
        train_jepa(model, train_loader, config, device)
    elif stage == "cilf_fusion":
        val_loader = None
        if data_cfg.get("val_manifest_path"):
            val_cfg = {**data_cfg, "manifest_path": data_cfg["val_manifest_path"]}
            val_vocab = DomainVocab.from_config(tokenizer, val_cfg)
            val_loader = build_dataloader(val_cfg, train_cfg, tokenizer, val_vocab, shuffle=False)
            _log(verbose, f"val_dataset_size={len(val_loader.dataset)} manifest={data_cfg['val_manifest_path']}")
        train_cilf(model, train_loader, val_loader, config, device)
    else:
        raise ValueError(f"Unknown training stage: {stage}")


if __name__ == "__main__":
    main()
