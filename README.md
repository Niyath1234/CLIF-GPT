# CLIF-GPT

<p align="center">
  <img src="https://img.shields.io/badge/System%201-Frozen%20LLM-4f46e5?style=for-the-badge&logo=openai&logoColor=white" alt="System 1"/>
  <img src="https://img.shields.io/badge/Cross--Attention-Fusion-8b5cf6?style=for-the-badge" alt="Fusion"/>
  <img src="https://img.shields.io/badge/System%202-Video%20ODE-f59e0b?style=for-the-badge" alt="System 2"/>
  <img src="https://img.shields.io/badge/Output-Hidden%20Bias%20%2B%20LM%20Head-10b981?style=for-the-badge" alt="Output"/>
</p>

Cross-attentive **C**ausal-**I**ntuitive **L**ogit **F**usion: frozen LLM + trainable video dynamics for zero-shot kinetic transfer.

CILF couples a **frozen small LLM** (language context) with a **trainable video dynamics branch** (Physion-style causal intuition). Text and video token sequences interact through **bidirectional cross-attention**; a low-rank projector turns the joint representation into a **hidden-space bias** that steers the LM head on novel English prompts—without per-video lookup.

## Architecture

Block flow (Transformer-style: **video dynamics** ≈ encoder, **frozen LLM** ≈ decoder, **cross-attention** fuses them before a single LM head):

```mermaid
flowchart TB
  subgraph VIDEO["🎬 System 2 · Video dynamics (trainable)"]
    direction TB
    VF["Video frames<br/><i>Physion clip</i>"]
    SIG["SigLIP vision encoder<br/><i>frozen</i>"]
    PROJ["State projection"]
    ODE["Neural ODE dynamics<br/>ds/dt = f(s, a)"]
    VT["Video tokens H_video<br/><i>per-frame + Δstate</i>"]
    VF --> SIG --> PROJ --> ODE --> VT
  end

  subgraph TEXT["💬 System 1 · Language (frozen)"]
    direction TB
    PT["Prompt tokens"]
    LLM["Frozen LLM<br/><i>last-layer hidden states</i>"]
    HT["Text tokens H_text"]
    PT --> LLM --> HT
  end

  subgraph FUSION["🔗 Cross-attention fusion (trainable)"]
    direction TB
    CA1["Multi-Head Cross-Attn<br/>Q = H_text · K,V = H_video"]
    N1["Add + Norm"]
    CA2["Multi-Head Cross-Attn<br/>Q = H_video · K,V = H_text"]
    N2["Add + Norm"]
    JOINT["Joint tokens<br/><i>text + pooled video</i>"]
    BIAS["Low-rank bias projector<br/>H_bias"]
    CA1 --> N1 --> JOINT
    CA2 --> N2 --> JOINT
    JOINT --> BIAS
  end

  subgraph OUT["✨ Decode"]
    direction TB
    FUSE["Residual fusion<br/>H_fused = H_text[last] + α · H_bias"]
    HEAD["LM head<br/><i>frozen</i>"]
    LOGITS["Fused logits → next token"]
    FUSE --> HEAD --> LOGITS
  end

  VT -->|K, V| CA1
  HT -->|Q| CA1
  VT -->|Q| CA2
  HT -->|K, V| CA2
  HT --> FUSE
  BIAS --> FUSE

  subgraph LOSS["📉 Training losses (fusion stage)"]
    direction LR
    L1["CE on fused logits"]
    L2["JEPA energy"]
    L3["VICReg"]
    L4["Text↔video contrastive"]
    L5["Bias L2"]
  end

  LOGITS -.-> LOSS
  ODE -.-> L2

  classDef videoNode fill:#fbbf24,stroke:#d97706,stroke-width:2px,color:#1c1917
  classDef textNode fill:#60a5fa,stroke:#2563eb,stroke-width:2px,color:#ffffff
  classDef fusionNode fill:#c4b5fd,stroke:#7c3aed,stroke-width:2px,color:#1e1b4b
  classDef outNode fill:#34d399,stroke:#059669,stroke-width:2px,color:#064e3b
  classDef lossNode fill:#fda4af,stroke:#e11d48,stroke-width:2px,color:#4c0519

  class VF,SIG,PROJ,ODE,VT videoNode
  class PT,LLM,HT textNode
  class CA1,N1,CA2,N2,JOINT,BIAS fusionNode
  class FUSE,HEAD,LOGITS outNode
  class L1,L2,L3,L4,L5 lossNode

  style VIDEO fill:#fff7ed,stroke:#ea580c,stroke-width:3px,color:#9a3412
  style TEXT fill:#eff6ff,stroke:#2563eb,stroke-width:3px,color:#1e3a8a
  style FUSION fill:#f5f3ff,stroke:#7c3aed,stroke-width:3px,color:#4c1d95
  style OUT fill:#ecfdf5,stroke:#059669,stroke-width:3px,color:#065f46
  style LOSS fill:#fff1f2,stroke:#e11d48,stroke-width:3px,color:#9f1239

  linkStyle 6,7,8,9 stroke:#8b5cf6,stroke-width:2px
  linkStyle 10,11 stroke:#10b981,stroke-width:2px
```

**Equations**

```text
H_text'  = LayerNorm(H_text  + CrossAttn(Q=H_text,  K/V=H_video))
H_video' = LayerNorm(H_video + CrossAttn(Q=H_video, K/V=H_text))
H_joint  = combine(H_text', pooled H_video')
H_bias   = LowRankProjector(concat(H_joint, pooled_video))
H_fused  = H_text[last] + α · H_bias[last]
logits   = lm_head(H_fused)
```

**Training stages**

1. **JEPA pretrain** — energy / trajectory losses on video latent dynamics (`ode` predictor).
2. **CILF fusion** — joint loss: CE on fused logits + JEPA energy + VICReg + text↔video contrastive alignment + bias L2.

## Repository layout

```text
cilf/
  model.py      # CILFModel, CrossAttentionFusionBlock, vision + dynamics
  losses.py     # Energy, VICReg, alignment, bias regularization
  train.py      # Two-stage training loop
  data.py       # GeneralCausalVideoDataset (JSONL manifests)
  vocab.py      # Full or bounded target vocab
  video_io.py   # Frame loading
  dynamics/ode.py

configs/cilf.yaml
data/
  physion/              # Physion videos (place or symlink here)
  kinetic_transfer/     # Train / val / transfer manifests

scripts/
  setup_venv.sh
  build_kinetic_transfer_manifests.py
  prove_transfer.py
```

## Setup

```bash
bash scripts/setup_venv.sh
source .venv/bin/activate
pip install -e .
```

Place Physion clips under `data/physion/` (paths in manifests are relative to each manifest file).

## Data manifests

```bash
python scripts/build_kinetic_transfer_manifests.py
```

Produces `data/kinetic_transfer/manifest_kinetic_{train,val,transfer}.jsonl` pairing videos with narrative prompts and kinetic target words.

## Train

```bash
cilf-train --config configs/cilf.yaml
```

Checkpoints are written to `runs/cilf/`. For a full run, set `training.stage` to `jepa_pretrain` first (same config), then `cilf_fusion` with `checkpoint_path` pointing at the JEPA checkpoint.

## Evaluate transfer

```bash
python scripts/prove_transfer.py \
  --config configs/cilf.yaml \
  --manifest data/kinetic_transfer/manifest_kinetic_transfer.jsonl
```

## Design intent

Video teaches **amortized causal intuition** (dynamics + cross-attention), not a memorized label per clip. The same checkpoint should lift rank on **unseen** prompt–video pairs when the dynamics match the narrative (e.g. domino collapse → `fell` / `shattered` depending on α and context).

After changing fusion architecture, retrain from scratch; older checkpoints are not weight-compatible.
