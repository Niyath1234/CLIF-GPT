#!/usr/bin/env python3
"""Localhost demo: ask any question → fused conclusion + source Physion video.

Usage:
  source .venv/bin/activate
  pip install flask   # or: pip install -e ".[demo]"
  python scripts/demo_localhost.py

Then open http://127.0.0.1:8765
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from flask import Flask, jsonify, request, send_file
except ImportError as exc:
    raise SystemExit(
        "Flask is required for this demo. Install with: pip install flask"
    ) from exc

from cilf.intuition_engine import IntuitionEngine

WEB_DIR = PROJECT_ROOT / "demo" / "web"
DEFAULT_CONFIG = PROJECT_ROOT / "configs/cilf_physion_office.yaml"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "runs/cilf_physion_office/cilf_checkpoint_step_2000.pt"
DEFAULT_MANIFEST = PROJECT_ROOT / "data/kinetic_transfer/manifest_kinetic_val.jsonl"
DEFAULT_INDEX_CACHE = PROJECT_ROOT / "runs/cilf_physion_office/prompt_index.pt"

app = Flask(__name__, static_folder=str(WEB_DIR), static_url_path="/static")
ENGINE: IntuitionEngine | None = None


def _discover_checkpoint(explicit: str) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    if DEFAULT_CHECKPOINT.exists():
        return DEFAULT_CHECKPOINT
    run_dir = PROJECT_ROOT / "runs/cilf_physion_office"
    ckpts = sorted(
        run_dir.glob("cilf_checkpoint_step_*.pt"),
        key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
    )
    if ckpts:
        return ckpts[-1]
    raise FileNotFoundError("No checkpoint found. Train first or pass --checkpoint.")


@app.get("/")
def index():
    return send_file(WEB_DIR / "index.html")


@app.get("/api/health")
def health():
    return jsonify(
        {
            "ok": True,
            "clips_indexed": len(ENGINE.clips) if ENGINE else 0,
            "checkpoint": str(ENGINE.checkpoint_path) if ENGINE else None,
        }
    )


@app.post("/api/ask")
def ask():
    if ENGINE is None:
        return jsonify({"error": "Model not loaded"}), 503
    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or payload.get("prompt") or "").strip()
    if not question:
        return jsonify({"error": "Provide a question in JSON body: {\"question\": \"...\"}"}), 400
    try:
        result = ENGINE.ask(question)
        return jsonify(ENGINE.result_to_json(result))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.get("/api/video/<stim_id>")
def video(stim_id: str):
    if ENGINE is None:
        return jsonify({"error": "Model not loaded"}), 503
    path = ENGINE.stim_id_to_video_path(stim_id)
    if path is None or not path.exists():
        return jsonify({"error": f"Video not found for stim_id={stim_id}"}), 404
    return send_file(path, mimetype="video/mp4", conditional=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--index-cache", default=str(DEFAULT_INDEX_CACHE))
    parser.add_argument("--candidates", type=int, default=12)
    parser.add_argument("--alpha-scale", type=float, default=1.0)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    global ENGINE
    args = parse_args()
    checkpoint = _discover_checkpoint(args.checkpoint)

    print("Loading CILF intuition engine (first run builds prompt index)...")
    ENGINE = IntuitionEngine(
        config_path=args.config,
        checkpoint_path=checkpoint,
        manifest_path=args.manifest,
        index_cache=args.index_cache,
        candidate_pool=args.candidates,
        alpha_scale=args.alpha_scale,
    )
    print(f"  clips: {len(ENGINE.clips)}")
    print(f"  checkpoint: {checkpoint}")
    print(f"  manifest: {args.manifest}")
    print(f"\nOpen http://{args.host}:{args.port}\n")

    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False)


if __name__ == "__main__":
    main()
