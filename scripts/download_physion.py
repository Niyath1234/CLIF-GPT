#!/usr/bin/env python3
"""Download the Physion dataset (test core + train dynamics) into ``data/physion``.

The dataset is hosted by the original authors on a public S3 bucket. We only
fetch the asset(s) actually requested via ``--split`` and skip already-extracted
content. Re-runs are idempotent.

Splits:
    test    -> PhysionTest-Core (~270 MB, 8 scenarios, used for human eval)
    train   -> PhysionTrain-Dynamics (~770 MB, ~2k clips per scenario)
    both    -> test + train

Example:

```
python scripts/download_physion.py --split test
python scripts/download_physion.py --split both --dest data/physion
```

The script needs ``curl`` and the system ``unzip``/``tar`` binaries. We avoid
loading the entire archive into memory and stream straight to disk.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

PHYSION_URLS = {
    "test": "https://physics-benchmarking-neurips2021-dataset.s3.amazonaws.com/Physion.zip",
    "train": "https://physics-benchmarking-neurips2021-dataset.s3.amazonaws.com/PhysionTrainMP4s.tar.gz",
}

TEST_MARKER = "PhysionTest-Core"
TRAIN_MARKER = "PhysionTrain-Dynamics"


def _run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def already_extracted(dest: Path, marker: str) -> bool:
    return (dest / marker).exists()


def download(url: str, target: Path) -> None:
    if target.exists() and target.stat().st_size > 0:
        print(f"[skip] {target} already present ({target.stat().st_size / 1e6:.1f} MB)", flush=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {target}", flush=True)
    _run(["curl", "-L", "--fail", "--progress-bar", "-o", str(target), url])


def extract_zip(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    _run(["unzip", "-q", "-o", str(archive), "-d", str(dest)])


def extract_tar(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    _run(["tar", "-xzf", str(archive), "-C", str(dest)])


def process_split(split: str, dest: Path, raw_dir: Path, keep_archive: bool) -> None:
    url = PHYSION_URLS[split]
    archive = raw_dir / Path(url).name
    if split == "test":
        marker = TEST_MARKER
    elif split == "train":
        marker = TRAIN_MARKER
    else:
        raise ValueError(split)

    if already_extracted(dest, marker):
        print(f"[skip] {marker} already extracted under {dest}", flush=True)
    else:
        download(url, archive)
        if archive.suffix == ".zip":
            extract_zip(archive, dest)
        else:
            extract_tar(archive, dest)

    if not keep_archive and archive.exists():
        try:
            archive.unlink()
            print(f"Removed archive {archive}", flush=True)
        except OSError as exc:
            print(f"WARN: failed to remove archive {archive}: {exc}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("test", "train", "both"), default="test")
    parser.add_argument("--dest", default="data/physion", help="Target directory for the dataset.")
    parser.add_argument("--raw-dir", default="data/physion/raw", help="Where to cache downloaded archives.")
    parser.add_argument("--keep-archive", action="store_true", help="Keep the .zip/.tar.gz on disk after extracting.")
    args = parser.parse_args()

    if shutil.which("curl") is None:
        sys.exit("curl is required to download Physion (please install curl).")
    if shutil.which("unzip") is None:
        sys.exit("unzip is required to extract the test split (please install unzip).")
    if shutil.which("tar") is None:
        sys.exit("tar is required to extract the train split (please install tar).")

    dest = Path(args.dest)
    raw_dir = Path(args.raw_dir)
    dest.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    splits = ("test", "train") if args.split == "both" else (args.split,)
    for split in splits:
        process_split(split, dest, raw_dir, args.keep_archive)

    print("\nPhysion ready at:", dest.resolve())
    print("Next steps:")
    print("  1. Build/refresh manifests:")
    print("     python scripts/build_physion_tracks_manifest.py --root", dest)
    print("  2. Precompute YOLO tracks:")
    print("     python scripts/precompute_yolo_tracks.py \\")
    print("       --manifest", dest / "manifest_train.jsonl", "\\")
    print("       --tracks-dir", dest / "tracks", "\\")
    print("       --model yolov8s-worldv2.pt --imgsz 480")


if __name__ == "__main__":
    main()
