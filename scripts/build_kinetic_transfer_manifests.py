"""Build cross-domain kinetic-transfer manifests from Physion videos.

Each physion video (collide/drop/dominoes/roll/...) is paired with several
human-narrative prompts that share the **same abstract kinetic outcome verb**.

The goal: train the CILF model so that any video of "impact + falling" pushes
probability toward kinetic-fall verbs (fell, tumbled, dropped, crashed) across
unrelated human contexts (walking, tripping, vases, cars, boxes).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHYSION_ROOT = PROJECT_ROOT / "data/physion/PhysionTest-Core/Physion"


SCENARIO_DYNAMICS: dict[str, str] = {
    "Dominoes": "support_loss_chain_fall",
    "Drop": "support_loss_freefall",
    "Collide": "momentum_transfer_collision",
    "Roll": "rolling_motion",
    "Drape": "support_loss_freefall",
    "Contain": "containment_settle",
    "Link": "contact_capture",
    "Support": "support_loss_freefall",
}


SCENARIO_OBJECTS: dict[str, dict[str, str | list[str]]] = {
    "Dominoes": {
        "objects": ["domino_a", "domino_b", "ground"],
        "subject_object": "domino_a",
        "affected_object": "domino_b",
        "interaction": "leading domino topples and contacts the next",
        "precondition": "tiles standing in chain",
        "postcondition": "tiles fallen on ground",
        "causal_state_change": "support_lost -> chain_fall -> impact",
    },
    "Drop": {
        "objects": ["object", "table", "ground"],
        "subject_object": "object",
        "affected_object": "object",
        "interaction": "object loses contact with support",
        "precondition": "object resting on support",
        "postcondition": "object on ground",
        "causal_state_change": "support_lost -> freefall -> impact",
    },
    "Collide": {
        "objects": ["object_a", "object_b"],
        "subject_object": "object_a",
        "affected_object": "object_b",
        "interaction": "moving body strikes another",
        "precondition": "object_a approaches object_b",
        "postcondition": "object_b deflected or displaced",
        "causal_state_change": "momentum_transfer -> displacement",
    },
    "Roll": {
        "objects": ["rolling_object", "ramp"],
        "subject_object": "rolling_object",
        "affected_object": "rolling_object",
        "interaction": "object rolls along surface under gravity",
        "precondition": "object at rest on incline",
        "postcondition": "object translating along surface",
        "causal_state_change": "gravity_acts -> rolling_motion",
    },
    "Drape": {
        "objects": ["cloth", "support"],
        "subject_object": "cloth",
        "affected_object": "cloth",
        "interaction": "flexible cloth detaches from support",
        "precondition": "cloth hung on support",
        "postcondition": "cloth on ground",
        "causal_state_change": "support_lost -> freefall",
    },
    "Contain": {
        "objects": ["small_object", "container"],
        "subject_object": "small_object",
        "affected_object": "container",
        "interaction": "object enters container",
        "precondition": "object above container opening",
        "postcondition": "object settled inside container",
        "causal_state_change": "freefall -> containment -> settle",
    },
    "Link": {
        "objects": ["flexible_link", "anchor"],
        "subject_object": "flexible_link",
        "affected_object": "anchor",
        "interaction": "moving link contacts and wraps anchor",
        "precondition": "link in motion toward anchor",
        "postcondition": "link constrained by anchor",
        "causal_state_change": "contact -> capture",
    },
    "Support": {
        "objects": ["object", "support"],
        "subject_object": "support",
        "affected_object": "object",
        "interaction": "support is removed, object follows gravity",
        "precondition": "object resting on support",
        "postcondition": "object on ground",
        "causal_state_change": "support_removed -> freefall -> impact",
    },
}


SCENARIO_TEMPLATES: dict[str, list[tuple[str, str]]] = {
    "Dominoes": [
        ("I was walking and hit my leg and", "fell"),
        ("She tripped over the curb and", "fell"),
        ("He stumbled over the rug and", "fell"),
        ("The runner caught his foot on a stone and", "fell"),
        ("The boy lost his balance and", "fell"),
        ("The child slipped on the wet floor and", "fell"),
        ("The old man missed the step and", "fell"),
        ("Her shoe caught the pavement and she", "fell"),
    ],
    "Drop": [
        ("The glass slipped from her hand and", "shattered"),
        ("He let go of the heavy box and it", "fell"),
        ("She knocked the cup off the counter and it", "fell"),
        ("The vase tipped off the shelf and", "shattered"),
        ("He dropped the phone and it", "fell"),
        ("The plate slid from the tray and", "shattered"),
        ("The painter lost his grip and the bucket", "fell"),
    ],
    "Collide": [
        ("The two cars met at the intersection and", "crashed"),
        ("He swung the bat into the ball and they", "collided"),
        ("The cyclist hit the curb and", "crashed"),
        ("The bumper struck the wall and", "crashed"),
        ("The truck rear-ended the sedan and they", "collided"),
        ("The football players ran into each other and", "collided"),
    ],
    "Roll": [
        ("The ball slid down the slope and", "rolled"),
        ("The marble started downhill and", "rolled"),
        ("The ball gathered speed and", "rolled"),
        ("The wheel began to spin and", "rolled"),
        ("The bottle tipped on its side and", "rolled"),
        ("The pebble slipped off the ledge and", "rolled"),
        ("The cart was pushed forward and", "rolled"),
    ],
    "Drape": [
        ("She let the blanket go and it", "fell"),
        ("The cloth slid off the table and", "fell"),
        ("The sheet came loose from the line and", "fell"),
        ("He released the curtain and it", "fell"),
        ("The flag was unhooked and it", "fell"),
    ],
    "Contain": [
        ("He dropped the marble into the cup and it", "fell"),
        ("She poured the beads into the jar and they", "settled"),
        ("The pebbles were tipped into the bowl and", "settled"),
        ("He flicked the coin into the dish and it", "fell"),
    ],
    "Link": [
        ("The chain caught on the post and", "caught"),
        ("The rope hit the hook and", "caught"),
        ("The cable wrapped the branch and", "caught"),
        ("The wire hit the fence and", "caught"),
    ],
    "Support": [
        ("The plank slid out and the box", "fell"),
        ("He pulled the prop and the shelf", "fell"),
        ("The leg of the table broke and it", "fell"),
        ("She kicked the stool and it", "fell"),
        ("The brace gave way and the beam", "fell"),
    ],
}


HELD_OUT_TRANSFER_PROMPTS: list[tuple[str, str, str]] = [
    # scenario_label_for_split, prompt, expected_kinetic_word
    ("fall_walking", "I was walking, hit my leg and", "fell"),
    ("fall_balance", "I lost my balance on the stairs and", "fell"),
    ("fall_vase", "The vase slipped off the table and", "shattered"),
    ("crash_cars", "The two cars met at the intersection and", "crashed"),
    ("roll_ball", "The ball started down the hill and", "rolled"),
    ("drop_cup", "The cup fell off the counter and", "shattered"),
]


def _collect_videos(scenario_dir: Path) -> list[Path]:
    mp4_dir = scenario_dir / "mp4s"
    if not mp4_dir.exists():
        return []
    return sorted(mp4_dir.glob("*.mp4"))


def _emit_rows(
    videos: list[Path],
    scenario: str,
    templates: list[tuple[str, str]],
    manifest_dir: Path,
    repeats_per_video: int,
    rng: random.Random,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for video in videos:
        for _ in range(repeats_per_video):
            prompt, target = rng.choice(templates)
            try:
                rel_video = video.resolve().relative_to(manifest_dir.resolve())
            except ValueError:
                import os
                rel_video = Path(os.path.relpath(video.resolve(), manifest_dir.resolve()))
            obj_meta = SCENARIO_OBJECTS.get(scenario, {})
            row: dict[str, object] = {
                "video_path": str(rel_video),
                "prompt": prompt,
                "causal_consequence": target,
                "causal_trigger": True,
                "scenario": f"kinetic_{scenario.lower()}",
                "stim_id": video.stem,
                "split": "kinetic",
                "abstract_dynamics": SCENARIO_DYNAMICS.get(scenario, "unknown_dynamics"),
            }
            row.update(obj_meta)
            rows.append(row)
    return rows


def _write_jsonl(rows: Iterable[dict[str, object]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
            count += 1
    return count


def build(
    physion_root: Path,
    output_dir: Path,
    project_root: Path,
    val_fraction: float,
    repeats_per_video: int,
    seed: int,
) -> dict[str, int]:
    rng = random.Random(seed)
    train_rows: list[dict[str, object]] = []
    val_rows: list[dict[str, object]] = []
    manifest_dir = output_dir

    for scenario, templates in SCENARIO_TEMPLATES.items():
        scenario_dir = physion_root / scenario
        videos = _collect_videos(scenario_dir)
        if not videos:
            print(f"[skip] no videos for {scenario} at {scenario_dir}")
            continue

        rng.shuffle(videos)
        val_count = max(1, int(len(videos) * val_fraction))
        val_videos = videos[:val_count]
        train_videos = videos[val_count:]

        train_rows.extend(
            _emit_rows(train_videos, scenario, templates, manifest_dir, repeats_per_video, rng)
        )
        val_rows.extend(_emit_rows(val_videos, scenario, templates, manifest_dir, 1, rng))
        print(f"{scenario}: total={len(videos)} train_videos={len(train_videos)} val_videos={len(val_videos)}")

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)

    train_path = output_dir / "manifest_kinetic_train.jsonl"
    val_path = output_dir / "manifest_kinetic_val.jsonl"
    transfer_path = output_dir / "manifest_kinetic_transfer.jsonl"

    # Transfer eval: held-out prompts the model never saw in training, paired with
    # one random video per scenario to test cross-domain generalization.
    transfer_rows: list[dict[str, object]] = []
    import os

    for scenario_label, prompt, expected_word in HELD_OUT_TRANSFER_PROMPTS:
        scenarios = list(SCENARIO_TEMPLATES.keys())
        for scenario in scenarios:
            scenario_dir = physion_root / scenario
            videos = _collect_videos(scenario_dir)
            if not videos:
                continue
            video = rng.choice(videos)
            rel_video = Path(os.path.relpath(video.resolve(), manifest_dir.resolve()))
            obj_meta = SCENARIO_OBJECTS.get(scenario, {})
            row: dict[str, object] = {
                "video_path": str(rel_video),
                "prompt": prompt,
                "causal_consequence": expected_word,
                "causal_trigger": True,
                "scenario": f"transfer_{scenario_label}_via_{scenario.lower()}",
                "stim_id": video.stem,
                "split": "transfer",
                "abstract_dynamics": SCENARIO_DYNAMICS.get(scenario, "unknown_dynamics"),
            }
            row.update(obj_meta)
            transfer_rows.append(row)

    train_count = _write_jsonl(train_rows, train_path)
    val_count = _write_jsonl(val_rows, val_path)
    transfer_count = _write_jsonl(transfer_rows, transfer_path)

    print(f"\nWrote {train_count} train rows  -> {train_path}")
    print(f"Wrote {val_count} val rows    -> {val_path}")
    print(f"Wrote {transfer_count} transfer rows -> {transfer_path}")
    return {"train": train_count, "val": val_count, "transfer": transfer_count}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physion-root", default=str(DEFAULT_PHYSION_ROOT))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "data/kinetic_transfer"))
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--repeats-per-video", type=int, default=3)
    parser.add_argument("--seed", type=int, default=23)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build(
        physion_root=Path(args.physion_root),
        output_dir=Path(args.output_dir),
        project_root=Path(args.project_root),
        val_fraction=float(args.val_fraction),
        repeats_per_video=int(args.repeats_per_video),
        seed=int(args.seed),
    )


if __name__ == "__main__":
    main()
