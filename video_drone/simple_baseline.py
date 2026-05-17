from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from .taxonomy import normalize_event_name


def image_embedding(path: Path, bins: int = 16) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("RGB").resize((224, 224))
        arr = np.asarray(img, dtype=np.float32) / 255.0
    feats = []
    for c in range(3):
        hist, _ = np.histogram(arr[:, :, c], bins=bins, range=(0.0, 1.0), density=True)
        feats.append(hist.astype(np.float32))
    mean = arr.mean(axis=(0, 1))
    std = arr.std(axis=(0, 1))
    vec = np.concatenate(feats + [mean, std]).astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _load_index(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _example_records(index: dict) -> list[dict]:
    return [
        r
        for r in index["records"]
        if r["sample_type"] == "example_image" and not r.get("is_coordinate_image", False)
    ]


def _target_images(index: dict, frame_root: Path | None) -> list[dict]:
    if frame_root is None:
        return [
            r
            for r in index["records"]
            if r["sample_type"] == "example_image" and not r.get("is_coordinate_image", False)
        ]
    targets = []
    for path in sorted(frame_root.rglob("*.jpg")):
        event = path.parents[1].name if len(path.parents) > 1 else normalize_event_name(path.parent.name)
        targets.append(
            {
                "sample_id": str(path.relative_to(frame_root)).replace("\\", "/"),
                "source_path": str(path.resolve()),
                "folder_event": event,
                "event": event,
                "sample_type": "frame",
            }
        )
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simple visual similarity baseline using color histograms. Replace with CLIP/VLM later."
    )
    parser.add_argument("--index", default="outputs/demo_index/dataset_index.json", type=Path)
    parser.add_argument("--frame-root", default=None, type=Path)
    parser.add_argument("--out-dir", default="outputs/simple_baseline", type=Path)
    parser.add_argument("--top-k", default=5, type=int)
    args = parser.parse_args()

    index = _load_index(args.index)
    examples = _example_records(index)
    targets = _target_images(index, args.frame_root)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    example_embs = []
    for rec in examples:
        path = Path(rec["source_path"])
        try:
            example_embs.append((rec, image_embedding(path)))
        except Exception as exc:
            print(f"Skipping example {path}: {exc}")

    pred_path = args.out_dir / ("frame_predictions.jsonl" if args.frame_root else "image_predictions.jsonl")
    with pred_path.open("w", encoding="utf-8") as f:
        for target in targets:
            path = Path(target["source_path"])
            try:
                emb = image_embedding(path)
            except Exception as exc:
                print(f"Skipping target {path}: {exc}")
                continue
            scores_by_event: dict[str, float] = {}
            nearest = []
            for ex_rec, ex_emb in example_embs:
                score = cosine(emb, ex_emb)
                event = ex_rec["event"]
                scores_by_event[event] = max(score, scores_by_event.get(event, -math.inf))
                nearest.append({"event": event, "source_path": ex_rec["source_path"], "score": score})
            predicted = [
                {"event": event, "score": round(score, 4)}
                for event, score in sorted(scores_by_event.items(), key=lambda x: x[1], reverse=True)[: args.top_k]
            ]
            record = {
                "sample_id": target["sample_id"],
                "source_path": target["source_path"],
                "ground_truth_event": target.get("event"),
                "method": "color-histogram-nearest-example",
                "predicted_events": predicted,
                "nearest_examples": sorted(nearest, key=lambda x: x["score"], reverse=True)[: args.top_k],
                "caption": "",
                "evidence": "",
                "department": [],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote predictions: {pred_path}")


if __name__ == "__main__":
    main()

