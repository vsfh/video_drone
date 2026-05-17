from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image

from .paths import IMAGE_EXTS, VIDEO_EXTS, safe_relpath
from .taxonomy import EVENT_TAXONOMY, PRIMARY_EVENTS, event_lookup, normalize_event_name


def _image_info(path: Path) -> dict:
    with Image.open(path) as img:
        return {"width": img.width, "height": img.height}


def scan_demo(demo_root: Path) -> dict:
    events = event_lookup()
    records: list[dict] = []
    folders: dict[str, dict] = {}

    for folder in sorted([p for p in demo_root.iterdir() if p.is_dir()]):
        folder_event = normalize_event_name(folder.name)
        folders[folder_event] = {
            "event": folder_event,
            "is_primary": folder_event in PRIMARY_EVENTS,
            "files": [],
        }
        for path in sorted(folder.rglob("*")):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix not in IMAGE_EXTS and suffix not in VIDEO_EXTS:
                continue
            stem_event = normalize_event_name(path.stem)
            is_coordinate = "坐标图" in path.stem
            sample_type = "video" if suffix in VIDEO_EXTS else ("coordinate_image" if is_coordinate else "example_image")
            event = stem_event if stem_event in events else folder_event
            rec = {
                "sample_id": f"{safe_relpath(path, demo_root)}",
                "source_path": str(path.resolve()),
                "relative_path": safe_relpath(path, demo_root),
                "folder_event": folder_event,
                "event": event,
                "sample_type": sample_type,
                "is_primary_event": event in PRIMARY_EVENTS,
                "is_coordinate_image": is_coordinate,
                "suffix": suffix,
            }
            if sample_type.endswith("image"):
                rec.update(_image_info(path))
            records.append(rec)
            folders[folder_event]["files"].append(rec["relative_path"])

    return {"demo_root": str(demo_root.resolve()), "records": records, "folders": folders}


def write_outputs(index: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dataset_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "dataset_records.jsonl").open("w", encoding="utf-8") as f:
        for rec in index["records"]:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    fields = sorted({k for rec in index["records"] for k in rec})
    with (out_dir / "dataset_records.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(index["records"])

    taxonomy_path = out_dir / "event_taxonomy.json"
    taxonomy_path.write_text(json.dumps(EVENT_TAXONOMY, ensure_ascii=False, indent=2), encoding="utf-8")

    prediction_template = out_dir / "prediction_template.jsonl"
    with prediction_template.open("w", encoding="utf-8") as f:
        for rec in index["records"]:
            if rec["sample_type"] == "video":
                f.write(
                    json.dumps(
                        {
                            "sample_id": rec["sample_id"],
                            "source_path": rec["source_path"],
                            "method": "replace-with-method-name",
                            "predicted_events": [],
                            "keyframes": [],
                            "caption": "",
                            "evidence": "",
                            "department": [],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Index demo videos, example PNGs, and coordinate images.")
    parser.add_argument("--demo-root", default="../demo", type=Path)
    parser.add_argument("--out-dir", default="outputs/demo_index", type=Path)
    args = parser.parse_args()

    if not args.demo_root.exists():
        raise FileNotFoundError(f"Demo root not found: {args.demo_root}")
    index = scan_demo(args.demo_root)
    write_outputs(index, args.out_dir)
    n_video = sum(1 for r in index["records"] if r["sample_type"] == "video")
    n_image = sum(1 for r in index["records"] if r["sample_type"] != "video")
    print(f"Indexed {n_video} videos and {n_image} images into {args.out_dir}")


if __name__ == "__main__":
    main()

