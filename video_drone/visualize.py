from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw


def _load_index(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path | None) -> dict[str, dict]:
    if path is None or not path.exists():
        return {}
    rows = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                rows[row["sample_id"]] = row
    return rows


def _thumb(src: Path, dst: Path, width: int = 260) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        img = img.convert("RGB")
        h = max(int(width * img.height / img.width), 1)
        img = img.resize((width, h))
        img.save(dst, quality=90)


def _prediction_html(pred: dict | None) -> str:
    if not pred:
        return "<span class='muted'>no prediction</span>"
    chips = []
    for item in pred.get("predicted_events", []):
        chips.append(f"<span class='chip'>{html.escape(item['event'])}: {item.get('score', '')}</span>")
    caption = html.escape(pred.get("caption") or "")
    evidence = html.escape(pred.get("evidence") or "")
    return "".join(chips) + (f"<p>{caption}</p>" if caption else "") + (f"<p class='muted'>{evidence}</p>" if evidence else "")


def build_report(index: dict, predictions: dict[str, dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    asset_dir = out_dir / "assets"
    asset_dir.mkdir(exist_ok=True)

    cards = []
    image_records = [r for r in index["records"] if r["sample_type"] != "video"]
    video_records = [r for r in index["records"] if r["sample_type"] == "video"]

    for rec in image_records:
        src = Path(rec["source_path"])
        thumb_rel = f"assets/{rec['sample_id'].replace('/', '__')}.jpg"
        _thumb(src, out_dir / thumb_rel)
        pred = predictions.get(rec["sample_id"])
        cards.append(
            f"""
            <article class="card">
              <img src="{html.escape(thumb_rel)}" />
              <h3>{html.escape(rec['event'])}</h3>
              <p class="muted">{html.escape(rec['relative_path'])}</p>
              <p>type: {html.escape(rec['sample_type'])}</p>
              <div>{_prediction_html(pred)}</div>
            </article>
            """
        )

    video_rows = []
    for rec in video_records:
        video_rows.append(
            f"<tr><td>{html.escape(rec['event'])}</td><td>{html.escape(rec['relative_path'])}</td><td>{html.escape(rec['source_path'])}</td></tr>"
        )

    page = f"""<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8" />
  <title>UAV Commercial Event Demo Report</title>
  <style>
    body {{ font-family: Arial, "Microsoft YaHei", sans-serif; margin: 24px; background: #f6f7f9; color: #17202a; }}
    h1, h2 {{ margin: 0 0 16px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 16px; }}
    .card {{ background: white; border: 1px solid #d8dde6; border-radius: 8px; padding: 12px; }}
    .card img {{ width: 100%; height: auto; border-radius: 4px; border: 1px solid #e5e7eb; }}
    .muted {{ color: #667085; font-size: 13px; word-break: break-all; }}
    .chip {{ display: inline-block; padding: 3px 7px; margin: 2px; border-radius: 999px; background: #e8f0fe; color: #174ea6; font-size: 12px; }}
    table {{ width: 100%; border-collapse: collapse; background: white; margin-bottom: 24px; }}
    th, td {{ border: 1px solid #d8dde6; padding: 8px; text-align: left; }}
  </style>
</head>
<body>
  <h1>UAV Commercial Event Demo Report</h1>
  <p class="muted">Demo root: {html.escape(index.get('demo_root', ''))}</p>
  <h2>Videos</h2>
  <table><thead><tr><th>event</th><th>relative path</th><th>source path</th></tr></thead><tbody>{''.join(video_rows)}</tbody></table>
  <h2>Images and Predictions</h2>
  <section class="grid">{''.join(cards)}</section>
</body>
</html>"""
    (out_dir / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an HTML report for demo examples and method predictions.")
    parser.add_argument("--index", default="outputs/demo_index/dataset_index.json", type=Path)
    parser.add_argument("--predictions", default=None, type=Path)
    parser.add_argument("--out-dir", default="outputs/visual_report", type=Path)
    args = parser.parse_args()

    index = _load_index(args.index)
    predictions = _load_jsonl(args.predictions)
    build_report(index, predictions, args.out_dir)
    print(f"Wrote report: {args.out_dir / 'index.html'}")


if __name__ == "__main__":
    main()

