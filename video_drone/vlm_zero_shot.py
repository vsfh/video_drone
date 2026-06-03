from __future__ import annotations

import argparse
import csv
import html
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image

try:
    from .taxonomy import EVENT_TAXONOMY, PRIMARY_EVENTS, normalize_event_name
except ImportError:  # pragma: no cover - supports direct script execution
    from taxonomy import EVENT_TAXONOMY, PRIMARY_EVENTS, normalize_event_name


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_DATA_ROOT = Path("/media/data1/feihong/video_drone_data")
DEFAULT_HF_CACHE = Path("/media/data1/feihong/hf_cache")
DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"


def primary_event_rows() -> list[dict]:
    return [row for row in EVENT_TAXONOMY if row["event"] in PRIMARY_EVENTS]


def _safe_rel_id(path: Path, root: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


def iter_image_records(data_root: Path, per_class_limit: int | None, limit: int | None) -> list[dict]:
    data_root = data_root.resolve()
    counts: dict[str, int] = {}
    records = []
    for path in sorted(data_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        rel = _safe_rel_id(path, data_root)
        folder_event = normalize_event_name(path.parent.name)
        if folder_event not in PRIMARY_EVENTS:
            continue
        if per_class_limit is not None and counts.get(folder_event, 0) >= per_class_limit:
            continue
        counts[folder_event] = counts.get(folder_event, 0) + 1
        records.append(
            {
                "sample_id": rel,
                "source_path": str(path.resolve()),
                "relative_path": rel,
                "ground_truth_event": folder_event,
                "sample_type": "image",
            }
        )
        if limit is not None and len(records) >= limit:
            break
    return records


def build_classify_prompt(event_rows: list[dict], top_k: int) -> str:
    event_lines = []
    for idx, row in enumerate(event_rows, 1):
        event_lines.append(
            "\n".join(
                [
                    f"{idx}. {row['event']}",
                    f"   定义: {row['definition']}",
                    f"   正例: {row['positive']}",
                    f"   反例: {row['negative']}",
                ]
            )
        )
    return f"""你是无人机巡检图像的城市治理事件识别助手。请只在下面 9 个事件中判断图像最可能包含哪些事件。

事件候选:
{chr(10).join(event_lines)}

要求:
- 只输出一个 JSON 对象，不要输出 Markdown、解释性前后缀或代码块。
- 如果图像没有明显属于 9 类之一的事件，has_relevant_event 设为 false，top_events 可以为空。
- score 是 0 到 1 的置信度，最多返回 {top_k} 个 top_events，按置信度从高到低排序。
- evidence 用一句中文说明关键视觉证据，caption 用一句中文概括画面。

JSON 格式:
{{
  "has_relevant_event": true,
  "top_events": [
    {{"event": "事件名", "score": 0.0}}
  ],
  "caption": "一句图像描述",
  "evidence": "一句判断依据"
}}"""


def _extract_json_object(text: str) -> dict[str, Any]:
    value = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", value, re.DOTALL | re.IGNORECASE)
    if fence:
        value = fence.group(1)
    else:
        start = value.find("{")
        end = value.rfind("}")
        if start >= 0 and end > start:
            value = value[start : end + 1]
    return json.loads(value)


def _score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = 0.0
    return max(0.0, min(score, 1.0))


def parse_model_json(text: str, event_names: list[str], top_k: int) -> dict:
    raw = text.strip()
    try:
        data = _extract_json_object(raw)
        parse_error = ""
    except Exception as exc:
        data = {}
        parse_error = str(exc)

    valid_events = set(event_names)
    top_events = data.get("top_events") or data.get("predicted_events") or []
    if isinstance(top_events, dict):
        top_events = [top_events]
    if not isinstance(top_events, list):
        top_events = []

    predicted = []
    for item in top_events:
        if not isinstance(item, dict):
            continue
        event = str(item.get("event") or "").strip()
        if event not in valid_events:
            continue
        predicted.append({"event": event, "score": round(_score(item.get("score")), 4)})
        if len(predicted) >= top_k:
            break

    if not predicted and str(data.get("event") or "").strip() in valid_events:
        predicted.append({"event": str(data["event"]).strip(), "score": round(_score(data.get("score", 0.0)), 4)})

    return {
        "has_relevant_event": bool(data.get("has_relevant_event", bool(predicted))),
        "predicted_events": predicted,
        "caption": str(data.get("caption") or "").strip(),
        "evidence": str(data.get("evidence") or "").strip(),
        "raw_response": raw,
        "parse_error": parse_error,
    }


def _model_snapshot_from_cache(model_id: str, cache_dir: Path) -> Path | None:
    candidates = [
        cache_dir / f"models--{model_id.replace('/', '--')}",
        cache_dir / "hub" / f"models--{model_id.replace('/', '--')}",
    ]
    for root in candidates:
        snapshots = root / "snapshots"
        if not snapshots.exists():
            continue
        valid = [p for p in snapshots.iterdir() if (p / "config.json").exists()]
        if valid:
            return sorted(valid)[-1]
    return None


def resolve_model_path(model: str | None, cache_dir: Path) -> Path | str:
    if model:
        path = Path(model)
        if path.exists():
            snapshots = path / "snapshots"
            if snapshots.exists():
                valid = [p for p in snapshots.iterdir() if (p / "config.json").exists()]
                if valid:
                    return sorted(valid)[-1]
            return path
        cached = _model_snapshot_from_cache(model, cache_dir)
        return cached if cached is not None else model
    cached = _model_snapshot_from_cache(DEFAULT_MODEL_ID, cache_dir)
    return cached if cached is not None else DEFAULT_MODEL_ID


def _torch_dtype(name: str):
    import torch

    if name == "auto":
        return "auto"
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _configure_generation_processor_padding(processor):
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "left"
    return processor


class QwenVLMClassifier:
    def __init__(self, model_path: Path | str, device: str, dtype: str, max_new_tokens: int) -> None:
        self.model_path = str(model_path)
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.processor = None

    def load(self) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration

        config_path = Path(self.model_path) / "config.json"
        model_type = ""
        if config_path.exists():
            model_type = json.loads(config_path.read_text(encoding="utf-8")).get("model_type", "")
        model_cls = Qwen3VLForConditionalGeneration if model_type == "qwen3_vl" else Qwen2_5_VLForConditionalGeneration
        kwargs = {
            "torch_dtype": _torch_dtype(self.dtype),
            "local_files_only": True,
        }
        if self.device == "auto":
            kwargs["device_map"] = "auto"
        self.model = model_cls.from_pretrained(self.model_path, **kwargs)
        if self.device != "auto":
            self.model.to(self.device)
        self.model.eval()
        self.processor = _configure_generation_processor_padding(AutoProcessor.from_pretrained(self.model_path, local_files_only=True))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def predict(self, image_path: Path, prompt: str) -> str:
        import torch

        if self.model is None or self.processor is None:
            self.load()
        with Image.open(image_path) as img:
            image = img.convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt")
        inputs = inputs.to(self.model.device)
        with torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        generated = generated[:, inputs.input_ids.shape[1] :]
        return self.processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def predict_records(records: list[dict], classifier: QwenVLMClassifier, prompt: str, event_names: list[str], top_k: int) -> list[dict]:
    output = []
    for idx, rec in enumerate(records, 1):
        path = Path(rec["source_path"])
        print(f"[{idx}/{len(records)}] {rec['relative_path']}")
        try:
            raw = classifier.predict(path, prompt)
            parsed = parse_model_json(raw, event_names, top_k)
            error = parsed.pop("parse_error", "")
        except Exception as exc:
            parsed = {
                "has_relevant_event": False,
                "predicted_events": [],
                "caption": "",
                "evidence": "",
                "raw_response": "",
            }
            error = str(exc)
        output.append(
            {
                **rec,
                "method": "qwen-vl-zero-shot-9way",
                **parsed,
                "error": error,
            }
        )
    return output


def write_outputs(records: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "vlm_predictions.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    (out_dir / "vlm_predictions.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    fields = [
        "sample_id",
        "ground_truth_event",
        "top1_event",
        "top1_score",
        "hit_top1",
        "has_relevant_event",
        "caption",
        "evidence",
        "source_path",
        "error",
    ]
    with (out_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in records:
            top1 = rec.get("predicted_events", [{}])[0] if rec.get("predicted_events") else {}
            writer.writerow(
                {
                    "sample_id": rec.get("sample_id", ""),
                    "ground_truth_event": rec.get("ground_truth_event", ""),
                    "top1_event": top1.get("event", ""),
                    "top1_score": top1.get("score", ""),
                    "hit_top1": top1.get("event") == rec.get("ground_truth_event"),
                    "has_relevant_event": rec.get("has_relevant_event", False),
                    "caption": rec.get("caption", ""),
                    "evidence": rec.get("evidence", ""),
                    "source_path": rec.get("source_path", ""),
                    "error": rec.get("error", ""),
                }
            )


def _thumb(src: Path, dst: Path, width: int = 260) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        img = img.convert("RGB")
        height = max(int(width * img.height / img.width), 1)
        img.resize((width, height)).save(dst, quality=90)


def _chips(predicted: list[dict]) -> str:
    return "".join(
        f"<span class='chip'>{html.escape(str(item.get('event', '')))}: {html.escape(str(item.get('score', '')))}</span>"
        for item in predicted
    )


def build_html_report(records: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    asset_dir = out_dir / "assets"
    cards = []
    hits = 0
    for rec in records:
        src = Path(rec["source_path"])
        top1 = rec.get("predicted_events", [{}])[0] if rec.get("predicted_events") else {}
        hit = top1.get("event") == rec.get("ground_truth_event")
        hits += int(hit)
        thumb_rel = f"assets/{rec['sample_id'].replace('/', '__')}"
        _thumb(src, out_dir / thumb_rel)
        status = "hit" if hit else "miss"
        cards.append(
            f"""
            <article class="card {status}">
              <img src="{html.escape(thumb_rel)}" loading="lazy" />
              <h3>{html.escape(rec.get('ground_truth_event', ''))}</h3>
              <p class="muted">{html.escape(rec.get('relative_path', ''))}</p>
              <div>{_chips(rec.get('predicted_events', []))}</div>
              <p>{html.escape(rec.get('caption', ''))}</p>
              <p class="muted">{html.escape(rec.get('evidence', ''))}</p>
              {f"<p class='error'>{html.escape(rec.get('error', ''))}</p>" if rec.get('error') else ""}
            </article>
            """
        )
    accuracy = hits / len(records) if records else 0.0
    page = f"""<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8" />
  <title>VLM Zero-Shot 9-Way Report</title>
  <style>
    body {{ font-family: Arial, "Microsoft YaHei", sans-serif; margin: 24px; background: #f5f7fb; color: #17202a; }}
    h1 {{ margin: 0 0 8px; }}
    .muted {{ color: #667085; font-size: 13px; word-break: break-all; }}
    .summary {{ margin: 0 0 18px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 16px; }}
    .card {{ background: white; border: 1px solid #d8dde6; border-left: 5px solid #98a2b3; border-radius: 8px; padding: 12px; }}
    .card.hit {{ border-left-color: #168a4a; }}
    .card.miss {{ border-left-color: #d92d20; }}
    .card img {{ width: 100%; height: auto; border-radius: 4px; border: 1px solid #e5e7eb; }}
    .chip {{ display: inline-block; padding: 3px 7px; margin: 2px; border-radius: 999px; background: #e8f0fe; color: #174ea6; font-size: 12px; }}
    .error {{ color: #b42318; font-size: 13px; }}
  </style>
</head>
<body>
  <h1>VLM Zero-Shot 9-Way Report</h1>
  <p class="summary muted">Images: {len(records)} | Top-1 hits: {hits} | Top-1 accuracy: {accuracy:.3f}</p>
  <section class="grid">{''.join(cards)}</section>
</body>
</html>"""
    asset_dir.mkdir(exist_ok=True)
    (out_dir / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-shot 9-way UAV commercial-event recognition with a local VLM.")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT, type=Path)
    parser.add_argument("--out-dir", default=Path("outputs/vlm_zero_shot"), type=Path)
    parser.add_argument("--model", default=None, help="Local model path or HF model id. Defaults to cached Qwen/Qwen3-VL-4B-Instruct.")
    parser.add_argument("--hf-cache", default=DEFAULT_HF_CACHE, type=Path)
    parser.add_argument("--limit", default=None, type=int)
    parser.add_argument("--per-class-limit", default=None, type=int)
    parser.add_argument("--top-k", default=3, type=int)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--max-new-tokens", default=512, type=int)
    args = parser.parse_args()

    event_rows = primary_event_rows()
    event_names = [row["event"] for row in event_rows]
    records = iter_image_records(args.data_root, args.per_class_limit, args.limit)
    if not records:
        raise RuntimeError(f"No primary-event images found under {args.data_root}")

    model_path = resolve_model_path(args.model, args.hf_cache)
    print(f"Model: {model_path}")
    print(f"Images: {len(records)}")
    prompt = build_classify_prompt(event_rows, args.top_k)
    classifier = QwenVLMClassifier(model_path=model_path, device=args.device, dtype=args.dtype, max_new_tokens=args.max_new_tokens)
    predictions = predict_records(records, classifier, prompt, event_names, args.top_k)
    write_outputs(predictions, args.out_dir)
    build_html_report(predictions, args.out_dir)
    print(f"Wrote predictions: {args.out_dir / 'vlm_predictions.jsonl'}")
    print(f"Wrote report: {args.out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
