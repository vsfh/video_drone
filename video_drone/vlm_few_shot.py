from __future__ import annotations

import argparse
import csv
import html
import json
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps

try:
    from .taxonomy import PRIMARY_EVENTS, normalize_event_name
    from .vlm_zero_shot import (
        DEFAULT_DATA_ROOT,
        DEFAULT_HF_CACHE,
        IMAGE_EXTS,
        _extract_json_object,
        _model_snapshot_from_cache,
        _score,
        _torch_dtype,
        iter_image_records,
        parse_model_json,
        primary_event_rows,
        write_outputs,
    )
except ImportError:  # pragma: no cover - supports direct script execution
    from taxonomy import PRIMARY_EVENTS, normalize_event_name
    from vlm_zero_shot import (
        DEFAULT_DATA_ROOT,
        DEFAULT_HF_CACHE,
        IMAGE_EXTS,
        _extract_json_object,
        _model_snapshot_from_cache,
        _score,
        _torch_dtype,
        iter_image_records,
        parse_model_json,
        primary_event_rows,
        write_outputs,
    )


DEFAULT_EXAMPLES_ROOT = Path("/media/data1/feihong/video_drone_data/vlm_prompt_examples")
DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
DEFAULT_SHOTS_PER_CLASS = 6
DEFAULT_EXAMPLE_LAYOUT = "multi-image"
DEFAULT_CLASSIFICATION_MODE = "multiclass"
DEFAULT_MAX_IMAGE_SIDE = 512
DEFAULT_MAX_PIXELS = DEFAULT_MAX_IMAGE_SIDE * DEFAULT_MAX_IMAGE_SIDE
DEFAULT_EXAMPLE_THUMB_SIZE = 224
MODEL_ALIASES = {
    "qwen3-vl-4b": DEFAULT_MODEL_ID,
    "qwen3-vl-4b-instruct": DEFAULT_MODEL_ID,
    "qwen": DEFAULT_MODEL_ID,
    "gamma-31b": "google/gemma-4-31B-it",
    "gemma-31b": "google/gemma-4-31B-it",
    "gemma-4-31b": "google/gemma-4-31B-it",
}


@dataclass(frozen=True)
class FewShotExample:
    event: str
    path: Path
    relative_path: str


def _safe_rel_id(path: Path, root: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


def _manifest_example_paths(examples_root: Path) -> list[tuple[str, Path, str]]:
    manifest_jsonl = examples_root / "manifest.jsonl"
    rows: list[tuple[str, Path, str]] = []
    if manifest_jsonl.exists():
        with manifest_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                event = normalize_event_name(str(item.get("event") or ""))
                rel = str(item.get("relative_image_path") or "").replace("\\", "/")
                path = examples_root / rel
                if event and rel and path.exists():
                    rows.append((event, path, rel))
        return rows

    manifest_csv = examples_root / "manifest.csv"
    if manifest_csv.exists():
        with manifest_csv.open("r", encoding="utf-8-sig", newline="") as f:
            for item in csv.DictReader(f):
                event = normalize_event_name(str(item.get("event") or ""))
                rel = str(item.get("relative_image_path") or "").replace("\\", "/")
                path = examples_root / rel
                if event and rel and path.exists():
                    rows.append((event, path, rel))
    return rows


def load_few_shot_examples(examples_root: Path, event_names: list[str], shots_per_class: int) -> list[FewShotExample]:
    if shots_per_class <= 0:
        return []

    examples_root = examples_root.resolve()
    allowed = set(event_names)
    grouped: dict[str, list[FewShotExample]] = {event: [] for event in event_names}

    manifest_rows = _manifest_example_paths(examples_root)
    if manifest_rows:
        candidates = manifest_rows
    else:
        candidates = []
        for path in sorted(examples_root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
                continue
            event = normalize_event_name(path.parent.name)
            candidates.append((event, path, _safe_rel_id(path, examples_root)))

    for event, path, rel in candidates:
        if event not in allowed or path.suffix.lower() not in IMAGE_EXTS:
            continue
        bucket = grouped[event]
        if len(bucket) >= shots_per_class:
            continue
        bucket.append(FewShotExample(event=event, path=path.resolve(), relative_path=rel))

    output: list[FewShotExample] = []
    for event in event_names:
        output.extend(grouped[event])
    return output


def group_examples_by_event(examples: list[FewShotExample], event_names: list[str]) -> dict[str, list[FewShotExample]]:
    grouped: dict[str, list[FewShotExample]] = {event: [] for event in event_names}
    for example in examples:
        if example.event in grouped:
            grouped[example.event].append(example)
    return grouped


def filter_records_excluding_root(records: list[dict], excluded_root: Path) -> list[dict]:
    excluded = excluded_root.resolve()
    filtered = []
    for rec in records:
        path = Path(str(rec.get("source_path", ""))).resolve()
        try:
            path.relative_to(excluded)
        except ValueError:
            filtered.append(rec)
    return filtered


def limit_records(records: list[dict], per_class_limit: int | None, limit: int | None) -> list[dict]:
    counts: dict[str, int] = {}
    output = []
    for rec in records:
        event = str(rec.get("ground_truth_event") or "")
        if per_class_limit is not None and counts.get(event, 0) >= per_class_limit:
            continue
        counts[event] = counts.get(event, 0) + 1
        output.append(rec)
        if limit is not None and len(output) >= limit:
            break
    return output


def is_fatal_inference_error(exc: BaseException) -> bool:
    if isinstance(exc, MemoryError):
        return True
    exc_name = type(exc).__name__.lower()
    message = str(exc).lower()
    fatal_markers = (
        "out of memory",
        "cuda error: out of memory",
        "cublas_status_alloc_failed",
        "cudnn_status_alloc_failed",
        "defaultcpuallocator",
        "can't allocate memory",
    )
    return "outofmemory" in exc_name or any(marker in message for marker in fatal_markers)


def build_few_shot_prompt(
    event_rows: list[dict],
    top_k: int,
    shots_per_class: int,
    missing_example_events: list[str],
    example_layout: str = DEFAULT_EXAMPLE_LAYOUT,
) -> str:
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
    missing_note = "无"
    if missing_example_events:
        missing_note = "、".join(missing_example_events)
    if example_layout == "contact-sheet":
        example_note = "第一张图片是标准示例拼图，第二张图片是待测图像。请只判断第二张待测图像。"
    else:
        example_note = "前面给出的图片是逐张标准示例，最后一张图片是待测图像。请只判断最后一张待测图像。"
    return f"""你是无人机巡检图像的城市治理事件识别助手。输入中包含已标注的标准示例和一张待测图像。

任务:
- {example_note}
- 不要把标准示例里的事件当成待测图像结果。
- 只在下面 9 个事件中判断图像最可能包含哪些事件。
- 标准示例每类最多 {shots_per_class} 张；没有标准示例的事件仍按文字定义判断。
- 当前缺少标准示例的事件: {missing_note}

事件候选:
{chr(10).join(event_lines)}

要求:
- 只输出一个 JSON 对象，不要输出 Markdown、解释性前后缀或代码块。
- 给出全部 9 个事件的概率，写入 event_probabilities，顺序与事件候选一致。
- probability 是 0 到 1 的概率，概率总和必须为 1。
- 即使待测图像不明显属于任何一类，也必须在 9 个候选事件之间分配概率；不要输出全部 0。
- has_relevant_event 表示最高概率事件是否足够可信，但不影响 event_probabilities 必须完整给出。
- top_events 最多返回 {top_k} 个，按概率从高到低排序。
- evidence 用一句中文说明最后一张待测图像中的关键视觉证据，caption 用一句中文概括最后一张待测图像。

JSON 格式:
{{
  "has_relevant_event": true,
  "event_probabilities": [
    {{"event": "事件名", "probability": 0.0}}
  ],
  "top_events": [
    {{"event": "事件名", "score": 0.0}}
  ],
  "caption": "一句图像描述",
  "evidence": "一句判断依据"
}}"""


def build_one_vs_rest_prompt(event_row: dict, shots_for_event: int) -> str:
    event = event_row["event"]
    return f"""你是无人机巡检图像的城市治理事件识别助手。输入中前面的图片是“{event}”的已标注标准示例，最后一张图片是待测图像。

任务:
- 只判断最后一张待测图像是否包含“{event}”这一种事件。
- 当前事件标准示例数量: {shots_for_event}。如果没有标准示例，请严格按文字定义判断。
- 不要把标准示例里的内容当成待测图像结果。

事件定义:
- 事件: {event}
- 定义: {event_row['definition']}
- 正例: {event_row['positive']}
- 反例: {event_row['negative']}

要求:
- 只输出一个 JSON 对象，不要输出 Markdown、解释性前后缀或代码块。
- has_event 表示待测图像是否包含“{event}”。
- score 是 0 到 1 的置信度。
- evidence 用一句中文说明最后一张待测图像中的关键视觉证据，caption 用一句中文概括最后一张待测图像。

JSON 格式:
{{
  "has_event": true,
  "event": "{event}",
  "score": 0.0,
  "caption": "一句图像描述",
  "evidence": "一句判断依据"
}}"""


def parse_multiclass_json(text: str, event_names: list[str], top_k: int) -> dict:
    raw = text.strip()
    try:
        data = _extract_json_object(raw)
        parse_error = ""
    except Exception as exc:
        data = {}
        parse_error = str(exc)

    valid_events = set(event_names)
    scores: dict[str, float] = {event: 0.0 for event in event_names}
    prob_data = (
        data.get("event_probabilities")
        or data.get("probabilities")
        or data.get("class_probabilities")
        or data.get("scores")
        or []
    )
    if isinstance(prob_data, dict):
        for event, value in prob_data.items():
            event_name = str(event).strip()
            if event_name in valid_events:
                scores[event_name] = _score(value)
    elif isinstance(prob_data, list):
        for item in prob_data:
            if not isinstance(item, dict):
                continue
            event_name = str(item.get("event") or item.get("label") or item.get("class") or "").strip()
            if event_name not in valid_events:
                continue
            value = item.get("probability", item.get("prob", item.get("score", 0.0)))
            scores[event_name] = _score(value)

    top_events = data.get("top_events") or data.get("predicted_events") or []
    if isinstance(top_events, dict):
        top_events = [top_events]
    if isinstance(top_events, list):
        for item in top_events:
            if not isinstance(item, dict):
                continue
            event_name = str(item.get("event") or "").strip()
            if event_name in valid_events and scores[event_name] == 0.0:
                scores[event_name] = _score(item.get("score", item.get("probability", 0.0)))

    event_probabilities = [
        {"event": event, "probability": round(scores[event], 4)}
        for event in event_names
    ]
    ranked = sorted(event_probabilities, key=lambda item: item["probability"], reverse=True)
    predicted = [
        {"event": item["event"], "score": item["probability"]}
        for item in ranked
        if item["probability"] > 0.0
    ][:top_k]

    has_relevant = data.get("has_relevant_event")
    if not isinstance(has_relevant, bool):
        has_relevant = bool(predicted)

    return {
        "has_relevant_event": bool(has_relevant),
        "predicted_events": predicted,
        "event_probabilities": event_probabilities,
        "caption": str(data.get("caption") or "").strip(),
        "evidence": str(data.get("evidence") or "").strip(),
        "raw_response": raw,
        "parse_error": parse_error,
    }


def parse_one_vs_rest_json(text: str, event_name: str) -> dict:
    raw = text.strip()
    try:
        data = _extract_json_object(raw)
        parse_error = ""
    except Exception as exc:
        data = {}
        parse_error = str(exc)

    has_event = data.get("has_event")
    if not isinstance(has_event, bool):
        has_event = data.get("has_relevant_event")
    if not isinstance(has_event, bool):
        has_event = False

    score = data.get("score", 0.0)
    top_events = data.get("top_events") or data.get("predicted_events") or []
    if isinstance(top_events, dict):
        top_events = [top_events]
    if isinstance(top_events, list):
        for item in top_events:
            if isinstance(item, dict) and str(item.get("event") or "").strip() == event_name:
                score = item.get("score", score)
                has_event = True
                break

    if str(data.get("event") or "").strip() == event_name and "has_event" not in data:
        has_event = True

    return {
        "event": event_name,
        "has_event": bool(has_event),
        "score": round(_score(score), 4),
        "caption": str(data.get("caption") or "").strip(),
        "evidence": str(data.get("evidence") or "").strip(),
        "raw_response": raw,
        "parse_error": parse_error,
    }


def build_few_shot_messages(examples: list[FewShotExample], target_image: Any, prompt: str) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": "下面是已标注的标准示例图片。每张示例后面的文字给出它所属的事件类别。",
        }
    ]
    for idx, example in enumerate(examples, 1):
        content.append({"type": "image", "image": example.path})
        content.append(
            {
                "type": "text",
                "text": f"标准示例 {idx}: 事件类别 = {example.event}。示例来源: {example.relative_path}",
            }
        )
    content.append({"type": "image", "image": target_image})
    content.append({"type": "text", "text": f"最后一张待测图像需要分类。{prompt}"})
    return [{"role": "user", "content": content}]


def _local_image_ref(path: Path) -> str:
    return str(path.resolve())


def build_qwen_standard_messages(
    examples: list[FewShotExample],
    target_image_path: Path,
    prompt: str,
    max_pixels: int | None,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": "下面是已标注的标准示例图片。每张示例后面的文字给出它所属的事件类别。",
        }
    ]
    for idx, example in enumerate(examples, 1):
        item: dict[str, Any] = {"type": "image", "image": _local_image_ref(example.path)}
        if max_pixels is not None and max_pixels > 0:
            item["max_pixels"] = max_pixels
        content.append(item)
        content.append(
            {
                "type": "text",
                "text": f"标准示例 {idx}: 事件类别 = {example.event}。示例来源: {example.relative_path}",
            }
        )
    target_item: dict[str, Any] = {"type": "image", "image": _local_image_ref(target_image_path)}
    if max_pixels is not None and max_pixels > 0:
        target_item["max_pixels"] = max_pixels
    content.append(target_item)
    content.append({"type": "text", "text": f"最后一张待测图像需要分类。{prompt}"})
    return [{"role": "user", "content": content}]


def _resize_max_side(image: Image.Image, max_side: int | None) -> Image.Image:
    if max_side is None or max_side <= 0:
        return image.copy()
    width, height = image.size
    side = max(width, height)
    if side <= max_side:
        return image.copy()
    scale = max_side / side
    size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def build_contact_sheet(examples: list[FewShotExample], thumb_size: int, cols: int = 3) -> tuple[Image.Image, str]:
    if not examples:
        raise ValueError("Cannot build a contact sheet without examples.")
    cols = max(cols, 1)
    rows = (len(examples) + cols - 1) // cols
    label_h = 26
    pad = 10
    cell_w = thumb_size + pad * 2
    cell_h = thumb_size + label_h + pad * 2
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    event_codes: dict[str, str] = {}
    legend_lines = []

    for idx, example in enumerate(examples):
        if example.event not in event_codes:
            event_codes[example.event] = f"E{len(event_codes) + 1:02d}"
            legend_lines.append(f"{event_codes[example.event]} = {example.event}")
        code = f"{event_codes[example.event]}-{sum(1 for prev in examples[:idx + 1] if prev.event == example.event)}"
        col = idx % cols
        row = idx // cols
        x = col * cell_w + pad
        y = row * cell_h + pad
        with Image.open(example.path) as img:
            thumb = ImageOps.contain(img.convert("RGB"), (thumb_size, thumb_size), Image.Resampling.LANCZOS)
        bg = Image.new("RGB", (thumb_size, thumb_size), (245, 247, 250))
        bg.paste(thumb, ((thumb_size - thumb.width) // 2, (thumb_size - thumb.height) // 2))
        sheet.paste(bg, (x, y + label_h))
        draw.rectangle((x, y, x + thumb_size, y + label_h - 1), fill=(30, 64, 175))
        draw.text((x + 6, y + 6), code, fill="white")
        draw.rectangle((x, y + label_h, x + thumb_size, y + label_h + thumb_size), outline=(210, 214, 220))

    return sheet, "\n".join(legend_lines)


def build_contact_sheet_messages(contact_sheet: Any, target_image: Any, legend: str, prompt: str) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": "第一张图片是已标注标准示例拼成的图谱。每个格子顶部有 E编号-序号，对应关系见后面的文字图例。",
        },
        {"type": "image", "image": contact_sheet},
        {"type": "text", "text": f"标准示例图例:\n{legend}"},
        {"type": "image", "image": target_image},
        {"type": "text", "text": f"第二张待测图像需要分类，只判断第二张图像。{prompt}"},
    ]
    return [{"role": "user", "content": content}]


def resolve_model_path(model: str | None, cache_dir: Path) -> Path | str:
    model_id = MODEL_ALIASES.get(model or "", model or DEFAULT_MODEL_ID)
    path = Path(model_id)
    if path.exists():
        snapshots = path / "snapshots"
        if snapshots.exists():
            valid = [p for p in snapshots.iterdir() if (p / "config.json").exists()]
            if valid:
                return sorted(valid)[-1]
        return path
    cached = _model_snapshot_from_cache(model_id, cache_dir)
    return cached if cached is not None else model_id


class FewShotVLMClassifier:
    def __init__(
        self,
        model_path: Path | str,
        device: str,
        dtype: str,
        max_new_tokens: int,
        example_layout: str = DEFAULT_EXAMPLE_LAYOUT,
        max_image_side: int = DEFAULT_MAX_IMAGE_SIDE,
        max_pixels: int = DEFAULT_MAX_PIXELS,
        example_thumb_size: int = DEFAULT_EXAMPLE_THUMB_SIZE,
    ) -> None:
        self.model_path = str(model_path)
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.example_layout = example_layout
        self.max_image_side = max_image_side
        self.max_pixels = max_pixels
        self.example_thumb_size = example_thumb_size
        self.model = None
        self.processor = None
        self.model_type = ""

    def load(self) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor, Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration

        config_path = Path(self.model_path) / "config.json"
        model_type = ""
        if config_path.exists():
            model_type = json.loads(config_path.read_text(encoding="utf-8")).get("model_type", "")
        self.model_type = model_type
        if model_type == "qwen3_vl":
            model_cls = Qwen3VLForConditionalGeneration
        elif model_type == "qwen2_5_vl":
            model_cls = Qwen2_5_VLForConditionalGeneration
        else:
            model_cls = AutoModelForImageTextToText

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
        self.processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _predict_qwen_standard(self, image_path: Path, examples: list[FewShotExample], prompt: str) -> str:
        import torch

        messages = build_qwen_standard_messages(examples, image_path, prompt, self.max_pixels)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        try:
            from qwen_vl_utils import process_vision_info

            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        except ModuleNotFoundError:
            images = []
            for example in examples:
                with Image.open(example.path) as img:
                    images.append(_resize_max_side(img.convert("RGB"), self.max_image_side))
            with Image.open(image_path) as img:
                images.append(_resize_max_side(img.convert("RGB"), self.max_image_side))
            inputs = self.processor(text=[text], images=images, return_tensors="pt")

        inputs = inputs.to(self.model.device)
        with torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        generated = generated[:, inputs.input_ids.shape[1] :]
        return self.processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    def predict(self, image_path: Path, examples: list[FewShotExample], prompt: str) -> str:
        import torch

        if self.model is None or self.processor is None:
            self.load()

        if self.model_type.startswith("qwen") and self.example_layout == "multi-image":
            return self._predict_qwen_standard(image_path, examples, prompt)

        with ExitStack() as stack:
            target_image = _resize_max_side(stack.enter_context(Image.open(image_path)).convert("RGB"), self.max_image_side)
            if self.example_layout == "contact-sheet":
                contact_sheet, legend = build_contact_sheet(examples, thumb_size=self.example_thumb_size)
                contact_sheet = _resize_max_side(contact_sheet, self.max_image_side)
                images = [contact_sheet, target_image]
                messages = build_contact_sheet_messages(contact_sheet, target_image, legend, prompt)
            else:
                opened_examples = []
                for example in examples:
                    opened_examples.append(_resize_max_side(stack.enter_context(Image.open(example.path)).convert("RGB"), self.max_image_side))
                images = opened_examples + [target_image]
                image_iter = iter(images)
                messages = build_few_shot_messages(examples, target_image, prompt)
                for item in messages[0]["content"]:
                    if item.get("type") == "image":
                        item["image"] = next(image_iter)

            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=images, return_tensors="pt")
            inputs = inputs.to(self.model.device)
            with torch.inference_mode():
                generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
            generated = generated[:, inputs.input_ids.shape[1] :]
            return self.processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def predict_records(
    records: list[dict],
    classifier: FewShotVLMClassifier,
    examples: list[FewShotExample],
    prompt: str,
    event_names: list[str],
    top_k: int,
) -> list[dict]:
    output = []
    for idx, rec in enumerate(records, 1):
        path = Path(rec["source_path"])
        print(f"[{idx}/{len(records)}] {rec['relative_path']}")
        try:
            raw = classifier.predict(path, examples, prompt)
            parsed = parse_multiclass_json(raw, event_names, top_k)
            error = parsed.pop("parse_error", "")
        except Exception as exc:
            if is_fatal_inference_error(exc):
                raise RuntimeError(f"Fatal inference error at {rec['relative_path']}: {exc}") from exc
            parsed = {
                "has_relevant_event": False,
                "predicted_events": [],
                "event_probabilities": [{"event": event, "probability": 0.0} for event in event_names],
                "caption": "",
                "evidence": "",
                "raw_response": "",
            }
            error = str(exc)
        output.append(
            {
                **rec,
                "method": "qwen-vl-few-shot-multiclass-9way",
                "few_shot_examples": [ex.relative_path for ex in examples],
                **parsed,
                "error": error,
            }
        )
    return output


def predict_records_one_vs_rest(
    records: list[dict],
    classifier: FewShotVLMClassifier,
    examples_by_event: dict[str, list[FewShotExample]],
    event_rows: list[dict],
    top_k: int,
) -> list[dict]:
    output = []
    for idx, rec in enumerate(records, 1):
        path = Path(rec["source_path"])
        print(f"[{idx}/{len(records)}] {rec['relative_path']}")
        checks = []
        errors = []
        captions = []
        positive_events = []
        for row in event_rows:
            event = row["event"]
            event_examples = examples_by_event.get(event, [])
            prompt = build_one_vs_rest_prompt(row, len(event_examples))
            try:
                raw = classifier.predict(path, event_examples, prompt)
                parsed = parse_one_vs_rest_json(raw, event)
                error = parsed.pop("parse_error", "")
            except Exception as exc:
                if is_fatal_inference_error(exc):
                    raise RuntimeError(f"Fatal inference error at {rec['relative_path']} while checking {event}: {exc}") from exc
                parsed = {
                    "event": event,
                    "has_event": False,
                    "score": 0.0,
                    "caption": "",
                    "evidence": "",
                    "raw_response": "",
                }
                error = str(exc)
            check = {**parsed, "example_count": len(event_examples), "error": error}
            checks.append(check)
            if error:
                errors.append(f"{event}: {error}")
            if check.get("caption"):
                captions.append(str(check["caption"]))
            if check.get("has_event"):
                positive_events.append({"event": event, "score": round(float(check.get("score", 0.0)), 4)})

        positive_events.sort(key=lambda item: item["score"], reverse=True)
        predicted_events = positive_events[:top_k]
        top_event_name = predicted_events[0]["event"] if predicted_events else ""
        top_check = next((check for check in checks if check["event"] == top_event_name), {})
        output.append(
            {
                **rec,
                "method": "qwen-vl-few-shot-one-vs-rest-9way",
                "few_shot_examples": {
                    event: [ex.relative_path for ex in examples_by_event.get(event, [])]
                    for event in [row["event"] for row in event_rows]
                },
                "has_relevant_event": bool(predicted_events),
                "predicted_events": predicted_events,
                "caption": str(top_check.get("caption") or (captions[0] if captions else "")),
                "evidence": str(top_check.get("evidence") or ""),
                "raw_response": json.dumps(checks, ensure_ascii=False),
                "one_vs_rest_checks": checks,
                "error": " | ".join(errors),
            }
        )
    return output


def _probability_map(rec: dict) -> dict[str, Any]:
    values = rec.get("event_probabilities") or []
    if isinstance(values, dict):
        return values
    output = {}
    if isinstance(values, list):
        for item in values:
            if isinstance(item, dict):
                event = str(item.get("event") or "").strip()
                if event:
                    output[event] = item.get("probability", item.get("score", ""))
    return output


def write_outputs(records: list[dict], out_dir: Path, event_names: list[str] | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "vlm_predictions.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    (out_dir / "vlm_predictions.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    if event_names is None:
        seen = []
        for rec in records:
            for item in rec.get("event_probabilities") or []:
                if isinstance(item, dict):
                    event = str(item.get("event") or "").strip()
                    if event and event not in seen:
                        seen.append(event)
        event_names = seen

    probability_fields = [f"prob_{event}" for event in event_names]
    fields = [
        "sample_id",
        "ground_truth_event",
        "top1_event",
        "top1_score",
        "hit_top1",
        "has_relevant_event",
        *probability_fields,
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
            prob_map = _probability_map(rec)
            row = {
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
            for event in event_names:
                row[f"prob_{event}"] = prob_map.get(event, "")
            writer.writerow(row)


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


def build_html_report(records: list[dict], out_dir: Path, examples: list[FewShotExample]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
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
    example_summary = ", ".join(f"{event}: {count}" for event, count in _example_counts(examples).items())
    page = f"""<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8" />
  <title>VLM Few-Shot 9-Way Report</title>
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
  <h1>VLM Few-Shot 9-Way Report</h1>
  <p class="summary muted">Images: {len(records)} | Top-1 hits: {hits} | Top-1 accuracy: {accuracy:.3f}</p>
  <p class="summary muted">Few-shot examples: {html.escape(example_summary)}</p>
  <section class="grid">{''.join(cards)}</section>
</body>
</html>"""
    (out_dir / "assets").mkdir(exist_ok=True)
    (out_dir / "index.html").write_text(page, encoding="utf-8")


def _example_counts(examples: list[FewShotExample]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for example in examples:
        counts[example.event] = counts.get(example.event, 0) + 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Few-shot 9-way UAV commercial-event recognition with a local VLM.")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT, type=Path)
    parser.add_argument("--examples-root", default=DEFAULT_EXAMPLES_ROOT, type=Path)
    parser.add_argument("--out-dir", default=Path("outputs/vlm_few_shot"), type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="Local model path or HF model id. Default is cached Qwen/Qwen3-VL-4B-Instruct.")
    parser.add_argument("--hf-cache", default=DEFAULT_HF_CACHE, type=Path)
    parser.add_argument("--limit", default=None, type=int)
    parser.add_argument("--per-class-limit", default=None, type=int)
    parser.add_argument("--shots-per-class", default=DEFAULT_SHOTS_PER_CLASS, type=int)
    parser.add_argument("--classification-mode", default=DEFAULT_CLASSIFICATION_MODE, choices=["one-vs-rest", "multiclass"])
    parser.add_argument("--example-layout", default=DEFAULT_EXAMPLE_LAYOUT, choices=["contact-sheet", "multi-image"])
    parser.add_argument("--max-image-side", default=DEFAULT_MAX_IMAGE_SIDE, type=int, help="Resize every image so its long side is at most this many pixels. Use 0 to disable.")
    parser.add_argument("--max-pixels", default=DEFAULT_MAX_PIXELS, type=int, help="Qwen standard image max_pixels per image. Use 0 to omit this field.")
    parser.add_argument("--example-thumb-size", default=DEFAULT_EXAMPLE_THUMB_SIZE, type=int, help="Contact-sheet thumbnail size in pixels.")
    parser.add_argument("--include-examples-in-data", action="store_true", help="Do not exclude examples-root images from evaluation records.")
    parser.add_argument("--top-k", default=3, type=int)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--max-new-tokens", default=512, type=int)
    args = parser.parse_args()

    event_rows = primary_event_rows()
    event_names = [row["event"] for row in event_rows]
    examples = load_few_shot_examples(args.examples_root, event_names, args.shots_per_class)
    example_counts = _example_counts(examples)
    missing_example_events = [event for event in event_names if example_counts.get(event, 0) == 0]
    records = iter_image_records(args.data_root, per_class_limit=None, limit=None)
    before_filter_count = len(records)
    if not args.include_examples_in_data:
        records = filter_records_excluding_root(records, args.examples_root)
    records = limit_records(records, args.per_class_limit, args.limit)
    if not records:
        raise RuntimeError(f"No primary-event images found under {args.data_root}")
    if not examples:
        raise RuntimeError(f"No few-shot examples found under {args.examples_root} for primary events: {', '.join(event_names)}")

    model_path = resolve_model_path(args.model, args.hf_cache)
    print(f"Model: {model_path}")
    print(f"Images: {len(records)}")
    print(f"Classification mode: {args.classification_mode}")
    print(f"Example layout: {args.example_layout}, max image side: {args.max_image_side}, max pixels: {args.max_pixels}, example thumb size: {args.example_thumb_size}")
    if len(records) != before_filter_count:
        print(f"Excluded prompt-example images from evaluation: {before_filter_count - len(records)}")
    print(f"Few-shot examples: {sum(example_counts.values())} ({example_counts})")
    if missing_example_events:
        print(f"Missing example events: {', '.join(missing_example_events)}")
    classifier = FewShotVLMClassifier(
        model_path=model_path,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        example_layout=args.example_layout,
        max_image_side=args.max_image_side,
        max_pixels=args.max_pixels,
        example_thumb_size=args.example_thumb_size,
    )
    if args.classification_mode == "one-vs-rest":
        predictions = predict_records_one_vs_rest(
            records=records,
            classifier=classifier,
            examples_by_event=group_examples_by_event(examples, event_names),
            event_rows=event_rows,
            top_k=args.top_k,
        )
    else:
        prompt = build_few_shot_prompt(event_rows, args.top_k, args.shots_per_class, missing_example_events, args.example_layout)
        predictions = predict_records(records, classifier, examples, prompt, event_names, args.top_k)
    write_outputs(predictions, args.out_dir, event_names=event_names)
    build_html_report(predictions, args.out_dir, examples)
    print(f"Wrote predictions: {args.out_dir / 'vlm_predictions.jsonl'}")
    print(f"Wrote report: {args.out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
