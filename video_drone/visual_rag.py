from __future__ import annotations

import argparse
import csv
import html
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image

try:
    from .taxonomy import EVENT_TAXONOMY, PRIMARY_EVENTS, event_lookup, normalize_event_name
    from .vlm_few_shot import DEFAULT_MODEL_ID as DEFAULT_VERIFY_MODEL_ID
    from .vlm_few_shot import is_fatal_inference_error, resolve_model_path
    from .vlm_zero_shot import DEFAULT_DATA_ROOT, DEFAULT_HF_CACHE, IMAGE_EXTS, _extract_json_object, _model_snapshot_from_cache, _score
except ImportError:  # pragma: no cover - supports direct script execution
    from taxonomy import EVENT_TAXONOMY, PRIMARY_EVENTS, event_lookup, normalize_event_name
    from vlm_few_shot import DEFAULT_MODEL_ID as DEFAULT_VERIFY_MODEL_ID
    from vlm_few_shot import is_fatal_inference_error, resolve_model_path
    from vlm_zero_shot import DEFAULT_DATA_ROOT, DEFAULT_HF_CACHE, IMAGE_EXTS, _extract_json_object, _model_snapshot_from_cache, _score


DEFAULT_EXAMPLES_ROOT = DEFAULT_DATA_ROOT / "vlm_prompt_examples"
DEFAULT_EMBEDDING_MODEL_ID = "openai/clip-vit-base-patch32"
DEFAULT_MODE = "retrieve-only"
DEFAULT_TOP_EXAMPLES = 24
DEFAULT_CANDIDATE_EVENTS = 5
DEFAULT_EXAMPLES_PER_EVENT = 3
DEFAULT_TOP_K = 5
DEFAULT_THRESHOLD = 0.35
DEFAULT_MAX_PIXELS = 512 * 512


@dataclass(frozen=True)
class EvidenceExample:
    event: str
    path: Path
    relative_path: str
    crop_path: Path | None = None
    crop_relative_path: str = ""
    red_box_found: bool = False


class ImageEmbedder(Protocol):
    def embed_image_paths(self, paths: list[Path]) -> np.ndarray:
        ...


def _safe_rel_id(path: Path, root: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


def evidence_image_path(example: EvidenceExample) -> Path:
    return example.crop_path if example.crop_path is not None else example.path


def evidence_relative_path(example: EvidenceExample) -> str:
    return example.crop_relative_path or example.relative_path


def _red_pixel_mask(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]
    return (red >= 150) & (green <= 110) & (blue <= 110) & ((red - green) >= 60) & ((red - blue) >= 60)


def _component_bboxes(mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    height, width = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    bboxes: list[tuple[int, int, int, int, int]] = []
    ys, xs = np.nonzero(mask)
    for seed_y, seed_x in zip(ys.tolist(), xs.tolist()):
        if visited[seed_y, seed_x]:
            continue
        stack = [(seed_x, seed_y)]
        visited[seed_y, seed_x] = True
        min_x = max_x = seed_x
        min_y = max_y = seed_y
        count = 0
        while stack:
            x, y = stack.pop()
            count += 1
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if nx < 0 or ny < 0 or nx >= width or ny >= height:
                    continue
                if visited[ny, nx] or not mask[ny, nx]:
                    continue
                visited[ny, nx] = True
                stack.append((nx, ny))
        bboxes.append((min_x, min_y, max_x, max_y, count))
    return bboxes


def detect_red_box_bbox(image_path: Path, min_red_pixels: int = 20) -> tuple[int, int, int, int] | None:
    with Image.open(image_path) as image:
        mask = _red_pixel_mask(image)
    candidates = []
    for min_x, min_y, max_x, max_y, count in _component_bboxes(mask):
        box_w = max_x - min_x + 1
        box_h = max_y - min_y + 1
        if count < min_red_pixels or box_w < 4 or box_h < 4:
            continue
        candidates.append((min_x, min_y, max_x, max_y, count))
    if not candidates:
        return None
    min_x, min_y, max_x, max_y, _count = max(candidates, key=lambda item: ((item[2] - item[0] + 1) * (item[3] - item[1] + 1), item[4]))
    return (min_x, min_y, max_x, max_y)


def _crop_inner_red_box(src: Path, dst: Path, bbox: tuple[int, int, int, int], inset: int) -> None:
    left, top, right, bottom = bbox
    crop_box = (left + inset, top + inset, right - inset + 1, bottom - inset + 1)
    if crop_box[0] >= crop_box[2] or crop_box[1] >= crop_box[3]:
        crop_box = (left, top, right + 1, bottom + 1)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as image:
        image.convert("RGB").crop(crop_box).save(dst)


def prepare_evidence_examples(
    examples: list[EvidenceExample],
    crop_root: Path,
    red_box_mode: str = "auto",
    red_box_inset: int = 2,
) -> list[EvidenceExample]:
    if red_box_mode == "off":
        return examples
    if red_box_mode not in {"auto", "require"}:
        raise ValueError(f"Unsupported red_box_mode: {red_box_mode}")

    prepared: list[EvidenceExample] = []
    crop_root = crop_root.resolve()
    for example in examples:
        bbox = detect_red_box_bbox(example.path)
        if bbox is None:
            if red_box_mode == "auto":
                prepared.append(example)
            continue
        rel_base = Path(example.relative_path)
        crop_rel = str(rel_base.with_name(f"{rel_base.stem}__redbox.png")).replace("\\", "/")
        crop_path = crop_root / crop_rel
        _crop_inner_red_box(example.path, crop_path, bbox, red_box_inset)
        prepared.append(
            replace(
                example,
                crop_path=crop_path,
                crop_relative_path=f"redbox_crops/{crop_rel}",
                red_box_found=True,
            )
        )
    return prepared


def _manifest_example_paths(examples_root: Path) -> list[tuple[str, Path, str]]:
    rows: list[tuple[str, Path, str]] = []
    manifest_jsonl = examples_root / "manifest.jsonl"
    if manifest_jsonl.exists():
        with manifest_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                event = normalize_event_name(str(item.get("event") or ""))
                rel = str(item.get("relative_image_path") or item.get("path") or "").replace("\\", "/")
                path = examples_root / rel
                if event and rel and path.exists():
                    rows.append((event, path, rel))
        return rows

    manifest_csv = examples_root / "manifest.csv"
    if manifest_csv.exists():
        with manifest_csv.open("r", encoding="utf-8-sig", newline="") as f:
            for item in csv.DictReader(f):
                event = normalize_event_name(str(item.get("event") or ""))
                rel = str(item.get("relative_image_path") or item.get("path") or "").replace("\\", "/")
                path = examples_root / rel
                if event and rel and path.exists():
                    rows.append((event, path, rel))
        return rows

    for path in sorted(examples_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        event = normalize_event_name(path.parent.name)
        rows.append((event, path, _safe_rel_id(path, examples_root)))
    return rows


def load_evidence_examples(
    examples_root: Path,
    event_names: list[str] | None,
    examples_per_event: int | None,
) -> list[EvidenceExample]:
    examples_root = examples_root.resolve()
    allowed = {normalize_event_name(event) for event in event_names} if event_names is not None else None
    counts: dict[str, int] = {}
    output: list[EvidenceExample] = []

    for event, path, rel in _manifest_example_paths(examples_root):
        if not event or path.suffix.lower() not in IMAGE_EXTS:
            continue
        if allowed is not None and event not in allowed:
            continue
        if examples_per_event is not None and counts.get(event, 0) >= examples_per_event:
            continue
        counts[event] = counts.get(event, 0) + 1
        output.append(EvidenceExample(event=event, path=path.resolve(), relative_path=rel))
    return output


def unique_events_from_examples(examples: list[EvidenceExample]) -> list[str]:
    events: list[str] = []
    seen: set[str] = set()
    for example in examples:
        if example.event not in seen:
            seen.add(example.event)
            events.append(example.event)
    return events


def event_rows_for_names(event_names: list[str]) -> list[dict]:
    lookup = event_lookup()
    rows = []
    for event in event_names:
        if event in lookup:
            rows.append(lookup[event])
        else:
            rows.append(
                {
                    "event": event,
                    "domain": "",
                    "definition": "由示例图片定义的可扩展事件类别。",
                    "positive": "与检索到的标准示例在主体、场景或状态上高度相似。",
                    "negative": "与标准示例缺少关键视觉相似性，或只存在背景相似。",
                }
            )
    return rows


def events_for_scope(scope: str, examples_root: Path) -> list[str] | None:
    if scope == "examples":
        return None
    if scope == "primary":
        return [row["event"] for row in EVENT_TAXONOMY if row["event"] in PRIMARY_EVENTS]
    if scope == "taxonomy":
        return [row["event"] for row in EVENT_TAXONOMY]
    raise ValueError(f"Unsupported event scope: {scope}")


def filter_records_excluding_root(records: list[dict], excluded_root: Path) -> list[dict]:
    excluded = excluded_root.resolve()
    output = []
    for rec in records:
        path = Path(str(rec.get("source_path", ""))).resolve()
        try:
            path.relative_to(excluded)
        except ValueError:
            output.append(rec)
    return output


def iter_image_records_for_events(
    data_root: Path,
    event_names: list[str],
    excluded_root: Path | None,
    per_class_limit: int | None,
    limit: int | None,
) -> list[dict]:
    data_root = data_root.resolve()
    allowed = set(event_names)
    counts: dict[str, int] = {}
    records: list[dict] = []
    for path in sorted(data_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        if excluded_root is not None:
            try:
                path.resolve().relative_to(excluded_root.resolve())
                continue
            except ValueError:
                pass
        event = normalize_event_name(path.parent.name)
        if event not in allowed:
            continue
        if per_class_limit is not None and counts.get(event, 0) >= per_class_limit:
            continue
        counts[event] = counts.get(event, 0) + 1
        rel = _safe_rel_id(path, data_root)
        records.append(
            {
                "sample_id": rel,
                "source_path": str(path.resolve()),
                "relative_path": rel,
                "ground_truth_event": event,
                "sample_type": "image",
            }
        )
        if limit is not None and len(records) >= limit:
            break
    return records


def _normalize_matrix(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def independent_similarity_score(cosine: float) -> float:
    return round(max(0.0, min((float(cosine) + 1.0) / 2.0, 1.0)), 4)


def rank_event_candidates(
    query_embedding: np.ndarray,
    evidence_embeddings: np.ndarray,
    examples: list[EvidenceExample],
    top_examples: int,
    top_events: int,
    examples_per_event: int,
) -> list[dict]:
    if len(examples) == 0:
        return []
    evidence = _normalize_matrix(evidence_embeddings)
    query = _normalize_matrix(query_embedding)[0]
    similarities = evidence @ query
    ranked_indices = np.argsort(-similarities)[: max(top_examples, top_events)]

    by_event: dict[str, dict] = {}
    for idx in ranked_indices:
        example = examples[int(idx)]
        score = independent_similarity_score(float(similarities[int(idx)]))
        bucket = by_event.setdefault(example.event, {"event": example.event, "score": 0.0, "examples": []})
        bucket["score"] = max(float(bucket["score"]), score)
        if len(bucket["examples"]) < examples_per_event:
            bucket["examples"].append(
                {
                    "relative_path": evidence_relative_path(example),
                    "source_path": str(evidence_image_path(example)),
                    "original_relative_path": example.relative_path,
                    "original_source_path": str(example.path),
                    "red_box_found": example.red_box_found,
                    "score": score,
                }
            )

    candidates = list(by_event.values())
    candidates.sort(key=lambda item: (-float(item["score"]), item["event"]))
    return candidates[:top_events]


def build_visual_rag_prompt(event_rows: list[dict], candidates: list[dict], top_k: int) -> str:
    row_by_event = {row["event"]: row for row in event_rows}
    candidate_lines = []
    for idx, candidate in enumerate(candidates, 1):
        event = candidate["event"]
        row = row_by_event.get(event) or event_rows_for_names([event])[0]
        example_paths = "、".join(str(item.get("relative_path", "")) for item in candidate.get("examples", []))
        candidate_lines.append(
            "\n".join(
                [
                    f"{idx}. {event}",
                    f"   检索相似度: {candidate.get('score', 0.0)}",
                    f"   定义: {row.get('definition', '')}",
                    f"   正例: {row.get('positive', '')}",
                    f"   反例: {row.get('negative', '')}",
                    f"   检索示例: {example_paths}",
                ]
            )
        )
    return f"""你是无人机巡检图像的城市治理事件识别助手。输入中先给出若干张从视觉证据库检索到的标准示例，最后一张是待测图像。

任务:
- 只判断最后一张待测图像，不要把示例图中的事件当作待测结果。
- 如果示例图来自红框裁剪，裁剪图就是原示意图红框内部的关键 object / region；如果仍是原图，也只有红框内部代表事件，红框外背景不是事件证据。
- 候选事件来自视觉 RAG 检索，可能不是固定 9 类，未来可以继续扩展。
- 每个事件的 score 是独立相关性，范围 0 到 1，不要求总和为 1，也不要为了归一化压低其他事件。
- 如果多个事件都明显成立，可以同时给较高分；如果都不成立，可以全部给低分。

候选事件与检索证据:
{chr(10).join(candidate_lines)}

要求:
- 只输出一个 JSON 对象，不要输出 Markdown、解释性前后缀或代码块。
- events 最多返回 {top_k} 个，按独立相关性 score 从高到低排序。
- evidence 用一句中文说明最后一张待测图像中支持该事件的关键视觉证据。
- 不要因为背景、拍摄角度、农田、水面、道路、屋顶颜色相似就输出事件；必须能在待测图中看到与示例红框内部对应的关键 object / region。
- caption 用一句中文概括最后一张待测图像。

JSON 格式:
{{
  "has_relevant_event": true,
  "events": [
    {{"event": "事件名", "score": 0.0, "evidence": "一句证据"}}
  ],
  "caption": "一句图像描述"
}}"""


def parse_visual_rag_json(text: str, candidate_events: list[str], top_k: int) -> dict:
    raw = text.strip()
    try:
        data = _extract_json_object(raw)
        parse_error = ""
    except Exception as exc:
        data = {}
        parse_error = str(exc)

    valid = set(candidate_events)
    events = data.get("events") or data.get("top_events") or data.get("predicted_events") or []
    if isinstance(events, dict):
        events = [events]
    predicted = []
    event_evidence: dict[str, str] = {}
    if isinstance(events, list):
        for item in events:
            if not isinstance(item, dict):
                continue
            event = str(item.get("event") or "").strip()
            if event not in valid:
                continue
            score = round(_score(item.get("score", item.get("probability", 0.0))), 4)
            predicted.append({"event": event, "score": score})
            if item.get("evidence"):
                event_evidence[event] = str(item["evidence"]).strip()

    predicted.sort(key=lambda item: item["score"], reverse=True)
    predicted = predicted[:top_k]
    evidence = ""
    if predicted:
        evidence = event_evidence.get(predicted[0]["event"], "")
    has_relevant = data.get("has_relevant_event")
    if not isinstance(has_relevant, bool):
        has_relevant = bool(predicted)

    return {
        "has_relevant_event": bool(has_relevant),
        "predicted_events": predicted,
        "event_evidence": event_evidence,
        "caption": str(data.get("caption") or "").strip(),
        "evidence": evidence,
        "raw_response": raw,
        "parse_error": parse_error,
    }


def resolve_embedding_model_path(model: str | None, cache_dir: Path) -> Path | str:
    model_id = model or DEFAULT_EMBEDDING_MODEL_ID
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


class ClipImageEmbedder:
    def __init__(self, model_path: Path | str, device: str = "auto", dtype: str = "auto", batch_size: int = 32) -> None:
        self.model_path = str(model_path)
        self.device_name = device
        self.dtype = dtype
        self.batch_size = batch_size
        self.model = None
        self.processor = None
        self.device = None

    def load(self) -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        if self.device_name == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.device_name)
        kwargs: dict[str, Any] = {"local_files_only": True}
        if self.dtype == "float16":
            kwargs["torch_dtype"] = torch.float16
        elif self.dtype == "bfloat16":
            kwargs["torch_dtype"] = torch.bfloat16
        elif self.dtype == "float32":
            kwargs["torch_dtype"] = torch.float32
        self.model = CLIPModel.from_pretrained(self.model_path, **kwargs)
        self.processor = CLIPProcessor.from_pretrained(self.model_path, local_files_only=True)
        self.model.to(self.device)
        self.model.eval()

    def embed_image_paths(self, paths: list[Path]) -> np.ndarray:
        import torch

        if self.model is None or self.processor is None:
            self.load()
        vectors = []
        for start in range(0, len(paths), self.batch_size):
            batch_paths = paths[start : start + self.batch_size]
            images = []
            for path in batch_paths:
                with Image.open(path) as img:
                    images.append(img.convert("RGB"))
            inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
            with torch.inference_mode():
                features = self.model.get_image_features(**inputs)
                if not torch.is_tensor(features):
                    tensor_features = getattr(features, "image_embeds", None)
                    if tensor_features is None:
                        tensor_features = getattr(features, "pooler_output", None)
                    if tensor_features is None:
                        tensor_features = features[0]
                    features = tensor_features
                features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            vectors.append(features.detach().cpu().float().numpy())
        if not vectors:
            return np.empty((0, 0), dtype=np.float32)
        return np.concatenate(vectors, axis=0).astype(np.float32)


def _candidate_examples(candidates: list[dict]) -> list[EvidenceExample]:
    output: list[EvidenceExample] = []
    seen: set[str] = set()
    for candidate in candidates:
        event = str(candidate.get("event") or "")
        for item in candidate.get("examples", []):
            rel = str(item.get("relative_path") or "")
            src = str(item.get("source_path") or "")
            if not src or src in seen:
                continue
            seen.add(src)
            output.append(EvidenceExample(event=event, path=Path(src), relative_path=rel, red_box_found=bool(item.get("red_box_found"))))
    return output


def build_qwen_visual_rag_messages(
    examples: list[EvidenceExample],
    target_image_path: Path,
    prompt: str,
    max_pixels: int | None,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": "下面是视觉 RAG 检索到的标准示例图片。优先使用原示意图的红框裁剪；红框内部才是事件关键目标，红框外背景不代表事件。",
        }
    ]
    for idx, example in enumerate(examples, 1):
        image_path = evidence_image_path(example)
        image_item: dict[str, Any] = {"type": "image", "image": str(image_path.resolve())}
        if max_pixels is not None and max_pixels > 0:
            image_item["max_pixels"] = max_pixels
        content.append(image_item)
        source_note = "红框裁剪" if example.red_box_found else "原图，只有红框内部代表事件"
        content.append(
            {
                "type": "text",
                "text": f"检索示例 {idx}: 事件类别 = {example.event}。示例来源: {evidence_relative_path(example)}。证据类型: {source_note}",
            }
        )
    target_item: dict[str, Any] = {"type": "image", "image": str(target_image_path.resolve())}
    if max_pixels is not None and max_pixels > 0:
        target_item["max_pixels"] = max_pixels
    content.append(target_item)
    content.append({"type": "text", "text": f"最后一张待测图像需要判断。{prompt}"})
    return [{"role": "user", "content": content}]


class QwenVisualRagVerifier:
    def __init__(self, model_path: Path | str, device: str, dtype: str, max_new_tokens: int, max_pixels: int) -> None:
        self.model_path = str(model_path)
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.max_pixels = max_pixels
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
        kwargs = {"torch_dtype": "auto" if self.dtype == "auto" else getattr(torch, self.dtype), "local_files_only": True}
        if self.device == "auto":
            kwargs["device_map"] = "auto"
        self.model = model_cls.from_pretrained(self.model_path, **kwargs)
        if self.device != "auto":
            self.model.to(self.device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)

    def verify(self, target_image_path: Path, examples: list[EvidenceExample], prompt: str) -> str:
        import torch

        if self.model is None or self.processor is None:
            self.load()
        messages = build_qwen_visual_rag_messages(examples, target_image_path, prompt, self.max_pixels)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        try:
            from qwen_vl_utils import process_vision_info

            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        except ModuleNotFoundError:
            images = []
            for example in examples:
                with Image.open(example.path) as img:
                    images.append(img.convert("RGB"))
            with Image.open(target_image_path) as img:
                images.append(img.convert("RGB"))
            inputs = self.processor(text=[text], images=images, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        with torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        generated = generated[:, inputs.input_ids.shape[1] :]
        return self.processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def build_evidence_index(embedder: ImageEmbedder, examples: list[EvidenceExample]) -> np.ndarray:
    return embedder.embed_image_paths([evidence_image_path(example) for example in examples])


def predict_records_retrieve_only(
    records: list[dict],
    embedder: ImageEmbedder,
    examples: list[EvidenceExample],
    event_rows: list[dict],
    top_examples: int,
    candidate_events: int,
    examples_per_event: int,
    top_k: int,
) -> list[dict]:
    evidence_embeddings = build_evidence_index(embedder, examples)
    output = []
    for idx, rec in enumerate(records, 1):
        print(f"[{idx}/{len(records)}] {rec['relative_path']}")
        query = embedder.embed_image_paths([Path(rec["source_path"])])[0]
        candidates = rank_event_candidates(query, evidence_embeddings, examples, top_examples, candidate_events, examples_per_event)
        predicted = [{"event": item["event"], "score": item["score"]} for item in candidates[:top_k]]
        output.append(
            {
                **rec,
                "method": "visual-rag-retrieve-only",
                "has_relevant_event": bool(predicted),
                "predicted_events": predicted,
                "retrieved_candidates": candidates,
                "caption": "",
                "evidence": "",
                "raw_response": "",
                "error": "",
            }
        )
    return output


def predict_records_verify(
    records: list[dict],
    embedder: ImageEmbedder,
    verifier: QwenVisualRagVerifier,
    examples: list[EvidenceExample],
    event_rows: list[dict],
    top_examples: int,
    candidate_events: int,
    examples_per_event: int,
    top_k: int,
    threshold: float,
) -> list[dict]:
    evidence_embeddings = build_evidence_index(embedder, examples)
    output = []
    for idx, rec in enumerate(records, 1):
        print(f"[{idx}/{len(records)}] {rec['relative_path']}")
        candidates: list[dict] = []
        try:
            query = embedder.embed_image_paths([Path(rec["source_path"])])[0]
            candidates = rank_event_candidates(query, evidence_embeddings, examples, top_examples, candidate_events, examples_per_event)
            prompt = build_visual_rag_prompt(event_rows, candidates, top_k)
            raw = verifier.verify(Path(rec["source_path"]), _candidate_examples(candidates), prompt)
            parsed = parse_visual_rag_json(raw, [item["event"] for item in candidates], top_k)
            error = parsed.pop("parse_error", "")
        except Exception as exc:
            if is_fatal_inference_error(exc):
                raise RuntimeError(f"Fatal visual-RAG inference error at {rec['relative_path']}: {exc}") from exc
            parsed = {
                "has_relevant_event": False,
                "predicted_events": [],
                "event_evidence": {},
                "caption": "",
                "evidence": "",
                "raw_response": "",
            }
            error = str(exc)
        if parsed["predicted_events"]:
            parsed["has_relevant_event"] = any(float(item["score"]) >= threshold for item in parsed["predicted_events"])
        output.append(
            {
                **rec,
                "method": "visual-rag-qwen-verify",
                "retrieved_candidates": candidates,
                **parsed,
                "error": error,
            }
        )
    return output


def write_outputs(records: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "visual_rag_predictions.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    (out_dir / "visual_rag_predictions.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    fields = [
        "sample_id",
        "ground_truth_event",
        "top1_event",
        "top1_score",
        "hit_top1",
        "retrieved_top1",
        "retrieved_top1_score",
        "retrieved_examples",
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
            retrieved = rec.get("retrieved_candidates", [{}])[0] if rec.get("retrieved_candidates") else {}
            retrieved_examples = " | ".join(str(item.get("relative_path", "")) for item in retrieved.get("examples", []))
            writer.writerow(
                {
                    "sample_id": rec.get("sample_id", ""),
                    "ground_truth_event": rec.get("ground_truth_event", ""),
                    "top1_event": top1.get("event", ""),
                    "top1_score": top1.get("score", ""),
                    "hit_top1": top1.get("event") == rec.get("ground_truth_event"),
                    "retrieved_top1": retrieved.get("event", ""),
                    "retrieved_top1_score": retrieved.get("score", ""),
                    "retrieved_examples": retrieved_examples,
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
        image = img.convert("RGB")
        height = max(int(width * image.height / image.width), 1)
        image.resize((width, height)).save(dst, quality=90)


def _chips(items: list[dict]) -> str:
    return "".join(
        f"<span class='chip'>{html.escape(str(item.get('event', '')))}: {html.escape(str(item.get('score', '')))}</span>"
        for item in items
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
        if src.exists():
            _thumb(src, out_dir / thumb_rel)
        retrieved = rec.get("retrieved_candidates", [])
        cards.append(
            f"""
            <article class="card {'hit' if hit else 'miss'}">
              {f'<img src="{html.escape(thumb_rel)}" loading="lazy" />' if src.exists() else ''}
              <h3>{html.escape(rec.get('ground_truth_event', ''))}</h3>
              <p class="muted">{html.escape(rec.get('relative_path', ''))}</p>
              <div>{_chips(rec.get('predicted_events', []))}</div>
              <p>{html.escape(rec.get('caption', ''))}</p>
              <p class="muted">{html.escape(rec.get('evidence', ''))}</p>
              <h4>Retrieved evidence</h4>
              <div>{_chips(retrieved)}</div>
              {f"<p class='error'>{html.escape(rec.get('error', ''))}</p>" if rec.get('error') else ""}
            </article>
            """
        )
    accuracy = hits / len(records) if records else 0.0
    page = f"""<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8" />
  <title>Visual RAG Report</title>
  <style>
    body {{ font-family: Arial, "Microsoft YaHei", sans-serif; margin: 24px; background: #f6f7f9; color: #17202a; }}
    h1 {{ margin: 0 0 8px; }}
    h4 {{ margin: 12px 0 4px; }}
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
  <h1>Visual RAG Report</h1>
  <p class="summary muted">Images: {len(records)} | Top-1 hits: {hits} | Top-1 accuracy: {accuracy:.3f}</p>
  <section class="grid">{''.join(cards)}</section>
</body>
</html>"""
    asset_dir.mkdir(exist_ok=True)
    (out_dir / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visual-RAG UAV commercial-event recognition from exemplar image folders.")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT, type=Path)
    parser.add_argument("--examples-root", default=DEFAULT_EXAMPLES_ROOT, type=Path)
    parser.add_argument("--out-dir", default=Path("outputs/visual_rag"), type=Path)
    parser.add_argument("--mode", default=DEFAULT_MODE, choices=["retrieve-only", "verify"])
    parser.add_argument("--event-scope", default="examples", choices=["examples", "primary", "taxonomy"])
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL_ID)
    parser.add_argument("--verify-model", default=DEFAULT_VERIFY_MODEL_ID)
    parser.add_argument("--hf-cache", default=DEFAULT_HF_CACHE, type=Path)
    parser.add_argument("--limit", default=None, type=int)
    parser.add_argument("--per-class-limit", default=None, type=int)
    parser.add_argument("--index-examples-per-event", default=None, type=int, help="Limit evidence images per event in the retrieval index. Defaults to all examples.")
    parser.add_argument("--examples-per-event", default=DEFAULT_EXAMPLES_PER_EVENT, type=int)
    parser.add_argument("--red-box-mode", default="auto", choices=["auto", "off", "require"], help="Use red rectangle crops from exemplar images. auto falls back to full images when no box is found.")
    parser.add_argument("--red-box-inset", default=2, type=int, help="Pixels to trim inward from the detected red rectangle before cropping.")
    parser.add_argument("--top-examples", default=DEFAULT_TOP_EXAMPLES, type=int)
    parser.add_argument("--candidate-events", default=DEFAULT_CANDIDATE_EVENTS, type=int)
    parser.add_argument("--top-k", default=DEFAULT_TOP_K, type=int)
    parser.add_argument("--threshold", default=DEFAULT_THRESHOLD, type=float)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--embed-device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--embed-dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--max-new-tokens", default=512, type=int)
    parser.add_argument("--max-pixels", default=DEFAULT_MAX_PIXELS, type=int)
    args = parser.parse_args()

    requested_events = events_for_scope(args.event_scope, args.examples_root)
    examples = load_evidence_examples(args.examples_root, requested_events, args.index_examples_per_event)
    if not examples:
        raise RuntimeError(f"No evidence examples found under {args.examples_root}")
    examples = prepare_evidence_examples(examples, args.out_dir / "redbox_crops", args.red_box_mode, args.red_box_inset)
    if not examples:
        raise RuntimeError(f"No evidence examples remained after red-box preparation under {args.examples_root}")
    event_names = requested_events if requested_events is not None else unique_events_from_examples(examples)
    event_names = [event for event in event_names if any(example.event == event for example in examples)]
    event_rows = event_rows_for_names(event_names)
    records = iter_image_records_for_events(args.data_root, event_names, args.examples_root, args.per_class_limit, args.limit)
    if not records:
        raise RuntimeError(f"No evaluation images found under {args.data_root} for events: {', '.join(event_names)}")

    embedding_model_path = resolve_embedding_model_path(args.embedding_model, args.hf_cache)
    print(f"Embedding model: {embedding_model_path}")
    print(f"Evidence examples: {len(examples)} across {len(event_names)} events")
    print(f"Images: {len(records)}")
    embedder = ClipImageEmbedder(embedding_model_path, device=args.embed_device, dtype=args.embed_dtype, batch_size=args.batch_size)

    if args.mode == "retrieve-only":
        predictions = predict_records_retrieve_only(
            records,
            embedder,
            examples,
            event_rows,
            args.top_examples,
            args.candidate_events,
            args.examples_per_event,
            args.top_k,
        )
    else:
        verify_model_path = resolve_model_path(args.verify_model, args.hf_cache)
        print(f"Verifier model: {verify_model_path}")
        verifier = QwenVisualRagVerifier(verify_model_path, args.device, args.dtype, args.max_new_tokens, args.max_pixels)
        predictions = predict_records_verify(
            records,
            embedder,
            verifier,
            examples,
            event_rows,
            args.top_examples,
            args.candidate_events,
            args.examples_per_event,
            args.top_k,
            args.threshold,
        )

    write_outputs(predictions, args.out_dir)
    build_html_report(predictions, args.out_dir)
    print(f"Wrote predictions: {args.out_dir / 'visual_rag_predictions.jsonl'}")
    print(f"Wrote report: {args.out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
