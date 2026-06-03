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
DEFAULT_REGION_SIZE = 448
DEFAULT_REGION_STRIDE = 336
DEFAULT_REGION_MIN_SCORE = 0.82
DEFAULT_REGION_MIN_MARGIN = 0.04
DEFAULT_REGION_MIN_EXAMPLE_HITS = 2
DEFAULT_MAX_REGIONS_PER_IMAGE = 96
DEFAULT_QWEN_SCAN_MAX_REGIONS = 32
DEFAULT_QWEN_SCAN_EXAMPLES_PER_EVENT = 5
DEFAULT_QWEN_BATCH_SIZE = 4


@dataclass(frozen=True)
class EvidenceExample:
    event: str
    path: Path
    relative_path: str
    crop_path: Path | None = None
    crop_relative_path: str = ""
    red_box_found: bool = False


@dataclass(frozen=True)
class RegionProposal:
    path: Path
    relative_path: str
    bbox: tuple[int, int, int, int]


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


def group_examples_by_event(examples: list[EvidenceExample], event_names: list[str], examples_per_event: int) -> dict[str, list[EvidenceExample]]:
    grouped: dict[str, list[EvidenceExample]] = {event: [] for event in event_names}
    for example in examples:
        if example.event not in grouped:
            continue
        if len(grouped[example.event]) >= examples_per_event:
            continue
        grouped[example.event].append(example)
    return grouped


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


def _safe_stem(value: str) -> str:
    stem = str(Path(value).with_suffix("")).replace("\\", "__").replace("/", "__")
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in stem)


def _axis_starts(length: int, window: int, stride: int) -> list[int]:
    if length <= window:
        return [0]
    starts = list(range(0, max(length - window, 0) + 1, max(stride, 1)))
    last = length - window
    if starts[-1] != last:
        starts.append(last)
    return starts


def _save_region_crop(image: Image.Image, bbox: tuple[int, int, int, int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.crop(bbox).save(path, quality=92)


def generate_region_proposals(
    image_path: Path,
    region_root: Path,
    sample_id: str,
    region_size: int,
    region_stride: int,
    include_full_image: bool,
    max_regions: int | None = None,
) -> list[RegionProposal]:
    region_root = region_root.resolve()
    base = _safe_stem(sample_id)
    proposals: list[RegionProposal] = []
    with Image.open(image_path) as src:
        image = src.convert("RGB")
        width, height = image.size
        if include_full_image:
            rel = f"regions/{base}__full.jpg"
            path = region_root / f"{base}__full.jpg"
            bbox = (0, 0, width, height)
            _save_region_crop(image, bbox, path)
            proposals.append(RegionProposal(path=path, relative_path=rel, bbox=bbox))
            if max_regions is not None and len(proposals) >= max_regions:
                return proposals

        size = max(1, int(region_size))
        if width <= size and height <= size:
            return proposals
        tile_w = min(size, width)
        tile_h = min(size, height)
        for top in _axis_starts(height, tile_h, region_stride):
            for left in _axis_starts(width, tile_w, region_stride):
                bbox = (left, top, left + tile_w, top + tile_h)
                if include_full_image and bbox == (0, 0, width, height):
                    continue
                rel = f"regions/{base}__x{left:04d}_y{top:04d}_w{tile_w}_h{tile_h}.jpg"
                path = region_root / f"{base}__x{left:04d}_y{top:04d}_w{tile_w}_h{tile_h}.jpg"
                _save_region_crop(image, bbox, path)
                proposals.append(RegionProposal(path=path, relative_path=rel, bbox=bbox))
                if max_regions is not None and len(proposals) >= max_regions:
                    return proposals
    return proposals


def _example_payload(example: EvidenceExample, score: float) -> dict[str, Any]:
    return {
        "relative_path": evidence_relative_path(example),
        "source_path": str(evidence_image_path(example)),
        "original_relative_path": example.relative_path,
        "original_source_path": str(example.path),
        "red_box_found": example.red_box_found,
        "score": score,
    }


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
            bucket["examples"].append(_example_payload(example, score))

    candidates = list(by_event.values())
    candidates.sort(key=lambda item: (-float(item["score"]), item["event"]))
    return candidates[:top_events]


def _rank_events_for_region(
    region_index: int,
    similarities: np.ndarray,
    examples: list[EvidenceExample],
    top_examples: int,
    examples_per_event: int,
    min_example_hits: int,
) -> list[dict]:
    ranked_indices = np.argsort(-similarities)[:top_examples]
    by_event: dict[str, list[tuple[EvidenceExample, float]]] = {}
    for idx in ranked_indices:
        example = examples[int(idx)]
        score = independent_similarity_score(float(similarities[int(idx)]))
        by_event.setdefault(example.event, []).append((example, score))

    events = []
    for event, hits in by_event.items():
        hits.sort(key=lambda item: item[1], reverse=True)
        selected = hits[:examples_per_event]
        if len(selected) < min_example_hits:
            continue
        scored_hits = selected[:max(min_example_hits, 1)]
        score = round(sum(item[1] for item in scored_hits) / len(scored_hits), 4)
        events.append(
            {
                "event": event,
                "score": score,
                "region_index": region_index,
                "example_hits": len(selected),
                "examples": [_example_payload(example, hit_score) for example, hit_score in selected],
            }
        )
    events.sort(key=lambda item: (-float(item["score"]), item["event"]))
    return events


def rank_region_event_candidates(
    region_embeddings: np.ndarray,
    regions: list[RegionProposal],
    evidence_embeddings: np.ndarray,
    examples: list[EvidenceExample],
    top_examples: int,
    top_events: int,
    examples_per_event: int,
    min_score: float,
    min_margin: float,
    min_example_hits: int,
) -> list[dict]:
    if not regions or len(examples) == 0:
        return []
    evidence = _normalize_matrix(evidence_embeddings)
    query = _normalize_matrix(region_embeddings)
    all_similarities = query @ evidence.T
    by_event: dict[str, dict] = {}

    for region_idx, region in enumerate(regions):
        ranked_events = _rank_events_for_region(region_idx, all_similarities[region_idx], examples, top_examples, examples_per_event, min_example_hits)
        if not ranked_events:
            continue
        best = ranked_events[0]
        runner_up = ranked_events[1]["score"] if len(ranked_events) > 1 else 0.0
        margin = round(float(best["score"]) - float(runner_up), 4)
        if float(best["score"]) < min_score or margin < min_margin:
            continue
        candidate = {
            "event": best["event"],
            "score": best["score"],
            "margin": margin,
            "region_relative_path": region.relative_path,
            "region_source_path": str(region.path),
            "region_bbox": list(region.bbox),
            "example_hits": best["example_hits"],
            "examples": best["examples"],
        }
        previous = by_event.get(str(best["event"]))
        if previous is None or float(candidate["score"]) > float(previous["score"]):
            by_event[str(best["event"])] = candidate

    candidates = list(by_event.values())
    candidates.sort(key=lambda item: (-float(item["score"]), -float(item["margin"]), item["event"]))
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


def build_region_verify_prompt(event_row: dict, candidate: dict) -> str:
    event = candidate["event"]
    examples = "、".join(str(item.get("relative_path", "")) for item in candidate.get("examples", []))
    return f"""你是无人机巡检图像的城市治理事件识别助手。输入中前面的图片是“{event}”的红框 object 示例，最后一张图片是从待测图像中裁剪出的候选区域。

任务:
- 只判断最后一张候选区域，不要判断整张原图，也不要把示例图里的事件当作结果。
- 只有候选区域中清楚可见“{event}”的关键 object / region / 状态时，has_event 才能为 true。
- 如果只是背景、颜色、道路、农田、水面、屋顶、拍摄角度相似，但缺少关键 object，必须判 false，score 不得超过 0.3。
- 如果候选区域太模糊、太小、被遮挡、证据不足，必须判 false。

事件定义:
- 事件: {event}
- 定义: {event_row.get('definition', '')}
- 正例: {event_row.get('positive', '')}
- 反例: {event_row.get('negative', '')}

检索信息:
- region_bbox: {candidate.get('region_bbox', [])}
- region_similarity_score: {candidate.get('score', 0.0)}
- margin_to_next_event: {candidate.get('margin', 0.0)}
- matched_redbox_examples: {examples}

要求:
- 只输出一个 JSON 对象，不要输出 Markdown、解释性前后缀或代码块。
- evidence 必须描述最后一张候选区域中实际可见的视觉证据，不能引用示例图作为证据。

JSON 格式:
{{
  "has_event": false,
  "event": "{event}",
  "score": 0.0,
  "caption": "一句候选区域描述",
  "evidence": "一句判断依据"
}}"""


def build_qwen_region_scan_prompt(event_row: dict, example_count: int, region_bbox: tuple[int, int, int, int]) -> str:
    event = event_row["event"]
    return f"""你是无人机巡检图像的城市治理事件识别助手。输入中前面最多 {example_count} 张红框 object 示例是“{event}”的视觉标准，最后一张图片是从待测图像中裁剪出的候选区域。

任务:
- 只判断最后一张候选区域，不要判断整张原图，也不要把示例图里的事件当作结果。
- 前面的示例图只用于理解“{event}”的关键 object / region / 状态；红框外背景不代表事件。
- 只有候选区域中清楚可见与示例红框内部同类的关键 object / region / 状态时，has_event 才能为 true。
- 如果只是背景、颜色、道路、农田、水面、屋顶、拍摄角度相似，但缺少关键 object，必须判 false，score 不得超过 0.3。
- 如果候选区域太模糊、太小、被遮挡、证据不足，必须判 false。

事件定义:
- 事件: {event}
- 定义: {event_row.get('definition', '')}
- 正例: {event_row.get('positive', '')}
- 反例: {event_row.get('negative', '')}

候选区域:
- region_bbox: {list(region_bbox)}

要求:
- 只输出一个 JSON 对象，不要输出 Markdown、解释性前后缀或代码块。
- evidence 必须描述最后一张候选区域中实际可见的视觉证据，不能引用示例图作为证据。

JSON 格式:
{{
  "has_event": false,
  "event": "{event}",
  "score": 0.0,
  "caption": "一句候选区域描述",
  "evidence": "一句判断依据"
}}"""


def parse_region_verify_json(text: str, event_name: str) -> dict:
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
    return {
        "event": event_name,
        "has_event": bool(has_event),
        "score": round(_score(data.get("score", 0.0)), 4),
        "caption": str(data.get("caption") or "").strip(),
        "evidence": str(data.get("evidence") or "").strip(),
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


def configure_generation_processor_padding(processor):
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "left"
    return processor


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
        self.processor = configure_generation_processor_padding(AutoProcessor.from_pretrained(self.model_path, local_files_only=True))

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
                with Image.open(evidence_image_path(example)) as img:
                    images.append(img.convert("RGB"))
            with Image.open(target_image_path) as img:
                images.append(img.convert("RGB"))
            inputs = self.processor(text=[text], images=images, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        with torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        generated = generated[:, inputs.input_ids.shape[1] :]
        output = self.processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return output

    def verify_batch(self, requests: list[dict[str, Any]]) -> list[str]:
        import torch

        if self.model is None or self.processor is None:
            self.load()
        messages_list = [
            build_qwen_visual_rag_messages(
                request["examples"],
                Path(request["target_image_path"]),
                str(request["prompt"]),
                self.max_pixels,
            )
            for request in requests
        ]
        texts = [self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in messages_list]
        try:
            from qwen_vl_utils import process_vision_info

            image_inputs, video_inputs = process_vision_info(messages_list)
            inputs = self.processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        except ModuleNotFoundError:
            images = []
            for request in requests:
                for example in request["examples"]:
                    with Image.open(evidence_image_path(example)) as img:
                        images.append(img.convert("RGB"))
                with Image.open(Path(request["target_image_path"])) as img:
                    images.append(img.convert("RGB"))
            inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        inputs = inputs.to(self.model.device)
        with torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        generated = generated[:, inputs.input_ids.shape[1] :]
        outputs = self.processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return outputs


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


def predict_records_region_retrieve(
    records: list[dict],
    embedder: ImageEmbedder,
    examples: list[EvidenceExample],
    event_rows: list[dict],
    top_examples: int,
    candidate_events: int,
    examples_per_event: int,
    top_k: int,
    region_root: Path,
    region_size: int,
    region_stride: int,
    include_full_region: bool,
    min_score: float,
    min_margin: float,
    min_example_hits: int,
    max_regions_per_image: int | None,
) -> list[dict]:
    evidence_embeddings = build_evidence_index(embedder, examples)
    output = []
    for idx, rec in enumerate(records, 1):
        print(f"[{idx}/{len(records)}] {rec['relative_path']}")
        regions = generate_region_proposals(
            Path(rec["source_path"]),
            region_root,
            str(rec.get("sample_id") or rec["relative_path"]),
            region_size,
            region_stride,
            include_full_region,
            max_regions_per_image,
        )
        region_embeddings = embedder.embed_image_paths([region.path for region in regions])
        candidates = rank_region_event_candidates(
            region_embeddings,
            regions,
            evidence_embeddings,
            examples,
            top_examples,
            candidate_events,
            examples_per_event,
            min_score,
            min_margin,
            min_example_hits,
        )
        predicted = [{"event": item["event"], "score": item["score"]} for item in candidates[:top_k]]
        output.append(
            {
                **rec,
                "method": "visual-rag-region-retrieve",
                "has_relevant_event": bool(predicted),
                "predicted_events": predicted,
                "retrieved_candidates": candidates,
                "region_count": len(regions),
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


def predict_records_region_verify(
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
    region_root: Path,
    region_size: int,
    region_stride: int,
    include_full_region: bool,
    min_score: float,
    min_margin: float,
    min_example_hits: int,
    max_regions_per_image: int | None,
) -> list[dict]:
    evidence_embeddings = build_evidence_index(embedder, examples)
    row_by_event = {row["event"]: row for row in event_rows}
    output = []
    for idx, rec in enumerate(records, 1):
        print(f"[{idx}/{len(records)}] {rec['relative_path']}")
        candidates: list[dict] = []
        checks = []
        errors = []
        try:
            regions = generate_region_proposals(
                Path(rec["source_path"]),
                region_root,
                str(rec.get("sample_id") or rec["relative_path"]),
                region_size,
                region_stride,
                include_full_region,
                max_regions_per_image,
            )
            region_embeddings = embedder.embed_image_paths([region.path for region in regions])
            candidates = rank_region_event_candidates(
                region_embeddings,
                regions,
                evidence_embeddings,
                examples,
                top_examples,
                candidate_events,
                examples_per_event,
                min_score,
                min_margin,
                min_example_hits,
            )
            positives = []
            for candidate in candidates[:top_k]:
                event = str(candidate["event"])
                prompt = build_region_verify_prompt(row_by_event.get(event) or event_rows_for_names([event])[0], candidate)
                raw = verifier.verify(Path(candidate["region_source_path"]), _candidate_examples([candidate]), prompt)
                parsed = parse_region_verify_json(raw, event)
                error = parsed.pop("parse_error", "")
                check = {**candidate, **parsed, "error": error}
                checks.append(check)
                if error:
                    errors.append(f"{event}: {error}")
                if check["has_event"] and float(check["score"]) >= threshold:
                    positives.append(
                        {
                            "event": event,
                            "score": check["score"],
                            "region_bbox": candidate.get("region_bbox", []),
                            "region_relative_path": candidate.get("region_relative_path", ""),
                        }
                    )
            positives.sort(key=lambda item: item["score"], reverse=True)
            parsed_output = {
                "has_relevant_event": bool(positives),
                "predicted_events": positives[:top_k],
                "caption": checks[0].get("caption", "") if checks else "",
                "evidence": next((check.get("evidence", "") for check in checks if check.get("has_event")), ""),
                "raw_response": json.dumps(checks, ensure_ascii=False),
            }
        except Exception as exc:
            if is_fatal_inference_error(exc):
                raise RuntimeError(f"Fatal region visual-RAG inference error at {rec['relative_path']}: {exc}") from exc
            parsed_output = {
                "has_relevant_event": False,
                "predicted_events": [],
                "caption": "",
                "evidence": "",
                "raw_response": "",
            }
            errors.append(str(exc))
        output.append(
            {
                **rec,
                "method": "visual-rag-region-qwen-verify",
                "retrieved_candidates": candidates,
                "region_verify_checks": checks,
                **parsed_output,
                "error": " | ".join(errors),
            }
        )
    return output


def predict_records_qwen_region_scan(
    records: list[dict],
    verifier: QwenVisualRagVerifier,
    examples: list[EvidenceExample],
    event_rows: list[dict],
    examples_per_event: int,
    top_k: int,
    threshold: float,
    region_root: Path,
    region_size: int,
    region_stride: int,
    include_full_region: bool,
    max_regions_per_image: int | None,
    qwen_batch_size: int = 1,
) -> list[dict]:
    event_names = [row["event"] for row in event_rows]
    examples_by_event = group_examples_by_event(examples, event_names, examples_per_event)
    output = []
    for idx, rec in enumerate(records, 1):
        print(f"[{idx}/{len(records)}] {rec['relative_path']}")
        checks = []
        errors = []
        positives_by_event: dict[str, dict] = {}
        try:
            regions = generate_region_proposals(
                Path(rec["source_path"]),
                region_root,
                str(rec.get("sample_id") or rec["relative_path"]),
                region_size,
                region_stride,
                include_full_region,
                max_regions_per_image,
            )
            requests = []
            for row in event_rows:
                event = row["event"]
                event_examples = examples_by_event.get(event, [])
                for region in regions:
                    prompt = build_qwen_region_scan_prompt(row, len(event_examples), region.bbox)
                    requests.append(
                        {
                            "event": event,
                            "row": row,
                            "region": region,
                            "examples": event_examples,
                            "prompt": prompt,
                            "target_image_path": region.path,
                        }
                    )

            batch_size = max(1, int(qwen_batch_size))
            for start in range(0, len(requests), batch_size):
                batch = requests[start : start + batch_size]
                if hasattr(verifier, "verify_batch"):
                    raw_outputs = verifier.verify_batch(batch)
                else:
                    raw_outputs = [
                        verifier.verify(Path(request["target_image_path"]), request["examples"], request["prompt"])
                        for request in batch
                    ]
                if len(raw_outputs) != len(batch):
                    raise RuntimeError(f"Batch verifier returned {len(raw_outputs)} outputs for {len(batch)} requests.")
                for request, raw in zip(batch, raw_outputs):
                    event = str(request["event"])
                    region = request["region"]
                    event_examples = request["examples"]
                    parsed = parse_region_verify_json(raw, event)
                    error = parsed.pop("parse_error", "")
                    check = {
                        **parsed,
                        "region_bbox": list(region.bbox),
                        "region_relative_path": region.relative_path,
                        "region_source_path": str(region.path),
                        "example_count": len(event_examples),
                        "example_paths": [evidence_relative_path(example) for example in event_examples],
                        "error": error,
                    }
                    checks.append(check)
                    if error:
                        errors.append(f"{event} {region.relative_path}: {error}")
                    if check["has_event"] and float(check["score"]) >= threshold:
                        previous = positives_by_event.get(event)
                        if previous is None or float(check["score"]) > float(previous["score"]):
                            positives_by_event[event] = {
                                "event": event,
                                "score": check["score"],
                                "region_bbox": check["region_bbox"],
                                "region_relative_path": check["region_relative_path"],
                            }
            predicted = list(positives_by_event.values())
            predicted.sort(key=lambda item: item["score"], reverse=True)
            first_positive = next((check for check in checks if check.get("has_event") and float(check.get("score", 0.0)) >= threshold), {})
            parsed_output = {
                "has_relevant_event": bool(predicted),
                "predicted_events": predicted[:top_k],
                "caption": str(first_positive.get("caption") or ""),
                "evidence": str(first_positive.get("evidence") or ""),
                "raw_response": json.dumps(checks, ensure_ascii=False),
                "region_count": len(regions),
            }
        except Exception as exc:
            if is_fatal_inference_error(exc):
                raise RuntimeError(f"Fatal Qwen region-scan inference error at {rec['relative_path']}: {exc}") from exc
            parsed_output = {
                "has_relevant_event": False,
                "predicted_events": [],
                "caption": "",
                "evidence": "",
                "raw_response": "",
                "region_count": 0,
            }
            errors.append(str(exc))
        output.append(
            {
                **rec,
                "method": "qwen-region-scan",
                "retrieved_candidates": [],
                "event_region_checks": checks,
                **parsed_output,
                "error": " | ".join(errors),
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
        "retrieved_region",
        "retrieved_region_bbox",
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
            region_rel = retrieved.get("region_relative_path", "") or top1.get("region_relative_path", "")
            region_bbox = retrieved.get("region_bbox", []) or top1.get("region_bbox", [])
            writer.writerow(
                {
                    "sample_id": rec.get("sample_id", ""),
                    "ground_truth_event": rec.get("ground_truth_event", ""),
                    "top1_event": top1.get("event", ""),
                    "top1_score": top1.get("score", ""),
                    "hit_top1": top1.get("event") == rec.get("ground_truth_event"),
                    "retrieved_top1": retrieved.get("event", ""),
                    "retrieved_top1_score": retrieved.get("score", ""),
                    "retrieved_region": region_rel,
                    "retrieved_region_bbox": json.dumps(region_bbox, ensure_ascii=False),
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
    parser.add_argument("--mode", default=DEFAULT_MODE, choices=["retrieve-only", "verify", "region-retrieve", "region-verify", "qwen-region-scan"])
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
    parser.add_argument("--region-size", default=DEFAULT_REGION_SIZE, type=int)
    parser.add_argument("--region-stride", default=DEFAULT_REGION_STRIDE, type=int)
    parser.add_argument("--no-full-region", action="store_true", help="Do not include the full target image as an extra region proposal.")
    parser.add_argument("--region-min-score", default=DEFAULT_REGION_MIN_SCORE, type=float)
    parser.add_argument("--region-min-margin", default=DEFAULT_REGION_MIN_MARGIN, type=float)
    parser.add_argument("--region-min-example-hits", default=DEFAULT_REGION_MIN_EXAMPLE_HITS, type=int)
    parser.add_argument("--max-regions-per-image", default=DEFAULT_MAX_REGIONS_PER_IMAGE, type=int, help="Cap generated target regions per image to limit runtime and memory. Use 0 for no cap.")
    parser.add_argument("--qwen-scan-max-regions", default=DEFAULT_QWEN_SCAN_MAX_REGIONS, type=int, help="Qwen-only scan region cap per image.")
    parser.add_argument("--qwen-scan-examples-per-event", default=DEFAULT_QWEN_SCAN_EXAMPLES_PER_EVENT, type=int, help="Qwen-only scan redbox examples per event.")
    parser.add_argument("--qwen-batch-size", default=DEFAULT_QWEN_BATCH_SIZE, type=int, help="Batch size for Qwen region scan generation. Lower this if CUDA OOM occurs.")
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
    if args.mode == "qwen-region-scan":
        requested_events = [row["event"] for row in EVENT_TAXONOMY if row["event"] in PRIMARY_EVENTS]
    examples = load_evidence_examples(args.examples_root, requested_events, args.index_examples_per_event)
    if not examples:
        raise RuntimeError(f"No evidence examples found under {args.examples_root}")
    examples = prepare_evidence_examples(examples, args.out_dir / "redbox_crops", args.red_box_mode, args.red_box_inset)
    if not examples:
        raise RuntimeError(f"No evidence examples remained after red-box preparation under {args.examples_root}")
    event_names = requested_events if requested_events is not None else unique_events_from_examples(examples)
    if args.mode != "qwen-region-scan":
        event_names = [event for event in event_names if any(example.event == event for example in examples)]
    event_rows = event_rows_for_names(event_names)
    records = iter_image_records_for_events(args.data_root, event_names, args.examples_root, args.per_class_limit, args.limit)
    if not records:
        raise RuntimeError(f"No evaluation images found under {args.data_root} for events: {', '.join(event_names)}")

    print(f"Evidence examples: {len(examples)} across {len(event_names)} events")
    print(f"Images: {len(records)}")
    max_regions_per_image = None if args.max_regions_per_image is not None and args.max_regions_per_image <= 0 else args.max_regions_per_image

    if args.mode == "retrieve-only":
        embedding_model_path = resolve_embedding_model_path(args.embedding_model, args.hf_cache)
        print(f"Embedding model: {embedding_model_path}")
        embedder = ClipImageEmbedder(embedding_model_path, device=args.embed_device, dtype=args.embed_dtype, batch_size=args.batch_size)
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
    elif args.mode == "region-retrieve":
        embedding_model_path = resolve_embedding_model_path(args.embedding_model, args.hf_cache)
        print(f"Embedding model: {embedding_model_path}")
        embedder = ClipImageEmbedder(embedding_model_path, device=args.embed_device, dtype=args.embed_dtype, batch_size=args.batch_size)
        predictions = predict_records_region_retrieve(
            records,
            embedder,
            examples,
            event_rows,
            args.top_examples,
            args.candidate_events,
            args.examples_per_event,
            args.top_k,
            args.out_dir / "regions",
            args.region_size,
            args.region_stride,
            not args.no_full_region,
            args.region_min_score,
            args.region_min_margin,
            args.region_min_example_hits,
            max_regions_per_image,
        )
    elif args.mode == "verify":
        embedding_model_path = resolve_embedding_model_path(args.embedding_model, args.hf_cache)
        print(f"Embedding model: {embedding_model_path}")
        embed_device = args.embed_device
        if embed_device == "auto":
            embed_device = "cpu"
            print("Embedding device: cpu (auto-selected for verify mode to leave GPU memory for Qwen)")
        embedder = ClipImageEmbedder(embedding_model_path, device=embed_device, dtype=args.embed_dtype, batch_size=args.batch_size)
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
    elif args.mode == "region-verify":
        embedding_model_path = resolve_embedding_model_path(args.embedding_model, args.hf_cache)
        print(f"Embedding model: {embedding_model_path}")
        embed_device = args.embed_device
        if embed_device == "auto":
            embed_device = "cpu"
            print("Embedding device: cpu (auto-selected for verify mode to leave GPU memory for Qwen)")
        embedder = ClipImageEmbedder(embedding_model_path, device=embed_device, dtype=args.embed_dtype, batch_size=args.batch_size)
        verify_model_path = resolve_model_path(args.verify_model, args.hf_cache)
        print(f"Verifier model: {verify_model_path}")
        verifier = QwenVisualRagVerifier(verify_model_path, args.device, args.dtype, args.max_new_tokens, args.max_pixels)
        predictions = predict_records_region_verify(
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
            args.out_dir / "regions",
            args.region_size,
            args.region_stride,
            not args.no_full_region,
            args.region_min_score,
            args.region_min_margin,
            args.region_min_example_hits,
            max_regions_per_image,
        )
    else:
        verify_model_path = resolve_model_path(args.verify_model, args.hf_cache)
        print(f"Verifier model: {verify_model_path}")
        verifier = QwenVisualRagVerifier(verify_model_path, args.device, args.dtype, args.max_new_tokens, args.max_pixels)
        scan_max_regions = None if args.qwen_scan_max_regions is not None and args.qwen_scan_max_regions <= 0 else args.qwen_scan_max_regions
        predictions = predict_records_qwen_region_scan(
            records,
            verifier,
            examples,
            event_rows,
            args.qwen_scan_examples_per_event,
            args.top_k,
            args.threshold,
            args.out_dir / "regions",
            args.region_size,
            args.region_stride,
            not args.no_full_region,
            scan_max_regions,
            args.qwen_batch_size,
        )

    write_outputs(predictions, args.out_dir)
    build_html_report(predictions, args.out_dir)
    print(f"Wrote predictions: {args.out_dir / 'visual_rag_predictions.jsonl'}")
    print(f"Wrote report: {args.out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
