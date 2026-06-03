from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence, TextIO

from PIL import Image


DEFAULT_DATA_ROOT = Path("/media/data1/feihong/video_drone_data/photos_2025-05-30_2025-06-30")
DEFAULT_OUTPUT_PATH = Path("result.json")
DEFAULT_MODEL_ID = "/media/data1/feihong/hf_cache/models--Qwen--Qwen3-VL-4B-Instruct"
DEFAULT_SHOTS_PER_CLASS = 3
DEFAULT_CLASS_CHUNK_SIZE = 4
DEFAULT_MAX_PIXELS = 262144
DEFAULT_PROGRESS = "auto"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

AUTO_RESEARCH_CHANGELOG = [
    {
        "time": "2026-06-03 11:45:00",
        "summary": "Initial auto-research pipeline: dataset discovery, chunked few-shot VLM inference, result.json output.",
    },
    {
        "time": "2026-06-03 12:30:00",
        "summary": "Force Qwen3-VL-4B loading from the local HuggingFace cache path with local_files_only=True.",
    },
    {
        "time": "2026-06-03 13:10:00",
        "summary": "Add sample and class-chunk progress reporting with tqdm/text/none modes.",
    },
    {
        "time": "2026-06-03 13:35:00",
        "summary": "Print per-chunk image loading/processor time and model forward generation time.",
    }
]


@dataclass(frozen=True)
class FewShotImage:
    event: str
    path: Path
    relative_path: str


@dataclass(frozen=True)
class ImageSample:
    sample_id: str
    ground_truth_event: str
    path: Path
    relative_path: str


@dataclass(frozen=True)
class DatasetBundle:
    data_root: Path
    classes: list[str]
    examples_by_class: dict[str, list[FewShotImage]]
    samples: list[ImageSample]


def iter_image_paths(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and not any(part.startswith(".") for part in path.parts)
    )


def discover_event_classes(data_root: Path) -> list[str]:
    data_root = Path(data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {data_root}")
    classes: list[str] = []
    for child in sorted(data_root.iterdir(), key=lambda item: item.name):
        if child.is_dir() and not child.name.startswith(".") and iter_image_paths(child):
            classes.append(child.name)
    if not classes:
        raise ValueError(f"No event class folders with images found under {data_root}")
    return classes


def _image_source(event_root: Path, subdir: str | None) -> Path:
    if subdir:
        preferred = event_root / subdir
        if preferred.exists():
            return preferred
    return event_root


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def build_dataset(
    data_root: Path,
    shots_per_class: int = DEFAULT_SHOTS_PER_CLASS,
    example_subdir: str = "cropped",
    target_subdir: str = "original",
    classes: Sequence[str] | None = None,
    exclude_shots: bool = True,
) -> DatasetBundle:
    data_root = Path(data_root)
    event_classes = list(classes) if classes is not None else discover_event_classes(data_root)
    examples_by_class: dict[str, list[FewShotImage]] = {}
    samples: list[ImageSample] = []

    for event in event_classes:
        event_root = data_root / event
        example_paths = iter_image_paths(_image_source(event_root, example_subdir))[:shots_per_class]
        target_paths = iter_image_paths(_image_source(event_root, target_subdir))
        if exclude_shots:
            target_paths = target_paths[shots_per_class:]

        examples_by_class[event] = [
            FewShotImage(event=event, path=path, relative_path=_relative(path, data_root)) for path in example_paths
        ]
        for path in target_paths:
            relative_path = _relative(path, data_root)
            samples.append(
                ImageSample(
                    sample_id=relative_path,
                    ground_truth_event=event,
                    path=path,
                    relative_path=relative_path,
                )
            )

    return DatasetBundle(data_root=data_root, classes=event_classes, examples_by_class=examples_by_class, samples=samples)


def chunked(items: Sequence[str], chunk_size: int) -> Iterable[list[str]]:
    if chunk_size <= 0:
        yield list(items)
        return
    for index in range(0, len(items), chunk_size):
        yield list(items[index : index + chunk_size])


def _progress_write(stream: TextIO, message: str) -> None:
    stream.write(message + "\n")
    stream.flush()


def _progress_mode(progress: str, stream: TextIO) -> str:
    if progress not in {"auto", "tqdm", "text", "none"}:
        raise ValueError("progress must be one of: auto, tqdm, text, none")
    if progress == "none":
        return "none"
    if progress == "text":
        return "text"
    try:
        import tqdm  # noqa: F401

        return "tqdm"
    except ImportError:
        if progress == "tqdm":
            _progress_write(stream, "[auto] tqdm is not installed; falling back to text progress")
        return "text"


def _progress_bar(items: Sequence[Any], mode: str, stream: TextIO, desc: str, unit: str):
    if mode != "tqdm":
        return items
    from tqdm.auto import tqdm

    return tqdm(items, total=len(items), desc=desc, unit=unit, file=stream, dynamic_ncols=True)


def _format_seconds(value: Any) -> str:
    try:
        return f"{float(value):.3f}s"
    except (TypeError, ValueError):
        return "n/a"


def build_agent_prompt(classes: Sequence[str], examples_by_class: dict[str, list[FewShotImage]]) -> str:
    class_lines = "\n".join(f"- {event}: {len(examples_by_class.get(event, []))} labeled examples" for event in classes)
    return f"""You are a strict UAV commercial-event image detection agent.

Rules:
1. The only valid event classes are the folder names listed below.
2. The labeled images before the target image are few-shot standards. Learn the visible event object and scene criteria from them.
3. The final image is the only target image to classify.
4. If several classes are plausible, rank them independently with scores from 0 to 1. Scores do not need to sum to 1.
5. Do not invent classes outside the allowed list.
6. Return compact JSON only.

Allowed classes:
{class_lines}

JSON schema:
{{
  "predicted_event": "one allowed class or null",
  "predicted_events": [{{"event": "class name", "score": 0.0}}],
  "caption": "brief visual description",
  "evidence": "why the top class is supported by the target image"
}}
"""


def build_vlm_messages(
    sample: ImageSample,
    examples_by_class: dict[str, list[FewShotImage]],
    classes: Sequence[str],
    prompt: str,
    max_pixels: int = DEFAULT_MAX_PIXELS,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for event in classes:
        for index, example in enumerate(examples_by_class.get(event, []), start=1):
            content.append({"type": "text", "text": f"Few-shot example {index} for class: {event}"})
            content.append({"type": "image", "image": str(example.path.resolve()), "max_pixels": max_pixels})
    content.append({"type": "text", "text": "Target image. Classify this final image only."})
    content.append({"type": "image", "image": str(sample.path.resolve()), "max_pixels": max_pixels})
    content.append({"type": "text", "text": "Return JSON now."})
    return [{"role": "user", "content": content}]


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
    return {}


def _float_score(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def parse_prediction_json(text: str, classes: Sequence[str], default_event: str | None = None) -> dict[str, Any]:
    allowed = set(classes)
    payload = _extract_json_object(text)
    predicted_events: list[dict[str, Any]] = []

    for item in payload.get("predicted_events", []) or []:
        if not isinstance(item, dict):
            continue
        event = str(item.get("event", "")).strip()
        if event in allowed:
            predicted_events.append({"event": event, "score": _float_score(item.get("score", item.get("probability", 0.0)))})

    predicted_event = payload.get("predicted_event")
    if predicted_event not in allowed:
        predicted_event = predicted_events[0]["event"] if predicted_events else default_event

    if predicted_event in allowed and not any(item["event"] == predicted_event for item in predicted_events):
        predicted_events.insert(0, {"event": predicted_event, "score": 0.0})

    predicted_events.sort(key=lambda item: item["score"], reverse=True)
    if predicted_events:
        predicted_event = predicted_events[0]["event"]

    return {
        "predicted_event": predicted_event if predicted_event in allowed else None,
        "predicted_events": predicted_events,
        "caption": str(payload.get("caption", "") or ""),
        "evidence": str(payload.get("evidence", "") or ""),
        "raw_response": text,
    }


def resolve_local_model_path(model_id: str | Path) -> Path:
    model_path = Path(model_id)
    if not model_path.exists():
        raise FileNotFoundError(f"Forced local model path not found: {model_path}")
    if (model_path / "config.json").exists():
        return model_path

    refs_main = model_path / "refs" / "main"
    snapshots_root = model_path / "snapshots"
    if refs_main.exists():
        revision = refs_main.read_text(encoding="utf-8").strip()
        snapshot = snapshots_root / revision
        if snapshot.exists():
            return snapshot

    if snapshots_root.exists():
        snapshots = sorted(
            (path for path in snapshots_root.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if snapshots:
            return snapshots[0]

    raise FileNotFoundError(
        "Could not resolve a HuggingFace snapshot under forced local model path: "
        f"{model_path}. Expected config.json or snapshots/<revision>."
    )


class QwenVlmPredictor:
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        max_new_tokens: int = 256,
    ) -> None:
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.model_path = resolve_local_model_path(model_id)
        self.last_timing: dict[str, float] = {}

        import torch
        from transformers import AutoProcessor

        try:
            from transformers import Qwen3VLForConditionalGeneration as ModelClass
        except ImportError:
            try:
                from transformers import AutoModelForImageTextToText as ModelClass
            except ImportError:
                from transformers import AutoModelForVision2Seq as ModelClass

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)
        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = "left"
        self.model = ModelClass.from_pretrained(
            self.model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            local_files_only=True,
        )
        self.model.eval()

    def _input_device(self):
        if self.torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def predict(
        self,
        sample: ImageSample,
        examples_by_class: dict[str, list[FewShotImage]],
        classes: Sequence[str],
        prompt: str,
        max_pixels: int = DEFAULT_MAX_PIXELS,
    ) -> str:
        total_start = time.perf_counter()
        load_start = time.perf_counter()
        messages = build_vlm_messages(sample, examples_by_class, classes, prompt, max_pixels=max_pixels)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        try:
            from qwen_vl_utils import process_vision_info

            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        except Exception:
            image_inputs = []
            for message in messages:
                for item in message["content"]:
                    if item.get("type") == "image":
                        image_inputs.append(Image.open(item["image"]).convert("RGB"))
            inputs = self.processor(text=[text], images=image_inputs, padding=True, return_tensors="pt")

        inputs = inputs.to(self._input_device())
        image_load_seconds = time.perf_counter() - load_start
        forward_start = time.perf_counter()
        with self.torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        forward_seconds = time.perf_counter() - forward_start
        generated_trimmed = [
            output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, generated_ids, strict=False)
        ]
        output = self.processor.batch_decode(generated_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
        self.last_timing = {
            "image_load_seconds": image_load_seconds,
            "forward_seconds": forward_seconds,
            "total_seconds": time.perf_counter() - total_start,
        }
        return output


def _merge_chunk_predictions(parsed_chunks: list[dict[str, Any]], classes: Sequence[str]) -> dict[str, Any]:
    best_by_event: dict[str, dict[str, Any]] = {}
    for parsed in parsed_chunks:
        for item in parsed.get("predicted_events", []):
            event = item["event"]
            if event not in best_by_event or item["score"] > best_by_event[event]["score"]:
                best_by_event[event] = {"event": event, "score": item["score"]}
    ranked = sorted(best_by_event.values(), key=lambda item: item["score"], reverse=True)
    predicted_event = ranked[0]["event"] if ranked else None
    caption = next((chunk.get("caption", "") for chunk in parsed_chunks if chunk.get("caption")), "")
    evidence = next((chunk.get("evidence", "") for chunk in parsed_chunks if chunk.get("evidence")), "")
    return {
        "predicted_event": predicted_event if predicted_event in set(classes) else None,
        "predicted_events": ranked,
        "caption": caption,
        "evidence": evidence,
        "chunk_responses": parsed_chunks,
    }


def forward(
    data_root: Path = DEFAULT_DATA_ROOT,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    predictor: Any | None = None,
    model_id: str = DEFAULT_MODEL_ID,
    shots_per_class: int = DEFAULT_SHOTS_PER_CLASS,
    example_subdir: str = "cropped",
    target_subdir: str = "original",
    class_chunk_size: int = DEFAULT_CLASS_CHUNK_SIZE,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    limit: int | None = None,
    progress: str = DEFAULT_PROGRESS,
    progress_stream: TextIO | None = None,
) -> list[dict[str, Any]]:
    progress_stream = progress_stream or sys.stderr
    resolved_progress = _progress_mode(progress, progress_stream)
    dataset = build_dataset(
        data_root=data_root,
        shots_per_class=shots_per_class,
        example_subdir=example_subdir,
        target_subdir=target_subdir,
    )
    if resolved_progress == "text":
        _progress_write(
            progress_stream,
            (
                f"[auto] dataset classes={len(dataset.classes)} "
                f"examples={sum(len(items) for items in dataset.examples_by_class.values())} "
                f"samples={len(dataset.samples if limit is None else dataset.samples[:limit])}"
            ),
        )
    predictor = predictor or QwenVlmPredictor(model_id=model_id)
    records: list[dict[str, Any]] = []
    samples = dataset.samples[:limit] if limit is not None else dataset.samples
    class_chunks = list(chunked(dataset.classes, class_chunk_size))

    sample_items = list(enumerate(samples, start=1))
    for sample_index, sample in _progress_bar(sample_items, resolved_progress, progress_stream, "samples", "sample"):
        if resolved_progress == "text":
            _progress_write(
                progress_stream,
                f"[auto] sample {sample_index}/{len(samples)} {sample.relative_path}",
            )
        parsed_chunks: list[dict[str, Any]] = []
        chunk_items = list(enumerate(class_chunks, start=1))
        for chunk_index, class_chunk in chunk_items:
            if resolved_progress == "text":
                _progress_write(
                    progress_stream,
                    f"[auto] chunk {chunk_index}/{len(class_chunks)} classes={','.join(class_chunk)}",
                )
            chunk_examples = {event: dataset.examples_by_class.get(event, []) for event in class_chunk}
            prompt = build_agent_prompt(class_chunk, chunk_examples)
            predict_start = time.perf_counter()
            raw = predictor.predict(sample, chunk_examples, class_chunk, prompt, max_pixels)
            predict_total_seconds = time.perf_counter() - predict_start
            timing = dict(getattr(predictor, "last_timing", {}) or {})
            timing.setdefault("total_seconds", predict_total_seconds)
            if resolved_progress != "none":
                _progress_write(
                    progress_stream,
                    (
                        f"[auto] timing sample={sample_index}/{len(samples)} "
                        f"chunk={chunk_index}/{len(class_chunks)} "
                        f"load_images={_format_seconds(timing.get('image_load_seconds'))} "
                        f"forward={_format_seconds(timing.get('forward_seconds'))} "
                        f"total={_format_seconds(timing.get('total_seconds'))}"
                    ),
                )
            parsed = parse_prediction_json(raw, class_chunk, default_event=None)
            parsed["timing"] = timing
            parsed_chunks.append(parsed)

        merged = _merge_chunk_predictions(parsed_chunks, dataset.classes)
        records.append(
            {
                "sample_id": sample.sample_id,
                "source_path": str(sample.path.resolve()),
                "relative_path": sample.relative_path,
                "ground_truth_event": sample.ground_truth_event,
                "predicted_event": merged["predicted_event"],
                "predicted_events": merged["predicted_events"],
                "caption": merged["caption"],
                "evidence": merged["evidence"],
                "chunk_responses": merged["chunk_responses"],
                "sample_index": sample_index,
            }
        )

    payload = {
        "dataset_root": str(Path(data_root).resolve()),
        "model_id": getattr(predictor, "model_id", model_id),
        "shots_per_class": shots_per_class,
        "example_subdir": example_subdir,
        "target_subdir": target_subdir,
        "class_chunk_size": class_chunk_size,
        "max_pixels": max_pixels,
        "classes": dataset.classes,
        "num_examples": sum(len(items) for items in dataset.examples_by_class.values()),
        "num_samples": len(records),
        "results": records,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if resolved_progress == "text":
        _progress_write(progress_stream, f"[auto] wrote result.json path={output_path}")
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Few-shot VLM auto-research for UAV commercial event images.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--shots-per-class", type=int, default=DEFAULT_SHOTS_PER_CLASS)
    parser.add_argument("--example-subdir", default="cropped")
    parser.add_argument("--target-subdir", default="original")
    parser.add_argument("--class-chunk-size", type=int, default=DEFAULT_CLASS_CHUNK_SIZE)
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress", choices=["auto", "tqdm", "text", "none"], default=DEFAULT_PROGRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    forward(
        data_root=args.data_root,
        output_path=args.output,
        model_id=args.model_id,
        shots_per_class=args.shots_per_class,
        example_subdir=args.example_subdir,
        target_subdir=args.target_subdir,
        class_chunk_size=args.class_chunk_size,
        max_pixels=args.max_pixels,
        limit=args.limit,
        progress=args.progress,
    )


if __name__ == "__main__":
    main()
