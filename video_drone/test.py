from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence


DEFAULT_RESULT_PATH = Path("result.json")
DEFAULT_PROGRESS_CSV = Path("progress.csv")
DEFAULT_PROGRESS_TXT = Path("progress.txt")


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _top_prediction(record: dict[str, Any]) -> str | None:
    predicted = record.get("predicted_event")
    if predicted:
        return str(predicted)
    predicted_events = record.get("predicted_events") or []
    if predicted_events:
        return str(predicted_events[0].get("event") or "")
    return None


def compute_metrics(records: Sequence[dict[str, Any]], classes: Sequence[str]) -> dict[str, Any]:
    per_class: dict[str, dict[str, float]] = {}
    for event in classes:
        tp = fp = fn = 0
        for record in records:
            truth = record.get("ground_truth_event")
            pred = _top_prediction(record)
            if truth == event and pred == event:
                tp += 1
            elif truth != event and pred == event:
                fp += 1
            elif truth == event and pred != event:
                fn += 1
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[event] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    total_tp = sum(int(item["tp"]) for item in per_class.values())
    total_fp = sum(int(item["fp"]) for item in per_class.values())
    total_fn = sum(int(item["fn"]) for item in per_class.values())
    micro_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if micro_precision + micro_recall else 0.0

    class_count = len(classes) or 1
    correct = sum(1 for record in records if record.get("ground_truth_event") == _top_prediction(record))
    return {
        "total": len(records),
        "correct": correct,
        "accuracy": correct / len(records) if records else 0.0,
        "class_count": len(classes),
        "macro_precision": sum(item["precision"] for item in per_class.values()) / class_count,
        "macro_recall": sum(item["recall"] for item in per_class.values()) / class_count,
        "macro_f1": sum(item["f1"] for item in per_class.values()) / class_count,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "per_class": per_class,
    }


def _sha256(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def _latest_auto_change(auto_py_path: Path) -> str:
    if not auto_py_path.exists():
        return "auto.py not found"
    try:
        tree = ast.parse(auto_py_path.read_text(encoding="utf-8"))
    except SyntaxError:
        return "auto.py changelog unavailable: syntax error"
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if "AUTO_RESEARCH_CHANGELOG" in names:
                try:
                    value = ast.literal_eval(node.value)
                except (ValueError, SyntaxError):
                    return "auto.py changelog unavailable"
                if isinstance(value, list) and value:
                    latest = value[-1]
                    if isinstance(latest, dict):
                        return f"{latest.get('time', '')} {latest.get('summary', '')}".strip()
    return "auto.py changelog empty"


def append_progress_csv(
    progress_csv: Path,
    result_path: Path,
    payload: dict[str, Any],
    metrics: dict[str, Any],
    auto_py_path: Path,
) -> None:
    progress_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run_time",
        "result_path",
        "dataset_root",
        "model_id",
        "total",
        "correct",
        "class_count",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "micro_precision",
        "micro_recall",
        "micro_f1",
        "auto_py_sha256",
    ]
    row = {
        "run_time": _now(),
        "result_path": str(result_path.resolve()),
        "dataset_root": payload.get("dataset_root", ""),
        "model_id": payload.get("model_id", ""),
        "total": metrics["total"],
        "correct": metrics["correct"],
        "class_count": metrics["class_count"],
        "accuracy": f"{metrics['accuracy']:.6f}",
        "macro_precision": f"{metrics['macro_precision']:.6f}",
        "macro_recall": f"{metrics['macro_recall']:.6f}",
        "macro_f1": f"{metrics['macro_f1']:.6f}",
        "micro_precision": f"{metrics['micro_precision']:.6f}",
        "micro_recall": f"{metrics['micro_recall']:.6f}",
        "micro_f1": f"{metrics['micro_f1']:.6f}",
        "auto_py_sha256": _sha256(auto_py_path),
    }
    write_header = not progress_csv.exists() or progress_csv.stat().st_size == 0
    with progress_csv.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def append_progress_txt(progress_txt: Path, result_path: Path, metrics: dict[str, Any], auto_py_path: Path) -> None:
    progress_txt.parent.mkdir(parents=True, exist_ok=True)
    line = (
        f"[{_now()}] result={result_path.resolve()} "
        f"macro_recall={metrics['macro_recall']:.6f} macro_f1={metrics['macro_f1']:.6f} "
        f"micro_recall={metrics['micro_recall']:.6f} micro_f1={metrics['micro_f1']:.6f} "
        f"auto.py={_latest_auto_change(auto_py_path)} sha={_sha256(auto_py_path)}\n"
    )
    with progress_txt.open("a", encoding="utf-8") as handle:
        handle.write(line)


def evaluate_result_json(
    result_path: Path = DEFAULT_RESULT_PATH,
    progress_csv: Path = DEFAULT_PROGRESS_CSV,
    progress_txt: Path = DEFAULT_PROGRESS_TXT,
    auto_py_path: Path | None = None,
) -> dict[str, Any]:
    result_path = Path(result_path)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    classes = payload.get("classes") or sorted({record.get("ground_truth_event") for record in payload.get("results", [])})
    classes = [str(event) for event in classes if event]
    records = payload.get("results", [])
    metrics = compute_metrics(records, classes)
    auto_py_path = auto_py_path or Path(__file__).with_name("auto.py")
    append_progress_csv(Path(progress_csv), result_path, payload, metrics, auto_py_path)
    append_progress_txt(Path(progress_txt), result_path, metrics, auto_py_path)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate auto-research result.json and append progress files.")
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT_PATH)
    parser.add_argument("--progress-csv", type=Path, default=DEFAULT_PROGRESS_CSV)
    parser.add_argument("--progress-txt", type=Path, default=DEFAULT_PROGRESS_TXT)
    parser.add_argument("--auto-py", type=Path, default=Path(__file__).with_name("auto.py"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_result_json(
        result_path=args.result,
        progress_csv=args.progress_csv,
        progress_txt=args.progress_txt,
        auto_py_path=args.auto_py,
    )
    print(
        json.dumps(
            {
                "total": metrics["total"],
                "macro_recall": metrics["macro_recall"],
                "macro_f1": metrics["macro_f1"],
                "micro_recall": metrics["micro_recall"],
                "micro_f1": metrics["micro_f1"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
