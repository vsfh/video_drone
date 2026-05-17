from __future__ import annotations

PREDICTION_SCHEMA = {
    "sample_id": "string; video, image, or frame id",
    "source_path": "absolute or relative path to source media",
    "method": "method name, e.g. qwen2.5-vl-zero-shot",
    "predicted_events": [{"event": "event name", "score": "float"}],
    "keyframes": [{"frame_id": "frame path or index", "score": "float", "timestamp_sec": "optional float"}],
    "detections": [{"event": "event name", "bbox_xyxy": [0, 0, 0, 0], "score": "float"}],
    "caption": "generated caption",
    "evidence": "short evidence explanation",
    "department": ["department names"],
}

