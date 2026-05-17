from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path


LOCAL_LIB_CANDIDATES = [
    Path.home() / ".cache" / "codex-video-drone-pythonlibs",
    Path(__file__).resolve().parents[1] / ".pythonlibs",
]
for local_libs in reversed(LOCAL_LIB_CANDIDATES):
    if local_libs.exists():
        sys.path.insert(0, str(local_libs))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import imageio.v3 as iio  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def safe_name(value: str) -> str:
    value = re.sub(r'[<>:"/\\|?*\s]+', "_", value.strip())
    value = value.strip("._")
    return value or "unnamed"


def frame_score(prev_gray: np.ndarray | None, gray: np.ndarray) -> float:
    if prev_gray is None:
        return 1.0
    diff = np.abs(prev_gray.astype(np.float32) - gray.astype(np.float32))
    return float(diff.mean() / 255.0)


def small_gray(frame: np.ndarray) -> np.ndarray:
    img = Image.fromarray(frame).convert("L").resize((96, 54), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def read_video_meta(video_path: Path) -> tuple[float, int, int, int, float]:
    meta = iio.immeta(video_path, plugin="pyav") if False else iio.immeta(video_path)
    fps = float(meta.get("fps") or 25.0)
    duration = float(meta.get("duration") or 0.0)
    size = meta.get("size") or meta.get("source_size") or (0, 0)
    width, height = int(size[0] or 0), int(size[1] or 0)
    total_frames = int(round(duration * fps)) if duration > 0 and fps > 0 else 0
    return fps, total_frames, width, height, duration


def should_keep(
    frame_idx: int,
    fps: float,
    prev_kept_frame: int | None,
    uniform_step: int,
    score: float,
    scene_threshold: float,
    min_gap_frames: int,
) -> tuple[bool, str]:
    if frame_idx == 0:
        return True, "first"

    enough_gap = prev_kept_frame is None or (frame_idx - prev_kept_frame) >= min_gap_frames
    if uniform_step > 0 and frame_idx % uniform_step == 0 and enough_gap:
        return True, "uniform"
    if score >= scene_threshold and enough_gap:
        return True, "scene_change"
    return False, ""


def frame_brightness(gray: np.ndarray) -> float:
    return float(gray.mean())


def extract_for_video(
    video_path: Path,
    out_root: Path,
    event_name: str | None,
    sample_fps: float,
    scene_threshold: float,
    min_gap_sec: float,
    max_frames: int | None,
    jpg_quality: int,
    min_brightness: float,
    skip_start_sec: float,
    skip_end_sec: float,
) -> list[dict]:
    fps, total_frames, width, height, duration_sec = read_video_meta(video_path)

    uniform_step = max(int(round(fps / sample_fps)), 1) if sample_fps > 0 else 0
    min_gap_frames = max(int(round(min_gap_sec * fps)), 1)

    event_name = event_name or video_path.parent.name
    video_stem = safe_name(video_path.stem)
    event_safe = safe_name(event_name)
    video_out_dir = out_root / event_safe / video_stem
    video_out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    prev_gray = None
    prev_kept_frame = None
    frame_idx = 0

    for frame in iio.imiter(video_path):
        if frame.ndim == 2:
            frame = np.stack([frame, frame, frame], axis=-1)
        elif frame.shape[-1] == 4:
            frame = frame[:, :, :3]
        timestamp_sec = frame_idx / fps if fps else 0.0
        if timestamp_sec < skip_start_sec:
            frame_idx += 1
            continue
        if skip_end_sec > 0 and duration_sec > 0 and timestamp_sec > max(duration_sec - skip_end_sec, skip_start_sec):
            break

        gray = small_gray(frame)
        brightness = frame_brightness(gray)
        if brightness < min_brightness:
            prev_gray = gray
            frame_idx += 1
            continue
        score = frame_score(prev_gray, gray)
        keep, reason = should_keep(
            frame_idx=frame_idx,
            fps=fps,
            prev_kept_frame=prev_kept_frame,
            uniform_step=uniform_step,
            score=score,
            scene_threshold=scene_threshold,
            min_gap_frames=min_gap_frames,
        )

        if keep:
            filename = (
                f"{event_safe}__{video_stem}"
                f"__t{timestamp_sec:09.3f}s__f{frame_idx:08d}__{reason}.jpg"
            )
            out_path = video_out_dir / filename
            Image.fromarray(frame).save(out_path, quality=jpg_quality)
            prev_kept_frame = frame_idx
            records.append(
                {
                    "event": event_name,
                    "video_name": video_path.name,
                    "video_path": str(video_path.resolve()),
                    "keyframe_path": str(out_path.resolve()),
                    "relative_keyframe_path": str(out_path.relative_to(out_root)).replace("\\", "/"),
                    "frame_index": frame_idx,
                    "timestamp_sec": round(timestamp_sec, 3),
                    "reason": reason,
                    "scene_score": round(score, 6),
                    "brightness": round(brightness, 3),
                    "video_fps": round(fps, 3),
                    "video_total_frames": total_frames,
                    "video_duration_sec": round(duration_sec, 3),
                    "video_width": width,
                    "video_height": height,
                }
            )
            if max_frames is not None and len(records) >= max_frames:
                break

        prev_gray = gray
        frame_idx += 1

    if total_frames == 0:
        total_frames = frame_idx
    if duration_sec == 0 and fps:
        duration_sec = frame_idx / fps
    return records


def write_manifest(records: list[dict], out_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_root / "keyframes_manifest.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    json_path = out_root / "keyframes_manifest.json"
    json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = out_root / "keyframes_manifest.csv"
    if records:
        fields = list(records[0].keys())
        with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract keyframes from a single MP4/video file.")
    parser.add_argument("--mp4-path", required=True, type=Path, help="Path to one MP4/video file.")
    parser.add_argument("--out-dir", default="outputs/keyframes", type=Path, help="Output folder for keyframe images.")
    parser.add_argument(
        "--event-name",
        default=None,
        help="Optional event label. Defaults to the input video's parent folder name.",
    )
    parser.add_argument("--sample-fps", default=1.0, type=float, help="Uniform sampling rate in frames per second.")
    parser.add_argument(
        "--scene-threshold",
        default=0.12,
        type=float,
        help="Mean grayscale frame-difference threshold for scene-change keyframes.",
    )
    parser.add_argument("--min-gap-sec", default=1.0, type=float, help="Minimum time gap between saved keyframes.")
    parser.add_argument(
        "--max-frames-per-video",
        default=None,
        type=int,
        help="Optional hard cap per video. Omit for no cap.",
    )
    parser.add_argument("--jpg-quality", default=92, type=int)
    parser.add_argument(
        "--min-brightness",
        default=8.0,
        type=float,
        help="Skip near-black frames whose mean grayscale brightness is below this threshold.",
    )
    parser.add_argument(
        "--skip-start-sec",
        default=0.0,
        type=float,
        help="Skip the first N seconds of each video, useful for removing takeoff/UI/covered-lens startup frames.",
    )
    parser.add_argument(
        "--skip-end-sec",
        default=0.0,
        type=float,
        help="Skip the last N seconds of each video, useful for removing landing/exit/UI ending frames.",
    )
    args = parser.parse_args()

    video_path = args.mp4_path.resolve()
    out_root = args.out_dir.resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if video_path.suffix.lower() not in VIDEO_EXTS:
        raise ValueError(f"Unsupported video extension: {video_path.suffix}")

    records = extract_for_video(
        video_path=video_path,
        out_root=out_root,
        event_name=args.event_name,
        sample_fps=args.sample_fps,
        scene_threshold=args.scene_threshold,
        min_gap_sec=args.min_gap_sec,
        max_frames=args.max_frames_per_video,
        jpg_quality=args.jpg_quality,
        min_brightness=args.min_brightness,
        skip_start_sec=args.skip_start_sec,
        skip_end_sec=args.skip_end_sec,
    )
    write_manifest(records, out_root)
    print(f"{video_path} -> {len(records)} keyframes")
    print(f"Wrote {len(records)} keyframes to {out_root}")
    print(f"Manifest: {out_root / 'keyframes_manifest.csv'}")


if __name__ == "__main__":
    main()
