from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def _load_index(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _backend() -> str | None:
    try:
        import cv2  # noqa: F401

        return "cv2"
    except Exception:
        pass
    try:
        import imageio  # noqa: F401

        return "imageio"
    except Exception:
        pass
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    return None


def _sample_cv2(video_path: Path, out_dir: Path, fps: float, max_frames: int) -> list[Path]:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(int(round(native_fps / fps)), 1)
    written = []
    frame_idx = 0
    selected = 0
    while selected < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step == 0:
            out = out_dir / f"frame_{selected:06d}.jpg"
            cv2.imwrite(str(out), frame)
            written.append(out)
            selected += 1
        frame_idx += 1
    cap.release()
    return written


def _sample_imageio(video_path: Path, out_dir: Path, fps: float, max_frames: int) -> list[Path]:
    import imageio.v3 as iio
    from PIL import Image

    meta = iio.immeta(video_path)
    native_fps = float(meta.get("fps") or 25)
    step = max(int(round(native_fps / fps)), 1)
    written = []
    for frame_idx, frame in enumerate(iio.imiter(video_path)):
        if frame_idx % step != 0:
            continue
        out = out_dir / f"frame_{len(written):06d}.jpg"
        Image.fromarray(frame).save(out, quality=90)
        written.append(out)
        if len(written) >= max_frames:
            break
    return written


def _sample_ffmpeg(video_path: Path, out_dir: Path, fps: float, max_frames: int) -> list[Path]:
    pattern = out_dir / "frame_%06d.jpg"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps}",
        "-frames:v",
        str(max_frames),
        str(pattern),
    ]
    subprocess.run(cmd, check=True)
    return sorted(out_dir.glob("frame_*.jpg"))


def sample_video(video_path: Path, out_dir: Path, fps: float, max_frames: int, backend: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if backend == "cv2":
        return _sample_cv2(video_path, out_dir, fps, max_frames)
    if backend == "imageio":
        return _sample_imageio(video_path, out_dir, fps, max_frames)
    if backend == "ffmpeg":
        return _sample_ffmpeg(video_path, out_dir, fps, max_frames)
    raise ValueError(f"Unsupported backend: {backend}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract candidate frames from indexed videos.")
    parser.add_argument("--index", default="outputs/demo_index/dataset_index.json", type=Path)
    parser.add_argument("--out-dir", default="outputs/frames", type=Path)
    parser.add_argument("--fps", default=1.0, type=float)
    parser.add_argument("--max-frames", default=64, type=int)
    parser.add_argument("--backend", choices=["auto", "cv2", "imageio", "ffmpeg"], default="auto")
    args = parser.parse_args()

    backend = _backend() if args.backend == "auto" else args.backend
    if backend is None:
        raise RuntimeError(
            "No video decoding backend found. Install opencv-python or imageio, or add ffmpeg to PATH. "
            "You can still run build_dataset, simple_baseline on PNGs, and visualize."
        )

    index = _load_index(args.index)
    manifest = []
    for rec in index["records"]:
        if rec["sample_type"] != "video":
            continue
        video_path = Path(rec["source_path"])
        video_out = args.out_dir / rec["folder_event"] / video_path.stem
        frames = sample_video(video_path, video_out, args.fps, args.max_frames, backend)
        manifest.append(
            {
                "video_sample_id": rec["sample_id"],
                "video_path": str(video_path),
                "event": rec["event"],
                "frame_dir": str(video_out.resolve()),
                "frames": [str(p.resolve()) for p in frames],
            }
        )
        print(f"{rec['sample_id']}: wrote {len(frames)} frames")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "frames_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

