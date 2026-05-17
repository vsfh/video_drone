# video_drone

Utilities for preparing the UAV commercial-event demo data for zero-shot baselines, keyframe selection, VLM captioning, and visualization.

The code is intentionally lightweight. It can index the current `../demo` folder and visualize PNG examples without video decoding dependencies. Frame extraction is available when a video backend such as OpenCV, imageio, or ffmpeg is installed.

## Quick Start

From this folder:

```powershell
& 'C:\Users\yc58103\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m video_drone.build_dataset --demo-root ..\demo --out-dir outputs\demo_index
& 'C:\Users\yc58103\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m video_drone.simple_baseline --index outputs\demo_index\dataset_index.json --out-dir outputs\simple_baseline
& 'C:\Users\yc58103\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m video_drone.visualize --index outputs\demo_index\dataset_index.json --predictions outputs\simple_baseline\image_predictions.jsonl --out-dir outputs\visual_report
```

Open `outputs/visual_report/index.html` in a browser to inspect examples and predictions.

## Optional Frame Extraction

Frame extraction needs one available backend:

- `opencv-python` (`cv2`)
- `imageio`
- `ffmpeg` command line

```powershell
& 'C:\Users\yc58103\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m video_drone.sample_frames --index outputs\demo_index\dataset_index.json --out-dir outputs\frames --fps 1 --max-frames 64
```

Then run the baseline on frames:

```powershell
& 'C:\Users\yc58103\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m video_drone.simple_baseline --index outputs\demo_index\dataset_index.json --frame-root outputs\frames --out-dir outputs\simple_baseline_frames
```

## Keyframe Extraction for One MP4

The dedicated extractor reads one MP4/video file, saves keyframes into a new folder, and writes a manifest. File names include the event name, source video, timestamp, frame index, and extraction reason.

Dependencies were installed locally into:

```text
C:\Users\yc58103\.cache\codex-video-drone-pythonlibs
```

Run:

```powershell
& 'C:\Users\yc58103\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m video_drone.extract_keyframes --mp4-path '..\demo\疑似烟火\疑似烟火.mp4' --out-dir outputs\keyframes_single --sample-fps 1 --scene-threshold 0.12 --min-gap-sec 1 --min-brightness 8 --skip-start-sec 20 --skip-end-sec 5
```

Outputs:

- `outputs/keyframes_single/<event>/<video_stem>/*.jpg`
- `outputs/keyframes_single/keyframes_manifest.csv`
- `outputs/keyframes_single/keyframes_manifest.json`
- `outputs/keyframes_single/keyframes_manifest.jsonl`

To process every MP4 under `../demo`, wrap the single-file command in a PowerShell loop:

```powershell
Get-ChildItem ..\demo -Recurse -Filter *.mp4 | ForEach-Object {
  & 'C:\Users\yc58103\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m video_drone.extract_keyframes --mp4-path $_.FullName --out-dir outputs\keyframes_batch --sample-fps 1 --scene-threshold 0.12 --min-gap-sec 1 --min-brightness 8 --skip-start-sec 20 --skip-end-sec 5
}
```

## Prediction Format

All future methods should write JSONL records like:

```json
{
  "sample_id": "疑似烟火/疑似烟火.mp4#frame_000001",
  "source_path": "outputs/frames/疑似烟火/疑似烟火/frame_000001.jpg",
  "method": "qwen2.5-vl-zero-shot",
  "predicted_events": [{"event": "疑似烟火", "score": 0.82}],
  "keyframe_rank": 1,
  "caption": "画面中村庄附近出现疑似烟雾，存在火情风险。",
  "evidence": "烟雾区域位于画面右侧房屋附近。",
  "department": ["应急管理", "消防安全"]
}
```

This shared schema lets keyframe selectors, captioning VLMs, and open-vocabulary detectors be visualized together.
