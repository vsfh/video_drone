# AutoResearch Rules

Goal:
Improve few-shot VLM detection of UAV commercial-event images on:
/media/data1/feihong/video_drone_data/photos_2025-05-30_2025-06-30

Editable files:
- video_drone/auto.py

Do not edit:
- dataset files under /media/data1/feihong/video_drone_data/photos_2025-05-30_2025-06-30
- model cache files under /media/data1/feihong/hf_cache
- evaluation protocol: folder name is the ground-truth class
- result.json by hand
- progress.csv by hand
- progress.txt by hand

Class space:
Read all first-level folders under:
/media/data1/feihong/video_drone_data/photos_2025-05-30_2025-06-30

Only these folder names are valid event classes.
Do not add, translate, merge, rename, or delete classes.

Model:
Load Qwen3-VL-4B-Instruct only from the local cache path:
/media/data1/feihong/hf_cache/models--Qwen--Qwen3-VL-4B-Instruct

Use local_files_only=True.
Do not download model files during experiments.

Experiment command:
python -m video_drone.auto \
  --data-root /media/data1/feihong/video_drone_data/photos_2025-05-30_2025-06-30 \
  --output result.json \
  --shots-per-class 1 \
  --class-chunk-size 12 \
  --max-pixels 262144 \
  --progress auto

Evaluation command:
python -m video_drone.test \
  --result result.json \
  --progress-csv progress.csv \
  --progress-txt progress.txt

Metric:
Read progress.csv after running the evaluation command.
The main objective is "macro_f1".
Higher is better.

Secondary metrics:
- macro_recall
- micro_f1
- accuracy

Output:
auto.py must write result.json.
test.py must append one row to progress.csv.
test.py must append the matching auto.py change note to progress.txt.

Rules:
1. Make exactly one meaningful change to auto.py per experiment.
2. Update AUTO_RESEARCH_CHANGELOG in auto.py for that change.
3. Run the experiment command.
4. Run the evaluation command.
5. Compare macro_f1 with the previous best in progress.csv.
6. Keep the change only if macro_f1 improves.
7. Revert failed changes.
8. Do not manually edit result.json, progress.csv, or progress.txt.
9. If an experiment fails because of OOM, reduce class_chunk_size, max_pixels, or shots_per_class.
10. If an experiment fails before producing result.json, record the failure reason in progress.txt.
