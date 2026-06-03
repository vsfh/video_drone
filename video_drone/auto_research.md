# Auto Research Agent Rules

修改时间：2026-06-03 11:45:00

## 目标

在 `/media/data1/feihong/video_drone_data/photos_2025-05-30_2025-06-30` 上做 few-shot 大模型图片检测。类别集合只能来自数据根目录下的一层文件夹名，例如 `construction_vehicle`、`straw_burning`、`waterside_garbage` 等。Agent 不允许自行增加、翻译、合并或删除类别。

## 数据规则

- 数据根目录的一层子文件夹就是事件类别。
- 每个类别优先读取 `cropped/` 作为 few-shot 示例图，因为它更接近事件目标。
- 每个类别优先读取 `original/` 作为待测图，因为它保留完整无人机画面。
- 默认每类取前 `3` 张示例图，待测图从该类排序后的第 `4` 张开始，避免示例图进入评估。
- 如果指定子目录不存在，则回退到类别文件夹内递归查找图片。
- 支持图片后缀：`.jpg`、`.jpeg`、`.png`、`.bmp`、`.webp`。

## 推理规则

- 默认模型为 `Qwen/Qwen3-VL-4B-Instruct`。
- 为降低 OOM 风险，`auto.py` 默认把类别分块，每次只让模型比较 `4` 个类别。
- 每次请求包含当前类别块的 few-shot 示例图和最后一张待测图。
- 模型必须只判断最后一张待测图，示例图只作为类别标准。
- 输出必须是 JSON，并包含：
  - `predicted_event`
  - `predicted_events`
  - `caption`
  - `evidence`
- `predicted_events` 的分数是独立置信度，不要求总和为 1。
- 如果模型输出了数据目录之外的类别，程序会丢弃该类别。

## 输出规则

`auto.py` 输出 `result.json`，核心字段包括：

- `dataset_root`
- `model_id`
- `shots_per_class`
- `classes`
- `results`

每条 `results` 记录包括：

- `sample_id`
- `source_path`
- `relative_path`
- `ground_truth_event`
- `predicted_event`
- `predicted_events`
- `caption`
- `evidence`
- `chunk_responses`

## 评估规则

`test.py` 读取 `result.json` 后，把文件夹名作为真实类别：

- `ground_truth_event` 来自样本所在类别文件夹。
- `predicted_event` 是模型预测类别。
- 计算每类 precision、recall、F1。
- 计算 macro precision、macro recall、macro F1。
- 计算 micro precision、micro recall、micro F1。
- 每次评估追加写入 `progress.csv`。
- 每次评估追加写入 `progress.txt`，并记录对应 `auto.py` 的最新简要修改记录和 sha256 短 hash。

## 运行方式

生成 `result.json`：

```bash
python -m video_drone.auto \
  --data-root /media/data1/feihong/video_drone_data/photos_2025-05-30_2025-06-30 \
  --output result.json \
  --model-id Qwen/Qwen3-VL-4B-Instruct \
  --shots-per-class 3 \
  --class-chunk-size 4 \
  --max-pixels 262144
```

评估并追加进度：

```bash
python -m video_drone.test \
  --result result.json \
  --progress-csv progress.csv \
  --progress-txt progress.txt
```

## 修改约定

后续每次修改 `auto.py` 时，必须同步更新 `AUTO_RESEARCH_CHANGELOG` 的最后一条或追加新条目。`test.py` 会读取该记录并写入 `progress.txt`，用于追踪每次实验对应的代码版本。
