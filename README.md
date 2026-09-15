# YOLO11 Apple Detection & Tracking

基于 Ultralytics YOLO11 的人工苹果目标检测与视频累计计数项目。

## 1. 项目简介

本项目针对模型果树场景中的红色人工苹果，实现：

- 苹果目标检测
- 当前帧苹果数量统计
- 重复框 / 包含框过滤
- 遮挡目标短时稳定跟踪
- 相机运动补偿
- 跨帧 Track 关联
- 视频累计苹果数量统计
- Tracklet 全局拼接

整体流程：

```text
Video
  ↓
YOLO11 Apple Detection
  ↓
Duplicate Box Removal
  ↓
Camera Motion Estimation
  ↓
Apple Tracking
  ↓
Current Apple Count
  ↓
Global Tracklet Stitching
  ↓
Total Apple Count
  ↓
Output Video
```

当前最终检测模型采用：

```text
YOLO11s
imgsz = 960
Hard Negative Samples
```

最终权重：

```text
weights/exp003_best.pt
```

---

## 2. 环境

```text
Windows 64-bit
Python 3.10.20
PyTorch 2.1.0 + CUDA 12.1
TorchVision 0.16.0
Ultralytics 8.4.7
OpenCV 4.9
NumPy 1.26.4
GPU: NVIDIA GeForce RTX 4060 Laptop GPU (8 GB)
```

安装依赖：

```bash
pip install -r requirements.txt
```

---

## 3. 项目结构

```text
YOLO11-Apple-Detection-Tracking/
│
├── analysis/
├── configs/
├── data/
│   ├── dataset.yaml
│   └── split/
│       ├── train/
│       │   ├── images/
│       │   └── labels/
│       └── val/
│           ├── images/
│           └── labels/
│
├── models/
├── predict/
│   ├── track_videos_apple_scene_tracker.py
│   ├── track_videos_apple_scene_tracker_v2.py
│   └── track_videos_apple_scene_tracker_v3.py
│
├── tools/
├── train/
│   └── train.py
│
├── weights/
│   └── exp003_best.pt
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 4. 数据集

类别：

```text
0: pingguo
```

数据来自两个连续视频序列。

为降低相邻视频帧随机划分带来的数据泄漏风险，数据集按照来源视频内部的时间顺序进行训练集 / 验证集划分。

当前提交数据：

```text
data/split/train/images
data/split/train/labels
data/split/val/images
data/split/val/labels
```

训练集中加入了 Hard Negative，用于降低红色背景物、反光区域、地面红色小块等造成的误检。

数据配置：

```text
data/dataset.yaml
```

---

## 5. 主要实验

### Exp001

```text
YOLO11s
imgsz = 640

Precision   ≈ 0.906
Recall      ≈ 0.871
mAP50       ≈ 0.928
mAP50-95    ≈ 0.599
```

### Exp002

```text
YOLO11s
imgsz = 640
+ Hard Negative

Precision   ≈ 0.926
Recall      ≈ 0.866
mAP50       ≈ 0.929
mAP50-95    ≈ 0.602
```

### Exp003（最终模型）

```text
YOLO11s
imgsz = 960
+ Hard Negative

Precision   = 0.906
Recall      = 0.926
mAP50       = 0.947
mAP50-95    = 0.622
```

最终模型：

```text
weights/exp003_best.pt
```

---

## 6. 训练

训练入口：

```text
train/train.py
```

最终实验配置：

```text
configs/Exp003_HighRes_HardNegative_YOLO11s_960.yaml
```

运行：

```bash
python train/train.py
```

---

## 7. Tracking 版本

### V1

```text
predict/track_videos_apple_scene_tracker.py
```

主要思路：

```text
YOLO Detection
+
Camera Motion Compensation
+
Scene Coordinate Tracking
```

相机运动估计使用：

```text
Shi-Tomasi Feature Detection
+
Lucas-Kanade Optical Flow
+
RANSAC Affine Motion Estimation
```

---

### V2

```text
predict/track_videos_apple_scene_tracker_v2.py
```

在 V1 基础上增加：

```text
Active Track Matching
+
Dormant Track Re-Association
```

用于减少苹果因遮挡或镜头移动导致的 Track ID 断裂。

---

### V3（推荐）

```text
predict/track_videos_apple_scene_tracker_v3.py
```

V3 使用两遍处理：

```text
PASS 1
YOLO + Camera Motion + Online Tracking
        ↓
生成 Tracklets
        ↓
Global Tracklet Stitching
        ↓
PASS 2
重新生成最终视频
```

通过对时间不重叠、场景位置接近、尺寸相似的 Tracklets 进行全局拼接，进一步降低：

```text
Track Fragmentation
ID Switching
Repeated Counting
```

---

## 8. 视频推理

推荐运行：

```bash
python predict/track_videos_apple_scene_tracker_v3.py
```

默认加载：

```text
weights/exp003_best.pt
```

最终视频左上角显示：

```text
Current Apples
Total Apples
```

其中：

- `Current Apples`：当前帧中的苹果数量
- `Total Apples`：视频中累计识别到的不同苹果数量

---

## 9. Tracking 设计说明

本项目没有直接使用通用行人 ReID 模型作为最终方案。

本任务的场景特点是：

```text
Apple: mostly static
Camera: moving
```

因此重点利用：

```text
Camera Motion Estimation
        ↓
Scene Coordinate Compensation
        ↓
Geometric Association
        ↓
Track Lifecycle
        ↓
Global Tracklet Stitching
```

整体设计借鉴 Multi-Object Tracking 中的 Track ID、Data Association、Track Lifecycle 和 Re-Association 思想，并针对固定人工苹果 + 移动相机场景进行适配。

---

## 10. 推荐入口

最终模型：

```text
weights/exp003_best.pt
```

推荐脚本：

```bash
python predict/track_videos_apple_scene_tracker_v3.py
```
