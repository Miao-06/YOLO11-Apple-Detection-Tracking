from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# 1. 项目路径
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VIDEO_PATH = (
    PROJECT_ROOT
    / "data"
    / "vidio"
    / "final.mp4"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "detect"
    / "baseline_yolo11s_640"
    / "weights"
    / "best.pt"
)


# ============================================================
# 2. 输出目录
# ============================================================

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "hard_negative_candidates"
)

FALSE_POSITIVE_DIR = (
    OUTPUT_ROOT
    / "false_positive_candidates"
)

BACKGROUND_DIR = (
    OUTPUT_ROOT
    / "background_candidates"
)


# ============================================================
# 3. 参数
# ============================================================

# 每隔多少秒抽一次视频帧
SAMPLE_INTERVAL_SEC = 0.5

# 用一个较低的阈值扫视频
# 目的：尽量把真正的苹果提前排除掉
DETECT_CONF = 0.10

# false-positive 候选置信度范围
FP_CONF_MIN = 0.25
FP_CONF_MAX = 0.80

# 每一帧最多保存多少个“疑似误检区域”
MAX_FP_PER_FRAME = 3

# 背景裁剪尺寸
BACKGROUND_CROP_SIZE = 384

# 背景滑窗步长
BACKGROUND_STRIDE = 192

# 每一帧最多保存背景区域
MAX_BACKGROUND_PER_FRAME = 2

# 检测框扩张比例
# 背景区域必须离苹果远一点
BOX_EXPAND_RATIO = 0.40

# 红色像素比例阈值
# 超过这个比例的区域暂时不作为背景候选
MAX_RED_RATIO = 0.012

# 背景复杂度阈值
# 太平坦的白墙意义不大
MIN_LAPLACIAN_VAR = 25.0

# 去重阈值
# 越接近 1，表示图片越相似
DEDUP_SIMILARITY = 0.965


# ============================================================
# 4. 创建目录
# ============================================================

def prepare_dirs():

    FALSE_POSITIVE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    BACKGROUND_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# 5. 判断两个矩形是否相交
# ============================================================

def boxes_intersect(box1, box2):

    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    if x1_max <= x2_min:
        return False

    if x2_max <= x1_min:
        return False

    if y1_max <= y2_min:
        return False

    if y2_max <= y1_min:
        return False

    return True


# ============================================================
# 6. 扩张检测框
# ============================================================

def expand_box(
    box,
    frame_width,
    frame_height,
    ratio=0.4
):

    x1, y1, x2, y2 = box

    width = x2 - x1
    height = y2 - y1

    dx = width * ratio
    dy = height * ratio

    x1 = max(
        0,
        int(x1 - dx)
    )

    y1 = max(
        0,
        int(y1 - dy)
    )

    x2 = min(
        frame_width,
        int(x2 + dx)
    )

    y2 = min(
        frame_height,
        int(y2 + dy)
    )

    return (
        x1,
        y1,
        x2,
        y2
    )


# ============================================================
# 7. 计算红色像素比例
# ============================================================

def calculate_red_ratio(image):

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    # 红色有两个 Hue 区域
    lower_red_1 = np.array(
        [0, 80, 60]
    )

    upper_red_1 = np.array(
        [12, 255, 255]
    )

    lower_red_2 = np.array(
        [165, 80, 60]
    )

    upper_red_2 = np.array(
        [179, 255, 255]
    )

    mask1 = cv2.inRange(
        hsv,
        lower_red_1,
        upper_red_1
    )

    mask2 = cv2.inRange(
        hsv,
        lower_red_2,
        upper_red_2
    )

    mask = cv2.bitwise_or(
        mask1,
        mask2
    )

    red_pixels = np.count_nonzero(
        mask
    )

    total_pixels = (
        image.shape[0]
        *
        image.shape[1]
    )

    if total_pixels == 0:
        return 1.0

    return (
        red_pixels
        /
        total_pixels
    )


# ============================================================
# 8. 背景复杂度
# ============================================================

def calculate_laplacian_variance(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    value = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    return float(value)


# ============================================================
# 9. 图像特征，用于去重
# ============================================================

def make_feature(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.resize(
        gray,
        (32, 32)
    )

    return gray.astype(
        np.float32
    )


# ============================================================
# 10. 两张图简单相似度
# ============================================================

def feature_similarity(
    feature1,
    feature2
):

    difference = np.mean(
        np.abs(
            feature1 - feature2
        )
    )

    similarity = (
        1.0
        -
        difference / 255.0
    )

    return float(similarity)


# ============================================================
# 11. 判断是否和已有候选过于相似
# ============================================================

def is_duplicate(
    feature,
    saved_features
):

    for old_feature in saved_features:

        similarity = feature_similarity(
            feature,
            old_feature
        )

        if similarity >= DEDUP_SIMILARITY:
            return True

    return False


# ============================================================
# 12. 从检测框周围裁一个带上下文区域
# ============================================================

def crop_detection_context(
    frame,
    box,
    scale=2.0
):

    frame_height, frame_width = (
        frame.shape[:2]
    )

    x1, y1, x2, y2 = box

    bbox_width = x2 - x1
    bbox_height = y2 - y1

    center_x = (
        x1 + x2
    ) / 2

    center_y = (
        y1 + y2
    ) / 2

    crop_size = int(
        max(
            bbox_width,
            bbox_height
        )
        *
        scale
    )

    crop_size = max(
        crop_size,
        160
    )

    crop_x1 = int(
        center_x
        -
        crop_size / 2
    )

    crop_y1 = int(
        center_y
        -
        crop_size / 2
    )

    crop_x2 = (
        crop_x1
        +
        crop_size
    )

    crop_y2 = (
        crop_y1
        +
        crop_size
    )

    if crop_x1 < 0:

        crop_x2 -= crop_x1
        crop_x1 = 0

    if crop_y1 < 0:

        crop_y2 -= crop_y1
        crop_y1 = 0

    if crop_x2 > frame_width:

        offset = (
            crop_x2
            -
            frame_width
        )

        crop_x1 = max(
            0,
            crop_x1 - offset
        )

        crop_x2 = frame_width

    if crop_y2 > frame_height:

        offset = (
            crop_y2
            -
            frame_height
        )

        crop_y1 = max(
            0,
            crop_y1 - offset
        )

        crop_y2 = frame_height

    crop = frame[
        crop_y1:crop_y2,
        crop_x1:crop_x2
    ].copy()

    return crop


# ============================================================
# 13. 找背景候选
# ============================================================

def find_background_candidates(
    frame,
    expanded_boxes
):

    frame_height, frame_width = (
        frame.shape[:2]
    )

    crop_size = min(
        BACKGROUND_CROP_SIZE,
        frame_width,
        frame_height
    )

    candidates = []

    for y1 in range(
        0,
        max(
            1,
            frame_height
            -
            crop_size
            +
            1
        ),
        BACKGROUND_STRIDE
    ):

        for x1 in range(
            0,
            max(
                1,
                frame_width
                -
                crop_size
                +
                1
            ),
            BACKGROUND_STRIDE
        ):

            x2 = (
                x1
                +
                crop_size
            )

            y2 = (
                y1
                +
                crop_size
            )

            crop_box = (
                x1,
                y1,
                x2,
                y2
            )

            # --------------------------------------------
            # 与任何苹果检测区域相交就不要
            # --------------------------------------------

            collision = False

            for detection_box in expanded_boxes:

                if boxes_intersect(
                    crop_box,
                    detection_box
                ):

                    collision = True
                    break

            if collision:
                continue

            crop = frame[
                y1:y2,
                x1:x2
            ]

            if crop.size == 0:
                continue

            # --------------------------------------------
            # 防止里面可能还有漏掉的红苹果
            # --------------------------------------------

            red_ratio = (
                calculate_red_ratio(
                    crop
                )
            )

            if (
                red_ratio
                >
                MAX_RED_RATIO
            ):

                continue

            # --------------------------------------------
            # 白墙等太简单区域价值不大
            # --------------------------------------------

            texture_score = (
                calculate_laplacian_variance(
                    crop
                )
            )

            if (
                texture_score
                <
                MIN_LAPLACIAN_VAR
            ):

                continue

            candidates.append(
                (
                    texture_score,
                    red_ratio,
                    crop.copy(),
                    crop_box
                )
            )

    # 优先保存纹理复杂的背景
    candidates.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return candidates


# ============================================================
# 14. 主程序
# ============================================================

def main():

    prepare_dirs()

    print("=" * 70)
    print("Hard-Negative Candidate Extraction")
    print("=" * 70)

    print(f"Video : {VIDEO_PATH}")
    print(f"Model : {MODEL_PATH}")
    print()

    if not VIDEO_PATH.exists():

        raise FileNotFoundError(
            f"找不到视频:\n{VIDEO_PATH}"
        )

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"找不到模型:\n{MODEL_PATH}"
        )

    model = YOLO(
        str(MODEL_PATH)
    )

    capture = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not capture.isOpened():

        raise RuntimeError(
            "无法打开视频"
        )

    fps = capture.get(
        cv2.CAP_PROP_FPS
    )

    total_frames = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if fps <= 0:
        fps = 30.0

    frame_interval = max(
        1,
        int(
            fps
            *
            SAMPLE_INTERVAL_SEC
        )
    )

    print(
        f"FPS               : {fps:.2f}"
    )

    print(
        f"Total frames      : {total_frames}"
    )

    print(
        f"Sample every      : {frame_interval} frames"
    )

    print()

    frame_index = 0

    saved_fp = 0
    saved_background = 0

    fp_features = []
    background_features = []

    while True:

        success, frame = (
            capture.read()
        )

        if not success:
            break

        # --------------------------------------------
        # 不是抽样帧则跳过
        # --------------------------------------------

        if (
            frame_index
            %
            frame_interval
            != 0
        ):

            frame_index += 1
            continue

        frame_height, frame_width = (
            frame.shape[:2]
        )

        # --------------------------------------------
        # YOLO 低阈值检测
        # --------------------------------------------

        result = model.predict(
            source=frame,
            imgsz=640,
            conf=DETECT_CONF,
            iou=0.7,
            device=0,
            verbose=False
        )[0]

        detection_boxes = []
        detections = []

        if (
            result.boxes
            is not None
        ):

            for box in result.boxes:

                xyxy = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                )

                confidence = float(
                    box.conf[0]
                )

                x1, y1, x2, y2 = (
                    xyxy.tolist()
                )

                detection_box = (
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2)
                )

                detection_boxes.append(
                    detection_box
                )

                detections.append(
                    (
                        detection_box,
                        confidence
                    )
                )

        # ====================================================
        # A. 模型“疑似误检”候选
        # ====================================================

        fp_candidates = []

        for (
            detection_box,
            confidence
        ) in detections:

            if not (
                FP_CONF_MIN
                <= confidence
                <= FP_CONF_MAX
            ):
                continue

            crop = crop_detection_context(
                frame,
                detection_box,
                scale=2.2
            )

            if crop.size == 0:
                continue

            feature = make_feature(
                crop
            )

            if is_duplicate(
                feature,
                fp_features
            ):
                continue

            fp_candidates.append(
                (
                    confidence,
                    crop,
                    feature
                )
            )

        # 更高置信度优先看
        fp_candidates.sort(
            key=lambda x: x[0],
            reverse=True
        )

        for (
            confidence,
            crop,
            feature
        ) in fp_candidates[
            :MAX_FP_PER_FRAME
        ]:

            filename = (
                f"fp_"
                f"f{frame_index:06d}_"
                f"conf{confidence:.2f}.jpg"
            )

            output_path = (
                FALSE_POSITIVE_DIR
                /
                filename
            )

            cv2.imwrite(
                str(output_path),
                crop
            )

            fp_features.append(
                feature
            )

            saved_fp += 1

        # ====================================================
        # B. 纯背景候选
        # ====================================================

        expanded_boxes = []

        for box in detection_boxes:

            expanded_boxes.append(
                expand_box(
                    box,
                    frame_width,
                    frame_height,
                    BOX_EXPAND_RATIO
                )
            )

        background_candidates = (
            find_background_candidates(
                frame,
                expanded_boxes
            )
        )

        background_saved_this_frame = 0

        for (
            texture_score,
            red_ratio,
            crop,
            crop_box
        ) in background_candidates:

            if (
                background_saved_this_frame
                >=
                MAX_BACKGROUND_PER_FRAME
            ):
                break

            feature = make_feature(
                crop
            )

            if is_duplicate(
                feature,
                background_features
            ):
                continue

            x1, y1, x2, y2 = (
                crop_box
            )

            filename = (
                f"bg_"
                f"f{frame_index:06d}_"
                f"x{x1}_y{y1}.jpg"
            )

            output_path = (
                BACKGROUND_DIR
                /
                filename
            )

            cv2.imwrite(
                str(output_path),
                crop
            )

            background_features.append(
                feature
            )

            saved_background += 1
            background_saved_this_frame += 1

        print(
            f"Frame {frame_index:6d}"
            f" | detections={len(detections):2d}"
            f" | FP total={saved_fp:3d}"
            f" | BG total={saved_background:3d}"
        )

        frame_index += 1

    capture.release()

    print()
    print("=" * 70)
    print("Extraction completed")
    print("=" * 70)

    print(
        f"False-positive candidates : "
        f"{saved_fp}"
    )

    print(
        f"Background candidates     : "
        f"{saved_background}"
    )

    print()

    print(
        "False-positive candidates:"
    )

    print(
        FALSE_POSITIVE_DIR
    )

    print()

    print(
        "Background candidates:"
    )

    print(
        BACKGROUND_DIR
    )

    print("=" * 70)


# ============================================================
# 15. Windows 主入口
# ============================================================

if __name__ == "__main__":
    main()

    