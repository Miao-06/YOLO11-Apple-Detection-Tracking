from pathlib import Path
import math

import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# 1. 项目根目录
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# 2. 模型
#
# 固定使用当前效果最好的 Exp003 best.pt。
# 不更换检测模型。
# ============================================================

MODEL_PATH = (
    PROJECT_ROOT
    / "weights"
    / "exp003_best.pt"
)


# ============================================================
# 3. 视频目录
# ============================================================

VIDEO_DIR = (
    PROJECT_ROOT
    / "data"
    / "vidio"
)

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
}


# ============================================================
# 4. 输出目录
# ============================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "runs"
    / "video_final"
    / "Exp003_AppleSceneTracker"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 5. YOLO 参数
#
# 保持你当前已经满意的视频检测参数。
# ============================================================

IMG_SIZE = 960
DETECT_CONF = 0.47
IOU_THRES = 0.70
MAX_DET = 300
DEVICE = 0


# ============================================================
# 6. 原始检测去重参数
#
# 与之前版本保持同样思想：
# 1. 小框基本被大框包含 -> 删除小框
# 2. 两框高度重叠 -> 只保留高置信度框
# ============================================================

CONTAINMENT_THRESHOLD = 0.85
DUPLICATE_IOU_THRESHOLD = 0.75


# ============================================================
# 7. 苹果专用 Tracker 参数
#
# 设计思想：
#
# 不是使用通用行人 ReID。
#
# 苹果固定在树上，相机移动，所以：
#
# 当前帧检测框
#     ↓
# 相机运动估计
#     ↓
# 映射到“参考场景坐标”
#     ↓
# 与历史 Apple Track 做几何匹配
#     ↓
# 维持同一个 Track ID
#
# ============================================================

# 新目标至少连续匹配多少次才进入“累计数量”。
# 第一帧是特殊情况：直接确认，避免第一帧累计为 0。
N_INIT = 2

# 当前帧计数允许短暂漏检几帧。
CURRENT_COUNT_TTL = 3

# 框最多残留显示几帧。
BOX_TTL = 1

# 活跃匹配允许丢失多少帧。
# 超过后不再参与普通短时匹配，
# 但历史 Track 仍保留，可用于“重新进入画面”时的累计去重。
ACTIVE_MAX_MISSED = 15

# 场景坐标匹配阈值。
# 以当前检测框对角线为尺度进行归一化。
SCENE_DISTANCE_FACTOR = 1.50

# 当前图像坐标中的辅助中心距离阈值。
IMAGE_DISTANCE_FACTOR = 1.20

# 如果预测框和当前框 IoU 达到这个值，可增强匹配可信度。
TRACK_IOU_THRESHOLD = 0.05

# 尺寸差异限制。
# max(area1, area2) / min(area1, area2) 不超过该值。
MAX_AREA_RATIO = 3.0

# 匹配代价权重。
SCENE_DISTANCE_WEIGHT = 3.0
IMAGE_DISTANCE_WEIGHT = 1.0
IOU_WEIGHT = 1.5
SIZE_WEIGHT = 0.5

# 框平滑。
BOX_SMOOTH_ALPHA = 0.75

# 场景坐标更新的平滑系数。
# 相机估计存在误差，因此不直接覆盖历史 scene_center。
SCENE_CENTER_ALPHA = 0.25


# ============================================================
# 8. 相机运动估计参数
#
# 使用 Shi-Tomasi + LK 光流 + RANSAC 仿射变换。
#
# 这里没有使用 DeepSORT 的 ReID。
# 对本项目而言，“相机移动”比“苹果自身运动”更重要。
# ============================================================

MAX_CORNERS = 500
QUALITY_LEVEL = 0.01
MIN_DISTANCE = 8
BLOCK_SIZE = 7

LK_WIN_SIZE = (21, 21)
LK_MAX_LEVEL = 3

MIN_GOOD_POINTS = 20

RANSAC_REPROJ_THRESHOLD = 3.0

# 如果某一帧相机运动估计失败，
# 先假设这一帧运动较小，沿用上一累计变换。
MOTION_DEBUG = False


# ============================================================
# 9. 基础几何函数
# ============================================================

def box_area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def box_center(box):
    return (
        (box[0] + box[2]) / 2.0,
        (box[1] + box[3]) / 2.0,
    )


def box_diagonal(box):
    w = max(1.0, box[2] - box[0])
    h = max(1.0, box[3] - box[1])
    return math.sqrt(w * w + h * h)


def intersection_area(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)

    return w * h


def calculate_iou(box1, box2):
    inter = intersection_area(box1, box2)

    if inter <= 0:
        return 0.0

    area1 = box_area(box1)
    area2 = box_area(box2)

    union = area1 + area2 - inter

    if union <= 0:
        return 0.0

    return inter / union


def point_distance(p1, p2):
    return math.sqrt(
        (p1[0] - p2[0]) ** 2
        +
        (p1[1] - p2[1]) ** 2
    )


def to_homogeneous_affine(matrix_2x3):
    """2x3 仿射矩阵 -> 3x3 齐次矩阵。"""

    matrix_3x3 = np.eye(
        3,
        dtype=np.float64
    )

    matrix_3x3[:2, :] = matrix_2x3

    return matrix_3x3


def transform_point(point, matrix_3x3):
    """使用 3x3 变换矩阵映射一个二维点。"""

    p = np.array(
        [
            point[0],
            point[1],
            1.0
        ],
        dtype=np.float64
    )

    q = matrix_3x3 @ p

    if abs(q[2]) < 1e-9:
        return point

    return (
        float(q[0] / q[2]),
        float(q[1] / q[2])
    )


def transform_box(box, matrix_3x3):
    """
    将框四个角经过仿射变换后，
    重新求一个轴对齐 bbox。
    """

    corners = [
        (box[0], box[1]),
        (box[2], box[1]),
        (box[2], box[3]),
        (box[0], box[3]),
    ]

    transformed = [
        transform_point(
            point,
            matrix_3x3
        )
        for point in corners
    ]

    xs = [p[0] for p in transformed]
    ys = [p[1] for p in transformed]

    return [
        min(xs),
        min(ys),
        max(xs),
        max(ys),
    ]


# ============================================================
# 10. Detection 去重
# ============================================================

def remove_duplicate_detections(detections):

    if len(detections) <= 1:
        return detections

    # 置信度高的优先保留。
    detections = sorted(
        detections,
        key=lambda item: item["conf"],
        reverse=True
    )

    removed = set()

    for i in range(len(detections)):

        if i in removed:
            continue

        box1 = detections[i]["box"]
        area1 = box_area(box1)

        for j in range(i + 1, len(detections)):

            if j in removed:
                continue

            box2 = detections[j]["box"]
            area2 = box_area(box2)

            if area1 <= 0 or area2 <= 0:
                continue

            inter = intersection_area(
                box1,
                box2
            )

            if inter <= 0:
                continue

            smaller_area = min(
                area1,
                area2
            )

            containment = (
                inter
                /
                smaller_area
            )

            # ----------------------------------------------------
            # A. 包含关系：
            # 用户要求删除被包含的小框。
            # ----------------------------------------------------

            if containment >= CONTAINMENT_THRESHOLD:

                if area1 >= area2:
                    removed.add(j)

                else:
                    removed.add(i)
                    break

            # ----------------------------------------------------
            # B. 高 IoU：
            # 按置信度排序后，删除后面的低置信度框。
            # ----------------------------------------------------

            else:

                iou = calculate_iou(
                    box1,
                    box2
                )

                if iou >= DUPLICATE_IOU_THRESHOLD:
                    removed.add(j)

    return [
        detection

        for index, detection
        in enumerate(detections)

        if index not in removed
    ]


# ============================================================
# 11. YOLO 检测
# ============================================================

def detect_frame(
    model,
    frame
):

    result = model.predict(
        source=frame,
        imgsz=IMG_SIZE,
        conf=DETECT_CONF,
        iou=IOU_THRES,
        max_det=MAX_DET,
        device=DEVICE,
        verbose=False,
    )[0]

    detections = []

    if (
        result.boxes is None
        or
        len(result.boxes) == 0
    ):
        return detections

    boxes = (
        result.boxes
        .xyxy
        .cpu()
        .numpy()
    )

    confidences = (
        result.boxes
        .conf
        .cpu()
        .numpy()
    )

    for box, confidence in zip(
        boxes,
        confidences
    ):

        detections.append(
            {
                "box": [
                    float(box[0]),
                    float(box[1]),
                    float(box[2]),
                    float(box[3]),
                ],

                "conf":
                    float(confidence),
            }
        )

    return remove_duplicate_detections(
        detections
    )


# ============================================================
# 12. 相机运动估计
#
# 输入：
# prev_gray -> 上一帧灰度图
# curr_gray -> 当前帧灰度图
#
# 输出：
# prev_to_curr:
# 上一帧图像坐标 -> 当前帧图像坐标 的 3x3 仿射矩阵
#
# 这一步是整个“苹果专用 Tracker”的核心。
# ============================================================

def estimate_camera_motion(
    prev_gray,
    curr_gray
):

    identity = np.eye(
        3,
        dtype=np.float64
    )

    points_prev = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=MAX_CORNERS,
        qualityLevel=QUALITY_LEVEL,
        minDistance=MIN_DISTANCE,
        blockSize=BLOCK_SIZE
    )

    if (
        points_prev is None
        or
        len(points_prev) < MIN_GOOD_POINTS
    ):
        return identity, 0

    points_curr, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        curr_gray,
        points_prev,
        None,
        winSize=LK_WIN_SIZE,
        maxLevel=LK_MAX_LEVEL,
        criteria=(
            cv2.TERM_CRITERIA_EPS
            |
            cv2.TERM_CRITERIA_COUNT,
            30,
            0.01
        )
    )

    if (
        points_curr is None
        or
        status is None
    ):
        return identity, 0

    status = status.reshape(-1).astype(bool)

    good_prev = (
        points_prev.reshape(-1, 2)[status]
    )

    good_curr = (
        points_curr.reshape(-1, 2)[status]
    )

    if len(good_prev) < MIN_GOOD_POINTS:
        return identity, len(good_prev)

    matrix_2x3, inliers = cv2.estimateAffinePartial2D(
        good_prev,
        good_curr,
        method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_REPROJ_THRESHOLD
    )

    if matrix_2x3 is None:
        return identity, len(good_prev)

    matrix_3x3 = to_homogeneous_affine(
        matrix_2x3
    )

    inlier_count = 0

    if inliers is not None:
        inlier_count = int(
            inliers.sum()
        )

    return (
        matrix_3x3,
        inlier_count
    )


# ============================================================
# 13. 将当前检测映射到“参考场景坐标”
#
# cumulative_ref_to_current：
#   第 0 帧坐标 -> 当前帧坐标
#
# 取逆矩阵后：
#   当前帧坐标 -> 第 0 帧参考坐标
#
# 对静止在树上的苹果来说：
# 同一个苹果在参考场景坐标中应该大致稳定。
# ============================================================

def add_scene_coordinates(
    detections,
    cumulative_ref_to_current
):

    try:
        current_to_ref = np.linalg.inv(
            cumulative_ref_to_current
        )

    except np.linalg.LinAlgError:
        current_to_ref = np.eye(
            3,
            dtype=np.float64
        )

    for detection in detections:

        center = box_center(
            detection["box"]
        )

        detection["center"] = center

        detection["scene_center"] = (
            transform_point(
                center,
                current_to_ref
            )
        )

        detection["diag"] = box_diagonal(
            detection["box"]
        )

        detection["area"] = box_area(
            detection["box"]
        )


# ============================================================
# 14. Apple Scene Tracker
# ============================================================

class AppleSceneTracker:

    def __init__(self):

        self.tracks = {}

        self.next_track_id = 0

        # 只要一个 Track 曾经确认过，
        # 它的 ID 就永久进入 counted_ids。
        self.counted_ids = set()


    # --------------------------------------------------------
    # 创建 Track
    # --------------------------------------------------------

    def _create_track(
        self,
        detection,
        frame_index,
        confirm_immediately=False
    ):

        track_id = self.next_track_id

        self.next_track_id += 1

        confirmed = bool(
            confirm_immediately
        )

        self.tracks[
            track_id
        ] = {

            "id":
                track_id,

            "box":
                detection["box"],

            "conf":
                detection["conf"],

            "center":
                detection["center"],

            "scene_center":
                detection["scene_center"],

            "diag":
                detection["diag"],

            "area":
                detection["area"],

            "hits":
                1,

            "missed":
                0,

            "last_frame":
                frame_index,

            "confirmed":
                confirmed,

            # 是否已经加入累计计数。
            "counted":
                False,
        }

        if confirmed:

            self._count_track(
                track_id
            )

        return track_id


    # --------------------------------------------------------
    # 一个 Track 首次确认时，累计数量 +1。
    # --------------------------------------------------------

    def _count_track(
        self,
        track_id
    ):

        track = self.tracks[
            track_id
        ]

        if track["counted"]:
            return False

        track["counted"] = True

        self.counted_ids.add(
            track_id
        )

        return True


    # --------------------------------------------------------
    # 计算某一个 Track 和 Detection 的匹配代价。
    #
    # 返回：
    #   None -> 不允许匹配
    #   float -> 越小越好
    # --------------------------------------------------------

    def _match_cost(
        self,
        track,
        detection,
        predicted_box
    ):

        # ====================================================
        # A. 场景坐标距离
        #
        # 同一个固定苹果在参考坐标里应接近。
        # ====================================================

        scene_dist = point_distance(
            track["scene_center"],
            detection["scene_center"]
        )

        scene_scale = max(
            track["diag"],
            detection["diag"],
            20.0
        )

        normalized_scene_dist = (
            scene_dist
            /
            scene_scale
        )


        # ====================================================
        # B. 当前图像坐标距离
        #
        # 使用相机运动预测后的 predicted_box。
        # ====================================================

        predicted_center = box_center(
            predicted_box
        )

        image_dist = point_distance(
            predicted_center,
            detection["center"]
        )

        image_scale = max(
            box_diagonal(predicted_box),
            detection["diag"],
            20.0
        )

        normalized_image_dist = (
            image_dist
            /
            image_scale
        )


        # ====================================================
        # C. IoU
        # ====================================================

        iou = calculate_iou(
            predicted_box,
            detection["box"]
        )


        # ====================================================
        # D. 尺寸变化
        # ====================================================

        area1 = max(
            track["area"],
            1.0
        )

        area2 = max(
            detection["area"],
            1.0
        )

        area_ratio = (
            max(area1, area2)
            /
            min(area1, area2)
        )


        # ====================================================
        # E. 硬门限
        # ====================================================

        if area_ratio > MAX_AREA_RATIO:
            return None

        scene_ok = (
            normalized_scene_dist
            <=
            SCENE_DISTANCE_FACTOR
        )

        image_ok = (
            normalized_image_dist
            <=
            IMAGE_DISTANCE_FACTOR
        )

        iou_ok = (
            iou
            >=
            TRACK_IOU_THRESHOLD
        )

        # 普通活跃 Track：
        # 场景坐标、图像坐标、IoU 任一可信即可参与。
        if track["missed"] <= ACTIVE_MAX_MISSED:

            if not (
                scene_ok
                or
                image_ok
                or
                iou_ok
            ):
                return None

        # 长时间消失的历史 Track：
        # 重新进入时不能仅靠图像坐标。
        # 必须依赖场景坐标。
        else:

            if not scene_ok:
                return None


        # ====================================================
        # F. 最终 cost
        #
        # 场景坐标权重最高。
        # ====================================================

        size_penalty = abs(
            math.log(
                max(
                    area_ratio,
                    1e-6
                )
            )
        )

        cost = (
            SCENE_DISTANCE_WEIGHT
            *
            normalized_scene_dist
            +
            IMAGE_DISTANCE_WEIGHT
            *
            normalized_image_dist
            +
            IOU_WEIGHT
            *
            (
                1.0 - iou
            )
            +
            SIZE_WEIGHT
            *
            size_penalty
        )

        return cost


    # --------------------------------------------------------
    # 更新
    #
    # prev_to_curr：
    # 上一帧 -> 当前帧 的相机运动矩阵。
    # --------------------------------------------------------

    def update(
        self,
        detections,
        prev_to_curr,
        frame_index
    ):

        # ====================================================
        # A. 所有 Track 先 missed +1
        # ====================================================

        for track in self.tracks.values():
            track["missed"] += 1


        # ====================================================
        # B. 根据相机运动预测当前框
        # ====================================================

        predicted_boxes = {}

        for (
            track_id,
            track
        ) in self.tracks.items():

            # 只有最近帧仍活跃的 Track，
            # 才适合通过逐帧相机运动预测当前框。
            if track["missed"] <= ACTIVE_MAX_MISSED + 1:

                predicted_boxes[
                    track_id
                ] = transform_box(
                    track["box"],
                    prev_to_curr
                )

            else:

                # 长期 Dormant Track 的当前图像位置没有意义。
                # 这里仍保留旧框，
                # 但真正重新识别主要依靠 scene_center。
                predicted_boxes[
                    track_id
                ] = track["box"]


        # ====================================================
        # C. 构建所有候选匹配
        #
        # 思想对应 DeepSORT：
        #   Track <-> Detection 数据关联
        #
        # 但这里的核心特征是：
        #   场景几何坐标
        # 而不是行人 ReID embedding。
        # ====================================================

        candidates = []

        for (
            track_id,
            track
        ) in self.tracks.items():

            predicted_box = predicted_boxes[
                track_id
            ]

            for (
                detection_index,
                detection
            ) in enumerate(
                detections
            ):

                cost = self._match_cost(
                    track,
                    detection,
                    predicted_box
                )

                if cost is None:
                    continue

                candidates.append(
                    (
                        cost,
                        track_id,
                        detection_index
                    )
                )


        # ====================================================
        # D. 小 cost 优先匹配
        #
        # 这里使用一对一贪心。
        #
        # 对你的单类别静态苹果场景：
        # 配合相机运动补偿 + scene 坐标，
        # 比直接套行人 ReID 更符合任务。
        # ====================================================

        candidates.sort(
            key=lambda item: item[0]
        )

        matched_tracks = set()
        matched_detections = set()

        newly_counted_ids = []


        for (
            cost,
            track_id,
            detection_index
        ) in candidates:

            if track_id in matched_tracks:
                continue

            if (
                detection_index
                in
                matched_detections
            ):
                continue

            track = self.tracks[
                track_id
            ]

            detection = detections[
                detection_index
            ]


            # ------------------------------------------------
            # 框平滑
            # ------------------------------------------------

            old_box = track["box"]
            new_box = detection["box"]

            smoothed_box = [

                (
                    BOX_SMOOTH_ALPHA
                    *
                    new_box[i]
                    +
                    (
                        1.0
                        -
                        BOX_SMOOTH_ALPHA
                    )
                    *
                    old_box[i]
                )

                for i in range(4)
            ]


            # ------------------------------------------------
            # 场景坐标平滑
            #
            # 相机运动估计会有小误差，
            # 不直接用单帧结果覆盖历史值。
            # ------------------------------------------------

            old_scene = track[
                "scene_center"
            ]

            new_scene = detection[
                "scene_center"
            ]

            smooth_scene = (

                (
                    1.0
                    -
                    SCENE_CENTER_ALPHA
                )
                *
                old_scene[0]
                +
                SCENE_CENTER_ALPHA
                *
                new_scene[0],

                (
                    1.0
                    -
                    SCENE_CENTER_ALPHA
                )
                *
                old_scene[1]
                +
                SCENE_CENTER_ALPHA
                *
                new_scene[1],
            )


            # ------------------------------------------------
            # 更新 Track
            # ------------------------------------------------

            track["box"] = smoothed_box
            track["conf"] = detection["conf"]
            track["center"] = detection["center"]
            track["scene_center"] = smooth_scene
            track["diag"] = detection["diag"]
            track["area"] = detection["area"]
            track["missed"] = 0
            track["hits"] += 1
            track["last_frame"] = frame_index


            # ------------------------------------------------
            # Tentative -> Confirmed
            # ------------------------------------------------

            if (
                not track["confirmed"]
                and
                track["hits"] >= N_INIT
            ):

                track["confirmed"] = True

                if self._count_track(
                    track_id
                ):
                    newly_counted_ids.append(
                        track_id
                    )


            matched_tracks.add(
                track_id
            )

            matched_detections.add(
                detection_index
            )


        # ====================================================
        # E. 未匹配 Detection -> 新 Track
        # ====================================================

        for (
            detection_index,
            detection
        ) in enumerate(
            detections
        ):

            if (
                detection_index
                in
                matched_detections
            ):
                continue

            new_id = self._create_track(
                detection,
                frame_index,
                confirm_immediately=False
            )

            # 新 Track 先不计入 Total，
            # 需要下一帧再次匹配成功。
            _ = new_id


        return newly_counted_ids


    # --------------------------------------------------------
    # 第一帧初始化
    #
    # 第一帧检测质量已经经过实际验证较好，
    # 所以直接认为第一帧所有去重后苹果都是真实苹果。
    # --------------------------------------------------------

    def initialize_first_frame(
        self,
        detections,
        frame_index=0
    ):

        newly_counted_ids = []

        for detection in detections:

            track_id = self._create_track(
                detection,
                frame_index,
                confirm_immediately=True
            )

            newly_counted_ids.append(
                track_id
            )

        return newly_counted_ids


    # --------------------------------------------------------
    # 当前帧苹果数量
    # --------------------------------------------------------

    def current_count(self):

        count = 0

        for track in self.tracks.values():

            if not track["confirmed"]:
                continue

            if (
                track["missed"]
                <=
                CURRENT_COUNT_TTL
            ):
                count += 1

        return count


    # --------------------------------------------------------
    # 累计唯一苹果数量
    # --------------------------------------------------------

    def total_count(self):

        return len(
            self.counted_ids
        )


    # --------------------------------------------------------
    # 当前需要绘制的 Track
    # --------------------------------------------------------

    def drawable_tracks(self):

        result = []

        for (
            track_id,
            track
        ) in self.tracks.items():

            if not track["confirmed"]:
                continue

            if (
                track["missed"]
                >
                BOX_TTL
            ):
                continue

            result.append(
                (
                    track_id,
                    track
                )
            )

        return result


# ============================================================
# 15. 绘制计数
# ============================================================

def draw_count(
    frame,
    current_count,
    total_count
):

    cv2.rectangle(
        frame,
        (0, 0),
        (430, 110),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        f"Current Apples: {current_count}",
        (10, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        f"Total Apples: {total_count}",
        (10, 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )


# ============================================================
# 16. 获取视频
# ============================================================

def get_video_list():

    if not VIDEO_DIR.exists():

        raise FileNotFoundError(
            f"找不到视频目录:\n{VIDEO_DIR}"
        )

    return sorted(
        [
            path

            for path in VIDEO_DIR.iterdir()

            if (
                path.is_file()
                and
                path.suffix.lower()
                in
                VIDEO_EXTENSIONS
            )
        ]
    )


# ============================================================
# 17. 处理单个视频
# ============================================================

def process_video(
    model,
    video_path
):

    print()
    print("=" * 75)
    print(
        f"开始处理: {video_path.name}"
    )
    print("=" * 75)

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"无法打开视频:\n{video_path}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if fps <= 0:
        fps = 25.0


    # --------------------------------------------------------
    # 输出
    # --------------------------------------------------------

    output_path = (
        OUTPUT_DIR
        /
        f"{video_path.stem}_apple_scene_tracker.mp4"
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (
            width,
            height
        )
    )

    if not writer.isOpened():

        cap.release()

        raise RuntimeError(
            f"无法创建输出视频:\n{output_path}"
        )


    # --------------------------------------------------------
    # Tracker
    # --------------------------------------------------------

    tracker = AppleSceneTracker()

    frame_index = 0

    prev_gray = None

    # 第 0 帧参考坐标 -> 当前帧坐标
    cumulative_ref_to_current = np.eye(
        3,
        dtype=np.float64
    )


    # ========================================================
    # 视频循环
    # ========================================================

    while True:

        success, frame = cap.read()

        if not success:
            break

        curr_gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )


        # ====================================================
        # A. 估计相机运动
        # ====================================================

        if prev_gray is None:

            prev_to_curr = np.eye(
                3,
                dtype=np.float64
            )

            inlier_count = 0

        else:

            (
                prev_to_curr,
                inlier_count
            ) = estimate_camera_motion(
                prev_gray,
                curr_gray
            )

            # 累计：
            # ref -> prev
            # prev -> current
            # 所以：
            # ref -> current
            cumulative_ref_to_current = (
                prev_to_curr
                @
                cumulative_ref_to_current
            )


        # ====================================================
        # B. Exp003 YOLO 检测
        # ====================================================

        detections = detect_frame(
            model,
            frame
        )


        # ====================================================
        # C. 将 Detection 加上 scene 坐标
        # ====================================================

        add_scene_coordinates(
            detections,
            cumulative_ref_to_current
        )


        # ====================================================
        # D. Tracking
        # ====================================================

        if frame_index == 0:

            newly_counted_ids = (
                tracker.initialize_first_frame(
                    detections,
                    frame_index
                )
            )

        else:

            newly_counted_ids = (
                tracker.update(
                    detections,
                    prev_to_curr,
                    frame_index
                )
            )


        current_count = (
            tracker.current_count()
        )

        total_count = (
            tracker.total_count()
        )


        # ====================================================
        # E. 绘制
        # ====================================================

        output_frame = frame.copy()

        for (
            track_id,
            track
        ) in tracker.drawable_tracks():

            box = track["box"]

            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])

            thickness = (
                2
                if track["missed"] == 0
                else 1
            )

            cv2.rectangle(
                output_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                thickness
            )

            # 为了调试 ID 稳定性，
            # 框左上角显示 Track ID。
            cv2.putText(
                output_frame,
                f"ID {track_id}",
                (
                    x1,
                    max(
                        18,
                        y1 - 5
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA
            )


        draw_count(
            output_frame,
            current_count,
            total_count
        )


        # ====================================================
        # F. 可选：显示相机运动估计状态
        # ====================================================

        if MOTION_DEBUG:

            cv2.putText(
                output_frame,
                f"Motion inliers: {inlier_count}",
                (
                    10,
                    height - 20
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                1,
                cv2.LINE_AA
            )


        writer.write(
            output_frame
        )


        # ====================================================
        # G. 日志
        # ====================================================

        if len(newly_counted_ids) > 0:

            print(
                f"[Frame {frame_index}] "
                f"累计 +{len(newly_counted_ids)} "
                f"| 新 IDs: {newly_counted_ids} "
                f"| Current: {current_count} "
                f"| Total: {total_count}"
            )

        elif (
            frame_index % 30 == 0
            or
            frame_index == total_frames - 1
        ):

            print(
                f"[{video_path.name}] "
                f"{frame_index + 1}/{total_frames} "
                f"| YOLO: {len(detections)} "
                f"| Current: {current_count} "
                f"| Total: {total_count} "
                f"| Motion inliers: {inlier_count}"
            )


        prev_gray = curr_gray

        frame_index += 1


    # ========================================================
    # 结束
    # ========================================================

    cap.release()
    writer.release()

    print()
    print("[完成]")
    print(
        f"输出视频:\n{output_path}"
    )
    print(
        f"最终累计唯一苹果数: "
        f"{tracker.total_count()}"
    )


# ============================================================
# 18. 主函数
# ============================================================

def main():

    print("=" * 75)
    print(
        "Fruit YOLO Apple Scene Tracker"
    )
    print(
        "Exp003 + Camera Motion Compensation + Scene Tracking + Total Count"
    )
    print("=" * 75)

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"找不到 Exp003 best.pt:\n{MODEL_PATH}"
        )

    videos = get_video_list()

    if len(videos) == 0:

        print(
            "data/vidio 中没有找到视频。"
        )

        return

    print(
        f"Model:\n{MODEL_PATH}"
    )

    print()
    print(
        f"Videos: {len(videos)}"
    )

    for video in videos:
        print(
            f" - {video.name}"
        )

    print()
    print(
        f"imgsz = {IMG_SIZE}"
    )

    print(
        f"Detect conf = {DETECT_CONF}"
    )

    print(
        f"N_INIT = {N_INIT}"
    )

    print(
        f"Scene distance factor = "
        f"{SCENE_DISTANCE_FACTOR}"
    )

    print(
        f"Active max missed = "
        f"{ACTIVE_MAX_MISSED}"
    )

    print()


    model = YOLO(
        str(MODEL_PATH)
    )


    for video_path in videos:

        process_video(
            model,
            video_path
        )


    print()
    print("=" * 75)
    print(
        "全部视频处理完成"
    )
    print("=" * 75)
    print(
        f"输出目录:\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
