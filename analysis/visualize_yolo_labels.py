from pathlib import Path

import cv2


# ============================================================
# 1. 数据路径
# ============================================================

IMAGE_DIR = Path(
    r"E:\workspace\01_Projects\AI_Vision_Learning\Fruit_YOLO\data\raw\images"
)

LABEL_DIR = Path(
    r"E:\workspace\01_Projects\AI_Vision_Learning\Fruit_YOLO\data\raw\labels"
)

OUTPUT_DIR = Path(
    r"E:\workspace\01_Projects\AI_Vision_Learning\Fruit_YOLO\runs\yolo_label_visualization"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. 类别名称
# ============================================================

CLASS_NAMES = {
    0: "pingguo",
}


# ============================================================
# 3. 读取 YOLO 标签
# ============================================================

def read_yolo_label(label_path):
    """
    YOLO Detect 格式：

    class_id x_center y_center width height

    其中坐标均为 0~1 的归一化坐标。
    """

    objects = []

    with open(label_path, "r", encoding="utf-8") as f:

        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            # YOLO 检测格式正常应该有 5 个值
            if len(parts) != 5:
                print(
                    f"[标签格式异常] {label_path.name} "
                    f"第 {line_number} 行: {line}"
                )
                continue

            try:

                class_id = int(parts[0])

                x_center = float(parts[1])
                y_center = float(parts[2])
                bbox_width = float(parts[3])
                bbox_height = float(parts[4])

            except ValueError:

                print(
                    f"[无法解析] {label_path.name} "
                    f"第 {line_number} 行"
                )
                continue

            objects.append(
                {
                    "class_id": class_id,
                    "x_center": x_center,
                    "y_center": y_center,
                    "width": bbox_width,
                    "height": bbox_height,
                }
            )

    return objects


# ============================================================
# 4. YOLO 坐标 → 像素坐标
# ============================================================

def yolo_to_pixel(obj, image_width, image_height):

    x_center = obj["x_center"] * image_width
    y_center = obj["y_center"] * image_height

    bbox_width = obj["width"] * image_width
    bbox_height = obj["height"] * image_height

    xmin = int(round(x_center - bbox_width / 2))
    ymin = int(round(y_center - bbox_height / 2))

    xmax = int(round(x_center + bbox_width / 2))
    ymax = int(round(y_center + bbox_height / 2))

    # 防止绘图时越出图像
    xmin = max(0, min(xmin, image_width - 1))
    ymin = max(0, min(ymin, image_height - 1))

    xmax = max(0, min(xmax, image_width - 1))
    ymax = max(0, min(ymax, image_height - 1))

    return xmin, ymin, xmax, ymax


# ============================================================
# 5. 绘制 YOLO 标签
# ============================================================

def draw_labels(image, objects):

    image_height, image_width = image.shape[:2]

    for index, obj in enumerate(objects, start=1):

        class_id = obj["class_id"]

        class_name = CLASS_NAMES.get(
            class_id,
            f"class_{class_id}"
        )

        xmin, ymin, xmax, ymax = yolo_to_pixel(
            obj,
            image_width,
            image_height
        )

        # ------------------------------
        # 画检测框
        # ------------------------------

        cv2.rectangle(
            image,
            (xmin, ymin),
            (xmax, ymax),
            (0, 255, 0),
            2
        )

        # ------------------------------
        # 标注编号
        # ------------------------------

        text = f"{class_name} {index}"

        cv2.putText(
            image,
            text,
            (xmin, max(ymin - 5, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

    # ------------------------------
    # 左上角显示目标总数
    # ------------------------------

    cv2.putText(
        image,
        f"Objects: {len(objects)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 255),
        3,
        cv2.LINE_AA
    )

    return image


# ============================================================
# 6. 主程序
# ============================================================

def main():

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp"
    }

    image_files = sorted(
        [
            p
            for p in IMAGE_DIR.iterdir()
            if p.is_file()
            and p.suffix.lower() in image_extensions
        ]
    )

    print("=" * 70)
    print("YOLO 标签可视化检查")
    print("=" * 70)

    print(f"图片目录 : {IMAGE_DIR}")
    print(f"标签目录 : {LABEL_DIR}")
    print(f"图片数量 : {len(image_files)}")
    print()

    if not image_files:

        print("没有找到图片。")
        return

    checked_images = 0
    missing_labels = 0

    for image_path in image_files:

        label_path = LABEL_DIR / f"{image_path.stem}.txt"

        # ------------------------------
        # 检查标签是否存在
        # ------------------------------

        if not label_path.exists():

            print(
                f"[缺少标签] {image_path.name}"
            )

            missing_labels += 1

            continue

        # ------------------------------
        # 读取图像
        # ------------------------------

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            print(
                f"[图片读取失败] {image_path.name}"
            )

            continue

        # ------------------------------
        # 读取 YOLO 标签
        # ------------------------------

        objects = read_yolo_label(
            label_path
        )

        print(
            f"{image_path.name:<22}"
            f"目标数: {len(objects)}"
        )

        # ------------------------------
        # 绘制
        # ------------------------------

        result = draw_labels(
            image.copy(),
            objects
        )

        # ------------------------------
        # 保存可视化结果
        # ------------------------------

        output_path = (
            OUTPUT_DIR / image_path.name
        )

        cv2.imwrite(
            str(output_path),
            result
        )

        checked_images += 1

        # ------------------------------
        # 为了适配屏幕，缩小显示
        # ------------------------------

        image_height, image_width = result.shape[:2]

        max_display_width = 1200
        max_display_height = 750

        scale = min(
            max_display_width / image_width,
            max_display_height / image_height,
            1.0
        )

        display_width = int(
            image_width * scale
        )

        display_height = int(
            image_height * scale
        )

        display_image = cv2.resize(
            result,
            (display_width, display_height)
        )

        # ------------------------------
        # 显示
        # ------------------------------

        cv2.imshow(
            "YOLO Label Visualization",
            display_image
        )

        print(
            "  任意键 = 下一张 | q = 退出"
        )

        key = cv2.waitKey(0) & 0xFF

        if key == ord("q"):
            print("用户提前退出可视化。")
            break

    cv2.destroyAllWindows()

    # ========================================================
    # 7. 最终统计
    # ========================================================

    print()
    print("=" * 70)
    print("检查结束")
    print("=" * 70)

    print(
        f"成功检查图片 : {checked_images}"
    )

    print(
        f"缺失标签     : {missing_labels}"
    )

    print()

    print(
        "可视化结果保存到："
    )

    print(
        OUTPUT_DIR
    )

    print("=" * 70)


if __name__ == "__main__":
    main()

    