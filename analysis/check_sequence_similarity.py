from pathlib import Path

import cv2
import numpy as np


# ============================================================
# 1. 数据路径
# ============================================================

IMAGE_DIR = Path(
    r"E:\workspace\01_Projects\AI_Vision_Learning\Fruit_YOLO\data\raw\images"
)


# ============================================================
# 2. 图像预处理
# ============================================================

def preprocess_image(image_path):
    """
    为了快速比较相似度：
    1. 读取图像
    2. 转灰度
    3. 缩小尺寸
    """

    image = cv2.imread(str(image_path))

    if image is None:
        return None

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.resize(
        gray,
        (160, 90)
    )

    return gray


# ============================================================
# 3. 计算相邻图像相似度
# ============================================================

def calculate_similarity(img1, img2):
    """
    使用归一化平均绝对差异计算简单相似度。

    similarity 越接近 1：
        图像越相似

    similarity 越低：
        场景变化越明显
    """

    diff = cv2.absdiff(
        img1,
        img2
    )

    mean_diff = np.mean(diff)

    similarity = 1.0 - (
        mean_diff / 255.0
    )

    return similarity


# ============================================================
# 4. 主程序
# ============================================================

def main():

    image_files = sorted(
        IMAGE_DIR.glob("*.jpg")
    )

    print("=" * 70)
    print("连续帧 / 场景相似度检查")
    print("=" * 70)

    print(f"图片数量: {len(image_files)}")
    print()

    if len(image_files) < 2:
        print("图片数量不足，无法比较。")
        return

    previous_image = None
    previous_path = None

    similarities = []

    for image_path in image_files:

        current_image = preprocess_image(
            image_path
        )

        if current_image is None:

            print(
                f"[读取失败] {image_path.name}"
            )

            continue

        if previous_image is not None:

            similarity = calculate_similarity(
                previous_image,
                current_image
            )

            similarities.append(
                (
                    previous_path.name,
                    image_path.name,
                    similarity
                )
            )

            print(
                f"{previous_path.name:<22}"
                f" -> "
                f"{image_path.name:<22}"
                f" 相似度: {similarity:.4f}"
            )

        previous_image = current_image
        previous_path = image_path

    # ========================================================
    # 5. 统计
    # ========================================================

    if not similarities:
        return

    values = [
        item[2]
        for item in similarities
    ]

    print()
    print("=" * 70)
    print("统计结果")
    print("=" * 70)

    print(
        f"平均相似度 : "
        f"{np.mean(values):.4f}"
    )

    print(
        f"最高相似度 : "
        f"{np.max(values):.4f}"
    )

    print(
        f"最低相似度 : "
        f"{np.min(values):.4f}"
    )

    # --------------------------------------------------------
    # 找出变化最大的相邻图
    # --------------------------------------------------------

    print()
    print("变化最大的 10 组相邻图片：")
    print("-" * 70)

    sorted_pairs = sorted(
        similarities,
        key=lambda x: x[2]
    )

    for prev_name, curr_name, similarity in sorted_pairs[:10]:

        print(
            f"{prev_name:<22}"
            f" -> "
            f"{curr_name:<22}"
            f" 相似度: {similarity:.4f}"
        )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
    