from pathlib import Path
import shutil


# ============================================================
# 1. 项目根目录
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# 2. Hard Negative 来源目录
# ============================================================

HARD_NEG_DIR = (
    PROJECT_ROOT
    / "data"
    / "hard_negative_candidates"
    / "manual_hard_negatives"
)


# ============================================================
# 3. 训练集目录
# ============================================================

TRAIN_IMAGES = (
    PROJECT_ROOT
    / "data"
    / "split"
    / "train"
    / "images"
)

TRAIN_LABELS = (
    PROJECT_ROOT
    / "data"
    / "split"
    / "train"
    / "labels"
)


# ============================================================
# 4. 支持的图片格式
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
}


# ============================================================
# 5. 主程序
# ============================================================

def main():

    print("=" * 70)
    print("Add Hard Negatives to Training Set")
    print("=" * 70)

    # --------------------------------------------------------
    # 路径检查
    # --------------------------------------------------------

    if not HARD_NEG_DIR.exists():
        raise FileNotFoundError(
            f"找不到 Hard Negative 目录:\n{HARD_NEG_DIR}"
        )

    if not TRAIN_IMAGES.exists():
        raise FileNotFoundError(
            f"找不到训练图片目录:\n{TRAIN_IMAGES}"
        )

    if not TRAIN_LABELS.exists():
        raise FileNotFoundError(
            f"找不到训练标签目录:\n{TRAIN_LABELS}"
        )

    # --------------------------------------------------------
    # 获取 hard negative 图片
    # --------------------------------------------------------

    hard_negative_images = sorted(
        [
            p
            for p in HARD_NEG_DIR.iterdir()
            if (
                p.is_file()
                and p.suffix.lower() in IMAGE_EXTENSIONS
            )
        ]
    )

    print(
        f"Hard Negative 数量: "
        f"{len(hard_negative_images)}"
    )

    if not hard_negative_images:
        print("没有找到可加入的图片。")
        return

    # --------------------------------------------------------
    # 加入训练集
    # --------------------------------------------------------

    added_count = 0
    skipped_count = 0

    for index, image_path in enumerate(
        hard_negative_images,
        start=1
    ):

        # 统一重新命名
        output_stem = (
            f"hardneg_{index:03d}"
        )

        output_image_path = (
            TRAIN_IMAGES
            /
            (
                output_stem
                +
                image_path.suffix.lower()
            )
        )

        output_label_path = (
            TRAIN_LABELS
            /
            f"{output_stem}.txt"
        )

        # ----------------------------------------------------
        # 防止重复覆盖
        # ----------------------------------------------------

        if (
            output_image_path.exists()
            or
            output_label_path.exists()
        ):

            print(
                f"[跳过] 已存在: "
                f"{output_stem}"
            )

            skipped_count += 1
            continue

        # ----------------------------------------------------
        # 复制图片
        # ----------------------------------------------------

        shutil.copy2(
            image_path,
            output_image_path
        )

        # ----------------------------------------------------
        # 创建空标签
        # ----------------------------------------------------

        output_label_path.touch()

        print(
            f"[OK] "
            f"{image_path.name}"
            f" -> "
            f"{output_image_path.name}"
        )

        added_count += 1

    # ========================================================
    # 6. 最终统计
    # ========================================================

    train_image_count = len(
        [
            p
            for p in TRAIN_IMAGES.iterdir()
            if (
                p.is_file()
                and
                p.suffix.lower() in IMAGE_EXTENSIONS
            )
        ]
    )

    train_label_count = len(
        list(
            TRAIN_LABELS.glob("*.txt")
        )
    )

    print()
    print("=" * 70)
    print("添加完成")
    print("=" * 70)

    print(
        f"本次加入 Hard Negative : "
        f"{added_count}"
    )

    print(
        f"跳过数量              : "
        f"{skipped_count}"
    )

    print()

    print(
        f"训练图片总数          : "
        f"{train_image_count}"
    )

    print(
        f"训练标签总数          : "
        f"{train_label_count}"
    )

    print()

    if train_image_count == train_label_count:
        print(
            "[OK] 图片和标签数量一致"
        )
    else:
        print(
            "[WARNING] 图片和标签数量不一致"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()
    