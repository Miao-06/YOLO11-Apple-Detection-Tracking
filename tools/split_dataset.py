from pathlib import Path
import shutil


# ============================================================
# 1. 路径配置
# ============================================================

RAW_IMAGES = Path(
    r"E:\workspace\01_Projects\AI_Vision_Learning\Fruit_YOLO\data\raw\images"
)

RAW_LABELS = Path(
    r"E:\workspace\01_Projects\AI_Vision_Learning\Fruit_YOLO\data\raw\labels"
)

SPLIT_ROOT = Path(
    r"E:\workspace\01_Projects\AI_Vision_Learning\Fruit_YOLO\data\split"
)

TRAIN_IMAGES = SPLIT_ROOT / "train" / "images"
TRAIN_LABELS = SPLIT_ROOT / "train" / "labels"

VAL_IMAGES = SPLIT_ROOT / "val" / "images"
VAL_LABELS = SPLIT_ROOT / "val" / "labels"


# ============================================================
# 2. 划分比例
# ============================================================

TRAIN_RATIO = 0.8


# ============================================================
# 3. 准备输出目录
# ============================================================

def prepare_dirs():

    # 清空旧划分，防止重复运行造成文件残留
    for path in [
        TRAIN_IMAGES,
        TRAIN_LABELS,
        VAL_IMAGES,
        VAL_LABELS,
    ]:

        if path.exists():
            shutil.rmtree(path)

        path.mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# 4. 获取某一个视频的数据
# ============================================================

def get_video_samples(prefix):
    """
    prefix:
        set01
        set02

    返回：
        [(image_path, label_path), ...]
    """

    image_files = sorted(
        RAW_IMAGES.glob(f"{prefix}_*.jpg")
    )

    samples = []

    for image_path in image_files:

        label_path = (
            RAW_LABELS / f"{image_path.stem}.txt"
        )

        if not label_path.exists():

            print(
                f"[警告] 缺少标签: {image_path.name}"
            )

            continue

        samples.append(
            (
                image_path,
                label_path
            )
        )

    return samples


# ============================================================
# 5. 按时间顺序划分
# ============================================================

def split_video(samples):
    """
    不随机打乱。

    前 80%：
        train

    后 20%：
        val
    """

    total = len(samples)

    split_index = int(
        total * TRAIN_RATIO
    )

    train_samples = (
        samples[:split_index]
    )

    val_samples = (
        samples[split_index:]
    )

    return train_samples, val_samples


# ============================================================
# 6. 复制样本
# ============================================================

def copy_samples(
    samples,
    image_output_dir,
    label_output_dir
):

    for image_path, label_path in samples:

        shutil.copy2(
            image_path,
            image_output_dir / image_path.name
        )

        shutil.copy2(
            label_path,
            label_output_dir / label_path.name
        )


# ============================================================
# 7. 主程序
# ============================================================

def main():

    print("=" * 70)
    print("Fruit YOLO 数据集时间顺序划分")
    print("=" * 70)

    prepare_dirs()

    all_train = []
    all_val = []

    # --------------------------------------------------------
    # 两个视频分别划分
    # --------------------------------------------------------

    for prefix in [
        "set01",
        "set02",
    ]:

        samples = get_video_samples(
            prefix
        )

        train_samples, val_samples = (
            split_video(samples)
        )

        print()
        print(
            f"[{prefix}]"
        )

        print(
            f"总图片数 : {len(samples)}"
        )

        print(
            f"Train   : {len(train_samples)}"
        )

        print(
            f"Val     : {len(val_samples)}"
        )

        # 显示边界文件，方便确认时间顺序
        if train_samples:

            print(
                f"Train 范围:"
            )

            print(
                f"  {train_samples[0][0].name}"
            )

            print(
                "        ↓"
            )

            print(
                f"  {train_samples[-1][0].name}"
            )

        if val_samples:

            print(
                f"Val 范围:"
            )

            print(
                f"  {val_samples[0][0].name}"
            )

            print(
                "        ↓"
            )

            print(
                f"  {val_samples[-1][0].name}"
            )

        all_train.extend(
            train_samples
        )

        all_val.extend(
            val_samples
        )

    # --------------------------------------------------------
    # 复制文件
    # --------------------------------------------------------

    copy_samples(
        all_train,
        TRAIN_IMAGES,
        TRAIN_LABELS
    )

    copy_samples(
        all_val,
        VAL_IMAGES,
        VAL_LABELS
    )

    # ========================================================
    # 8. 最终统计
    # ========================================================

    total = (
        len(all_train)
        +
        len(all_val)
    )

    train_ratio = (
        len(all_train) / total
        if total > 0
        else 0
    )

    val_ratio = (
        len(all_val) / total
        if total > 0
        else 0
    )

    print()
    print("=" * 70)
    print("最终划分结果")
    print("=" * 70)

    print(
        f"总图片数 : {total}"
    )

    print(
        f"Train    : {len(all_train)} "
        f"({train_ratio * 100:.2f}%)"
    )

    print(
        f"Val      : {len(all_val)} "
        f"({val_ratio * 100:.2f}%)"
    )

    print()
    print("Train:")
    print(TRAIN_IMAGES)
    print(TRAIN_LABELS)

    print()
    print("Val:")
    print(VAL_IMAGES)
    print(VAL_LABELS)

    print()
    print("=" * 70)
    print("划分完成")
    print("=" * 70)


if __name__ == "__main__":
    main()

    