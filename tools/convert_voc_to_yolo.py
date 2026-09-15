from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import cv2


# ============================================================
# 1. 项目配置
# ============================================================

SOURCE_DIRS = [
    Path(
        r"E:\workspace\01_Projects\AI_Vision_Learning\Fruit_YOLO\picked(2)\picked(2)"
    ),
    Path(
        r"E:\workspace\01_Projects\AI_Vision_Learning\Fruit_YOLO\picked2(1)(2)(1)\picked2(1)(2)"
    ),
]

OUTPUT_ROOT = Path(
    r"E:\workspace\01_Projects\AI_Vision_Learning\Fruit_YOLO\data\raw"
)

OUTPUT_IMAGES = OUTPUT_ROOT / "images"
OUTPUT_LABELS = OUTPUT_ROOT / "labels"


# 当前只有一个类别
CLASS_MAP = {
    "pingguo": 0,
}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
}


# ============================================================
# 2. 清空 / 创建输出目录
# ============================================================

def prepare_output_dirs():
    """
    为了防止重复运行导致旧数据残留，
    每次运行先清空 raw/images 和 raw/labels。
    """

    if OUTPUT_IMAGES.exists():
        shutil.rmtree(OUTPUT_IMAGES)

    if OUTPUT_LABELS.exists():
        shutil.rmtree(OUTPUT_LABELS)

    OUTPUT_IMAGES.mkdir(parents=True, exist_ok=True)
    OUTPUT_LABELS.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. 读取 VOC XML
# ============================================================

def parse_voc_xml(xml_path):
    """
    读取 Pascal VOC XML。

    返回：
        xml_width
        xml_height
        objects
    """

    tree = ET.parse(xml_path)
    root = tree.getroot()

    size_node = root.find("size")

    if size_node is None:
        raise ValueError("XML 缺少 <size>")

    width_node = size_node.find("width")
    height_node = size_node.find("height")

    if width_node is None or height_node is None:
        raise ValueError("XML 缺少 width / height")

    width = int(float(width_node.text))
    height = int(float(height_node.text))

    if width <= 0 or height <= 0:
        raise ValueError(
            f"非法图像尺寸: {width}x{height}"
        )

    objects = []

    for obj in root.findall("object"):

        name_node = obj.find("name")

        if name_node is None or name_node.text is None:
            continue

        class_name = name_node.text.strip()

        bbox_node = obj.find("bndbox")

        if bbox_node is None:
            continue

        try:
            xmin = float(bbox_node.find("xmin").text)
            ymin = float(bbox_node.find("ymin").text)
            xmax = float(bbox_node.find("xmax").text)
            ymax = float(bbox_node.find("ymax").text)

        except Exception:
            continue

        objects.append(
            {
                "class_name": class_name,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
            }
        )

    return width, height, objects


# ============================================================
# 4. 清洗 bbox
# ============================================================

def clean_bbox(
    xmin,
    ymin,
    xmax,
    ymax,
    image_width,
    image_height,
):
    """
    清洗 VOC bbox。

    规则：
    1. 自动限制在图像边界内
    2. 删除宽高 <= 0 的框
    """

    # ----------
    # 自动裁剪越界框
    # ----------
    xmin = max(0.0, min(xmin, image_width))
    ymin = max(0.0, min(ymin, image_height))

    xmax = max(0.0, min(xmax, image_width))
    ymax = max(0.0, min(ymax, image_height))

    bbox_width = xmax - xmin
    bbox_height = ymax - ymin

    # ----------
    # 删除非法 bbox
    # ----------
    if bbox_width <= 0 or bbox_height <= 0:
        return None

    return xmin, ymin, xmax, ymax


# ============================================================
# 5. VOC bbox → YOLO bbox
# ============================================================

def voc_to_yolo(
    xmin,
    ymin,
    xmax,
    ymax,
    image_width,
    image_height,
):
    """
    YOLO 格式：
        class_id x_center y_center width height

    坐标全部归一化到 0~1。
    """

    x_center = (
        (xmin + xmax) / 2.0
    ) / image_width

    y_center = (
        (ymin + ymax) / 2.0
    ) / image_height

    bbox_width = (
        xmax - xmin
    ) / image_width

    bbox_height = (
        ymax - ymin
    ) / image_height

    return (
        x_center,
        y_center,
        bbox_width,
        bbox_height,
    )


# ============================================================
# 6. 给输出文件生成唯一名称
# ============================================================

def build_output_name(
    source_index,
    image_path,
):
    """
    两个数据集有可能都存在：

        000001.jpg

    所以统一加数据源编号。

    例如：
        set01_000001.jpg
        set02_000001.jpg
    """

    return (
        f"set{source_index:02d}_"
        f"{image_path.stem}"
    )


# ============================================================
# 7. 主处理函数
# ============================================================

def process_dataset():

    prepare_output_dirs()

    stats = {
        "images_found": 0,
        "xml_found": 0,
        "saved_images": 0,

        "missing_xml": 0,
        "bad_images": 0,
        "bad_xml": 0,
        "empty_samples": 0,

        "total_objects": 0,
        "saved_objects": 0,

        "unknown_classes": 0,
        "invalid_bbox": 0,
        "clipped_bbox": 0,
    }

    print("=" * 70)
    print("VOC -> YOLO 数据转换 + 清洗")
    print("=" * 70)

    for source_index, source_dir in enumerate(
        SOURCE_DIRS,
        start=1,
    ):

        print()
        print(
            f"[数据源 {source_index}]"
        )
        print(source_dir)
        print("-" * 70)

        if not source_dir.exists():
            print(
                "[错误] 数据目录不存在"
            )
            continue

        image_files = sorted(
            [
                p
                for p in source_dir.iterdir()
                if (
                    p.is_file()
                    and
                    p.suffix.lower()
                    in IMAGE_EXTENSIONS
                )
            ]
        )

        print(
            f"发现图片: {len(image_files)}"
        )

        stats["images_found"] += len(
            image_files
        )

        for image_path in image_files:

            xml_path = (
                image_path.with_suffix(".xml")
            )

            # ------------------------------------------------
            # 1. XML 是否存在
            # ------------------------------------------------

            if not xml_path.exists():

                print(
                    f"[跳过] 缺少 XML: "
                    f"{image_path.name}"
                )

                stats["missing_xml"] += 1

                continue

            stats["xml_found"] += 1

            # ------------------------------------------------
            # 2. 检查图片
            # ------------------------------------------------

            image = cv2.imread(
                str(image_path)
            )

            if image is None:

                print(
                    f"[跳过] 图片损坏: "
                    f"{image_path.name}"
                )

                stats["bad_images"] += 1

                continue

            real_height, real_width = (
                image.shape[:2]
            )

            # ------------------------------------------------
            # 3. 解析 XML
            # ------------------------------------------------

            try:

                (
                    xml_width,
                    xml_height,
                    objects,
                ) = parse_voc_xml(
                    xml_path
                )

            except Exception as e:

                print(
                    f"[跳过] XML 异常: "
                    f"{xml_path.name}"
                    f" | {e}"
                )

                stats["bad_xml"] += 1

                continue

            # ------------------------------------------------
            # 4. XML 尺寸与真实图像尺寸检查
            # ------------------------------------------------

            if (
                xml_width != real_width
                or
                xml_height != real_height
            ):

                print(
                    f"[尺寸不一致] "
                    f"{image_path.name}: "
                    f"XML={xml_width}x{xml_height}, "
                    f"真实={real_width}x{real_height}"
                )

                # 这里以真实图片尺寸为准
                width = real_width
                height = real_height

            else:

                width = xml_width
                height = xml_height

            # ------------------------------------------------
            # 5. 清洗目标
            # ------------------------------------------------

            yolo_lines = []

            for obj in objects:

                stats["total_objects"] += 1

                class_name = (
                    obj["class_name"]
                )

                # ----------
                # 未知类别
                # ----------

                if class_name not in CLASS_MAP:

                    print(
                        f"[未知类别] "
                        f"{xml_path.name}: "
                        f"{class_name}"
                    )

                    stats[
                        "unknown_classes"
                    ] += 1

                    continue

                class_id = (
                    CLASS_MAP[class_name]
                )

                old_bbox = (
                    obj["xmin"],
                    obj["ymin"],
                    obj["xmax"],
                    obj["ymax"],
                )

                cleaned = clean_bbox(
                    xmin=obj["xmin"],
                    ymin=obj["ymin"],
                    xmax=obj["xmax"],
                    ymax=obj["ymax"],
                    image_width=width,
                    image_height=height,
                )

                if cleaned is None:

                    print(
                        f"[非法 bbox] "
                        f"{xml_path.name}: "
                        f"{old_bbox}"
                    )

                    stats[
                        "invalid_bbox"
                    ] += 1

                    continue

                (
                    xmin,
                    ymin,
                    xmax,
                    ymax,
                ) = cleaned

                new_bbox = (
                    xmin,
                    ymin,
                    xmax,
                    ymax,
                )

                if new_bbox != old_bbox:

                    stats[
                        "clipped_bbox"
                    ] += 1

                (
                    x_center,
                    y_center,
                    bbox_width,
                    bbox_height,
                ) = voc_to_yolo(
                    xmin=xmin,
                    ymin=ymin,
                    xmax=xmax,
                    ymax=ymax,
                    image_width=width,
                    image_height=height,
                )

                # ----------
                # 最后的 YOLO 合法性检查
                # ----------

                values = [
                    x_center,
                    y_center,
                    bbox_width,
                    bbox_height,
                ]

                if not all(
                    0.0 <= value <= 1.0
                    for value in values
                ):

                    print(
                        f"[YOLO 坐标异常] "
                        f"{xml_path.name}"
                    )

                    stats[
                        "invalid_bbox"
                    ] += 1

                    continue

                yolo_line = (
                    f"{class_id} "
                    f"{x_center:.6f} "
                    f"{y_center:.6f} "
                    f"{bbox_width:.6f} "
                    f"{bbox_height:.6f}"
                )

                yolo_lines.append(
                    yolo_line
                )

                stats[
                    "saved_objects"
                ] += 1

            # ------------------------------------------------
            # 6. 没有有效目标 → 跳过图片
            # ------------------------------------------------

            if not yolo_lines:

                print(
                    f"[跳过] 无有效目标: "
                    f"{image_path.name}"
                )

                stats[
                    "empty_samples"
                ] += 1

                continue

            # ------------------------------------------------
            # 7. 构建唯一输出名
            # ------------------------------------------------

            output_stem = (
                build_output_name(
                    source_index,
                    image_path,
                )
            )

            output_image_path = (
                OUTPUT_IMAGES
                /
                (
                    output_stem
                    +
                    image_path.suffix.lower()
                )
            )

            output_label_path = (
                OUTPUT_LABELS
                /
                f"{output_stem}.txt"
            )

            # ------------------------------------------------
            # 8. 保存图片
            # ------------------------------------------------

            shutil.copy2(
                image_path,
                output_image_path,
            )

            # ------------------------------------------------
            # 9. 保存 YOLO 标签
            # ------------------------------------------------

            with open(
                output_label_path,
                "w",
                encoding="utf-8",
            ) as f:

                f.write(
                    "\n".join(
                        yolo_lines
                    )
                )

                f.write("\n")

            stats[
                "saved_images"
            ] += 1

            print(
                f"[OK] "
                f"{image_path.name:<15} "
                f"目标数: {len(yolo_lines)}"
            )

    # ========================================================
    # 8. 输出最终统计
    # ========================================================

    print()
    print("=" * 70)
    print("转换完成")
    print("=" * 70)

    print(
        f"发现图片总数        : "
        f"{stats['images_found']}"
    )

    print(
        f"找到 XML 数量       : "
        f"{stats['xml_found']}"
    )

    print(
        f"最终保存图片        : "
        f"{stats['saved_images']}"
    )

    print(
        f"原始目标总数        : "
        f"{stats['total_objects']}"
    )

    print(
        f"最终保存目标数      : "
        f"{stats['saved_objects']}"
    )

    print()

    print(
        f"缺失 XML            : "
        f"{stats['missing_xml']}"
    )

    print(
        f"损坏图片            : "
        f"{stats['bad_images']}"
    )

    print(
        f"异常 XML            : "
        f"{stats['bad_xml']}"
    )

    print(
        f"无有效目标样本      : "
        f"{stats['empty_samples']}"
    )

    print(
        f"未知类别目标        : "
        f"{stats['unknown_classes']}"
    )

    print(
        f"非法 bbox           : "
        f"{stats['invalid_bbox']}"
    )

    print(
        f"自动裁剪越界 bbox   : "
        f"{stats['clipped_bbox']}"
    )

    print()

    print(
        f"图片输出目录:"
    )
    print(
        OUTPUT_IMAGES
    )

    print()

    print(
        f"标签输出目录:"
    )
    print(
        OUTPUT_LABELS
    )

    print("=" * 70)


# ============================================================
# 9. 程序入口
# ============================================================

if __name__ == "__main__":
    process_dataset()
    