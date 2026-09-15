from pathlib import Path
import xml.etree.ElementTree as ET

import cv2


# =========================================================
# 1. 数据集路径
# =========================================================
DATASET_DIR = Path(
    r"E:\workspace\01_Projects\AI_Vision_Learning\Fruit_YOLO\picked(2)\picked(2)"
)

# 输出目录：保存画好框的图片
OUTPUT_DIR = Path(
    r"E:\workspace\01_Projects\AI_Vision_Learning\Fruit_YOLO\runs\label_visualization"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2. 读取 VOC XML 标注
# =========================================================
def read_voc_xml(xml_path):
    """
    读取 Pascal VOC 格式 XML 标注

    返回：
        width, height, objects

    objects 中每个元素格式：
        {
            "name": "pingguo",
            "xmin": 100,
            "ymin": 100,
            "xmax": 200,
            "ymax": 200
        }
    """

    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")

    width = int(size.find("width").text)
    height = int(size.find("height").text)

    objects = []

    for obj in root.findall("object"):

        name = obj.find("name").text

        bbox = obj.find("bndbox")

        xmin = int(float(bbox.find("xmin").text))
        ymin = int(float(bbox.find("ymin").text))
        xmax = int(float(bbox.find("xmax").text))
        ymax = int(float(bbox.find("ymax").text))

        objects.append(
            {
                "name": name,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
            }
        )

    return width, height, objects


# =========================================================
# 3. 在图片上画框
# =========================================================
def draw_annotations(image, objects):

    for i, obj in enumerate(objects):

        xmin = obj["xmin"]
        ymin = obj["ymin"]
        xmax = obj["xmax"]
        ymax = obj["ymax"]

        name = obj["name"]

        # 画 bbox
        cv2.rectangle(
            image,
            (xmin, ymin),
            (xmax, ymax),
            (0, 255, 0),
            2,
        )

        # 标签文字
        text = f"{name} {i + 1}"

        cv2.putText(
            image,
            text,
            (xmin, max(ymin - 5, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    return image


# =========================================================
# 4. 主程序
# =========================================================
def main():

    image_files = sorted(DATASET_DIR.glob("*.jpg"))

    print("=" * 60)
    print("VOC 标签可视化")
    print("=" * 60)

    print(f"数据集路径: {DATASET_DIR}")
    print(f"图片数量: {len(image_files)}")
    print()

    if len(image_files) == 0:
        print("没有找到 jpg 图片，请检查 DATASET_DIR")
        return

    for image_path in image_files:

        xml_path = image_path.with_suffix(".xml")

        # 没有对应 XML
        if not xml_path.exists():
            print(f"[缺少 XML] {image_path.name}")
            continue

        # 读取图片
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"[读取失败] {image_path.name}")
            continue

        # 读取 XML
        xml_width, xml_height, objects = read_voc_xml(xml_path)

        img_height, img_width = image.shape[:2]

        print(
            f"{image_path.name:<15}"
            f" 图像: {img_width}x{img_height}"
            f" XML: {xml_width}x{xml_height}"
            f" 目标数: {len(objects)}"
        )

        # 画框
        result = draw_annotations(image.copy(), objects)

        # 在左上角显示目标总数
        cv2.putText(
            result,
            f"Objects: {len(objects)}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        # 保存
        output_path = OUTPUT_DIR / image_path.name

        cv2.imwrite(
            str(output_path),
            result,
        )

        # 缩小显示，避免 1280x720 太大
        show_image = cv2.resize(
            result,
            None,
            fx=0.8,
            fy=0.8,
        )

        cv2.imshow(
            "VOC Label Visualization",
            show_image,
        )

        print(
            "按任意键查看下一张，按 q 退出"
        )

        key = cv2.waitKey(0)

        if key == ord("q"):
            break

    cv2.destroyAllWindows()

    print()
    print("=" * 60)
    print("可视化完成")
    print(f"结果保存到: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()