from ultralytics import YOLO


DATA_YAML = (
    r"E:\workspace\01_Projects\AI_Vision_Learning"
    r"\Fruit_YOLO\data\dataset.yaml"
)

MODEL_PATH = (
    r"E:\workspace\01_Projects\AI_Vision_Learning"
    r"\Fruit_YOLO\yolo11s.pt"
)


def main():

    model = YOLO(MODEL_PATH)

    model.val(
        data=DATA_YAML,
        imgsz=640,
        batch=4,
        device=0,
        workers=4,
    )


if __name__ == "__main__":
    main()
    