from pathlib import Path
import argparse
import random

import numpy as np
import torch
import yaml
from ultralytics import YOLO


# ============================================================
# 1. 项目根目录
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# 2. 固定随机种子
# ============================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    print(f"[INFO] Random seed fixed: {seed}")


# ============================================================
# 3. 读取配置
# ============================================================

def load_config(config_path):

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config not found:\n{config_path}"
        )

    with open(
        config_path,
        "r",
        encoding="utf-8"
    ) as f:

        config = yaml.safe_load(f)

    return config


# ============================================================
# 4. 路径检查
# ============================================================

def check_paths(config):

    model_path = Path(
        config["model"]
    )

    data_path = Path(
        config["data"]
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found:\n{model_path}"
        )

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset yaml not found:\n{data_path}"
        )


# ============================================================
# 5. 主训练函数
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="训练配置 YAML 路径"
    )

    args = parser.parse_args()

    config = load_config(
        args.config
    )

    seed = config.get(
        "seed",
        42
    )

    set_seed(seed)

    check_paths(config)

    print()
    print("=" * 70)
    print("Fruit YOLO Training")
    print("=" * 70)

    for key, value in config.items():
        print(
            f"{key:<15}: {value}"
        )

    print("=" * 70)
    print()

    # --------------------------------------------------------
    # 加载模型
    # --------------------------------------------------------

    model = YOLO(
        config["model"]
    )

    # --------------------------------------------------------
    # model 已经在上面加载过
    # 所以 train 参数里删除 model
    # --------------------------------------------------------

    train_args = config.copy()

    train_args.pop(
        "model",
        None
    )

    # --------------------------------------------------------
    # 开始训练
    # --------------------------------------------------------

    results = model.train(
        **train_args
    )

    print()
    print("=" * 70)
    print("Training completed")
    print("=" * 70)

    save_dir = getattr(
        results,
        "save_dir",
        None
    )

    if save_dir:
        print(
            f"Results: {save_dir}"
        )


# ============================================================
# Windows 主入口
# ============================================================

if __name__ == "__main__":
    main()
    