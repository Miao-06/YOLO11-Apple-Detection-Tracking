from pathlib import Path

import cv2


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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "hard_negative_candidates"
    / "manual_hard_negatives"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. 参数
# ============================================================

# 每次左右移动多少帧
STEP_FRAMES = 5


# ============================================================
# 3. 保存 ROI
# ============================================================

def save_roi(frame, frame_index, save_index):

    print()
    print("请用鼠标框选负样本区域。")
    print("选好以后按 ENTER / SPACE")
    print("取消按 C")
    print()

    # --------------------------------------------------------
    # OpenCV 自带 ROI 选择器
    # --------------------------------------------------------

    x, y, w, h = cv2.selectROI(
        "Select Hard Negative",
        frame,
        showCrosshair=True,
        fromCenter=False
    )

    cv2.destroyWindow(
        "Select Hard Negative"
    )

    # 用户取消
    if w == 0 or h == 0:

        print("[取消] 没有保存")

        return save_index

    crop = frame[
        y:y + h,
        x:x + w
    ].copy()

    if crop.size == 0:

        print("[错误] ROI 为空")

        return save_index

    filename = (
        f"hardneg_"
        f"f{frame_index:06d}_"
        f"{save_index:03d}.jpg"
    )

    output_path = (
        OUTPUT_DIR
        /
        filename
    )

    cv2.imwrite(
        str(output_path),
        crop
    )

    print(
        f"[保存] {output_path.name}"
    )

    return save_index + 1


# ============================================================
# 4. 主程序
# ============================================================

def main():

    print("=" * 70)
    print("Manual Hard Negative Cropper")
    print("=" * 70)

    print(f"Video : {VIDEO_PATH}")
    print(f"Output: {OUTPUT_DIR}")

    if not VIDEO_PATH.exists():

        raise FileNotFoundError(
            f"找不到视频:\n{VIDEO_PATH}"
        )

    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not cap.isOpened():

        raise RuntimeError(
            "无法打开视频"
        )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    print(
        f"FPS         : {fps:.2f}"
    )

    print(
        f"Total frames: {total_frames}"
    )

    print()
    print("操作方法：")
    print()
    print("A / ← : 向前退 5 帧")
    print("D / → : 向后进 5 帧")
    print("S     : 框选并保存 Hard Negative")
    print("SPACE : 暂停 / 查看当前帧")
    print("Q     : 退出")
    print()

    frame_index = 0
    save_index = 1

    while True:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_index
        )

        success, frame = (
            cap.read()
        )

        if not success:
            break

        display = frame.copy()

        # ----------------------------------------------------
        # 显示当前帧号和时间
        # ----------------------------------------------------

        current_time = (
            frame_index / fps
            if fps > 0
            else 0
        )

        text = (
            f"Frame: {frame_index}/{total_frames} "
            f"Time: {current_time:.2f}s"
        )

        cv2.putText(
            display,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        cv2.imshow(
            "Hard Negative Video Browser",
            display
        )

        key = cv2.waitKey(0) & 0xFF

        # ====================================================
        # Q：退出
        # ====================================================

        if key == ord("q"):

            break

        # ====================================================
        # D：向后
        # ====================================================

        elif key == ord("d"):

            frame_index += STEP_FRAMES

        # ====================================================
        # A：向前
        # ====================================================

        elif key == ord("a"):

            frame_index -= STEP_FRAMES

        # ====================================================
        # S：框选 hard negative
        # ====================================================

        elif key == ord("s"):

            save_index = save_roi(
                frame,
                frame_index,
                save_index
            )

        # ----------------------------------------------------
        # 防止越界
        # ----------------------------------------------------

        frame_index = max(
            0,
            min(
                frame_index,
                total_frames - 1
            )
        )

    cap.release()

    cv2.destroyAllWindows()

    print()
    print("=" * 70)
    print("Hard Negative 采集结束")
    print("=" * 70)

    print(
        f"共保存: {save_index - 1} 张"
    )

    print(
        f"目录: {OUTPUT_DIR}"
    )


# ============================================================
# Windows 主入口
# ============================================================

if __name__ == "__main__":
    main()