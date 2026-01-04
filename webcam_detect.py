import cv2
import math
import time
import torch
from ultralytics import YOLO

# ==========================================
# ⚙️ CONFIG
# ==========================================
MODEL_PATH = "facemask_final/run_split_70_20_10/weights/best.pt"

CLASS_NAMES = ["with_mask", "mask_weared_incorrect", "without_mask"]

COLORS = {
    "with_mask": (0, 255, 0),
    "mask_weared_incorrect": (0, 255, 255),
    "without_mask": (0, 0, 255)
}


# ==========================================
# 🎨 DRAWING UTILS
# ==========================================
def draw_box(img, box, cls_name, conf, color):
    x1, y1, x2, y2 = map(int, box)

    # Draw rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    # Label
    label = f"{cls_name} {conf:.2f}"

    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x1, y1 - h - 6), (x1 + w, y1), color, -1)

    cv2.putText(
        img, label, (x1, y1 - 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA
    )


# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    # Cek device
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Menggunakan Device: {device}")

    # Load model
    try:
        print(f"[INFO] Loading model dari {MODEL_PATH} ...")
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"[ERROR] Gagal load model: {e}")
        return

    # Buka webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("[ERROR] Webcam tidak terdeteksi!")
        return

    print("[INFO] Mulai Deteksi... Tekan 'Q' untuk keluar.")

    prev_time = time.time()
    smoothed_fps = 0

    while True:
        ret, img = cap.read()
        if not ret:
            break

        # INFERENCE (lebih cepat dari predict())
        results = model(img, conf=0.5, device=device, verbose=False)

        for result in results:
            for box in result.boxes:
                xyxy = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                if cls >= len(CLASS_NAMES):
                    continue

                cls_name = CLASS_NAMES[cls]
                color = COLORS.get(cls_name, (255, 255, 255))

                draw_box(img, xyxy, cls_name, conf, color)

        # FPS stabil (moving average)
        now = time.time()
        fps = 1 / (now - prev_time)
        prev_time = now
        smoothed_fps = (smoothed_fps * 0.8) + (fps * 0.2)

        cv2.putText(
            img, f"FPS: {int(smoothed_fps)}",
            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
            (0, 255, 0), 2
        )

        cv2.imshow("Face Mask Detection System", img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
