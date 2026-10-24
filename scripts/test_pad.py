#!/usr/bin/env python3
"""Local Test & Benchmarking Suite for Fine-Tuned MobileNetV3-Small PAD Model.

Supports:
1. Single image evaluation (--image path/to/face.jpg)
2. Batch evaluation over an image directory (--dir path/to/folder)
3. Live Real-Time Webcam testing (--webcam)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None


def preprocess_image(image_input, target_size=(224, 224)) -> np.ndarray:
    """Preprocess image matching MobileNetV3 ImageNet training pipeline:
    RGB, resize to 224x224, normalize with mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
    Returns (1, 3, 224, 224) float32 tensor.
    """
    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, np.ndarray):
        # Assumed BGR from OpenCV, convert to RGB
        img = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
    else:
        img = image_input.convert("RGB")

    img = img.resize(target_size, Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0  # (224, 224, 3)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std

    # Transpose to (3, 224, 224) and add batch dim -> (1, 3, 224, 224)
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0).astype(np.float32)
    return arr


def sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class PADPredictor:
    def __init__(self, model_path: Path, metrics_path: Path | None = None):
        if ort is None:
            raise RuntimeError("onnxruntime is not installed. Install via: pip install onnxruntime")
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at: {model_path}\n"
                f"Please download 'mobilenetv3_pad.onnx' from Kaggle and place it in 'models/releases/'."
            )

        print(f"Loading ONNX PAD model from: {model_path}...")
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

        # Load calibrated threshold if available
        self.threshold = 0.5
        if metrics_path and metrics_path.exists():
            try:
                with open(metrics_path) as f:
                    data = json.load(f)
                    self.threshold = float(data.get("calibrated_threshold", 0.5))
                    print(f"[✓] Loaded calibrated threshold from {metrics_path.name}: τ* = {self.threshold:.4f} (EER: {data.get('eer', 0)*100:.2f}%)")
            except Exception as e:
                print(f"[!] Warning reading metrics: {e}; defaulting to τ = 0.5")
        else:
            print(f"[i] Using default threshold τ = {self.threshold:.4f}")

    def predict(self, image_input) -> tuple[str, float, float]:
        """Predicts whether an image is REAL (Bona-fide) or SPOOF (Attack).
        Returns:
            label: 'REAL' or 'SPOOF'
            score: Probability of being real (0.0 to 1.0)
            latency_ms: Inference time in milliseconds
        """
        tensor = preprocess_image(image_input)
        t0 = time.perf_counter()
        raw_logits = self.session.run(None, {self.input_name: tensor})[0]
        latency_ms = (time.perf_counter() - t0) * 1000.0

        logit = float(raw_logits[0, 0] if raw_logits.ndim == 2 else raw_logits[0])
        score = float(sigmoid(logit))

        label = "REAL" if score >= self.threshold else "SPOOF"
        return label, score, latency_ms


def run_single_image(predictor: PADPredictor, image_path: Path):
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    label, score, latency_ms = predictor.predict(image_path)
    confidence = score if label == "REAL" else (1.0 - score)
    color = "\033[92m" if label == "REAL" else "\033[91m"
    reset = "\033[0m"

    print("\n" + "=" * 50)
    print(f"🖼️  Image:      {image_path.name}")
    print(f"📊 Result:     {color}{label}{reset} (Live Prob: {score*100:.2f}%, Conf: {confidence*100:.2f}%)")
    print(f"⏱️  Latency:    {latency_ms:.2f} ms")
    print("=" * 50)


def run_directory(predictor: PADPredictor, dir_path: Path):
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Directory not found: {dir_path}")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [p for p in dir_path.glob("**/*") if p.suffix.lower() in exts]
    if not images:
        print(f"No images found in {dir_path}")
        return

    print(f"\nEvaluating {len(images)} images in {dir_path}...")
    real_count = 0
    spoof_count = 0
    total_time = 0.0

    for i, p in enumerate(images):
        label, score, lat = predictor.predict(p)
        total_time += lat
        if label == "REAL":
            real_count += 1
        else:
            spoof_count += 1

        if (i + 1) % 50 == 0 or (i + 1) == len(images):
            print(f"  Processed {i+1}/{len(images)}... (Current: {real_count} REAL, {spoof_count} SPOOF)")

    avg_lat = total_time / len(images)
    print("\n" + "=" * 50)
    print("📈 BATCH EVALUATION SUMMARY:")
    print(f"  • Total Images:   {len(images)}")
    print(f"  • Predicted REAL:  {real_count} ({real_count/len(images)*100:.1f}%)")
    print(f"  • Predicted SPOOF: {spoof_count} ({spoof_count/len(images)*100:.1f}%)")
    print(f"  • Avg Latency:     {avg_lat:.2f} ms / frame ({1000/avg_lat:.1f} FPS)")
    print("=" * 50)


def run_webcam(predictor: PADPredictor, camera_id: int = 0):
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required for webcam testing. Install via: pip install opencv-python")

    # Load YuNet face detector if available
    yunet_model = Path(__file__).resolve().parents[1] / "models/face_detection_yunet_2023mar.onnx"
    detector = None
    if yunet_model.exists():
        detector = cv2.FaceDetectorYN.create(str(yunet_model), "", (320, 320), 0.6, 0.3, 5000)
        print(f"[✓] Loaded YuNet face detector from {yunet_model.name}")
    else:
        print("[!] YuNet not found; using full-frame / center crop.")

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {camera_id}")

    print("\n" + "=" * 55)
    print("📷 LIVE WEBCAM ANTI-SPOOFING TEST STARTED")
    print("   • Press 'q' or 'ESC' on the video window to EXIT.")
    print("=" * 55 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        face_boxes = []

        if detector is not None:
            detector.setInputSize((w, h))
            _, faces = detector.detect(frame)
            if faces is not None:
                for f in faces:
                    x, y, fw, fh = int(f[0]), int(f[1]), int(f[2]), int(f[3])
                    face_boxes.append((x, y, fw, fh))

        if not face_boxes:
            # Fallback to center 50% box
            cw, ch = int(w * 0.5), int(h * 0.5)
            cx, cy = (w - cw) // 2, (h - ch) // 2
            face_boxes.append((cx, cy, cw, ch))

        for x, y, fw, fh in face_boxes:
            # Expand bbox by 20% margin
            margin_x = int(0.2 * fw)
            margin_y = int(0.2 * fh)
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(w, x + fw + margin_x)
            y2 = min(h, y + fh + margin_y)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            label, score, latency_ms = predictor.predict(crop)
            is_real = (label == "REAL")
            box_color = (0, 255, 0) if is_real else (0, 0, 255)  # BGR
            text_label = f"{label} ({score*100:.1f}%) [{latency_ms:.1f}ms]"

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(
                frame,
                text_label,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                box_color,
                2
            )

        cv2.imshow("FaceGuard AI - Live Presentation Attack Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    project_root = Path(__file__).resolve().parents[1]
    default_model = project_root / "models/releases/mobilenetv3_pad.onnx"
    default_metrics = project_root / "models/releases/pad_metrics.json"

    parser = argparse.ArgumentParser(description="Test & Benchmark Fine-Tuned PAD Model")
    parser.add_argument("--model", type=Path, default=default_model, help="Path to mobilenetv3_pad.onnx")
    parser.add_argument("--metrics", type=Path, default=default_metrics, help="Path to pad_metrics.json")
    parser.add_argument("--image", type=Path, help="Path to a single image to classify")
    parser.add_argument("--dir", type=Path, help="Path to a folder of images to benchmark")
    parser.add_argument("--webcam", action="store_true", help="Launch live webcam testing")
    parser.add_argument("--camera-id", type=int, default=0, help="Webcam device index (default: 0)")
    args = parser.parse_args()

    predictor = PADPredictor(args.model, args.metrics)

    if args.image:
        run_single_image(predictor, args.image)
    elif args.dir:
        run_directory(predictor, args.dir)
    elif args.webcam:
        run_webcam(predictor, args.camera_id)
    else:
        print("\n[!] Please specify a test mode:")
        print("  1. Single image: python3 scripts/test_pad.py --image path/to/face.jpg")
        print("  2. Directory:    python3 scripts/test_pad.py --dir path/to/folder")
        print("  3. Live webcam:  python3 scripts/test_pad.py --webcam\n")


if __name__ == "__main__":
    main()
