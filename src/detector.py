"""Thin wrapper over Ultralytics YOLO; works with .pt / .onnx (incl. INT8 QDQ) weights."""
from __future__ import annotations
from ultralytics import YOLO


class Detector:
    def __init__(self, weights: str, device: str = "cpu", imgsz: int = 640, conf: float = 0.25):
        self.model = YOLO(weights, task="detect")
        self.device, self.imgsz, self.conf = device, imgsz, conf

    def __call__(self, frame_bgr):
        r = self.model.predict(frame_bgr, imgsz=self.imgsz, conf=self.conf,
                               device=self.device, verbose=False)[0]
        return [{"box": b.xyxy[0].tolist(), "conf": float(b.conf[0]), "cls": r.names[int(b.cls[0])]}
                for b in r.boxes]
