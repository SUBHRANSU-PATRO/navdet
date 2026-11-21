"""Transfer learning: fine-tune COCO-pretrained YOLOv8 on cone / barrier / stop sign.
python -m src.train --data configs/data.yaml --model yolov8n.pt --epochs 50
"""
import argparse
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="configs/data.yaml")
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--freeze", type=int, default=10, help="freeze first N layers (backbone)")
    p.add_argument("--device", default="0")
    p.add_argument("--name", default="navdet")
    a = p.parse_args()
    YOLO(a.model).train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
                        freeze=a.freeze, device=a.device, project="runs/train", name=a.name,
                        patience=15, cos_lr=True)


if __name__ == "__main__":
    main()
