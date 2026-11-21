"""Detection + distance estimation on an image, video, or webcam index.
python -m src.infer --weights runs/train/navdet/weights/best.pt --source demo.mp4 --out out.mp4
"""
import argparse
from pathlib import Path
import cv2
from .detector import Detector
from .distance import Camera, estimate_distance
from .annotate import draw


def process(frame, det, cam, method="auto"):
    dets = det(frame)
    for d in dets:
        d["dist"] = estimate_distance(d["box"], d["cls"], cam, method)
    return draw(frame, dets), dets


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--source", required=True, help="image path, video path, or webcam index")
    p.add_argument("--camera", default="configs/camera.yaml")
    p.add_argument("--method", default="auto", choices=["auto", "height", "ground"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--out", default="output.mp4")
    a = p.parse_args()
    det = Detector(a.weights, a.device, a.imgsz, a.conf)

    src = int(a.source) if a.source.isdigit() else a.source
    if isinstance(src, str) and Path(src).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
        img = cv2.imread(src)
        cam = Camera.from_yaml(a.camera, img.shape[1], img.shape[0])
        out, dets = process(img, det, cam, a.method)
        out_path = a.out if a.out.lower().endswith((".jpg", ".png")) else "output.jpg"
        cv2.imwrite(out_path, out)
        for d in dets:
            print(d["cls"], f'{d["dist"]["z"]:.2f} m', f'conf={d["conf"]:.2f}')
        return
    cap = cv2.VideoCapture(src)
    w, h = int(cap.get(3)), int(cap.get(4))
    cam = Camera.from_yaml(a.camera, w, h)
    wr = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), cap.get(5) or 30, (w, h))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        wr.write(process(frame, det, cam, a.method)[0])
    cap.release(); wr.release()
    print("saved", a.out)


if __name__ == "__main__":
    main()
