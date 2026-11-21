"""FPS (CPU/GPU) + mAP comparison across model variants -> results/benchmark.md
python -m src.benchmark --models best.pt best_pruned30.pt best_int8.onnx --images datasets/navobjects/images/val --data configs/data.yaml
"""
import argparse, time
from pathlib import Path
import cv2
import numpy as np


def fps(weights, imgs, device, imgsz, warmup=10, runs=100):
    from ultralytics import YOLO
    m = YOLO(weights, task="detect")
    frames = [cv2.imread(str(p)) for p in imgs[:runs]]
    for f in frames[:warmup]:
        m.predict(f, imgsz=imgsz, device=device, verbose=False)
    t = []
    for f in frames:
        s = time.perf_counter()
        m.predict(f, imgsz=imgsz, device=device, verbose=False)
        t.append(time.perf_counter() - s)  # end-to-end incl. pre/post-processing
    return 1.0 / np.mean(t), np.mean(t) * 1000

def size_mb(p):
    p = Path(p)
    return (sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else p.stat().st_size) / 1e6
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--data", default=None, help="data yaml; if given, mAP50-95 is computed too")
    p.add_argument("--imgsz", type=int, default=640)
    a = p.parse_args()
    imgs = sorted(q for q in Path(a.images).glob("*") if q.suffix.lower() in {".jpg", ".png", ".jpeg"})
    import torch
    devices = ["cpu"] + (["0"] if torch.cuda.is_available() else [])
    rows = ["| model | size MB | device | FPS | ms/img | mAP50-95 |", "|---|---|---|---|---|---|"]
    for w in a.models:
        mp = ""
        if a.data:
            from ultralytics import YOLO
            try:
                mp = f"{YOLO(w, task='detect').val(data=a.data, imgsz=a.imgsz, device='cpu', verbose=False).box.map:.3f}"
            except Exception:
                mp = "n/a"
        for d in devices:
            # if d == "0" and w.endswith(".onnx"):
            if d == "0" and (w.endswith(".onnx") or Path(w).is_dir()):
                continue  # onnxruntime-gpu not assumed
            f, ms = fps(w, imgs, d, a.imgsz)
            rows.append(f"| {Path(w).name} | {size_mb(w):.1f} | {'GPU' if d=='0' else 'CPU'} | {f:.1f} | {ms:.1f} | {mp} |")
    Path("results").mkdir(exist_ok=True)
    Path("results/benchmark.md").write_text("\n".join(rows) + "\n")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
