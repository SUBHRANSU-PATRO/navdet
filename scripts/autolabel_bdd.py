"""Auto-label BDD100K images with YOLO-World (text prompts) -> YOLO dataset for cone/barrier/stop sign."""
import argparse, random, shutil
from pathlib import Path
from ultralytics import YOLOWorld

PROMPTS = ["traffic cone", "road barrier", "stop sign"]  # class ids 0, 1, 2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True)
    p.add_argument("--out", default="datasets/navobjects")
    p.add_argument("--n", type=int, default=4000)
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--neg-frac", type=float, default=0.1)
    p.add_argument("--device", default="0")
    a = p.parse_args()

    random.seed(0)
    files = sorted(Path(a.images).rglob("*.jpg"))
    random.shuffle(files)
    files = files[: a.n]
    print(f"scanning {len(files)} images")

    model = YOLOWorld("yolov8s-worldv2.pt")
    model.set_classes(PROMPTS)

    kept, negs = [], []
    for i in range(0, len(files), 16):
        batch = files[i:i + 16]
        for f, r in zip(batch, model.predict([str(x) for x in batch], conf=a.conf, device=a.device, verbose=False)):
            lines = [f"{int(b.cls[0])} " + " ".join(f"{v:.6f}" for v in b.xywhn[0].tolist()) for b in r.boxes]
            (kept if lines else negs).append((f, lines))
        print(f"\r{min(i + 16, len(files))}/{len(files)}  with objects: {len(kept)}", end="")
    print()

    negs = negs[: int(len(kept) * a.neg_frac)]
    items = kept + negs
    random.shuffle(items)
    n_val = int(len(items) * a.val_frac)
    for idx, (f, lines) in enumerate(items):
        split = "val" if idx < n_val else "train"
        di, dl = Path(a.out, "images", split), Path(a.out, "labels", split)
        di.mkdir(parents=True, exist_ok=True); dl.mkdir(parents=True, exist_ok=True)
        shutil.copy(f, di / f.name)
        (dl / (f.stem + ".txt")).write_text("\n".join(lines))
    print(f"done: {len(items)} images ({len(kept)} with objects) -> {a.out}")


if __name__ == "__main__":
    main()