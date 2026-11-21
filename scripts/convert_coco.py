"""Convert COCO-format annotations (COCO, Roboflow, Open Images exports...) to YOLO format,
keeping only the target classes via a name map. Run once per source dataset/split.

python scripts/convert_coco.py --images imgs/ --ann instances.json --split train --prefix coco_ \
   --map "stop sign=stop sign" "traffic cone=cone" "barricade=barrier"
"""
import argparse, json, shutil
from pathlib import Path

TARGETS = ["cone", "barrier", "stop sign"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True)
    p.add_argument("--ann", required=True)
    p.add_argument("--out", default="datasets/navobjects")
    p.add_argument("--split", default="train")
    p.add_argument("--map", nargs="+", required=True, help="'source name=target name'")
    p.add_argument("--prefix", default="", help="filename prefix to avoid collisions between sources")
    a = p.parse_args()
    nm = dict(s.split("=") for s in a.map)
    coco = json.load(open(a.ann))
    cats = {c["id"]: c["name"].lower() for c in coco["categories"]}
    imgs = {i["id"]: i for i in coco["images"]}
    out_i, out_l = Path(a.out, "images", a.split), Path(a.out, "labels", a.split)
    out_i.mkdir(parents=True, exist_ok=True); out_l.mkdir(parents=True, exist_ok=True)
    lines = {}
    for an in coco["annotations"]:
        tgt = nm.get(cats[an["category_id"]])
        if tgt is None or an.get("iscrowd"):
            continue
        im = imgs[an["image_id"]]
        x, y, w, h = an["bbox"]
        lines.setdefault(im["id"], []).append(
            f"{TARGETS.index(tgt)} {(x+w/2)/im['width']:.6f} {(y+h/2)/im['height']:.6f} {w/im['width']:.6f} {h/im['height']:.6f}")
    kept = 0
    for iid, ls in lines.items():
        f = Path(imgs[iid]["file_name"]).name
        src = Path(a.images, f)
        if not src.exists():
            continue
        shutil.copy(src, out_i / (a.prefix + f))
        (out_l / (a.prefix + Path(f).stem + ".txt")).write_text("\n".join(ls))
        kept += 1
    print(f"{kept} images with target classes -> {a.out}")


if __name__ == "__main__":
    main()
