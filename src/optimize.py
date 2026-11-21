"""Edge optimisation: ONNX export, static INT8 quantisation, L1 pruning.
  python -m src.optimize onnx  --weights best.pt
  python -m src.optimize int8  --weights best.pt --calib-dir datasets/navobjects/images/val
  python -m src.optimize prune --weights best.pt --amount 0.3
"""
import argparse
from pathlib import Path
import cv2
import numpy as np
import onnx

def export_onnx(weights, imgsz=640):
    from ultralytics import YOLO
    return YOLO(weights).export(format="onnx", imgsz=imgsz, simplify=True, opset=1)



def _prep(path, imgsz):
    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (imgsz, imgsz)).astype(np.float32) / 255.0
    return img.transpose(2, 0, 1)[None]


def quantize_int8(weights, calib_dir, imgsz=640, n=100):
    """Static QDQ INT8 quantisation calibrated on real images."""
    import onnxruntime as ort
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    onnx_path = Path(export_onnx(weights, imgsz))
    out = onnx_path.with_name(onnx_path.stem + "_int8.onnx")
    name = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"]).get_inputs()[0].name
    files = sorted(p for p in Path(calib_dir).glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})[:n]
    exclude = [n.name for n in onnx.load(str(onnx_path)).graph.node if n.name.startswith("/model.22")]
    print("keeping", len(exclude), "head nodes in float")
    class Reader(CalibrationDataReader):
        def __init__(self): self.it = iter(files)
        def get_next(self):
            f = next(self.it, None)
            return None if f is None else {name: _prep(f, imgsz)}

    # quantize_static(str(onnx_path), str(out), Reader(), quant_format=QuantFormat.QDQ,
    #                 activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True)
    # print("INT8 model:", out, f"({out.stat().st_size/1e6:.1f} MB vs {onnx_path.stat().st_size/1e6:.1f} MB)")
    quantize_static(str(onnx_path), str(out), Reader(), quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                per_channel=True, nodes_to_exclude=exclude)
    return out


def prune(weights, amount=0.3):
    """Global L1 unstructured pruning of Conv weights; fine-tune afterwards to recover mAP.
    Unstructured sparsity gives little speed-up without a sparse runtime - the benchmark shows this."""
    import torch
    import torch.nn.utils.prune as prune_
    from ultralytics import YOLO
    y = YOLO(weights)
    convs = [(m, "weight") for m in y.model.modules() if isinstance(m, torch.nn.Conv2d)]
    prune_.global_unstructured(convs, pruning_method=prune_.L1Unstructured, amount=amount)
    for m, n in convs:
        prune_.remove(m, n)
    zeros = sum((m.weight == 0).sum().item() for m, _ in convs)
    total = sum(m.weight.numel() for m, _ in convs)
    out = Path(weights).with_name(f"{Path(weights).stem}_pruned{int(amount*100)}.pt")
    y.save(out)
    print(f"sparsity {zeros/total:.1%} -> {out}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["onnx", "int8", "prune"])
    p.add_argument("--weights", required=True)
    p.add_argument("--calib-dir")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--amount", type=float, default=0.3)
    a = p.parse_args()
    if a.mode == "onnx": export_onnx(a.weights, a.imgsz)
    elif a.mode == "int8": quantize_int8(a.weights, a.calib_dir, a.imgsz)
    else: prune(a.weights, a.amount)
