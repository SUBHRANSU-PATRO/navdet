# Object Detection + Distance Estimation for Robotics Navigation

Detects **cones, barriers, stop signs**, estimates their distance from the camera using
pinhole geometry, annotates boxes as `Cone, 1.5m`, and benchmarks edge-optimised variants.

```
src/
  train.py       transfer learning (COCO-pretrained YOLOv8, frozen backbone)
  detector.py    inference wrapper (.pt / .onnx / INT8 .onnx)
  distance.py    pinhole-model distance estimation (height-based + ground-plane)
  annotate.py    "Cone, 1.5m" drawing
  infer.py       image / video / webcam pipeline
  optimize.py    ONNX export, static INT8 quantisation, L1 pruning
  benchmark.py   FPS (CPU/GPU) + mAP table -> results/benchmark.md
  bev.py         extra credit: homography bird's-eye view
scripts/convert_coco.py   COCO-format -> YOLO converter with class remapping
configs/                  data.yaml, camera.yaml
tests/test_distance.py    geometry round-trip tests
```

## Important: dataset note
BDD100K's detection classes are bus, traffic light, traffic sign, person, bike, truck, motor,
car, train, rider. **It has no cone, barrier or stop-sign class** ("traffic sign" is one merged class),
so it cannot be used directly for this task. The assignment allows other datasets; use e.g.:
- **COCO** (`stop sign`) via `scripts/convert_coco.py`
- **Roboflow Universe / Kaggle** traffic-cone and road-barrier datasets exported as COCO JSON
- Optionally BDD100K images as extra negatives (background) to reduce false positives

```bash
python scripts/convert_coco.py --images coco/train2017 --ann coco/instances_train2017.json \
  --split train --prefix coco_ --map "stop sign=stop sign"
python scripts/convert_coco.py --images cones/train --ann cones/_annotations.coco.json \
  --split train --prefix cone_ --map "traffic cone=cone" "cone=cone"
# repeat with --split val
```

## Quick start
```bash
pip install -r requirements.txt
python -m src.train --data configs/data.yaml --model yolov8n.pt --epochs 50
python -m src.infer --weights runs/train/navdet/weights/best.pt --source demo.mp4 --out out.mp4
pytest -q
```

## Distance estimation
Camera intrinsics: `fx = (W/2) / tan(HFOV/2)` (or real calibration values in `configs/camera.yaml`).
1. **Known-size (similar triangles):** `Z = fy · H_real / h_pixels`
2. **Ground plane:** for objects whose base touches the road, `Z = fy · h_cam / (y_bottom − y_horizon)`
3. `auto`: cones/barriers average (1) and (2); stop signs use (1) only (sign face is elevated).

Lateral offset: `X = (u_center − cx) · Z / fx`. Reported Z is forward depth along the optical axis.
Error sources: wrong real-size assumption (cone sizes vary 0.45–0.9 m), uncalibrated FOV, camera pitch
(shifts the horizon, so set `horizon_y`), and loose/truncated boxes. Validate against a few tape-measured
distances and tune `real_heights_m` / `hfov_deg`.

## Edge optimisation
```bash
python -m src.optimize onnx  --weights best.pt
python -m src.optimize int8  --weights best.pt --calib-dir datasets/navobjects/images/val
python -m src.optimize prune --weights best.pt --amount 0.3      # then re-fine-tune
python -m src.train --model yolov8n.pt ...                       # lightweight backbone vs yolov8s.pt
python -m src.benchmark --models best.pt best_pruned30.pt best.onnx best_int8.onnx \
   --images datasets/navobjects/images/val --data configs/data.yaml
```
`results/benchmark.md` gives the before/after table (size, CPU/GPU FPS, mAP50-95).
Expect INT8 ONNX to help most on CPU; unstructured pruning mainly reduces compressed size, not latency.

## Extra credit
- **Disparity-depth (epipolar, rectified stereo):** from similar triangles, `d = x_L − x_R = f·B / Z`, so `Z = f·B / d`.
- **Bird's-eye view:** `src/bev.py` — `homography()` from four ground points of a known rectangle, `to_bev()` to warp and plot detections.
- **Optical flow tracking:** not implemented; `cv2.calcOpticalFlowPyrLK` on box centres is the natural next step.
