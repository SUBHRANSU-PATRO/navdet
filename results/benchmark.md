| model | size MB | device | FPS | ms/img | mAP50-95 |
|---|---|---|---|---|---|
| best.pt | 6.2 | CPU | 17.9 | 55.8 | 0.680 |
| best.pt | 6.2 | GPU | 121.7 | 8.2 | 0.680 |
| best_pruned30.pt | 6.2 | CPU | 16.1 | 62.3 | 0.652 |
| best_pruned30.pt | 6.2 | GPU | 74.1 | 13.5 | 0.652 |
| best_openvino_model | 12.4 | CPU | 79.5 | 12.6 | 0.678 |
| best_int8_openvino_model | 3.6 | CPU | 97.3 | 10.3 | 0.649 |
