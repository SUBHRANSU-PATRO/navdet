"""Extra credit: bird's-eye view via homography from a known ground rectangle."""
import cv2
import numpy as np


def homography(img_pts, width_m, length_m, px_per_m=50):
    """img_pts: 4 pixel points (TL, TR, BR, BL) of a ground rectangle of known size."""
    W, H = int(width_m * px_per_m), int(length_m * px_per_m)
    dst = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    return cv2.getPerspectiveTransform(np.float32(img_pts), dst), (W, H)


def to_bev(frame, dets, Hm, size):
    bev = cv2.warpPerspective(frame, Hm, size)
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        p = cv2.perspectiveTransform(np.float32([[[(x1 + x2) / 2, y2]]]), Hm)[0, 0]
        cv2.circle(bev, (int(p[0]), int(p[1])), 6, (0, 0, 255), -1)
        cv2.putText(bev, d["cls"], (int(p[0]) + 8, int(p[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return bev
