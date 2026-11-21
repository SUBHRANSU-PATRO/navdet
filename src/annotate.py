import cv2

COLORS = {"cone": (0, 140, 255), "barrier": (0, 200, 255), "stop sign": (0, 0, 255)}


def draw(frame, dets):
    """Label format: 'Cone, 1.5m' / 'Stop Sign, 3.2m'."""
    for d in dets:
        x1, y1, x2, y2 = map(int, d["box"])
        col = COLORS.get(d["cls"], (0, 255, 0))
        z = (d.get("dist") or {}).get("z")
        label = d["cls"].title() + (f", {z:.1f}m" if z else "")
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), col, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame
