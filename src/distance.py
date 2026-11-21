"""Geometry-based monocular distance estimation (pinhole camera model)."""
from __future__ import annotations
import math
from dataclasses import dataclass, field
import yaml


@dataclass
class Camera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    camera_height_m: float = 1.4
    horizon_y: float | None = None
    real_heights_m: dict = field(default_factory=lambda: {"cone": 0.45, "barrier": 1.0, "stop sign": 0.75})

    @classmethod
    def from_fov(cls, width, height, hfov_deg=70.0, **kw):
        """fx = (W/2) / tan(HFOV/2); square pixels so fy = fx."""
        fx = (width / 2) / math.tan(math.radians(hfov_deg) / 2)
        return cls(width, height, fx, fx, width / 2, height / 2, **kw)

    @classmethod
    def from_yaml(cls, path, width=None, height=None):
        c = yaml.safe_load(open(path))
        w, h = width or c["image_width"], height or c["image_height"]
        kw = dict(camera_height_m=c.get("camera_height_m", 1.4), horizon_y=c.get("horizon_y"),
                  real_heights_m=c.get("real_heights_m", {}))
        if "fx" in c:  # explicit intrinsics, rescaled if frame size differs from calibration
            sx, sy = w / c["image_width"], h / c["image_height"]
            return cls(w, h, c["fx"] * sx, c.get("fy", c["fx"]) * sy,
                       c.get("cx", c["image_width"] / 2) * sx, c.get("cy", c["image_height"] / 2) * sy, **kw)
        return cls.from_fov(w, h, c.get("hfov_deg", 70.0), **kw)

    @property
    def horizon(self) -> float:
        return self.cy if self.horizon_y is None else self.horizon_y


def distance_from_height(h_px: float, real_h_m: float, fy: float) -> float:
    """Similar triangles: h_px / fy = H / Z  ->  Z = fy * H / h_px."""
    return fy * real_h_m / max(h_px, 1e-6)


def distance_from_ground(y_bottom: float, cam: Camera) -> float | None:
    """Object base on the ground plane: Z = fy * cam_height / (y_bottom - y_horizon)."""
    dy = y_bottom - cam.horizon
    return cam.fy * cam.camera_height_m / dy if dy > 1 else None


def estimate_distance(box_xyxy, cls_name: str, cam: Camera, method: str = "auto") -> dict:
    """Return forward depth Z (m), lateral offset X (m, +right) and the method used.

    method: 'height' (known object size), 'ground' (ground-plane contact),
            'auto' (mean of both for ground-standing objects, height for signs on poles).
    """
    x1, y1, x2, y2 = map(float, box_xyxy)
    H = cam.real_heights_m.get(cls_name)
    z_h = distance_from_height(y2 - y1, H, cam.fy) if H else None
    z_g = distance_from_ground(y2, cam)
    on_ground = cls_name != "stop sign"  # sign face is elevated, its bbox bottom isn't on the ground
    if method == "height" or (method == "auto" and not on_ground):
        z, used = z_h, "height"
    elif method == "ground":
        z, used = z_g, "ground"
    else:
        vals = [v for v in (z_h, z_g) if v is not None]
        z, used = (sum(vals) / len(vals), "height+ground") if vals else (None, "none")
    if z is None:
        z, used = z_h, "height"
    x = ((x1 + x2) / 2 - cam.cx) * z / cam.fx if z else None
    return {"z": z, "x": x, "method": used}
