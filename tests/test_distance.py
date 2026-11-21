import math
from src.distance import Camera, estimate_distance


def cam():
    return Camera.from_fov(1280, 720, 70, camera_height_m=1.4)


def test_fx_from_fov():
    assert abs(cam().fx - 640 / math.tan(math.radians(35))) < 1e-6


def test_height_method_roundtrip():
    c = cam(); Z = 5.0
    h_px = c.fy * 0.45 / Z
    r = estimate_distance([600, 400, 620, 400 + h_px], "cone", c, "height")
    assert abs(r["z"] - Z) < 1e-6


def test_ground_method_roundtrip():
    c = cam(); Z = 8.0
    y_bottom = c.horizon + c.fy * c.camera_height_m / Z
    r = estimate_distance([600, y_bottom - 20, 620, y_bottom], "cone", c, "ground")
    assert abs(r["z"] - Z) < 1e-6


def test_closer_is_bigger():
    c = cam()
    assert estimate_distance([0, 0, 25, 50], "stop sign", c)["z"] > estimate_distance([0, 0, 50, 100], "stop sign", c)["z"]


def test_lateral_sign():
    assert estimate_distance([900, 300, 940, 400], "stop sign", cam())["x"] > 0
