import numpy as np

from ostk import geometry as g
from ostk import spine


def _vertebra(center, tilt_deg, ap_extent=30.0, lr_extent=45.0, height=24.0,
              n=6000, seed=0):
    """A solid box vertebral BODY with its superior endplate tilted `tilt_deg`
    about the L–R axis (anterior edge lower), plus a posterior 'spinous process'
    spike that must NOT influence the endplate fit."""
    rng = np.random.default_rng(seed)
    # body in local coords: x=L-R, y=A-P, z=cranio-caudal
    b = rng.uniform(-1, 1, size=(n, 3)) * np.array([lr_extent, ap_extent, height]) / 2
    a = np.deg2rad(tilt_deg)
    R = np.array([[1, 0, 0],
                  [0, np.cos(a), -np.sin(a)],
                  [0, np.sin(a), np.cos(a)]])
    body = b @ R.T + np.asarray(center)
    # posterior spike (−y), well below the endplate, off to one side
    s = rng.uniform(-1, 1, size=(n // 4, 3)) * np.array([8, 10, 30])
    s += np.array([0, -ap_extent * 0.9, -height * 0.2])
    spike = s @ R.T + np.asarray(center)
    return np.vstack([body, spike])


def test_fit_endplate_recovers_tilt_and_ignores_posterior():
    pts = _vertebra(center=[0, 0, 100.0], tilt_deg=20.0)
    res = spine.fit_endplate(pts, which="superior")
    assert res is not None
    c, n, rms = res
    # superior endplate normal tilted 20° from vertical, cranial (+z)
    assert n[2] > 0
    tilt = np.degrees(np.arccos(np.clip(abs(n @ np.array([0, 0, 1.0])), 0, 1)))
    assert abs(tilt - 20.0) < 6.0          # recovered tilt despite posterior spike


def test_fit_endplate_flat_is_horizontal():
    pts = _vertebra(center=[0, 0, 0.0], tilt_deg=0.0, seed=3)
    c, n, rms = spine.fit_endplate(pts, which="superior")
    tilt = np.degrees(np.arccos(np.clip(abs(n @ np.array([0, 0, 1.0])), 0, 1)))
    assert tilt < 5.0


def test_fit_endplate_too_few_points():
    assert spine.fit_endplate(np.zeros((5, 3))) is None


def test_anterior_axis_orientation():
    ap = spine.anterior_axis()
    assert ap[1] > 0                       # points to +Y (anterior)
    assert abs(ap @ np.array([0, 0, 1.0])) < 1e-9
