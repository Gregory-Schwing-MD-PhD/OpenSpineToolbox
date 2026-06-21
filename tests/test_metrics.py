import numpy as np

from ostk import geometry as g
from ostk import metrics


def _sphere_points(center, radius, n=1500, seed=0):
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return np.asarray(center) + radius * d


def _plane_points(midpoint, normal, half=40.0, step=4.0):
    n = g.unit(normal)
    u1 = g.unit(g.project_out(np.array([1.0, 0, 0]), n))
    u2 = g.unit(np.cross(n, u1))
    ts = np.arange(-half, half + step, step)
    a, b = np.meshgrid(ts, ts)
    grid = a.ravel()[:, None] * u1 + b.ravel()[:, None] * u2
    return np.asarray(midpoint) + grid


def test_pelvic_incidence_phantom_known_angles():
    # Phantom with analytic PI=60, SS=40, PT=20 (opposite-tilt convention).
    cL, cR = np.array([-80.0, 0, 0]), np.array([80.0, 0, 0])
    PT, SS = np.deg2rad(20.0), np.deg2rad(40.0)
    P = 150.0 * np.array([0.0, np.sin(PT), np.cos(PT)])          # hip→endplate, PT from vertical
    n = np.array([0.0, -np.sin(SS), np.cos(SS)])                 # endplate normal, SS from vertical

    ep = _plane_points(P, n)
    fhl = _sphere_points(cL, 25.0, seed=1)
    fhr = _sphere_points(cR, 25.0, seed=2)

    r = metrics.pelvic_incidence(ep, fhl, fhr)
    assert abs(r["PI"] - 60.0) < 0.5
    assert abs(r["SS"] - 40.0) < 0.5
    assert abs(r["PT"] - 20.0) < 0.5
    assert abs(r["SS"] + r["PT"] - r["PI"]) < 0.5               # geometric identity


def test_pelvic_incidence_is_reproducible():
    cL, cR = np.array([-70.0, 0, 0]), np.array([70.0, 0, 0])
    P = np.array([0.0, 40.0, 130.0])
    n = np.array([0.0, -0.5, 0.8])
    ep, fhl, fhr = (_plane_points(P, n),
                    _sphere_points(cL, 22.0, seed=5),
                    _sphere_points(cR, 22.0, seed=6))
    a = metrics.pelvic_incidence(ep, fhl, fhr)
    b = metrics.pelvic_incidence(ep, fhl, fhr)
    assert a["PI"] == b["PI"] and a["SS"] == b["SS"] and a["PT"] == b["PT"]
