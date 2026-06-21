import numpy as np
import pytest

from ostk import geometry as g


def test_fit_sphere_recovers_center_and_radius():
    rng = np.random.default_rng(0)
    c0, r0 = np.array([10.0, -5.0, 30.0]), 25.0
    dirs = rng.normal(size=(2000, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    pts = c0 + r0 * dirs + rng.normal(scale=0.05, size=(2000, 3))
    c, r, rms = g.fit_sphere(pts)
    assert np.allclose(c, c0, atol=0.2)
    assert abs(r - r0) < 0.2
    assert rms < 0.1


def test_fit_sphere_partial_cap():
    # only a cap of the sphere (FOV-clipped femoral head) still recovers center
    rng = np.random.default_rng(1)
    c0, r0 = np.array([0.0, 0.0, 0.0]), 20.0
    d = rng.normal(size=(4000, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    d = d[d[:, 2] > 0.4]                       # keep a cap
    pts = c0 + r0 * d
    c, r, _ = g.fit_sphere(pts)
    assert np.allclose(c, c0, atol=0.5)
    assert abs(r - r0) < 0.5


def test_fit_plane_normal_and_residual():
    rng = np.random.default_rng(2)
    a, b, c = 0.3, -0.2, 5.0                   # plane z = a x + b y + c
    xy = rng.uniform(-50, 50, size=(1000, 2))
    z = a * xy[:, 0] + b * xy[:, 1] + c
    pts = np.c_[xy, z]
    m, n, rms = g.fit_plane_tls(pts)
    true_n = g.unit(np.array([-a, -b, 1.0]))
    assert abs(abs(n @ true_n) - 1.0) < 1e-6   # parallel (sign-agnostic)
    assert rms < 1e-6


def test_principal_axes_anisotropic_and_deterministic():
    rng = np.random.default_rng(3)
    pts = rng.normal(size=(3000, 3)) * np.array([10.0, 3.0, 1.0])
    V, w, m = g.principal_axes(pts)
    assert w[0] > w[1] > w[2]
    assert abs(abs(V[:, 0] @ np.array([1.0, 0, 0])) - 1.0) < 0.05  # long axis ~ x
    V2, _, _ = g.principal_axes(pts)
    assert np.array_equal(V, V2)               # reproducible


def test_angle_and_projection():
    assert abs(g.angle_between([1, 0, 0], [0, 1, 0]) - 90.0) < 1e-9
    assert g.angle_between([1, 0, 0], [2, 0, 0]) < 1e-9
    out = g.project_out(np.array([1.0, 2.0, 3.0]), [0, 0, 1])
    assert abs(out @ np.array([0, 0, 1.0])) < 1e-12
