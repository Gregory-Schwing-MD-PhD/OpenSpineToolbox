"""ostk.spine — vertebral-body endplate fitting (a reusable primitive).

The superior/inferior endplate is the disc-bearing surface of the vertebral
BODY. Two facts make a naive fit wrong:

  * Posterior elements (canal, facets, spinous/transverse processes) are one
    connected component with the body in a 3-D mask, so they can't be split off
    by connectivity — they must be dropped by ANTERIOR position.
  * The endplate is tilted (sacral slope, wedging), so a flat "top-N% by height"
    slab under-reads the tilt. The true face is the extreme voxel per in-plane
    column along the cranio-caudal axis.

`fit_endplate` handles both and returns a plane (centroid, cranial unit normal,
rms). It's used by `ostk.metrics` (lumbar lordosis) and the demo exporter, and is
the place to improve endplate fitting for the whole toolbox.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .geometry import WORLD_SUPERIOR, fit_plane_tls, unit


def anterior_axis(normal_axis=WORLD_SUPERIOR, lr=(1.0, 0.0, 0.0)) -> np.ndarray:
    """Unit anterior axis: in the sagittal plane (⊥ L–R and ⊥ cranial), oriented
    to world +Y (RAS anterior)."""
    ap = unit(np.cross(np.asarray(lr, float), unit(normal_axis)))
    return ap if ap @ np.array([0.0, 1.0, 0.0]) >= 0 else -ap


def endplate_surface(points, normal_axis=WORLD_SUPERIOR, which: str = "superior",
                     ant_frac: float = 0.6, nbins: int = 22,
                     lr=(1.0, 0.0, 0.0)) -> np.ndarray:
    """The endplate face of a body point cloud (N,3 world mm): keep the anterior
    `ant_frac` of the body (drops posterior elements), then the extreme voxel per
    in-plane column along `normal_axis` (topmost for 'superior')."""
    P = np.asarray(points, dtype=np.float64)
    if len(P) == 0:
        return P
    a = unit(normal_axis)
    if 0.0 < ant_frac < 1.0:
        proj = (P - P.mean(0)) @ anterior_axis(a, lr)
        P = P[proj >= np.quantile(proj, 1.0 - ant_frac)]
        if len(P) == 0:
            return P
    ref = np.array([1.0, 0, 0]) if abs(a @ np.array([1.0, 0, 0])) < 0.9 else np.array([0, 1.0, 0])
    e1 = unit(ref - (ref @ a) * a)
    e2 = np.cross(a, e1)
    u, v, w = P @ e1, P @ e2, P @ a
    ui = np.floor((u - u.min()) / (np.ptp(u) + 1e-9) * nbins).astype(int)
    vi = np.floor((v - v.min()) / (np.ptp(v) + 1e-9) * nbins).astype(int)
    key = ui * (nbins + 1) + vi
    sgn = -1.0 if which == "superior" else 1.0          # superior -> max w first
    order = np.lexsort((sgn * w, key))
    sk = key[order]
    first = np.ones(len(order), bool)
    first[1:] = sk[1:] != sk[:-1]
    return P[order[first]]


def fit_endplate(points, normal_axis=WORLD_SUPERIOR, which: str = "superior",
                 ant_frac: float = 0.6, lr=(1.0, 0.0, 0.0), min_points: int = 30
                 ) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """Fit the superior/inferior endplate plane of a vertebral-body point cloud.
    Returns (centroid, unit normal oriented cranially for 'superior', rms) or None
    if there aren't enough points."""
    P = np.asarray(points, dtype=np.float64)
    if len(P) < min_points:
        return None
    surf = endplate_surface(P, normal_axis, which, ant_frac, lr=lr)
    if len(surf) < min_points:
        surf = P
    c, n, rms = fit_plane_tls(surf)
    # one robust pass: drop the worst-fitting quartile (rim voxels where the
    # per-column extreme is a side wall, not the face) and refit.
    d = np.abs((surf - c) @ n)
    keep = d <= np.quantile(d, 0.75)
    if keep.sum() >= min_points:
        c, n, rms = fit_plane_tls(surf[keep])
    a = unit(normal_axis)
    cranial = n @ a >= 0
    if (which == "superior") != cranial:
        n = -n
    return c, n, rms


def endplate_from_label(label, affine, level: str, which: str = "superior",
                        normal_axis=WORLD_SUPERIOR, ant_frac: float = 0.6,
                        lr=(1.0, 0.0, 0.0), min_points: int = 30):
    """Convenience: fit an endplate straight from a label volume + structure name.
    For S1 falls back to the sacrum label if the carved S1 is absent."""
    from .labels import lid
    from .masks import binary_mask, largest_component, mask_world
    m = binary_mask(label, lid(level))
    if level == "S1" and not m.any():
        m = binary_mask(label, lid("sacrum"))
    pts = mask_world(largest_component(m), affine)
    return fit_endplate(pts, normal_axis, which, ant_frac, lr, min_points)
