"""ostk.masks — mask extraction + surface helpers over integer label volumes.

Vectorised; the one connected-components call uses scipy.ndimage (C-level)."""
from __future__ import annotations

from typing import Optional

import numpy as np

from .io import voxels_to_world


def binary_mask(label, idx: int) -> np.ndarray:
    return np.asarray(label) == idx


def mask_indices(mask) -> np.ndarray:
    """(N,3) voxel indices of the True voxels."""
    return np.argwhere(np.asarray(mask, dtype=bool))


def mask_world(mask, affine) -> np.ndarray:
    """(N,3) world-mm coords of the mask voxels (empty (0,3) if none)."""
    idx = mask_indices(mask)
    if len(idx) == 0:
        return np.empty((0, 3))
    return voxels_to_world(idx, affine)


def world_centroid(mask, affine) -> Optional[np.ndarray]:
    w = mask_world(mask, affine)
    return None if len(w) == 0 else w.mean(axis=0)


def largest_component(mask) -> np.ndarray:
    """Keep the single largest 3-D connected component (deterministic)."""
    from scipy import ndimage
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return m
    lab, n = ndimage.label(m)
    if n <= 1:
        return m
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    return lab == int(counts.argmax())


def surface_slab(points_world, axis, which: str = "superior",
                 frac: float = 0.12) -> np.ndarray:
    """The `frac` of points furthest along `axis` (which='superior') or against
    it ('inferior') — i.e. the cranial / caudal articular slab of a body.
    `points_world` is (N,3)."""
    P = np.asarray(points_world, dtype=np.float64)
    if len(P) == 0:
        return P
    a = axis / np.linalg.norm(axis)
    proj = P @ a
    if which == "superior":
        return P[proj >= np.quantile(proj, 1.0 - frac)]
    if which == "inferior":
        return P[proj <= np.quantile(proj, frac)]
    raise ValueError("which must be 'superior' or 'inferior'")


def endplate_points(body_mask, affine, axis, which: str = "superior",
                    frac: float = 0.12) -> np.ndarray:
    """World points on the superior/inferior endplate of a vertebral body:
    the cranial/caudal slab along `axis` (e.g. the body's long axis, or
    geometry.WORLD_SUPERIOR)."""
    return surface_slab(mask_world(body_mask, affine), axis, which, frac)
