"""ostk.io — NIfTI load + voxel↔world helpers. All geometry downstream is in
world millimetres via the affine (never voxel-index space)."""
from __future__ import annotations

from typing import Tuple

import numpy as np


def load_nifti(path) -> Tuple[np.ndarray, np.ndarray]:
    """Return (data array, 4x4 affine). nibabel is imported lazily so the pure
    geometry layer has no hard NIfTI dependency."""
    import nibabel as nib
    img = nib.load(str(path))
    return np.asarray(img.dataobj), np.asarray(img.affine, dtype=np.float64)


def load_label(path) -> Tuple[np.ndarray, np.ndarray]:
    arr, affine = load_nifti(path)
    return np.asarray(arr).astype(np.int32), affine


def load_ct(path) -> Tuple[np.ndarray, np.ndarray]:
    arr, affine = load_nifti(path)
    return np.asarray(arr).astype(np.float32), affine


def voxels_to_world(ijk, affine) -> np.ndarray:
    """Map voxel indices (N,3) or (3,) to world mm via the affine. Returns (N,3)."""
    ijk = np.asarray(ijk, dtype=np.float64)
    if ijk.ndim == 1:
        ijk = ijk[None, :]
    homog = np.c_[ijk, np.ones(len(ijk))]
    return (homog @ np.asarray(affine, dtype=np.float64).T)[:, :3]


def voxel_volume_mm3(affine) -> float:
    """Volume of one voxel in mm^3 (|det| of the affine's 3x3 block)."""
    return float(abs(np.linalg.det(np.asarray(affine, dtype=np.float64)[:3, :3])))
