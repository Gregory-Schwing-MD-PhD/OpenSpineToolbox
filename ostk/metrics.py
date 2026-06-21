"""ostk.metrics — measurements composed from the primitives.

Flagship: pelvic incidence (PI) + sacral slope (SS) + pelvic tilt (PT). PI is the
one sagittal parameter valid on supine CT (posture-invariant). Convention:
endplate normal oriented cranially; pelvic radius = hip-axis → sacral-endplate
midpoint; all angles taken in the patient sagittal plane (normal ⟂ the L–R
bicoxofemoral axis). The absolute convention is to be confirmed against manual
radiographic PI (Paper 2, Aim 2) — the geometry/identity is unit-tested.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from .geometry import (WORLD_SUPERIOR, angle_between, fit_plane_tls, fit_sphere,
                       project_out, unit)
from .record import Measurement

PI_METHOD_VERSION = "pi-v1"


def pelvic_incidence(endplate_points, femhead_left_points, femhead_right_points,
                     sup_axis=WORLD_SUPERIOR) -> Dict:
    """PI/SS/PT from three world-mm point clouds: the S1 superior endplate and
    the two femoral heads. Returns a dict with the angles, landmarks, residuals."""
    m, n, ep_rms = fit_plane_tls(endplate_points)
    cL, rL, eL = fit_sphere(femhead_left_points)
    cR, rR, eR = fit_sphere(femhead_right_points)
    bicox = 0.5 * (cL + cR)

    lr = unit(cR - cL)                                  # left–right axis
    n_s = unit(project_out(n, lr))                     # endplate normal in sagittal plane
    radius = project_out(m - bicox, lr)                # hip axis → endplate midpoint
    sup_s = unit(project_out(sup_axis, lr))            # vertical in sagittal plane
    if n_s @ sup_s < 0:                                # orient endplate normal cranially
        n_s = -n_s

    PI = angle_between(n_s, radius)
    SS = angle_between(n_s, sup_s)
    PT = angle_between(radius, sup_s)
    return {
        "PI": PI, "SS": SS, "PT": PT,
        "landmarks_world_mm": {
            "endplate_midpoint": m.tolist(),
            "femhead_left": cL.tolist(), "femhead_right": cR.tolist(),
            "bicoxofemoral": bicox.tolist()},
        "fit_residuals": {
            "s1_endplate_rms": ep_rms,
            "femhead_left_rms": eL, "femhead_right_rms": eR,
            "femhead_left_radius": rL, "femhead_right_radius": rR},
    }


def pelvic_incidence_from_label(label, affine, *, case_id: str = "",
                                sup_axis=WORLD_SUPERIOR, endplate_frac: float = 0.15,
                                head_frac: float = 0.35,
                                min_voxels: int = 50) -> Measurement:
    """Compose PI from a v3 label volume. Extraction (approximate — flagged for
    the manual-validation pass): S1 superior endplate = cranial slab of S1
    (fallback sacrum); femoral head = cranial slab of each femur (proximal end).
    Returns a Measurement with QC flags (never silently drops a bad case)."""
    from .labels import lid
    from .masks import (binary_mask, endplate_points, largest_component,
                        mask_world, surface_slab)

    flags: list = []
    s1 = binary_mask(label, lid("S1"))
    src = s1 if s1.any() else binary_mask(label, lid("sacrum"))
    ep = endplate_points(largest_component(src), affine, sup_axis, "superior",
                         endplate_frac)
    fl = mask_world(largest_component(binary_mask(label, lid("femur_left"))), affine)
    fr = mask_world(largest_component(binary_mask(label, lid("femur_right"))), affine)
    fhl = surface_slab(fl, sup_axis, "superior", head_frac)
    fhr = surface_slab(fr, sup_axis, "superior", head_frac)

    for nm, arr in (("S1", ep), ("femur_left", fhl), ("femur_right", fhr)):
        if len(arr) < min_voxels:
            flags.append(f"low_voxels:{nm}")
    if flags:
        return Measurement(case_id=case_id, parameter="pelvic_incidence",
                           value=None, qc_flags=flags,
                           method_version=PI_METHOD_VERSION)

    r = pelvic_incidence(ep, fhl, fhr, sup_axis)
    if abs(r["SS"] + r["PT"] - r["PI"]) > 1.0:         # geometric identity check
        flags.append("identity_violation")
    flags = flags or ["ok"]
    return Measurement(
        case_id=case_id, parameter="pelvic_incidence", value=round(r["PI"], 3),
        landmarks_world_mm=r["landmarks_world_mm"], fit_residuals=r["fit_residuals"],
        qc_flags=flags, method_version=PI_METHOD_VERSION)
