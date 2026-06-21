"""OpenSpineToolbox kit (ostk) — reusable, tested primitives for building
spinopelvic measurements from CTSpinoPelvic1K masks. See SPEC.md."""
from . import geometry, io, labels, masks, metrics, parallel, record
from .geometry import (WORLD_SUPERIOR, angle_between, fit_plane_tls, fit_sphere,
                       principal_axes, project_out, unit)
from .io import load_ct, load_label, voxel_volume_mm3, voxels_to_world
from .labels import LABELS, lid
from .masks import (binary_mask, endplate_points, largest_component,
                    mask_world, surface_slab, world_centroid)
from .metrics import pelvic_incidence, pelvic_incidence_from_label
from .parallel import map_cases
from .record import Measurement

__all__ = [
    "geometry", "io", "labels", "masks", "metrics", "parallel", "record",
    "WORLD_SUPERIOR", "angle_between", "fit_plane_tls", "fit_sphere",
    "principal_axes", "project_out", "unit",
    "load_ct", "load_label", "voxel_volume_mm3", "voxels_to_world",
    "LABELS", "lid",
    "binary_mask", "endplate_points", "largest_component", "mask_world",
    "surface_slab", "world_centroid",
    "pelvic_incidence", "pelvic_incidence_from_label",
    "map_cases", "Measurement",
]
