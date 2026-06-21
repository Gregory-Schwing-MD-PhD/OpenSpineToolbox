import numpy as np
import pytest

from ostk import masks


def test_world_centroid_block_identity_affine():
    lab = np.zeros((10, 10, 10), dtype=np.int32)
    lab[2:5, 2:5, 2:5] = 7                       # block, true centre = (3,3,3)
    c = masks.world_centroid(masks.binary_mask(lab, 7), np.eye(4))
    assert np.allclose(c, [3.0, 3.0, 3.0])


def test_world_centroid_empty():
    assert masks.world_centroid(np.zeros((4, 4, 4)), np.eye(4)) is None


def test_surface_slab_superior_vs_inferior():
    pts = np.random.default_rng(0).uniform(0, 100, size=(2000, 3))
    sup = masks.surface_slab(pts, [0, 0, 1], "superior", 0.1)
    inf = masks.surface_slab(pts, [0, 0, 1], "inferior", 0.1)
    assert sup[:, 2].mean() > inf[:, 2].mean()
    assert len(sup) > 0 and len(inf) > 0


def test_largest_component_picks_bigger_blob():
    scipy = pytest.importorskip("scipy")          # noqa: F841
    m = np.zeros((20, 20, 20), dtype=bool)
    m[1:3, 1:3, 1:3] = True                        # small blob (8 vox)
    m[10:15, 10:15, 10:15] = True                  # big blob (125 vox)
    keep = masks.largest_component(m)
    assert keep[12, 12, 12] and not keep[2, 2, 2]
    assert keep.sum() == 125
