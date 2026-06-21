import numpy as np

from ostk import io


def test_voxels_to_world_identity():
    ijk = np.array([[1, 2, 3], [4, 5, 6]])
    out = io.voxels_to_world(ijk, np.eye(4))
    assert np.allclose(out, ijk)


def test_voxels_to_world_scaled_translated():
    aff = np.diag([2.0, 3.0, 4.0, 1.0])
    aff[:3, 3] = [10, 20, 30]
    out = io.voxels_to_world([1, 1, 1], aff)
    assert np.allclose(out, [[12, 23, 34]])


def test_voxel_volume():
    aff = np.diag([2.0, 3.0, 4.0, 1.0])
    assert abs(io.voxel_volume_mm3(aff) - 24.0) < 1e-9
