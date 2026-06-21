# ostk — OpenSpineToolbox kit

Reusable, **tested** primitives for building spinopelvic measurements from
CTSpinoPelvic1K masks. Functions are pure/stateless → reproducible (no RNG; fixed
sign conventions), low-latency (vectorised + closed-form fits), and picklable for
process-pool parallelism. See the shared contract in [`../SPEC.md`](../SPEC.md).

```bash
pip install -r ../requirements.txt
python -m pytest          # 14 analytic unit tests
```

**Modules**
- `io` — `load_label`, `load_ct`, `voxels_to_world`, `voxel_volume_mm3`
- `geometry` — `fit_sphere`, `fit_plane_tls`, `principal_axes`, `angle_between`, `project_out`, `unit`, `WORLD_SUPERIOR`
- `masks` — `binary_mask`, `mask_world`, `world_centroid`, `largest_component`, `surface_slab`, `endplate_points`
- `labels` — `LABELS`, `lid()` (the v3/v4 id scheme — no magic numbers)
- `record` — `Measurement` (the per-case output contract)
- `metrics` — `pelvic_incidence(...)` + `pelvic_incidence_from_label(...)`
- `parallel` — `map_cases(fn, items, workers)`

**Compose a measurement**
```python
from ostk import load_label, pelvic_incidence_from_label, map_cases

def pi_for(case_id):
    label, affine = load_label(f"labels/{case_id}.nii.gz")
    return pelvic_incidence_from_label(label, affine, case_id=case_id).to_dict()

results = map_cases(pi_for, case_ids, workers=8)   # parallel, one record per case
```

PI is the flagship (valid on supine CT); its absolute convention is to be confirmed
against manual radiographic PI (Paper 2, Aim 2). The geometry + the PI=SS+PT
identity are unit-tested on an analytic phantom.
