# projects/

One folder per student miniproject. See the top-level README for how to add yours,
and [`../SPEC.md`](../SPEC.md) §6 for the clinical-utility build order.

| Project | Parameter | Status |
|---|---|---|
| [sacral-slope-pelvic-incidence](sacral-slope-pelvic-incidence/) | PI / SS / PT | ✅ reference impl in `ostk` (`main.py`) — open task is **manual validation** |
| [lordosis-trall-angle](lordosis-trall-angle/) | Lumbar lordosis (+ PI–LL, Schwab) | ✅ reference impl in `ostk` (`main.py`) — open task is **manual validation** |
| [scoliosis-cobb-angle](scoliosis-cobb-angle/) | Coronal Cobb | 🔨 open (reuse `ostk.geometry.cobb_angle`) |
| [lordosis-distribution-index](lordosis-distribution-index/) | LDI | 🔨 open (consumes per-segment LL) |
| [vertebral-body-wedging-index](vertebral-body-wedging-index/) | Wedging | 🔨 open |
| [disc-spacing](disc-spacing/) | Disc height | 🔨 open |
| [lumbar-vertebral-spacing](lumbar-vertebral-spacing/) | Centroid spacing | 🔨 open |
| [spondylolisthesis](spondylolisthesis/) | Slip / Meyerding | 🔨 open |
| [centroid-trajectory-tortuosity](centroid-trajectory-tortuosity/) | Tortuosity | 🔨 open |
| [osteoporosis-hu](osteoporosis-hu/) | Trabecular HU | 🔨 open (needs CT) |
| [spinal-stenosis](spinal-stenosis/) | Bony canal | ⚠️ limited (no canal label in v3) |
| [sagittal-vertical-axis](sagittal-vertical-axis/) | SVA | ❌ out of scope (needs C7 + standing) |
| [t1-pelvic-angle](t1-pelvic-angle/) | TPA | ❌ out of scope (needs T1 + standing) |

The ✅ projects have a working `ostk`-based reference implementation; the headline
student deliverable for those is the **validation** (MAE / ICC / Bland–Altman vs
manual). The 🔨 projects are open — reuse the tested `ostk` primitives.
