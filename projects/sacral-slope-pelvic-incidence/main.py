"""Reference implementation — PI / SS / PT from CTSpinoPelvic1K v3 masks.

Thin wrapper over the tested `ostk` primitives (femoral-head sphere fit + S1
endplate plane fit + sagittal projection). Run over a labels folder:

    python main.py --labels /path/to/labels --out pi.csv

Equivalent to `python -m ostk pi --labels ... --out ...`. The open student task
here is VALIDATION: compare `value` against manual radiographic PI on a subset
and report MAE / ICC / Bland–Altman (SPEC §7).
"""
import argparse

from ostk.cli import main as ostk_main


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", required=True, help="directory of NIfTI label maps")
    ap.add_argument("--out", help="output .csv or .jsonl (prints if omitted)")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    argv = ["pi", "--labels", a.labels]
    if a.out:
        argv += ["--out", a.out]
    if a.workers is not None:
        argv += ["--workers", str(a.workers)]
    return ostk_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
