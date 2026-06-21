"""Reference implementation — Lumbar Lordosis (+ PI-LL mismatch, SRS-Schwab).

Thin wrapper over the tested `ostk` primitives. `all` emits the full Greenberg
§73 spinopelvic summary (PI/SS/PT, LL + per-segment, PI-LL, Schwab, Eq. 73.1):

    python main.py --labels /path/to/labels --out summary.csv

Equivalent to `python -m ostk all --labels ... --out ...`. Open student task:
VALIDATION of LL against manual measurement (SPEC §7).
"""
import argparse

from ostk.cli import main as ostk_main


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", required=True, help="directory of NIfTI label maps")
    ap.add_argument("--out", help="output .csv or .jsonl (prints if omitted)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--ll-only", action="store_true",
                    help="emit only lumbar lordosis (default emits full summary)")
    a = ap.parse_args()
    argv = ["ll" if a.ll_only else "all", "--labels", a.labels]
    if a.out:
        argv += ["--out", a.out]
    if a.workers is not None:
        argv += ["--workers", str(a.workers)]
    return ostk_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
