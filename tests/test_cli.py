import csv
import json

import numpy as np
import pytest

from ostk import cli
from ostk import geometry as g
from ostk import metrics
from ostk.labels import lid

nib = pytest.importorskip("nibabel")


def _phantom_label(D=96):
    ijk = np.argwhere(np.ones((D, D, D), dtype=bool)).astype(float)
    label = np.zeros((D, D, D), dtype=np.int16)
    flat = label.reshape(-1)

    def ball(c, r):
        return np.linalg.norm(ijk - np.asarray(c), axis=1) <= r

    def body(c, normal, radius=18.0, half=7.0):
        n = g.unit(normal)
        d = (ijk - np.asarray(c)) @ n
        inplane = np.linalg.norm((ijk - np.asarray(c)) - np.outer(d, n), axis=1)
        return (np.abs(d) <= half) & (inplane <= radius)

    flat[ball([22, 48, 22], 13.0)] = lid("femur_left")
    flat[ball([74, 48, 22], 13.0)] = lid("femur_right")
    flat[ball([22, 48, 30], 9.0)] = lid("left_hip")
    flat[ball([74, 48, 30], 9.0)] = lid("right_hip")
    tilts = np.linspace(12.0, -18.0, len(metrics.LL_ENDPLATE_CHAIN))
    zs = np.linspace(80.0, 40.0, len(metrics.LL_ENDPLATE_CHAIN))
    for lv, t, z in zip(metrics.LL_ENDPLATE_CHAIN, tilts, zs):
        a = np.deg2rad(t)
        flat[body([48, 48, z], [0.0, np.sin(a), np.cos(a)])] = lid(lv)
    return label


def test_spinopelvic_summary_from_label_phantom():
    lab = _phantom_label()
    s = metrics.spinopelvic_summary_from_label(lab, np.eye(4), case_id="p1")
    assert s["case_id"] == "p1" and s["supine_ct"] is True
    for k in ("PI", "SS", "PT", "LL"):
        assert s[k] is not None
    # both PI and LL present -> mismatch + schwab computed
    assert s["PI-LL"]["pi_minus_ll"] == round(s["PI"] - s["LL"], 3)
    assert set(s["schwab"]) >= {"PI-LL", "PT", "SVA", "objectives"}
    assert s["schwab"]["SVA"] == "out_of_scope"


def test_cli_all_writes_csv_and_jsonl(tmp_path):
    labels = tmp_path / "labels"
    labels.mkdir()
    img = nib.Nifti1Image(_phantom_label(), np.eye(4))
    nib.save(img, str(labels / "0001_seg.nii.gz"))

    out_csv = tmp_path / "summary.csv"
    rc = cli.main(["all", "--labels", str(labels), "--out", str(out_csv), "--workers", "1"])
    assert rc == 0 and out_csv.exists()
    rows = list(csv.DictReader(open(out_csv, encoding="utf-8")))
    assert len(rows) == 1 and rows[0]["case_id"] == "0001"
    assert rows[0]["PI"] and rows[0]["LL"]

    out_jsonl = tmp_path / "summary.jsonl"
    cli.main(["all", "--labels", str(labels), "--out", str(out_jsonl), "--workers", "1"])
    recs = [json.loads(l) for l in open(out_jsonl, encoding="utf-8")]
    assert len(recs) == 1 and recs[0]["case_id"] == "0001"


def test_cli_pi_csv(tmp_path):
    labels = tmp_path / "labels"
    labels.mkdir()
    nib.save(nib.Nifti1Image(_phantom_label(), np.eye(4)), str(labels / "0007_x.nii.gz"))
    out = tmp_path / "pi.csv"
    cli.main(["pi", "--labels", str(labels), "--out", str(out), "--workers", "1"])
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    assert rows[0]["parameter"] == "pelvic_incidence" and rows[0]["case_id"] == "0007"
