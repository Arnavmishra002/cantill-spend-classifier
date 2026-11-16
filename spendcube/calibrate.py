"""`run.py calibrate`: fit the offline-confidence model on labelled companies with
their labels hidden, check it transfers across companies, write the coefficients."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .classify import load_labels
from .classify.calibration import CALIB_FILE, FAMILY_SCORE, ece, family, fit
from .config import load_config
from .paths import REPO_ROOT
from .pipeline import run


def collect(config: str, data: str) -> pd.DataFrame:
    """Zero-shot offline run (labels hidden); one record per scored line."""
    cfg = load_config(config)
    tag = Path(config).stem
    rep = run(config, data, str(REPO_ROOT / "out" / "eval" / f"calib_{tag}.csv"), use_llm=False,
              use_labels=False, cache_dir=str(REPO_ROOT / ".cache"), health={"usable": False})
    df = rep["_df"]
    labels = load_labels(cfg, Path(data), rep["_tax"])
    lab = labels[labels["node"].notna()].set_index("record_id")["node"]
    d = df[(df["include"] == 1) & df["record_id"].isin(lab.index)].copy()
    d["family"] = d["class_source"].map(family)
    d = d[d["family"].notna()]
    d["score"] = [r[FAMILY_SCORE[f]] for f, r in zip(d["family"], d.to_dict("records"))]
    d = d[d["score"].notna()]
    d["correct"] = (d["node"] == d["record_id"].map(lab)).astype(int)
    d["company"] = cfg.get("company")
    d["raw_conf"] = d["confidence"].astype(float)
    return d[["company", "family", "score", "correct", "raw_conf"]]


def calibrate(pairs: list[tuple[str, str]], out: Path = CALIB_FILE) -> dict:
    if Path(out).exists():
        Path(out).unlink()          # collect with the uncalibrated scores
    recs = pd.concat([collect(c, d) for c, d in pairs], ignore_index=True)
    recs["weight"] = 1.0 / recs.groupby("company")["company"].transform("size")   # companies weigh the same
    report = {}
    for test in sorted(recs["company"].unique()):   # honest check: fit on the others, test on this one
        train, tst = recs[recs["company"] != test], recs[recs["company"] == test]
        maps = fit(train)
        pred = tst["raw_conf"].values.copy()
        for fam, m in maps.items():
            sel = (tst["family"] == fam).values
            pred[sel] = np.interp(tst.loc[sel, "score"].values, m["x"], m["y"])
        pooled = fit(recs)
        pin = tst["raw_conf"].values.copy()
        for fam, m in pooled.items():
            sel = (tst["family"] == fam).values
            pin[sel] = np.interp(tst.loc[sel, "score"].values, m["x"], m["y"])
        report[test] = {"lines": int(len(tst)), "leaf_accuracy": round(float(tst["correct"].mean()), 4),
                        "median_conf_raw": round(float(np.median(tst["raw_conf"])), 4),
                        "median_conf_calibrated": round(float(np.median(pin)), 4),
                        "ece_raw": round(ece(tst["raw_conf"], tst["correct"]), 4),
                        "ece_fit_on_other_company": round(ece(pred, tst["correct"]), 4),
                        "ece_fit_on_both": round(ece(pin, tst["correct"]), 4)}
    payload = {"families": fit(recs), "fitted_on": sorted(recs["company"].unique().tolist()),
               "fitted_at": time.strftime("%Y-%m-%d"),
               "target": "probability that the chosen leaf (deepest level) is correct",
               "method": "isotonic regression per kind of evidence, companies weighted equally",
               "report": report}
    Path(out).write_text(json.dumps(payload, indent=1))
    return payload
