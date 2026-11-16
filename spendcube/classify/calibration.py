"""Calibrated confidence for the evidence-based (non-LLM) classification layers.

Raw offline scores (TF-IDF similarity of a line to the tree, the share of a
supplier's or group's category mix) are not probabilities. For each kind of
evidence a monotone mapping (isotonic regression) turns the score into the
probability that the chosen leaf is right. The mappings are fitted once on
companies A and B with their labels hidden (`python run.py calibrate`), each
company weighted equally, and stored as breakpoints in
spendcube/data/offline_calibration.json, so no labels are needed at run time.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

CALIB_FILE = Path(__file__).resolve().parents[1] / "data" / "offline_calibration.json"

# which score each family is calibrated on
FAMILY_SCORE = {"retrieval": "calib_sim1", "propagated": "calib_p1", "group": "calib_p1"}


def family(source: str) -> str | None:
    s = str(source or "")
    if s.endswith("+group"):
        return "group"
    if s.startswith("propagated"):
        return "propagated"
    if s == "retrieval":
        return "retrieval"
    return None


class OfflineCalibrator:
    def __init__(self, path: Path = CALIB_FILE):
        self.maps = {}
        if Path(path).exists():
            self.maps = json.loads(Path(path).read_text()).get("families", {})

    def available(self, fam: str) -> bool:
        return fam in self.maps

    def predict(self, fam: str, score, fallback) -> np.ndarray:
        """Probability that the leaf is right; `fallback` when the family has no mapping."""
        if fam not in self.maps:
            return np.asarray(fallback, dtype=float)
        m = self.maps[fam]
        return np.interp(np.asarray(score, dtype=float), m["x"], m["y"])


def ece(prob, correct, w=None, bins: int = 10) -> float:
    """Expected calibration error (weighted)."""
    prob, correct = np.asarray(prob, float), np.asarray(correct, float)
    w = np.ones(len(prob)) if w is None else np.asarray(w, float)
    idx = np.clip((prob * bins).astype(int), 0, bins - 1)
    tot, e = w.sum(), 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += w[m].sum() / tot * abs(np.average(correct[m], weights=w[m]) - np.average(prob[m], weights=w[m]))
    return float(e)


def fit_isotonic(score, correct, weight) -> dict:
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)
    iso.fit(np.asarray(score, float), np.asarray(correct, float), sample_weight=np.asarray(weight, float))
    return {"x": [float(v) for v in iso.X_thresholds_], "y": [round(float(v), 4) for v in iso.y_thresholds_]}


def fit(records: pd.DataFrame) -> dict:
    """records: family, score, correct (0/1), weight. Returns {family: {x, y, n}}."""
    out = {}
    for fam, g in records.groupby("family"):
        if len(g) < 50 or g["correct"].nunique() < 2:
            continue
        out[fam] = {**fit_isotonic(g["score"], g["correct"], g["weight"]), "n": int(len(g))}
    return out
