"""FX chain: line rate (checked) -> client table (orientation checked) -> shipped public
ECB table -> median of the file's own rates. Synthetic data only."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spendcube.clean import PUBLIC_FX, clean, load_public_fx, orient_table  # noqa: E402
from spendcube.config import DEFAULTS, _deep_merge  # noqa: E402


def _cfg(fx=None):
    c = _deep_merge(DEFAULTS, {"reporting_currency": "USD",
                               "sources": [{"name": "s", "file": "x", "record_id": "id", "columns": {"amount": "a"}}]})
    if fx:
        c["fx"].update(fx)
    return c


def _frame(rows):
    df = pd.DataFrame(rows)
    df["source"] = "s"
    df["record_id"] = [f"r{i}" for i in range(len(df))]
    for c in ("description", "supplier_name", "status", "item_code", "supplier_id", "fx_rate"):
        if c not in df:
            df[c] = ""
    return df


def test_public_table_is_shipped_and_sane():
    assert PUBLIC_FX.exists()
    pub = load_public_fx("USD")
    eur = pub[(pub.currency == "EUR") & (pub.month == "2024-03-01")]["rate"].iloc[0]
    inr = pub[(pub.currency == "INR") & (pub.month == "2024-03-01")]["rate"].iloc[0]
    assert 0.85 < eur < 1.0 and 75 < inr < 90
    assert pub.groupby("currency").size().get("SAR", 0) > 0          # USD peg included


def test_no_table_uses_public_rates_and_never_zero(tmp_path):
    df = _frame([{"amount": "1000", "currency": "EUR", "date": "2024-03-10"},
                 {"amount": "8300", "currency": "INR", "date": "2024-03-10"},
                 {"amount": "50", "currency": "USD", "date": "2024-03-10"}])
    out = clean(df, _cfg(), tmp_path)
    assert (out["spend_rc"] > 0).all()
    assert out.loc[0, "fx_source"] == "public_table"
    assert out.loc[1, "spend_rc"] == pytest.approx(100, rel=0.05)
    assert out.loc[2, "spend_rc"] == 50


def test_inverted_line_rate_is_detected(tmp_path):
    # a rate of 1.087 USD per EUR given where units of EUR per USD (~0.92) is expected
    df = _frame([{"amount": "1000", "currency": "EUR", "date": "2024-03-10", "fx_rate": "1.087"},
                 {"amount": "1000", "currency": "EUR", "date": "2024-03-10", "fx_rate": "0.92"},
                 {"amount": "1000", "currency": "EUR", "date": "2024-03-10", "fx_rate": "18.6"}])
    out = clean(df, _cfg(), tmp_path)
    assert out.loc[0, "fx_source"] == "line_inverted" and "FX_RATE_INVERTED" in out.loc[0, "flags"]
    assert out.loc[0, "spend_rc"] == pytest.approx(1087, rel=0.01)
    assert out.loc[1, "fx_source"] == "line"
    assert "FX_RATE_REJECTED" in out.loc[2, "flags"] and out.loc[2, "spend_rc"] == pytest.approx(1087, rel=0.05)


def test_client_table_on_inverse_base_is_inverted():
    pub = load_public_fx("USD")
    m = pub[pub.month >= "2024-01-01"]
    wrong = m.assign(rate=1.0 / m["rate"])           # USD per unit instead of units per USD
    fixed, note = orient_table(wrong, pub)
    assert "inverted" in note
    assert np.allclose(fixed["rate"].values, m["rate"].values)


def test_unknown_currency_uses_file_median(tmp_path):
    df = _frame([{"amount": "100", "currency": "XYZ", "date": "2024-03-10", "fx_rate": "2.0"},
                 {"amount": "100", "currency": "XYZ", "date": "2024-03-10", "fx_rate": ""}])
    out = clean(df, _cfg(), tmp_path)
    assert out.loc[0, "fx_source"] == "line"
    assert out.loc[1, "fx_source"] == "file_median" and out.loc[1, "spend_rc"] == 50


def test_offline_calibration_is_monotone_and_bounded():
    from spendcube.classify.calibration import CALIB_FILE, OfflineCalibrator
    assert CALIB_FILE.exists()
    cal = OfflineCalibrator()
    assert cal.available("retrieval")
    s = np.linspace(0, 0.6, 50)
    p = cal.predict("retrieval", s, np.zeros(50))
    assert (np.diff(p) >= -1e-12).all() and p.min() >= 0.0 and p.max() <= 1.0
    assert cal.predict("unknown-family", [0.3], [0.123])[0] == 0.123   # falls back
