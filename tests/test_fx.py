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


def test_reporting_currency_lines_are_not_flagged_public_table(tmp_path):
    df = _frame([{"amount": "50", "currency": "USD", "date": "2024-03-10"},
                 {"amount": "50", "currency": "EUR", "date": "2024-03-10"}])
    out = clean(df, _cfg(), tmp_path)
    assert out.loc[0, "fx_source"] == "reporting" and "FX_PUBLIC_TABLE" not in out.loc[0, "flags"]
    assert "FX_PUBLIC_TABLE" in out.loc[1, "flags"]


def test_high_inflation_rate_from_posting_month_is_accepted(tmp_path):
    # client table: ARS from 100 to 1,000 units per USD over 33 months
    months = pd.date_range("2022-01-01", "2024-10-01", freq="MS")
    pd.DataFrame({"currency": "ARS", "month": months.strftime("%Y-%m"),
                  "units_per_usd": np.geomspace(100, 1000, len(months)).round(2)}).to_csv(tmp_path / "fx.csv",
                                                                                           index=False)
    rows = [{"amount": "990000", "currency": "ARS", "date": "2022-01-20", "posting_date": "2024-10-05",
             "fx_rate": "990"},                                   # booked at the posting month
            {"amount": "1000", "currency": "EUR", "date": "2024-03-10", "posting_date": "2024-03-20",
             "fx_rate": "18.6"}]                                  # another currency's rate: still rejected
    df = _frame(rows)
    cfg = _cfg({"table": "fx.csv"})
    cfg["sources"][0]["columns"]["posting_date"] = "posting_date"
    cfg["dates"] = {**cfg.get("dates", {}), "max_days_before_posting": 1100}
    out = clean(df, cfg, tmp_path)
    assert out.loc[0, "fx_source"] == "line" and out.loc[0, "spend_rc"] == pytest.approx(1000, rel=0.01)
    assert "FX_RATE_REJECTED" in out.loc[1, "flags"]


def test_stale_public_month_does_not_override_a_plausible_line_rate(tmp_path):
    # the shipped table stops early for some currencies; a 2024 ARS line rate must stand
    df = _frame([{"amount": "900000", "currency": "ARS", "date": "2024-06-10", "fx_rate": "900"},
                 {"amount": "450000", "currency": "ARS", "date": "2024-06-12"}])
    out = clean(df, _cfg(), tmp_path)
    assert out.loc[0, "fx_source"] == "line" and out.loc[0, "spend_rc"] == pytest.approx(1000, rel=0.01)
    assert out.loc[1, "fx_source"] == "file_median" and out.loc[1, "spend_rc"] == pytest.approx(500, rel=0.01)
