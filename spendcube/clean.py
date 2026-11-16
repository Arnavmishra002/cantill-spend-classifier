"""Parse amounts/dates/rates, remove tax, convert to reporting currency.

Adds columns:
  amount_num   parsed line amount (source currency, as given)
  net_amount   tax removed when the source is gross
  quantity_num, tax_rate_num
  date_parsed, posting_date_parsed, fx_date
  currency_norm, fx_rate_used, fx_source
  spend_rc     net amount in the reporting currency
  flags        ';'-joined data-quality flags (UNPARSED_AMOUNT, UNPARSED_DATE, ...)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .ingest import read_table
from .paths import resolve_file
from .textnorm import clean_text, parse_date_series, parse_number_series

log = logging.getLogger(__name__)


def _add_flag(df: pd.DataFrame, mask, flag: str):
    mask = pd.Series(mask, index=df.index).fillna(False).astype(bool)
    if mask.any():
        df.loc[mask, "flags"] = df.loc[mask, "flags"].map(lambda f: f"{f};{flag}" if f else flag)


def _col(df, name):
    return df[name] if name in df.columns else pd.Series("", index=df.index)


def load_fx_table(cfg: dict, data_dir: Path) -> pd.DataFrame | None:
    fx = cfg.get("fx") or {}
    if not fx.get("table"):
        return None
    t = read_table(resolve_file(fx["table"], cfg, data_dir))
    c = fx.get("table_columns", {})
    out = pd.DataFrame({
        "currency": t[c.get("currency", "currency")].str.strip().str.upper(),
        "month": pd.to_datetime(t[c.get("month", "month")].str.strip().str[:7], format="%Y-%m",
                                errors="coerce"),
        "rate": parse_number_series(t[c.get("rate", "units_per_usd")]),
    }).dropna()
    return out


def _table_rate(fx: pd.DataFrame, cur: pd.Series, when: pd.Series) -> pd.Series:
    """Rate for (currency, month); nearest available month for that currency."""
    res = pd.Series(np.nan, index=cur.index)
    if fx is None or fx.empty:
        return res
    month = when.dt.to_period("M").dt.to_timestamp()
    for c, grp in fx.groupby("currency"):
        idx = cur.index[cur == c]
        if len(idx) == 0:
            continue
        g = grp.sort_values("month")
        months = g["month"].values
        rates = g["rate"].values
        m = month.loc[idx].values
        ok = ~pd.isna(m)
        if ok.any():
            # nearest month by absolute distance
            mi = m[ok].astype("datetime64[ns]").astype("int64")[:, None]
            gm = months.astype("datetime64[ns]").astype("int64")[None, :]
            j = np.abs(mi - gm).argmin(axis=1)
            res.loc[idx[ok]] = rates[j]
        if (~ok).any():  # no date at all: use the median rate for the currency
            res.loc[idx[~ok]] = float(np.median(rates))
    return res


def clean(df: pd.DataFrame, cfg: dict, data_dir: Path) -> pd.DataFrame:
    df = df.copy()
    df["flags"] = ""
    rc = cfg["reporting_currency"].upper()
    srcs = {s["name"]: s for s in cfg["sources"]}
    dcfg = cfg.get("dates", {})
    dayfirst = dcfg.get("dayfirst", True)

    # ---- numbers
    df["amount_num"] = parse_number_series(_col(df, "amount"))
    _add_flag(df, df["amount_num"].isna(), "UNPARSED_AMOUNT")
    df["quantity_num"] = parse_number_series(_col(df, "quantity"))
    df["fx_rate_given"] = parse_number_series(_col(df, "fx_rate"))
    df["tax_amount_num"] = parse_number_series(_col(df, "tax_amount"))
    tr = parse_number_series(_col(df, "tax_rate"))
    df["tax_rate_num"] = np.where(tr > 1.0, tr / 100.0, tr)

    # ---- dates (each source may use its own format; parser handles mixed layouts)
    for f in ("date", "posting_date"):
        df[f"{f}_parsed"] = pd.NaT
        if f in df.columns:
            for name, s in srcs.items():
                m = df["source"] == name
                if m.any():
                    df.loc[m, f"{f}_parsed"] = parse_date_series(df.loc[m, f], s.get("dayfirst", dayfirst))
            bad = (df[f].str.strip() != "") & df[f"{f}_parsed"].isna()
            _add_flag(df, bad, f"UNPARSED_{f.upper()}")
    lo, hi = dcfg.get("valid_window_years", [2000, 2035])
    for f in ("date_parsed", "posting_date_parsed"):
        yr = df[f].dt.year
        bad = yr.notna() & ((yr < lo) | (yr > hi))
        _add_flag(df, bad, f"IMPOSSIBLE_{f.split('_parsed')[0].upper()}")
        df.loc[bad, f] = pd.NaT
    # invoice date implausibly far from posting date -> flag, do not trust for FX month
    if df["posting_date_parsed"].notna().any():
        gap = (df["date_parsed"] - df["posting_date_parsed"]).dt.days
        susp = gap.notna() & ((gap > dcfg.get("max_days_after_posting", 120)) |
                              (gap < -dcfg.get("max_days_before_posting", 1100)))
        _add_flag(df, susp, "SUSPECT_DATE")
        df["date_trusted"] = df["date_parsed"].where(~susp)
    else:
        df["date_trusted"] = df["date_parsed"]
    df["fx_date"] = df["date_trusted"].fillna(df["posting_date_parsed"])
    _add_flag(df, df["fx_date"].isna(), "NO_DATE")

    # ---- tax removal
    df["net_amount"] = df["amount_num"]
    for name, s in srcs.items():
        m = df["source"] == name
        if not m.any() or s.get("amount_type", "net") != "gross":
            continue
        default = s.get("tax_rate_default", 0.0)
        rate = df.loc[m, "tax_rate_num"]
        miss = m & df["tax_rate_num"].isna()
        _add_flag(df, miss, "TAX_RATE_DEFAULTED")
        rate = rate.fillna(default)
        ta = df.loc[m, "tax_amount_num"]
        net = np.where(ta.notna(), df.loc[m, "amount_num"] - ta, df.loc[m, "amount_num"] / (1.0 + rate))
        df.loc[m, "net_amount"] = np.round(net, 2)
        df.loc[m, "tax_rate_used"] = rate

    # ---- currency
    cur = _col(df, "currency").str.strip().str.upper()
    for name, s in srcs.items():
        m = (df["source"] == name) & (cur == "")
        if m.any():
            cur.loc[m] = (s.get("currency_default") or rc).upper()
    df["currency_norm"] = cur
    fxcfg = cfg.get("fx") or {}
    fx = load_fx_table(cfg, data_dir)
    is_rc = cur == rc
    table_rate = _table_rate(fx, cur, df["fx_date"])
    given = df["fx_rate_given"].where(df["fx_rate_given"] > 0)
    if fxcfg.get("ignore_rate_for_reporting_currency", True):
        given = given.where(~is_rc)
    lo_b, hi_b = fxcfg.get("sanity_band", [0.33, 3.0])
    ratio = given / table_rate
    insane = given.notna() & table_rate.notna() & ((ratio < lo_b) | (ratio > hi_b))
    _add_flag(df, insane, "FX_RATE_REJECTED")
    rate = given.where(~insane)
    df["fx_source"] = np.where(is_rc, "reporting", np.where(rate.notna(), "line", "table"))
    rate = rate.fillna(table_rate)
    rate = rate.where(~is_rc, 1.0)
    missing = rate.isna()
    _add_flag(df, missing, "NO_FX_RATE")
    if missing.any():
        log.warning("%d lines have no FX rate (currencies %s); spend set to 0 and flagged",
                    int(missing.sum()), sorted(cur[missing].unique())[:10])
    sem = fxcfg.get("rate_semantics", "units_per_reporting")
    df["fx_rate_used"] = rate
    if sem == "units_per_reporting":
        df["spend_rc"] = df["net_amount"] / rate
    else:  # reporting units per line-currency unit
        df["spend_rc"] = df["net_amount"] * rate
    df.loc[missing | df["net_amount"].isna(), "spend_rc"] = 0.0
    df["spend_rc"] = df["spend_rc"].round(2)

    # ---- text helpers
    for f in ("description", "supplier_name", "status", "item_code", "supplier_id"):
        if f not in df.columns:
            df[f] = ""
    _add_flag(df, df["description"].str.strip() == "", "BLANK_DESCRIPTION")
    return df
