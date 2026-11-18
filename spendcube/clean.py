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
from .textnorm import clean_text, normalize_currency, parse_dates_with_flags, parse_number_series

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


PUBLIC_FX = Path(__file__).resolve().parent / "data" / "fx_monthly_usd.csv"


def load_public_fx(reporting: str) -> pd.DataFrame | None:
    """Shipped public table (ECB monthly average reference rates, crossed through EUR,
    plus official fixed pegs), as units of currency per 1 reporting-currency unit."""
    if not PUBLIC_FX.exists():
        return None
    t = pd.read_csv(PUBLIC_FX, dtype={"currency": str, "month": str})
    pub = pd.DataFrame({"currency": t["currency"].str.upper(),
                        "month": pd.to_datetime(t["month"], format="%Y-%m"),
                        "rate": t["units_per_usd"].astype(float)})
    if reporting != "USD":
        base = pub[pub["currency"] == reporting][["month", "rate"]].rename(columns={"rate": "base"})
        if base.empty:
            return None
        pub = pub.merge(base, on="month")
        pub["rate"] = pub["rate"] / pub["base"]
        pub = pub[["currency", "month", "rate"]]
    return pub


def orient_table(fx: pd.DataFrame, pub: pd.DataFrame | None) -> tuple[pd.DataFrame, str]:
    """Check a supplied FX table against public rates; invert it if it is on the wrong base
    (reporting units per currency unit instead of currency units per reporting unit)."""
    if fx is None or pub is None:
        return fx, "unchecked"
    m = fx.merge(pub, on=["currency", "month"], suffixes=("", "_pub"))
    m = m[(m["rate"] > 0) & (m["rate_pub"] > 0)]
    if len(m) < 3:
        return fx, "unchecked"
    direct = float(np.median(m["rate"] / m["rate_pub"]))
    inverse = float(np.median((1.0 / m["rate"]) / m["rate_pub"]))
    if 0.5 <= direct <= 2.0:
        return fx, "agrees with public rates"
    if 0.5 <= inverse <= 2.0:
        log.warning("FX table is on the inverse base (reporting units per currency unit); inverted")
        fx = fx.assign(rate=1.0 / fx["rate"])
        return fx, "inverted to match public rates"
    log.warning("FX table differs from public rates (median ratio %.3f); used as given", direct)
    return fx, f"differs from public rates (median ratio {direct:.3f})"


def _table_rate(fx: pd.DataFrame, cur: pd.Series, when: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Rate for (currency, month) from the nearest available month for that currency, and
    the distance in days to that month (0 for a line without a date, which gets the median)."""
    res = pd.Series(np.nan, index=cur.index)
    gap = pd.Series(np.nan, index=cur.index)
    if fx is None or fx.empty:
        return res, gap
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
            dist = np.abs(mi - gm)
            j = dist.argmin(axis=1)
            res.loc[idx[ok]] = rates[j]
            gap.loc[idx[ok]] = dist[np.arange(len(j)), j] / 86_400e9
        if (~ok).any():  # no date at all: use the median rate for the currency
            res.loc[idx[~ok]] = float(np.median(rates))
            gap.loc[idx[~ok]] = 0.0
    return res, gap


def _table_range(fx: pd.DataFrame, cur: pd.Series, d1: pd.Series, d2: pd.Series,
                 max_gap_days: float) -> tuple[pd.Series, pd.Series]:
    """Lowest and highest table rate for the currency over the months from the earlier to
    the later of two dates (invoice and posting). Months further than max_gap_days from
    both dates are not used; NaN when the table has no month in or near the window."""
    lo = pd.Series(np.nan, index=cur.index)
    hi = pd.Series(np.nan, index=cur.index)
    if fx is None or fx.empty:
        return lo, hi
    a, b = d1.fillna(d2), d2.fillna(d1)
    start = pd.concat([a, b], axis=1).min(axis=1).dt.to_period("M").dt.to_timestamp()
    end = pd.concat([a, b], axis=1).max(axis=1).dt.to_period("M").dt.to_timestamp()
    slack = np.timedelta64(int(max_gap_days), "D")
    for c, grp in fx.groupby("currency"):
        idx = cur.index[(cur == c) & start.notna()]
        if len(idx) == 0:
            continue
        g = grp.sort_values("month")
        months = g["month"].values.astype("datetime64[ns]")
        rates = g["rate"].values
        keys = pd.DataFrame({"s": start.loc[idx].values, "e": end.loc[idx].values}, index=idx)
        for (s0, e0), sub in keys.groupby(["s", "e"]):
            sel = (months >= np.datetime64(s0) - slack) & (months <= np.datetime64(e0) + slack)
            if sel.any():
                lo.loc[sub.index] = rates[sel].min()
                hi.loc[sub.index] = rates[sel].max()
    return lo, hi


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
        df[f"{f}_parsed"] = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
        if f in df.columns:
            oor = pd.Series(False, index=df.index)
            for name, s in srcs.items():
                m = df["source"] == name
                if m.any():
                    d, o = parse_dates_with_flags(df.loc[m, f], s.get("dayfirst", dayfirst))
                    df.loc[m, f"{f}_parsed"] = d
                    oor.loc[m] = o
            _add_flag(df, oor, f"IMPOSSIBLE_{f.upper()}")
            bad = (df[f].str.strip() != "") & df[f"{f}_parsed"].isna() & ~oor
            _add_flag(df, bad, f"UNPARSED_{f.upper()}")
    lo, hi = dcfg.get("valid_window_years", [2000, 2035])
    for f in ("date_parsed", "posting_date_parsed"):
        yr = df[f].dt.year
        bad = yr.notna() & ((yr < lo) | (yr > hi))
        _add_flag(df, bad, f"IMPOSSIBLE_{f.split('_parsed')[0].upper()}")
        df.loc[bad, f] = pd.NaT
    # an invoice date implausibly far from the posting date is flagged and not used for the FX month
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
    cur = normalize_currency(_col(df, "currency"), rc, (cfg.get("fx") or {}).get("currency_aliases"))
    for name, s in srcs.items():
        m = (df["source"] == name) & (cur == "")
        if m.any():
            cur.loc[m] = (s.get("currency_default") or rc).upper()
    # a rate other than 1 with no currency: the line is treated as reporting currency, but flagged
    blank_cur = _col(df, "currency").str.strip() == ""
    _add_flag(df, blank_cur & df["fx_rate_given"].notna() & ((df["fx_rate_given"] - 1.0).abs() > 1e-9),
              "FX_RATE_WITHOUT_CURRENCY")
    df["currency_norm"] = cur
    fxcfg = cfg.get("fx") or {}
    sem = fxcfg.get("rate_semantics", "units_per_reporting")
    is_rc = cur == rc
    pub = load_public_fx(rc)
    fx, fx_check = orient_table(load_fx_table(cfg, data_dir), pub)
    df.attrs["fx_table_check"] = fx_check
    # A table month far from the line's date is stale (the public table stops early for some
    # currencies, such as ARS in 2020). A stale rate is not used to judge a line rate and is
    # used to fill only after the file's own rates.
    stale_days = float(fxcfg.get("stale_table_days", 95))
    ref_cfg_any, gap_cfg = _table_rate(fx, cur, df["fx_date"])     # client's table (if any)
    ref_pub_any, gap_pub = _table_rate(pub, cur, df["fx_date"])    # shipped public table
    ref_cfg = ref_cfg_any.where(gap_cfg <= stale_days)
    ref_pub = ref_pub_any.where(gap_pub <= stale_days)
    ref = ref_cfg.fillna(ref_pub)
    # the line may have been booked at any month from invoice to posting (high inflation)
    lo_c, hi_c = _table_range(fx, cur, df["date_trusted"], df["posting_date_parsed"], stale_days)
    lo_p, hi_p = _table_range(pub, cur, df["date_trusted"], df["posting_date_parsed"], stale_days)
    has_c = lo_c.notna()
    win_lo, win_hi = lo_c.where(has_c, lo_p), hi_c.where(has_c, hi_p)
    given = df["fx_rate_given"].where(df["fx_rate_given"] > 0)
    if sem != "units_per_reporting":                  # normalise to currency units per reporting unit
        given = 1.0 / given
    if fxcfg.get("ignore_rate_for_reporting_currency", True):
        given = given.where(~is_rc)
    lo_b, hi_b = fxcfg.get("sanity_band", [0.33, 3.0])
    win_lo, win_hi = win_lo.fillna(ref), win_hi.fillna(ref)
    has_ref = win_lo.notna()

    def _dist(x):
        """log distance from a rate to the reference range (0 inside it)."""
        return np.maximum(np.log(win_lo / x), 0.0) + np.maximum(np.log(x / win_hi), 0.0)

    # two hypotheses per line rate: as given, or on the inverse base. The inverse wins
    # when it is much closer to the reference range and close in absolute terms (within 25%),
    # which also catches near-parity currencies (EUR 1.09 given where 0.92 is expected).
    d_dir = _dist(given)
    d_inv = _dist(1.0 / given)
    inverted = given.notna() & has_ref & (d_inv < np.log(1.25)) & (d_inv + 0.05 < d_dir)
    in_band = (given / win_hi <= hi_b) & (given / win_lo >= lo_b)
    ok = given.notna() & ~inverted & (~has_ref | in_band)
    rejected = given.notna() & ~ok & ~inverted
    _add_flag(df, inverted, "FX_RATE_INVERTED")
    _add_flag(df, rejected, "FX_RATE_REJECTED")
    rate = given.where(ok)
    rate[inverted] = 1.0 / given[inverted]
    src = pd.Series(np.where(ok, "line", np.where(inverted, "line_inverted", "")), index=df.index, dtype=object)
    fill = rate.isna() & ref_cfg.notna() & ~is_rc
    rate[fill], src[fill] = ref_cfg[fill], "config_table"
    fill = rate.isna() & ref_pub.notna() & ~is_rc
    rate[fill], src[fill] = ref_pub[fill], "public_table"
    # then the median of the accepted line rates for that currency in this file
    med = rate[src.isin(["line", "line_inverted"])].groupby(cur).median()
    fill = rate.isna() & cur.isin(med.index) & ~is_rc
    rate[fill], src[fill] = cur[fill].map(med), "file_median"
    _add_flag(df, src == "file_median", "FX_FILE_MEDIAN")
    # last: a stale table month, the client's first
    fill = rate.isna() & ref_cfg_any.notna() & ~is_rc
    rate[fill], src[fill] = ref_cfg_any[fill], "config_table_stale"
    fill = rate.isna() & ref_pub_any.notna() & ~is_rc
    rate[fill], src[fill] = ref_pub_any[fill], "public_table_stale"
    _add_flag(df, src.isin(["public_table", "public_table_stale"]), "FX_PUBLIC_TABLE")
    _add_flag(df, src.isin(["config_table_stale", "public_table_stale"]), "FX_STALE_TABLE")
    rate = rate.where(~is_rc, 1.0)
    src[is_rc] = "reporting"
    missing = rate.isna()
    src[missing] = "none"
    df["fx_source"] = src
    _add_flag(df, missing, "NO_FX_RATE")
    if missing.any():
        log.warning("%d lines have no FX rate in any source (currencies %s); spend set to 0 and flagged",
                    int(missing.sum()), sorted(cur[missing].unique())[:10])
    df["fx_rate_used"] = rate
    df["spend_rc"] = df["net_amount"] / rate
    df.loc[missing | df["net_amount"].isna(), "spend_rc"] = 0.0
    df["spend_rc"] = df["spend_rc"].round(2)

    # ---- text helpers
    for f in ("description", "supplier_name", "status", "item_code", "supplier_id"):
        if f not in df.columns:
            df[f] = ""
    _add_flag(df, df["description"].str.strip() == "", "BLANK_DESCRIPTION")
    return df
