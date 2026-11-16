"""Decide what counts as spend. Every exclusion gets a code and a rule name.

Rules (all from config):
  status            per-source status map -> CANCELLED / NOT_APPROVED
  non_procurement   regex / value-list rules on any canonical or raw column
  duplicates        exact duplicates inside one source (same values on a key)
Cross-source duplicates (same purchase in two systems) are marked in match.py.
The final code per line follows `exclusion_precedence`.
"""
from __future__ import annotations

import logging
import re

import pandas as pd

log = logging.getLogger(__name__)


def field_series(df: pd.DataFrame, field: str) -> pd.Series:
    """A rule can name a canonical field or a raw column of the source file."""
    if field in df.columns:
        return df[field].astype(str)
    if f"raw__{field}" in df.columns:
        return df[f"raw__{field}"].fillna("").astype(str)
    raise KeyError(f"Rule references unknown field/column `{field}`")


def _mark(df, mask, code, rule):
    mask = mask.fillna(False).astype(bool)
    if mask.any():
        df.loc[mask, "cand_codes"] = df.loc[mask, "cand_codes"].map(
            lambda c: c + [(code, rule)])
    return int(mask.sum())


def apply_status_rules(df: pd.DataFrame, cfg: dict, stats: dict):
    base = cfg.get("status", {})
    for s in cfg["sources"]:
        m = df["source"] == s["name"]
        if not m.any() or "status" not in (s.get("columns") or {}):
            continue
        smap = {**base, **(s.get("status_map") or {})}
        st = df.loc[m, "status"].str.strip().str.lower()
        known = set()
        for code in ("CANCELLED", "NOT_APPROVED"):
            vals = {v.lower() for v in smap.get(code, [])}
            known |= vals
            n = _mark(df, m & df["status"].str.strip().str.lower().isin(vals), code, f"status:{code.lower()}")
            stats[f"status_{code}_{s['name']}"] = n
        inc = {v.lower() for v in smap.get("include", [])}
        known |= inc
        unknown = st[(~st.isin(known)) & (st != "")]
        if len(unknown):
            outcome = smap.get("unknown_status_outcome", "include")
            log.warning("source %s: unknown status values %s -> %s",
                        s["name"], sorted(unknown.unique())[:10], outcome)
            if outcome in ("CANCELLED", "NOT_APPROVED"):
                _mark(df, m & df["status"].str.strip().str.lower().isin(set(unknown)), outcome,
                      "status:unknown")


def apply_non_procurement(df: pd.DataFrame, cfg: dict, stats: dict):
    for r in cfg.get("non_procurement") or []:
        name = r.get("name", r["field"])
        scope = df["source"].isin(r["sources"]) if r.get("sources") else pd.Series(True, index=df.index)
        try:
            s = field_series(df, r["field"])
        except KeyError as e:
            log.warning("non_procurement rule %s skipped: %s", name, e)
            continue
        if "pattern" in r:
            hit = s.str.contains(r["pattern"], flags=0 if r.get("case_sensitive") else re.I,
                                 regex=True, na=False)
        else:
            vals = {str(v).strip().lower() for v in r["values"]}
            hit = s.str.strip().str.lower().isin(vals)
        if r.get("unless"):
            u = r["unless"]
            us = field_series(df, u["field"])
            hit &= ~us.str.contains(u["pattern"], flags=re.I, regex=True, na=False)
        stats[f"non_procurement:{name}"] = _mark(df, scope & hit, "NON_PROCUREMENT", name)


def apply_within_source_duplicates(df: pd.DataFrame, cfg: dict, stats: dict):
    dcfg = cfg.get("duplicates", {})
    if not dcfg.get("within_source", True):
        return
    for s in cfg["sources"]:
        m = df["source"] == s["name"]
        if not m.any():
            continue
        key = s.get("duplicate_key") or dcfg.get("key")
        sub = df.loc[m]
        if key:
            cols = [c if c in sub.columns else f"raw__{c}" for c in key]
        else:  # every original column except the record id column
            rid = s["record_id"]["column"] if isinstance(s["record_id"], dict) else s["record_id"]
            rid = [rid] if isinstance(rid, str) else rid
            cols = [c for c in sub.columns if c.startswith("raw__") and c[5:] not in rid]
        dup = sub.duplicated(cols, keep="first")
        stats[f"exact_duplicate_{s['name']}"] = _mark(df, m & dup.reindex(df.index, fill_value=False),
                                                      "DUPLICATE", "exact_duplicate")


def rule_confidence(rule: str) -> float:
    """How sure we are that an exclusion is right, by the kind of evidence."""
    if not rule:
        return 0.0
    if rule.startswith("status") or rule == "exact_duplicate":
        return 0.99
    if "P1" in rule:
        return 0.97
    if rule == "exported_twice":
        return 0.9
    if "P2" in rule:
        return 0.88
    if "P3" in rule:
        return 0.8
    return 0.9  # non-procurement rules


def resolve_codes(df: pd.DataFrame, cfg: dict):
    order = {c: i for i, c in enumerate(cfg["exclusion_precedence"])}

    def pick(cands):
        if not cands:
            return ("", "")
        return min(cands, key=lambda cr: order.get(cr[0], 99))
    picked = df["cand_codes"].map(pick)
    df["exclusion_reason"] = picked.map(lambda x: x[0])
    df["exclusion_rule"] = picked.map(lambda x: x[1])
    df["exclusion_confidence"] = df["exclusion_rule"].map(rule_confidence)
    df["include"] = (df["exclusion_reason"] == "").astype(int)
    df["spend"] = df["spend_rc"].where(df["include"] == 1, 0.0).fillna(0.0)


def apply_rules(df: pd.DataFrame, cfg: dict) -> dict:
    stats: dict = {}
    if "cand_codes" not in df.columns:
        df["cand_codes"] = [[] for _ in range(len(df))]
    apply_status_rules(df, cfg, stats)
    apply_non_procurement(df, cfg, stats)
    apply_within_source_duplicates(df, cfg, stats)
    return stats
