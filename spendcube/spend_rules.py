"""Decide what counts as spend. Every exclusion gets a code and a rule name.

Rules (all from config):
  status            per-source status map to CANCELLED or NOT_APPROVED
  non_procurement   regex / value-list rules on any canonical or raw column
  duplicates        exact duplicates inside one source (same values on a key)
Cross-source duplicates (same purchase in two systems) are marked in match.py.
The final code per line follows `exclusion_precedence`.
"""
from __future__ import annotations

import logging
import re
import warnings

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
            stats[f"unmapped_status_{s['name']}"] = unknown.value_counts().head(20).to_dict()
            outcome = smap.get("unknown_status_outcome", "NOT_APPROVED")
            log.warning("source %s: unknown status values %s treated as %s",
                        s["name"], sorted(unknown.unique())[:10], outcome)
            if outcome in ("CANCELLED", "NOT_APPROVED"):
                _mark(df, m & df["status"].str.strip().str.lower().isin(set(unknown)), outcome,
                      "status:unknown")


def apply_non_procurement(df: pd.DataFrame, cfg: dict, stats: dict):
    """Config rules on any field or raw column. `non_procurement` rules give NON_PROCUREMENT;
    `exclusion_rules` carry their own code, for approval or cancellation kept in flag columns."""
    rules = [(r, "NON_PROCUREMENT", "non_procurement") for r in cfg.get("non_procurement") or []] + \
            [(r, r["code"], "exclusion_rules") for r in cfg.get("exclusion_rules") or []]
    for r, code, kind in rules:
        name = r.get("name", r["field"])
        if kind == "exclusion_rules" and code in ("CANCELLED", "NOT_APPROVED") and not name.startswith("status"):
            name = f"status:{name}"      # counted with the status rules in the exclusion warning
        scope = df["source"].isin(r["sources"]) if r.get("sources") else pd.Series(True, index=df.index)
        try:
            s = field_series(df, r["field"])
        except KeyError as e:
            log.warning("%s rule %s skipped: %s", kind, name, e)
            continue
        if "pattern" in r:
            with warnings.catch_warnings():  # groups in user patterns are fine here
                warnings.simplefilter("ignore", UserWarning)
                hit = s.str.contains(r["pattern"], flags=0 if r.get("case_sensitive") else re.I,
                                     regex=True, na=False)
        else:
            vals = {str(v).strip().lower() for v in r["values"]}
            hit = s.str.strip().str.lower().isin(vals)
        if r.get("unless"):
            u = r["unless"]
            us = field_series(df, u["field"])
            hit &= ~us.str.contains(u["pattern"], flags=re.I, regex=True, na=False)
        stats[f"{kind}:{name}"] = _mark(df, scope & hit, code, name)


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
        if s.get("fuzzy_duplicates", dcfg.get("fuzzy", False)):
            near = near_duplicates(df[m & ~dup.reindex(df.index, fill_value=False)], s, dcfg)
            stats[f"near_duplicate_{s['name']}"] = _mark(df, pd.Series(df.index.isin(near), index=df.index) & m,
                                                         "DUPLICATE", "near_duplicate")


def near_duplicates(sub: pd.DataFrame, src: dict, dcfg: dict) -> list:
    """Same invoice exported twice with different formatting: same normalised invoice
    number (+ line number if the file has one), same net amount, same quantity and
    near-identical text/supplier. Returns the index of every copy except the first."""
    from rapidfuzz import fuzz

    from .match import norm_invoice_number
    from .textnorm import compact_key, norm_key
    if "invoice_number" not in sub.columns or sub.empty:
        return []
    prefixes = dcfg.get("invoice_prefixes", ["TAX INV", "INV/", "INV", "SI-", "SI", "IN", "#"])
    t = pd.DataFrame({
        "inv": sub["invoice_number"].map(lambda x: norm_invoice_number(x, prefixes)),
        "line": sub["invoice_line"].str.lstrip("0") if "invoice_line" in sub.columns else "",
        "amt": sub["net_amount"].round(2),
        "qty": sub["quantity_num"].round(4) if "quantity_num" in sub.columns else 0.0,
        "txt": sub["description"].map(norm_key),
        "sup": sub["supplier_name"].map(compact_key) if "supplier_name" in sub.columns else "",
        "ord": sub["row_order"],
        # copies whose status counts sort first, so the kept copy is one that counts
        "excl": sub["cand_codes"].map(lambda c: any(k in ("CANCELLED", "NOT_APPROVED") for k, _ in c))
        if "cand_codes" in sub.columns else False,
    }, index=sub.index)
    t = t[(t["inv"] != "") & t["amt"].notna()]
    thr = float(dcfg.get("fuzzy_text_min", 90))
    out = []
    for _, g in t.groupby(["inv", "line", "amt", "qty"], sort=False, dropna=False):
        if len(g) < 2:
            continue
        g = g.sort_values(["excl", "ord"])
        reps = []
        for i, r in g.iterrows():
            same = any(fuzz.ratio(r["txt"], rt) >= thr and
                       (not r["sup"] or not rs or fuzz.ratio(r["sup"], rs) >= 85) for rt, rs in reps)
            if same:
                out.append(i)
            else:
                reps.append((r["txt"], r["sup"]))
    return out


def rule_confidence(rule: str) -> float:
    """How sure we are that an exclusion is right, by the kind of evidence."""
    if not rule:
        return 0.0
    if rule.startswith("status") or rule == "exact_duplicate":
        return 0.99
    if "P1" in rule:
        return 0.97
    if rule in ("exported_twice", "near_duplicate"):
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


def status_exclusion_warning(df: pd.DataFrame, limit: float = 0.30) -> str | None:
    """A loud warning when status rules exclude more than `limit` of lines or of spend: an
    unrecognised status vocabulary can silently exclude a whole company."""
    if not len(df) or "exclusion_reason" not in df:
        return None
    st = df["exclusion_reason"].isin(["CANCELLED", "NOT_APPROVED"]) & \
        df["exclusion_rule"].astype(str).str.startswith("status")
    tot = float(df["spend_rc"].abs().sum()) if "spend_rc" in df else 0.0
    share_n = float(st.mean())
    share_s = float(df.loc[st, "spend_rc"].abs().sum()) / tot if tot > 0 else 0.0
    if max(share_n, share_s) <= limit:
        return None
    vals = df.loc[st, "status"].astype(str).value_counts().head(8).to_dict() if "status" in df else {}
    if not any(k.strip() for k in vals):          # the exclusions come from flag rules, not a status column
        vals = df.loc[st, "exclusion_rule"].astype(str).value_counts().head(8).to_dict()
    return (f"status rules exclude {share_n:.0%} of lines and {share_s:.0%} of spend. Check status_map "
            f"before using this output. Excluded status values: {vals}")
