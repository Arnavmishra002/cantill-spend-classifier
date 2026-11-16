"""Classification units: identical (supplier, text, account, code) lines are classified once.

Collapsing lines into units cuts LLM calls by an order of magnitude and makes
results consistent: the same description from the same supplier always gets the
same category.
"""
from __future__ import annotations

import hashlib

import pandas as pd

from ..textnorm import clean_text, norm_key

UNIT_FIELDS = ("description", "supplier_name", "gl_description", "context", "item_code",
               "uom", "document_type")


def _get(df, f):
    return df[f].fillna("").astype(str) if f in df.columns else pd.Series("", index=df.index)


def build_units(df: pd.DataFrame, cfg: dict) -> tuple[pd.Series, pd.DataFrame]:
    """Return (unit_id per row, units frame).

    units frame columns: unit_id, description, supplier_name, supplier_key,
    gl_description, context, item_code, n_lines, spend, sources, text (display text),
    desc_key (normalised description).
    """
    ccfg = cfg.get("classification", {})
    key_parts = pd.DataFrame({
        "desc": _get(df, "description").map(norm_key),
        "sup": _get(df, "supplier_key"),
        "gl": _get(df, "gl_description").map(norm_key),
        "ctx": _get(df, "context").map(norm_key),
        "code": _get(df, "item_code").str.strip().str.upper() if ccfg.get("unit_uses_item_code", True)
        else "",
    })
    raw = key_parts.astype(str).agg("\x1f".join, axis=1)
    uid = raw.map(lambda s: "U" + hashlib.sha1(s.encode()).hexdigest()[:12])
    tmp = pd.DataFrame({
        "unit_id": uid,
        "description": _get(df, "description"),
        "supplier_name": _get(df, "supplier_name"),
        "supplier_key": _get(df, "supplier_key"),
        "supplier_id": _get(df, "supplier_id"),
        "gl_description": _get(df, "gl_description"),
        "context": _get(df, "context"),
        "item_code": _get(df, "item_code"),
        "desc_key": key_parts["desc"],
        "spend_abs": df["spend_rc"].abs().fillna(0.0) if "spend_rc" in df else 0.0,
        "source": df["source"],
    })
    agg = tmp.groupby("unit_id", sort=False).agg(
        description=("description", "first"),
        supplier_name=("supplier_name", lambda s: s.mode().iloc[0] if len(s) else ""),
        supplier_key=("supplier_key", "first"),
        supplier_id=("supplier_id", "first"),
        gl_description=("gl_description", "first"),
        context=("context", "first"),
        item_code=("item_code", "first"),
        desc_key=("desc_key", "first"),
        n_lines=("description", "size"),
        spend=("spend_abs", "sum"),
        sources=("source", lambda s: ",".join(sorted(set(s)))),
    ).reset_index()
    agg["text"] = agg.apply(unit_text, axis=1)
    return uid, agg


def unit_text(r) -> str:
    """Human/LLM readable one-line description of a unit."""
    parts = []
    d = clean_text(r.get("description", ""))
    parts.append(f"desc: {d if d else '(blank)'}")
    if r.get("supplier_name"):
        parts.append(f"supplier: {r['supplier_name']}")
    if r.get("gl_description"):
        parts.append(f"account: {r['gl_description']}")
    if r.get("context"):
        parts.append(f"context: {r['context']}")
    if r.get("item_code"):
        parts.append(f"item code: {r['item_code']}")
    return " | ".join(parts)
