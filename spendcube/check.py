"""`run.py check-config`: fast dry run to review a (drafted) config before the real run.

Prints, per source and mapped field: column, blank share, parse rate (numbers/dates)
and sample values; then exclusion counts, FX/date flags and spend totals. No
classification and no API calls, so it takes seconds.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .clean import clean
from .config import load_config
from .ingest import ingest_all
from .spend_rules import apply_rules, resolve_codes, status_exclusion_warning
from .taxonomy import load_taxonomy

NUMERIC = {"amount", "quantity", "unit_price", "tax_amount", "tax_rate", "fx_rate"}
DATES = {"date": "date_parsed", "posting_date": "posting_date_parsed"}


def check_config(config: str, data: str) -> str:
    cfg = load_config(config)
    d = Path(data)
    out = []
    tax = load_taxonomy(cfg, d)
    out.append(f"tree: {len(tax.nodes)} nodes, {len(tax.leaves)} leaves, depth {tax.max_depth}")
    for n in tax.leaves[:5]:
        out.append(f"   e.g. {tax.nodes[n].label}")
    raw = ingest_all(cfg, d)
    df = clean(raw, cfg, d)
    for s in cfg["sources"]:
        m = df["source"] == s["name"]
        sub = df[m]
        out.append(f"\nsource {s['name']}: {int(m.sum())} rows")
        for f, col in (s.get("columns") or {}).items():
            v = sub[f].astype(str) if f in sub else pd.Series(dtype=str)
            blank = float((v.str.strip() == "").mean()) if len(v) else 1.0
            extra = ""
            if f in NUMERIC:
                parsed = {"amount": "amount_num", "quantity": "quantity_num", "tax_rate": "tax_rate_num",
                          "fx_rate": "fx_rate_given", "tax_amount": "tax_amount_num"}.get(f)
                if parsed and parsed in sub:
                    ok = sub[parsed].notna() | (v.str.strip() == "")
                    extra = f" parsed {ok.mean():.1%}"
            if f in DATES and DATES[f] in sub:
                nb = v.str.strip() != ""
                extra = f" parsed {sub.loc[nb, DATES[f]].notna().mean():.1%} of non-blank" if nb.any() else ""
            samples = list(dict.fromkeys(v[v.str.strip() != ""].head(200)))[:3]
            out.append(f"  {f:15s} <- {str(col)[:40]:40s} blank {blank:5.1%}{extra}  e.g. {samples}")
        if "status" in sub:
            out.append(f"  status values: {sub['status'].value_counts().head(10).to_dict()}")
        if "currency_norm" in sub:
            out.append(f"  currencies: {sub['currency_norm'].value_counts().head(10).to_dict()}  "
                       f"fx source: {sub['fx_source'].value_counts().to_dict()}")
    apply_rules(df, cfg)
    resolve_codes(df, cfg)
    flags = df["flags"].str.split(";").explode()
    sw = status_exclusion_warning(df, float((cfg.get("status") or {}).get("warn_above_share", 0.30)))
    if sw:
        out.insert(0, f"WARNING: {sw}\n")
        out.append(f"\nWARNING: {sw}")
    out.append(f"\nflags: {flags[flags != ''].value_counts().to_dict()}")
    out.append(f"exclusions: {df['exclusion_reason'].replace('', 'INCLUDED').value_counts().to_dict()}")
    out.append(f"rules fired: {df.loc[df['exclusion_rule'] != '', 'exclusion_rule'].value_counts().head(12).to_dict()}")
    out.append(f"spend ({cfg['reporting_currency']}): total {df['spend_rc'].sum():,.2f}  included "
               f"{df['spend'].sum():,.2f}  (cross-source matching runs only in the full run)")
    return "\n".join(out)
