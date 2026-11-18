"""Profile a company's raw files to review a drafted config. It reads and prints; it changes nothing.

    python scripts/profile_data.py --data data/C --config configs/company_c.yaml

For each source file it prints: the id column's uniqueness, every column with its blank share
and number of distinct values, the values of columns with few distinct values (status, document
type, account), non-procurement keywords found in any text column, the largest suppliers and how
concentrated the spend is. Then the tree's size and shape. The output holds values from the data,
so read it on screen and do not commit it.
"""
from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spendcube.config import load_config  # noqa: E402
from spendcube.ingest import read_table  # noqa: E402
from spendcube.paths import resolve_file  # noqa: E402
from spendcube.taxonomy import load_taxonomy  # noqa: E402
from spendcube.textnorm import clean_text, parse_number_series  # noqa: E402

# Words that often mark a line that is not a purchase from a third-party supplier.
KEYWORDS = {
    "tax": r"\b(tax(es)?|vat|gst|hst|pst|tva|iva|mwst|withholding|customs|excise|duty|duties|levy|"
           r"revenue (service|authority|agency)|irs|hmrc|sars|treasury)\b",
    "intercompany": r"\b(inter[\s-]?company|interco|intra[\s-]?group|i/c|affiliate|subsidiar(y|ies)|"
                    r"related party|recharge|cross[\s-]?charge|group (company|charge))\b",
    "payroll and staff": r"\b(payroll|salar(y|ies)|wages?|bonus|pension|employee|staff (claim|reimb)|"
                         r"reimbursements?|expense claim|per diem|petty cash)\b",
    "finance": r"\b(interest|bank (charge|fee)s?|loan|dividend|royalt(y|ies)|fx (gain|loss)|"
               r"depreciation|amorti[sz]ation|accrual|write[\s-]?off|provision)\b",
    "donation and fines": r"\b(donations?|charit(y|able)|sponsorship|grant|fines?|penalt(y|ies))\b",
    "refund and customer": r"\b(refunds?|rebates?|customer (refund|credit)|chargeback)\b",
}


def _short(v, n=38) -> str:
    v = clean_text(v)
    return v if len(v) <= n else v[: n - 1] + "…"


def profile_source(src: dict, cfg: dict, data_dir: Path) -> None:
    path = resolve_file(src["file"], cfg, data_dir, dirs=[data_dir])
    df = read_table(path, sheet=src.get("sheet"), delimiter=src.get("delimiter"), encoding=src.get("encoding"),
                    header_row=src.get("header_row"))
    df.columns = [clean_text(c) for c in df.columns]
    cols = src.get("columns") or {}
    mapped = {}
    for f, c in cols.items():
        for one in (c if isinstance(c, list) else [c]):
            mapped[one] = f
    print(f"\n===== source {src['name']}: {path.name}, {len(df)} rows, {len(df.columns)} columns")

    rid = src["record_id"]["column"] if isinstance(src["record_id"], dict) else src["record_id"]
    mapped.setdefault(rid, "record_id")
    if rid in df:
        ids = df[rid].astype(str).str.strip()
        print(f"record id `{rid}`: {ids.nunique()} distinct, {int(ids.duplicated().sum())} repeated, "
              f"{int((ids == '').sum())} blank")
    else:
        print(f"record id `{rid}` IS NOT A COLUMN of this file")

    amt_col = cols.get("amount") if isinstance(cols.get("amount"), str) else None
    amt = parse_number_series(df[amt_col].astype(str)).fillna(0.0) if amt_col in df else pd.Series(0.0, index=df.index)
    tot = float(amt.abs().sum()) or 1.0

    print("\ncolumns (role, blank share, distinct values, numeric share, samples):")
    for c in df.columns:
        v = df[c].astype(str).map(clean_text)
        nb = v[v != ""]
        num = float(parse_number_series(nb).notna().mean()) if len(nb) else 0.0
        samples = [_short(x, 24) for x in list(dict.fromkeys(nb.head(300)))[:3]]
        print(f"  {_short(c, 30):30s} {mapped.get(c, '-'):15s} blank {1 - len(nb) / max(len(v), 1):5.1%}  "
              f"distinct {nb.nunique():6d}  numeric {num:4.0%}  {samples}")

    print("\ncolumns with 2 to 40 distinct values (value: rows, share of amount):")
    for c in df.columns:
        v = df[c].astype(str).map(clean_text)
        if not 2 <= v[v != ""].nunique() <= 40:
            continue
        g = pd.DataFrame({"v": v, "a": amt.abs()}).groupby("v").agg(n=("a", "size"), a=("a", "sum"))
        g = g.sort_values("n", ascending=False).head(14)
        items = ", ".join(f"{_short(k, 26) or '(blank)'}: {int(r.n)}, {r.a / tot:.0%}" for k, r in g.iterrows())
        print(f"  {_short(c, 28)} [{mapped.get(c, '-')}]: {items}")

    print("\nnon-procurement keywords by column (rows, share of amount, most common values):")
    found = False
    id_roles = {"record_id", "invoice_number", "po_number", "item_code", "supplier_id"}
    for c in df.columns:
        v = df[c].astype(str).map(clean_text)
        if mapped.get(c) in id_roles or v.str.contains(r"[A-Za-z]{3,}").mean() < 0.3:
            continue
        for name, rx in KEYWORDS.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)      # the patterns use groups on purpose
                hit = v.str.contains(rx, flags=re.I, regex=True, na=False)
            if hit.any():
                found = True
                top = [f"{_short(k, 34)} ({n})" for k, n in v[hit].value_counts().head(4).items()]
                print(f"  {name:20s} in {_short(c, 24):24s} {int(hit.sum()):6d} rows, {amt[hit].abs().sum() / tot:5.1%}  {top}")
    if not found:
        print("  none")

    sup_col = cols.get("supplier_name") or cols.get("supplier_id")
    sup_col = sup_col[0] if isinstance(sup_col, list) else sup_col
    if sup_col in df:
        g = pd.DataFrame({"s": df[sup_col].astype(str).map(clean_text), "a": amt.abs()}).groupby("s").agg(
            n=("a", "size"), a=("a", "sum")).sort_values("a", ascending=False)
        print(f"\nlargest suppliers by amount in `{sup_col}` ({len(g)} distinct):")
        for k, r in g.head(25).iterrows():
            print(f"  {r.a / tot:6.1%}  {int(r.n):6d} rows  {_short(k, 60) or '(blank)'}")
    if amt_col in df:
        srt = amt.abs().sort_values(ascending=False)
        k1 = max(1, len(srt) // 100)
        print(f"\namount `{amt_col}`: {int((amt < 0).sum())} negative, {int((amt == 0).sum())} zero or unparsed; "
              f"the top 1% of rows ({k1}) hold {srt.head(k1).sum() / tot:.0%}, the top 30 rows {srt.head(30).sum() / tot:.0%}")


def profile_tree(cfg: dict, data_dir: Path) -> None:
    tax = load_taxonomy(cfg, data_dir)
    depth = pd.Series([tax.nodes[n].depth for n in tax.leaves]).value_counts().sort_index().to_dict()
    general = [n for n in tax.leaves if tax.nodes[n].general]
    tops = [n.name for n in tax.nodes.values() if n.depth == 1]
    shortlist = cfg.get("classification", {}).get("llm", {}).get("shortlist_above_leaves", 400)
    print(f"\n===== tree: {len(tax.nodes)} nodes, {len(tax.leaves)} leaves, depth {tax.max_depth}; leaves by depth {depth}")
    print(f"mode: {'full tree in the prompt' if len(tax.leaves) <= shortlist else 'SHORTLIST per batch'} "
          f"(shortlist_above_leaves {shortlist}); general leaves {len(general)}; "
          f"catch-all leaf: {' > '.join(tax.nodes[tax.catch_all_leaf()].path) if tax.catch_all_leaf() else 'none'}")
    print(f"top level ({len(tops)}): {[_short(t, 30) for t in tops[:40]]}")
    codey = [t for t in tops if re.fullmatch(r"[A-Za-z]{0,4}[-_.#]?\d[\dA-Za-z.\-]*", t.strip())]
    if codey:
        print(f"WARNING: top-level names look like codes {codey[:5]}; check taxonomy.level_columns")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    cfg, data_dir = load_config(a.config), Path(a.data)
    for src in cfg["sources"]:
        try:
            profile_source(src, cfg, data_dir)
        except Exception as e:                      # keep going: this is a review aid
            print(f"\n===== source {src.get('name')}: could not profile ({type(e).__name__}: {e})")
    try:
        profile_tree(cfg, data_dir)
    except Exception as e:
        print(f"\n===== tree: could not load ({type(e).__name__}: {e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
