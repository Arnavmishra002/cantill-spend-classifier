"""Draft a config for an unfamiliar company (Company C) from its files.

1. Find the tree file and the transaction file(s) in the folder.
2. Detect each column's role from its header (synonym regexes) and its values
   (numeric/date parse rates, uniqueness, ISO currency codes, text length...).
3. If an LLM key is available, ask Claude to map headers+samples to roles and use
   it where the heuristics are unsure (validated against parse rates).
4. Draft status mapping, non-procurement rules (generic tax / intercompany /
   finance patterns, plus any GL accounts that look like them), FX settings and a
   one-line company context for the classifier.
5. Write YAML with CHECK comments for a human to review. No code edits needed.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .config import DEFAULTS
from .ingest import read_table
from .textnorm import clean_text, parse_date_series, parse_number_series

log = logging.getLogger(__name__)

ISO_CCY = set("""AED AFN ALL AMD ANG AOA ARS AUD AWG AZN BAM BBD BDT BGN BHD BIF BMD BND BOB BRL BSD BTN BWP
BYN BZD CAD CDF CHF CLP CNY COP CRC CUP CVE CZK DJF DKK DOP DZD EGP ERN ETB EUR FJD FKP GBP GEL GHS GIP GMD
GNF GTQ GYD HKD HNL HRK HTG HUF IDR ILS INR IQD IRR ISK JMD JOD JPY KES KGS KHR KMF KPW KRW KWD KYD KZT LAK
LBP LKR LRD LSL LYD MAD MDL MGA MKD MMK MNT MOP MRU MUR MVR MWK MXN MYR MZN NAD NGN NIO NOK NPR NZD OMR PAB
PEN PGK PHP PKR PLN PYG QAR RON RSD RUB RWF SAR SBD SCR SDG SEK SGD SHP SLL SOS SRD SSP STN SYP SZL THB TJS
TMT TND TOP TRY TTD TWD TZS UAH UGX USD UYU UZS VES VND VUV WST XAF XCD XOF XPF YER ZAR ZMW ZWL RCN""".split())

# canonical field and its header patterns (earlier patterns score higher)
HEADER_HINTS = {
    "record_id": [r"^(line|record|row|txn|transaction|invoice[\s_]*line|entry)[\s_]?(id|key)$", r"^id$",
                  r"line[\s_]*id", r"(record|row|line)[\s_]*(no|number|#)$"],
    "invoice_number": [r"inv(oice)?[\s_]*(no|num|number|#|id|ref)", r"^document[\s_]*(no|number)$", r"bill[\s_]*no"],
    "invoice_line": [r"inv(oice)?[\s_]*line", r"^line[\s_]*(no|num|number)$", r"^item[\s_]*no$"],
    "date": [r"inv(oice)?[\s_]*date", r"doc(ument)?[\s_]*date", r"bill[\s_]*date", r"purchase[\s_]*date",
             r"order[\s_]*date", r"po[\s_]*date", r"^date$", r"txn[\s_]*date", r"transaction[\s_]*date",
             r"receipt[\s_]*date"],
    "posting_date": [r"post(ing)?[\s_]*date", r"gl[\s_]*date", r"entry[\s_]*date", r"accounting[\s_]*date",
                     r"creat(ion|ed)[\s_]*(date|on)", r"approv(al|ed)[\s_]*date"],
    "supplier_name": [r"(supplier|vendor|creditor|payee|seller)[\s_]*(name)?$", r"(supplier|vendor)[\s_]*name", r"party[\s_]*name"],
    "supplier_id": [r"(supplier|vendor|creditor|payee)[\s_]*(id|code|no|nr|num|number|#|key|ref|account|acct)$",
                    r"(supplier|vendor|creditor)[\s_]*(id|code|no|number|#)\b", r"^vendor$"],
    "supplier_tax_id": [r"gstin", r"\bvat[\s_]*(no|number|id|reg)", r"tax[\s_]*(id|no|number|reg)", r"\bpan\b", r"\babn\b", r"\bein\b", r"\bcnpj\b"],
    "description": [r"(line|item|material|product|service)?[\s_]*desc(ription)?", r"line[\s_]*text", r"short[\s_]*text", r"narration", r"memo", r"particulars", r"^text$", r"item[\s_]*name"],
    "item_code": [r"(material|item|product|part|sku|article)[\s_]*(no|num|number|code|id)?$", r"^sku$", r"part[\s_]*(no|number)"],
    "quantity": [r"^(qty|quantity)", r"quantity", r"\bqty\b"],
    "uom": [r"\buom\b", r"unit[\s_]*of[\s_]*measure", r"^unit$"],
    "unit_price": [r"unit[\s_]*(price|cost|rate)", r"^price$", r"^rate$"],
    "amount": [r"net[\s_]*(amount|value|amt)", r"(line|invoice|item)?[\s_]*amount", r"^value", r"amount", r"spend", r"total"],
    "tax_amount": [r"(vat|tax|gst)[\s_]*(amount|amt|value)"],
    "tax_rate": [r"(vat|tax|gst)[\s_]*(rate|%|pct|percent)", r"^(vat|tax|gst)$"],
    "currency": [r"^(currency|ccy|curr|cur)([\s_]*code)?$", r"currency"],
    "fx_rate": [r"(fx|exchange|currency|conversion)[\s_]*rate", r"^rate[\s_]*to"],
    "status": [r"status", r"state$", r"approval"],
    "po_number": [r"\bpo\b[\s_]*(no|num|number|#)?$", r"purchase[\s_]*order", r"^po[\s_]*(ref|reference)"],
    "gl_account": [r"gl[\s_]*(account|acct|code)?$", r"(account|acct)[\s_]*(code|no|number)$", r"^account$", r"nominal"],
    "gl_description": [r"(gl|account|acct)[\s_]*(desc|description|name)", r"expense[\s_]*type", r"spend[\s_]*category"],
    "cost_centre": [r"cost[\s_]*cent(er|re)", r"\bcc\b", r"department"],
    "document_type": [r"doc(ument)?[\s_]*type", r"transaction[\s_]*type", r"invoice[\s_]*type"],
    "entity": [r"company[\s_]*code", r"entity", r"business[\s_]*unit", r"legal"],
}
CONTEXT_HINTS = [r"category", r"commodity", r"type", r"class", r"group", r"segment", r"plant", r"site",
                 r"region", r"capex|opex", r"buyer", r"contract"]

STATUS_WORDS = {"include": ["paid", "approved", "posted", "cleared", "complete", "closed", "processed"],
                "CANCELLED": ["cancel", "void", "reversed", "deleted"],
                "NOT_APPROVED": ["reject", "hold", "submitted", "dispute", "pending", "draft", "review", "await",
                                 "parked", "blocked", "open"]}

GENERIC_NP_RULES = [
    {"name": "intercompany_text", "field": "description",
     "pattern": r"(?i)\b(intercompany|inter-company|inter company|interco|i/c (charge|recharge|payable))\b"},
    {"name": "tax_lines", "field": "description",
     "pattern": r"(?i)^\s*(accrued[\s-]*)?((state\s*(&|and)\s*local\s+tax(es)?)|(sales\s*(&\s*use\s*)?tax(\s+expense)?)|(gst|vat|igst|cgst|sgst|tds|withholding tax|wht)(\s+(input|output|payable|paid|amount|adjustment))?)\s*-?\s*[\d.%]*\s*$"},
    {"name": "finance_and_other_non_supplier", "field": "description",
     "pattern": r"(?i)\b(interest (charge|expense|payment|on loan)s?|bank (charge|fee)s?|royalt(y|ies)|dividends?|loan repayment|revenue share|partner billing|penalty interest|salary|salaries|payroll|donation)\b"},
]


def _score_header(col: str, field: str) -> float:
    # 'Supplier No.', 'Vendor-ID' and 'Vendor #' become 'supplier no', 'vendor id' and 'vendor #'
    c = re.sub(r"[.\-:]+", " ", col.strip().lower()).strip()
    c = re.sub(r"\s+", " ", c)
    for k, rx in enumerate(HEADER_HINTS.get(field, [])):
        if re.search(rx, c):
            return 1.0 - 0.1 * k
    return 0.0


def profile_columns(df: pd.DataFrame, n=3000) -> dict:
    s = df.head(n)
    prof = {}
    for c in s.columns:
        v = s[c].astype(str).map(clean_text)
        nb = v[v != ""]
        num = parse_number_series(nb) if len(nb) else pd.Series(dtype=float)
        dt = parse_date_series(nb.head(500)) if len(nb) else pd.Series(dtype="datetime64[ns]")
        prof[c] = {
            "blank": float((v == "").mean()),
            "unique_ratio": float(nb.nunique() / max(len(nb), 1)),
            "nunique": int(nb.nunique()),
            "num_rate": float(num.notna().mean()) if len(nb) else 0.0,
            "date_rate": float(dt.notna().mean()) if len(nb) else 0.0,
            "ccy_rate": float(nb.str.upper().isin(ISO_CCY).mean()) if len(nb) else 0.0,
            "avg_len": float(nb.str.len().mean()) if len(nb) else 0.0,
            "alpha_rate": float(nb.str.contains(r"[A-Za-z]{3,}", regex=True).mean()) if len(nb) else 0.0,
            "has_decimals": float(nb.str.contains(r"[.,]\d{2}$", regex=True).mean()) if len(nb) else 0.0,
            "samples": list(dict.fromkeys(nb.head(200)))[:6],
        }
    return prof


def heuristic_roles(prof: dict) -> dict:
    """Return {field: (column, score)}."""
    cands = {}
    for f in HEADER_HINTS:
        best = None
        for c, p in prof.items():
            h = _score_header(c, f)
            v = 0.0
            if f in ("amount", "tax_amount", "unit_price", "quantity", "fx_rate", "tax_rate"):
                v = p["num_rate"] if p["num_rate"] > 0.9 else -1
            elif f in ("date", "posting_date"):
                v = p["date_rate"] if p["date_rate"] > 0.6 else -1
            elif f == "currency":
                v = p["ccy_rate"] if p["ccy_rate"] > 0.8 else -1
            elif f == "record_id":
                v = 1.0 if p["unique_ratio"] > 0.999 and p["blank"] == 0 else -1
            elif f in ("invoice_line", "po_item"):
                v = 1.0 if p["unique_ratio"] < 0.5 else -1     # line numbers repeat across invoices
            elif f == "description":
                v = min(p["avg_len"] / 30, 1.0) * p["alpha_rate"]
            elif f == "supplier_name":
                v = p["alpha_rate"] * (1 if p["num_rate"] < 0.2 else 0)
                if p["avg_len"] < 5 or _code_like(p["samples"]):
                    v = -1                       # codes such as V10023 are ids, not names
            elif f == "supplier_id":
                v = 1.0 if (_code_like(p["samples"]) or p["num_rate"] > 0.8) else 0.2
            elif f == "status":
                v = 1.0 if 1 < p["nunique"] <= 15 and p["alpha_rate"] > 0.8 else -1
            if h <= 0 or v < 0:
                continue
            sc = h + 0.5 * v
            if best is None or sc > best[1]:
                best = (c, sc)
        if best:
            cands[f] = best
    # a column may serve only one role: keep the higher score (a unique id column is
    # always the record id first)
    if "record_id" in cands:
        cands["record_id"] = (cands["record_id"][0], cands["record_id"][1] + 10)
    used = {}
    for f, (c, sc) in sorted(cands.items(), key=lambda x: -x[1][1]):
        if c in used:
            continue
        used[c] = f
    roles = {f: cands[f] for f in cands if used.get(cands[f][0]) == f}
    # fallback for record id: first fully unique column
    if "record_id" not in roles:
        for c, p in prof.items():
            if p["unique_ratio"] > 0.999 and p["blank"] == 0 and c not in used:
                roles["record_id"] = (c, 0.4)
                break
    if "description" not in roles:
        txt = [(c, p["avg_len"] * p["alpha_rate"]) for c, p in prof.items() if c not in used]
        if txt:
            c = max(txt, key=lambda x: x[1])[0]
            roles["description"] = (c, 0.3)
    return roles


def llm_roles(prof: dict) -> dict | None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
    except Exception:
        return None
    fields = sorted(HEADER_HINTS) + ["context", "ignore"]
    cols = {c: {"samples": [str(x)[:40] for x in p["samples"][:5]], "blank": round(p["blank"], 2),
                "numeric": round(p["num_rate"], 2), "date": round(p["date_rate"], 2)} for c, p in prof.items()}
    schema = {"type": "object", "properties": {"mapping": {"type": "array", "items": {
        "type": "object", "properties": {"column": {"type": "string", "enum": list(prof)},
                                         "role": {"type": "string", "enum": fields}},
        "required": ["column", "role"], "additionalProperties": False}}},
        "required": ["mapping"], "additionalProperties": False}
    prompt = ("You map columns of a company's purchase/accounts-payable export to canonical roles for a "
              "spend analysis tool. Each role at most once except `context` (extra text useful to categorise "
              "spend, e.g. GL/category/commodity/plant) and `ignore`. `amount` is the line amount (prefer net "
              "of tax); `date` is the invoice/document date; `posting_date` the accounting date.\n"
              f"Roles: {fields}\nColumns:\n{json.dumps(cols, ensure_ascii=False)}")
    try:
        c = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], base_url="https://api.anthropic.com")
        r = c.messages.create(model="claude-opus-5-5", max_tokens=8000,
                              output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}},
                              messages=[{"role": "user", "content": prompt}])
        text = next(b.text for b in r.content if b.type == "text")
        return {m["column"]: m["role"] for m in json.loads(text)["mapping"]}
    except Exception as e:
        log.warning("LLM column mapping unavailable: %s", str(e)[:150])
        return None


def _merge_roles(h: dict, l: dict | None, prof: dict) -> tuple[dict, list[str], dict]:
    roles = {f: c for f, (c, _) in h.items()}
    notes = []
    context = []
    if l:
        by_role = {}
        for col, role in l.items():
            if role == "context":
                context.append(col)
            elif role != "ignore":
                by_role.setdefault(role, col)
        for f, col in by_role.items():
            p = prof[col]
            ok = True
            if f in ("amount", "quantity", "tax_amount", "fx_rate", "tax_rate", "unit_price"):
                ok = p["num_rate"] > 0.85
            if f in ("date", "posting_date"):
                ok = p["date_rate"] > 0.5
            if not ok:
                notes.append(f"LLM suggested {col} as {f} but values do not parse; kept heuristic")
                continue
            if roles.get(f) != col:
                if f in roles:
                    notes.append(f"{f}: LLM chose `{col}` over heuristic `{roles[f]}`")
                roles[f] = col
        # one column serves one role
        seen = {}
        for f in list(roles):
            c = roles[f]
            if c in seen:
                keep = seen[c] if (l.get(c) == seen[c]) else f
                drop = f if keep == seen[c] else seen[c]
                roles.pop(drop, None)
                seen[c] = keep
            else:
                seen[c] = f
    used = set(roles.values())
    if not context:
        context = [c for c in prof if c not in used and any(re.search(rx, c.lower()) for rx in CONTEXT_HINTS)
                   and prof[c]["alpha_rate"] > 0.5 and prof[c]["nunique"] > 1][:3]
    return roles, notes, {"context": context}


def _code_like(samples: list) -> bool:
    """Values like 'V10023', '100045', 'SUP-0042': short tokens with digits, no spaces."""
    vals = [str(x).strip() for x in samples if str(x).strip()]
    if not vals:
        return False
    hits = sum(1 for v in vals if re.fullmatch(r"[A-Za-z]{0,4}[-_#]?\d{2,}[A-Za-z0-9-]*", v))
    return hits >= 0.8 * len(vals)


def guess_dayfirst(values: pd.Series) -> bool | None:
    """Decide day-first vs month-first from the values themselves.
    Any second part above 12 (03/25/2024) means month first; any first part above 12
    (25/03/2024) means day first; otherwise undecided (None)."""
    v = values.astype(str).str.strip()
    m = v.str.extract(r"^(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{2,4})")
    m = m.dropna()
    if m.empty:
        return None
    a, b = m[0].astype(int), m[1].astype(int)
    if (b > 12).any() and not (a > 12).any():
        return False
    if (a > 12).any() and not (b > 12).any():
        return True
    return None


def _uniq(seq):
    return list(dict.fromkeys(x for x in seq if x))


def _status_map(values: list[str]) -> dict:
    m = {"include": [], "CANCELLED": [], "NOT_APPROVED": []}
    unknown = []
    for v in values:
        lv = v.strip().lower()
        if not lv:
            continue
        hit = None
        for code in ("CANCELLED", "NOT_APPROVED", "include"):
            if any(w in lv for w in STATUS_WORDS[code]):
                hit = code
                break
        (m[hit] if hit else unknown).append(lv)
    m = {k: _uniq(v) for k, v in m.items()}
    m["unknown_status_outcome"] = "NOT_APPROVED"
    return m, _uniq(unknown)


# rough units of currency per 1 USD, only to tell the direction of a rate column
_TYPICAL_PER_USD = {"EUR": 0.92, "GBP": 0.79, "INR": 83.0, "JPY": 150.0, "CNY": 7.2, "ZAR": 18.5, "BRL": 5.2,
                    "MXN": 17.5, "CAD": 1.36, "AUD": 1.52, "CHF": 0.89, "SEK": 10.5, "KRW": 1350.0,
                    "AED": 3.67, "SAR": 3.75, "SGD": 1.35, "PLN": 4.0, "TRY": 32.0}


def _guess_rate_semantics(df, cur_col, rate_col, reporting="USD") -> str | None:
    if reporting != "USD":
        return None
    cur = df[cur_col].astype(str).str.strip().str.upper()
    rate = parse_number_series(df[rate_col].astype(str))
    votes = {"units_per_reporting": 0, "reporting_per_unit": 0}
    for c, typ in _TYPICAL_PER_USD.items():
        r = rate[(cur == c) & (rate > 0)]
        if len(r) < 3:
            continue
        med = float(r.median())
        a, b = abs(np.log(med / typ)), abs(np.log(med * typ))
        votes["units_per_reporting" if a < b else "reporting_per_unit"] += len(r)
    if not any(votes.values()):
        return None
    return max(votes, key=votes.get)


def _company_context(tax_paths: list[str], samples: list[str]) -> str | None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
        c = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], base_url="https://api.anthropic.com")
        prompt = ("From this company's procurement category tree and a sample of its purchase lines, write a "
                  "2-3 sentence description of the company (industry, what it buys) to brief a spend "
                  "classifier. No preamble.\nTree:\n" + "\n".join(tax_paths[:300]) +
                  "\nSample lines:\n" + "\n".join(samples[:60]))
        r = c.messages.create(model="claude-opus-5-5", max_tokens=4000, output_config={"effort": "low"},
                              messages=[{"role": "user", "content": prompt}])
        return next(b.text for b in r.content if b.type == "text").strip()
    except Exception as e:
        log.warning("company context draft failed: %s", str(e)[:150])
        return None


def _taxonomy_block(path: Path) -> dict:
    t = read_table(path)
    cols = list(t.columns)
    lower = {c.lower(): c for c in cols}
    if any("parent" in k for k in lower):
        from .taxonomy import guess_parent_child_columns
        idc, pc, nc = guess_parent_child_columns(cols)
        block = {"file": path.name, "format": "parent_child", "id_column": idc, "parent_column": pc,
                 "name_column": nc}
        return block, t
    lvl = [c for c in cols if re.search(r"(?i)(level|lvl|\bl)[\s_-]?\d|l\d$|category[\s_]*l?\d", c)]
    desc = [c for c in cols if re.search(r"(?i)desc|definition|keywords|examples|scope", c)]
    block = {"file": path.name, "format": "wide", "level_columns": lvl or [c for c in cols if c not in desc]}
    if desc:
        block["description_column"] = desc[0]
    return block, t


def draft_config(data: str, taxonomy: str, out: str, company: str = "C", use_llm: bool = True,
                 files: list[str] | None = None) -> Path:
    data_dir = Path(data)
    tax_path = data_dir / taxonomy
    if not tax_path.exists():
        hits = sorted(data_dir.glob(taxonomy))
        if not hits:
            raise FileNotFoundError(f"taxonomy {taxonomy} not found in {data_dir}")
        tax_path = hits[0]
    tax_block, tax_df = _taxonomy_block(tax_path)
    cand = [data_dir / f for f in files] if files else \
        sorted(p for p in data_dir.iterdir() if p.is_file() and p.suffix.lower() in
               (".csv", ".tsv", ".txt", ".xlsx", ".xls") and p.resolve() != tax_path.resolve()
               and not p.name.startswith("."))
    sources, notes, fx_table = [], [], None
    window_years: list[tuple[int, int]] = []
    fx_sem_any = None
    consistency_groups: list[str] = []
    all_context_samples = []
    np_rules = list(GENERIC_NP_RULES)
    reporting = "USD"
    for i, p in enumerate(cand):
        df = read_table(p)
        df.columns = [clean_text(c) for c in df.columns]
        low = {c.lower() for c in df.columns}
        if {"record_id", "include", "exclusion_reason", "supplier_key"} <= low or p.stem.endswith("_audit"):
            notes.append(f"{p.name}: looks like an output of this tool; skipped")
            continue
        if {"record_id", "include", "exclusion_reason"} <= low and len(df.columns) <= 12 and len(df) <= 5:
            notes.append(f"{p.name}: looks like the output template; skipped")
            continue
        if {"currency"} <= low and any("rate" in c or "per" in c for c in low) and len(df) < 5000 and \
                any("month" in c or "date" in c or "period" in c for c in low):
            cols = {c.lower(): c for c in df.columns}
            rate_col = next(c for c in df.columns if "rate" in c.lower() or "per" in c.lower())
            month_col = next(c for c in df.columns if any(k in c.lower() for k in ("month", "date", "period")))
            fx_table = {"table": p.name, "table_columns": {"currency": cols["currency"], "month": month_col,
                                                            "rate": rate_col}}
            notes.append(f"{p.name}: looks like an FX table (CHECK rate direction: units per reporting currency)")
            continue
        if any(re.search(r"label|category_l1|seed", c.lower()) for c in df.columns) and len(df.columns) <= 8:
            notes.append(f"{p.name}: looks like a labels file; not used as input (add under classification.labels if wanted)")
            continue
        prof = profile_columns(df)
        h = heuristic_roles(prof)
        lr = llm_roles(prof) if use_llm else None
        roles, rnotes, extra = _merge_roles(h, lr, prof)
        notes += [f"{p.name}: {n}" for n in rnotes]
        # an amount already in the reporting currency beats converting a local amount
        rc_amt = [c for c in df.columns if re.search(rf"(?i)({reporting}.*(amount|amt|value|spend|total))|"
                                                     rf"((amount|amt|value|spend|total).*{reporting})", c)
                  and prof[c]["num_rate"] > 0.9]
        if rc_amt and roles.get("amount") != rc_amt[0]:
            notes.append(f"{p.name}: using `{rc_amt[0]}` (already {reporting}) as amount instead of "
                         f"`{roles.get('amount')}`; currency/rate columns dropped (CHECK)")
            roles["amount"] = rc_amt[0]
            roles.pop("currency", None)
            roles.pop("fx_rate", None)
        fx_sem = None
        if "fx_rate" in roles and "currency" in roles:
            fx_sem = _guess_rate_semantics(df, roles["currency"], roles["fx_rate"], reporting)
            if fx_sem:
                notes.append(f"{p.name}: `{roles['fx_rate']}` looks like {fx_sem} (CHECK fx.rate_semantics)")
        cols = {f: c for f, c in roles.items() if f != "record_id"}
        # join every descriptive text column (Item Name, Item Description, Short Text, ...)
        if "description" in roles:
            desc_rx = r"(?i)(item|material|product|service|line)[\s_]*(name|desc|description|text)|short[\s_]*text|" \
                      r"long[\s_]*text|^description$|narration|particulars"
            more = [c for c in df.columns if c != roles["description"] and c not in roles.values()
                    and re.search(desc_rx, c) and prof[c]["alpha_rate"] > 0.5 and prof[c]["avg_len"] > 4]
            if more:
                first = sorted([roles["description"]] + more,
                               key=lambda c: (0 if re.search(r"(?i)name", c) else 1, list(df.columns).index(c)))
                cols["description"] = first
                notes.append(f"{p.name}: description joins {first} (CHECK)")
        if extra["context"]:
            cols["context"] = extra["context"]
        groups = [c for c in df.columns if re.search(r"(?i)contract|agreement", c) and c not in roles.values()
                  and 1 < prof[c]["nunique"] < 0.5 * max(len(df), 1)]
        if groups:
            consistency_groups.extend(groups[:2])
            notes.append(f"{p.name}: {groups[:2]} used as consistency groups (lines on one contract tend to "
                         f"share a category) (CHECK)")
        name = re.sub(r"[^a-z0-9]+", "_", p.stem.lower()).strip("_") or f"src{i}"
        src = {"name": name, "file": _uniq([p.name, re.sub(r"\d+", "*", p.stem) + p.suffix]),
               "record_id": {"column": roles.get("record_id", df.columns[0])},
               "amount_type": "net", "columns": cols}
        amt_col = roles.get("amount", "")
        if re.search(r"(?i)gross|incl|total", amt_col) and ("tax_rate" in roles or "tax_amount" in roles):
            src["amount_type"] = "gross"
            src["tax_rate_default"] = 0.0
            notes.append(f"{p.name}: amount `{amt_col}` looks gross, so tax is removed (CHECK)")
        # plausible year window from the data itself (1st to 99th percentile, plus one year)
        yrs = []
        for f in ("date", "posting_date"):
            if f in roles:
                d = parse_date_series(df[roles[f]].astype(str).head(20000))
                yrs += list(d.dt.year.dropna().astype(int))
        if yrs:
            ys = pd.Series(yrs)
            window_years.append((int(ys.quantile(0.01)) - 1, int(ys.quantile(0.99)) + 1))
        # day-first or month-first, decided from the values of the date columns
        votes = {guess_dayfirst(df[roles[f]]) for f in ("date", "posting_date") if f in roles} - {None}
        if len(votes) == 1:
            src["dayfirst"] = votes.pop()
            notes.append(f"{p.name}: dates are {'day' if src['dayfirst'] else 'month'}-first (from values)")
        elif len(votes) > 1:
            notes.append(f"{p.name}: date columns disagree on day/month order (CHECK dayfirst)")
        elif any(f in roles for f in ("date", "posting_date")):
            notes.append(f"{p.name}: day/month order not decidable from values; ISO or textual dates "
                         f"are parsed either way (CHECK dayfirst)")
        if "status" in roles:
            vals = sorted(df[roles["status"]].astype(str).map(clean_text).unique())[:50]
            smap, unknown = _status_map(vals)
            src["status_map"] = smap
            if unknown:
                notes.append(f"{p.name}: status values not recognised {unknown}, treated as NOT_APPROVED (CHECK)")
        if "currency" in roles:
            cur = df[roles["currency"]].astype(str).str.strip().str.upper()
            top = cur[cur != ""].value_counts()
            notes.append(f"{p.name}: currencies {top.head(8).to_dict()}")
            foreign = [c for c in top.index if c != reporting]
            if foreign and "fx_rate" not in roles:
                notes.append(f"{p.name}: non-{reporting} currencies {foreign[:8]} and no rate column: the shipped "
                             f"ECB monthly table converts them (flag FX_PUBLIC_TABLE); add the client's table under "
                             f"`fx.table` if one is provided (CHECK)")
        else:
            src["currency_default"] = reporting
            notes.append(f"{p.name}: no currency column, assumed {reporting} (CHECK)")
        if "gl_description" in roles:
            gl = df[roles["gl_description"]].astype(str).map(clean_text)
            susp = sorted({g for g in gl.unique() if re.search(
                r"(?i)interco|inter-co|\bi/c\b|^ic\d|tax|vat|gst|interest|royalt|dividend|loan|payroll|salar|bank charge", g)})
            if susp:
                np_rules.append({"name": "suspect_gl_accounts", "field": "gl_description",
                                 "values": susp, "_note": "CHECK: drafted from account names"})
                notes.append(f"{p.name}: GL accounts drafted as NON_PROCUREMENT: {susp[:12]}")
        if "description" in roles:
            all_context_samples += list(df[roles["description"]].astype(str).map(clean_text)
                                        .replace("", np.nan).dropna().sample(
                                            min(60, len(df)), random_state=0))
        if fx_sem:
            fx_sem_any = fx_sem
        sources.append(src)
    if not sources:
        raise ValueError("no transaction file found")

    # tree paths for context
    lv = tax_block.get("level_columns") or []
    tax_paths = [" > ".join(x for x in r if x) for r in tax_df[lv].astype(str).itertuples(index=False)] if lv else []
    ctx = _company_context(tax_paths, all_context_samples) if use_llm else None

    cfg = {
        "company": company,
        "reporting_currency": reporting,
        "search_dirs": [],
        "taxonomy": tax_block,
        "sources": sources,
        "exclusion_precedence": ["CANCELLED", "NOT_APPROVED", "NON_PROCUREMENT", "DUPLICATE"],
        "non_procurement": [{k: v for k, v in r.items() if not k.startswith("_")} for r in np_rules],
        "duplicates": {"within_source": True, "fuzzy": True},
        "cross_source": {"enabled": False},
        "dates": {"dayfirst": next((s_["dayfirst"] for s_ in sources if "dayfirst" in s_), True),
                  "valid_window_years": [min(w[0] for w in window_years), max(w[1] for w in window_years)]
                  if window_years else [2000, 2035]},
        "dedupe": {"supplier": {"use_supplier_id": True, "fuzzy_threshold": 92},
                   "item": {"use_item_code": True, "fuzzy_threshold": 95}},
        "classification": {
            "company_context": ctx or "TODO: one or two sentences on the industry and what it buys",
            "consistency": {"group_fields": consistency_groups, "strong": 0.8, "weak": 0.7, "share": 0.7},
            "labels": None,
            # Claude only, same keys as the defaults in spendcube/config.py
            "llm": {**copy.deepcopy(DEFAULTS["classification"]["llm"]), "cost_cap_usd": 60.0, "concurrency": 16},
            "embeddings": {"provider": "lsa", "dim": 256},
        },
        "output": {"template": "output_template.csv"},
    }
    if fx_table or fx_sem_any:
        cfg["fx"] = {**(fx_table or {}), "rate_semantics": fx_sem_any or "units_per_reporting",
                     "ignore_rate_for_reporting_currency": True, "sanity_band": [0.33, 3.0]}
    if len(sources) > 1:
        notes.append("More than one transaction file: if they overlap, enable cross_source and set primary/secondary (CHECK)")
    header = ["# DRAFT config generated by `run.py init-config`. Review every CHECK note, then run:",
              f"#   python run.py --config {out} --data {data} --out submission/arnav_mishra__{company}.csv",
              "# Review notes:"] + [f"#  - {n}" for n in notes] + [""]
    outp = Path(out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text("\n".join(header) + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True, width=110))
    return outp
