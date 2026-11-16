"""Load and validate a company config (YAML or JSON).

The config is the only place that knows about a company's files, column names,
formats and business rules. Paths inside the config are relative to the data
folder passed on the command line, so the same config works for A and A2.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml

EXCLUSION_CODES = ("CANCELLED", "NOT_APPROVED", "NON_PROCUREMENT", "DUPLICATE")

# Canonical fields a source column can be mapped to. A field may map to one column
# or a list of columns (text fields are joined with " | ", other fields take the
# first non-blank value).
CANONICAL_FIELDS = {
    "line_id", "invoice_number", "invoice_line", "date", "posting_date",
    "supplier_name", "supplier_id", "supplier_tax_id",
    "description", "item_code", "quantity", "uom", "unit_price",
    "amount", "tax_amount", "tax_rate", "currency", "fx_rate",
    "status", "po_number", "po_item", "po_reference",
    "gl_account", "gl_description", "cost_centre", "document_type", "entity",
    "context",  # extra text used only as classification context
}

DEFAULTS: dict[str, Any] = {
    "company": "X",
    "reporting_currency": "USD",
    "taxonomy": {"file": None, "format": "auto", "level_columns": None},
    "sources": [],
    "exclusion_precedence": list(EXCLUSION_CODES),
    "status": {
        # lower-cased status values -> outcome. Unknown statuses are logged and
        # treated as `unknown_status_outcome`.
        "include": ["paid", "approved", "posted", "cleared", "complete", "completed", "closed"],
        "CANCELLED": ["cancelled", "canceled", "void", "voided", "reversed", "deleted"],
        "NOT_APPROVED": ["rejected", "on hold", "hold", "submitted", "disputed", "pending",
                         "draft", "in review", "awaiting approval", "parked", "blocked"],
        "unknown_status_outcome": "NOT_APPROVED",   # brief: only approved or paid lines count
    },
    "non_procurement": [],
    "duplicates": {"within_source": True, "key": None},
    "cross_source": {"enabled": False},
    "fx": {
        "table": None,                 # file with currency, month, units_per_usd style columns
        "table_columns": {"currency": "currency", "month": "month", "rate": "units_per_usd"},
        "rate_semantics": "units_per_reporting",  # line currency units per 1 reporting unit
        "ignore_rate_for_reporting_currency": True,
        "sanity_band": [0.33, 3.0],    # given rate / table rate must fall in band, else table
        "month_from": "date",          # canonical date field used for the FX month
    },
    "dates": {
        "dayfirst": True,
        "valid_window_years": [2000, 2035],
        "max_days_after_posting": 120,
        "max_days_before_posting": 1100,
    },
    "dedupe": {
        "supplier": {
            "use_supplier_id": True,
            "fuzzy_threshold": 92,
            "legal_suffixes": None,  # None -> built-in list
        },
        "item": {"use_item_code": True, "fuzzy_threshold": 95, "scope": "global"},
    },
    "classification": {
        "text_fields": ["description", "supplier_name", "gl_description", "context"],
        "labels": None,
        # Claude only. These keys are exactly the ones spendcube/classify/llm.py reads, so a
        # config without an llm block still gets the full Claude setup.
        "llm": {
            "enabled": True,
            "primary": {"vendor": "anthropic", "model": "claude-opus-5-5", "effort": "low"},
            "second": {"vendor": "anthropic", "model": "claude-sonnet-5-5", "effort": "low"},
            "judge": {"vendor": "anthropic", "model": "claude-opus-5-5", "effort": "low"},
            "tail": {"vendor": "anthropic", "model": "claude-sonnet-5-5", "effort": "low"},
            "fallbacks": [{"vendor": "anthropic", "model": "claude-sonnet-5-5", "effort": "low"}],
            "primary_spend_share": 0.95,       # units holding the top 95% of spend go to the primary model
            "shortlist_above_leaves": 400,     # bigger trees: per-batch candidate shortlist, not the full tree
            "shortlist_size": 25,
            "shortlist_batch_cap": 150,
            "batch_size": 25,
            "concurrency": 10,
            "low_confidence_threshold": 0.5,   # labelled runs: LLM only below this supervised confidence
            "vote_threshold": 0.7,             # second opinion below this primary confidence
            "second_max_units": 1500,
            "second_top_spend_share": 0.5,     # the units holding the top half of spend always get one
            "second_top_spend_max_units": 400,
            "max_units": 20000,
            "cost_cap_usd": 15.0,
            "supplier_profiles": True,
            "scope_notes": True,
        },
        "embeddings": {"provider": "lsa", "dim": 256},
    },
    "output": {"template": None},
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


class ConfigError(ValueError):
    pass


def load_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config not found: {path}")
    text = path.read_text(encoding="utf-8")
    raw = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise ConfigError(f"Config {path} is not a mapping")
    cfg = _deep_merge(DEFAULTS, raw)
    cfg["_path"] = str(path)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    errs = []
    if not cfg.get("sources"):
        errs.append("`sources` must list at least one input file")
    if not (cfg.get("taxonomy") or {}).get("file"):
        errs.append("`taxonomy.file` is required")
    names = set()
    for i, s in enumerate(cfg.get("sources", [])):
        where = f"sources[{i}]"
        for req in ("name", "file", "record_id"):
            if req not in s:
                errs.append(f"{where}: missing `{req}`")
        if s.get("name") in names:
            errs.append(f"{where}: duplicate source name {s.get('name')}")
        names.add(s.get("name"))
        cols = s.get("columns", {})
        bad = set(cols) - CANONICAL_FIELDS
        if bad:
            errs.append(f"{where}: unknown canonical fields {sorted(bad)}; allowed: {sorted(CANONICAL_FIELDS)}")
        if "amount" not in cols:
            errs.append(f"{where}: columns.amount is required")
        if s.get("amount_type", "net") not in ("net", "gross"):
            errs.append(f"{where}: amount_type must be net or gross")
    for code in cfg.get("exclusion_precedence", []):
        if code not in EXCLUSION_CODES:
            errs.append(f"exclusion_precedence: unknown code {code}")
    for r in cfg.get("non_procurement", []) or []:
        if "field" not in r or not ({"pattern", "values"} & set(r)):
            errs.append(f"non_procurement rule {r.get('name')}: needs `field` and `pattern` or `values`")
    xs = cfg.get("cross_source", {})
    if xs.get("enabled"):
        for k in ("primary", "secondary"):
            if xs.get(k) not in names:
                errs.append(f"cross_source.{k} must name a source ({sorted(names)})")
    if errs:
        raise ConfigError("Invalid config:\n  - " + "\n  - ".join(errs))
