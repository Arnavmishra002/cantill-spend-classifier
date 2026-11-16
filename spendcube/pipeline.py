"""End-to-end run: one config + one data folder -> one validated output file."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd

from . import classify as clf
from .clean import clean
from .config import load_config
from .dedupe import dedupe_items, dedupe_suppliers
from .export import build_output, template_columns, write_outputs
from .ingest import ingest_all
from .match import match_sources
from .paths import REPO_ROOT
from .spend_rules import apply_rules, resolve_codes
from .taxonomy import load_taxonomy
from .validate import validate_output

log = logging.getLogger("spendcube")


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_jsonable(x) for x in o]
    if hasattr(o, "item"):
        return o.item()
    return o


def run(config_path: str, data_dir: str, out_path: str, *, use_llm: bool = True,
        use_labels: bool = True, cache_dir: str = ".cache", llm_cost_cap: float | None = None,
        label_filter=None) -> dict:
    t0 = time.time()
    cfg = load_config(config_path)
    if llm_cost_cap is not None:
        cfg["classification"]["llm"]["cost_cap_usd"] = llm_cost_cap
    data_dir, out_path = Path(data_dir), Path(out_path)
    cache = Path(cache_dir) / str(cfg.get("company", "X"))
    report: dict = {"company": cfg.get("company"), "config": str(config_path), "data": str(data_dir)}

    raw = ingest_all(cfg, data_dir)
    n_input = len(raw)
    report["input_rows_by_source"] = raw["source"].value_counts().to_dict()
    df = clean(raw, cfg, data_dir)
    tax = load_taxonomy(cfg, data_dir)
    report["taxonomy"] = {"nodes": len(tax.nodes), "leaves": len(tax.leaves), "depth": tax.max_depth}

    labels = clf.load_labels(cfg, data_dir, tax) if use_labels else None
    if labels is not None and label_filter is not None:
        labels = label_filter(labels)
    report["labels_used"] = 0 if labels is None else int(len(labels))

    # Labels may belong to an earlier drop of the same company (A when running A2):
    # then that drop's data is loaded as training rows only.
    train_df = None
    if labels is not None:
        hit = labels["record_id"].isin(set(df["record_id"])).mean()
        tdir = (cfg["classification"].get("labels") or {}).get("training_data_dir")
        if hit < 0.5 and tdir:
            tpath = Path(tdir) if Path(tdir).is_absolute() else REPO_ROOT / tdir
            if tpath.resolve() != data_dir.resolve() and tpath.exists():
                log.info("labels refer to an earlier drop: loading training rows from %s", tpath)
                train_df = clean(ingest_all(cfg, tpath), cfg, tpath)
        report["labels_match_current_rows"] = round(float(hit), 3)
    rule_stats = apply_rules(df, cfg)
    mem_src = train_df if train_df is not None else df
    memory = clf.build_memory(mem_src, labels, mem_src["supplier_name"]) if labels is not None else {}
    df["supplier_key"] = dedupe_suppliers(df, cfg, memory)
    if train_df is not None:
        train_df["supplier_key"] = dedupe_suppliers(train_df, cfg, memory)
    match_stats = match_sources(df, cfg, supplier_key=df["supplier_key"])
    resolve_codes(df, cfg)
    link = df["matched_to"] if "matched_to" in df.columns else None
    df["item_key"] = dedupe_items(df, cfg, df["supplier_key"], memory, link=link)
    report["rules"] = rule_stats
    report["cross_source"] = match_stats

    inc = df["include"] == 1
    sub = df[inc].copy()
    cstats = clf.classify(sub, cfg, tax, data_dir, cache, labels=labels, use_llm=use_llm,
                          train_df=train_df)
    for c in ("node", "confidence", "class_source", "unit_id"):
        df[c] = sub[c].reindex(df.index)
    report["classification"] = cstats

    columns = template_columns(cfg, data_dir)
    out = build_output(df, tax, columns)
    report["validation"] = validate_output(out, df, tax, columns, n_input)
    audit = write_outputs(df, out, out_path)
    report["runtime_sec"] = round(time.time() - t0, 1)
    report["output"] = str(out_path)
    report["audit"] = str(audit)
    rp = out_path.with_name(out_path.stem + "_report.json")
    rp.write_text(json.dumps(_jsonable(report), indent=2, default=str))
    report["_df"], report["_out"], report["_tax"], report["_units"] = df, out, tax, cstats.get("_units")
    return report
