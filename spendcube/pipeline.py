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
from .export import build_output, sidecar_path, template_columns, write_outputs
from .ingest import ingest_all
from .match import match_sources
from .paths import REPO_ROOT
from .spend_rules import apply_rules, resolve_codes
from .taxonomy import load_taxonomy
from .validate import validate_output

log = logging.getLogger("spendcube")


class LLMUnavailableError(RuntimeError):
    """--require-llm was given but no Claude model answered the health check."""


def llm_health_check(cfg: dict, use_llm: bool, require_llm: bool) -> dict:
    """Ask each configured Claude model once, print the result, and decide.
    Returns {'enabled', 'models': {model: 'ok'|reason}, 'usable', 'message'}."""
    from .classify.llm import check_llm_health, llm_settings
    lcfg = llm_settings(cfg)
    if not use_llm or not lcfg.get("enabled", True):
        why = "--no-llm" if not use_llm else "disabled in config"
        info = {"enabled": False, "models": {}, "usable": False, "message": f"LLM off ({why})"}
        print(f"LLM health: off ({why})", flush=True)
        if require_llm:
            raise LLMUnavailableError(f"--require-llm given but the LLM is {why}")
        return info
    models = check_llm_health(cfg)
    for m, st in models.items():
        print(f"LLM health: {m:20s} {st}", flush=True)
    primary_models = [lcfg.get("primary", {}).get("model")] + [f.get("model") for f in lcfg.get("fallbacks") or []]
    usable = any(models.get(m) == "ok" for m in primary_models if m)
    msg = "LLM ready" if usable else ("LLM ENABLED BUT UNAVAILABLE: no primary/fallback model answered "
                                      f"({'; '.join(f'{m}: {v}' for m, v in models.items())}). "
                                      "Classification falls back to the offline layers.")
    if not usable:
        if require_llm:
            raise LLMUnavailableError(msg.replace("Classification falls back to the offline layers.",
                                                  "Stopping because --require-llm was given."))
        log.warning(msg)
    return {"enabled": True, "models": models, "usable": usable, "message": msg}


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_jsonable(x) for x in o]
    if hasattr(o, "item"):
        return o.item()
    return o


def _write_report(report: dict, out_path: Path) -> Path:
    rp = sidecar_path(out_path, "_report.json")
    report["report"] = str(rp)
    rp.write_text(json.dumps(_jsonable(report), indent=2, default=str))
    return rp


def run(config_path: str, data_dir: str, out_path: str, *, use_llm: bool = True,
        use_labels: bool = True, cache_dir: str = ".cache", llm_cost_cap: float | None = None,
        label_filter=None, require_llm: bool = False, health: dict | None = None) -> dict:
    t0 = time.time()
    cfg = load_config(config_path)
    if llm_cost_cap is not None:
        cfg["classification"]["llm"]["cost_cap_usd"] = llm_cost_cap
    data_dir, out_path = Path(data_dir), Path(out_path)
    cache = Path(cache_dir) / str(cfg.get("company", "X"))
    report: dict = {"company": cfg.get("company"), "config": str(config_path), "data": str(data_dir)}

    # 1. provider health first, so a missing key is visible before any work is done
    health = health if health is not None else llm_health_check(cfg, use_llm, require_llm)
    report["llm_health"] = health
    use_llm = use_llm and health.get("usable", False)

    tax = load_taxonomy(cfg, data_dir)
    report["taxonomy"] = {"nodes": len(tax.nodes), "leaves": len(tax.leaves), "depth": tax.max_depth}
    raw = ingest_all(cfg, data_dir)
    n_input = len(raw)
    report["input_rows_by_source"] = raw["source"].value_counts().to_dict()
    columns = template_columns(cfg, data_dir)
    if n_input == 0:
        # header-only input: write a header-only output instead of failing
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=columns).to_csv(out_path, index=False)
        log.warning("input files contain no records: wrote a header-only output")
        report.update({"validation": {"rows": 0, "included_rows": 0, "excluded_by_reason": {},
                                      "input_spend_rc": 0.0, "included_spend": 0.0, "excluded_spend": 0.0,
                                      "suppliers": 0, "items": 0, "mean_confidence_included": None},
                       "classification": {"llm": {}}, "runtime_sec": round(time.time() - t0, 1),
                       "output": str(out_path), "audit": None})
        _write_report(report, out_path)
        report["_df"], report["_out"], report["_tax"], report["_units"] = raw, pd.DataFrame(columns=columns), tax, None
        return report
    df = clean(raw, cfg, data_dir)

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
    report["supplier_hubs"] = dict(df["supplier_key"].attrs.get("hub_stats", {}))
    if train_df is not None:
        train_df["supplier_key"] = dedupe_suppliers(train_df, cfg, memory)
    # matching compares real suppliers, so it uses the name/id clusters *before* golden
    # keys are applied (a client's catch-all golden key must not make two firms "the same")
    match_key = dedupe_suppliers(df, cfg, None) if memory else df["supplier_key"]
    match_stats = match_sources(df, cfg, supplier_key=match_key)
    resolve_codes(df, cfg)
    link = df["matched_to"] if "matched_to" in df.columns else None
    df["item_key"] = dedupe_items(df, cfg, df["supplier_key"], memory, link=link)
    report["rules"] = rule_stats
    if match_stats:
        sec = df["source"] == cfg["cross_source"]["secondary"]
        match_stats["secondary_only_included_spend"] = round(float(df.loc[sec & (df["include"] == 1), "spend"].sum()), 2)
        match_stats["secondary_only_included_rows"] = int((sec & (df["include"] == 1)).sum())
    report["cross_source"] = match_stats

    inc = df["include"] == 1
    sub = df[inc].copy()
    if len(sub):
        cstats = clf.classify(sub, cfg, tax, data_dir, cache, labels=labels, use_llm=use_llm,
                              train_df=train_df, llm_health=health.get("models"))
        for c in ("node", "confidence", "class_source", "unit_id", "calib_p1", "calib_margin",
                  "calib_sim1", "calib_sim2", "calib_weak"):
            df[c] = sub[c].reindex(df.index) if c in sub else None
    else:
        cstats = {"llm": {}, "class_source_lines": {}}
        for c in ("node", "confidence", "class_source", "unit_id", "calib_p1", "calib_margin",
                  "calib_sim1", "calib_sim2", "calib_weak"):
            df[c] = None
    report["classification"] = cstats

    out = build_output(df, tax, columns)
    report["validation"] = validate_output(out, df, tax, columns, n_input)
    audit = write_outputs(df, out, out_path)
    report["runtime_sec"] = round(time.time() - t0, 1)
    report["output"] = str(out_path)
    report["audit"] = str(audit)
    _write_report(report, out_path)
    report["_df"], report["_out"], report["_tax"], report["_units"] = df, out, tax, cstats.get("_units")
    return report
