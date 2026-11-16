"""Honest evaluation by simulating the blind test.

Modes
  zero_shot         labels switched off (simulates Company C on A's or B's tree)
  supplier_holdout  labels hidden for a random 20% of suppliers; scored on them
  fresh_slice       rows split into an 'old' drop (with labels) and a 'fresh'
                    drop (no labels), written to separate folders; the frozen
                    pipeline runs on the fresh folder exactly like A2/B2 would
Metrics: accuracy at each tree level, line-weighted and spend-weighted, on
included lines that carry a label; plus exclusion agreement when labels have
include / reason columns (B seeds).
"""
from __future__ import annotations

import copy
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .classify import load_labels
from .config import load_config
from .dedupe import supplier_compact
from .ingest import ingest_all, read_table
from .paths import REPO_ROOT, resolve_file
from .pipeline import run
from .taxonomy import load_taxonomy

log = logging.getLogger(__name__)


def _level_metrics(pred: pd.DataFrame, truth: pd.DataFrame, depth: int, weight: pd.Series) -> dict:
    res = {}
    ok_all = pd.Series(True, index=pred.index)
    for k in range(depth):
        p = pred[f"l{k+1}"].fillna("")
        t = truth[f"t{k+1}"].fillna("")
        ok = (p == t)
        ok_all &= ok
        res[f"L{k+1}_line"] = round(float(ok.mean()), 4)
        res[f"L{k+1}_spend"] = round(float((ok * weight).sum() / max(weight.sum(), 1e-9)), 4)
    return res


def score(out: pd.DataFrame, labels: pd.DataFrame, tax, restrict_ids=None) -> dict:
    lv = [f"l{k+1}" for k in range(tax.max_depth)]
    lab = labels[labels["node"].notna()].copy()
    for k in range(tax.max_depth):
        lab[f"t{k+1}"] = lab["node"].map(lambda n: tax.path_of(n)[k] if k < len(tax.path_of(n)) else "")
    m = out.merge(lab[["record_id"] + [f"t{k+1}" for k in range(tax.max_depth)]], on="record_id")
    m = m[m["include"] == 1]
    if restrict_ids is not None:
        m = m[m["record_id"].isin(restrict_ids)]
    if m.empty:
        return {"n": 0}
    w = m["spend"].abs()
    res = {"n_lines": int(len(m)), "spend": round(float(w.sum()), 2)}
    res.update(_level_metrics(m[lv], m, tax.max_depth, w))
    # accuracy by confidence band (calibration check)
    bands = pd.cut(m["confidence"], [0, 0.5, 0.7, 0.85, 0.95, 1.0001], right=False)
    leaf_ok = m[f"l{tax.max_depth}"] == m[f"t{tax.max_depth}"]
    res["leaf_acc_by_conf"] = {str(b): [int(n), round(float(a), 3)] for b, (n, a) in
                               leaf_ok.groupby(bands, observed=True).agg(["size", "mean"]).iterrows()}
    return res


def exclusion_agreement(out: pd.DataFrame, labels: pd.DataFrame) -> dict | None:
    if "label_include" not in labels:
        return None
    m = out.merge(labels[["record_id", "label_include", "label_reason"]], on="record_id")
    m = m[m["label_include"].isin(["0", "1"])]
    t = m["label_include"].astype(int)
    p = m["include"].astype(int)
    tp = int(((p == 0) & (t == 0)).sum())
    fp = int(((p == 0) & (t == 1)).sum())
    fn = int(((p == 1) & (t == 0)).sum())
    return {"n": int(len(m)), "agreement": round(float((p == t).mean()), 4),
            "excl_precision": round(tp / max(tp + fp, 1), 4), "excl_recall": round(tp / max(tp + fn, 1), 4),
            "fp": fp, "fn": fn}


def _cost_per_1k(rep: dict, n_lines: int) -> float | None:
    llm = rep["classification"].get("llm") or {}
    if not llm.get("calls"):
        return None
    return round(1000 * llm["cost_usd"] / max(n_lines, 1), 4)


def evaluate(config: str, data: str, mode: str = "zero_shot", use_llm: bool = True,
             cache_dir: str = ".cache", holdout: float = 0.2, seed: int = 0,
             max_llm_cost: float | None = None, sample_units: int | None = None) -> dict:
    cfg = load_config(config)
    data_dir = Path(data)
    tax = load_taxonomy(cfg, data_dir)
    labels = load_labels(cfg, data_dir, tax)
    if labels is None:
        raise SystemExit("evaluation needs a labels file in the config")
    comp = cfg.get("company", "X")
    outdir = REPO_ROOT / "out" / "eval"
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"{comp}_{mode}{'_llm' if use_llm else '_offline'}"
    res: dict = {"company": comp, "mode": mode, "llm": use_llm}

    cfg_path = config
    if sample_units:
        # classify a random sample of units with the LLM (cheap, unbiased estimate)
        c2 = copy.deepcopy(_strip(cfg))
        c2["classification"]["llm"]["unit_sample"] = {"n": int(sample_units), "seed": seed}
        cfg_path = outdir / f"{tag}_cfg.yaml"
        cfg_path.write_text(yaml.safe_dump(c2, sort_keys=False))

    if mode == "zero_shot":
        rep = run(str(cfg_path), str(data_dir), str(outdir / f"{tag}.csv"), use_llm=use_llm,
                  use_labels=False, cache_dir=cache_dir, llm_cost_cap=max_llm_cost)
        out = rep["_out"]
        restrict = None
        if sample_units:
            df = rep["_df"]
            restrict = set(df.loc[df["class_source"].fillna("").str.startswith("llm"), "record_id"])
            res["sampled_lines"] = len(restrict)
        res["classification"] = score(out, labels, tax, restrict)
        res["exclusions"] = exclusion_agreement(out, labels)
    elif mode == "supplier_holdout":
        raw = ingest_all(cfg, data_dir)
        names = raw["supplier_name"] if "supplier_name" in raw else pd.Series("", index=raw.index)
        ids = raw["supplier_id"] if "supplier_id" in raw else pd.Series("", index=raw.index)
        # group suppliers by name when names exist (ids can be a catch-all bucket), else by id
        if (names != "").mean() > 0.5:
            key = names.map(supplier_compact)
        else:
            key = ids.where(ids != "", names.map(supplier_compact))
        rid_key = dict(zip(raw["record_id"], key))
        sups = sorted(set(key))
        rng = np.random.default_rng(seed)
        held = set(rng.choice(sups, size=max(1, int(len(sups) * holdout)), replace=False))
        held_ids = {r for r, k in rid_key.items() if k in held}
        rep = run(str(cfg_path), str(data_dir), str(outdir / f"{tag}.csv"), use_llm=use_llm,
                  use_labels=True, cache_dir=cache_dir, llm_cost_cap=max_llm_cost,
                  label_filter=lambda L: L[~L["record_id"].isin(held_ids)])
        out = rep["_out"]
        res["held_suppliers"] = len(held)
        res["classification"] = score(out, labels, tax, held_ids)
    elif mode == "fresh_slice":
        rep, fresh_ids = _fresh_slice_run(cfg, config, data_dir, outdir, tag, use_llm, cache_dir,
                                          holdout, seed, max_llm_cost)
        out = rep["_out"]
        res["fresh_rows"] = int(len(out))
        res["classification"] = score(out, labels, tax, fresh_ids)
        res["exclusions"] = exclusion_agreement(out, labels)
        res["validation"] = rep["validation"]
    else:
        raise ValueError(mode)
    n_inc = int((rep["_out"]["include"] == 1).sum())
    res["runtime_sec"] = rep["runtime_sec"]
    res["llm_usage"] = rep["classification"].get("llm")
    res["llm_cost_per_1k_lines"] = _cost_per_1k(rep, n_inc)
    res["class_source_lines"] = rep["classification"].get("class_source_lines")
    (outdir / f"{tag}_metrics.json").write_text(json.dumps(res, indent=2, default=str))
    return res


def _strip(cfg):
    return {k: v for k, v in cfg.items() if not k.startswith("_")}


def _fresh_slice_run(cfg, config, data_dir, outdir, tag, use_llm, cache_dir, frac, seed, cap):
    """Split every source file by row (random, seeded) into old/ and fresh/ folders."""
    base = outdir / f"{tag}_split"
    old, fresh = base / "old", base / "fresh"
    for d in (old, fresh):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    fresh_ids = set()
    for s in cfg["sources"]:
        p = resolve_file(s["file"], cfg, data_dir, dirs=[data_dir])
        t = read_table(p)
        m = rng.random(len(t)) < frac
        t[~m].to_csv(old / p.name, index=False)
        t[m].to_csv(fresh / p.name, index=False)
        rid = s["record_id"]["column"] if isinstance(s["record_id"], dict) else s["record_id"]
        pre = s["record_id"].get("prefix", "") if isinstance(s["record_id"], dict) else ""
        fresh_ids |= set(pre + t.loc[m, rid].astype(str))
    # labels / tree / fx stay in the original folder (found through search_dirs);
    # the old drop is the training folder
    c2 = copy.deepcopy(_strip(cfg))
    c2["search_dirs"] = [str(data_dir.resolve())] + list(cfg.get("search_dirs", []))
    c2["classification"]["labels"]["training_data_dir"] = str(old.resolve())
    cfg_path = base / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(c2, sort_keys=False))
    rep = run(str(cfg_path), str(fresh), str(outdir / f"{tag}.csv"), use_llm=use_llm, use_labels=True,
              cache_dir=cache_dir, llm_cost_cap=cap)
    return rep, fresh_ids
