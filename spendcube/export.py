"""Write the output file in the exact template layout, plus an audit sidecar."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .taxonomy import Taxonomy

OUTPUT_COLUMNS = ["record_id", "include", "exclusion_reason", "supplier_key", "item_key", "spend",
                  "l1", "l2", "l3", "l4", "confidence"]


def template_columns(cfg: dict, data_dir: Path | None = None) -> list[str]:
    """Columns of output_template.csv if found (data folder, its parent, search dirs),
    else the built-in copy of the template header (identical)."""
    from .paths import candidate_dirs, resolve_file
    t = (cfg.get("output") or {}).get("template")
    if t and data_dir is not None:
        dirs = candidate_dirs(cfg, data_dir)
        dirs = dirs + [d.parent for d in dirs] + [Path.cwd()]
        p = resolve_file(t, cfg, data_dir, required=False, dirs=dirs)
        if p is not None:
            cols = list(pd.read_csv(p, nrows=0).columns)
            if set(OUTPUT_COLUMNS[:6]).issubset(cols):
                return cols
    return list(OUTPUT_COLUMNS)


def build_output(df: pd.DataFrame, tax: Taxonomy, columns: list[str]) -> pd.DataFrame:
    levels = [c for c in columns if c.startswith("l") and c[1:].isdigit()]
    out = pd.DataFrame({
        "record_id": df["record_id"],
        "include": df["include"].astype(int),
        "exclusion_reason": df["exclusion_reason"].fillna(""),
        "supplier_key": df["supplier_key"].fillna(""),
        "item_key": df["item_key"].fillna(""),
        "spend": df["spend"].round(2),
        "confidence": df["confidence"].fillna(0.0).round(3),
    })
    paths = df["node"].map(lambda n: tax.path_of(n) if isinstance(n, str) and n in tax.nodes else ())
    inc = out["include"] == 1
    for k, lv in enumerate(levels):
        out[lv] = paths.map(lambda p: p[k] if k < len(p) else "").where(inc, "")
    # As in the template, excluded rows carry no category; their confidence is the
    # confidence of the exclusion decision.
    if "exclusion_confidence" in df.columns:
        out.loc[~inc, "confidence"] = df.loc[~inc, "exclusion_confidence"].round(3)
    return out[columns]


def sidecar_path(out_path: Path, suffix: str) -> Path:
    """Audit and report files go to <repo>/out/ (git-ignored), never next to a
    submission file: the audit holds raw descriptions, which are confidential.
    SPENDCUBE_SIDECAR_DIR overrides the folder; the tests point it at a temp folder."""
    import os
    from .paths import REPO_ROOT
    d = Path(os.environ["SPENDCUBE_SIDECAR_DIR"]) if os.environ.get("SPENDCUBE_SIDECAR_DIR") else REPO_ROOT / "out"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{Path(out_path).stem}{suffix}"


def write_outputs(df: pd.DataFrame, out: pd.DataFrame, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, encoding="utf-8")
    audit_cols = [c for c in ["record_id", "source", "exclusion_rule", "matched_to", "match_pass",
                              "match_cluster", "flags", "currency_norm", "fx_rate_used", "fx_source",
                              "amount_num", "net_amount", "spend_rc", "class_source", "node", "confidence",
                              "calib_p1", "calib_margin",
                              "unit_id", "supplier_name", "description"] if c in df.columns]
    audit = sidecar_path(path, "_audit.csv")
    df[audit_cols].to_csv(audit, index=False, encoding="utf-8")
    return audit
