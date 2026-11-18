"""Post-run validation. Any failure raises ValidationError with a clear message."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import EXCLUSION_CODES
from .taxonomy import Taxonomy


class ValidationError(RuntimeError):
    pass


def validate_output(out: pd.DataFrame, df: pd.DataFrame, tax: Taxonomy, columns: list[str],
                    n_input: int) -> dict:
    errs, report = [], {}
    if list(out.columns) != list(columns):
        errs.append(f"columns {list(out.columns)} != template {list(columns)}")
    if len(out) != n_input:
        errs.append(f"{len(out)} output rows but {n_input} input records")
    d = out["record_id"].duplicated()
    if d.any():
        errs.append(f"{int(d.sum())} duplicated record_id values, e.g. {out.loc[d, 'record_id'].head(3).tolist()}")
    if out["record_id"].astype(str).str.strip().eq("").any():
        errs.append("blank record_id")
    if not out["include"].isin([0, 1]).all():
        errs.append("include must be 0/1")
    bad = out.loc[out["include"] == 0, "exclusion_reason"]
    if (~bad.isin(EXCLUSION_CODES)).any():
        errs.append(f"excluded rows with invalid reason: {sorted(set(bad) - set(EXCLUSION_CODES))}")
    if (out.loc[out["include"] == 1, "exclusion_reason"].fillna("") != "").any():
        errs.append("included rows must have blank exclusion_reason")
    if (out.loc[out["include"] == 0, "spend"] != 0).any():
        errs.append("excluded rows must have spend 0")
    if out["spend"].isna().any():
        errs.append("spend has NaN")
    c = out["confidence"]
    if c.isna().any() or (c < 0).any() or (c > 1).any():
        errs.append("confidence must be within [0,1]")
    # categories exist in the tree, exact names, blank below the tree's depth
    levels = [x for x in columns if x.startswith("l") and x[1:].isdigit()]
    valid_paths = {tax.output_path(n.id) for n in tax.nodes.values()} | {n.path for n in tax.nodes.values()}
    paths = out[levels].fillna("").apply(lambda r: tuple(v for v in r if v), axis=1)
    inc = out["include"] == 1
    not_in_tree = ~paths.isin(valid_paths)
    if (not_in_tree & inc).any():
        ex = paths[not_in_tree & inc].head(3).tolist()
        errs.append(f"{int((not_in_tree & inc).sum())} included rows have a category path not in the tree, e.g. {ex}")
    if (paths.map(len) == 0)[inc].any():
        errs.append("included rows without a category")
    for k, lv in enumerate(levels):
        if k >= tax.max_depth and (out[lv].fillna("") != "").any():
            errs.append(f"{lv} must be blank: tree has {tax.max_depth} levels")
    # spend reconciliation: included + excluded == total input spend (reporting ccy)
    total = float(df["spend_rc"].sum())
    inc_sum = float(out.loc[inc, "spend"].sum())
    exc_sum = float(df.loc[~inc.values, "spend_rc"].sum())
    if not np.isclose(inc_sum + exc_sum, total, atol=max(1.0, abs(total) * 1e-9)):
        errs.append(f"spend does not reconcile: included {inc_sum:.2f} + excluded {exc_sum:.2f} != {total:.2f}")
    report.update({
        "rows": int(len(out)), "included_rows": int(inc.sum()),
        "excluded_by_reason": out.loc[~inc, "exclusion_reason"].value_counts().to_dict(),
        "input_spend_rc": round(total, 2), "included_spend": round(inc_sum, 2),
        "excluded_spend": round(exc_sum, 2),
        "suppliers": int(out.loc[inc, "supplier_key"].nunique()),
        "items": int(out.loc[inc, "item_key"].nunique()),
        "mean_confidence_included": round(float(out.loc[inc, "confidence"].mean()), 3) if inc.any() else None,
    })
    if errs:
        raise ValidationError("OUTPUT VALIDATION FAILED:\n  - " + "\n  - ".join(errs))
    return report
