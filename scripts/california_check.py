"""External check of the unseen-tree method on public data.

Source: "Purchase Order Data" (fiscal years 2012 to 2015), California Department of General
Services, https://data.ca.gov/dataset/purchase-order-data ("No Restrictions on Public Use"),
also mirrored on Kaggle as sohier/large-purchases-by-the-state-of-ca. No data is stored in this
repository: download the CSV yourself and pass its path.

The script samples lines from the most common UNSPSC leaves, hides the UNSPSC columns, writes a
company folder with an invoice-line file and a 4-level tree, drafts a config with init-config,
runs the pipeline and scores the output against the hidden codes.

  python scripts/california_check.py --csv ca_po.csv --out out/ca --lines 2000 --leaves 200
  python scripts/california_check.py --csv ca_po.csv --out out/ca --lines 2000 --leaves 200 --no-llm
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spendcube.classify.llm import load_dotenv_if_present  # noqa: E402
from spendcube.init_config import draft_config  # noqa: E402
from spendcube.pipeline import run  # noqa: E402

LEVELS = ["Segment Title", "Family Title", "Class Title", "Commodity Title"]


def build(csv: Path, out: Path, n_lines: int, n_leaves: int, seed: int = 11) -> pd.DataFrame:
    d = pd.read_csv(csv, dtype=str, keep_default_na=False, low_memory=False)
    for c in LEVELS + ["Item Name", "Supplier Name", "Total Price", "Supplier Code"]:
        d = d[d[c].str.strip() != ""]
    d = d[~d["Total Price"].str.fullmatch(r"\$?0(?:\.0+)?")]
    top = d.groupby(LEVELS).size().sort_values(ascending=False).head(n_leaves).index
    d = d.set_index(LEVELS).loc[top].reset_index()
    d = d.sample(n=min(n_lines, len(d)), random_state=seed).reset_index(drop=True)
    d["Line ID"] = [f"CA{i:07d}" for i in range(1, len(d) + 1)]
    data = out / "data"
    data.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "Line ID": d["Line ID"], "PO Number": d["Purchase Order Number"], "Creation Date": d["Creation Date"],
        "Purchase Date": d["Purchase Date"], "Department": d["Department Name"],
        "Acquisition Type": d["Acquisition Type"], "Vendor ID": d["Supplier Code"],
        "Vendor Name": d["Supplier Name"], "Item Name": d["Item Name"], "Item Description": d["Item Description"],
        "Quantity": d["Quantity"], "Unit Price": d["Unit Price"], "Total Price": d["Total Price"],
    }).to_csv(data / "invoice_lines.csv", index=False)
    pd.DataFrame(list(top), columns=["Level 1", "Level 2", "Level 3", "Level 4"]).to_csv(
        data / "taxonomy.csv", index=False)
    truth = pd.DataFrame({"record_id": d["Line ID"], "true_supplier": d["Supplier Code"],
                          **{f"t{i + 1}": d[c] for i, c in enumerate(LEVELS)}})
    truth.to_csv(out / "truth.csv", index=False)
    return truth


def pairwise_f1(pred, gold) -> float:
    df = pd.DataFrame({"p": list(pred), "g": list(gold)})
    c2 = lambda s: (s * (s - 1) / 2).sum()  # noqa: E731
    tp, pp, gp = c2(df.groupby(["p", "g"]).size()), c2(df.groupby("p").size()), c2(df.groupby("g").size())
    prec, rec = tp / max(pp, 1), tp / max(gp, 1)
    return 2 * prec * rec / max(prec + rec, 1e-12)


def score(out_csv: Path, truth: pd.DataFrame) -> dict:
    o = pd.read_csv(out_csv, dtype=str, keep_default_na=False).merge(truth, on="record_id")
    inc = o[o["include"] == "1"].copy()
    w = pd.to_numeric(inc["spend"], errors="coerce").abs().fillna(0)
    res = {"lines": len(o), "included": len(inc),
           "top_1pct_spend_share": round(float(w.sort_values(ascending=False).head(max(1, len(w) // 100)).sum()
                                               / max(w.sum(), 1e-9)), 3)}
    for k in range(1, 5):
        ok = inc[f"l{k}"] == inc[f"t{k}"]
        res[f"L{k}"] = {"line": round(float(ok.mean()), 3), "spend": round(float((ok * w).sum() / w.sum()), 3)}
    res["supplier_pairwise_f1"] = round(float(pairwise_f1(inc["supplier_key"], inc["true_supplier"])), 3)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--out", default=ROOT / "out" / "ca", type=Path)
    ap.add_argument("--lines", type=int, default=2000)
    ap.add_argument("--leaves", type=int, default=200)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--max-llm-cost", type=float, default=6.0)
    a = ap.parse_args()
    load_dotenv_if_present(ROOT)
    truth = build(a.csv, a.out, a.lines, a.leaves)
    cfg = draft_config(str(a.out / "data"), "taxonomy.csv", str(a.out / "config.yaml"), company="CA",
                       use_llm=not a.no_llm)
    text = cfg.read_text().replace(
        "TODO: one or two sentences on the industry and what it buys",
        "A US state government buying goods and services for its agencies.")
    cfg.write_text(text)
    rep = run(str(cfg), str(a.out / "data"), str(a.out / "output.csv"), use_llm=not a.no_llm,
              cache_dir=str(ROOT / ".cache"), llm_cost_cap=a.max_llm_cost)
    print({"runtime_sec": rep["runtime_sec"], "llm_cost_usd": (rep["classification"].get("llm") or {}).get("cost_usd"),
           **score(a.out / "output.csv", truth)})


if __name__ == "__main__":
    main()
