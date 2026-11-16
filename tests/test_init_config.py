"""init-config on a synthetic unfamiliar company: roles, tree format, and an end-to-end
offline run of the drafted config. No API calls (use_llm=False)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spendcube.init_config import draft_config  # noqa: E402
from spendcube.pipeline import run  # noqa: E402


def _make_company(tmp: Path):
    rows = []
    sups = ["Hotel Grand Pvt Ltd", "HOTEL GRAND PRIVATE LIMITED", "Swift Couriers", "Acme Steel & Co",
            "Bharat Telecom Ltd", "Interco Services (India)"]
    descs = ["Room booking for sales team", "Room booking for sales team", "Courier charges March",
             "MS steel plate 10mm", "Mobile data plan", "Intercompany recharge Q1"]
    for i in range(60):
        k = i % 6
        rows.append({
            "Txn Line ID": f"L{i:05d}", "Bill No": f"BILL-{1000 + i // 2}", "Bill Date": f"{(i % 28) + 1:02d}/03/2025",
            "Vendor Name": sups[k], "GSTIN": "27AAACH1234A1Z5" if k < 2 else "",
            "Line Description": descs[k], "Qty": "1", "Net Amount": f"{1000 + 10 * k:,.2f}",
            "Currency": "USD", "Invoice Status": ["Paid", "Approved", "Rejected", "Paid", "Cancelled", "Paid"][k],
        })
    # an exported-twice copy with different formatting
    dup = dict(rows[0])
    dup["Txn Line ID"] = "L99999"
    dup["Bill No"] = "BILL 1000"
    dup["Line Description"] = "ROOM BOOKING FOR SALES TEAM"
    rows.append(dup)
    pd.DataFrame(rows).to_csv(tmp / "ap_lines.csv", index=False)
    tree = [
        {"Level 1": "Travel", "Level 2": "Lodging", "Level 3": "Hotels", "Level 4": "Domestic Hotels"},
        {"Level 1": "Logistics", "Level 2": "Courier", "Level 3": "Express", "Level 4": "Domestic Courier"},
        {"Level 1": "Materials", "Level 2": "Metals", "Level 3": "Steel", "Level 4": "Plates"},
        {"Level 1": "IT & Telecom", "Level 2": "Telecom", "Level 3": "Mobile", "Level 4": "Data Plans"},
    ]
    pd.DataFrame(tree).to_csv(tmp / "category_tree.csv", index=False)
    pd.DataFrame(columns=["record_id", "include", "exclusion_reason", "supplier_key", "item_key", "spend",
                          "l1", "l2", "l3", "l4", "confidence"]).to_csv(tmp / "output_template.csv", index=False)


def test_draft_and_run_offline(tmp_path):
    _make_company(tmp_path)
    cfg_path = draft_config(str(tmp_path), "category_tree.csv", str(tmp_path / "c.yaml"), company="T",
                            use_llm=False)
    cfg = yaml.safe_load(cfg_path.read_text())
    src = cfg["sources"][0]
    cols = src["columns"]
    assert src["record_id"]["column"] == "Txn Line ID"
    assert cols["amount"] == "Net Amount"
    assert cols["supplier_name"] == "Vendor Name"
    assert cols["description"] == "Line Description"
    assert cols["date"] == "Bill Date"
    assert cols["status"] == "Invoice Status"
    assert cols.get("supplier_tax_id") == "GSTIN"
    assert cfg["taxonomy"]["level_columns"] == ["Level 1", "Level 2", "Level 3", "Level 4"]
    assert "rejected" in src["status_map"]["NOT_APPROVED"]
    assert "cancelled" in src["status_map"]["CANCELLED"]

    rep = run(str(cfg_path), str(tmp_path), str(tmp_path / "out.csv"), use_llm=False,
              cache_dir=str(tmp_path / "cache"))
    out = rep["_out"]
    assert len(out) == 61
    reasons = dict(zip(out["record_id"], out["exclusion_reason"]))
    assert reasons["L00002"] == "NOT_APPROVED"
    assert reasons["L00004"] == "CANCELLED"
    assert reasons["L00005"] == "NON_PROCUREMENT"      # intercompany text
    assert reasons["L99999"] == "DUPLICATE"            # exported twice, different formatting
    inc = out[out["include"] == 1]
    assert set(inc["l4"]) <= {"Domestic Hotels", "Domestic Courier", "Plates", "Data Plans"}
    # the two spellings of the hotel (same GSTIN) are one supplier
    keys = out.set_index("record_id")["supplier_key"]
    assert keys["L00000"] == keys["L00001"]
