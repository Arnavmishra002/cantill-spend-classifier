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


def test_supplier_id_synonyms_dayfirst_and_unique_lists(tmp_path):
    from spendcube.init_config import guess_dayfirst
    assert guess_dayfirst(pd.Series(["03/25/2024", "04/01/2024"])) is False   # month first
    assert guess_dayfirst(pd.Series(["25/03/2024", "01/04/2024"])) is True    # day first
    assert guess_dayfirst(pd.Series(["01/02/2024", "2024-03-05"])) is None    # undecidable
    for header in ["Supplier No.", "Vendor ID", "Vendor #", "Vendor Code"]:
        d = tmp_path / header.replace(" ", "_").replace("#", "n").replace(".", "")
        d.mkdir()
        n = 40
        pd.DataFrame({
            "Line ID": [f"X{i}" for i in range(n)], header: [f"V{1000 + i % 7}" for i in range(n)],
            "Supplier Name": [["Acme Ltd", "Beta Corp", "Gamma Inc"][i % 3] for i in range(n)],
            "Invoice Date": [f"{(i % 12) + 1:02d}/{(i % 27) + 2:02d}/2025" for i in range(n)],
            "Description": ["steel plate 10mm"] * n, "Amount": ["12.50"] * n,
            "Status": ["Paid", "paid ", "Approved", "Paid"] * (n // 4),
        }).to_csv(d / "lines.csv", index=False)
        pd.DataFrame({"L1": ["A"], "L2": ["B"]}).to_csv(d / "tree.csv", index=False)
        cfg = yaml.safe_load(draft_config(str(d), "tree.csv", str(d / "c.yaml"), use_llm=False).read_text())
        src = cfg["sources"][0]
        assert src["columns"].get("supplier_id") == header, (header, src["columns"])
        assert src["columns"].get("supplier_name") == "Supplier Name"
        assert src["dayfirst"] is False
        assert len(src["file"]) == len(set(src["file"]))
        inc = src["status_map"]["include"]
        assert len(inc) == len(set(inc)) and "paid" in inc


def test_real_style_file_joins_text_maps_dates_and_window(tmp_path):
    n = 60
    pd.DataFrame({
        "Line ID": [f"C{i}" for i in range(n)], "Vendor Name": ["Acme Corp", "Beta LLC", "Gamma Inc"] * 20,
        "Item Name": ["Laptop computer", "Toner cartridge", "Office chair"] * 20,
        "Item Description": ["14 inch, 16GB", "black, high yield", "ergonomic mesh"] * 20,
        "Creation Date": [f"0{(i % 9) + 1}/2{(i % 8) + 1}/2013" for i in range(n)],
        "Purchase Date": [f"0{(i % 9) + 1}/1{(i % 9) + 1}/2013" for i in range(n)],
        "Total Price": ["$1,200.00"] * n,
    }).to_csv(tmp_path / "po.csv", index=False)
    pd.DataFrame({"Level 1": ["IT", "Office"], "Level 2": ["Hardware", "Furniture"]}).to_csv(tmp_path / "t.csv", index=False)
    cfg = yaml.safe_load(draft_config(str(tmp_path), "t.csv", str(tmp_path / "c.yaml"), use_llm=False).read_text())
    src = cfg["sources"][0]
    assert src["columns"]["description"] == ["Item Name", "Item Description"]
    assert src["columns"]["date"] == "Purchase Date" and src["columns"]["posting_date"] == "Creation Date"
    assert src["dayfirst"] is False and cfg["dates"]["dayfirst"] is False
    lo, hi = cfg["dates"]["valid_window_years"]
    assert lo <= 2012 and 2014 <= hi <= 2015


def test_small_invoice_file_with_currency_rate_and_date_is_not_an_fx_table(tmp_path):
    n = 3000
    pd.DataFrame({
        "Line ID": [f"L{i}" for i in range(n)], "Supplier": ["Acme Ltd", "Beta GmbH"] * (n // 2),
        "Description": ["valve 2in", "pump seal"] * (n // 2), "Invoice Date": ["2025-03-01"] * n,
        "Currency": ["EUR", "USD"] * (n // 2), "FX Rate": ["0.92", "1"] * (n // 2), "Amount": ["100.00"] * n,
        "Operator": ["x"] * n, "Period": ["2025-03"] * n,
    }).to_csv(tmp_path / "invoice_lines.csv", index=False)
    pd.DataFrame({"L1": ["Equipment"], "L2": ["Valves"]}).to_csv(tmp_path / "tree.csv", index=False)
    for files in (None, ["invoice_lines.csv"]):
        cfg = yaml.safe_load(draft_config(str(tmp_path), "tree.csv", str(tmp_path / "c.yaml"), use_llm=False,
                                          files=files).read_text())
        cols = cfg["sources"][0]["columns"]
        assert cols["amount"] == "Amount" and cols["currency"] == "Currency" and cols["fx_rate"] == "FX Rate"
        assert "table" not in (cfg.get("fx") or {})


def test_narrow_fx_table_next_to_invoices_is_used_as_fx(tmp_path):
    pd.DataFrame({"Line ID": ["A", "B"] * 50, "Supplier": ["X", "Y"] * 50, "Description": ["bolt", "nut"] * 50,
                  "Invoice Date": ["2025-01-02"] * 100, "Currency": ["EUR"] * 100, "Amount": ["5"] * 100}
                 ).assign(**{"Line ID": [f"L{i}" for i in range(100)]}).to_csv(tmp_path / "lines.csv", index=False)
    pd.DataFrame({"currency": ["EUR", "GBP"], "month": ["2025-01", "2025-01"], "units_per_usd": ["0.95", "0.8"]}
                 ).to_csv(tmp_path / "fx.csv", index=False)
    pd.DataFrame({"L1": ["A"], "L2": ["B"]}).to_csv(tmp_path / "tree.csv", index=False)
    cfg = yaml.safe_load(draft_config(str(tmp_path), "tree.csv", str(tmp_path / "c.yaml"), use_llm=False).read_text())
    assert cfg["fx"]["table"] == "fx.csv" and len(cfg["sources"]) == 1


def test_only_data_file_is_used_even_if_it_looks_like_labels(tmp_path):
    pd.DataFrame({"Line ID": [f"L{i}" for i in range(50)], "Vendor": ["Acme"] * 50, "Seed note": ["x"] * 50,
                  "Item": ["gloves"] * 50, "Date": ["2025-01-05"] * 50, "Amount": ["12.5"] * 50}
                 ).to_csv(tmp_path / "lines.csv", index=False)
    pd.DataFrame({"L1": ["A"], "L2": ["B"]}).to_csv(tmp_path / "tree.csv", index=False)
    cfg = yaml.safe_load(draft_config(str(tmp_path), "tree.csv", str(tmp_path / "c.yaml"), use_llm=False).read_text())
    assert cfg["sources"][0]["columns"]["amount"] == "Amount"
