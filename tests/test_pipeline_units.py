"""Unit tests: cleaning, spend rules, matching, de-duplication, output validation.
All data here is synthetic; no task data is used or needed."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spendcube.config import DEFAULTS, _deep_merge  # noqa: E402
from spendcube.dedupe import dedupe_items, dedupe_suppliers, supplier_norm  # noqa: E402
from spendcube.export import OUTPUT_COLUMNS, build_output  # noqa: E402
from spendcube.match import match_sources, norm_invoice_number  # noqa: E402
from spendcube.spend_rules import apply_rules, resolve_codes  # noqa: E402
from spendcube.taxonomy import Taxonomy  # noqa: E402
from spendcube.textnorm import compact_key, parse_date, parse_number, parse_rate  # noqa: E402
from spendcube.validate import ValidationError, validate_output  # noqa: E402


# ------------------------------------------------------------------ cleaning
@pytest.mark.parametrize("raw,val", [
    ("1,234.56", 1234.56), ("1.234,56", 1234.56), ("1,00,000", 100000.0), ("12,5", 12.5),
    ("R 1 500,00", 1500.0), ("$2,000", 2000.0), ("(123.45)", -123.45), ("123.45-", -123.45),
    ("-116,020.33", -116020.33), ("₹ 2,50,000.75", 250000.75), ("", np.nan), ("abc", np.nan),
])
def test_parse_number(raw, val):
    got = parse_number(raw)
    if np.isnan(val):
        assert np.isnan(got)
    else:
        assert got == pytest.approx(val)


@pytest.mark.parametrize("raw,val", [("15", 0.15), ("15%", 0.15), ("0.15", 0.15), ("0", 0.0)])
def test_parse_rate(raw, val):
    assert parse_rate(raw) == pytest.approx(val)


def test_parse_rate_blank_default():
    assert parse_rate("", default=0.15) == 0.15


@pytest.mark.parametrize("raw,iso", [
    ("2024-03-29", "2024-03-29"), ("29/03/2024", "2024-03-29"), ("08-May-24", "2024-05-08"),
    ("May 8, 2024", "2024-05-08"), ("20240131", "2024-01-31"), ("45000", "2023-03-15"),
])
def test_parse_date(raw, iso):
    assert str(parse_date(raw)) == iso


def test_parse_date_garbage():
    assert parse_date("31/31/2024") is None
    assert parse_date("not a date") is None


def test_ocr_compact_key():
    assert compact_key("KEL MONT (PTY) LTD") == compact_key("KELMONT (PTY) LTD")
    assert compact_key("C0NVEYOR") == compact_key("CONVEYOR")
    assert compact_key("4510047148") == "4510047148"  # numbers untouched


# -------------------------------------------------------------- spend rules
def _cfg(**kw):
    base = _deep_merge(DEFAULTS, {"sources": [{"name": "s", "file": "x", "record_id": "id",
                                               "columns": {"amount": "a", "status": "st"}}]})
    return _deep_merge(base, kw)


def _frame(rows):
    df = pd.DataFrame(rows)
    df["source"] = "s"
    df["row_order"] = range(len(df))
    for c in ("description", "supplier_name", "supplier_id", "item_code", "status"):
        if c not in df:
            df[c] = ""
    df["spend_rc"] = df.get("spend_rc", 100.0)
    df["cand_codes"] = [[] for _ in range(len(df))]
    return df


def test_status_and_precedence():
    cfg = _cfg(non_procurement=[{"name": "tax", "field": "description", "pattern": "(?i)^sales tax"}])
    df = _frame([
        {"record_id": "1", "status": "Paid", "description": "valve", "raw__x": "a"},
        {"record_id": "2", "status": "Cancelled", "description": "Sales tax", "raw__x": "b"},
        {"record_id": "3", "status": "On Hold", "description": "pump", "raw__x": "c"},
        {"record_id": "4", "status": "Approved", "description": "Sales tax expense", "raw__x": "d"},
    ])
    apply_rules(df, cfg)
    resolve_codes(df, cfg)
    assert df["exclusion_reason"].tolist() == ["", "CANCELLED", "NOT_APPROVED", "NON_PROCUREMENT"]
    assert df["include"].tolist() == [1, 0, 0, 0]
    assert df.loc[df["include"] == 0, "spend"].eq(0).all()


def test_within_source_exact_duplicates():
    cfg = _cfg()
    df = _frame([{"record_id": "1", "raw__id": "1", "raw__amt": "5", "raw__d": "x"},
                 {"record_id": "2", "raw__id": "2", "raw__amt": "5", "raw__d": "x"},
                 {"record_id": "3", "raw__id": "3", "raw__amt": "6", "raw__d": "x"}])
    apply_rules(df, cfg)
    resolve_codes(df, cfg)
    assert df["exclusion_reason"].tolist() == ["", "DUPLICATE", ""]


# ----------------------------------------------------------------- matching
def test_cross_source_matching_and_exported_twice():
    cfg = _deep_merge(DEFAULTS, {"cross_source": {
        "enabled": True, "primary": "erp", "secondary": "tool", "po_pattern": r"(\d{10})",
        "invoice_prefixes": ["INV/", "SI-"]}})
    rows = [
        # primary
        dict(record_id="ERP:1", source="erp", po_number="4510000001", net_amount=100.0, quantity_num=1.0,
             description="VALVE BALL 2IN", supplier_name="ACME (PTY) LTD", date_parsed=pd.NaT),
        dict(record_id="ERP:2", source="erp", po_number="4510000002", net_amount=50.0, quantity_num=2.0,
             description="ROOF BOLT 1.8M", supplier_name="BOLTCO", date_parsed=pd.NaT),
        # same purchase, PO reformatted -> duplicate of ERP:1
        dict(record_id="TOOL:a", source="tool", po_reference="PO#4510000001", invoice_number="INV/77",
             net_amount=100.0, quantity_num=1.0, description="VALVE BALL 2IN", supplier_name="Acme",
             date_parsed=pd.NaT),
        # same invoice exported twice (different formatting, no PO) -> duplicate too
        dict(record_id="TOOL:b", source="tool", po_reference="", invoice_number="77",
             net_amount=100.0, quantity_num=1.0, description="VALVE BALL 2lN", supplier_name="ACME",
             date_parsed=pd.NaT),
        # tool-only purchase -> counted
        dict(record_id="TOOL:c", source="tool", po_reference="4519999999", invoice_number="SI-5",
             net_amount=70.0, quantity_num=1.0, description="CABLE GLAND", supplier_name="OTHER",
             date_parsed=pd.NaT),
    ]
    df = pd.DataFrame(rows).fillna("")
    df["row_order"] = range(len(df))
    df["cand_codes"] = [[] for _ in range(len(df))]
    df["spend_rc"] = df["net_amount"]
    match_sources(df, cfg)
    resolve_codes(df, cfg)
    got = dict(zip(df["record_id"], df["exclusion_reason"]))
    assert got["ERP:1"] == "" and got["ERP:2"] == ""
    assert got["TOOL:a"] == "DUPLICATE" and got["TOOL:b"] == "DUPLICATE"
    assert got["TOOL:c"] == ""
    assert df["spend"].sum() == pytest.approx(220.0)


def test_norm_invoice_number():
    p = ["TAX INV", "INV/", "SI-", "IN"]
    assert norm_invoice_number("INV/2023/759857", p) == norm_invoice_number("2023/759857", p)
    assert norm_invoice_number("TAX INV 000123", p) == "123"


# ------------------------------------------------------------------ dedupe
def test_supplier_norm_suffixes():
    assert supplier_norm("Acme Engineering (Pty) Ltd") == supplier_norm("ACME ENGINEERING PTY LIMITED")
    assert supplier_norm("Tata & Sons Pvt. Ltd.") == supplier_norm("TATA AND SONS PRIVATE LIMITED")
    assert supplier_norm("Mining SA (PTY) LTD") == "MINING"


def test_dedupe_suppliers_groups_variants_not_different_firms():
    df = pd.DataFrame({
        "source": "s",
        "supplier_name": ["KELMONT (PTY) LTD", "KEL MONT PTY LTD", "Kelmont", "KELMORE HYDRAULICS",
                          "PRAHAVEN (PTY) LTD", "PRAH AVEN"],
        "supplier_id": ["", "", "", "", "", ""],
    })
    k = dedupe_suppliers(df, DEFAULTS)
    assert k[0] == k[1] == k[2]
    assert k[3] != k[0]
    assert k[4] == k[5] and k[4] != k[0]


def test_dedupe_suppliers_tax_id_and_vendor_id():
    df = pd.DataFrame({"source": "s", "supplier_name": ["Alpha Ltd", "Totally Different Name", "Beta"],
                       "supplier_id": ["V1", "V1", "V2"], "supplier_tax_id": ["", "", ""]})
    k = dedupe_suppliers(df, DEFAULTS)
    assert k[0] == k[1] != k[2]


def test_dedupe_items_code_and_text():
    df = pd.DataFrame({
        "record_id": list("abcde"),
        "item_code": ["M1", "M2", "", "", "M3"],
        "description": ["REPR;GEARBOX;P/N:0423-RH", "REPR;GEARBOX;P/N:0423-RH", "VALVE BALL 2 IN",
                        "VALVE BALL 2IN", "VALVE BALL 3IN"],
    })
    k = dedupe_items(df, DEFAULTS, pd.Series(["S"] * 5))
    assert k[0] == k[1]          # different codes, same text
    assert k[2] == k[3]          # unit spacing
    assert k[3] != k[4]          # different size stays apart


# --------------------------------------------------------------- validation
def _tax():
    return Taxonomy.from_paths([("A", "A1", "A1x"), ("A", "A2", "A2x"), ("B", "B1", "B1x")])


def _out_df(tax):
    leaf = tax.leaves[0]
    df = pd.DataFrame({"record_id": ["r1", "r2"], "include": [1, 0], "exclusion_reason": ["", "DUPLICATE"],
                       "supplier_key": ["s", "s"], "item_key": ["i", "i"], "spend": [10.0, 0.0],
                       "spend_rc": [10.0, 10.0], "node": [leaf, None], "confidence": [0.9, None],
                       "exclusion_confidence": [0.0, 0.95]})
    return df


def test_validation_passes_and_matches_template():
    tax = _tax()
    df = _out_df(tax)
    out = build_output(df, tax, OUTPUT_COLUMNS)
    rep = validate_output(out, df, tax, OUTPUT_COLUMNS, 2)
    assert list(out.columns) == OUTPUT_COLUMNS
    assert rep["included_spend"] == 10.0 and rep["excluded_spend"] == 10.0
    assert out.loc[1, "l1"] == "" and out.loc[0, "l4"] == ""


def test_validation_catches_bad_category_and_duplicate_ids():
    tax = _tax()
    df = _out_df(tax)
    out = build_output(df, tax, OUTPUT_COLUMNS)
    bad = out.copy()
    bad.loc[0, "l2"] = "Not In Tree"
    with pytest.raises(ValidationError):
        validate_output(bad, df, tax, OUTPUT_COLUMNS, 2)
    dup = out.copy()
    dup.loc[1, "record_id"] = "r1"
    with pytest.raises(ValidationError):
        validate_output(dup, df, tax, OUTPUT_COLUMNS, 2)
    wrong = out.copy()
    wrong.loc[1, "spend"] = 5.0
    with pytest.raises(ValidationError):
        validate_output(wrong, df, tax, OUTPUT_COLUMNS, 2)


# ------------------------------------------------------------------ reading
def test_read_table_title_rows_and_semicolons(tmp_path):
    from spendcube.ingest import read_table
    p = tmp_path / "x.csv"
    p.write_text('AP export\nGenerated 2025-01-01\nLine ID,Vendor,Amount\n1,A,"1,000.50"\n2,B,20\n')
    t = read_table(p)
    assert list(t.columns) == ["Line ID", "Vendor", "Amount"] and len(t) == 2
    q = tmp_path / "y.csv"
    q.write_text("Line ID;Vendor;Amount\n1;A;1.000,50\n2;B;20\n")
    t = read_table(q)
    assert list(t.columns) == ["Line ID", "Vendor", "Amount"]
    assert parse_number(t.loc[0, "Amount"]) == pytest.approx(1000.5)


def test_taxonomy_parent_child_and_depth():
    tax = Taxonomy.from_paths([("A", "B", "C", "D"), ("A", "B", "E")])
    assert tax.max_depth == 4
    assert {tax.nodes[n].name for n in tax.leaves} == {"D", "E"}
    assert tax.node_for_labels(["A", "B", "E", ""]) == tax.by_path[("A", "B", "E")]


def test_supplier_hubs_do_not_chain_suppliers():
    # a generic name used with many ids, and one id used with many unrelated names
    firms = ["Alpha Engineering", "Beta Hydraulics", "Gamma Mining Supplies", "Delta Electrical",
             "Epsilon Filters", "Zeta Bearings"]
    rows = []
    for i, f in enumerate(firms):
        rows.append({"supplier_name": f, "supplier_id": f"V{i}"})
        rows.append({"supplier_name": "SUNDRY VENDOR", "supplier_id": f"V{i}"})   # hub name
    for f in ["Tax Office", "Water Board", "Municipality Rates", "Courier Express", "Cash Payee"]:
        rows.append({"supplier_name": f, "supplier_id": "V999"})                   # hub id
    df = pd.DataFrame(rows)
    df["source"] = "s"
    k = dedupe_suppliers(df, DEFAULTS)
    named = k[df["supplier_name"].isin(firms)]
    assert named.nunique() == len(firms)                      # not chained into one key
    # each sundry line joins the real firm that owns its id
    for i, f in enumerate(firms):
        same_id = k[df["supplier_id"] == f"V{i}"]
        assert same_id.nunique() == 1
    hub_id_rows = k[df["supplier_id"] == "V999"]
    assert hub_id_rows.nunique() == 5                         # unrelated names stay apart
    assert k.attrs["hub_stats"] == {"hub_names": 1, "hub_ids": 1, "hub_tax_ids": 0}


def test_supplier_with_few_ids_still_merges():
    df = pd.DataFrame({"source": "s", "supplier_name": ["Acme Ltd", "ACME LIMITED", "Acme Ltd"],
                       "supplier_id": ["V1", "V2", "V3"]})
    k = dedupe_suppliers(df, DEFAULTS)
    assert k.nunique() == 1


def test_general_leaf_detection():
    tax = Taxonomy.from_paths([("Eq", "Pumps", "Pumps"), ("Eq", "Pumps", "Pump Spares"),
                               ("Eq", "Fans", "Fan Spares"), ("Eq", "Fans", "Other Fans"),
                               ("Eq", "Belts", "Belts")])
    gen = {tax.nodes[n].path for n in tax.leaves if tax.nodes[n].general}
    assert gen == {("Eq", "Pumps", "Pumps"), ("Eq", "Fans", "Other Fans")}   # single-leaf branch is not
    spares = tax.by_path[("Eq", "Pumps", "Pump Spares")]
    assert tax.general_sibling(spares) == tax.by_path[("Eq", "Pumps", "Pumps")]
    assert tax.display_path(tax.by_path[("Eq", "Pumps", "Pumps")]).endswith("[general]")
