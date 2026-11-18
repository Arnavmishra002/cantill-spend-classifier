"""Awkward file layouts an unseen company could send. Synthetic data only."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spendcube.config import ConfigError, load_config  # noqa: E402
from spendcube.ingest import drop_summary_rows, read_table  # noqa: E402
from spendcube.init_config import draft_config  # noqa: E402
from spendcube.pipeline import run  # noqa: E402
from spendcube.taxonomy import load_taxonomy  # noqa: E402
from spendcube.textnorm import normalize_currency  # noqa: E402

TREE = [("Facilities", "Cleaning", "Janitorial Services"), ("Facilities", "Maintenance", "HVAC Repair"),
        ("IT", "Hardware", "Laptops"), ("IT", "Hardware", "Monitors"), ("IT", "Software", "Office Software"),
        ("Services", "Legal", "External Counsel")]
ITEMS = ["daily office cleaning", "HVAC compressor repair", "Dell Latitude laptop", "27 inch monitor",
         "Microsoft 365 licence", "litigation fees"]
SUPPLIERS = ["Acme Cleaning Ltd", "Tailspin HVAC Co", "Northwind IT Inc", "Contoso Legal LLP"]


@pytest.fixture(autouse=True)
def no_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def lines(n=120, **extra) -> pd.DataFrame:
    df = pd.DataFrame({"Line ID": [f"C{i:05d}" for i in range(n)], "Supplier Name": [SUPPLIERS[i % 4] for i in range(n)],
                       "Line Description": [f"{ITEMS[i % 6]} batch {i}" for i in range(n)],
                       "Invoice Date": [f"2025-03-{1 + i % 27:02d}" for i in range(n)],
                       "Amount": [f"{100 + i}.00" for i in range(n)], "Status": ["Paid", "Approved"] * (n // 2)})
    for k, v in extra.items():
        df[k] = v
    return df


def tree() -> pd.DataFrame:
    return pd.DataFrame(TREE, columns=["category_l1", "category_l2", "category_l3"])


def draft(folder: Path, tree_name: str = "tree.csv") -> tuple[dict, str]:
    text = draft_config(str(folder), tree_name, str(folder / "c.yaml"), use_llm=False).read_text()
    return yaml.safe_load(text), text


def run_it(folder: Path) -> pd.DataFrame:
    run(str(folder / "c.yaml"), str(folder), str(folder / "out.csv"), use_llm=False, cache_dir=str(folder / "cache"))
    return pd.read_csv(folder / "out.csv", dtype=str, keep_default_na=False)


def test_title_rows_blank_rows_and_a_totals_row_are_not_records(tmp_path):
    df = lines()
    body = df.to_csv(index=False)
    total = f"Grand Total,,,,{df['Amount'].astype(float).sum():.2f},\n"
    (tmp_path / "export.csv").write_text("Accounts Payable Export,,,,,\nRun 2025-04-01,,,,,\n,,,,,\n" + body
                                         + ",,,,,\n,,,,,\n" + total)
    tree().to_csv(tmp_path / "tree.csv", index=False)
    cfg, _ = draft(tmp_path)
    assert cfg["sources"][0]["columns"]["amount"] == "Amount"
    out = run_it(tmp_path)
    assert len(out) == len(df) and set(out["record_id"]) == set(df["Line ID"])
    assert out["spend"].astype(float).sum() == pytest.approx(df["Amount"].astype(float).sum())


def test_only_a_real_totals_line_is_dropped():
    df = lines(10)
    df.loc[9, "Line Description"] = "Total station survey instrument"
    kept, n = drop_summary_rows(df, "Line ID")
    assert n == 0 and len(kept) == 10
    df.loc[10] = ["C99999", "TOTAL", "", "", "880.00", ""]        # a sparse record from a supplier called Total
    kept, n = drop_summary_rows(df, "Line ID")
    assert n == 0 and len(kept) == 11
    df.loc[11] = ["Total", "", "", "", "1234.00", ""]
    kept, n = drop_summary_rows(df, "Line ID")
    assert n == 1 and len(kept) == 11
    df.loc[11] = ["", "Grand Total:", "", "", "1,234.00", ""]
    kept, n = drop_summary_rows(df, "Line ID")
    assert n == 1 and len(kept) == 11


def test_empty_rows_in_a_workbook_are_dropped(tmp_path):
    df = lines(20)
    gap = pd.DataFrame([[""] * len(df.columns)] * 3, columns=df.columns)
    pd.concat([df.iloc[:10], gap, df.iloc[10:]]).to_excel(tmp_path / "x.xlsx", index=False)
    assert len(read_table(tmp_path / "x.xlsx")) == 20


def test_utf16_text_export_is_read(tmp_path):
    df = lines(30)
    (tmp_path / "lines.txt").write_bytes(df.to_csv(index=False, sep="\t").encode("utf-16"))
    got = read_table(tmp_path / "lines.txt")
    assert list(got.columns) == list(df.columns) and len(got) == 30
    (tmp_path / "nobom.txt").write_bytes(df.to_csv(index=False, sep="\t").encode("utf-16-le"))
    assert len(read_table(tmp_path / "nobom.txt")) == 30


def test_currency_names_and_symbols_become_codes():
    v = pd.Series(["usd", "US Dollar", "€", "Euro", "£", "$", "USD - US Dollar", "GBP", "", "Doubloons", "Rand", "RMB"])
    got = normalize_currency(v, "USD").tolist()
    assert got == ["USD", "USD", "EUR", "EUR", "GBP", "USD", "USD", "GBP", "", "DOUBLOONS", "ZAR", "CNY"]
    assert normalize_currency(pd.Series(["$"]), "EUR").tolist() == ["$"]          # no guess without a dollar base
    assert normalize_currency(pd.Series(["Doubloons"]), "USD", {"doubloons": "xcd"}).tolist() == ["XCD"]


def test_currency_names_convert_and_unknown_ones_are_flagged(tmp_path):
    cur = ["US Dollar", "Euro", "€", "GBP", "$", "Doubloons"] * 20
    lines(Currency=cur).to_csv(tmp_path / "lines.csv", index=False)
    tree().to_csv(tmp_path / "tree.csv", index=False)
    cfg, text = draft(tmp_path)
    assert cfg["sources"][0]["columns"]["currency"] == "Currency"
    assert "currency values read as codes" in text and "DOUBLOONS" in text and "WARNING" in text
    rep = run(str(tmp_path / "c.yaml"), str(tmp_path), str(tmp_path / "o.csv"), use_llm=False,
              cache_dir=str(tmp_path / "cache"))
    df = rep["_df"]
    assert set(df["currency_norm"]) == {"USD", "EUR", "GBP", "DOUBLOONS"}
    eur = df[df["currency_norm"] == "EUR"]
    assert (eur["spend_rc"] > eur["net_amount"]).all()                     # converted, not read as dollars
    assert df.loc[df["currency_norm"] == "DOUBLOONS", "flags"].str.contains("NO_FX_RATE").all()
    # a config alias fixes the unknown value without code
    c2 = yaml.safe_load((tmp_path / "c.yaml").read_text())
    c2["fx"] = {**(c2.get("fx") or {}), "currency_aliases": {"Doubloons": "USD"}}
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(c2))
    rep = run(str(tmp_path / "c.yaml"), str(tmp_path), str(tmp_path / "o.csv"), use_llm=False,
              cache_dir=str(tmp_path / "cache"))
    assert not rep["_df"]["flags"].str.contains("NO_FX_RATE").any()


def test_flag_columns_become_exclusion_rules(tmp_path):
    n = 140
    df = lines(n).drop(columns=["Status"])
    df["Approved"] = (["Y"] * 6 + ["N"]) * (n // 7)
    df["Cancelled Flag"] = ["Yes" if i % 10 == 0 else "No" for i in range(n)]
    df["Paid"] = ["true", "false"] * (n // 2)
    df.to_csv(tmp_path / "lines.csv", index=False)
    tree().to_csv(tmp_path / "tree.csv", index=False)
    cfg, text = draft(tmp_path)
    rules = {r["field"]: r for r in cfg["exclusion_rules"]}
    assert rules["Approved"]["code"] == "NOT_APPROVED" and rules["Approved"]["values"] == ["n"]
    assert rules["Cancelled Flag"]["code"] == "CANCELLED" and rules["Cancelled Flag"]["values"] == ["yes"]
    assert "Paid" not in rules and "paid flag" in text
    out = run_it(tmp_path).set_index("record_id")
    exp_cancel = {f"C{i:05d}" for i in range(n) if i % 10 == 0}
    exp_na = {f"C{i:05d}" for i in range(n) if i % 7 == 6} - exp_cancel
    assert set(out.index[out["exclusion_reason"] == "CANCELLED"]) == exp_cancel
    assert set(out.index[out["exclusion_reason"] == "NOT_APPROVED"]) == exp_na


def test_exclusion_rule_needs_a_valid_code(tmp_path):
    lines(20).to_csv(tmp_path / "lines.csv", index=False)
    tree().to_csv(tmp_path / "tree.csv", index=False)
    cfg, _ = draft(tmp_path)
    cfg["exclusion_rules"] = [{"name": "x", "field": "Status", "values": ["paid"], "code": "REMOVED"}]
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg))
    with pytest.raises(ConfigError, match="code"):
        load_config(str(tmp_path / "c.yaml"))


def test_tree_given_as_one_path_column(tmp_path):
    lines().to_csv(tmp_path / "lines.csv", index=False)
    pd.DataFrame({"Category Path": [" > ".join(t) for t in TREE]}).to_csv(tmp_path / "tree.csv", index=False)
    cfg, text = draft(tmp_path)
    assert cfg["taxonomy"]["path_separator"] == ">" and "whole paths" in text
    tax = load_taxonomy(load_config(str(tmp_path / "c.yaml")), tmp_path)
    assert tax.max_depth == 3 and {tax.nodes[n].path for n in tax.leaves} == set(TREE)
    out = run_it(tmp_path)
    assert set(out["l1"]) <= {"Facilities", "IT", "Services"} and set(out["l3"]) <= {t[2] for t in TREE}


def test_flat_category_list_with_slashes_is_not_split(tmp_path):
    lines().to_csv(tmp_path / "lines.csv", index=False)
    pd.DataFrame({"Category": ["HVAC/R", "IT/Telecom", "Legal/Compliance", "Travel/Entertainment", "Cleaning"]}
                 ).to_csv(tmp_path / "tree.csv", index=False)
    cfg, _ = draft(tmp_path)
    assert "path_separator" not in cfg["taxonomy"]
    assert load_taxonomy(load_config(str(tmp_path / "c.yaml")), tmp_path).max_depth == 1


def test_one_workbook_holding_the_tree_and_the_lines(tmp_path):
    with pd.ExcelWriter(tmp_path / "company_c.xlsx") as w:
        pd.DataFrame({"About": ["Company C blind test data"]}).to_excel(w, sheet_name="ReadMe", index=False)
        lines().to_excel(w, sheet_name="Invoice Lines", index=False)
        tree().to_excel(w, sheet_name="Taxonomy", index=False)
    cfg, text = draft(tmp_path, "company_c.xlsx")
    assert cfg["taxonomy"]["sheet"] == "Taxonomy" and cfg["sources"][0]["sheet"] == "Invoice Lines"
    out = run_it(tmp_path)
    assert len(out) == 120 and set(out["l1"]) - {""} <= {"Facilities", "IT", "Services"}


def test_stray_files_in_the_folder_are_not_sources(tmp_path):
    lines().to_csv(tmp_path / "c_invoice_lines.csv", index=False)
    tree().to_csv(tmp_path / "c_taxonomy.csv", index=False)
    (tmp_path / "README.txt").write_text("Company C blind test\nFiles: c_invoice_lines.csv, c_taxonomy.csv\n")
    pd.DataFrame({"Field": ["Line ID", "Amount"], "Description": ["unique id of the line", "amount in USD"]}
                 ).to_csv(tmp_path / "data_dictionary.csv", index=False)
    pd.DataFrame(columns=["record_id", "include", "exclusion_reason", "supplier_key", "item_key", "spend", "l1", "l2",
                          "l3", "l4", "confidence"]).to_csv(tmp_path / "output_template.csv", index=False)
    cfg, text = draft(tmp_path, "c_taxonomy.csv")
    assert [s["file"][0] for s in cfg["sources"]] == ["c_invoice_lines.csv"]
    assert "not a transaction file" in text and "TODO" not in yaml.safe_dump(cfg["sources"])
    assert len(run_it(tmp_path)) == 120


ODD_TREE = [("IT  Hardware", "End User ", "Laptops and Notebooks"), ("IT  Hardware", "End User ", "Monitors™"),
            ("Facilities", "Cleaning m²", " Janitorial"), ("Facilities", "Cleaning m²", "Window  Cleaning"),
            ("Services", "Legal – External", "Litigation")]


def _exact(folder: Path) -> set:
    out = run_it(folder)
    inc = out[out["include"] == "1"]
    assert len(inc) > 0
    return set(map(tuple, inc[["l1", "l2", "l3"]].values))


def test_category_names_are_written_exactly_as_in_a_wide_tree(tmp_path):
    lines().to_csv(tmp_path / "lines.csv", index=False)
    pd.DataFrame(ODD_TREE, columns=["category_l1", "category_l2", "category_l3"]).to_csv(tmp_path / "tree.csv", index=False)
    draft(tmp_path)
    assert _exact(tmp_path) <= set(ODD_TREE)             # double spaces, trailing spaces and symbols kept


def test_category_names_are_written_exactly_as_in_a_parent_child_tree(tmp_path):
    lines().to_csv(tmp_path / "lines.csv", index=False)
    rows, ids = [], {}
    for path in ODD_TREE:
        for d in range(1, 4):
            if path[:d] not in ids:
                ids[path[:d]] = f"K{len(ids) + 1:03d}"
                rows.append({"Category ID": ids[path[:d]], "Parent ID": ids.get(path[:d - 1], ""), "Category Name": path[d - 1]})
    pd.DataFrame(rows).to_csv(tmp_path / "tree.csv", index=False)
    cfg, _ = draft(tmp_path)
    assert cfg["taxonomy"]["format"] == "parent_child"
    assert _exact(tmp_path) <= set(ODD_TREE)


def test_category_names_from_a_path_column_keep_their_symbols(tmp_path):
    lines().to_csv(tmp_path / "lines.csv", index=False)
    pd.DataFrame({"Category Path": [" > ".join(t) for t in ODD_TREE]}).to_csv(tmp_path / "tree.csv", index=False)
    draft(tmp_path)
    assert _exact(tmp_path) <= {tuple(x.strip() for x in t) for t in ODD_TREE}


def _cols(folder: Path) -> tuple[dict, dict, str]:
    cfg, text = draft(folder)
    return cfg, cfg["sources"][0]["columns"], text


@pytest.mark.parametrize("order", [["VAT Amount", "Gross Amount"], ["Gross Amount", "VAT Amount"],
                                   ["Importe IVA", "Importe IVA incluido"], ["Montant TVA", "Montant TTC"]])
def test_amount_and_tax_amount_are_told_apart_in_any_column_order(tmp_path, order):
    df = lines().drop(columns=["Amount"])
    net = [100.0 + i for i in range(len(df))]
    tax_col, gross_col = sorted(order, key=lambda c: 0 if c in ("VAT Amount", "Importe IVA", "Montant TVA") else 1)
    vals = {tax_col: [f"{x * 0.2:.2f}" for x in net], gross_col: [f"{x * 1.2:.2f}" for x in net]}
    for c in order:                                  # the file's column order
        df[c] = vals[c]
    df.to_csv(tmp_path / "lines.csv", index=False)
    tree().to_csv(tmp_path / "tree.csv", index=False)
    cfg, cols, _ = _cols(tmp_path)
    assert cols["amount"] == gross_col and cols["tax_amount"] == tax_col and cfg["sources"][0]["amount_type"] == "gross"
    out = run_it(tmp_path)
    assert out["spend"].astype(float).sum() == pytest.approx(sum(net), rel=1e-4)       # tax removed


@pytest.mark.parametrize("name_col,id_col", [("Nombre del proveedor", "Código proveedor"), ("Nom du fournisseur", "Code fournisseur"),
                                             ("Kreditor Name", "Lieferantennummer"), ("Proveedor", "Código proveedor")])
def test_supplier_headers_in_other_languages(tmp_path, name_col, id_col):
    df = lines().rename(columns={"Supplier Name": name_col})
    df[id_col] = [f"V{1000 + i % 4}" for i in range(len(df))]
    df.to_csv(tmp_path / "lines.csv", index=False)
    tree().to_csv(tmp_path / "tree.csv", index=False)
    _, cols, text = _cols(tmp_path)
    assert cols["supplier_name"] == name_col and cols["supplier_id"] == id_col and "no supplier column" not in text


def test_no_supplier_column_is_a_warning(tmp_path):
    lines().drop(columns=["Supplier Name"]).to_csv(tmp_path / "lines.csv", index=False)
    tree().to_csv(tmp_path / "tree.csv", index=False)
    _, _, text = _cols(tmp_path)
    assert "WARNING" in text and "no supplier column found" in text


def test_status_column_with_case_and_space_variants_is_still_the_status(tmp_path):
    base = ["Paid", "Approved", "Posted", "Cancelled", "Rejected", "On Hold", "Disputed"]
    n = 210
    st = [[b.upper(), f" {b} ", b.lower()][i % 3] for i, b in enumerate(base * (n // 7))]     # 21 spellings
    lines(n, Status=st).to_csv(tmp_path / "lines.csv", index=False)
    tree().to_csv(tmp_path / "tree.csv", index=False)
    cfg, cols, _ = _cols(tmp_path)
    assert cols["status"] == "Status"
    out = run_it(tmp_path)
    assert out["exclusion_reason"].value_counts().to_dict() == {"": 90, "NOT_APPROVED": 90, "CANCELLED": 30}


def test_a_status_header_that_is_not_used_is_a_warning(tmp_path):
    lines(120, Status=[f"batch state {i}" for i in range(120)]).to_csv(tmp_path / "lines.csv", index=False)
    tree().to_csv(tmp_path / "tree.csv", index=False)
    _, cols, text = _cols(tmp_path)
    assert "status" not in cols and "look like a status but are not used" in text


def test_generic_tax_labels_are_non_procurement(tmp_path):
    df = lines()
    labels = ["GST Input Tax", "VAT payable", "Sales Tax", "Withholding tax", "Input VAT", "VAT payment Q1"]
    keep = ["GST consulting fees", "Tax advisory services", "Customs duty on imports"]
    for i, t in enumerate(labels + keep):
        df.loc[i, "Line Description"] = t
    df.to_csv(tmp_path / "lines.csv", index=False)
    tree().to_csv(tmp_path / "tree.csv", index=False)
    draft(tmp_path)
    out = run_it(tmp_path).set_index("record_id")
    assert (out.loc[[f"C{i:05d}" for i in range(len(labels))], "exclusion_reason"] == "NON_PROCUREMENT").all()
    assert (out.loc[[f"C{i:05d}" for i in range(len(labels), len(labels) + len(keep))], "exclusion_reason"] == "").all()


def test_a_totals_line_does_not_hide_a_flag_column(tmp_path):
    df = lines().drop(columns=["Status"])
    df.insert(0, "Approved", ["Y", "Y", "N"] * (len(df) // 3))
    body = df.to_csv(index=False)
    (tmp_path / "lines.csv").write_text(body + "Total" + "," * (len(df.columns) - 1) + "\n")
    tree().to_csv(tmp_path / "tree.csv", index=False)
    cfg, _, text = _cols(tmp_path)
    assert [r["field"] for r in cfg["exclusion_rules"]] == ["Approved"] and "totals line" in text
    out = run_it(tmp_path)
    assert len(out) == len(df) and (out["exclusion_reason"] == "NOT_APPROVED").sum() == len(df) // 3


def test_path_tree_with_few_branches_is_split(tmp_path):
    lines().to_csv(tmp_path / "lines.csv", index=False)
    paths = ["Fleet / Fuel / Diesel", "Fleet / Tyres / Truck Tyres", "IT / Tracking / Telematics", "Insurance / Motor / Fleet Cover",
             "Haulage / Third Party / Loads", "Warehouse / Equipment / Forklifts", "Warehouse / Consumables / Pallets"]
    pd.DataFrame({"Category": paths}).to_csv(tmp_path / "tree.csv", index=False)
    cfg, _ = draft(tmp_path)
    assert cfg["taxonomy"].get("path_separator") in ("/", " / ")
    assert load_taxonomy(load_config(str(tmp_path / "c.yaml")), tmp_path).max_depth == 3
