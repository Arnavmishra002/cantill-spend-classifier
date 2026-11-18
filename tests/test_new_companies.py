"""init-config and the run on two awkward synthetic companies (tests/synthetic_data.py)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from synthetic_data import D_TREE, E_STATUS, E_TREE, make_company_d, make_company_e  # noqa: E402

from spendcube.check import check_config  # noqa: E402
from spendcube.dedupe import supplier_norm  # noqa: E402
from spendcube.ingest import ConfigColumnError  # noqa: E402
from spendcube.init_config import _status_map, _taxonomy_block, draft_config  # noqa: E402
from spendcube.pipeline import run  # noqa: E402


@pytest.fixture(autouse=True)
def no_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def _draft(folder: Path, tree: str) -> tuple[dict, str]:
    path = draft_config(str(folder), tree, str(folder / "c.yaml"), use_llm=False)
    text = path.read_text()
    return yaml.safe_load(text), text


def _run(folder: Path):
    rep = run(str(folder / "c.yaml"), str(folder), str(folder / "out.csv"), use_llm=False,
              cache_dir=str(folder / "cache"))
    return rep, pd.read_csv(folder / "out.csv", dtype=str, keep_default_na=False)


def test_status_words_in_seven_languages():
    vals = ["Received", "Invoiced", "Processed", "Posted", "Settled", "Released", "Matched", "Bezahlt",
            "Genehmigt", "Payé", "Pagado", "Aprobado", "Pagato", "Aprovado", "Betaald", "Storniert", "Annulé",
            "Cancelado", "Abgelehnt", "Rechazado", "En attente", "In behandeling", "Not approved", "Unpaid"]
    smap, unknown, _ = _status_map(vals)
    assert set(smap["include"]) == {v.lower() for v in vals[:15]}
    assert set(smap["CANCELLED"]) == {"storniert", "annulé", "cancelado"}
    assert set(smap["NOT_APPROVED"]) == {"abgelehnt", "rechazado", "en attente", "in behandeling", "not approved"}
    assert unknown == ["unpaid"]          # never read as "paid"


def test_claude_answer_fills_values_the_keywords_do_not_know():
    smap, unknown, notes = _status_map(["Zugestellt", "Paid"], llm={"Zugestellt": "include", "Paid": "NOT_APPROVED"})
    assert smap["include"] == ["zugestellt", "paid"] and not unknown
    assert notes and "kept include" in notes[0]


def test_company_d_received_status_and_name_tree(tmp_path):
    folder = make_company_d(tmp_path / "D")
    df = pd.read_csv(folder / "orders.csv")
    df.loc[::7, "status"] = "Invoiced"          # two values, so status is mapped as a column
    df.to_csv(folder / "orders.csv", index=False)
    cfg, text = _draft(folder, "unspsc_tree.csv")
    assert cfg["taxonomy"]["level_columns"] == ["Segment", "Family", "Class"]
    src = cfg["sources"][0]
    assert src["columns"]["amount"] == "line_total" and src["columns"]["date"] == "order_date"
    assert set(src["status_map"]["include"]) == {"received", "invoiced"}
    assert "WARNING" not in text
    rep, out = _run(folder)
    assert (out["include"] == "1").all() and not rep["warnings"]
    assert set(out["l3"]) <= {t[2] for t in D_TREE}


def test_company_e_codes_gross_amounts_languages_and_dates(tmp_path):
    folder = make_company_e(tmp_path / "E")
    cfg, text = _draft(folder, "kategorien.csv")
    assert cfg["taxonomy"]["level_columns"] == ["L1 Name", "L2 Name", "L3 Name"]
    src = cfg["sources"][0]
    assert src["columns"]["amount"] == "Bruttobetrag" and src["columns"]["tax_amount"] == "Steuerbetrag"
    assert src["amount_type"] == "gross" and src["columns"]["currency"] == "Währung"
    expect = {s.lower(): o for s, o in E_STATUS}
    for outcome in ("include", "CANCELLED", "NOT_APPROVED"):
        assert set(src["status_map"][outcome]) == {s for s, o in expect.items() if o == outcome}
    assert "WARNING" in text                    # 3 of 8 statuses excluded: 38% of lines
    rep, out = _run(folder)
    raw = pd.read_csv(folder / "rechnungen_export.csv", sep=";", encoding="cp1252", dtype=str)
    inc = out[out["include"] == "1"]
    assert set(inc["l1"]) <= {t[1] for t in E_TREE} and set(inc["l3"]) <= {t[5] for t in E_TREE}
    neg = raw["Bruttobetrag"].str.endswith("-")
    assert (out.loc[neg.values & (out["include"] == "1").values, "spend"].astype(float) < 0).all()
    audit = rep["_df"]
    assert audit["date_parsed"].notna().all()   # German, ISO and Excel serial dates
    gross = audit["amount_num"].abs()
    assert (audit["net_amount"].abs() < gross).all()          # VAT removed
    assert rep["warnings"] and "38%" in rep["warnings"][0]
    assert "WARNING" in check_config(str(folder / "c.yaml"), str(folder))


def test_tree_code_and_title_pairs_keep_names(tmp_path):
    pd.DataFrame({"Segment Code": ["43000000"], "Segment Title": ["IT Equipment"], "Family Code": ["43210000"],
                  "Family Title": ["Computers"], "Class Code": ["43211500"], "Class Title": ["Notebooks"]}
                 ).to_csv(tmp_path / "t.csv", index=False)
    block, _ = _taxonomy_block(tmp_path / "t.csv")
    assert block["level_columns"] == ["Segment Title", "Family Title", "Class Title"]


def test_amount_with_placeholders_is_still_mapped(tmp_path):
    n = 200
    vals = [f"{10 + i}.50" for i in range(n)]
    for i in range(0, n, 8):
        vals[i] = "-" if i % 16 else "N/A"      # about 13% placeholders
    pd.DataFrame({"Line ID": [f"L{i}" for i in range(n)], "Vendor Name": ["Acme Inc", "Beta LLC"] * (n // 2),
                  "Item Description": ["copy paper", "toner"] * (n // 2), "Invoice Date": ["2025-01-10"] * n,
                  "Line Amount": vals}).to_csv(tmp_path / "lines.csv", index=False)
    pd.DataFrame({"L1": ["Office"], "L2": ["Paper"]}).to_csv(tmp_path / "tree.csv", index=False)
    cfg, text = _draft(tmp_path, "tree.csv")
    assert cfg["sources"][0]["columns"]["amount"] == "Line Amount" and "placeholders" in text


def test_no_amount_column_is_a_todo_and_a_clear_error(tmp_path):
    pd.DataFrame({"Line ID": [f"L{i}" for i in range(50)], "Vendor Name": ["Acme Inc"] * 50,
                  "Item Description": ["copy paper"] * 50, "Invoice Date": ["2025-01-10"] * 50}
                 ).to_csv(tmp_path / "lines.csv", index=False)
    pd.DataFrame({"L1": ["Office"], "L2": ["Paper"]}).to_csv(tmp_path / "tree.csv", index=False)
    cfg, text = _draft(tmp_path, "tree.csv")
    assert cfg["sources"][0]["columns"]["amount"] == "TODO" and "WARNING" in text
    with pytest.raises(ConfigColumnError):
        _run(tmp_path)


@pytest.mark.parametrize("gross", [True, False])
def test_neutral_amount_header_is_judged_from_tax_values(tmp_path, gross):
    n = 60
    net = [100.0 + 7 * i for i in range(n)]
    tax = [round(x * 0.2, 2) for x in net]
    amt = [round(a + t, 2) for a, t in zip(net, tax)] if gross else net
    pd.DataFrame({"Line ID": [f"L{i}" for i in range(n)], "Vendor Name": ["Acme Inc"] * n,
                  "Item Description": ["toner"] * n, "Invoice Date": ["2025-01-10"] * n,
                  "Amount": amt, "VAT Amount": tax}).to_csv(tmp_path / "lines.csv", index=False)
    pd.DataFrame({"L1": ["Office"], "L2": ["Paper"]}).to_csv(tmp_path / "tree.csv", index=False)
    cfg, _ = _draft(tmp_path, "tree.csv")
    assert cfg["sources"][0]["amount_type"] == ("gross" if gross else "net")


def test_quebec_and_spaced_legal_suffixes():
    for a, b in [("Bombardier Ltée", "Bombardier"), ("Groupe ABC S.E.N.C.", "Groupe ABC"),
                 ("Acme Limitée", "Acme Inc."), ("Informática Ibérica S.L.", "Informatica Iberica"),
                 ("Acme S.A. de C.V.", "Acme SA")]:
        assert supplier_norm(a) == supplier_norm(b)


def test_a_status_column_without_any_status_word_is_not_used(tmp_path):
    n = 60
    pd.DataFrame({"Line ID": [f"L{i}" for i in range(n)], "Vendor Name": ["Acme Inc", "Beta LLC"] * (n // 2),
                  "Item Description": ["toner", "paper"] * (n // 2), "Invoice Date": ["2025-01-10"] * n,
                  "Amount": ["10.00"] * n, "Contract Status": ["On Contract", "Non Contract", "On Contract"] * (n // 3)}
                 ).to_csv(tmp_path / "lines.csv", index=False)
    pd.DataFrame({"L1": ["Office"], "L2": ["Paper"]}).to_csv(tmp_path / "tree.csv", index=False)
    cfg, text = _draft(tmp_path, "tree.csv")
    assert "status" not in cfg["sources"][0]["columns"] and "not used as status" in text and "WARNING" not in text


def test_unrelated_values_from_claude_are_included():
    smap, unknown, notes = _status_map(["On Contract", "Paid"], llm={"On Contract": "unrelated", "Paid": "include"})
    assert set(smap["include"]) == {"on contract", "paid"} and not unknown


def test_vague_and_reference_only_text():
    from spendcube.classify import is_vague
    from spendcube.textnorm import norm_key
    for t in ["INV-123", "Misc", "Product", "as per quote", "PO 4500012345", "Q1 2024", "Invoice March 2025", ""]:
        assert is_vague(norm_key(t)), t
    for t in ["Monthly rent March", "A4 copier paper 80gsm", "Consulting fees", "Freight charges", "Dell 24in LED"]:
        assert not is_vague(norm_key(t)), t


def test_vague_lines_get_low_confidence_and_no_specific_guess(tmp_path):
    tree = pd.DataFrame({"L1": ["Office", "Office", "IT", "Other"], "L2": ["Stationery", "Paper", "Laptops", "Other"]})
    tree.to_csv(tmp_path / "tree.csv", index=False)
    rows = [("L1", "Staples Inc", "ballpoint pens blue"), ("L2", "Staples Inc", "stapler and staples"),
            ("L3", "Staples Inc", "highlighter pens"), ("L4", "Staples Inc", "INV-123"),
            ("L5", "", "Misc"), ("L6", "", "as per quote"), ("L7", "", "Product"),
            ("L8", "Dell", "Latitude laptop 16GB")]
    pd.DataFrame(rows, columns=["id", "vendor", "text"]).assign(amt="100.00").to_csv(tmp_path / "lines.csv", index=False)
    cfg = {"company": "V", "taxonomy": {"file": "tree.csv", "level_columns": ["L1", "L2"]},
           "sources": [{"name": "s", "file": "lines.csv", "record_id": {"column": "id"},
                        "columns": {"supplier_name": "vendor", "description": "text", "amount": "amt"}}]}
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg))
    rep, out = _run(tmp_path)
    o = out.set_index("record_id")
    for rid in ("L5", "L6", "L7"):                       # no text, no supplier: the catch-all leaf
        assert o.loc[rid, "l1"] == "Other" and float(o.loc[rid, "confidence"]) <= 0.3
    assert o.loc["L4", "l2"] == "Stationery"             # the supplier's other lines decide
    rv = rep["review"]
    assert rv["lines"] >= 3 and 0 < rv["spend_share"] <= 1 and rv["coverage_l2_spend"] == 1.0


def test_workbook_with_a_cover_sheet_reads_the_lines_sheet(tmp_path):
    n = 120
    lines = pd.DataFrame({"Line ID": [f"L{i}" for i in range(n)], "Supplier Name": ["Acme Inc", "Beta LLC"] * (n // 2),
                          "Description": ["copy paper", "toner"] * (n // 2), "Invoice Date": ["2025-01-10"] * n,
                          "Amount": ["12.50"] * n, "Status": ["Paid", "Approved"] * (n // 2)})
    with pd.ExcelWriter(tmp_path / "export.xlsx") as w:
        pd.DataFrame({"Report": ["Accounts payable export", "Generated 2025-02-01"]}).to_excel(w, sheet_name="Cover",
                                                                                             index=False)
        lines.to_excel(w, sheet_name="Lines", index=False)
    pd.DataFrame({"L1": ["Office"], "L2": ["Paper"]}).to_csv(tmp_path / "tree.csv", index=False)
    cfg, text = _draft(tmp_path, "tree.csv")
    src = cfg["sources"][0]
    assert src["sheet"] == "Lines" and src["columns"]["amount"] == "Amount" and "CHECK sheet" in text
    rep, out = _run(tmp_path)
    assert len(out) == n and set(out["exclusion_reason"]) <= {"", "DUPLICATE"}
    # without the sheet the error names the sheets
    cfg2 = yaml.safe_load((tmp_path / "c.yaml").read_text())
    cfg2["sources"][0].pop("sheet")
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg2))
    with pytest.raises(ConfigColumnError, match="Lines"):
        _run(tmp_path)


def test_profile_script_runs_on_a_drafted_config(tmp_path, capsys, monkeypatch):
    import importlib.util
    folder = make_company_e(tmp_path / "E")
    _draft(folder, "kategorien.csv")
    spec = importlib.util.spec_from_file_location(
        "profile_data", Path(__file__).resolve().parent.parent / "scripts" / "profile_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(sys, "argv", ["profile_data.py", "--data", str(folder), "--config", str(folder / "c.yaml")])
    assert mod.main() == 0
    text = capsys.readouterr().out
    assert "record id `Zeilen-ID`: 400 distinct, 0 repeated" in text and "Status [status]" in text
    assert "===== tree: " in text and "could not" not in text
