"""Robustness: Claude-only defaults, LLM health reporting, header-only input,
missing config columns. No network calls (the API key is removed)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spendcube.classify.llm import LLMClassifier, check_llm_health, llm_settings  # noqa: E402
from spendcube.config import DEFAULTS, load_config  # noqa: E402
from spendcube.ingest import ConfigColumnError  # noqa: E402
from spendcube.pipeline import LLMUnavailableError, run  # noqa: E402
from spendcube.taxonomy import Taxonomy  # noqa: E402


def _company(tmp: Path, rows: int = 3, llm_block: bool = False):
    lines = pd.DataFrame({"id": [f"L{i}" for i in range(rows)], "vendor": ["Acme Ltd"] * rows,
                          "text": ["steel plate"] * rows, "amt": ["10.00"] * rows})
    lines.to_csv(tmp / "lines.csv", index=False)
    pd.DataFrame({"L1": ["Materials", "Services"], "L2": ["Metals", "Cleaning"]}).to_csv(tmp / "tree.csv", index=False)
    cfg = {"company": "T", "taxonomy": {"file": "tree.csv", "level_columns": ["L1", "L2"]},
           "sources": [{"name": "s", "file": "lines.csv", "record_id": {"column": "id"},
                        "columns": {"supplier_name": "vendor", "description": "text", "amount": "amt"}}]}
    if llm_block:
        cfg["classification"] = {"llm": {"enabled": True}}
    (tmp / "c.yaml").write_text(yaml.safe_dump(cfg))
    return tmp / "c.yaml"


@pytest.fixture(autouse=True)
def no_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def test_defaults_are_claude_only_and_use_the_keys_the_code_reads(tmp_path):
    llm = DEFAULTS["classification"]["llm"]
    for role in ("primary", "second", "judge"):
        assert llm[role]["vendor"] == "anthropic" and llm[role]["model"].startswith("claude-")
    assert llm["primary"]["model"] == "claude-opus-5-5"
    assert llm["second"]["model"] == "claude-sonnet-5-5"
    assert llm["judge"]["model"] == "claude-opus-5-5"
    cfg = load_config(_company(tmp_path))          # config without any llm block
    lc = LLMClassifier(cfg, Taxonomy.from_paths([("A", "B")]), tmp_path)
    assert (lc.primary.model, lc.second.model, lc.judge.model) == \
        ("claude-opus-5-5", "claude-sonnet-5-5", "claude-opus-5-5")
    assert "gemini" not in str(DEFAULTS).lower()


def test_health_check_reports_missing_key_and_require_llm_stops(tmp_path, capsys):
    cfg = load_config(_company(tmp_path))
    h = check_llm_health(cfg)
    assert set(h) == {"claude-opus-5-5", "claude-sonnet-5-5"}
    assert all("ANTHROPIC_API_KEY" in v for v in h.values())
    with pytest.raises(LLMUnavailableError):
        run(str(tmp_path / "c.yaml"), str(tmp_path), str(tmp_path / "o.csv"), use_llm=True,
            require_llm=True, cache_dir=str(tmp_path / "cache"))
    rep = run(str(tmp_path / "c.yaml"), str(tmp_path), str(tmp_path / "o.csv"), use_llm=True,
              cache_dir=str(tmp_path / "cache"))
    assert rep["llm_health"]["enabled"] and not rep["llm_health"]["usable"]
    assert "UNAVAILABLE" in rep["llm_health"]["message"]
    assert "LLM health" in capsys.readouterr().out


def test_header_only_input_writes_header_only_output(tmp_path):
    cfgp = _company(tmp_path, rows=0)
    rep = run(str(cfgp), str(tmp_path), str(tmp_path / "o.csv"), use_llm=False, cache_dir=str(tmp_path / "cache"))
    out = pd.read_csv(tmp_path / "o.csv")
    assert len(out) == 0
    assert list(out.columns)[:3] == ["record_id", "include", "exclusion_reason"]
    assert rep["validation"]["rows"] == 0


def test_missing_config_column_is_one_clear_error(tmp_path):
    cfgp = _company(tmp_path)
    cfg = yaml.safe_load(cfgp.read_text())
    cfg["sources"][0]["columns"]["supplier_id"] = "Vendor No"
    cfgp.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ConfigColumnError) as e:
        run(str(cfgp), str(tmp_path), str(tmp_path / "o.csv"), use_llm=False, cache_dir=str(tmp_path / "cache"))
    assert "Vendor No" in str(e.value) and "lines.csv" in str(e.value)


def test_cli_exit_code_2_without_traceback(tmp_path, capsys):
    import run as cli
    cfgp = _company(tmp_path)
    cfg = yaml.safe_load(cfgp.read_text())
    cfg["sources"][0]["columns"]["amount"] = "Net Amount"
    cfgp.write_text(yaml.safe_dump(cfg))
    code = cli.main(["--config", str(cfgp), "--data", str(tmp_path), "--out", str(tmp_path / "o.csv"),
                     "--no-llm", "--log-level", "ERROR"])
    err = capsys.readouterr().err
    assert code == 2
    assert err.startswith("ERROR:") and "Traceback" not in err and len(err.strip().splitlines()) == 1


def test_llm_settings_merges_partial_block(tmp_path):
    cfg = load_config(_company(tmp_path, llm_block=True))
    s = llm_settings(cfg)
    assert s["primary"]["model"] == "claude-opus-5-5" and s["enabled"] is True
