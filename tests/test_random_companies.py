"""Draft a config without Claude for random unseen companies, run offline, and compare every
record with the truth: exclusion code, spend, supplier grouping and category names."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from random_companies import make  # noqa: E402

from spendcube.init_config import draft_config  # noqa: E402
from spendcube.pipeline import run  # noqa: E402


@pytest.fixture(autouse=True)
def no_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


@pytest.mark.parametrize("seed", range(1, 61))
def test_random_unseen_company(seed, tmp_path):
    folder, _, tree_file, truth, feats, leaf_paths = make(seed, tmp_path)
    cfg = draft_config(str(folder), tree_file, str(tmp_path / "c.yaml"), use_llm=False)
    run(str(cfg), str(folder), str(tmp_path / "out.csv"), use_llm=False, cache_dir=str(tmp_path / "cache"))
    out = pd.read_csv(tmp_path / "out.csv", dtype=str, keep_default_na=False)
    m = truth.merge(out, on="record_id", how="left")
    assert len(out) == len(truth) and m["include"].notna().all(), feats            # one row per record
    wrong = m[m["code"] != m["exclusion_reason"]]
    assert wrong.empty, (feats, wrong[["code", "exclusion_reason"]].value_counts().to_dict())
    inc = m[m["include"] == "1"]
    spend = pd.to_numeric(inc["spend"])
    assert ((spend - inc["usd"]).abs() <= inc["usd"].abs() * 0.01 + 0.02).all(), feats  # net USD per line
    assert (pd.to_numeric(m.loc[m["include"] == "0", "spend"]) == 0).all(), feats
    got = set(inc[["l1", "l2", "l3", "l4"]].apply(lambda s: tuple(x for x in s if x), axis=1))
    assert got <= leaf_paths, feats                                                # names as in the tree file
    assert inc.groupby("supplier_key")["ent"].nunique().max() == 1, feats           # one firm per key
    assert inc.groupby("ent")["supplier_key"].nunique().max() == 1, feats           # one key per firm
