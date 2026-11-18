"""No regular expression in the code or the configs may take long on a hostile description.

A pattern with nested optional repeats can take minutes on one line (catastrophic
backtracking) and stall a whole run. This test collects every pattern it can find, runs
each one against long strings built to trigger that, and fails if any match is slow. The
scan runs in a child process with a time limit, so a bad pattern fails the test instead of
hanging it.
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RE_FUNCS = {"search", "match", "fullmatch", "sub", "subn", "findall", "finditer", "split", "compile"}
STR_FUNCS = {"contains", "extract", "extractall", "fullmatch", "match", "findall", "count"}
LIMIT_SECONDS = 1.0


def _patterns() -> dict[str, str]:
    """Every regular expression in configs/*.yaml and spendcube/**/*.py, keyed by where it is."""
    import yaml
    found: dict[str, str] = {}

    def walk(node, where):
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "pattern" and isinstance(v, str):
                    found[f"{where}:{node.get('name', '?')}"] = v
                else:
                    walk(v, where)
        elif isinstance(node, list):
            for v in node:
                walk(v, where)
    for p in sorted((ROOT / "configs").glob("*.yaml")):
        walk(yaml.safe_load(p.read_text()), p.name)

    for p in sorted((ROOT / "spendcube").rglob("*.py")) + [ROOT / "run.py"] + sorted((ROOT / "scripts").glob("*.py")):
        tree = ast.parse(p.read_text())
        for node in ast.walk(tree):
            # string constants that look like patterns: inside re.<func>(...), .str.<func>(...) or pattern lists
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.args:
                f = node.func
                is_re = isinstance(f.value, ast.Name) and f.value.id == "re" and f.attr in RE_FUNCS
                is_str = isinstance(f.value, ast.Attribute) and f.value.attr == "str" and f.attr in STR_FUNCS
                arg = node.args[0]
                if (is_re or is_str) and isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    found[f"{p.name}:{node.lineno}"] = arg.value
            elif isinstance(node, ast.Constant) and isinstance(node.value, str) and re.search(r"\\[sdwb]|\(\?[i:]|\[\^?[\\\w]", node.value):
                found.setdefault(f"{p.name}:{node.lineno}:const", node.value)
    out = {}
    for where, pat in found.items():
        try:
            re.compile(pat)
            out[where] = pat
        except re.error:
            pass                                  # a string that is not a pattern
    return out


def _hostile() -> list[str]:
    heads = ["VAT ", "TDS payable ", "accrued sales tax ", "gst input ", "income tax payment ", "intercompany ", "interest charge ",
             "INV-", "PO ", "Total ", "Supplier No. ", "", "category l", "a b c "]
    tails = ["1" * 40 + " refund", "-" * 40 + " see note", " " * 60 + "x", "1/" * 30 + "!", "q1 " * 25 + "x", ". % - / " * 12 + "end",
             "ab " * 40 + "!", "01/04/2023-31/03/2024 for vendor " * 3, "_" * 50 + "#", "(" * 30 + ")" * 29, "A" * 120 + "!"]
    return [h + t for h in heads for t in tails]


def _scan() -> dict:
    pats = _patterns()
    texts = _hostile()
    slow, worst = [], (0.0, "")
    for where, pat in pats.items():
        rx = re.compile(pat)
        for s in texts:
            t0 = time.perf_counter()
            rx.search(s)
            dt = time.perf_counter() - t0
            if dt > worst[0]:
                worst = (dt, where)
            if dt > LIMIT_SECONDS:
                slow.append({"where": where, "seconds": round(dt, 2), "text": s[:40]})
                break
    return {"patterns": len(pats), "texts": len(texts), "slow": slow, "worst_ms": round(worst[0] * 1000, 2), "worst_at": worst[1]}


def test_no_pattern_is_slow_on_hostile_text():
    try:
        r = subprocess.run([sys.executable, __file__], capture_output=True, text=True, timeout=120, cwd=ROOT)
    except subprocess.TimeoutExpired:
        raise AssertionError("a regular expression in the code or configs backtracks for minutes on a hostile string")
    assert r.returncode == 0, r.stderr[-500:]
    res = json.loads(r.stdout.strip().splitlines()[-1])
    assert res["patterns"] >= 80, res                 # the collector still finds the patterns
    assert not res["slow"], res["slow"]


def test_the_scan_catches_a_bad_pattern():
    bad = re.compile(r"^(a+)+$")                        # the classic slow pattern, kept short so the test stays fast
    t0 = time.perf_counter()
    bad.search("a" * 22 + "!")
    assert time.perf_counter() - t0 > 0.05


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    print(json.dumps(_scan()))


def test_a_slow_or_invalid_pattern_in_a_config_is_rejected_at_load(tmp_path):
    import pandas as pd
    import pytest
    import yaml

    sys.path.insert(0, str(ROOT))
    from spendcube.config import ConfigError, load_config, pattern_problem

    slow = r"(?i)^\s*vat(\s*[-/]?\s*(q[1-4]|fy)?\s*[\d.%/-]*)*\s*$"     # the shape that once stalled a run
    assert "too slow" in pattern_problem(slow)
    assert "not a valid" in pattern_problem(r"(unclosed")
    assert pattern_problem(r"(?i)\b(intercompany|bank charges?|royalt(y|ies))\b") is None
    pd.DataFrame({"id": ["1"], "vendor": ["Acme"], "text": ["x"], "amt": ["1"]}).to_csv(tmp_path / "lines.csv", index=False)
    pd.DataFrame({"L1": ["A"], "L2": ["B"]}).to_csv(tmp_path / "tree.csv", index=False)
    cfg = {"company": "T", "taxonomy": {"file": "tree.csv", "level_columns": ["L1", "L2"]},
           "sources": [{"name": "s", "file": "lines.csv", "record_id": {"column": "id"},
                        "columns": {"supplier_name": "vendor", "description": "text", "amount": "amt"}}],
           "non_procurement": [{"name": "bad_rule", "field": "description", "pattern": slow}]}
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg))
    t0 = time.perf_counter()
    with pytest.raises(ConfigError, match="bad_rule"):
        load_config(str(tmp_path / "c.yaml"))
    assert time.perf_counter() - t0 < 30                 # rejected quickly, not after a hang
    for name in ("company_a.yaml", "company_b.yaml", "company_c_template.yaml"):
        load_config(str(ROOT / "configs" / name))        # the shipped configs load
