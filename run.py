#!/usr/bin/env python3
"""Spend cube CLI.

  python run.py --config configs/company_a.yaml --data data/company_A --out out/arnav_mishra__A.csv
  python run.py init-config --data data/company_C --taxonomy taxonomy_C.csv --out configs/company_c.yaml
  python run.py check-config --config configs/company_c.yaml --data data/C
  python run.py evaluate --config configs/company_a.yaml --data data/company_A --mode zero_shot
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from spendcube.classify.llm import load_dotenv_if_present  # noqa: E402


def _setup_logging(level: str):
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpx2", "anthropic", "google_genai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def cmd_run(a):
    from spendcube.pipeline import run
    from spendcube.validate import ValidationError
    try:
        rep = run(a.config, a.data, a.out, use_llm=not a.no_llm, use_labels=not a.no_labels,
                  cache_dir=a.cache_dir, llm_cost_cap=a.max_llm_cost)
    except ValidationError as e:
        print(str(e), file=sys.stderr)
        return 2
    v = rep["validation"]
    llm = rep["classification"].get("llm") or {}
    print("\n=== RUN OK ===")
    print(f"output      : {rep['output']}  ({v['rows']} rows)")
    print(f"included    : {v['included_rows']} rows, spend {v['included_spend']:,.2f}")
    print(f"excluded    : {v['excluded_by_reason']}")
    print(f"reconcile   : input {v['input_spend_rc']:,.2f} = included {v['included_spend']:,.2f} "
          f"+ excluded {v['excluded_spend']:,.2f}")
    print(f"suppliers   : {v['suppliers']}   items: {v['items']}")
    print(f"class source: {rep['classification'].get('class_source_lines')}")
    if llm:
        print(f"LLM         : {llm.get('calls', 0)} calls, ${llm.get('cost_usd', 0):.3f}, "
              f"cache hits {llm.get('cache_hits', 0)}")
    print(f"runtime     : {rep['runtime_sec']}s")
    print(f"audit/report: {rep['audit']}  {rep['report']}")
    return 0


def cmd_init(a):
    from spendcube.init_config import draft_config
    path = draft_config(a.data, a.taxonomy, a.out, company=a.company, use_llm=not a.no_llm,
                        files=a.files)
    print(f"Drafted config written to {path}. Review the TODO/CHECK comments, then run it.")
    return 0


def cmd_eval(a):
    from spendcube.evaluate import evaluate
    res = evaluate(a.config, a.data, mode=a.mode, use_llm=not a.no_llm, cache_dir=a.cache_dir,
                   holdout=a.holdout, seed=a.seed, max_llm_cost=a.max_llm_cost, sample_units=a.sample_units,
                   labelled_only=a.labelled_only)
    print(json.dumps(res, indent=2, default=str))
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    load_dotenv_if_present(ROOT)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--log-level", default="INFO")
    common.add_argument("--no-llm", action="store_true", help="offline mode: no API calls")
    common.add_argument("--cache-dir", default=str(ROOT / ".cache"))
    common.add_argument("--max-llm-cost", type=float, default=None, help="USD cap for this run")

    if argv and argv[0] == "init-config":
        p = argparse.ArgumentParser(prog="run.py init-config", parents=[common])
        p.add_argument("--data", required=True)
        p.add_argument("--taxonomy", required=True, help="taxonomy file name inside --data")
        p.add_argument("--files", nargs="*", default=None, help="input file names (default: all others)")
        p.add_argument("--company", default="C")
        p.add_argument("--out", default="configs/company_c.yaml")
        a = p.parse_args(argv[1:])
        _setup_logging(a.log_level)
        return cmd_init(a)
    if argv and argv[0] == "check-config":
        p = argparse.ArgumentParser(prog="run.py check-config", parents=[common])
        p.add_argument("--config", required=True)
        p.add_argument("--data", required=True)
        a = p.parse_args(argv[1:])
        _setup_logging("WARNING")
        from spendcube.check import check_config
        print(check_config(a.config, a.data))
        return 0
    if argv and argv[0] == "evaluate":
        p = argparse.ArgumentParser(prog="run.py evaluate", parents=[common])
        p.add_argument("--config", required=True)
        p.add_argument("--data", required=True)
        p.add_argument("--mode", choices=["zero_shot", "supplier_holdout", "fresh_slice"], default="zero_shot")
        p.add_argument("--holdout", type=float, default=0.2)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--sample-units", type=int, default=None)
        p.add_argument("--labelled-only", action="store_true", help="LLM only on units that carry labels")
        a = p.parse_args(argv[1:])
        _setup_logging(a.log_level)
        return cmd_eval(a)
    p = argparse.ArgumentParser(prog="run.py", parents=[common])
    p.add_argument("--config", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--no-labels", action="store_true", help="ignore labels (zero-shot run)")
    a = p.parse_args(argv)
    _setup_logging(a.log_level)
    return cmd_run(a)


if __name__ == "__main__":
    sys.exit(main())
