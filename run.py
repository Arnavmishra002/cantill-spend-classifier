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
    for noisy in ("httpx", "httpx2", "anthropic", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _input_errors():
    from spendcube.config import ConfigError
    from spendcube.ingest import ConfigColumnError
    return (ConfigError, ConfigColumnError, FileNotFoundError)


def cmd_run(a):
    from spendcube.pipeline import LLMUnavailableError, run
    from spendcube.validate import ValidationError
    try:
        rep = run(a.config, a.data, a.out, use_llm=not a.no_llm, use_labels=not a.no_labels,
                  cache_dir=a.cache_dir, llm_cost_cap=a.max_llm_cost, require_llm=a.require_llm)
    except ValidationError as e:
        print(str(e), file=sys.stderr)
        return 2
    except _input_errors() as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    except LLMUnavailableError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3
    v = rep["validation"]
    llm = rep["classification"].get("llm") or {}
    health = rep.get("llm_health") or {}
    print("\n=== RUN OK ===")
    print(f"output      : {rep['output']}  ({v['rows']} rows)")
    print(f"included    : {v['included_rows']} rows, spend {v['included_spend']:,.2f}")
    print(f"excluded    : {v['excluded_by_reason']}")
    print(f"reconcile   : input {v['input_spend_rc']:,.2f} = included {v['included_spend']:,.2f} "
          f"+ excluded {v['excluded_spend']:,.2f}")
    print(f"suppliers   : {v['suppliers']}   items: {v['items']}")
    xs = rep.get("cross_source") or {}
    if xs.get("secondary_only_included_rows") is not None:
        print(f"only in 2nd : {xs['secondary_only_included_rows']} rows, spend "
              f"{xs['secondary_only_included_spend']:,.2f} (spend found in one system only)")
    unmapped = {k: v for k, v in (rep.get("rules") or {}).items() if k.startswith("unmapped_status")}
    for k, vals in unmapped.items():
        print(f"WARNING     : status values not in the status map, treated as NOT_APPROVED: {vals}")
    for w in rep.get("warnings") or []:
        print(f"WARNING     : {w}")
    print(f"class source: {rep['classification'].get('class_source_lines')}")
    rv = rep.get("review") or {}
    if rv:
        print(f"review      : {rv['lines']} included lines below confidence {rv['threshold']} "
              f"({rv['line_share']:.1%} of lines, {rv['spend_share']:.1%} of spend) for human review")
        print(f"coverage    : {rv['coverage_l2_spend']:.1%} of spend classified to level 2 or deeper, "
              f"{rv['confident_l2_spend']:.1%} at confidence {rv['threshold']} or more")
    if health.get("enabled") and not health.get("usable"):
        print("LLM         : NOT USED. " + health.get("message", "no provider answered"))
    elif not health.get("enabled"):
        print(f"LLM         : {health.get('message', 'off')}")
    else:
        print(f"LLM         : {llm.get('calls', 0)} calls, ${llm.get('cost_usd', 0):.3f}, "
              f"cache hits {llm.get('cache_hits', 0)}, models {llm.get('primary')} / {llm.get('second')} "
              f"/ {llm.get('judge')}")
    print(f"runtime     : {rep['runtime_sec']}s")
    print(f"audit/report: {rep['audit']}  {rep['report']}")
    return 0


def cmd_init(a):
    from spendcube.init_config import draft_config
    try:
        path = draft_config(a.data, a.taxonomy, a.out, company=a.company, use_llm=not a.no_llm,
                            files=a.files)
    except _input_errors() + (ValueError,) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    for line in Path(path).read_text().splitlines():
        if line.startswith("#  - WARNING"):
            print(line[5:])
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
        try:
            print(check_config(a.config, a.data))
        except _input_errors() as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2
        return 0
    if argv and argv[0] == "calibrate":
        p = argparse.ArgumentParser(prog="run.py calibrate", parents=[common])
        p.add_argument("--pairs", nargs="+", required=True, help="config.yaml:data_dir pairs with labels")
        a = p.parse_args(argv[1:])
        _setup_logging(a.log_level)
        from spendcube.calibrate import calibrate
        res = calibrate([tuple(x.split(":", 1)) for x in a.pairs])
        print(json.dumps(res, indent=2))
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
    p.add_argument("--require-llm", action="store_true",
                   help="stop (exit 3) if the LLM is enabled but no Claude model answers")
    a = p.parse_args(argv)
    _setup_logging(a.log_level)
    return cmd_run(a)


if __name__ == "__main__":
    sys.exit(main())
