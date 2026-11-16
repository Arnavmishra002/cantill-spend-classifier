# Cantill spend cube builder

This tool takes a company's raw purchase data and its category tree. It combines and cleans the data, decides
what counts as spend (with a reason code for every exclusion), de-duplicates suppliers and items, and classifies
every line into the company's tree with a confidence. Each company is described by one config file; a new company
needs a new config, not new code. It works with labels (companies A and B) and without any (an unseen company C).

- Method, AI use, measured accuracy and cost: [APPROACH.md](APPROACH.md)
- What I found in the data and what I decided: [DECISIONS.md](DECISIONS.md)
- Blind-test checklist: [RUNBOOK.md](RUNBOOK.md)

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.11 or newer. The LLM layer uses the Claude API and is optional. Without a key the tool runs on its offline
layers, says so in the run summary, and accuracy drops to the offline numbers in APPROACH.md. With a key, a re-run
on the same machine reads the disk cache in `.cache/` and reproduces the same output; on a new machine Claude is
asked again, so a few answers can differ. To enable it, put the key in a git-ignored `.env` file or export it:

```bash
cp .env.example .env      # then fill in ANTHROPIC_API_KEY
```

The task data is confidential and is not in this repository. Put the kit under `data/`, which git ignores:

```
data/output_template.csv
data/company_A/erp_goods_receipts.csv, invoice_tool_export.csv, labels_A.csv, taxonomy_A.csv
data/company_B/invoice_lines.csv, fx_rates_usd.csv, seed_labels.csv, taxonomy_B.csv
```

## Run (one command per company)

```bash
python run.py --config configs/company_a.yaml --data data/company_A --out submission/arnav_mishra__A.csv
python run.py --config configs/company_b.yaml --data data/company_B --out submission/arnav_mishra__B.csv
```

Blind test. A2 and B2 use the same configs; C gets a drafted config that I review:

```bash
python run.py --config configs/company_a.yaml --data data/A2 --out submission/arnav_mishra__A2.csv
python run.py --config configs/company_b.yaml --data data/B2 --out submission/arnav_mishra__B2.csv
python run.py init-config --data data/C --taxonomy "<tree file>" --company C --out configs/company_c.yaml
python run.py check-config --config configs/company_c.yaml --data data/C
python run.py --config configs/company_c.yaml --data data/C --out submission/arnav_mishra__C.csv
```

If `init-config` fails on C, copy `configs/company_c_template.yaml` to `configs/company_c.yaml` and fill in the
CHANGE_ME names (RUNBOOK.md, "If something breaks").

Useful flags: `--no-llm` (offline, no API calls), `--require-llm` (stop with exit code 3 if no Claude model
answers), `--max-llm-cost 5` (USD cap for the run), `--no-labels` (zero-shot), `--log-level DEBUG`.

Every run starts with a health check of each Claude model it will use and ends with a validation of its output:
template columns and order, one row per input record, unique ids, valid exclusion codes, categories that exist in
the tree, and spend that reconciles with the input. A failed check stops the run with exit code 2. A config column
that is missing from the input file prints one error line and exits with code 2. Next to each output,
`out/<name>_audit.csv` (or the folder in `SPENDCUBE_SIDECAR_DIR`) explains every row (rule, matched record, flags, FX source, classification layer) and
`out/<name>_report.json` summarises the run. Both are git-ignored because they contain raw descriptions.

## How to change things (config only)

- A new exclusion rule: add an entry under `non_procurement` with a `field` (any mapped field or raw column) and a
  `pattern` or a list of `values`. The run summary shows how many lines each rule caught.
- A new input column: map it under `sources[].columns` to a canonical field (`description`, `gl_description`,
  `context` and so on). Several columns can be listed for one text field; they are joined.
- A different model: change `classification.llm.primary`, `second`, `judge` or `tail`, for example
  `{vendor: anthropic, model: claude-sonnet-5-5, effort: low}`. `cost_cap_usd` limits spend per run.
- A new status value: add it to the right list in `status_map`. Unknown values count as `NOT_APPROVED`.
- A new company: `python run.py init-config ...`, review the draft, run `check-config`, then run.

## Evaluate

```bash
python run.py evaluate --config configs/company_a.yaml --data data/company_A --mode zero_shot        # labels hidden
python run.py evaluate --config configs/company_a.yaml --data data/company_A --mode supplier_holdout
python run.py evaluate --config configs/company_a.yaml --data data/company_A --mode fresh_slice      # simulates A2
python run.py calibrate --pairs configs/company_a.yaml:data/company_A configs/company_b.yaml:data/company_B
python scripts/california_check.py --csv <downloaded ca_po.csv> --out out/ca --lines 2000 --leaves 200
python -m pytest -q
```

## Layout

```
run.py                       command line: run, init-config, check-config, evaluate, calibrate
configs/company_a.yaml       everything specific to company A
configs/company_b.yaml       everything specific to company B
configs/company_c_template.yaml  fallback config for an unseen company, filled in by hand
spendcube/
  ingest.py clean.py spend_rules.py match.py dedupe.py export.py validate.py
  taxonomy.py paths.py textnorm.py config.py pipeline.py evaluate.py init_config.py check.py calibrate.py
  classify/   units.py vectors.py llm.py enrich.py calibration.py __init__.py (layer orchestration)
  data/       fx_monthly_usd.csv (public ECB rates), offline_calibration.json
scripts/california_check.py  external check on public California purchase data (not stored here)
tests/                       unit tests on synthetic data
submission/                  output files only
```
