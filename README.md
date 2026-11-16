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

Python 3.11 or newer. The LLM layer uses the Claude API and is optional: without a key the tool runs on its
offline layers and says so in the run summary. To enable it, put the key in a git-ignored `.env` file or export it:

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

Useful flags: `--no-llm` (offline, no API calls), `--require-llm` (stop with exit code 3 if no Claude model
answers), `--max-llm-cost 5` (USD cap for the run), `--no-labels` (zero-shot), `--log-level DEBUG`.

Every run starts with a health check of each Claude model it will use and ends with a validation of its output:
template columns and order, one row per input record, unique ids, valid exclusion codes, categories that exist in
the tree, and spend that reconciles with the input. A failed check stops the run with exit code 2. A config column
that is missing from the input file prints one error line and exits with code 2. Next to each output,
`out/<name>_audit.csv` explains every row (rule, matched record, flags, FX source, classification layer) and
`out/<name>_report.json` summarises the run. Both are git-ignored because they contain raw descriptions.

## Evaluate

```bash
python run.py evaluate --config configs/company_a.yaml --data data/company_A --mode zero_shot        # labels hidden
python run.py evaluate --config configs/company_a.yaml --data data/company_A --mode supplier_holdout
python run.py evaluate --config configs/company_a.yaml --data data/company_A --mode fresh_slice      # simulates A2
python run.py calibrate --pairs configs/company_a.yaml:data/company_A configs/company_b.yaml:data/company_B
python -m pytest -q
```

## Layout

```
run.py                       command line: run, init-config, check-config, evaluate, calibrate
configs/company_a.yaml       everything specific to company A
configs/company_b.yaml       everything specific to company B
spendcube/
  ingest.py clean.py spend_rules.py match.py dedupe.py export.py validate.py
  taxonomy.py paths.py textnorm.py config.py pipeline.py evaluate.py init_config.py check.py calibrate.py
  classify/   units.py vectors.py llm.py enrich.py calibration.py __init__.py (layer orchestration)
  data/       fx_monthly_usd.csv (public ECB rates), offline_calibration.json
tests/                       unit tests on synthetic data
submission/                  output files only
```
