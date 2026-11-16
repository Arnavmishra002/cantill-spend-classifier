# Cantill spend cube builder

Takes a company's raw purchase data and its category tree, then **combines and cleans** the data, decides **what
counts as spend** (with a reason code for every exclusion), **de-duplicates suppliers and items**, and **classifies
every line** into the company's tree with a confidence. One config file per company, no code changes per company.
Works with labels (A, B) and without any (an unseen company C).

- Method, AI use, metrics and cost: [APPROACH.md](APPROACH.md)
- What I found in the data and decided: [DECISIONS.md](DECISIONS.md)
- Blind-test checklist: [RUNBOOK.md](RUNBOOK.md)

## Setup
```bash
python3.12 -m venv .venv && source .venv/bin/activate      # Python 3.11+
pip install -r requirements.txt
```
Optional LLM layer (the tool runs without it, on its offline layers):
```bash
cp .env.example .env     # then put your keys in .env (git-ignored), or export them in the shell
# ANTHROPIC_API_KEY=...   Claude: line classification, supplier profiles, tree scope notes
# GEMINI_API_KEY=...      Gemini: second opinion on uncertain lines, embeddings, failover
```
The task data is confidential and is **not** in this repo. Put the kit under `data/` (git-ignored):
```
data/output_template.csv
data/company_A/{erp_goods_receipts.csv, invoice_tool_export.csv, labels_A.csv, taxonomy_A.csv}
data/company_B/{invoice_lines.csv, fx_rates_usd.csv, seed_labels.csv, taxonomy_B.csv}
```

## Run (one command per company)
```bash
python run.py --config configs/company_a.yaml --data data/company_A --out submission/arnav_mishra__A.csv
python run.py --config configs/company_b.yaml --data data/company_B --out submission/arnav_mishra__B.csv
```
Blind test (same configs for the new drops; C gets a drafted config):
```bash
python run.py --config configs/company_a.yaml --data data/A2 --out submission/arnav_mishra__A2.csv
python run.py --config configs/company_b.yaml --data data/B2 --out submission/arnav_mishra__B2.csv
python run.py init-config --data data/C --taxonomy "<tree file>" --company C --out configs/company_c.yaml
python run.py --config configs/company_c.yaml --data data/C --out submission/arnav_mishra__C.csv
```
Useful flags: `--no-llm` (offline, no API calls), `--max-llm-cost 5` (USD cap for the run),
`--no-labels` (zero-shot), `--log-level DEBUG`.

Every run validates its output (template columns and order, one row per input record, unique ids, valid codes,
categories that exist in the tree, spend reconciliation) and stops with exit code 2 if anything is wrong.
Next to each output, `out/<name>_audit.csv` explains every row (rule, matched record, flags, FX source,
classification layer) and `out/<name>_report.json` summarises the run.

## Evaluate
```bash
python run.py evaluate --config configs/company_a.yaml --data data/company_A --mode zero_shot      # labels hidden
python run.py evaluate --config configs/company_a.yaml --data data/company_A --mode supplier_holdout
python run.py evaluate --config configs/company_a.yaml --data data/company_A --mode fresh_slice    # simulates A2
python -m pytest -q
```

## Layout
```
run.py                      CLI: run | init-config | evaluate
configs/company_{a,b}.yaml  everything company-specific
spendcube/
  ingest.py   clean.py   spend_rules.py   match.py   dedupe.py   export.py   validate.py
  taxonomy.py  paths.py  textnorm.py  config.py  pipeline.py  evaluate.py  init_config.py
  classify/   units.py  vectors.py  llm.py  enrich.py  __init__.py (layer orchestration)
tests/                      synthetic-data unit tests
submission/                 output files only
```
