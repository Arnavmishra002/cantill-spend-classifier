# RUNBOOK: blind test, Thursday 19 November 2026, 10:00 to 12:00 IST

Rules: no code changes after 10:00. Push only the Company C config and the three output files, into `submission/`.
Target: pushed by 11:15. Hard stop: 12:00.

## 0. Before 10:00

Check the inbox that received the challenge email, including its spam folder, for the blind-test email at 10:00.
Run every command in a normal terminal, not inside a Claude Code sandbox, which blocks the API calls.


```bash
cd cantill-spend-classifier
git status                       # clean, on main, at tag v1.0-freeze
source .venv/bin/activate        # or: python3.12 -m venv .venv && pip install -r requirements.txt
cut -d= -f1 .env                 # ANTHROPIC_API_KEY listed (the value is never printed)
python -m pytest -q              # all tests pass
```

## 1. Put the new data in `data/` (git-ignored)

Unzip the email attachment so that it looks like this (file names may differ; the configs accept patterns):

```
data/A2/   erp_goods_receipts*.csv, invoice_tool_export*.csv   (and the tree if shipped)
data/B2/   invoice_lines*.csv                                   (and the FX table or tree if shipped)
data/C/    <invoice lines file>, <tree file>
```

The A and B trees, labels and FX table are read from `data/company_A` and `data/company_B` when the new folders
do not repeat them. Do not delete `data/company_A` or `data/company_B`: they are the training rows for A2 and B2.

## 2. Draft the Company C config (config only, no code)

```bash
python run.py init-config --data data/C --taxonomy "<tree file name>" --company C --out configs/company_c.yaml
python run.py check-config --config configs/company_c.yaml --data data/C
```

Review `configs/company_c.yaml` and the `check-config` output (10 minutes at most):

- [ ] `classification.company_context`: fill it in now, at 10:00, in one or two plain sentences (industry, what
      the company buys), for example: "A US hospital group that buys medical supplies, drugs, imaging equipment,
      facility services and IT." The draft writes one when the API is reachable; check it.
- [ ] `record_id.column` is the line id column (the output uses it as `record_id`).
- [ ] `columns.amount` is the line amount; `amount_type` net or gross is right; tax rate column if gross.
- [ ] `currency` and `fx_rate`; `fx.table` if the email ships rates. Without a table the shipped ECB monthly table
      converts other currencies to USD (lines are flagged `FX_PUBLIC_TABLE`).
- [ ] `date` is the invoice date, `posting_date` if present; `dayfirst` was decided from the values.
- [ ] `supplier_name`, `supplier_id`, `description`, `item_code`, `gl_description`, `context` look sensible.
- [ ] `status_map`: paid and approved are included; cancelled is CANCELLED; rejected, on hold and the like are
      NOT_APPROVED.
- [ ] `non_procurement`: keep the generic tax, intercompany and finance rules; check the drafted GL account list.
- [ ] `taxonomy.level_columns` are in order L1 to L4 (or `format: parent_child`).
- [ ] `check-config` shows parse rates near 100% for amount and date, and plausible exclusions.

## 3. Run the three companies

Start C first: it takes longest and is worth 50 points. Run A2 and B2 at the same time in a second terminal.

```bash
python run.py --config configs/company_c.yaml --data data/C  --out submission/arnav_mishra__C.csv
```

```bash
python run.py --config configs/company_a.yaml --data data/A2 --out submission/arnav_mishra__A2.csv
python run.py --config configs/company_b.yaml --data data/B2 --out submission/arnav_mishra__B2.csv
```

Measured on Monday: A (65,549 lines) about 2 minutes, B (41,028 lines) about 12 minutes with no cache, 2,000
California lines with a 200-leaf tree 2 minutes and 3.30 USD. A 25,000-line C is expected to take about 25 minutes
and cost about 40 USD; the drafted cap is 60 USD.

Each run prints an LLM health line per model first, then ends with `=== RUN OK ===` (validation passed), or prints
`OUTPUT VALIDATION FAILED` or an `ERROR:` line and exits with code 2. If the summary says `LLM : NOT USED`, the key
or the network is the problem: fix `.env` and re-run. Finished batches are cached, so a re-run only pays for what is
missing.

## 4. Check

- [ ] Three `=== RUN OK ===` blocks; row counts equal input rows.
- [ ] Exclusion counts look plausible.
- [ ] The reconcile line shows input = included + excluded.
- [ ] In `out/arnav_mishra__C_report.json`, `classification.class_source_lines` is mostly `llm`, not `retrieval`.
- [ ] Spot-check 10 lines of C in `out/arnav_mishra__C_audit.csv` (description against l1 to l4).

If the C config needs a fix (a wrong column), edit only `configs/company_c.yaml` and re-run step 3 for C.

## 5. Submit (config and outputs only)

```bash
cp configs/company_c.yaml submission/company_c.yaml
git status --short                       # expect only submission/ files
git add submission/arnav_mishra__A2.csv submission/arnav_mishra__B2.csv submission/arnav_mishra__C.csv submission/company_c.yaml
git diff --cached --name-only            # nothing outside submission/, no audit or report files
git commit -m "Blind test outputs (A2, B2, C) and Company C config"
git push origin main
```

Open the repository on GitHub and confirm the four files are there.

## If something breaks

| Symptom | Action (no code edits) |
|---|---|
| `No file matching ...` | The file name differs. Pass the right folder; for A2 or B2 copy the file inside `data/A2` to the expected name; for C fix `configs/company_c.yaml`. |
| `ERROR: config ... column ... is not in the header` | Fix the column name in `configs/company_c.yaml`. |
| `LLM : NOT USED` | Check `.env` and the network, then re-run. Use `--require-llm` to make the run stop instead. |
| Validation failure on categories | Check the order of `taxonomy.level_columns` in the C config. |
| Out of time | `--no-llm` runs everything offline in under two minutes (lower accuracy). |
