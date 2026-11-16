# RUNBOOK – blind test, Thu 19 Nov 2026, 10:00–12:00 IST

Rules: **no code changes after 10:00.** Only the C config and the three output files are pushed, into
`submission/`. Target: pushed by **11:15**.

## 0. Before 10:00 (once)
```bash
cd cantill-spend-classifier
git status                      # clean, on main, at tag v1.0-freeze
source .venv/bin/activate       # or: python3.12 -m venv .venv && pip install -r requirements.txt
cat .env | cut -d= -f1          # ANTHROPIC_API_KEY / GEMINI_API_KEY present (values never printed)
python -m pytest -q             # all green
```

## 1. Put the new data in `data/` (git-ignored)
Unzip the email attachment so that it looks like (names may differ – configs accept patterns):
```
data/A2/   erp_goods_receipts*.csv, invoice_tool_export*.csv   (+ taxonomy if shipped)
data/B2/   invoice_lines*.csv                                   (+ fx table / taxonomy if shipped)
data/C/    <invoice lines file>, <taxonomy file>
```
The A/B trees, labels and FX table are found in `data/company_A` / `data/company_B` when the new folders do not
repeat them. **Do not delete `data/company_A` or `data/company_B`** (they are the training rows for A2/B2).

## 2. Draft the Company C config (config only, no code)
```bash
python run.py init-config --data data/C --taxonomy "<tree file name>" --company C --out configs/company_c.yaml
```
Review `configs/company_c.yaml` (10 minutes max):
- [ ] `sources[0].file` and `record_id.column` = the line id column (output uses it as `record_id`)
- [ ] `columns.amount` is the line amount; `amount_type` net/gross correct; tax rate column if gross
- [ ] `currency` / `fx_rate` / `fx.table` (reporting currency is USD for C per brief)
- [ ] `date` = invoice date, `posting_date` if present; `dates.dayfirst` matches the file
- [ ] `supplier_name`, `supplier_id`, `description`, `item_code`, `gl_description`, `context` sensible
- [ ] `status_map`: paid/approved → include; cancelled → CANCELLED; rejected/hold/... → NOT_APPROVED
- [ ] `non_procurement`: keep generic tax/intercompany/finance rules; check drafted GL account list
- [ ] `taxonomy.level_columns` in order L1..L4 (or `format: parent_child`)
- [ ] `classification.company_context` reads right (one or two sentences)
- [ ] `cost_cap_usd` (default 8) is fine

## 3. Run the three companies
```bash
python run.py --config configs/company_a.yaml --data data/A2 --out submission/arnav_mishra__A2.csv
python run.py --config configs/company_b.yaml --data data/B2 --out submission/arnav_mishra__B2.csv
python run.py --config configs/company_c.yaml --data data/C  --out submission/arnav_mishra__C.csv
```
Each ends with `=== RUN OK ===` (validation passed) or prints `OUTPUT VALIDATION FAILED` and exits 2.
If the LLM is unavailable the run still finishes on the offline layers (lower accuracy) – that is acceptable.
Expected runtime: A2/B2 a few minutes, C ≈ 10–25 min for 25k lines.

## 4. Check
- [ ] three `=== RUN OK ===` blocks; row counts = input rows
- [ ] `excluded` counts look plausible (C: some NON_PROCUREMENT? any DUPLICATE?)
- [ ] `reconcile` line: input = included + excluded
- [ ] open each `_report.json`: `classification.class_source_lines` mostly memory/supervised/llm, not retrieval
- [ ] spot-check 10 lines of C in the `_audit.csv` (description vs l1–l4)
If C's config needs a fix (a wrong column), edit **only** `configs/company_c.yaml` and re-run step 3 for C.

## 5. Submit (config + outputs only)
```bash
cp configs/company_c.yaml submission/company_c.yaml
git status --short                      # expect only submission/ files
git add submission/arnav_mishra__A2.csv submission/arnav_mishra__B2.csv submission/arnav_mishra__C.csv submission/company_c.yaml
git diff --cached --name-only           # nothing outside submission/, no audit/report files
git commit -m "Blind test outputs (A2, B2, C) and Company C config"
git push origin main
```
Then open the repo on GitHub and confirm the four files are visible. Done by 11:15; hard stop 12:00.

## If something breaks
| Symptom | Action (no code edits) |
|---|---|
| `No file matching ...` | file name differs: pass the right folder, or edit only `configs/company_c.yaml` for C. For A2/B2 copy/rename the file inside `data/A2` to the expected name. |
| `column(s) [...] not in file` (C) | fix the column name in `configs/company_c.yaml` |
| LLM errors / rate limits | let it finish (offline fallback) or re-run: finished batches are cached |
| Validation failure on categories | check `taxonomy.level_columns` order in the C config |
| Out of time | `--no-llm` runs everything offline in under 2 minutes |
