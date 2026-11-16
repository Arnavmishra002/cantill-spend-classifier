# DECISIONS.md

What I found in the data, what was unclear, the options I considered, what I chose and why.
Counts refer to the kit data (A: 37,029 ERP + 28,520 tool lines; B: 41,028 lines). No raw data rows are
reproduced here; examples are format patterns with digits masked. Every rule below is a config value, not code,
unless it says "generic".

## 0. Ground rules I set for myself

| Topic | Decision | Why |
|---|---|---|
| Company specifics | Only in `configs/*.yaml` (file names, columns, formats, rules, thresholds). The package has no column names, company names or category names. | Brief §3.1 and §3.7; Company C must need a config, not code. |
| Brief vs. my instructions | Where they differ, the brief wins. Example: the brief requires `__A.csv` / `__B.csv` in the repo at the build deadline, so they are committed under `submission/`. | The brief is the source of truth. |
| Output for excluded rows | One row per input record. Excluded rows: `include=0`, a code, `spend=0`, **blank** l1–l4 (as in `output_template.csv`), supplier/item keys filled, `confidence` = confidence of the exclusion decision. | Matches the template's example row exactly. |
| Unclassifiable lines | Every included line still gets a leaf (deepest level). Confidence carries the uncertainty. | Scoring is accuracy per level; a leaf guess can be right, a blank level cannot. |

## 1. Company A – two systems

### 1.1 What is spend in each system
* **ERP goods receipts** (`value_zar`) are net of VAT and already in ZAR, even on the 15 lines whose
  `document_currency` is EUR/CHF (that column is the PO currency; the value column is ZAR). → `amount_type: net`,
  currency fixed to ZAR.
* **ERP negative lines (434)** are goods-receipt reversals (quantity −1, same text, same PO). Options: exclude the pair
  as `CANCELLED`, or keep both so they net to zero. **Chose: keep them** (net spend is right either way) because
  `labels_A.csv` labels every ERP line and has no include/exclusion column, which signals that every ERP record is
  counted. Exclusions in A therefore come from the tool file.
* **Tool amounts are gross.** VAT column has five spellings: `15`, `0.15`, `15%`, blank and `0`. Parsed to a rate;
  blank → 15% (brief). **`0` is kept as 0% (no VAT removed).** Evidence: tool lines with VAT `0` match an ERP line
  exactly on *gross* amount (71% match rate, same as the other rates after removing VAT), so they really carry no VAT.
* Tool amounts sometimes use thousands separators (`1,234.56`, 7,920 lines) → generic number parser (also handles
  `1.234,56`, `1,00,000`, currency symbols, `(123.45)` and trailing minus).
* Tool `invoice_date` has three layouts (`YYYY-MM-DD`, `DD/MM/YYYY`, `DD-Mon-YY`) and is blank on ~65% of lines.
  Dates are not needed for A's spend, only as a matching tie-breaker.

### 1.2 Status
Paid / Approved → counted. Cancelled → `CANCELLED`. Rejected / On Hold → `NOT_APPROVED`. Any status value not in
the map → `NOT_APPROVED` (brief: "only approved or paid count"), and it is logged.
Precedence when a line qualifies twice: `CANCELLED > NOT_APPROVED > NON_PROCUREMENT > DUPLICATE`. A cancelled tool
line that also matches an ERP receipt is reported as `CANCELLED` (it is cancelled whatever else is true; the ERP
record is counted anyway). In practice only 10 such lines exist.

### 1.3 Matching the two systems ("count once")
Unclear: how to recognise the same purchase when PO references are reformatted or missing and invoice numbers
differ in format. Findings: PO reference appears in five layouts (`45100xxxxx`, `PO#45100xxxxx`, `PO 45100xxxxx`,
`45100xxxxx/10.0`, blank on 18%); invoice numbers carry prefixes (`INV/`, `SI-`, `TAX INV`, `IN`).
Net-of-VAT tool amounts equal the ERP value to the cent for most duplicates (median ratio 1.000000).

Chosen algorithm (`spendcube/match.py`, all thresholds in config):
1. **"Exported twice" clusters inside the tool file**: same normalised invoice number + same net amount + same
   quantity + near-identical OCR-normalised text (≥80 similarity) = one invoice line. 2,342 lines sit in exact pairs
   that differ only in formatting (supplier spelling, PO layout); their statuses always agree.
2. Each cluster is matched **one-to-one** to an ERP receipt, strictest evidence first; an ERP line can absorb only one
   cluster:
   * **P1** same PO (+ PO item when both have one) and net amount within max(0.05, 0.05%) → 15,644 clusters.
   * **P2** no usable PO (blank, or a PO the ERP has never seen): same supplier (after de-duplication) + same amount
     + similar text, and dates within 400 days when both are known → ~5,000 clusters. Most of these are tool lines
     whose PO reference is blank.
   * **P3** same PO + same unit price (±0.5%) + quantity within 6% + similar text → 133 clusters (weighbridge-type
     quantity differences where unit price is identical).
   * **P4** one invoice line = several receipts of the same PO at the same unit price (subset sum) → a handful.
3. Matched clusters → every tool row `DUPLICATE`. Unmatched clusters → first row counted, the other copies
   `DUPLICATE` (exported twice).

Options rejected: matching on amount only (recurring identical charges would collide); many-to-one without unit-price
agreement (too loose). Remaining unmatched tool lines that share a PO with ERP lines (~690) mostly invoice a
different quantity than any free receipt (partial invoicing or deliveries not in the ERP extract); I count them as
tool-only spend rather than guess.

### 1.4 Non-procurement in A
A's ERP lines are all purchases from third parties (they are all labelled). A light regex guards tool-only lines
against obvious tax/interest/penalty/intercompany text. It fires on nothing in the kit; it is there for A2.

## 2. Company B – many ERPs and currencies

### 2.1 Currency
* `currency_rate` is "units of line currency per 1 USD" (brief). USD lines are already USD: their rate column is
  ignored (1,279 USD lines carry a rate around 17–19, which is the MXN rate of the JDE company, not a USD rate).
* **Sanity check on given rates**: 64 EUR/CHF/ARS lines in one JDE system carry rates 18–20× the FX table
  (consistent with a rate to the local MXN book currency). Options: trust the line rate (brief wording), or the FX
  table. **Chose: if the given rate differs from the table by more than ×3 / ÷3 it is rejected and the table rate
  is used** (`fx.sanity_band`), and the line is flagged `FX_RATE_REJECTED` in the audit file. Legitimate drift
  (ARS devaluation, daily vs monthly rates; worst ratio 0.45) stays inside the band.
* Missing rate (CAD, MMK, THB, HKD, RCN always; ~49% of lines) → FX table for the **invoice month**. If the invoice
  date is unparseable, outside 2019–2026, or implausible (>120 days after posting or >400 days before it), the
  **posting month** is used and the line is flagged. If the month is not in the table, the nearest month for that
  currency is used.

### 2.2 Dates
Posting dates are clean (2024). Invoice dates include 2018, 2020–2022 and 2027 values; due dates up to 2027.
Impossible dates are flagged and never drive FX. No line is dropped for a bad date.

### 2.3 What is not procurement spend (`NON_PROCUREMENT`)
Found in the seed labels (450 of 3,000 are excluded, all `NON_PROCUREMENT`) and confirmed on the full file:
* **Intercompany**: GL account descriptions `IC<nn>`, `IC Payables`, `I/C Payable`, `Intercompany - <country>`;
  vendor types `5406` and `2742` occur only on intercompany accounts. → excluded.
* **Tax lines**: descriptions that are the tax itself – `State & Local Taxes`, `Sales Tax (Expense)`,
  `GST Input Tax`, `Accrued Sales & Use Tax`. → excluded. Tax *services* (tax consultants, "assistance with property
  tax") and customs duties stay as spend: B's tree has a "Customs Clearance & Duties" leaf, so duties are treated as
  procurement of a logistics service.
* Generic finance patterns (interest, bank charges, royalties, dividends, revenue share, partner billing) are in
  the config for B2 even though the kit has none.
* Result on the 3,000 seeds: exclusion agreement 100% (precision 1.0, recall 1.0). The rules are generic patterns
  (intercompany, tax), not a lookup of the seed lines.

### 2.4 Status (B has none)
B has no status column; `paid_date` is blank on 25% of lines. Options: treat unpaid as not approved, or treat
posted AP lines as approved. **Chose: posted AP invoice lines are approved** – an invoice posted to AP has passed
approval; unpaid only means not yet due/paid. The seeds agree (no `NOT_APPROVED` among them).

### 2.5 Duplicates
613 lines are exact copies of another line (every column equal except `line_id`; 512 in one JDE system).
Same invoice + same invoice line number + same amount cannot be two purchases → `DUPLICATE` (first copy counted).
None of the 3,000 seeds are such copies, consistent with copies injected after labelling.
Lines sharing invoice/line number but differing in amount or account (~1,000) are kept: in JDE one invoice line
can be split across GL distributions.

### 2.6 Zero and negative amounts
918 zero-amount lines are kept as included with spend 0 (the seeds label them `include=1`). Negative lines (833,
credit notes / adjustments) are kept and net against spend (the seeds include them).

### 2.7 Blank descriptions
41% of B lines have no description (blank or a single space). They are still classified, from the vendor's other
lines, the vendor's label history, the GL account and the PO type. Their confidence is lower. The seeds contain no
included blank-description lines, so accuracy on these cannot be measured; I report it as a limitation.

### 2.8 Suppliers in B
No names, only vendor IDs (`V5xxxxx`), which look global across the 8 ERPs (one numbering scheme). → supplier key =
vendor ID (`supplier_id_global: true`). Without names, two IDs of the same real supplier cannot be merged.

## 3. Supplier de-duplication (generic)
* Normalisation: ASCII fold, upper case, `&`/`+` → AND, OCR digit/letter repair inside words, legal suffixes stripped
  (PTY LTD, PVT LTD, LLC, GMBH, SA DE CV, ... list in code, overridable in config), then a **compact key without
  spaces** (the tool file splits names: "KEL MONT" vs "KELMONT").
* Union-find over: same compact key; same supplier id (per source unless ids are global); same tax id (GSTIN/VAT/PAN
  when a column exists); fuzzy match inside blocks (first 3 letters / first token) with ratio ≥92 *and* token-set
  ratio ≥87; names under 5 letters need ≥97. Conservative on purpose: merging two real suppliers is worse than
  leaving two spellings apart.
* **A only – learned memory.** The client's golden key in `labels_A.csv` groups 122 different supplier names under one
  key; 99% of that key's lines have a blank vendor ID (an "un-mastered vendor" bucket). No generic rule could recover
  that, so for A (and A2) the tool learns `name → golden key` and `vendor-id (incl. blank) → golden key` from the
  labels and applies it after clustering; tool-file spellings inherit the key through their cluster. New suppliers
  in A2 get our own cluster key.

## 4. Item de-duplication (generic)
* Item code (material number) groups lines; codes with the same normalised description are the same item (the
  golden item keys do exactly this: several material numbers share one key when their text is identical).
* Normalised description: OCR fold, unit spacing (`2 IN` = `2IN`, `20 MM` = `20MM`), punctuation removed; fuzzy ≥95
  inside blocks only when all numbers agree (sizes/part numbers must match exactly).
* Placeholder codes used with many unrelated descriptions (>25, e.g. a "dummy" material number used for free-text
  purchases) never glue items together.
* Generic texts ("No Object", "Parts", blank) cannot identify an item: key = supplier + code.
* Tool lines that duplicate an ERP line inherit that line's item key; A learns `code → golden item` from labels.
* The golden `item_key` is blank for lines without a material number; for those lines I still output a text-based key
  (assumed to be ignored by the scorer).

## 5. Classification
* **Unit of classification**: identical (normalised description, supplier, account, context, item code) lines are
  classified once (A: 42k lines → 7.6k units; B: 38k → 17.7k).
* **Layers** (first confident one wins): memory of labels → supervised TF-IDF + logistic regression (A/B only) →
  LLM → offline retrieval + propagation. Threshold for calling the LLM in labelled runs: supervised confidence < 0.5.
  Measured on held-out suppliers, supervised predictions at ≥0.5 confidence were ~99% right and below 0.5 ~20% right
  (A and B both), so 0.5 is where the LLM adds value.
* **Unseen tree (C)**: the LLM gets the whole tree (leaf IDs + paths) in a cached system prompt, plus per-leaf scope
  notes written once per tree by Claude Opus from the tree and a sample of the company's own **unlabelled** lines, plus
  per-supplier context. The JSON schema allows only valid leaf IDs, so an invalid category is impossible.
* **Supplier profiles**: A's tree is organised by equipment fleet / supplier specialisation (75% of A's suppliers map
  to a single L3), so each supplier first gets a "profile" category (from label history when labels exist,
  otherwise one LLM decision from its 15 biggest lines). Line classification sees it and follows it when the line is
  ambiguous. On a 300-unit zero-shot sample of A this lifted L1 from 78% to 85% (and scope notes to 90%).
* **Temperature 0**: current Claude models reject sampling parameters, so determinism comes from the on-disk cache
  keyed by a hash of (model, prompt version, tree, line payload). Gemini is called with temperature 0.
* **Item codes are not shown to the LLM** (they are opaque IDs) so identical lines share one LLM decision.
* **B seed labels are "useful, not perfect"**: they train the supervised layer with a quality factor of 0.92 on
  memory confidence, and they are the yardstick for B metrics (so B numbers inherit their noise).

## 6. LLM use, cost and robustness
* Primary: Claude Sonnet 5.5 (effort low) for supplier profiles and lines. Judge: Claude Opus 5.5 for scope notes,
  column mapping in `init-config`, and lines where the two vendors disagree. Second opinion / failover: Gemini 2.5
  Flash. The key in this project is free tier: it hits quota after a few calls, so it votes on at most 400 of the
  least certain units and a circuit breaker switches it off for the run on repeated quota errors (decided with the
  owner). Embeddings for kNN hints use Gemini when available, otherwise LSA (TF-IDF + SVD).
* Batching (25 lines per request, lines of one supplier together, supplier context sent once), prompt caching for
  the tree, disk cache, retries with backoff, a per-run USD cap (`cost_cap_usd`), and an offline fallback: with no key
  or after the cap, lines get retrieval + propagation and the run still writes a valid file.
* Keys are read from the environment or a git-ignored `.env`; never from config, never logged.

## 7. Output and validation
Every run validates before it writes: exact template columns and order; one row per input record; no duplicate
`record_id`; `include` ∈ {0,1}; codes from the brief's list; spend 0 when excluded; every included row's category
path exists in the tree (exact names) and levels below the tree depth are blank; confidence in [0,1];
included + excluded spend = total input spend. Any failure stops the run with a message (exit code 2).
An `_audit.csv` (rule, matched record, flags, FX source, classification layer) and a `_report.json` sit next to each
output.

## 8. Assumptions for the blind test
* A2/B2 use the same layouts; file names may differ slightly (configs accept glob patterns). The tree, labels and FX
  table are taken from the A2/B2 folder if present, else from the original kit folder (`search_dirs`).
* For A2/B2 the labelled A/B data is loaded as training rows only (never output).
* C: one invoice-line file + tree (≤4 levels), no labels, reporting currency USD (brief §4). `init-config` drafts the
  config; I review it (config only), run, validate, and push.
