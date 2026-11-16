# DECISIONS.md

What I found in the data, what was unclear, the options I considered, what I chose and why. Counts refer to the
kit (A: 37,029 ERP lines and 28,520 tool lines; B: 41,028 lines). No raw data rows are reproduced; examples are
format patterns with digits masked. Every rule is a config value unless it is described as generic code.

## 0. Ground rules

| Topic | Decision | Why |
|---|---|---|
| Company specifics | Only in `configs/*.yaml`: file names, columns, formats, rules, thresholds. The package has no column names, company names or category names. | Brief sections 3.1 and 3.7. Company C must need a config, not code. |
| Brief against my own instructions | Where they differ, the brief wins. The brief asks for `__A.csv` and `__B.csv` in the repository at the build deadline, so they are committed under `submission/`. | The brief is the source of truth. |
| Excluded rows | One row per input record. An excluded row has `include=0`, a code, `spend=0`, blank l1 to l4 (as in the template's example row), its supplier and item keys, and as confidence the confidence of the exclusion decision. | Matches `output_template.csv`. |
| Lines that are hard to classify | Every included line still gets a leaf (deepest level); the confidence carries the doubt. | Scoring is accuracy per level. A guessed leaf can be right; a blank level cannot. |

## 1. Company A: two systems

### 1.1 What is spend in each system
* ERP goods receipts (`value_zar`) are net of VAT and already in ZAR, also on the 15 lines whose
  `document_currency` is EUR or CHF (that column is the PO currency). Config: `amount_type: net`, currency ZAR.
* ERP negative lines (434) are goods-receipt reversals: quantity minus one, same text, same PO. Options: exclude
  the pair as `CANCELLED`, or keep both so they net to zero. Chose to keep them. Net spend is the same either way,
  and `labels_A.csv` labels every ERP line without any include column, which says every ERP record is counted.
* Tool amounts are gross. The VAT column has five spellings: `15`, `0.15`, `15%`, blank and `0`. Blank means 15%
  (brief). `0` means no VAT: tool lines with VAT `0` match an ERP line exactly on the gross amount, at the same rate
  as the other lines match after VAT is removed.
* 7,920 tool amounts use thousands separators. The generic number parser also reads `1.234,56`, `1,00,000`,
  currency symbols, `(123.45)` and a trailing minus.
* Tool invoice dates come in three layouts and are blank on about 65% of lines. A's spend does not need them; they
  only break ties in matching.

### 1.2 Status
Paid and Approved are counted. Cancelled is `CANCELLED`. Rejected and On Hold are `NOT_APPROVED`. Any value not in
the map is `NOT_APPROVED` (brief: only approved or paid count) and is logged. When a line qualifies twice the order
is `CANCELLED`, `NOT_APPROVED`, `NON_PROCUREMENT`, `DUPLICATE`. A cancelled tool line that also matches an ERP
receipt is reported as `CANCELLED`; the ERP record is counted anyway. Only 10 such lines exist.

### 1.3 Matching the two systems
Unclear: how to recognise the same purchase when PO references are reformatted or missing and invoice numbers are
written differently. The PO reference appears in five layouts (`45100xxxxx`, `PO#45100xxxxx`, `PO 45100xxxxx`,
`45100xxxxx/10.0`, blank on 18%). Invoice numbers carry prefixes such as `INV/`, `SI-`, `TAX INV` and `IN`. After
VAT is removed, a duplicate's amount equals the ERP value to the cent (median ratio 1.000000).

Method (`spendcube/match.py`, thresholds in the config):
1. Copies inside the tool file ("exported twice"): same normalised invoice number, same net amount, same quantity
   and near-identical text after OCR repair form one cluster, that is one invoice line. 2,342 lines sit in exact
   pairs that differ only in formatting; their statuses always agree.
2. Each cluster is matched one to one to an ERP receipt, strictest evidence first. An ERP line can take only one
   cluster.
   * P1: same PO (and PO item when both have one) and net amount within 0.05 or 0.05%. About 15,600 clusters.
   * P2: no usable PO (blank, or a PO the ERP has never seen): same supplier, same amount, similar text, and dates
     within 400 days when both are known. About 5,000 clusters, mostly tool lines with a blank PO reference.
   * P3: same PO, same unit price within 0.5%, quantity within 6%, similar text. 133 clusters, where only a
     weighbridge-type quantity differs.
   * P4: one invoice line covering several receipts of one PO at one unit price (subset sum). A handful.
3. A matched cluster makes every tool row `DUPLICATE`. In an unmatched cluster the first row is counted and the
   other copies are `DUPLICATE`.

Rejected: matching on amount alone (recurring identical charges would collide) and many-to-one matching without the
same unit price (too loose). About 690 unmatched tool lines share a PO with ERP lines but invoice a quantity that no
free receipt has (partial invoicing, or deliveries missing from the ERP extract). I count them as tool-only spend
rather than guess. The supplier test inside matching uses the generic name and ID clusters, not the client's golden
keys, so the client's catch-all key (1.5 below) cannot make two different firms look like one supplier.

### 1.4 Non-procurement in A
A's ERP lines are all purchases from third parties (all of them are labelled). A light pattern guards tool-only
lines against tax, interest, penalty or intercompany text. It fires on nothing in the kit; it is there for A2.

### 1.5 Supplier keys in A
The client's golden key groups 122 different supplier names under one key, and 99% of that key's lines have a blank
vendor ID: an "un-mastered vendor" bucket. No generic rule can recover that. For A and A2 the tool learns name to
golden key and vendor ID (blank included) to golden key from the labels and applies them after clustering. Tool-file
spellings inherit the key through their cluster. New suppliers in A2 get the tool's own cluster key.

### 1.6 Item keys in A
In `labels_A.csv` 43% of ERP lines (15,816) have a blank golden `item_key`, and they are exactly the lines without
a material number: the client does not treat a free-text or service line as an item. My first version gave those
lines text-based keys. Scored pairwise over all labelled lines, with a blank key counted as one group as a simple
grouping check would count it, item F1 was 0.10 (recall 0.05). Options: keep text keys and assume blanks are
ignored, or follow the client's convention. Chose to follow it. The tool learns from the labels whether lines without
an item code carry a blank golden key (it requires at least 95% of them; A has 100%) and then outputs a blank
`item_key` for such lines. A line matched to another record takes that record's final key, blank included. On the A2
simulation pairwise item F1 is 0.994 and B-cubed F1 on lines with a golden key is 0.976. Without labels (C) the rule
is off and every line gets a key.

## 2. Company B: many ERPs and currencies

### 2.1 Currency
* `currency_rate` means units of line currency per 1 USD (brief). USD lines are already in USD, so their rate column
  is ignored: 1,279 USD lines carry a rate near 17 to 19, which is the MXN rate of the JDE company.
* 60 EUR and CHF lines in one JDE system carry rates 18 to 20 times the FX table, consistent with a rate to the
  local MXN book currency. Options: trust the line rate (brief wording) or the table. Chose: a line rate more than
  three times away from the reference is rejected, the reference is used, and the line is flagged
  `FX_RATE_REJECTED`. Legitimate drift (ARS devaluation, daily against monthly rates; worst ratio 0.45) stays inside
  that band.
* A rate on the inverse base (USD per unit instead of units per USD) is detected by testing both readings against
  the reference: when the reciprocal is within 25% of the reference and clearly closer than the rate itself, the
  rate is inverted and flagged `FX_RATE_INVERTED`. This also catches currencies near parity, where a wide band alone
  would not.
* About 49% of lines have no rate (CAD, MMK, THB, HKD and RCN never have one). They use the FX table for the invoice
  month. If the invoice date is unparseable, outside 2019 to 2026, or implausible (more than 120 days after posting or
  more than 400 days before it), the posting month is used and the line is flagged. A month missing from the table
  takes the nearest month for that currency.

### 2.2 FX when no usable rate exists (generic)
The repository ships a small public table, `spendcube/data/fx_monthly_usd.csv`: ECB monthly average reference
rates for 2018 to October 2026, crossed through EUR into units per USD, plus official fixed pegs (for example SAR,
AED, XOF). It is not the kit's file. The order of sources is: the line's own rate (checked), the client's table, the
public table, then the median of the file's own accepted line rates for that currency. A client table is checked
against the public rates; a table on the inverse base is inverted with a warning. The ECB-derived rates agree with
B's kit table (median ratio 1.000), and B's totals are unchanged by the new chain. Only a currency unknown to every
source would get spend 0, and it is flagged `NO_FX_RATE` with a warning in the log.

### 2.3 Dates
Posting dates are clean (2024). Invoice dates include 2018, 2020 to 2022 and 2027 values. Impossible dates are
flagged and never drive FX. No line is dropped for a bad date.

### 2.4 What is not procurement spend
Found in the seed labels (450 of 3,000 are excluded, all as `NON_PROCUREMENT`) and confirmed on the full file:
* Intercompany: GL account descriptions `IC<nn>`, `IC Payables`, `I/C Payable`, `Intercompany - <country>`. Vendor
  types `5406` and `2742` occur only on intercompany accounts.
* Tax lines: descriptions that are the tax itself, such as `State & Local Taxes`, `Sales Tax Expense`,
  `GST Input Tax`, `Accrued Sales & Use Tax`. Tax services (tax consultants, help with property tax) and customs
  duties stay as spend: B's tree has a "Customs Clearance & Duties" leaf, so duties are a logistics purchase.
* Generic finance patterns (interest, bank charges, royalties, dividends, revenue share, partner billing) are in the
  config for B2 although the kit has none.
* On the 3,000 seeds the exclusions agree 100% (precision 1.0, recall 1.0). The rules are generic patterns, not a
  lookup of seed lines.

### 2.5 Status (B has none)
B has no status column, and `paid_date` is blank on 25% of lines. Options: treat unpaid as not approved, or treat
posted AP lines as approved. Chose the second: an invoice posted to AP has passed approval; unpaid only means not yet
paid. The seeds agree (none is `NOT_APPROVED`).

### 2.6 Duplicates
613 lines are exact copies of another line (every column equal except `line_id`; 512 in one JDE system). The same
invoice, invoice line number and amount cannot be two purchases, so the copies are `DUPLICATE`. None of the seeds is
such a copy, which fits copies added after labelling. About 1,000 lines share an invoice and line number but differ in
amount or account; they are kept, because in JDE one invoice line can be split across GL distributions.

### 2.7 Zero and negative amounts
918 zero-amount lines are included with spend 0 (the seeds include them). 833 negative lines (credit notes and
adjustments) are included and net against spend (the seeds include them).

### 2.8 Blank descriptions
41% of B lines have no description. They are still classified from the vendor's other lines, the vendor's label
history, the GL account and the PO type, with lower confidence. The seeds contain no included blank-description
line, so accuracy on these lines cannot be measured. I report this as a limitation.

### 2.9 Suppliers in B
B has no names, only vendor IDs (`V5xxxxx`), which follow one numbering scheme across the 8 ERPs. Supplier key =
vendor ID (`supplier_id_global: true`). Without names, two IDs of one real supplier cannot be merged.

## 3. Supplier de-duplication (generic)
* Normalisation: ASCII fold, upper case, `&` and `+` read as AND, OCR repair of digits inside words, legal suffixes
  removed (PTY LTD, PVT LTD, LLC, GMBH, SA DE CV and others; the list can be replaced in the config), then a key
  without spaces, because the tool file splits names ("KEL MONT" and "KELMONT").
* Union-find over the same name key, the same supplier ID (per source unless IDs are global), the same tax ID when
  a column exists, and fuzzy matches inside blocks (same first three letters or first word) with ratio at least 92
  and token-set ratio at least 87; names under five letters need 97. Merging two real suppliers is worse than leaving
  two spellings apart, so the thresholds are conservative.
* Hubs are never used to merge. A name linked to more than three supplier IDs (sundry, one-time vendor, tax office,
  cash) is split by ID instead of joining every supplier behind it; an ID or tax ID linked to more than three
  unrelated names does not link them. Hubs are logged and counted in the run report. In A, 19 names have two or three
  vendor IDs (one firm with several vendor records) and no ID has two names, so A is unchanged.
* Measured on A without labels: B-cubed precision 1.00 and recall 1.00 on all named suppliers, including the tool
  file's spelling variants. Only the client's 122-name bucket (1.5) is out of reach without labels.

## 4. Item de-duplication (generic)
* The item code (material number) groups lines, and codes with the same normalised description are one item; the
  golden keys do the same.
* Normalised description: OCR repair, unit spacing (`2 IN` equals `2IN`), punctuation removed. Fuzzy matches (at
  least 95) only inside blocks and only when all numbers agree, so sizes and part numbers must match exactly.
* Placeholder codes used with more than 25 unrelated descriptions never join items.
* Generic texts ("No Object", "Parts", blank) cannot identify an item; the key is supplier plus code.
* For blank keys learned from labels, see 1.6.

## 5. Classification

### 5.1 Layers
* Identical lines (same normalised description, supplier, account, context and item code) are classified once.
  A: 42,000 lines become 7,600 units; B: 38,000 become 17,700.
* The first confident layer wins: memory of labels, then a supervised TF-IDF plus logistic regression model (only
  with labels), then the LLM, then offline retrieval with propagation, then group consistency.
* In labelled runs the LLM is called only below a supervised-confidence threshold: A 0.3, B 0.5 (see 5.3). On
  held-out suppliers, supervised predictions at 0.5 or above were about 99% right and below 0.5 about 20% right.

### 5.2 An unseen tree (C)
* Claude sees the whole tree (leaf IDs and paths) in a cached system prompt, plus a scope note per leaf, written once
  per tree by Claude Opus from the tree and a sample of the company's own unlabelled lines. A tree file that has its
  own descriptions replaces the notes.
* Each supplier first gets a profile category: from label history when labels exist, otherwise one LLM decision
  from its 15 biggest lines. Lines see their supplier's profile and other lines, which matters because many clients
  file spend by supplier or by equipment fleet (75% of A's suppliers map to a single L3).
* The JSON schema allows only valid leaf IDs, so an invalid category is impossible.
* Roles: Claude Opus 5.5 classifies every unit; Claude Sonnet 5.5 gives an independent second opinion on uncertain
  units and on the units that hold the top half of spend; when they disagree, Claude Opus 5.5 sees both answers and
  the supplier context and decides. Sonnet takes over if Opus is unavailable.
* Current Claude models reject temperature settings, so runs are made deterministic by the on-disk cache, keyed by
  a hash of the model, prompt version, tree and line payload.
* Item codes are not shown to the LLM (they are opaque IDs), so identical lines share one decision.
* B's seed labels are described as useful but not perfect. Their memory confidence is scaled by 0.92, and B's
  metrics inherit their noise.

### 5.3 Decisions made from measurements
| Question | Options | Choice | Evidence |
|---|---|---|---|
| When to call the LLM in labelled runs | always, or below a supervised-confidence threshold | A below 0.3, B below 0.5 | A2 simulation: the LLM beats supervised only below 0.3 (56% against 17% at L3). B2 simulation: the LLM lifts L3 from 82% to 91% at 0.5. |
| Supplier profiles | none, from labels, from the LLM | labels when the supplier has labels, else the LLM | A zero-shot sample: L1 from 78% to 85% |
| Scope notes for an unseen tree | none, or written once per tree | written by Opus from the tree and unlabelled lines | A zero-shot sample: L1 from 85% to 90%, L3 from 67% to 74% (line-weighted) |
| Model roles | Sonnet or Opus as primary | Opus primary, Sonnet second opinion, Opus judge (owner's decision) | On a sample Opus was 1 to 3 points better than Sonnet at about twice the cost |
| LLM vendor | Claude only, or Claude with Gemini | Claude only (owner's decision) | The free-tier Gemini key returned quota errors after a few calls |
| Vectors for nearest-neighbour hints | API embeddings or local LSA | local LSA (TF-IDF reduced with SVD), indexed with FAISS | no API dependency, deterministic |
| Group consistency and general leaves | off, or on | on (see 5.4) | A with labels hidden: L3 from 69.4% to 71.6% of spend |

### 5.4 Top confusions on an unseen tree, and the generic fixes
With Claude Opus as primary and A's labels hidden, I listed the ten largest L3 confusions by spend. Three generic
causes explain them:
1. General leaves. A leaf that repeats its parent's name ("Continuous Miners > Continuous Miners") is the branch's
   general bucket. In every A branch that has one, the client files most of the branch's spend there (48% to 99.9%),
   while the model picks a specific sibling whenever the text names one ("build seal" lines go to "Seal Building";
   the client files them under the general leaf). About 9% of A's spend sat in such confusions. The old prompt
   collapsed "X > X" into "X", so the model could not even see the general leaf. Fixes: the prompt and the scope
   notes now show full paths and mark general leaves, with a rule to prefer them unless a line clearly names a
   specific sibling; and a specific leaf picked with confidence below 0.6 falls back to its branch's general leaf.
   The marker alone did not change the model's choices; the fallback lifted L3 from 69.4% to 71.1% of spend.
2. Supplier and contract habits. Vague lines ("Mining services", "General mechanical") are filed by the client
   according to the supplier's or contract's main category. Group consistency handles this: a line below 0.7
   confidence takes the dominant category of its group's confident lines (0.8 or above) when that category holds at
   least 70% of the group's confident spend and the group has at least 3 confident lines. Groups are the supplier and,
   for A, the contract number. A first version with a 0.85 threshold changed lines that were right only 31% of the
   time; the re-tuned version, together with the fallback, takes L3 from 69.4% to 71.6% of spend (62.3% to 67.4% of
   lines).
3. Sibling ambiguity the text cannot settle ("mini rebuild" filed as a full rather than an intervention overhaul,
   rock bits filed under drill steel). No generic fix; these stay as errors.
Nothing in the code or prompts names an A category. The thresholds were measured on A, which is a risk for C; they
are config values and were kept moderate.

### 5.5 Confidence
The confidence is the probability that the chosen leaf (deepest level) is right.
* LLM layer: the model's own probability, raised when the second opinion agrees.
* Supervised layer: the logistic regression probability.
* Offline layers: raw scores (similarity to the tree, a supplier's category share) are not probabilities; on A and
  B the median raw score was 0.02 to 0.03 while 15% (A) and 36% (B) of leaves were right. `run.py calibrate` fits a
  monotone mapping (isotonic regression) from the top similarity to the probability of a right leaf, on A and B with
  labels hidden and each company weighted equally, and stores it in `spendcube/data/offline_calibration.json`. The
  median offline confidence becomes 0.16, with expected calibration error 0.066 (A) and 0.071 (B). Fitted on one
  company and tested on the other the error is 0.21 and 0.13, which is the honest expectation for an unseen company.

## 6. LLM use, cost and robustness
* Claude only. Every run starts with a health check of each model it will use and prints the result. If the LLM is
  enabled but no model answers, the run continues on the offline layers and the summary says so plainly;
  `--require-llm` stops the run with exit code 3 instead.
* Batches of 25 lines with one supplier's lines together and its context sent once, prompt caching for the tree,
  the disk cache, SDK retries with backoff, a per-run USD cap (`cost_cap_usd`), and the offline fallback, so a run
  always writes a valid file.
* The key is read from the environment or a git-ignored `.env`, never from config, never logged.

## 7. Inputs, output and validation
* A header-only input file writes a header-only output. A config column missing from the input prints one error line
  and exits with code 2. Files may be CSV (any common delimiter and encoding, title rows above the header) or Excel.
* Every run validates before it finishes: exact template columns and order; one row per input record; no duplicate
  `record_id`; `include` is 0 or 1; codes from the brief; spend 0 when excluded; each included row's category path
  exists in the tree with exact names; levels below the tree depth are blank; confidence within 0 and 1; included
  plus excluded spend equals total input spend. A failure stops the run with exit code 2.

## 8. Single-file companies (C): duplicates
With one invoice file there is no second system, but A's tool file shows how duplicates look in practice: the same
invoice line exported twice with different formatting. Drafted configs therefore turn on `duplicates.fuzzy`: same
normalised invoice number (and line number when the file has one), same net amount, same quantity, text similarity at
least 90 and a similar supplier make later copies `DUPLICATE`. Exact copies are always duplicates. On a disguised copy
of A's tool file this flags 2,095 lines, against about 2,350 copies expected from the cluster analysis.

## 9. Assumptions for the blind test
* A2 and B2 have the same layouts; file names may differ slightly (configs accept patterns). The tree, labels and
  FX table come from the A2 or B2 folder if present, else from the original kit folder.
* For A2 and B2 the labelled A and B data are loaded as training rows only, never output.
* A new drop is self-contained: a tool invoice in A2 duplicates an ERP receipt in A2, not one from A. The A2
  simulation splits records by PO number for the same reason.
* C: one invoice-line file and a tree of up to four levels, no labels, reporting currency USD (brief section 4).
  `init-config` drafts the config: column roles from headers and values, checked by Claude when the API is
  reachable and validated against parse rates; supplier ID synonyms (Supplier No., Vendor ID, Vendor #, Vendor Code);
  day or month first decided from the values; status map; tax, intercompany and finance patterns; suspicious GL
  accounts; FX direction; contract-like consistency groups; company context. `check-config` shows parse rates per
  field. I review the config, run, validate and push.
* If C has an amount column already in USD, it is used directly.
