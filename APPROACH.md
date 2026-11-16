# APPROACH.md

## Pipeline
One command per company, driven by a YAML config; no company names, columns or categories in code. `ingest` reads
any CSV or Excel layout and maps columns to canonical fields. `clean` parses amounts, tax rates and dates, removes
VAT from gross sources and converts currency (line rate checked against a reference, then the client's table, then
a shipped ECB monthly table). `spend_rules` applies status, non-procurement and duplicate rules, one reason code per
exclusion. `match` counts a purchase once across two systems. `dedupe` builds supplier and item keys. `classify`
assigns a leaf and a confidence. `export` writes the exact template and `validate` stops the run on any
inconsistency. `run.py init-config` drafts the config for a new company; `check-config` shows parse rates per field.

## Matching and de-duplication
Copies of one invoice line in the tool file are clustered first (normalised invoice number, net amount, quantity,
OCR-repaired text). Each cluster is matched one to one to an ERP receipt: PO and amount; supplier, amount and text
when there is no usable PO; PO, unit price and quantity within 6%; one invoice covering several receipts. Matched
tool rows are `DUPLICATE`. Suppliers: legal suffixes stripped, OCR repair, a space-free key, union-find over name,
supplier ID and tax ID, conservative fuzzy matching; hub names and IDs (sundry, one-time vendor) never link. Items:
item code joined with normalised description, fuzzy only when all numbers agree. With labels, the client's golden
supplier and item conventions are learned (including blank item keys for lines without a code).

## Classifying into a tree the tool has never seen
Identical lines are classified once. With labels, label memory and a TF-IDF logistic regression handle confident
lines (p of 0.5 or more was about 99% right on held-out suppliers). Everything else goes to Claude Opus 5.5 with the
whole tree in a cached system prompt, a scope note per leaf (written once by Opus from the tree and a sample of the
company's own unlabelled lines), and a profile category per supplier, because many trees are organised by supplier
or equipment. Output is JSON restricted to valid leaf IDs. Claude Sonnet 5.5 gives a second opinion on uncertain and
high-spend units; Opus judges disagreements. A specific leaf picked with low confidence falls back to its branch's
general leaf, and uncertain lines follow the clear majority of their supplier's or contract's confident lines.
Without a key, or past the cost cap, offline retrieval with calibrated confidence keeps the run going.

## Measured accuracy (labels hidden)

| Setting | L1 line / spend | L2 line / spend | L3 line / spend |
|---|---:|---:|---:|
| A, all 37,029 labelled lines, Claude | 87.9% / 93.0% | 76.0% / 82.8% | 67.4% / 71.6% |
| A, offline only | 48.2% / 53.9% | 19.1% / 25.2% | 16.9% / 21.0% |
| B, 2,550 included seed lines, Claude | 94.3% / 98.5% | 93.4% / 93.1% | 88.6% / 92.6% |
| A2 simulation (20% of POs held back, A labels for the rest) | 98.0% / 98.2% | 93.6% / 94.1% | 92.5% / 93.6% |
| B2 simulation (20% held back) | 93.5% / 98.0% | 92.8% / 97.7% | 89.3% / 96.9% |

Exclusions agree 100% with B's 3,000 seed labels. Supplier de-duplication on A without labels: B-cubed precision and
recall 1.00 on named suppliers. Items on the A2 simulation: pairwise F1 0.994. Lines with Claude confidence 0.95 or
more were right 94.5% of the time.

## Runtime and cost
| Run | lines | time | Claude cost | per 1,000 lines |
|---|---:|---:|---:|---:|
| A with labels hidden, every unit through Claude | 65,549 | 8.6 min | $7.97 | $0.19 |
| B build output | 41,028 | 11.7 min | $13.87 | $0.34 |
| Small 4-level company (fresh-clone test) | 3,000 | 4.1 min | $2.29 | $0.76 |
| Offline only | any | 0.5 to 2 min | $0 | $0 |

25,000 lines take about 10 minutes. Re-runs read a disk cache and cost almost nothing; a per-run cap stops spending.

## AI tools used
In the tool: the Anthropic Claude API only (Opus 5.5: classification, supplier profiles, scope notes, column mapping
and company context in `init-config`, judge; Sonnet 5.5: second opinion and fallback). The key comes from the
environment or a git-ignored `.env`; without it the tool runs offline and says so. To build it: Claude Code (Claude
Opus 5.5) as a pair programmer for code, tests and documents, under my review. No data was pasted into a chatbot.

## Limitations
Zero-shot L3 on A's equipment-based tree reaches 71.6% of spend; most remaining errors are the client's own filing
conventions, which the text cannot reveal. Thresholds were measured on two companies. B's blank-description lines
have no labels, so their accuracy is unknown. Details and the evidence for each decision are in DECISIONS.md.
