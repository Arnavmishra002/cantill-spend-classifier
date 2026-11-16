# APPROACH.md

## Pipeline
One command per company, driven by a YAML config; no company names, columns or categories in code. `ingest` reads
any CSV or Excel layout and maps columns to canonical fields. `clean` parses amounts, tax rates and dates, removes
VAT from gross sources and converts currency (line rate checked against a reference, then the client's table, then a
shipped ECB monthly table). `spend_rules` applies status, non-procurement and duplicate rules, one reason code per
exclusion. `match` counts a purchase once across two systems. `dedupe` builds supplier and item keys. `classify`
assigns a leaf and a confidence. `export` writes the exact template and `validate` stops the run on any
inconsistency. `run.py init-config` drafts a new company's config; `check-config` shows parse rates per field.

## Matching and de-duplication
Copies of one invoice line in the tool file are clustered first (normalised invoice number, net amount, quantity,
OCR-repaired text). Each cluster is matched one to one to an ERP receipt: PO and amount; supplier, amount and text
when there is no usable PO; PO, unit price and quantity within 6%; one invoice covering several receipts. Matched
tool rows are `DUPLICATE`. Suppliers: legal suffixes stripped, OCR repair, a space-free key, union-find over name,
supplier ID and tax ID, conservative fuzzy matching; hub names and IDs (sundry, one-time vendor) never link. Items:
item code joined with normalised description, fuzzy only when all numbers agree. With labels, the client's golden
supplier and item conventions are learned, including blank item keys for lines without a code.

## Classifying into a tree the tool has never seen
Identical lines are classified once. With labels, label memory and a TF-IDF logistic regression handle confident
lines (p of 0.5 or more was about 99% right on held-out suppliers). The rest go to Claude: Opus 5.5 for the units
that hold the top 95% of spend, Sonnet 5.5 for the tail. The whole tree sits in a cached system prompt with a scope
note per leaf, written once by Opus from the tree and a sample of the company's unlabelled lines, and each supplier
first gets a profile category, because many trees are organised by supplier or equipment. Output is JSON restricted
to valid leaf IDs. Sonnet gives a second opinion on uncertain and high-spend units; Opus judges disagreements. A
specific leaf picked with low confidence falls back to its branch's general leaf, and uncertain lines follow the
clear majority of their supplier's or contract's confident lines. Trees above 400 leaves use per-batch retrieval
shortlists instead of the full tree. Without a key, or past the cost cap, offline retrieval with calibrated
confidence classifies what is left.

## Measured accuracy (labels hidden)

| Setting | L1 line / spend | L2 line / spend | L3 line / spend |
|---|---:|---:|---:|
| A, all 37,029 labelled lines, Claude | 87.9% / 93.0% | 76.0% / 82.8% | 67.4% / 71.6% |
| A, offline only | 42.7% / 42.9% | 18.1% / 22.5% | 15.1% / 15.3% |
| B, 2,550 included seed lines, Claude | 94.3% / 98.5% | 93.4% / 93.1% | 88.6% / 92.6% |
| A2 simulation (20% of POs held back, A labels for the rest) | 98.0% / 98.2% | 93.6% / 94.1% | 92.5% / 93.6% |
| B2 simulation (20% held back) | 93.5% / 98.0% | 92.8% / 97.7% | 89.3% / 96.9% |
| Public California purchases, 2,000 lines, 4-level tree with 200 leaves, Claude | 74.9% / 96.7% | 70.8% / 96.6% | 67.2% / 96.5% |
| Same California lines, offline only | 51.4% / 63.9% | 45.8% / 63.5% | 40.6% / 62.2% |

California L4: 57.6% of lines with Claude, 33.6% offline. The top 1% of lines hold 22% of spend in A, 38% in B and
93% in the California sample, so there the line-weighted number is the better guide. The California data is the
public Purchase Order Data of the California Department of General Services (data.ca.gov, "No Restrictions on Public
Use"); `scripts/california_check.py` rebuilds the test. Exclusions agree 100% with B's 3,000 seed labels. Supplier
de-duplication on A without labels: precision and recall 1.00 on named suppliers. Items on the A2 simulation:
pairwise F1 0.994. Lines with Claude confidence 0.95 or more were right 94.5% of the time.

## Runtime and cost
| Run | lines | time | Claude cost | per 1,000 lines |
|---|---:|---:|---:|---:|
| A with labels hidden, every unit through Claude | 65,549 | 8.6 min | $7.97 | $0.19 |
| B build output | 41,028 | 11.7 min | $13.87 | $0.34 |
| California, almost every line unique, 200 leaves | 2,000 | 2.1 min | $3.27 | $1.64 |
| Offline only | any | 0.5 to 2 min | $0 | $0 |

25,000 lines take 10 to 26 minutes and 5 to 41 USD, depending on how many lines repeat; the limit is 60 minutes.
Re-runs read a disk cache and cost almost nothing, and no call starts if it could pass the per-run cap. Without an
API key the tool runs offline and reaches the offline numbers above.

## AI tools used
In the tool: the Anthropic Claude API only. Opus 5.5 classifies, profiles suppliers, writes scope notes, maps
columns and drafts the company context in `init-config`, and judges disagreements. Sonnet 5.5 gives second opinions,
classifies the low-spend tail and is the fallback. To build it: Claude Code (Claude Opus 5.5) as a pair programmer
for code, tests and documents, under my review. No data was pasted into a chatbot.

## Limitations
Zero-shot L3 on A's equipment-based tree reaches 71.6% of spend; most remaining errors are the client's filing
conventions, which the text cannot show. Thresholds were measured on two companies. B's blank-description lines have
no labels, so their accuracy is unknown. The evidence for each decision is in DECISIONS.md.
