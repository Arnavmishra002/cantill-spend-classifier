# APPROACH.md

## Pipeline
One command per company, driven by a YAML config; nothing company-specific is in code. The tool reads the files,
parses amounts, tax rates and dates, removes VAT, converts currency, applies status, non-procurement and duplicate
rules with one reason code each, counts each purchase once across two systems, builds supplier and item keys,
classifies every included line, writes the exact template and validates it. `init-config` drafts a new company's
config. Details are in README.md and DECISIONS.md.

## Matching and de-duplication
Copies of one invoice line in the tool file are clustered, then matched one to one to an ERP receipt: PO and amount;
supplier, amount and text without a usable PO; PO, unit price and quantity within 6%; one invoice for several
receipts. Suppliers: suffix stripping, OCR repair, a space-free key, union-find over name, ID and tax ID, cautious
fuzzy matching; hub names and IDs never link. Items: code joined with normalised text, fuzzy only when numbers agree.

## Classifying into an unseen tree
Identical lines are classified once. With labels, label memory and a logistic regression take the confident lines.
The rest go to Claude with the whole tree, a scope note per leaf written from the company's unlabelled lines, and a
category profile per supplier. Opus takes the units holding the top 95% of spend, Sonnet the rest; Sonnet gives a
second opinion on uncertain and high-spend units and Opus judges disagreements. Output is limited to valid leaf IDs.
Without a key, offline retrieval with calibrated confidence classifies every line. Vague text such as "Misc" or
an invoice number counts as no evidence and gets low confidence.

## Measured accuracy (labels hidden)
| Setting | L1 | L2 | L3 |
|---|---:|---:|---:|
| A, 37,029 labelled lines, by line / by spend | 87.9% / 93.0% | 76.0% / 82.8% | 67.4% / 71.6% |
| A, offline only, by line / by spend | 42.7% / 42.9% | 18.1% / 22.5% | 15.1% / 15.3% |
| B, 2,550 seed lines, by line / by spend | 94.3% / 98.5% | 93.4% / 93.1% | 88.6% / 92.6% |
| A2 simulation, by line / by spend | 98.0% / 98.2% | 93.6% / 94.1% | 92.5% / 93.6% |
| B2 simulation, by line / by spend | 93.5% / 98.0% | 92.8% / 97.7% | 89.3% / 96.9% |
| California, 5,000 lines, 4 levels, by line / by log spend | 74.0% / 77.1% | 69.9% / 73.2% | 66.3% / 69.6% |

California L4: 56.4% by line, 59.6% by log spend; offline on 2,000 lines it was 33.6%. The top 1% of California
lines hold 94% of spend, so plain spend weighting would rest on a few lines. Data: Purchase Order Data, California
Department of General Services, data.ca.gov ("No Restrictions on Public Use"); `scripts/california_check.py`.
Lines below confidence 0.7 form a review queue printed with every run: 3.7% of spend on A's build output, 48% on
B's (mostly lines without a description) and 1.7% on the California run; C's share is in its run summary. Every
included line gets a leaf, so coverage at level 2 or deeper is 100% of spend. Exclusions agree 100% with B's 3,000 seeds. Suppliers on A without labels: precision and recall 1.00. Items on the
A2 simulation: pairwise F1 0.994.

## Runtime and cost
| Run | lines | units | time | cost | per 1,000 lines |
|---|---:|---:|---:|---:|---:|
| A, labels hidden, repetitive text | 65,549 | 8,483 | 8.6 min | $7.97 | $0.19 |
| B build output | 41,028 | 15,100 | 11.7 min | $13.87 | $0.34 |
| California, mostly unique text | 5,000 | 4,528 | 2.3 min | $5.94 | $1.19 |
| Offline only | 25,362 | | 4 s | $0 | $0 |

Projection from the 5,000-line run, not measured: 25,000 lines of unique text take about 12 to 15 minutes and
about 30 USD; repetitive text costs less. The limit is 60 minutes. No call starts if it could pass the cost cap.

## AI tools used
In the tool: the Anthropic Claude API only (Opus 5.5 and Sonnet 5.5). To build it: Claude Code (Claude Opus 5.5) as
a pair programmer, under my review. No data was pasted into a chatbot.

## Limitations
Zero-shot L3 on A's equipment-based tree reaches 71.6% of spend; most errors are client filing conventions the text
cannot show. Thresholds were measured on two companies. B's blank-description lines have no labels to score.
