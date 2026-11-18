## Pipeline
One command per company, driven by a YAML config; no company-specific code. The tool reads the files, cleans
amounts, tax and dates, converts currency, gives every exclusion a reason code, counts each purchase once across
two systems, builds supplier and item keys, classifies every included line and validates its output.
`init-config` drafts a new company's config.

## Matching and de-duplication
Copies of one tool invoice line are clustered, then matched one to one to an ERP receipt, strictest evidence first:
PO and amount; supplier, amount and text; PO, unit price and quantity; one invoice for several receipts.
Suppliers: legal suffixes stripped, OCR repair, union-find over name, ID and tax ID, cautious fuzzy matching;
catch-all names never link. Items: code joined with normalised text; fuzzy only when the numbers agree.

## Classifying into an unseen tree
Identical lines are classified once. With labels, label memory and a logistic regression take the confident lines.
The rest go to Claude with the whole tree, a scope note per leaf written from the company's unlabelled lines, and
a category profile per supplier. Opus takes the lines holding the top 95% of spend, Sonnet the rest; Sonnet gives
a second opinion on uncertain lines and Opus judges disagreements. Answers are limited to valid leaf IDs.
Without a key, offline retrieval classifies every line.

## Measured accuracy (labels hidden), by line / by spend
| Setting | L1 | L2 | L3 |
|---|---:|---:|---:|
| A, all 37,029 labelled lines | 87.9% / 93.0% | 76.0% / 82.8% | 67.4% / 71.6% |
| A, offline only (no key) | 42.7% / 42.9% | 18.1% / 22.5% | 15.1% / 15.3% |
| B, 2,550 seed lines | 94.3% / 98.5% | 93.4% / 93.1% | 88.6% / 92.6% |
| A2 simulation (a held-out slice of A) | 98.0% / 98.2% | 93.6% / 94.1% | 92.5% / 93.6% |
| B2 simulation (a held-out slice of B) | 93.5% / 98.0% | 92.8% / 97.7% | 89.3% / 96.9% |

On 5,000 public California purchase lines (4-level tree): L1 74.0%, L4 56.4% by line. Exclusions agree 100% with
B's 3,000 seeds. Suppliers on A without labels: precision and recall 1.00. Items on A2: pairwise F1 0.994. Most
L3 errors on A are client filing conventions the text cannot show.

## Runtime and cost
| Run | lines | time | LLM cost | per 1,000 lines |
|---|---:|---:|---:|---:|
| A, labels hidden | 65,549 | 8.6 min | $7.97 | $0.12 |
| B build output | 41,028 | 6.2 min | $8.18 | $0.20 |
| California, mostly unique text | 5,000 | 2.3 min | $5.94 | $1.19 |

Projected, not measured: 25,000 lines of unique text take 12 to 15 minutes and about 30 USD (limit 60 minutes).
A cost cap stops new calls before it is passed. Without a key a run is free.

## AI tools used
In the tool: the Anthropic Claude API only (Opus 5.5, Sonnet 5.5). To build it: Claude Code (Opus 5.5; Fable 5.2
for the final checks) as a pair programmer under my review. No data was pasted into a chatbot.
