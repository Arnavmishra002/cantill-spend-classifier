# APPROACH.md

## Pipeline (one command per company, config-driven)
`ingest` (files/columns from YAML, glob names, CSV/XLSX, encoding + delimiter + header-row detection) →
`clean` (numbers incl. `1,00,000` / `1.234,56` / symbols, rates `15|0.15|15%`, any date layout, VAT removal for
gross sources, FX: line rate → sanity check → monthly table) → `spend_rules` (status map, non-procurement rules,
exact and near duplicates; one reason code per exclusion, fixed precedence) → `match` (count once across two
systems) → `dedupe` (suppliers, items) → `classify` → `export` (exact template) → `validate` (fails loudly).
Nothing company-specific is in code: a new company is a new YAML, drafted by `run.py init-config`.

## Matching and de-duplication
* **Two systems (A)**: tool lines that are the same invoice "exported twice" are clustered first (normalised invoice
  no. + net amount + quantity + OCR-normalised text). Each cluster is matched one-to-one to an ERP receipt in
  passes of falling strictness: PO + amount → supplier + amount + text (no usable PO) → PO + unit price + quantity
  within 6% → PO aggregate (one invoice covers several receipts). Matched → `DUPLICATE` (ERP record counted);
  unmatched → tool-only spend, copies `DUPLICATE`.
* **Suppliers**: legal-suffix stripping, `&`→AND, OCR repair, space-insensitive key, union-find over name key /
  vendor id / tax id / blocked fuzzy matching (rapidfuzz, conservative thresholds). With labels (A) the client's
  golden keys are learned (name → key, vendor id → key) and spread through clusters.
* **Items**: item code ∪ normalised description (units, OCR, punctuation), fuzzy only when all numbers agree;
  placeholder codes and generic texts never merge items; matched tool lines inherit the ERP item.

## Classifying into a tree the tool has never seen
Lines are collapsed into ~5–10× fewer **units** (same text, supplier, account, context). Layers:
1. **Memory / supervised** (only when labels exist): exact label memory, then TF-IDF (word + char n-grams) +
   logistic regression. Measured: predictions with p ≥ 0.5 are ~99% correct on held-out suppliers, so only
   p < 0.5 goes on.
2. **LLM, zero-shot capable** (Claude Sonnet 5.5): the whole tree with leaf IDs in a cached system prompt;
   **scope notes per leaf** written once by Claude Opus 5.5 from the tree + a sample of the company's own
   unlabelled lines (teaches the company's abbreviations); a **supplier profile** per supplier (from label history,
   or one LLM decision from its biggest lines), because many trees are organised by supplier/equipment; nearest
   labelled examples (FAISS) and model suggestions as hints when labels exist. JSON-schema output restricted to
   valid leaf IDs; 25 lines per call, grouped by supplier; on-disk cache keyed by input hash.
3. **Second opinion**: Gemini 2.5 Flash votes on the least certain lines; agreement raises confidence,
   disagreement goes to Claude Opus 5.5 as judge (free-tier Gemini is capped and has a circuit breaker).
4. **Offline fallback** (no key / cost cap hit): TF-IDF retrieval against tree paths + propagation of each
   supplier's / account's category mix to blank lines. A run always finishes with a valid file.
Every included line gets a leaf (deepest level) and a confidence; confidence bands are checked against accuracy.

## Measured accuracy (labels hidden unless stated)
METRICS_TABLE

## Runtime and cost
RUNTIME_COST

## AI tools used
* **In the tool**: Anthropic Claude API – Sonnet 5.5 (line classification, supplier profiles), Opus 5.5 (tree scope
  notes, `init-config` column mapping and company context, judge on disagreements); Google Gemini API –
  gemini-2.5-flash (second opinion, failover), gemini-embedding-001 (optional embeddings; default is local LSA).
  Keys come from the environment / git-ignored `.env`; without keys the tool runs offline.
* **To build it**: Claude Code (Claude Opus 5.5) as a pair-programmer for code, tests and docs, under my review.
  Data was never pasted into a chatbot; the tool runs unattended.

## Limitations / what I would do next
LIMITATIONS
