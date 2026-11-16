# DECISIONS.md

Each entry says what was unclear, what I chose and why. Counts come from the kit: A has 37,029 ERP lines and 28,520
tool lines, B has 41,028 lines. No data rows are reproduced. Every rule is a config value unless it says otherwise.

## General

1. Excluded rows. One row per input record. An excluded row has `include=0`, a code, `spend=0`, blank l1 to l4 as in
   the template's example row, its supplier and item keys, and the confidence of the exclusion.
2. Hard lines. Every included line gets a leaf at the deepest level, because a guessed leaf can be right and a
   blank level cannot. The confidence carries the doubt.
3. Confidence. It is the probability that the chosen leaf is right. Claude's own estimate on the LLM path; the
   logistic regression probability on the supervised path; on the offline path an isotonic mapping of the top
   similarity, fitted on A and B with labels hidden (median 0.16 instead of 0.02; error 0.07 when fitted on both
   companies, 0.13 to 0.21 across companies). Calibration only rescales the reported number; it never changes the
   chosen category.
4. Status. Paid, approved, posted and complete count. Cancelled, void and reversed are `CANCELLED`. Rejected, on
   hold, pending, draft, disputed and submitted are `NOT_APPROVED`. A status value the map does not know is
   `NOT_APPROVED`, because the brief says only approved or paid lines count, and the run summary lists it.
5. Precedence when a line qualifies twice: `CANCELLED`, `NOT_APPROVED`, `NON_PROCUREMENT`, `DUPLICATE`.
6. Dates. Any layout is parsed. A year outside 1900 to 2200 is flagged `IMPOSSIBLE_DATE` and treated as missing;
   no line is dropped for a bad date.

## Company A: two systems

7. ERP amounts. `value_zar` is net and already in ZAR, also on the 15 lines whose document currency is EUR or CHF
   (that column is the PO currency).
8. ERP reversals (434 negative lines). Kept, so they net to zero. The labels cover every ERP line and have no
   include column, so every ERP record counts.
9. Tool VAT. The rate is written as 15, 0.15, 15%, blank or 0. Blank means 15% (brief). 0 means no VAT: those lines
   match an ERP line on the gross amount.
10. Counting a purchase once. Copies of one invoice line inside the tool file are clustered first (same normalised
    invoice number, net amount, quantity and near-identical text). Each cluster is matched one to one to an ERP
    receipt, strictest evidence first: same PO and amount (15,643 clusters); same supplier, amount and text when the
    PO is blank or unknown, with dates within 400 days (5,011); same PO, unit price and a quantity within 6% (133);
    one invoice covering several receipts (4). Matched tool rows are `DUPLICATE`. In an unmatched cluster the first
    copy whose status counts is kept and the others are `DUPLICATE`. Rejected: matching on amount alone, because
    recurring identical charges would collide.
11. Spend found in only one system. 5,103 tool rows (ZAR 363.8 million) match no receipt and are counted. About 690
    of them share a PO with ERP lines but invoice a quantity no free receipt has; I count them rather than guess.
    The run summary prints this total.
12. Supplier keys. The client's golden key groups 122 names under one key, and 99% of its lines have a blank vendor
    ID. No generic rule finds that, so for A and A2 the tool learns name to key and vendor ID to key from the
    labels. Matching itself uses the generic name clusters, so that catch-all key cannot join two firms.
13. Item keys. 15,816 ERP lines (43%) have a blank golden item key, exactly the lines without a material number.
    The tool learns this from the labels (it requires at least 95% of no-code lines to be blank) and outputs a blank
    key for them. Pairwise item F1 over all labelled lines goes from 0.10 to 0.994 on the A2 simulation. The tool
    file has no item code, so a tool-only line in A2 also gets a blank key; that is an inference from the ERP lines,
    not something the labels show.
14. A new drop is self-contained. A tool invoice in A2 is matched only to receipts in A2.

## Company B: many ERPs and currencies

15. USD lines ignore their rate column. 1,279 of them carry about 17 to 19, which is the MXN book rate.
16. Line rates are checked against a reference rate. A rate more than three times away is replaced and flagged
    (59 lines). A rate on the inverse base is detected by comparing both readings and is inverted. A rate that
    matches the table for any month from the invoice date to the posting date is accepted, because a line can be
    booked at the posting month and a high-inflation currency such as ARS can move more than three times in that
    span. This changed one B line by 0.10 USD after the build output was committed.
17. Missing rates use the FX table for the invoice month, or the posting month when the invoice date is broken or
    implausible, or the nearest month in the table.
18. No status column. 10,322 lines have no paid date, and document types include CA, Z6, PD and PR. A line posted to
    AP has passed approval; unpaid only means not yet paid. None is excluded for status. The 3,000 seeds agree: none
    is `NOT_APPROVED`.
19. Non-procurement: intercompany accounts (`IC<nn>`, `I/C Payable`, `IC Payables`), the two vendor types used only
    on them, and lines that are a tax itself (sales tax, state and local taxes, GST input tax). Tax services and
    customs duties stay as spend, because B's tree has a customs leaf. Agreement with the seeds is 100%.
20. Duplicates. 613 lines repeat every column except the line ID and are `DUPLICATE`. About 1,000 lines share an
    invoice and line number but differ in amount or account; they are kept, since JDE splits one line across GL
    distributions.
21. Zero and negative lines are kept (918 and 833); the seeds include them.
22. Blank descriptions (41% of lines) are classified from the vendor's other lines, its label history and the
    account. The seeds have no such included line, so their accuracy is unknown.
23. B has vendor IDs and no names, so the supplier key is the vendor ID.

## Generic parts used for C

24. FX without a usable rate. The repository ships `spendcube/data/fx_monthly_usd.csv`: ECB monthly average rates
    (2018 to October 2026) crossed into units per USD, plus official fixed pegs. Order of sources: the line's own
    rate, the client's table, this table, then the median of the file's own rates. A client table on the inverse base
    is inverted. Only a currency unknown to every source gets spend 0, flagged `NO_FX_RATE`. A table month more
    than 95 days from the line's date is stale (the shipped table stops in 2020 for ARS): it never judges a line
    rate and fills a rate only after the file's own rates. `FX_PUBLIC_TABLE` marks only lines in another currency
    than the reporting one that took a public rate.
25. Supplier de-duplication: legal suffixes stripped, OCR repair, a key without spaces, union-find over name, ID and
    tax ID, fuzzy matching with conservative thresholds. A name with more than 3 IDs (sundry, one-time vendor, tax
    office) or an ID with more than 3 unrelated names never links suppliers. On A without labels, precision and
    recall are 1.00 on named suppliers.
26. Item de-duplication: item code joined with normalised description; fuzzy matching only when all numbers agree;
    placeholder codes and generic texts never merge items.
27. Single-file duplicates. A line exported twice with different formatting (same invoice and line number, amount,
    quantity and similar text and supplier) is `DUPLICATE`. On a disguised copy of A's tool file this finds 2,095 of
    about 2,350 expected copies.
28. init-config. Column roles come from headers and values and are checked by Claude when the API is reachable.
    Supplier No., Vendor ID, Vendor # and Vendor Code map to `supplier_id`. All item text columns are joined. The
    day and month order and the plausible year window come from the values. Statuses are mapped by keyword. A file
    is read as an FX table only when it has 5 columns or fewer, with currency, rate and month columns, and another
    data file exists. A file named with `--files`, or the only data file, is always read as transactions. If
    init-config still fails, `configs/company_c_template.yaml` is a complete config to fill in by hand.

## Classification

29. Models. Claude only, by the owner's decision. Opus 5.5 classifies the units that hold the top 95% of spend,
    Sonnet 5.5 the rest. Sonnet gives a second opinion on uncertain units and on the units holding the top half of
    spend; Opus judges disagreements. Opus was 1 to 3 points better than Sonnet on a sample at twice the cost.
30. When to call the LLM with labels. Below supervised confidence 0.3 for A and 0.5 for B. On the A2 simulation the
    LLM beats the supervised model only below 0.3; on the B2 simulation it lifts L3 from 82% to 91%.
31. Unseen tree. The whole tree goes in a cached prompt with a scope note per leaf, written by Opus from the tree and
    a sample of the company's unlabelled lines. On an A sample this lifted L1 from 85% to 90%. Each supplier gets a
    profile category first, because 75% of A's suppliers map to one L3; this lifted L1 from 78% to 85%.
32. Big trees. Above 400 leaves the full tree no longer fits a fast, cheap prompt, so each batch chooses from its
    lines' retrieval shortlists. On a 3,000-leaf UNSPSC sample the right leaf is in the top 30 for 64% of lines and
    L1 is right on 54% of lines, against 75% with a 200-leaf tree in full-tree mode.
33. The top 10 L3 confusions on A with labels hidden. First cause: a leaf that repeats its parent's name is the
    branch's general bucket, and the client files most of the branch's spend there (48% to 99.9%), while the model
    picks a specific sibling. A specific leaf picked with confidence below 0.6 now falls back to that general leaf.
    Second cause: vague lines are filed by supplier or contract. An uncertain line takes the dominant category of its
    group's confident lines. Together: L3 from 69.4% to 71.6% of spend. The rest are client conventions the text
    cannot show. Nothing in the code names an A category.
34. Group consistency on the offline path is off by default. With it, A offline L3 was 21.0% of spend instead of
    15.3%, but a fake hospital company fell from 82.8% to 74.2% (L4 from 85.4% to 66.6%), and A always runs with
    labels.
35. Label ceiling. 22% of A's rows have a text the client filed under more than one L3, so a text-only reader tops
    out at 94%; with the supplier it is 99%. B's seeds have none.
36. Spend weighting. The top 1% of lines hold 22% of spend in A, 38% in B and 93% in the California sample, so
    spend-weighted accuracy rests on a few lines. Line-weighted accuracy is always reported next to it.
37. Determinism. Claude rejects temperature settings, so a disk cache keyed by model, prompt, tree and line makes
    re-runs on the same machine identical. Item codes are not sent to the LLM, so identical lines share one answer.
38. Runtime and cost at C size. Units, not lines, drive cost. A's 65,549 lines collapse to 8,483 units; 5,000
    California lines are 4,528 units. That run took 137 s and 5.94 USD with 57 units on Opus and 4,471 on Sonnet;
    the 6 USD cap trimmed only second opinions. Drafted C configs use concurrency 16 and a 60 USD cap.
