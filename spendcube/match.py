"""Cross-source matching: find the same purchase in two systems and count it once.

Generic over any primary/secondary pair named in the config (for A: ERP goods
receipts = primary, AP invoice tool = secondary).

Steps
 1. Secondary self-duplicates ("invoice exported twice"): rows with the same
    normalised invoice number, same net amount, same quantity and near-identical
    text form one cluster = one real invoice line.
 2. Each cluster is matched one-to-one to a primary line, in passes of falling
    strictness (only primary lines not yet taken are eligible):
      P1  same PO (+ PO item when both have it) and net amount within tolerance
      P2  no usable PO: same supplier (normalised) and net amount within tolerance
          and similar text
      P3  same PO, same unit price, quantity within tolerance, similar text
          (weighbridge / partial-quantity differences)
 3. Matched clusters -> every row DUPLICATE (primary record is counted).
    Unmatched clusters -> first row counted, the others DUPLICATE.
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from .textnorm import compact_key, norm_key

log = logging.getLogger(__name__)


def _digits_or_alnum(s: str, pattern: str | None) -> str:
    if not s:
        return ""
    if pattern:
        m = re.search(pattern, s)
        return m.group(1) if m else ""
    return re.sub(r"[^0-9A-Za-z]", "", s).upper()


def norm_invoice_number(s: str, strip_prefixes: list[str]) -> str:
    u = (s or "").upper().strip()
    for p in sorted(strip_prefixes, key=len, reverse=True):
        if u.startswith(p.upper()):
            u = u[len(p):]
            break
    u = re.sub(r"[^0-9A-Z]", "", u)
    return u.lstrip("0") or u


def _po_fields(df: pd.DataFrame, mcfg: dict):
    pat = mcfg.get("po_pattern")  # e.g. '(\d{10})'
    item_pat = mcfg.get("po_item_pattern")  # e.g. '/(\d+)(?:\.0+)?$'
    po = pd.Series("", index=df.index)
    poi = pd.Series("", index=df.index)
    if "po_number" in df:
        po = df["po_number"].map(lambda s: _digits_or_alnum(s, pat))
    if "po_reference" in df:
        ref = df["po_reference"]
        po2 = ref.map(lambda s: _digits_or_alnum(s, pat))
        po = po.where(po != "", po2)
        if item_pat:
            poi2 = ref.map(lambda s: (re.search(item_pat, s or "") or [None, ""])[1] or "")
            poi = poi2
    if "po_item" in df:
        pi = df["po_item"].str.replace(r"\.0+$", "", regex=True).str.lstrip("0")
        poi = poi.where(poi != "", pi)
    poi = poi.astype(str).str.lstrip("0")
    return po, poi


def _subset_sum(ids, amounts, target, tol, max_parts):
    """Find a subset of amounts summing to target (within tol). Small n only.
    Tries 'all of them' first, then a bounded DFS over amounts sorted desc."""
    if abs(sum(amounts) - target) <= tol and len(ids) <= max_parts * 4:
        return list(ids)
    pairs = sorted(zip(amounts, ids), key=lambda x: -abs(x[0]))[:24]
    best = None
    steps = [0]

    def dfs(k, acc, chosen):
        nonlocal best
        steps[0] += 1
        if best is not None or steps[0] > 50_000:
            return
        if abs(acc - target) <= tol and chosen:
            best = list(chosen)
            return
        if k >= len(pairs) or len(chosen) >= max_parts:
            return
        a, i = pairs[k]
        if abs(acc + a) <= abs(target) + tol:
            chosen.append(i)
            dfs(k + 1, acc + a, chosen)
            chosen.pop()
        dfs(k + 1, acc, chosen)
    dfs(0, 0.0, [])
    return best


def _status_excluded(cands) -> bool:
    return any(code in ("CANCELLED", "NOT_APPROVED") for code, _ in (cands or []))


def _text_sim(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return max(fuzz.ratio(a, b), fuzz.token_set_ratio(a, b))


def match_sources(df: pd.DataFrame, cfg: dict, supplier_key: pd.Series | None = None) -> dict:
    """Mutates df: adds match columns and DUPLICATE candidate codes. Returns stats."""
    mcfg = cfg.get("cross_source", {})
    stats: dict = {}
    if not mcfg.get("enabled"):
        return stats
    P, S = mcfg["primary"], mcfg["secondary"]
    tol_abs = float(mcfg.get("amount_tolerance_abs", 0.05))
    tol_rel = float(mcfg.get("amount_tolerance_rel", 0.0005))
    text_min = float(mcfg.get("text_similarity_min", 80))
    qty_tol = float(mcfg.get("quantity_tolerance_rel", 0.06))
    price_tol = float(mcfg.get("unit_price_tolerance_rel", 0.005))
    prefixes = mcfg.get("invoice_prefixes", [])

    df["match_cluster"] = ""
    df["matched_to"] = ""
    df["match_pass"] = ""
    df["po_norm"], df["po_item_norm"] = _po_fields(df, mcfg)
    df["text_key"] = df["description"].map(norm_key)
    df["supp_ckey"] = (supplier_key if supplier_key is not None
                       else df["supplier_name"].map(compact_key))

    sec = df[df["source"] == S].copy()
    pri = df[df["source"] == P].copy()

    # ---------------------------------------------------------------- 1. clusters
    sec["inv_norm"] = sec.get("invoice_number", pd.Series("", index=sec.index)).map(
        lambda s: norm_invoice_number(s, prefixes))
    sec["amt_r"] = sec["net_amount"].round(2)
    sec["qty_r"] = sec["quantity_num"].round(4)
    cluster_of = {}
    cid = 0
    for _, g in sec.groupby(["inv_norm", "amt_r", "qty_r"], sort=False, dropna=False):
        idx = list(g.index)
        if len(idx) == 1 or g["inv_norm"].iloc[0] == "":
            for i in idx:
                cluster_of[i] = cid
                cid += 1
            continue
        # greedy grouping by text similarity inside the key group
        reps: list[tuple[int, str]] = []
        for i in idx:
            t = sec.at[i, "text_key"]
            for c, rt in reps:
                if _text_sim(t, rt) >= text_min:
                    cluster_of[i] = c
                    break
            else:
                cluster_of[i] = cid
                reps.append((cid, t))
                cid += 1
    sec["cluster"] = pd.Series(cluster_of)
    df.loc[sec.index, "match_cluster"] = "C" + sec["cluster"].astype(int).astype(str)

    # one representative row per cluster (prefer one with a PO, then first row)
    sec["has_po"] = sec["po_norm"] != ""
    clusters = (sec.sort_values(["cluster", "has_po", "row_order"], ascending=[True, False, True])
                .groupby("cluster", sort=False))
    rep = clusters.head(1).set_index("cluster")
    # union of POs / suppliers known for the cluster
    c_pos = clusters["po_norm"].agg(lambda s: {x for x in s if x})
    c_poi = clusters["po_item_norm"].agg(lambda s: {x for x in s if x})
    c_sup = clusters["supp_ckey"].agg(lambda s: {x for x in s if x})

    # ---------------------------------------------------------------- 2. matching
    pri_amt = pri["net_amount"].round(2)
    by_po_amt = defaultdict(list)
    by_po = defaultdict(list)
    by_sup_amt = defaultdict(list)
    for i, po, a, sk in zip(pri.index, pri["po_norm"], pri_amt, pri["supp_ckey"]):
        by_po_amt[(po, a)].append(i)
        by_po[po].append(i)
        by_sup_amt[(sk, a)].append(i)
    taken: set = set()
    matched: dict[int, tuple[int, str]] = {}

    def amt_ok(a, b):
        return abs(a - b) <= max(tol_abs, tol_rel * abs(b))

    def best(cands, c, need_text=False, po_items=None):
        r = rep.loc[c]
        best_i, best_s = None, -1.0
        for i in cands:
            if i in taken:
                continue
            s = _text_sim(r["text_key"], pri.at[i, "text_key"])
            if need_text and s < text_min:
                continue
            if po_items and pri.at[i, "po_item_norm"] in po_items:
                s += 50
            if s > best_s:
                best_i, best_s = i, s
        return best_i

    order = rep.sort_values("row_order").index
    # P1: PO + amount (allow +/- 1 cent neighbours)
    for c in order:
        a = rep.at[c, "amt_r"]
        if pd.isna(a):
            continue
        cands = []
        for po in c_pos.get(c, set()):
            for d in (0.0, -0.01, 0.01):
                cands += [i for i in by_po_amt.get((po, round(a + d, 2)), []) if amt_ok(a, pri_amt[i])]
        i = best(cands, c, po_items=c_poi.get(c))
        if i is not None:
            taken.add(i)
            matched[c] = (i, "P1_po_amount")
    # P2: supplier + amount + text (clusters without a PO, or PO not in primary).
    # When both dates are known they must be within max_date_gap_days.
    max_gap = float(mcfg.get("max_date_gap_days", 400))
    pdates = pri["date_parsed"] if "date_parsed" in pri else pd.Series(pd.NaT, index=pri.index)
    sdates = sec["date_parsed"] if "date_parsed" in sec else pd.Series(pd.NaT, index=sec.index)
    c_date = sdates.groupby(sec["cluster"]).first()

    def date_ok(c, i):
        a, b = c_date.get(c), pdates.get(i)
        if pd.isna(a) or pd.isna(b):
            return True
        return abs((a - b).days) <= max_gap

    for c in order:
        if c in matched:
            continue
        a = rep.at[c, "amt_r"]
        if pd.isna(a):
            continue
        pos = c_pos.get(c, set())
        if pos and any(po in by_po for po in pos):
            continue  # has a known PO: only P1/P3 may match it
        cands = []
        for sk in c_sup.get(c, set()):
            for d in (0.0, -0.01, 0.01):
                cands += [i for i in by_sup_amt.get((sk, round(a + d, 2)), [])
                          if amt_ok(a, pri_amt[i]) and date_ok(c, i)]
        i = best(cands, c, need_text=True)
        if i is not None:
            taken.add(i)
            matched[c] = (i, "P2_supplier_amount_text")
    # P3: same PO, same unit price, quantity close, text similar
    if mcfg.get("allow_quantity_variance", True):
        for c in order:
            if c in matched:
                continue
            r = rep.loc[c]
            a, q = r["amt_r"], r["qty_r"]
            if pd.isna(a) or pd.isna(q) or q == 0:
                continue
            up = a / q
            cands = []
            for po in c_pos.get(c, set()):
                for i in by_po.get(po, []):
                    if i in taken:
                        continue
                    pq = pri.at[i, "quantity_num"]
                    pa = pri_amt[i]
                    if not pq or pd.isna(pq) or np.sign(pq) != np.sign(q):
                        continue
                    if abs(pa / pq - up) <= price_tol * abs(up) and abs(pq - q) <= qty_tol * abs(pq):
                        cands.append(i)
            i = best(cands, c, need_text=True)
            if i is not None:
                taken.add(i)
                matched[c] = (i, "P3_po_unitprice_qty")

    # P4: one invoice line covering several receipts on the same PO
    #     (same unit price, receipt quantities add up to the invoiced quantity)
    multi: dict[int, list[int]] = {}
    if mcfg.get("allow_aggregate", True):
        max_parts = int(mcfg.get("aggregate_max_parts", 12))
        for c in order:
            if c in matched:
                continue
            r = rep.loc[c]
            a, q = r["amt_r"], r["qty_r"]
            if pd.isna(a) or pd.isna(q) or q == 0:
                continue
            up = a / q
            cands = []
            for po in c_pos.get(c, set()):
                for i in by_po.get(po, []):
                    if i in taken:
                        continue
                    pq, pa = pri.at[i, "quantity_num"], pri_amt[i]
                    if not pq or pd.isna(pq) or np.sign(pq) != np.sign(q):
                        continue
                    if abs(pa / pq - up) <= price_tol * abs(up) and \
                            _text_sim(r["text_key"], pri.at[i, "text_key"]) >= text_min:
                        cands.append(i)
            if len(cands) < 2:
                continue
            sub = _subset_sum(cands, [pri_amt[i] for i in cands], a,
                              max(tol_abs, tol_rel * abs(a)) * 2, max_parts)
            if sub:
                for i in sub:
                    taken.add(i)
                matched[c] = (sub[0], "P4_po_aggregate")
                multi[c] = sub

    # ---------------------------------------------------------------- 3. outcome
    dup_rows = []
    for c, g in clusters:
        idx = list(g.sort_values("row_order").index)
        if c in matched:
            pi, pas = matched[c]
            df.loc[idx, "matched_to"] = df.at[pi, "record_id"]
            df.loc[idx, "match_pass"] = pas
            dup_rows += [(i, "cross_source:" + pas) for i in idx]
        else:
            # keep the first copy whose status counts (a cancelled or rejected first copy must
            # not take a paid copy's spend down with it); idx[0] only when no copy counts
            counting = [i for i in idx if not _status_excluded(df.at[i, "cand_codes"])]
            keep = counting[0] if counting else idx[0]
            rest = [i for i in idx if i != keep]
            df.loc[rest, "matched_to"] = df.at[keep, "record_id"]
            df.loc[rest, "match_pass"] = "S0_exported_twice"
            dup_rows += [(i, "exported_twice") for i in rest]
    for i, rule in dup_rows:
        df.at[i, "cand_codes"] = df.at[i, "cand_codes"] + [("DUPLICATE", rule)]

    passes = pd.Series([m[1] for m in matched.values()]).value_counts().to_dict()
    stats.update({
        "secondary_rows": int(len(sec)),
        "secondary_clusters": int(sec["cluster"].nunique()),
        "clusters_matched": int(len(matched)),
        "match_passes": passes,
        "rows_exported_twice": int(sum(1 for _, r in dup_rows if r == "exported_twice")),
        "primary_lines_claimed": int(len(taken)),
        # spend found only in the second system: counted rows of unmatched clusters
        "secondary_only_rows": int(sum(1 for c in clusters.groups if c not in matched)),
    })
    log.info("cross-source: %s", stats)
    return stats
