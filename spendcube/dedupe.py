"""Supplier and item de-duplication.

Suppliers
  1. normalise names: ascii, upper, '&' -> AND, OCR digit fold, strip legal
     suffixes (PTY LTD, PVT LTD, LLP, INC, GMBH, SA DE CV, ...), drop punctuation
  2. compact key (no spaces) so 'KEL MONT' == 'KELMONT'
  3. union-find over: same compact key, same supplier id (if trusted), same tax
     id, and fuzzy name match inside blocks (rapidfuzz, conservative threshold)
  4. learned memory (when labels exist): known name / id -> client golden key

Items
  1. item code (material number) groups lines
  2. normalised description (OCR fold, unit spacing, punctuation) groups codes
     that describe the same thing; fuzzy match inside blocks for typos
  3. learned memory: item code -> golden item key
"""
from __future__ import annotations

import hashlib
import logging
import re
from collections import defaultdict

import pandas as pd
from rapidfuzz import fuzz, process

from .textnorm import ascii_fold, clean_text, norm_key

log = logging.getLogger(__name__)

DEFAULT_LEGAL_SUFFIXES = [
    "PROPRIETARY LIMITED", "PTY LTD", "PTY LIMITED", "PTY", "PRIVATE LIMITED", "PVT LTD", "PVT",
    "PRIVATE", "LIMITED", "LTD", "LLP", "LLC", "L L C", "INCORPORATED", "INC", "CORPORATION",
    "CORP", "COMPANY", "CO", "PLC", "GMBH", "AG", "KG", "SARL", "SAS", "SA DE CV", "SA", "SRL",
    "SPA", "S P A", "BV", "NV", "OY", "AB", "AS", "KFT", "SP Z O O", "SPZOO", "DOO", "D O O",
    "LTDA", "EIRELI", "ME", "CC", "NPC", "SOC", "PJSC", "OOO", "ZAO", "OAO", "PTE", "BHD",
    "SDN BHD", "SDN", "KK", "TBK", "CIA", "Y CIA", "AND CO", "ET CIE", "THE",
]


def _h(s: str, n=8) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]


class UnionFind:
    def __init__(self):
        self.p = {}

    def find(self, x):
        p = self.p
        p.setdefault(x, x)
        root = x
        while p[root] != root:
            root = p[root]
        while p[x] != root:
            p[x], x = root, p[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # deterministic: smaller string becomes root
            if rb < ra:
                ra, rb = rb, ra
            self.p[rb] = ra


# ------------------------------------------------------------------- suppliers

def supplier_norm(name: str, suffixes: list[str] | None = None) -> str:
    s = ascii_fold(clean_text(name)).upper()
    s = s.replace("&", " AND ").replace("+", " AND ")
    s = re.sub(r"\(.*?\)", lambda m: " " + m.group(0)[1:-1] + " ", s)  # '(PTY)' -> 'PTY'
    s = norm_key(s, ocr=True)
    toks = s.split()
    suf = [x.split() for x in (suffixes or DEFAULT_LEGAL_SUFFIXES)]
    suf.sort(key=len, reverse=True)
    changed = True
    while changed and len(toks) > 1:
        changed = False
        for sx in suf:
            if len(toks) > len(sx) and toks[-len(sx):] == sx:
                toks = toks[:-len(sx)]
                changed = True
                break
    # leading 'THE'
    if len(toks) > 1 and toks[0] == "THE":
        toks = toks[1:]
    return " ".join(toks)


def supplier_compact(name: str, suffixes=None) -> str:
    return supplier_norm(name, suffixes).replace(" ", "")


def dedupe_suppliers(df: pd.DataFrame, cfg: dict, memory: dict | None = None) -> pd.Series:
    """Return supplier_key per row."""
    scfg = cfg.get("dedupe", {}).get("supplier", {})
    suffixes = scfg.get("legal_suffixes")
    thr = float(scfg.get("fuzzy_threshold", 92))
    names = df["supplier_name"].fillna("")
    ids = df["supplier_id"].fillna("").str.strip() if "supplier_id" in df else pd.Series("", index=df.index)
    taxid = df["supplier_tax_id"].fillna("").str.upper().str.replace(r"[^0-9A-Z]", "", regex=True) \
        if "supplier_tax_id" in df else pd.Series("", index=df.index)

    uniq_names = pd.unique(names)
    norm = {n: supplier_norm(n, suffixes) for n in uniq_names}
    comp = {n: norm[n].replace(" ", "") for n in uniq_names}
    df_ck = names.map(comp)

    uf = UnionFind()
    # node per compact name; lines with no name use their id (or a single unknown node)
    node = pd.Series("N:" + df_ck, index=df.index)
    noname = df_ck == ""
    node[noname] = "I:" + ids[noname]
    node[noname & (ids == "")] = "UNKNOWN"
    for n in node.unique():
        uf.find(n)
    if scfg.get("use_supplier_id", True):
        # same supplier id -> same supplier (ids are per source system, so prefix source
        # unless the config says ids are global)
        scope = df["source"] if not scfg.get("supplier_id_global", False) else ""
        idk = (scope + "|" + ids).where(ids != "")
        for _, g in node.groupby(idk.values):
            vals = list(pd.unique(g))
            for v in vals[1:]:
                uf.union(vals[0], v)
    if (taxid != "").any():
        for _, g in node.groupby(taxid.where(taxid != "").values):
            vals = list(pd.unique(g))
            for v in vals[1:]:
                uf.union(vals[0], v)

    # fuzzy name matching with blocking on first 3 chars of compact key and on
    # first token; compare each name only within its blocks
    ck_list = sorted({c for c in comp.values() if c})
    norm_of_ck = {}
    for n, c in comp.items():
        if c:
            norm_of_ck.setdefault(c, norm[n])
    blocks = defaultdict(list)
    for c in ck_list:
        blocks["P:" + c[:3]].append(c)
        first = norm_of_ck[c].split()[0] if norm_of_ck[c] else ""
        if len(first) >= 3:
            blocks["T:" + first].append(c)
    for _, members in blocks.items():
        if len(members) < 2 or len(members) > 3000:
            continue
        for i, a in enumerate(members):
            res = process.extract(a, members[i + 1:], scorer=fuzz.ratio, score_cutoff=thr, limit=None)
            for b, score, _ in res:
                # guard: very short names need near-exact match
                if min(len(a), len(b)) < 5 and score < 97:
                    continue
                # token-level sanity: both normalised names must agree on most tokens
                if fuzz.token_set_ratio(norm_of_ck[a], norm_of_ck[b]) < thr - 5:
                    continue
                uf.union("N:" + a, "N:" + b)

    root = node.map(uf.find)
    # learned memory: map clusters to golden keys where we know them
    key = "S-" + root.map(lambda r: _h(r))
    if memory:
        by_name = memory.get("supplier_by_name", {})
        by_id = memory.get("supplier_by_id", {})
        gold = df_ck.map(by_name)
        # an id (including the blank id bucket) the labels map to one golden key
        idkey = (df["source"] + "|" + ids)
        gold_id = idkey.map(by_id)
        gold = gold.fillna(gold_id)
        # spread a cluster's golden key to cluster members without one
        cl_gold = gold.groupby(root).agg(lambda s: s.dropna().mode().iloc[0] if s.notna().any() else None)
        gold = gold.fillna(root.map(cl_gold))
        key = gold.where(gold.notna(), key)
    return key


# ----------------------------------------------------------------------- items

_UNIT_FIX = [
    (re.compile(r"(\d)\s+(MM|CM|M|KG|G|L|ML|KW|W|V|A|HP|BAR|PSI|IN|INCH|FT|T|TON|TONS|OZ|LB)\b"), r"\1\2"),
    (re.compile(r"\bP\s*/?\s*N\b"), "PN"),
]


def item_norm(desc: str) -> str:
    s = norm_key(desc, ocr=True)
    for rx, rep in _UNIT_FIX:
        s = rx.sub(rep, s)
    return s


GENERIC_DESCRIPTIONS = {"", "NO OBJECT", "PRODUCT", "PRODUCTS", "PARTS", "PART", "SERVICE",
                        "SERVICES", "MATERIAL", "MATERIALS", "MISC", "MISCELLANEOUS", "VARIOUS",
                        "GOODS", "ITEM", "ITEMS", "NA", "N A", "NONE", "OTHER", "SUNDRY"}


def dedupe_items(df: pd.DataFrame, cfg: dict, supplier_key: pd.Series, memory: dict | None = None,
                 link: pd.Series | None = None) -> pd.Series:
    """Return item_key per row.

    link: optional Series of record_id each row is matched to (cross-source
    matches) so a secondary row inherits its primary row's item.
    """
    icfg = cfg.get("dedupe", {}).get("item", {})
    thr = float(icfg.get("fuzzy_threshold", 95))
    code = df["item_code"].fillna("").str.strip().str.upper() if "item_code" in df else pd.Series("", index=df.index)
    desc = df["description"].fillna("")
    dn = desc.map(item_norm)
    generic = dn.isin(GENERIC_DESCRIPTIONS) | (dn.str.len() < 3)
    uf = UnionFind()
    use_code = icfg.get("use_item_code", True)
    generic_codes = set(icfg.get("generic_item_codes", []) or [])
    # Codes used with many unrelated descriptions are placeholders ("dummy"
    # materials); never let them glue different items together.
    if use_code:
        dpc = pd.DataFrame({"c": code, "d": dn})[code != ""].groupby("c")["d"].nunique()
        generic_codes |= set(dpc[dpc > int(icfg.get("max_descriptions_per_code", 25))].index)
    nodes = []
    for c, d, g, sk in zip(code, dn, generic, supplier_key):
        if use_code and c and c not in generic_codes:
            n = "C:" + c
            if not g:
                uf.union(n, "D:" + d)  # code <-> description
        elif not g:
            n = "D:" + d
        else:
            # generic or blank text: an item can't be identified beyond supplier + code
            n = "G:" + sk + "|" + (c or "")
        uf.find(n)
        nodes.append(n)
    nodes = pd.Series(nodes, index=df.index)
    # fuzzy on descriptions (typos/OCR) within blocks of first token
    descs = sorted({d for d, g in zip(dn, generic) if not g})
    blocks = defaultdict(list)
    for d in descs:
        blocks[d.split()[0][:6] if d.split() else ""].append(d)
    for _, members in blocks.items():
        if len(members) < 2 or len(members) > 4000:
            continue
        for i, a in enumerate(members):
            res = process.extract(a, members[i + 1:], scorer=fuzz.ratio, score_cutoff=thr, limit=None)
            for b, score, _ in res:
                # numbers must agree exactly (sizes, part numbers)
                if re.findall(r"\d+", a) != re.findall(r"\d+", b):
                    continue
                uf.union("D:" + a, "D:" + b)
    root = nodes.map(uf.find)
    key = "I-" + root.map(lambda r: _h(r, 10))
    if link is not None:
        # inherit the primary record's item key when matched across systems
        rid_to_key = pd.Series(key.values, index=df["record_id"].values)
        linked = link.map(lambda r: rid_to_key.get(r) if r else None)
        key = linked.where(linked.notna(), key)
    if memory:
        by_code = memory.get("item_by_code", {})
        by_desc = memory.get("item_by_desc", {})
        gold = code.map(by_code)
        gold = gold.fillna(dn.where(~generic).map(by_desc))
        cl_gold = gold.groupby(root).agg(lambda s: s.dropna().mode().iloc[0] if s.notna().any() else None)
        gold = gold.fillna(root.map(cl_gold))
        if link is not None:
            rid_to_gold = pd.Series(gold.values, index=df["record_id"].values)
            lg = link.map(lambda r: rid_to_gold.get(r) if r else None)
            gold = gold.fillna(lg)
        key = gold.where(gold.notna(), key)
    return key
