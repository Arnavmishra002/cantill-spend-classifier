"""Classify every line into the company's tree (deepest level), with a confidence.

Layers, cheapest first; each unit takes the first layer that is confident:
  1. memory      labels seen before for the same unit / supplier+text / item code
  2. supervised  TF-IDF + logistic regression trained on the company's labels
  3. LLM         Claude Opus (primary), Claude Sonnet (second opinion on uncertain and
                 high-spend units), Claude Opus judge on disagreements; the full tree,
                 per-leaf scope notes, supplier profiles and nearest labelled examples
  4. offline     zero-shot retrieval (line text vs. tree paths) + propagation of
                 supplier / account category mix to blank or generic lines
Works with zero labels: layers 1-2 switch off and 3-4 carry the load.
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ..dedupe import GENERIC_DESCRIPTIONS
from ..ingest import read_table
from ..paths import resolve_file
from ..taxonomy import Taxonomy
from ..textnorm import clean_text
from .calibration import OfflineCalibrator, family
from .llm import LLMClassifier, llm_settings
from .units import build_units
from .vectors import DenseEncoder, KNNIndex, SparseVectorizer

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------- labels

def load_labels(cfg: dict, data_dir: Path, tax: Taxonomy) -> pd.DataFrame | None:
    lcfg = cfg.get("classification", {}).get("labels")
    if not lcfg or not lcfg.get("file"):
        return None
    p = resolve_file(lcfg["file"], cfg, data_dir, required=False)
    if p is None:
        log.warning("labels file %s not found; running without labels", lcfg["file"])
        return None
    lab = read_table(p)
    idc = lcfg.get("id_column", "record_id")
    out = pd.DataFrame({"record_id": lcfg.get("id_prefix", "") + lab[idc].map(clean_text)})
    lv = lcfg.get("level_columns") or []
    out["node"] = [tax.node_for_labels([clean_text(x) for x in r]) for r in lab[lv].itertuples(index=False)]
    if lcfg.get("include_column"):
        out["label_include"] = lab[lcfg["include_column"]].map(clean_text)
        out["label_reason"] = lab[lcfg.get("reason_column", "")].map(clean_text) \
            if lcfg.get("reason_column") else ""
    for k in ("supplier_key_column", "item_key_column"):
        if lcfg.get(k):
            out[k.replace("_column", "")] = lab[lcfg[k]].map(clean_text)
    out["label_weight"] = float(lcfg.get("weight", 1.0))
    bad = out["node"].isna() & (lab[lv].apply(lambda r: any(clean_text(x) for x in r), axis=1))
    if bad.any():
        log.warning("%d labels do not map to the taxonomy and are ignored", int(bad.sum()))
    return out


def build_memory(df: pd.DataFrame, labels: pd.DataFrame | None, supplier_names: pd.Series) -> dict:
    """Golden supplier/item keys learned from labels (used by dedupe)."""
    mem: dict = {}
    if labels is None:
        return mem
    m = df[["record_id", "source", "supplier_id", "item_code", "description"]].merge(labels, on="record_id")
    from ..dedupe import item_norm, supplier_compact
    if "supplier_key" in m and (m["supplier_key"] != "").any():
        ck = df.set_index("record_id").loc[m["record_id"], "supplier_name"].map(supplier_compact).values
        m["ck"] = ck
        by_name = {}
        for k, g in m[m["supplier_key"] != ""].groupby("ck")["supplier_key"]:
            vc = g.value_counts(normalize=True)
            if k and vc.iloc[0] >= 0.9:
                by_name[k] = vc.index[0]
        by_id = {}
        idk = m["source"] + "|" + m["supplier_id"].fillna("").str.strip()
        for k, g in m[m["supplier_key"] != ""].groupby(idk)["supplier_key"]:
            vc = g.value_counts(normalize=True)
            if vc.iloc[0] >= 0.95 and len(g) >= 3:
                by_id[k] = vc.index[0]
        mem["supplier_by_name"], mem["supplier_by_id"] = by_name, by_id
    if "item_key" in m and (m["item_key"] != "").any():
        mi = m[m["item_key"] != ""]
        code = mi["item_code"].fillna("").str.strip().str.upper()
        by_code = {}
        for k, g in mi.groupby(code)["item_key"]:
            vc = g.value_counts(normalize=True)
            if k and vc.iloc[0] >= 0.9:
                by_code[k] = vc.index[0]
        by_desc = {}
        dn = mi["description"].map(item_norm)
        for k, g in mi.groupby(dn)["item_key"]:
            vc = g.value_counts(normalize=True)
            if k and k not in GENERIC_DESCRIPTIONS and vc.iloc[0] >= 0.95:
                by_desc[k] = vc.index[0]
        mem["item_by_code"], mem["item_by_desc"] = by_code, by_desc
    if "item_key" in m:
        nocode = m["item_code"].fillna("").str.strip() == ""
        if int(nocode.sum()) >= 50:
            share = float((m.loc[nocode, "item_key"].fillna("") == "").mean())
            mem["blank_item_share_without_code"] = round(share, 4)
            mem["blank_item_without_code"] = share >= 0.95
    return mem


# ------------------------------------------------------------------- helpers

def _softmax(x, t=1.0):
    x = np.asarray(x, dtype=float) / t
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def _supervised_text(u: pd.DataFrame) -> list[str]:
    toks = []
    for r in u.itertuples(index=False):
        extra = []
        if r.supplier_key:
            extra.append("supkey" + "".join(ch for ch in r.supplier_key if ch.isalnum()).lower())
        if r.item_code:
            extra.append("itm" + "".join(ch for ch in r.item_code if ch.isalnum()).lower())
        if r.gl_description:
            extra.append("gl" + "".join(ch for ch in r.gl_description if ch.isalnum()).lower())
        toks.append(" ".join([r.text] + extra))
    return toks


def is_weak(desc_key: str) -> bool:
    return (desc_key in GENERIC_DESCRIPTIONS) or len(desc_key.replace(" ", "")) < 3


# ---------------------------------------------------------------------- main

def classify(df: pd.DataFrame, cfg: dict, tax: Taxonomy, data_dir: Path, cache_dir: Path,
             labels: pd.DataFrame | None = None, use_llm: bool = True,
             train_df: pd.DataFrame | None = None, llm_health: dict | None = None) -> dict:
    """Adds columns node, confidence, class_source to df. Returns stats.

    train_df: labelled rows from an earlier drop of the same company (e.g. A when
    running A2). They train the memory / supervised layers but are not output.
    """
    out_df = df
    if train_df is not None and labels is not None:
        t = train_df[train_df["record_id"].isin(set(labels["record_id"]))].copy()
        t["record_id"] = "TRAIN|" + t["record_id"]
        t["spend_rc"] = 0.0
        labels = labels.assign(record_id="TRAIN|" + labels["record_id"])
        keep = [c for c in df.columns if c in t.columns]
        df = pd.concat([df.assign(_train=False), t[keep].assign(_train=True)], ignore_index=True)
    else:
        df = df.assign(_train=False)
    ccfg = cfg.get("classification", {})
    lcfg = llm_settings(cfg)
    stats: dict = {}
    leaf_ids = list(tax.leaves)
    L = len(leaf_ids)
    col = {nid: j for j, nid in enumerate(leaf_ids)}

    uid, units = build_units(df, cfg)
    df["unit_id"] = uid
    cur_units = set(uid[~df["_train"].values])
    units["current"] = units["unit_id"].isin(cur_units)
    U = len(units)
    units["weak"] = units["desc_key"].map(is_weak)
    stats["units"] = U
    stats["weak_units"] = int(units["weak"].sum())
    log.info("classification: %d lines -> %d units (%d weak)", len(df), U, stats["weak_units"])

    # unit label distributions from labels (leaf only; non-leaf labels are mapped
    # to themselves later if needed)
    lab_dist = np.zeros((U, L))
    has_label = np.zeros(U, dtype=bool)
    if labels is not None:
        lm = df[["record_id", "unit_id"]].merge(labels[labels["node"].notna()], on="record_id")
        lm = lm[lm["node"].isin(col)]
        uidx = pd.Series(range(U), index=units["unit_id"])
        for (u, n), c in lm.groupby(["unit_id", "node"]).size().items():
            lab_dist[uidx[u], col[n]] += c
        has_label = lab_dist.sum(1) > 0
        stats["labelled_units"] = int(has_label.sum())

    # ---- vectors
    leaf_text = [tax.leaf_text(n) for n in leaf_ids]
    sv = SparseVectorizer().fit(list(units["text"]) + leaf_text)
    Xu = sv.transform(units["text"])
    Xl = sv.transform(leaf_text)

    # ---- layer 4 (always available): zero-shot retrieval over tree paths
    sim = (Xu @ Xl.T).toarray()
    p_ret = _softmax(sim, t=float(ccfg.get("retrieval_temperature", 0.05)))

    probs = np.zeros((U, L))
    conf = np.zeros(U)
    source = np.array([""] * U, dtype=object)

    # ---- layer 1: memory (unit itself labelled)
    if has_label.any():
        d = lab_dist[has_label]
        n = d.sum(1)
        purity = d.max(1) / n
        probs[has_label] = d / n[:, None]
        # labels of the very same unit: trust them (scaled by the label set's quality)
        q = float((ccfg.get("labels") or {}).get("quality", 1.0))
        conf[has_label] = purity * (0.86 + 0.12 * np.minimum(n, 3) / 3) * q
        source[has_label] = "memory"

    # ---- layer 2: supervised model (needs labels)
    p_sup = None
    min_lab = int(ccfg.get("min_labelled_units", 50))
    if has_label.sum() >= min_lab:
        texts = _supervised_text(units)
        Xs = SparseVectorizer().fit(texts)
        XS = Xs.transform(texts)
        y = lab_dist[has_label].argmax(1)
        w = np.log1p(lab_dist[has_label].sum(1))
        if len(set(y)) >= 2:
            clf = LogisticRegression(C=float(ccfg.get("supervised_C", 8.0)), max_iter=2000)
            clf.fit(XS[has_label], y, sample_weight=w)
            pp = clf.predict_proba(XS)
            p_sup = np.zeros((U, L))
            p_sup[:, clf.classes_] = pp
            todo = ~has_label
            probs[todo] = p_sup[todo]
            conf[todo] = p_sup[todo].max(1)
            # a supplier never seen in the labels: the model mostly learned supplier
            # identity, so its confidence is not trustworthy -> halve it (routes to LLM)
            seen_sup = set(units.loc[has_label, "supplier_key"])
            unseen = todo & ~units["supplier_key"].isin(seen_sup).values
            conf[unseen] *= float(ccfg.get("unseen_supplier_conf_factor", 1.0))
            stats["units_unseen_supplier"] = int(unseen.sum())
            source[todo] = "supervised"
            stats["supervised_classes"] = int(len(clf.classes_))

    # ---- dense vectors + kNN exemplars (labelled units) for LLM hints
    knn_hint = defaultdict(list)
    if has_label.any() and use_llm:
        enc = DenseEncoder(cfg, list(units["text"]))
        D = enc.encode(list(units["text"]))
        lab_idx = np.where(has_label)[0]
        index = KNNIndex(D[lab_idx])
        q = np.where(~has_label | (conf < 0.95))[0]
        if len(q):
            s, ix = index.search(D[q], 4)
            for qi, row_s, row_i in zip(q, s, ix):
                for sc, li in zip(row_s, row_i):
                    j = lab_idx[li]
                    if j == qi or sc < 0.5:
                        continue
                    knn_hint[qi].append(f"like '{units.at[j, 'description'][:40]}' -> "
                                        f"{leaf_ids[int(lab_dist[j].argmax())]}")
        stats["dense_encoder"] = enc.mode

    # ---- layer 3: LLM
    llm_stats = {}
    zero_shot = p_sup is None
    if use_llm and lcfg.get("enabled", True):
        if lcfg.get("scope_notes", True):
            from .enrich import add_scope_notes
            stats["scope_notes"] = add_scope_notes(tax, units, cfg, cache_dir)
        llm = LLMClassifier(cfg, tax, cache_dir)
        if llm_health:
            llm.apply_health(llm_health)
        if llm.enabled():
            thr = float(lcfg.get("low_confidence_threshold", 0.5))
            cand = np.where(conf < thr)[0] if not zero_shot else np.arange(U)
            cand = cand[units["current"].values[cand]]
            # biggest spend first; cap the number of units sent
            cand = cand[np.argsort(-units["spend"].values[cand])][: int(lcfg.get("max_units", 20000))]
            if lcfg.get("only_records"):  # evaluation only: units holding these records
                keep_u = set(df.loc[df["record_id"].isin(set(lcfg["only_records"])), "unit_id"])
                cand = cand[units["unit_id"].isin(keep_u).values[cand]]
            if lcfg.get("unit_sample"):  # evaluation only: random sample of units
                us = lcfg["unit_sample"]
                rng = np.random.default_rng(int(us.get("seed", 0)))
                cand = np.sort(rng.choice(cand, size=min(int(us["n"]), len(cand)), replace=False))
            sup_lines = _supplier_lines(units)
            # supplier profiles: from label history when the supplier has labels,
            # otherwise one LLM decision per supplier from its biggest lines
            profiles = {}
            if has_label.any():
                sk_lab = units.loc[has_label, "supplier_key"].values
                d_lab = lab_dist[has_label]
                for sk in pd.unique(sk_lab):
                    d = d_lab[sk_lab == sk].sum(0)
                    if d.sum() > 0:
                        profiles[sk] = {"node": leaf_ids[int(d.argmax())], "confidence": float(d.max() / d.sum()),
                                        "from": "labels"}
            if lcfg.get("supplier_profiles", True):
                need = [i for i in cand if units.at[i, "supplier_key"] not in profiles]
                profiles.update(_profile_suppliers(llm, units, np.array(need, dtype=int), leaf_ids, tax))
            stats["supplier_profiles"] = len(profiles)
            items = []
            for i in cand:
                r = units.iloc[i]
                payload = {"line": _llm_line(r)}
                hints = []
                if p_sup is not None:
                    top = np.argsort(-p_sup[i])[:2]
                    hints += [f"model {leaf_ids[t]} p{p_sup[i, t]:.2f}" for t in top if p_sup[i, t] >= 0.15]
                hints += knn_hint.get(i, [])[:2]
                if hints:
                    payload["hints"] = hints
                sk = r["supplier_key"]
                gp = {"supplier": r["supplier_name"] or "(no name)"}
                pr = profiles.get(sk)
                if pr:
                    src_txt = "labelled history" if pr.get("from") == "labels" else "conf"
                    gp["profile"] = (f"{pr['node']} ({tax.nodes[pr['node']].label}) "
                                     f"{src_txt} {pr['confidence']:.2f}")
                sl = sup_lines.get(sk, [])[:5]
                if sl:
                    gp["other_lines"] = sl
                items.append({"uid": r["unit_id"], "payload": payload, "group": sk, "group_payload": gp})
            log.info("LLM: %d units to classify", len(items))
            # spend-weighted scoring: the units holding the top share of spend always get
            # a second, independent opinion (plus every unit the primary is unsure about)
            priority = []
            share = float(lcfg.get("second_top_spend_share", 0.5))
            if share > 0 and items:
                sp = units.set_index("unit_id")["spend"]
                tot = float(sum(sp.get(it["uid"], 0.0) for it in items)) or 1.0
                acc = 0.0
                for it in sorted(items, key=lambda it: -sp.get(it["uid"], 0.0))[
                        : int(lcfg.get("second_top_spend_max_units", 400))]:
                    if acc / tot >= share:
                        break
                    priority.append(it["uid"])
                    acc += sp.get(it["uid"], 0.0)
                stats["second_opinion_top_spend"] = {"units": len(priority), "spend_share": round(acc / tot, 3)}
            res = llm.classify_units(items, vote_threshold=float(lcfg.get("vote_threshold", 0.7)),
                                     priority=priority)
            uidx = pd.Series(range(U), index=units["unit_id"])
            for u, v in res.items():
                i = uidx[u]
                j = col.get(v["node"])
                if j is None:
                    continue
                c_llm = float(v["confidence"])
                if p_sup is not None and source[i] != "memory":
                    # blend: LLM choice, confidence informed by the supervised prob
                    c_llm = 0.5 * c_llm + 0.5 * max(p_sup[i, j], c_llm * 0.6)
                probs[i] = 0.0
                probs[i, j] = 1.0
                conf[i] = c_llm
                source[i] = "llm_judge" if v.get("judged") else ("llm_agree" if v.get("agree") else "llm")
            llm_stats = llm.summary()
    stats["llm"] = llm_stats

    # ---- layer 4: offline zero-shot for whatever is still unclassified
    rest = source == ""
    calib_p1 = np.full(U, np.nan)
    calib_margin = np.full(U, np.nan)
    if rest.any():
        probs[rest] = p_ret[rest]
        conf[rest] = p_ret[rest].max(1) * 0.6
        source[rest] = "retrieval"
        # propagate supplier / account mix to weak units
        _propagate(units, probs, conf, source, rest & units["weak"].values)
        top2 = np.sort(probs[rest], axis=1)[:, -2:]
        calib_p1[rest] = top2[:, 1]
        calib_margin[rest] = top2[:, 1] - top2[:, 0]
    srt = np.sort(sim, axis=1)
    units["calib_p1"] = calib_p1
    units["calib_margin"] = calib_margin
    units["calib_sim1"] = srt[:, -1]
    units["calib_sim2"] = srt[:, -2] if srt.shape[1] > 1 else 0.0
    if rest.any():
        # calibrated confidence: probability that the chosen leaf is right
        cal = OfflineCalibrator()
        fams = np.array([family(x) for x in source])
        for fam, col_ in (("retrieval", "calib_sim1"), ("propagated", "calib_p1")):
            m = rest & (fams == fam)
            if m.any():
                conf[m] = cal.predict(fam, units[col_].values[m], conf[m])

    node_idx = probs.argmax(1)
    units["node"] = [leaf_ids[j] for j in node_idx]
    units["confidence"] = np.clip(conf, 0.0, 1.0).round(3)
    units["class_source"] = source
    m = units.set_index("unit_id")
    cur = df[~df["_train"]]
    out_df["unit_id"] = cur["unit_id"].values
    out_df["node"] = cur["unit_id"].map(m["node"]).values
    out_df["confidence"] = cur["unit_id"].map(m["confidence"]).values
    out_df["class_source"] = cur["unit_id"].map(m["class_source"]).values
    out_df["calib_p1"] = cur["unit_id"].map(m["calib_p1"]).values
    out_df["calib_margin"] = cur["unit_id"].map(m["calib_margin"]).values
    out_df["calib_sim1"] = cur["unit_id"].map(m["calib_sim1"]).values
    out_df["calib_sim2"] = cur["unit_id"].map(m["calib_sim2"]).values
    out_df["calib_weak"] = cur["unit_id"].map(m["weak"]).values
    stats["consistency"] = _consistency_pass(out_df, ccfg)
    stats["class_source_lines"] = out_df["class_source"].value_counts().to_dict()
    stats["_units"] = units[units["current"]]
    return stats


def _profile_suppliers(llm, units: pd.DataFrame, cand, leaf_ids, tax) -> dict:
    """One LLM decision per supplier from its biggest lines (cached like everything else)."""
    cur = units[units["current"]]
    if len(cand) == 0:
        return {}
    need = set(units.iloc[cand]["supplier_key"])
    items = []
    for sk, g in cur.groupby("supplier_key", sort=False):
        if sk not in need or not sk:
            continue
        if int((~g["weak"]).sum()) < 2:
            continue  # nothing to learn: single line (it speaks for itself) or only blank text
        g = g.sort_values("spend", ascending=False)
        tot = float(g["spend"].sum()) or 1.0
        top = []
        for r in g.head(15).itertuples(index=False):
            d = r.description[:70] if r.description else "(blank)"
            extra = f" [{r.gl_description}]" if r.gl_description else ""
            top.append(f"{d}{extra} x{r.n_lines} ({100 * r.spend / tot:.0f}%)")
        name = g["supplier_name"].iloc[0] or "(no name, id only)"
        items.append({"uid": sk, "payload": {"supplier": name, "n_lines": int(g["n_lines"].sum()),
                                             "top_lines": top}})
    if not items:
        return {}
    log.info("LLM: profiling %d suppliers", len(items))
    return llm.profile_suppliers(items)


def _consistency_pass(df: pd.DataFrame, ccfg: dict) -> dict:
    """Low-confidence lines take the dominant category of their group's confident lines.

    Groups: the supplier, plus any configured fields (e.g. a contract number). Many
    clients file spend by supplier or contract; a vague line ("Mining services",
    "Hand tools") on a contract whose other lines are clearly one category usually
    belongs there. Measured on A with labels hidden: +3 to +4 points L3 (line).
    """
    c = ccfg.get("consistency", {}) or {}
    if c.get("enabled", True) is False:
        return {}
    hi, lo, share_min = float(c.get("strong", 0.85)), float(c.get("weak", 0.7)), float(c.get("share", 0.7))
    min_strong = int(c.get("min_strong_lines", 3))
    fields = ["supplier_key"] + list(c.get("group_fields", []) or [])
    stats = {}
    for f in fields:
        col = f if f in df.columns else (f"raw__{f}" if f"raw__{f}" in df.columns else None)
        if col is None:
            continue
        g = df[col].fillna("").astype(str)
        conf = df["confidence"].astype(float)
        w = df["spend_rc"].abs().fillna(0.0) + 1e-6 if "spend_rc" in df.columns else pd.Series(1.0, index=df.index)
        strong = (conf >= hi) & (g != "") & df["node"].notna()
        if not strong.any():
            continue
        dist = pd.DataFrame({"g": g[strong], "n": df.loc[strong, "node"], "w": w[strong]}) \
            .groupby(["g", "n"]).agg(w=("w", "sum"), k=("w", "size")).reset_index()
        tot = dist.groupby("g")["w"].transform("sum")
        cnt = dist.groupby("g")["k"].transform("sum")
        dist["share"] = dist["w"] / tot
        dist = dist[cnt >= min_strong].sort_values("share", ascending=False).drop_duplicates("g").set_index("g")
        weak = (conf < lo) & g.isin(dist.index)
        new = g.map(dist["n"])
        sh = g.map(dist["share"])
        ch = weak & (sh >= share_min) & (new != df["node"])
        df.loc[ch, "node"] = new[ch]
        cal = OfflineCalibrator()
        df.loc[ch, "calib_p1"] = sh[ch]
        df.loc[ch, "calib_margin"] = 2 * sh[ch] - 1
        df.loc[ch, "confidence"] = np.round(cal.predict("group", sh[ch], sh[ch] * 0.75), 3)
        df.loc[ch, "class_source"] = df.loc[ch, "class_source"].astype(str) + "+group"
        stats[f] = int(ch.sum())
    return stats


def _llm_line(r) -> str:
    """Line text for the LLM; the supplier travels in the batch's supplier block."""
    # item codes are opaque ids (M1000032, MAT-1000007): useless to a language model and
    # they stop identical lines from sharing one LLM decision, so they are left out
    parts = [f"desc: {r['description'] if r['description'] else '(blank)'}"]
    if r.get("gl_description"):
        parts.append(f"account: {r['gl_description']}")
    if r.get("context"):
        parts.append(f"context: {r['context']}")
    return " | ".join(parts)


def _supplier_lines(units: pd.DataFrame) -> dict[str, list[str]]:
    out = {}
    ok = units[~units["weak"]].sort_values("spend", ascending=False)
    for sk, g in ok.groupby("supplier_key", sort=False):
        out[sk] = list(dict.fromkeys(g["description"].map(lambda s: s[:70])))[:8]
    return out


def _propagate(units, probs, conf, source, mask):
    """Weak units take the spend-weighted category mix of the same supplier's
    (then same account's) confidently classified units."""
    strong = ~mask & (conf >= 0.3)
    for key in ("supplier_key", "gl_description"):
        groups = defaultdict(list)
        for i in np.where(strong)[0]:
            k = units.iat[i, units.columns.get_loc(key)]
            if k:
                groups[k].append(i)
        for i in np.where(mask & (source == "retrieval"))[0]:
            k = units.iat[i, units.columns.get_loc(key)]
            if k in groups:
                idx = groups[k]
                w = units["spend"].values[idx] + 1.0
                dist = (probs[idx] * (w * conf[idx])[:, None]).sum(0)
                if dist.sum() > 0:
                    dist = dist / dist.sum()
                    probs[i] = dist
                    conf[i] = float(dist.max()) * 0.7
                    source[i] = f"propagated_{key.split('_')[0]}"
