"""Taxonomy enrichment: one LLM call writes a short scope note per leaf.

The note says what belongs in the leaf in this company's own vocabulary and how
it differs from its siblings. Input: the tree (read at run time) plus a sample
of the company's *unlabelled* purchase lines, so it works for an unseen tree with
no labels. Cached on disk by a hash of tree + sample; skipped (no notes) if no
API key, or if the tree file already has descriptions.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


def add_scope_notes(tax, units: pd.DataFrame, cfg: dict, cache_dir: Path) -> int:
    leaves = list(tax.leaves)
    if any(tax.nodes[n].description for n in leaves):
        return 0  # the client's own descriptions win
    lcfg = cfg.get("classification", {}).get("llm", {})
    spec = lcfg.get("notes_model") or lcfg.get("judge") or lcfg.get("primary") or {}
    if spec.get("vendor") != "anthropic" or not os.environ.get("ANTHROPIC_API_KEY"):
        return 0
    cur = units[units["current"]] if "current" in units else units
    ok = cur[~cur["weak"]] if "weak" in cur else cur
    # sample: biggest-spend lines + a spread of the rest (deterministic)
    top = ok.sort_values("spend", ascending=False).head(120)
    rest = ok.drop(top.index)
    rest = rest.sample(min(120, len(rest)), random_state=0) if len(rest) else rest
    sample = pd.concat([top, rest])
    lines = [f"{r.description[:80]} | {r.supplier_name[:40]}" + (f" | {r.gl_description[:30]}" if r.gl_description else "")
             for r in sample.itertuples(index=False)]
    tree = "\n".join(f"{n}: {' > '.join(dict.fromkeys(tax.nodes[n].path))}" for n in leaves)
    ctx = cfg.get("classification", {}).get("company_context") or "not given"
    key = hashlib.sha256(f"notes-v1|{spec.get('model')}|{ctx}|{tree}|{json.dumps(lines)}".encode()).hexdigest()
    cache = Path(cache_dir) / "scope_notes.json"
    store = json.loads(cache.read_text()) if cache.exists() else {}
    notes = store.get(key)
    if notes is None:
        try:
            import anthropic
            from concurrent.futures import ThreadPoolExecutor
            client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"],
                                         base_url=spec.get("base_url") or "https://api.anthropic.com",
                                         max_retries=4, timeout=600.0)
            head = (f"Company context: {ctx}\n\nThis is the company's procurement category tree (leaf ID: path):\n"
                    f"{tree}\n\nHere is a sample of the company's purchase lines (description | supplier | account):\n"
                    + "\n".join(lines))
            chunk = 60   # big trees: several calls, each sees the whole tree but writes notes for <=60 leaves

            def one(ids):
                schema = {"type": "object", "properties": {"notes": {"type": "array", "items": {
                    "type": "object", "properties": {"id": {"type": "string", "enum": ids},
                                                     "note": {"type": "string"}},
                    "required": ["id", "note"], "additionalProperties": False}}},
                    "required": ["notes"], "additionalProperties": False}
                prompt = (head + f"\n\nFor EVERY one of these leaves: {', '.join(ids)} - write one scope note "
                          "(max 30 words) for a classifier: what belongs there, typical items/services in this "
                          "company's own vocabulary and abbreviations, and what distinguishes it from its sibling "
                          "leaves. Do not invent facts about suppliers.")
                with client.messages.stream(
                        model=spec.get("model", "claude-opus-5-5"), max_tokens=32000,
                        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": schema}},
                        messages=[{"role": "user", "content": prompt}]) as stream:
                    msg = stream.get_final_message()
                text = next(b.text for b in msg.content if b.type == "text")
                return {d["id"]: d["note"].strip() for d in json.loads(text)["notes"]}

            parts = [leaves[i:i + chunk] for i in range(0, len(leaves), chunk)]
            notes = {}
            with ThreadPoolExecutor(max_workers=4) as ex:
                for got in ex.map(one, parts):
                    notes.update(got)
            store[key] = notes
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(store, indent=1))
            log.info("scope notes written for %d leaves", len(notes))
        except Exception as e:
            log.warning("scope notes unavailable: %s", str(e)[:150])
            return 0
    for n, note in notes.items():
        if n in tax.nodes:
            tax.nodes[n].description = note
    return len(notes)
