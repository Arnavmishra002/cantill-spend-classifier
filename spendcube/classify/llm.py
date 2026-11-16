"""LLM classification layer (Anthropic Claude + Google Gemini), unattended-safe.

Design
  * Units are sent in batches (default 25 per request) with the full category
    tree in a cached system prompt; the model returns JSON constrained by a
    schema whose `node` field is an enum of valid leaf IDs, so an invalid
    category is impossible.
  * Results are cached on disk keyed by a hash of (model, prompt version, tree,
    unit payload). Re-runs are fast, free and deterministic.
  * Retries with exponential backoff on rate limits / 5xx; a run-wide cost cap
    stops new requests; anything not classified falls back to the offline
    layers, so a run always finishes with a valid file.
  * Roles: `primary` model classifies; optional `second` model (other vendor)
    votes on uncertain units; when they disagree a `judge` model decides.
API keys are read from the environment (or a local .env), never from config.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

PROMPT_VERSION = "v4"

# USD per 1M tokens (input, output, cache_read). Override in config llm.prices.
DEFAULT_PRICES = {
    "claude-opus-5-5": (4.00, 20.00, 0.20),
    "claude-sonnet-5-5": (2.00, 10.00, 0.10),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
    "gemini-2.5-flash": (0.30, 2.50, 0.03),
    "gemini-2.5-flash-lite": (0.10, 0.40, 0.01),
    "gemini-2.5-pro": (1.25, 10.00, 0.31),
}

SYSTEM_TEMPLATE = """You are a procurement spend analyst. You classify purchase lines into one company's category tree.

Company context: {context}

Rules:
- Pick exactly one category ID for every line, always a leaf ID from the tree below.
- The line description is the strongest evidence. Supplier name, GL account, item code and context fields are supporting evidence.
- Descriptions are short, abbreviated (e.g. BRG=bearing, VLV=valve, ASSY=assembly, REPR=repair, MNTN=maintenance, SPRT=support, CNVY=conveyor), OCR-damaged (0/O, 5/S, 8/B swapped) or in other languages; read through that.
- Each line points to a supplier block `s` with the supplier name, its `profile` (the category that best describes the supplier's overall business with this company) and some of its `other_lines`.
- When the description is blank or generic ("Product", "Parts", "No Object"), infer from the supplier block and the account.
- Many companies categorise spend by the supplier's line of business or by the equipment fleet a part is for. When a line is ambiguous (a generic spare part, a repair, a service), prefer the supplier profile; override it when the line clearly belongs elsewhere.
- `hints` (if present) are candidate categories from statistical models or similar labelled lines; use them as evidence, not as orders.
- confidence = your probability (0 to 1) that the chosen leaf is exactly right.

Category tree (ID: L1 > L2 > ... > leaf){desc_note}:
{tree}
"""


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def load_dotenv_if_present(start: Path | None = None):
    """Load KEY=VALUE pairs from a .env file (project root or cwd) without overriding env."""
    cands = [Path.cwd() / ".env"]
    if start:
        cands.insert(0, Path(start) / ".env")
    for p in cands:
        if p.exists():
            for line in p.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if v and not os.environ.get(k):
                    os.environ[k] = v
            break


class ResultCache:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data: dict[str, dict] = {}
        self.lock = threading.Lock()
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                try:
                    r = json.loads(line)
                    self.data[r["k"]] = r["v"]
                except Exception:
                    continue

    def get(self, k):
        return self.data.get(k)

    def put_many(self, items: list[tuple[str, dict]]):
        with self.lock, self.path.open("a") as f:
            for k, v in items:
                self.data[k] = v
                f.write(json.dumps({"k": k, "v": v}) + "\n")


@dataclass
class Budget:
    cap_usd: float
    spent: float = 0.0
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_cache_read: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, usd, tin, tout):
        with self.lock:
            self.spent += usd
            self.calls += 1
            self.tokens_in += tin
            self.tokens_out += tout

    def exhausted(self) -> bool:
        return self.spent >= self.cap_usd


class QuotaError(RuntimeError):
    """Provider refuses for quota/billing reasons; retrying will not help this run."""


class Provider:
    vendor = "base"

    def __init__(self, model: str, prices: dict):
        self.model = model
        self.price = prices.get(model, (5.0, 25.0, 0.5))
        self.disabled = False          # circuit breaker: set after repeated quota errors
        self.quota_failures = 0

    def cost(self, tin, tout, tcache=0):
        pi, po, pc = self.price
        return (tin * pi + tout * po + tcache * pc) / 1e6

    def available(self) -> bool:
        raise NotImplementedError

    def classify(self, system: str, user: str, schema: dict) -> tuple[dict, float, int, int]:
        raise NotImplementedError


class AnthropicProvider(Provider):
    vendor = "anthropic"

    def __init__(self, model, prices, effort="low", base_url=None):
        super().__init__(model, prices)
        self.effort = effort
        self.base_url = base_url or "https://api.anthropic.com"
        self._client = None
        self._lock = threading.Lock()

    def available(self):
        return bool(os.environ.get("ANTHROPIC_API_KEY")) and not self.disabled

    @property
    def client(self):
        with self._lock:
            if self._client is None:
                import anthropic
                self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"],
                                                   base_url=self.base_url, max_retries=5, timeout=180.0)
            return self._client

    def classify(self, system, user, schema):
        r = self.client.messages.create(
            model=self.model,
            max_tokens=16000,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
        )
        if r.stop_reason == "refusal":
            raise RuntimeError("model refused")
        text = next((b.text for b in r.content if b.type == "text"), "")
        u = r.usage
        tin = (u.input_tokens or 0) + int((getattr(u, "cache_creation_input_tokens", 0) or 0) * 1.25)
        tcache = getattr(u, "cache_read_input_tokens", 0) or 0
        usd = self.cost(tin, u.output_tokens, tcache)
        self.last_cache_read = tcache
        return json.loads(text), usd, tin + tcache, u.output_tokens


class GeminiProvider(Provider):
    vendor = "gemini"

    def available(self):
        return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")) and not self.disabled

    def __init__(self, model, prices):
        super().__init__(model, prices)
        self._client = None
        self._lock = threading.Lock()

    @property
    def client(self):
        # one client per provider, created once under a lock: a second client created
        # by a racing thread would be garbage-collected and close its HTTP session
        # while a request is still using it
        with self._lock:
            if self._client is None:
                from google import genai
                from google.genai import types
                self._client = genai.Client(
                    api_key=os.environ.get("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"],
                    http_options=types.HttpOptions(api_version="v1beta", timeout=120_000))
            return self._client

    def classify(self, system, user, schema):
        from google.genai import types
        last = None
        for attempt in range(3):
            try:
                r = self.client.models.generate_content(
                    model=self.model, contents=user,
                    config=types.GenerateContentConfig(
                        system_instruction=system, temperature=0, response_mime_type="application/json",
                        response_json_schema=schema, max_output_tokens=16000))
                um = r.usage_metadata
                tin = um.prompt_token_count or 0
                tout = (um.candidates_token_count or 0) + (getattr(um, "thoughts_token_count", 0) or 0)
                return json.loads(r.text), self.cost(tin, tout), tin, tout
            except Exception as e:
                last = e
                msg = str(e)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    if "quota" in msg.lower() and attempt >= 1:
                        raise QuotaError(msg[:200])
                    time.sleep(5 * (attempt + 1))
                    continue
                if "503" in msg or "500" in msg or "504" in msg or "DEADLINE" in msg:
                    time.sleep(5 * (attempt + 1))
                    continue
                raise
        raise QuotaError(f"gemini failed after retries: {str(last)[:200]}")


def make_provider(spec: dict, prices: dict) -> Provider | None:
    if not spec:
        return None
    v = spec.get("vendor")
    if v == "anthropic":
        return AnthropicProvider(spec["model"], prices, effort=spec.get("effort", "low"),
                                 base_url=spec.get("base_url"))
    if v == "gemini":
        return GeminiProvider(spec["model"], prices)
    raise ValueError(f"unknown LLM vendor {v}")


def tree_block(tax, leaf_ids) -> tuple[str, str]:
    lines = []
    has_desc = False
    for nid in leaf_ids:
        n = tax.nodes[nid]
        path = " > ".join(dict.fromkeys(n.path))
        if n.description:
            has_desc = True
            lines.append(f"{nid}: {path} [{n.description}]")
        else:
            lines.append(f"{nid}: {path}")
    return "\n".join(lines), (" [description]" if has_desc else "")


def result_schema(leaf_ids: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "i": {"type": "integer"},
                        "node": {"type": "string", "enum": list(leaf_ids)},
                        "confidence": {"type": "number"},
                    },
                    "required": ["i", "node", "confidence"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["results"],
        "additionalProperties": False,
    }


class LLMClassifier:
    def __init__(self, cfg: dict, tax, cache_dir: Path):
        lcfg = cfg.get("classification", {}).get("llm", {})
        self.cfg = lcfg
        self.tax = tax
        prices = {**DEFAULT_PRICES, **{k: tuple(v) for k, v in (lcfg.get("prices") or {}).items()}}
        self.primary = make_provider(lcfg.get("primary"), prices)
        self.second = make_provider(lcfg.get("second"), prices)
        self.judge = make_provider(lcfg.get("judge"), prices)
        self.fallbacks = [make_provider(s, prices) for s in (lcfg.get("fallbacks") or [])]
        self.batch_size = int(lcfg.get("batch_size", 25))
        self.concurrency = int(lcfg.get("concurrency", 6))
        self.budget = Budget(float(lcfg.get("cost_cap_usd", 4.0)))
        self.cache = ResultCache(Path(cache_dir) / "llm_results.jsonl")
        self.leaf_ids = list(tax.leaves)
        tree, desc_note = tree_block(tax, self.leaf_ids)
        ctx = cfg.get("classification", {}).get("company_context") or "not given"
        self.system = SYSTEM_TEMPLATE.format(context=ctx, tree=tree, desc_note=desc_note)
        self.schema = result_schema(self.leaf_ids)
        self.tax_hash = _sha(self.system)[:16]
        self.stats = {"cache_hits": 0, "classified": 0, "failed_batches": 0}
        self.lock = threading.Lock()

    # ------------------------------------------------------------------ utils
    def enabled(self) -> bool:
        provs = [p for p in [self.primary, *self.fallbacks] if p]
        return bool(self.cfg.get("enabled", True)) and any(p.available() for p in provs)

    def _key(self, prov: Provider, payload: dict, role: str, group: dict | None = None) -> str:
        return _sha(f"{PROMPT_VERSION}|{role}|{prov.model}|{self.tax_hash}|"
                    f"{json.dumps(payload, sort_keys=True, ensure_ascii=False)}|"
                    f"{json.dumps(group or {}, sort_keys=True, ensure_ascii=False)}")

    def _run_batches(self, prov: Provider, items: list[dict], role: str, system=None,
                     schema=None, instruction: str | None = None) -> dict[str, dict]:
        """items: [{'uid':..., 'payload': {...}}]. Returns uid -> {node, confidence}."""
        system = system or self.system
        schema = schema or self.schema
        out: dict[str, dict] = {}
        todo = []
        same_key: dict[str, list[str]] = {}
        for it in items:
            k = self._key(prov, it["payload"], role, it.get("group_payload"))
            hit = self.cache.get(k)
            if hit is not None:
                out[it["uid"]] = hit
                self.stats["cache_hits"] += 1
            elif k in same_key:          # identical request already queued: share its answer
                same_key[k].append(it["uid"])
            else:
                same_key[k] = [it["uid"]]
                todo.append((it, k))
        if not todo or not prov.available():
            return out
        # lines of the same supplier travel together so supplier context is sent once
        todo.sort(key=lambda x: (x[0].get("group") or "", x[0]["uid"]))
        batches = [todo[i:i + self.batch_size] for i in range(0, len(todo), self.batch_size)]

        def run(batch):
            if self.budget.exhausted() or prov.disabled:
                return []
            groups, gid = [], {}
            lines = []
            for j, (it, _) in enumerate(batch):
                g = it.get("group")
                rec = {"i": j}
                if g and it.get("group_payload") is not None:
                    if g not in gid:
                        gid[g] = f"S{len(gid) + 1}"
                        groups.append({"s": gid[g], **it["group_payload"]})
                    rec["s"] = gid[g]
                rec.update(it["payload"])
                lines.append(rec)
            head = instruction or ("Classify each line. Lines refer to their supplier `s` (context below). "
                                   "Return one result per line `i`.")
            parts = [head]
            if groups:
                parts.append("SUPPLIERS:")
                parts += [json.dumps(x, ensure_ascii=False) for x in groups]
                parts.append("LINES:")
            parts += [json.dumps(x, ensure_ascii=False) for x in lines]
            user = "\n".join(parts)
            res, usd, tin, tout = prov.classify(system, user, schema)
            self.budget.add(usd, tin, tout)
            with self.budget.lock:
                self.budget.tokens_cache_read += int(getattr(prov, "last_cache_read", 0) or 0)
            got = {}
            for r in res.get("results", []):
                j = r.get("i")
                if isinstance(j, int) and 0 <= j < len(batch) and r.get("node") in schema_enum(schema):
                    conf = float(r.get("confidence", 0.5))
                    got[j] = {"node": r["node"], "confidence": max(0.0, min(1.0, conf)),
                              "model": prov.model}
            done = [(batch[j][1], batch[j][0]["uid"], v) for j, v in got.items()]
            self.cache.put_many([(k, v) for k, _, v in done])
            return done

        with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            futs = {ex.submit(run, b): b for b in batches}
            for f in as_completed(futs):
                try:
                    for k, uid, v in f.result():
                        for u in same_key.get(k, [uid]):
                            out[u] = v
                except QuotaError as e:
                    self.stats["failed_batches"] += 1
                    prov.quota_failures += 1
                    if prov.quota_failures >= 2 and not prov.disabled:
                        prov.disabled = True
                        log.warning("%s: quota exhausted, provider switched off for this run (%s)",
                                    prov.model, str(e)[:120])
                except Exception as e:
                    self.stats["failed_batches"] += 1
                    log.warning("%s batch failed: %s", prov.model, str(e)[:200])
        if self.budget.exhausted():
            log.warning("LLM cost cap reached ($%.2f); remaining units use offline layers", self.budget.spent)
        return out

    # ------------------------------------------------------------------ main
    def profile_suppliers(self, items: list[dict]) -> dict[str, dict]:
        """items: [{'uid': supplier_key, 'payload': {supplier, top_lines, ...}}]."""
        instr = ("Each item is a SUPPLIER with its largest purchase lines (description, line count, share "
                 "of spend). Choose the single leaf category that best describes most of this supplier's "
                 "spend. Return one result per item `i`.")
        res = {}
        chain = [p for p in [self.primary, *self.fallbacks] if p and p.available()]
        remaining = items
        for prov in chain:
            got = self._run_batches(prov, remaining, "supplier", instruction=instr)
            res.update(got)
            remaining = [it for it in remaining if it["uid"] not in res]
            if not remaining:
                break
        return res

    def classify_units(self, items: list[dict], vote_threshold: float = 0.8) -> dict[str, dict]:
        """items: [{'uid', 'payload'}]. Returns uid -> {node, confidence, model, agree}."""
        res: dict[str, dict] = {}
        chain = [p for p in [self.primary, *self.fallbacks] if p and p.available()]
        remaining = items
        for prov in chain:
            got = self._run_batches(prov, remaining, "primary")
            res.update(got)
            remaining = [it for it in remaining if it["uid"] not in res]
            if not remaining:
                break
        # second opinion from the other vendor on uncertain units
        if self.second and self.second.available() and res:
            unsure = [it for it in items if it["uid"] in res and res[it["uid"]]["confidence"] < vote_threshold]
            unsure.sort(key=lambda it: res[it["uid"]]["confidence"])
            unsure = unsure[: int(self.cfg.get("second_max_units", 400))]
            if unsure:
                sec = self._run_batches(self.second, unsure, "second")
                disagree = []
                for it in unsure:
                    a, b = res[it["uid"]], sec.get(it["uid"])
                    if not b:
                        continue
                    if a["node"] == b["node"]:
                        a["confidence"] = min(0.99, 1 - (1 - a["confidence"]) * (1 - b["confidence"]) ** 0.5)
                        a["agree"] = True
                    else:
                        a["alt"] = b
                        disagree.append(it)
                if disagree and self.judge and self.judge.available():
                    jitems = []
                    for it in disagree:
                        a = res[it["uid"]]
                        p = dict(it["payload"])
                        p["candidates"] = [self.tax.nodes[a["node"]].label, self.tax.nodes[a["alt"]["node"]].label]
                        p["candidate_ids"] = [a["node"], a["alt"]["node"]]
                        jitems.append({"uid": it["uid"], "payload": p})
                    judged = self._run_batches(self.judge, jitems, "judge")
                    for uid, v in judged.items():
                        v["judged"] = True
                        res[uid] = v
        self.stats["classified"] = len(res)
        return res

    def summary(self) -> dict:
        return {**self.stats, "calls": self.budget.calls, "cost_usd": round(self.budget.spent, 4),
                "tokens_in": self.budget.tokens_in, "tokens_out": self.budget.tokens_out,
                "tokens_cache_read": self.budget.tokens_cache_read,
                "primary": self.primary.model if self.primary else None,
                "second": self.second.model if self.second else None,
                "judge": self.judge.model if self.judge else None}


def schema_enum(schema: dict) -> set:
    return set(schema["properties"]["results"]["items"]["properties"]["node"]["enum"])
