"""LLM classification layer (Anthropic Claude only), safe to run unattended.

Design
  * Units are sent in batches (default 25 per request) with the full category
    tree in a cached system prompt. The model returns JSON constrained by a
    schema whose `node` field is an enum of valid leaf IDs, so an invalid
    category is impossible.
  * Results are cached on disk, keyed by a hash of (model, prompt version, tree,
    unit payload). Re-runs are fast, free and deterministic.
  * Retries with exponential backoff on rate limits and server errors (SDK), a
    run-wide cost cap, and an offline fallback for anything not classified, so a
    run always finishes with a valid file.
  * Roles: `primary` (Claude Opus) classifies every unit; `second` (Claude Sonnet)
    gives an independent second opinion on uncertain and high-spend units; when
    they disagree the `judge` (Claude Opus) sees both candidates and decides.
    `fallbacks` take over if the primary model is unavailable.
The API key is read from the environment (or a local, git-ignored .env), never from config.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from ..config import DEFAULTS

log = logging.getLogger(__name__)

PROMPT_VERSION = "v6"

# USD per 1M tokens (input, output, cache read). Override in config llm.prices.
DEFAULT_PRICES = {
    "claude-opus-5-5": (4.00, 20.00, 0.20),
    "claude-sonnet-5-5": (2.00, 10.00, 0.10),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
}

SYSTEM_TEMPLATE = """You are a procurement spend analyst. You classify purchase lines into one company's category tree.

Company context: {context}

Rules:
- Pick exactly one category ID for every line, always a leaf ID from the tree below.
- The line description is the strongest evidence. Supplier name, GL account and context fields are supporting evidence.
- Descriptions are short, abbreviated (e.g. BRG=bearing, VLV=valve, ASSY=assembly, REPR=repair, MNTN=maintenance, SPRT=support, CNVY=conveyor), OCR-damaged (0/O, 5/S, 8/B swapped) or in other languages; read through that.
- Each line points to a supplier block `s` with the supplier name, its `profile` (the category that best describes the supplier's overall business with this company) and some of its `other_lines`.
- When the description is blank or generic ("Product", "Parts", "No Object"), infer from the supplier block and the account.
- Many companies categorise spend by the supplier's line of business or by the equipment fleet a part is for. When a line is ambiguous (a generic spare part, a repair, a service), prefer the supplier profile; override it when the line clearly belongs elsewhere.
- `hints` (if present) come from this company's labelled history: "model N0xx p0.42" is a statistical model's suggestion with its probability; "like '<text>' -> N0xx" is a similar labelled line and its category. Use them as evidence, not as orders.
- `candidates` (if present) are the answers of two other analysts who disagreed; pick the better one unless both are clearly wrong.
- A leaf marked [general] is the general bucket of its branch (it repeats the branch name, or is named Other, General or similar). Companies file most of a branch's spend there and use the specific sibling leaves only when the line clearly names that specific category. When a line belongs to the branch but does not clearly name one of the specific leaves, choose the [general] leaf.
- confidence = your probability (0 to 1) that the chosen leaf is exactly right.

Category tree (ID: L1 > L2 > ... > leaf){desc_note}:
{tree}
"""

# single source of truth for the default roles: the config defaults
DEFAULT_LLM = DEFAULTS["classification"]["llm"]


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

    def add(self, usd, tin, tout, tcache=0):
        with self.lock:
            self.spent += usd
            self.calls += 1
            self.tokens_in += tin
            self.tokens_out += tout
            self.tokens_cache_read += tcache

    reserved: float = 0.0

    def exhausted(self) -> bool:
        return self.spent >= self.cap_usd

    def estimate(self) -> float:
        """Expected cost of the next call: the running average, or a cautious first guess."""
        with self.lock:
            return (self.spent / self.calls) if self.calls else 0.10

    def try_reserve(self, est: float) -> bool:
        with self.lock:
            if self.spent + self.reserved + est > self.cap_usd:
                return False
            self.reserved += est
            return True

    def release(self, est: float):
        with self.lock:
            self.reserved = max(0.0, self.reserved - est)


class AnthropicProvider:
    vendor = "anthropic"

    def __init__(self, model: str, prices: dict, effort: str = "low", base_url: str | None = None):
        self.model = model
        self.price = prices.get(model, (5.0, 25.0, 0.5))
        self.effort = effort
        # explicit base URL: an inherited ANTHROPIC_BASE_URL (e.g. from a dev tool) must not reroute calls
        self.base_url = base_url or "https://api.anthropic.com"
        self.disabled = False      # set when the health check fails or auth fails mid-run
        self._client = None
        self._lock = threading.Lock()

    def cost(self, tin, tout, tcache=0):
        pi, po, pc = self.price
        return (tin * pi + tout * po + tcache * pc) / 1e6

    def available(self) -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY")) and not self.disabled

    @property
    def client(self):
        with self._lock:   # one client per provider, created once
            if self._client is None:
                import anthropic
                self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"],
                                                   base_url=self.base_url, max_retries=5, timeout=180.0)
            return self._client

    def health(self) -> str:
        """'ok' or a one-line reason. Costs a few tokens."""
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return "no ANTHROPIC_API_KEY in environment or .env"
        try:
            r = self.client.with_options(max_retries=1, timeout=60.0).messages.create(
                model=self.model, max_tokens=64, output_config={"effort": "low"},
                messages=[{"role": "user", "content": "Reply with the single word OK."}])
            return "ok" if r.content else "empty response"
        except Exception as e:  # auth, model not found, network, overload
            return f"{type(e).__name__}: {str(e)[:120]}"

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
        return json.loads(text), usd, tin, u.output_tokens, tcache


def make_provider(spec: dict | None, prices: dict) -> AnthropicProvider | None:
    if not spec:
        return None
    v = spec.get("vendor", "anthropic")
    if v != "anthropic":
        raise ValueError(f"LLM vendor `{v}` is not supported; this tool uses Claude (vendor: anthropic)")
    return AnthropicProvider(spec["model"], prices, effort=spec.get("effort", "low"), base_url=spec.get("base_url"))


def llm_settings(cfg: dict) -> dict:
    return {**DEFAULT_LLM, **(cfg.get("classification", {}).get("llm") or {})}


def check_llm_health(cfg: dict) -> dict[str, str]:
    """Health of every distinct model the config uses: model -> 'ok' or a reason."""
    lcfg = llm_settings(cfg)
    prices = {**DEFAULT_PRICES, **{k: tuple(v) for k, v in (lcfg.get("prices") or {}).items()}}
    out: dict[str, str] = {}
    for spec in [lcfg.get("primary"), lcfg.get("tail"), lcfg.get("second"), lcfg.get("judge"),
                 *(lcfg.get("fallbacks") or [])]:
        p = make_provider(spec, prices)
        if p and p.model not in out:
            out[p.model] = p.health()
    return out


def tree_block(tax, leaf_ids) -> tuple[str, str]:
    lines = []
    has_desc = False
    for nid in leaf_ids:
        n = tax.nodes[nid]
        path = tax.display_path(nid)
        if n.description:
            has_desc = True
            lines.append(f"{nid}: {path} [{n.description}]")
        else:
            lines.append(f"{nid}: {path}")
    return "\n".join(lines), (" [description]" if has_desc else "")


def outline_block(tax, max_lines: int = 400) -> str:
    """Upper levels of a big tree (every node above the leaves), for orientation only."""
    rows = [" > ".join(n.path) for n in tax.nodes.values() if n.children]
    return "\n".join(rows[:max_lines])


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


def schema_enum(schema: dict) -> set:
    return set(schema["properties"]["results"]["items"]["properties"]["node"]["enum"])


class LLMClassifier:
    def __init__(self, cfg: dict, tax, cache_dir: Path):
        lcfg = llm_settings(cfg)
        self.cfg = lcfg
        self.tax = tax
        prices = {**DEFAULT_PRICES, **{k: tuple(v) for k, v in (lcfg.get("prices") or {}).items()}}
        self.primary = make_provider(lcfg.get("primary"), prices)
        self.second = make_provider(lcfg.get("second"), prices)
        self.judge = make_provider(lcfg.get("judge"), prices)
        self.tail = make_provider(lcfg.get("tail"), prices)        # cheaper model for the low-spend tail
        self.fallbacks = [make_provider(s, prices) for s in (lcfg.get("fallbacks") or [])]
        self.batch_size = int(lcfg.get("batch_size", 25))
        self.concurrency = int(lcfg.get("concurrency", 10))
        self.budget = Budget(float(lcfg.get("cost_cap_usd", 15.0)))
        self.cache = ResultCache(Path(cache_dir) / "llm_results.jsonl")
        self.leaf_ids = list(tax.leaves)
        ctx = cfg.get("classification", {}).get("company_context") or "not given"
        # Big trees: no full tree in the prompt; each batch gets the union of its lines'
        # retrieval shortlists instead (see classify()).
        self.shortlist_mode = len(self.leaf_ids) > int(lcfg.get("shortlist_above_leaves", 400))
        if self.shortlist_mode:
            tree = outline_block(tax)
            self.system = SYSTEM_TEMPLATE.format(context=ctx, tree=tree, desc_note=" (outline only; each "
                                                 "request lists the candidate leaves to choose from)")
        else:
            tree, desc_note = tree_block(tax, self.leaf_ids)
            self.system = SYSTEM_TEMPLATE.format(context=ctx, tree=tree, desc_note=desc_note)
        self.schema = result_schema(self.leaf_ids)
        self.valid = set(self.leaf_ids)
        self.tax_hash = _sha(self.system)[:16]
        self.stats = {"cache_hits": 0, "classified": 0, "failed_batches": 0, "refused_batches": 0,
                      "skipped_for_cap": 0}
        self.health_report: dict[str, str] = {}

    # ------------------------------------------------------------------ utils
    def providers(self) -> list[AnthropicProvider]:
        seen, out = set(), []
        for p in [self.primary, self.second, self.judge, self.tail, *self.fallbacks]:
            if p and id(p) not in seen:
                seen.add(id(p))
                out.append(p)
        return out

    def apply_health(self, health: dict[str, str]):
        """Switch off models that failed the run's start-up health check."""
        self.health_report = dict(health)
        for p in self.providers():
            if health.get(p.model, "ok") != "ok":
                p.disabled = True

    def enabled(self) -> bool:
        provs = [p for p in [self.primary, self.tail, *self.fallbacks] if p]
        return bool(self.cfg.get("enabled", True)) and any(p.available() for p in provs)

    def _key(self, prov, it: dict, role: str) -> str:
        return _sha(f"{PROMPT_VERSION}|{role}|{prov.model}|{self.tax_hash}|"
                    f"{json.dumps(it['payload'], sort_keys=True, ensure_ascii=False)}|"
                    f"{json.dumps(it.get('group_payload') or {}, sort_keys=True, ensure_ascii=False)}|"
                    f"{','.join(it.get('cands') or [])}")

    def _request(self, prov, batch, instruction):
        """Build and send one request. Returns {index: result}."""
        groups, gid, lines = [], {}, []
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
        schema, valid = self.schema, self.valid
        if self.shortlist_mode:
            cand = list(dict.fromkeys(c for it, _ in batch for c in (it.get("cands") or [])))
            cand = cand[: int(self.cfg.get("shortlist_batch_cap", 300))] or self.leaf_ids[:300]
            schema, valid = result_schema(cand), set(cand)
            parts += ["CANDIDATE CATEGORIES (choose only from these):"] + \
                     [f"{c}: {self.tax.display_path(c)}" for c in cand]
        if groups:
            parts += ["SUPPLIERS:"] + [json.dumps(x, ensure_ascii=False) for x in groups] + ["LINES:"]
        parts += [json.dumps(x, ensure_ascii=False) for x in lines]
        res, usd, tin, tout, tcache = prov.classify(self.system, "\n".join(parts), schema)
        self.budget.add(usd, tin, tout, tcache)
        got = {}
        for r in res.get("results", []):
            j = r.get("i")
            if isinstance(j, int) and 0 <= j < len(batch) and r.get("node") in valid:
                conf = float(r.get("confidence", 0.5))
                got[j] = {"node": r["node"], "confidence": max(0.0, min(1.0, conf)), "model": prov.model}
        return got

    def _run_batches(self, prov, items: list[dict], role: str, instruction: str | None = None) -> dict[str, dict]:
        """items: [{'uid', 'payload', 'group'?, 'group_payload'?, 'spend'?, 'cands'?}].
        Returns uid -> {node, confidence}. Batches run in descending spend order, so a cost
        cap only ever drops the low-spend tail."""
        out: dict[str, dict] = {}
        todo = []
        same_key: dict[str, list[str]] = {}
        for it in items:
            k = self._key(prov, it, role)
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
        # one supplier's lines travel together; suppliers ordered by their largest unit's spend
        gmax: dict[str, float] = {}
        for it, _ in todo:
            g = it.get("group") or it["uid"]
            gmax[g] = max(gmax.get(g, 0.0), float(it.get("spend", 0.0)))
        todo.sort(key=lambda x: (-gmax[x[0].get("group") or x[0]["uid"]], x[0].get("group") or "",
                                 -float(x[0].get("spend", 0.0)), x[0]["uid"]))
        batches = [todo[i:i + self.batch_size] for i in range(0, len(todo), self.batch_size)]

        def run(batch):
            if prov.disabled:
                return []
            est = self.budget.estimate()
            if not self.budget.try_reserve(est):        # never start a call that could pass the cap
                self.stats["skipped_for_cap"] += 1
                return []
            try:
                try:
                    got = self._request(prov, batch, instruction)
                except RuntimeError as e:
                    if "refused" not in str(e) or len(batch) <= 1:
                        raise
                    # a refusal is usually caused by one line: retry in pieces of 5
                    self.stats["refused_batches"] += 1
                    got = {}
                    for s0 in range(0, len(batch), 5):
                        piece = batch[s0:s0 + 5]
                        try:
                            sub = self._request(prov, piece, instruction)
                        except RuntimeError:
                            continue            # still refused: these lines go to the fallback model
                        got.update({s0 + j: v for j, v in sub.items()})
            finally:
                self.budget.release(est)
            done = [(batch[j][1], batch[j][0]["uid"], v) for j, v in got.items()]
            self.cache.put_many([(k, v) for k, _, v in done])
            return done

        with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            futs = [ex.submit(run, b) for b in batches]
            for f in as_completed(futs):
                try:
                    for k, uid, v in f.result():
                        for u in same_key.get(k, [uid]):
                            out[u] = v
                except Exception as e:
                    self.stats["failed_batches"] += 1
                    msg = str(e)
                    if "authentication" in msg.lower() or "401" in msg or "permission" in msg.lower():
                        prov.disabled = True     # no point sending more requests this run
                    log.warning("%s batch failed: %s", prov.model, msg[:200])
        if self.budget.exhausted() or self.stats["skipped_for_cap"]:
            log.warning("LLM cost cap reached ($%.2f of $%.2f); remaining units use offline layers",
                        self.budget.spent, self.budget.cap_usd)
        return out

    def _chain(self, items: list[dict], role: str, instruction: str | None = None,
               first=None) -> dict[str, dict]:
        """`first` model (default primary), then fallbacks; each retries omitted lines once."""
        res: dict[str, dict] = {}
        remaining = items
        chain = [p for p in [first or self.primary, *self.fallbacks, self.primary] if p]
        seen = set()
        for prov in chain:
            if id(prov) in seen:
                continue
            seen.add(id(prov))
            for _ in range(2):
                if not remaining or not prov.available():
                    break
                res.update(self._run_batches(prov, remaining, role, instruction))
                remaining = [it for it in remaining if it["uid"] not in res]
            if not remaining:
                break
        return res

    # ------------------------------------------------------------------ main
    def profile_suppliers(self, items: list[dict]) -> dict[str, dict]:
        """items: [{'uid': supplier_key, 'payload': {supplier, top_lines, ...}}]."""
        instr = ("Each item is a SUPPLIER with its largest purchase lines (description, line count, share "
                 "of spend). Choose the single leaf category that best describes most of this supplier's "
                 "spend. Return one result per item `i`.")
        return self._chain(items, "supplier", instr)

    def classify_units(self, items: list[dict], vote_threshold: float = 0.7,
                       priority: list[str] | None = None) -> dict[str, dict]:
        """items: [{'uid', 'payload', 'group', 'group_payload', 'spend', 'cands'?}].
        The units holding the top `primary_spend_share` of spend go to the primary model,
        the tail to the cheaper `tail` model. priority: uids that always get a second opinion."""
        share = float(self.cfg.get("primary_spend_share", 1.0))
        head, tail = items, []
        if self.tail and share < 1.0 and items:
            order = sorted(items, key=lambda it: -float(it.get("spend", 0.0)))
            tot = sum(max(float(it.get("spend", 0.0)), 0.0) for it in order) or 1.0
            acc, head = 0.0, []
            for it in order:
                if acc / tot < share:
                    head.append(it)
                    acc += max(float(it.get("spend", 0.0)), 0.0)
                else:
                    tail.append(it)
        res = self._chain(head, "primary")
        if tail:
            res.update(self._chain(tail, "primary", first=self.tail))
        self.stats["units_primary_model"] = len(head)
        self.stats["units_tail_model"] = len(tail)
        if self.second and self.second.available() and res:
            pr = set(priority or [])
            unsure = [it for it in items if it["uid"] in res and
                      (it["uid"] in pr or res[it["uid"]]["confidence"] < vote_threshold)]
            unsure.sort(key=lambda it: (it["uid"] not in pr, res[it["uid"]]["confidence"]))
            unsure = unsure[: int(self.cfg.get("second_max_units", 1500))]
            sec = self._run_batches(self.second, unsure, "second") if unsure else {}
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
            self.stats["second_opinions"] = len(sec)
            self.stats["disagreements"] = len(disagree)
            if disagree and self.judge and self.judge.available():
                jitems = []
                for it in disagree:
                    a = res[it["uid"]]
                    p = dict(it["payload"])
                    p["candidates"] = [f"{a['node']} {self.tax.nodes[a['node']].label}",
                                       f"{a['alt']['node']} {self.tax.nodes[a['alt']['node']].label}"]
                    cands = list(dict.fromkeys([a["node"], a["alt"]["node"], *(it.get("cands") or [])]))
                    jitems.append({**it, "payload": p, "cands": cands})
                instr = ("Two analysts disagreed on each line below; their answers are in `candidates`. "
                         "Lines refer to their supplier `s`. Return one result per line `i`.")
                judged = self._run_batches(self.judge, jitems, "judge", instr)
                for uid, v in judged.items():
                    res[uid] = {**v, "judged": True}
                self.stats["judged"] = len(judged)
        self.stats["classified"] = len(res)
        return res

    def summary(self) -> dict:
        return {**self.stats, "calls": self.budget.calls, "cost_usd": round(self.budget.spent, 4),
                "tokens_in": self.budget.tokens_in, "tokens_out": self.budget.tokens_out,
                "tokens_cache_read": self.budget.tokens_cache_read,
                "primary": self.primary.model if self.primary else None,
                "tail": self.tail.model if self.tail else None,
                "second": self.second.model if self.second else None,
                "judge": self.judge.model if self.judge else None,
                "shortlist_mode": self.shortlist_mode,
                "health": self.health_report}
