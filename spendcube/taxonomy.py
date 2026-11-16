"""Load a category tree from file (read at run time; no category names in code).

Supported layouts (auto-detected unless the config says):
  wide          one row per leaf path, one column per level (L1..Ln)
  parent_child  id, parent_id, name (+ optional level / description)
Optional description / keyword columns are kept and used as retrieval text.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .ingest import read_table
from .paths import resolve_file
from .textnorm import clean_text


@dataclass
class Node:
    id: str
    path: tuple[str, ...]          # names from L1 down to this node
    depth: int                     # 1-based
    children: list[str] = field(default_factory=list)
    description: str = ""
    general: bool = False          # the general bucket among its siblings (see Taxonomy._mark_general)

    @property
    def name(self) -> str:
        return self.path[-1]

    @property
    def label(self) -> str:
        return " > ".join(self.path)


class Taxonomy:
    def __init__(self, nodes: dict[str, Node], max_depth: int):
        self.nodes = nodes
        self.max_depth = max_depth
        self.by_path = {n.path: n.id for n in nodes.values()}
        self.leaves = [n.id for n in nodes.values() if not n.children]
        self._mark_general()

    def _mark_general(self):
        """Among siblings, a leaf that repeats its parent's name ('Pumps > Pumps') or is
        named Other / General / Miscellaneous / Unclassified is the branch's general bucket.
        Trees padded to a fixed depth use it for spend that is not clearly a specific leaf."""
        def norm(x):
            return re.sub(r"[^a-z0-9]", "", x.lower())
        rx = re.compile(r"(?i)\b(other|others|general|misc|miscellaneous|unclassified|various|sundry|not elsewhere)\b")
        for parent in self.nodes.values():
            kids = [self.nodes[c] for c in parent.children]
            if len(kids) < 2:
                continue
            same = [k for k in kids if not k.children and norm(k.name) == norm(parent.name)]
            other = [k for k in kids if not k.children and rx.search(k.name)]
            pick = same[:1] or (other if len(other) == 1 else [])
            for k in pick:
                k.general = True

    def general_sibling(self, nid: str) -> str | None:
        """The general leaf of nid's branch, if nid is a specific leaf in such a branch."""
        n = self.nodes[nid]
        if n.general or n.depth < 2:
            return None
        parent = self.nodes[self.by_path[n.path[:-1]]]
        for c in parent.children:
            if self.nodes[c].general:
                return c
        return None

    # ------------------------------------------------------------------ build
    @classmethod
    def from_paths(cls, paths: list[tuple[str, ...]], descriptions: dict | None = None) -> "Taxonomy":
        nodes: dict[str, Node] = {}
        by_path: dict[tuple, str] = {}
        max_depth = 0
        for p in paths:
            p = tuple(x for x in p if x)
            # collapse a trailing level that repeats its parent (e.g. L2 == L3) only
            # for retrieval text; the stored path keeps every level as given
            for d in range(1, len(p) + 1):
                sub = p[:d]
                if sub not in by_path:
                    nid = f"N{len(nodes) + 1:03d}"
                    by_path[sub] = nid
                    nodes[nid] = Node(nid, sub, d)
                    if d > 1:
                        nodes[by_path[sub[:-1]]].children.append(nid)
            max_depth = max(max_depth, len(p))
        if descriptions:
            for p, desc in descriptions.items():
                if p in by_path:
                    nodes[by_path[p]].description = desc
        return cls(nodes, max_depth)

    def path_of(self, nid: str) -> tuple[str, ...]:
        return self.nodes[nid].path

    def is_leaf(self, nid: str) -> bool:
        return not self.nodes[nid].children

    def ancestors(self, nid: str) -> list[str]:
        p = self.nodes[nid].path
        return [self.by_path[p[:d]] for d in range(1, len(p))]

    def node_for_labels(self, levels: list[str]) -> str | None:
        p = tuple(x for x in levels if x)
        while p:
            if p in self.by_path:
                return self.by_path[p]
            p = p[:-1]
        return None

    def display_path(self, nid: str) -> str:
        """Full path for prompts; a general leaf is marked so a model can see its role."""
        n = self.nodes[nid]
        return " > ".join(n.path) + (" [general]" if n.general else "")

    def leaf_text(self, nid: str) -> str:
        """Retrieval text: de-duplicated path + description."""
        n = self.nodes[nid]
        parts = list(dict.fromkeys(n.path))
        t = " > ".join(parts)
        return f"{t}. {n.description}" if n.description else t

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for n in self.nodes.values():
            rows.append({"id": n.id, "depth": n.depth, "leaf": not n.children, "path": n.label})
        return pd.DataFrame(rows)


def _level_number(col: str):
    c = col.strip().lower()
    m = re.search(r"(?:level|lvl|l)[\s_-]?(\d)", c)
    if m and any(k in c for k in ("level", "lvl", "cat", "l1", "l2", "l3", "l4", "l5", "l")):
        return int(m.group(1))
    return None


def guess_parent_child_columns(cols):
    low = [(c, c.strip().lower()) for c in cols]
    pc = next((c for c, l in low if "parent" in l), None)
    idc = next((c for c, l in low if c != pc and l in ("id", "node_id", "code", "node", "category_id",
                                                           "node_code", "category_code")), None) or \
        next((c for c, l in low if c != pc and (l.endswith("id") or l.endswith("code"))), None)
    nc = next((c for c, l in low if c not in (pc, idc) and any(k in l for k in ("name", "label", "title"))), None) or \
        next((c for c, l in low if c not in (pc, idc) and "category" in l), None)
    return idc, pc, nc


def load_taxonomy(cfg: dict, data_dir: Path) -> Taxonomy:
    tcfg = cfg["taxonomy"]
    df = read_table(resolve_file(tcfg["file"], cfg, data_dir), sheet=tcfg.get("sheet"))
    df.columns = [clean_text(c) for c in df.columns]
    df = df.map(clean_text)
    fmt = tcfg.get("format", "auto")
    cols = tcfg.get("level_columns")
    desc_col = tcfg.get("description_column")
    if fmt == "auto":
        lower = {c.lower(): c for c in df.columns}
        if ("parent_id" in lower or "parent" in lower) and not cols:
            fmt = "parent_child"
        else:
            fmt = "wide"
    if fmt == "parent_child":
        idc, pc, nc = guess_parent_child_columns(df.columns)
        idc = tcfg.get("id_column") or idc
        pc = tcfg.get("parent_column") or pc
        nc = tcfg.get("name_column") or nc
        if not (idc and pc and nc):
            raise ValueError(f"parent_child taxonomy needs id/parent/name columns; got {list(df.columns)}")
        name = dict(zip(df[idc], df[nc]))
        parent = dict(zip(df[idc], df[pc]))
        descs = dict(zip(df[idc], df[desc_col])) if desc_col else {}
        kids = set(parent.values())

        def path(i, seen=()):
            if not i or i not in name or i in seen:
                return ()
            return path(parent.get(i, ""), seen + (i,)) + (name[i],)
        paths = [path(i) for i in df[idc] if i not in kids] or [path(i) for i in df[idc]]
        d = {path(i): descs.get(i, "") for i in df[idc]} if descs else None
        return Taxonomy.from_paths(paths, d)
    # wide
    if not cols:
        numbered = sorted(((n, c) for c in df.columns if (n := _level_number(c)) is not None))
        cols = [c for _, c in numbered] or [c for c in df.columns if c != desc_col]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"taxonomy level columns {missing} not in {list(df.columns)}")
    paths = [tuple(r) for r in df[cols].itertuples(index=False)]
    descs = None
    if desc_col and desc_col in df.columns:
        descs = {tuple(x for x in r[:-1] if x): r[-1] for r in df[cols + [desc_col]].itertuples(index=False)}
    return Taxonomy.from_paths(paths, descs)
