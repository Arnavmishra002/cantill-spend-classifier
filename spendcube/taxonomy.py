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
        lower = {c.lower(): c for c in df.columns}
        idc = tcfg.get("id_column") or lower.get("id") or lower.get("node_id") or lower.get("code")
        pc = tcfg.get("parent_column") or lower.get("parent_id") or lower.get("parent")
        nc = tcfg.get("name_column") or lower.get("name") or lower.get("category") or lower.get("label")
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
