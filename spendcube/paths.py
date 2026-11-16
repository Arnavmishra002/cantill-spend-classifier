"""Resolve config file names against the data folder.

A config entry may be a file name, a glob pattern or a list of either. The
first match wins, searched in: the run's data folder, then any
`search_dirs` listed in the config (relative to the repo root). This lets the
A config run unchanged on the A2 folder even if file names change slightly,
and lets the tree/labels come from the original kit folder when a new drop does
not repeat them.
"""
from __future__ import annotations

import glob
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def candidate_dirs(cfg: dict, data_dir: Path) -> list[Path]:
    dirs = [Path(data_dir)]
    for d in cfg.get("search_dirs", []) or []:
        p = Path(d)
        dirs.append(p if p.is_absolute() else (REPO_ROOT / p))
    return dirs


def resolve_file(spec, cfg: dict, data_dir: Path, required: bool = True, dirs=None) -> Path | None:
    if spec is None:
        return None
    pats = spec if isinstance(spec, list) else [spec]
    for d in (dirs or candidate_dirs(cfg, data_dir)):
        for pat in pats:
            p = d / pat
            if p.exists():
                return p
            hits = sorted(glob.glob(str(d / pat)))
            if hits:
                return Path(hits[0])
    if required:
        tried = [str(d) for d in (dirs or candidate_dirs(cfg, data_dir))]
        raise FileNotFoundError(f"No file matching {pats} in {tried}")
    return None
