"""Read every source file named in the config and map it to canonical fields.

Output: one DataFrame, one row per input record, with
  - record_id   (prefix + id column, as the output format requires)
  - source      (config source name)
  - canonical fields (strings, still unparsed)
  - raw__<col>  (every original column, so rules can reference anything)
"""
from __future__ import annotations

import csv
import io
import logging
from pathlib import Path

import pandas as pd

from .paths import resolve_file
from .textnorm import clean_text

log = logging.getLogger(__name__)

TEXT_JOIN_FIELDS = {"description", "context", "supplier_name", "gl_description"}
_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")


def read_table(path: Path, sheet=None, delimiter=None, encoding=None) -> pd.DataFrame:
    """Read CSV/TSV/TXT/XLSX as all-string columns, blanks kept as ''."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    suf = path.suffix.lower()
    if suf in (".xlsx", ".xlsm", ".xls"):
        df = pd.read_excel(path, sheet_name=sheet or 0, dtype=str)
        return df.fillna("")
    if suf == ".parquet":
        return pd.read_parquet(path).astype(str).replace({"nan": "", "None": ""})
    raw = path.read_bytes()
    encs = [encoding] if encoding else list(_ENCODINGS)
    text = None
    for enc in encs:
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        text = raw.decode("latin-1", errors="replace")
    if delimiter is None:
        sample = text[:20000]
        try:
            delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
        except csv.Error:
            delimiter = ","
    return pd.read_csv(io.StringIO(text), sep=delimiter, dtype=str, keep_default_na=False,
                       na_filter=False, engine="c")


def _resolve(cols, df_cols, where):
    if cols is None:
        return []
    if isinstance(cols, str):
        cols = [cols]
    missing = [c for c in cols if c not in df_cols]
    if missing:
        raise KeyError(f"{where}: column(s) {missing} not in file. Available: {list(df_cols)}")
    return list(cols)


def ingest_source(src: dict, data_dir: Path, cfg: dict | None = None) -> pd.DataFrame:
    # input records must come from the run's own data folder (never a fallback dir)
    path = resolve_file(src["file"], cfg or {}, data_dir, dirs=[Path(data_dir)])
    df = read_table(path, sheet=src.get("sheet"), delimiter=src.get("delimiter"),
                    encoding=src.get("encoding"))
    df.columns = [clean_text(c) for c in df.columns]
    n = len(df)
    log.info("source %s: %d rows from %s", src["name"], n, path.name)

    rid = src["record_id"]
    if isinstance(rid, str):
        rid = {"column": rid}
    id_col = _resolve(rid["column"], df.columns, f"{src['name']}.record_id")[0]
    out = pd.DataFrame(index=df.index)
    ids = df[id_col].map(clean_text)
    if (ids == "").any():
        # synthesise ids for blank ones so every input record still gets a row
        blank = ids == ""
        ids.loc[blank] = [f"ROW{i+1}" for i in df.index[blank]]
        log.warning("source %s: %d blank ids synthesised", src["name"], int(blank.sum()))
    out["record_id"] = rid.get("prefix", "") + ids
    out["source"] = src["name"]
    out["row_order"] = range(n)

    for field, cols in (src.get("columns") or {}).items():
        cols = _resolve(cols, df.columns, f"{src['name']}.columns.{field}")
        vals = [df[c].map(clean_text) for c in cols]
        if len(vals) == 1:
            out[field] = vals[0]
        elif field in TEXT_JOIN_FIELDS:
            joined = pd.concat(vals, axis=1).apply(
                lambda r: " | ".join(dict.fromkeys(v for v in r if v)), axis=1)
            out[field] = joined
        else:  # first non-blank
            acc = vals[0].copy()
            for v in vals[1:]:
                acc = acc.where(acc != "", v)
            out[field] = acc
    for c in df.columns:
        out[f"raw__{c}"] = df[c]
    # constants from config (e.g. currency for a single-currency file)
    for field, val in (src.get("constants") or {}).items():
        if field not in out or (out[field] == "").all():
            out[field] = str(val)
    return out


def ingest_all(cfg: dict, data_dir: Path) -> pd.DataFrame:
    frames = [ingest_source(s, data_dir, cfg) for s in cfg["sources"]]
    df = pd.concat(frames, ignore_index=True, sort=False)
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].fillna("").astype(str)
    dup = df["record_id"].duplicated(keep=False)
    if dup.any():
        # Make ids unique but keep them traceable; validation reports this.
        log.warning("%d duplicate record_ids in input; suffixing", int(dup.sum()))
        df.loc[dup, "record_id"] = df.loc[dup, "record_id"] + "#" + \
            df[dup].groupby("record_id").cumcount().add(1).astype(str)
    return df
