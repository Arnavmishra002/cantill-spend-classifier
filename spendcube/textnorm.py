"""Text, number and date normalisation helpers (pure functions, no company logic)."""
from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- text

_WS = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^0-9A-Z]+")
# OCR confusions inside alphabetic words: 0->O, 1->I, 5->S, 8->B, 3->E, 6->G, 2->Z
_OCR_MAP = str.maketrans({"0": "O", "1": "I", "5": "S", "8": "B", "3": "E", "6": "G", "2": "Z", "|": "I"})
_HAS_ALPHA = re.compile(r"[A-Z]")


def clean_text(s) -> str:
    """Unicode NFKC, strip control chars, collapse whitespace. Keeps case."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    s = "".join(ch for ch in s if unicodedata.category(ch)[0] != "C" or ch in "\t\n")
    return _WS.sub(" ", s).strip()


def ascii_fold(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")


_INNER_DIGIT = re.compile(r"(?<=[A-Z])[0-8|](?=[A-Z])")
_LEAD_DIGIT = re.compile(r"^[0158](?=[A-Z]{3,}$)")


def ocr_fold_token(tok: str) -> str:
    """Undo OCR digit/letter swaps only where unambiguous: a digit between two
    letters ('C0NVEYOR', 'CHA1N', 'RU8BER') or a single leading digit before 3+
    letters ('5TEEL', '0BJECT'). Sizes and codes ('2IN', '10MM', '4510047148') stay."""
    if not _HAS_ALPHA.search(tok):
        return tok
    tok = _INNER_DIGIT.sub(lambda m: m.group(0).translate(_OCR_MAP), tok)
    return _LEAD_DIGIT.sub(lambda m: m.group(0).translate(_OCR_MAP), tok)


def norm_key(s: str, ocr: bool = True) -> str:
    """Aggressive comparison key: ascii, upper, OCR-folded, alnum tokens joined by space."""
    s = ascii_fold(clean_text(s)).upper()
    toks = [t for t in _NON_ALNUM.split(s) if t]
    if ocr:
        toks = [ocr_fold_token(t) for t in toks]
    return " ".join(toks)


def compact_key(s: str, ocr: bool = True) -> str:
    """norm_key without spaces: 'KEL MONT' == 'KELMONT'."""
    return norm_key(s, ocr).replace(" ", "")


def is_blank(s) -> bool:
    return s is None or (isinstance(s, float) and np.isnan(s)) or str(s).strip() == ""


# --------------------------------------------------------------------------- numbers

_CURRENCY_TOKENS = re.compile(
    r"(?i)(R\s?(?=\d)|ZAR|USD|EUR|GBP|INR|Rs\.?|₹|\$|€|£|¥|CHF|AUD|CAD)"
)


def parse_number(x) -> float:
    """Parse messy numeric strings.

    Handles: '1,234.56', '1.234,56', '1,00,000' (Indian grouping), '1 234,56',
    currency symbols/codes, '(123.45)' and '123.45-' negatives, '15%' (returned as 15).
    Returns NaN when nothing numeric can be read.
    """
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = clean_text(x)
    if not s:
        return np.nan
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg, s = True, s[1:-1]
    s = _CURRENCY_TOKENS.sub("", s).replace("%", "").strip()
    if s.endswith("-"):
        neg, s = True, s[:-1]
    if s.startswith("-"):
        neg, s = (not neg), s[1:]
    elif s.startswith("+"):
        s = s[1:]
    s = s.replace(" ", "").replace(" ", "").replace("'", "")
    if not s:
        return np.nan
    has_c, has_d = "," in s, "." in s
    if has_c and has_d:
        # the right-most separator is the decimal point
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif has_c:
        parts = s.split(",")
        # '1,234' / '1,00,000' / '12,345,678' -> grouping; '12,5' / '0,15' -> decimal comma
        if len(parts[-1]) == 3 and all(p.isdigit() for p in parts):
            s = s.replace(",", "")
        elif len(parts) == 2:
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")
    elif has_d and s.count(".") > 1:
        s = s.replace(".", "")  # '1.234.567' grouping
    try:
        v = float(s)
    except ValueError:
        return np.nan
    return -v if neg else v


def parse_number_series(s: pd.Series) -> pd.Series:
    fast = pd.to_numeric(s, errors="coerce")
    need = fast.isna() & s.astype(str).str.strip().ne("")
    if need.any():
        fast.loc[need] = s[need].map(parse_number)
    return fast.astype(float)


def parse_rate(x, default=np.nan) -> float:
    """Tax rate: '15', '15%', '0.15' -> 0.15. Blank -> default."""
    v = parse_number(x)
    if np.isnan(v):
        return default
    return v / 100.0 if v > 1.0 else v


# ----------------------------------------------------------------------------- dates

_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_EXCEL_EPOCH = date(1899, 12, 30)


def _mk(y, m, d):
    try:
        return date(int(y), int(m), int(d))
    except (ValueError, TypeError):
        return None


def parse_date(x, dayfirst: bool = True):
    """Parse one date in any common layout. Returns datetime.date or None."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, (datetime, pd.Timestamp)):
        return x.date()
    if isinstance(x, date):
        return x
    s = clean_text(x)
    if not s:
        return None
    s = s.split("T")[0] if re.match(r"^\d{4}-\d{2}-\d{2}T", s) else s
    s = re.sub(r"\s+\d{1,2}:\d{2}(:\d{2})?(\.\d+)?\s*(AM|PM)?$", "", s, flags=re.I)
    m = re.fullmatch(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", s)
    if m:
        return _mk(*m.groups())
    m = re.fullmatch(r"(\d{8})", s)
    if m:  # 20240131
        return _mk(s[:4], s[4:6], s[6:])
    m = re.fullmatch(r"(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})", s)
    if m:
        a, b, y = m.groups()
        y = int(y)
        if y < 100:
            y += 2000 if y < 70 else 1900
        a, b = int(a), int(b)
        if a > 12 and b <= 12:
            return _mk(y, b, a)
        if b > 12 and a <= 12:
            return _mk(y, a, b)
        return _mk(y, b, a) if dayfirst else _mk(y, a, b)
    m = re.fullmatch(r"(\d{1,2})[-/ .]([A-Za-z]{3,9})[-/ .,]*(\d{2,4})", s)
    if m:
        d, mon, y = m.groups()
        mi = _MONTHS.get(mon[:3].lower())
        y = int(y)
        if y < 100:
            y += 2000 if y < 70 else 1900
        return _mk(y, mi, d) if mi else None
    m = re.fullmatch(r"([A-Za-z]{3,9})[-/ .]+(\d{1,2})[,]?[-/ .]+(\d{2,4})", s)
    if m:
        mon, d, y = m.groups()
        mi = _MONTHS.get(mon[:3].lower())
        y = int(y)
        if y < 100:
            y += 2000 if y < 70 else 1900
        return _mk(y, mi, d) if mi else None
    m = re.fullmatch(r"\d{5}(\.\d+)?", s)
    if m:  # Excel serial
        v = float(s)
        if 20000 < v < 80000:
            return _EXCEL_EPOCH + timedelta(days=int(v))
    return None


def parse_date_series(s: pd.Series, dayfirst: bool = True) -> pd.Series:
    """Vectorised over unique values (dates repeat a lot)."""
    uniq = pd.unique(s.astype(str))
    mp = {u: parse_date(u, dayfirst) for u in uniq}
    out = s.astype(str).map(mp)
    return pd.to_datetime(out, errors="coerce")
