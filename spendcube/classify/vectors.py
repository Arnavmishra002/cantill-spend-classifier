"""Text vectors for retrieval and supervised learning.

Sparse: TF-IDF on words (1-2 grams) + character 3-5 grams (robust to the
abbreviations/OCR noise in procurement text, e.g. 'ANCHR', 'BRG', 'C0NVEYOR').
Dense:  Gemini embeddings when a key is available (cached on disk), otherwise
LSA (TruncatedSVD of the TF-IDF matrix). Dense vectors feed FAISS for k-nearest
neighbour search over labelled / pseudo-labelled units.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from ..textnorm import norm_key

log = logging.getLogger(__name__)

try:  # FAISS is optional: numpy brute force gives identical results for small sets
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None


def prep(text: str) -> str:
    return norm_key(text, ocr=True).lower()


class SparseVectorizer:
    def __init__(self, max_features=200_000):
        self.word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True,
                                    token_pattern=r"(?u)\b[a-z0-9][a-z0-9]+\b", max_features=max_features)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True,
                                    max_features=max_features)

    def fit(self, texts):
        t = [prep(x) for x in texts]
        self.word.fit(t)
        self.char.fit(t)
        return self

    def transform(self, texts):
        t = [prep(x) for x in texts]
        m = sparse.hstack([self.word.transform(t), self.char.transform(t) * 0.7]).tocsr()
        return normalize(m)


class KNNIndex:
    """Inner-product kNN over L2-normalised dense vectors (FAISS if available)."""

    def __init__(self, vecs: np.ndarray):
        self.vecs = np.ascontiguousarray(vecs.astype("float32"))
        self.index = None
        if faiss is not None and len(self.vecs):
            self.index = faiss.IndexFlatIP(self.vecs.shape[1])
            self.index.add(self.vecs)

    def search(self, q: np.ndarray, k: int):
        q = np.ascontiguousarray(q.astype("float32"))
        k = min(k, len(self.vecs))
        if k == 0:
            return np.zeros((len(q), 0)), np.zeros((len(q), 0), dtype=int)
        if self.index is not None:
            return self.index.search(q, k)
        sims = q @ self.vecs.T
        idx = np.argsort(-sims, axis=1)[:, :k]
        return np.take_along_axis(sims, idx, axis=1), idx


class EmbeddingCache:
    """Disk cache for API embeddings: sha1(model|text) -> vector."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data: dict[str, list[float]] = {}
        self.lock = threading.Lock()
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                try:
                    k, v = json.loads(line)
                    self.data[k] = v
                except Exception:
                    continue

    @staticmethod
    def key(model, text):
        return hashlib.sha1(f"{model}|{text}".encode()).hexdigest()

    def put_many(self, items):
        with self.lock, self.path.open("a") as f:
            for k, v in items:
                self.data[k] = v
                f.write(json.dumps([k, v]) + "\n")


def gemini_embed(texts: list[str], model: str, dim: int, cache: EmbeddingCache,
                 task: str = "CLASSIFICATION") -> np.ndarray | None:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        return None
    try:
        from google import genai
        from google.genai import types
    except Exception:
        return None
    client = genai.Client(api_key=key)
    out = [None] * len(texts)
    todo = []
    for i, t in enumerate(texts):
        k = EmbeddingCache.key(f"{model}:{dim}:{task}", t)
        if k in cache.data:
            out[i] = cache.data[k]
        else:
            todo.append((i, t, k))
    B = 100
    for s in range(0, len(todo), B):
        chunk = todo[s:s + B]
        for attempt in range(6):
            try:
                r = client.models.embed_content(
                    model=model, contents=[t for _, t, _ in chunk],
                    config=types.EmbedContentConfig(output_dimensionality=dim, task_type=task))
                vecs = [e.values for e in r.embeddings]
                cache.put_many([(k, v) for (_, _, k), v in zip(chunk, vecs)])
                for (i, _, _), v in zip(chunk, vecs):
                    out[i] = v
                break
            except Exception as e:  # rate limit / transient
                wait = min(60, 2 ** attempt * 2)
                log.warning("gemini embed failed (%s); retry in %ss", str(e)[:120], wait)
                time.sleep(wait)
        else:
            log.warning("gemini embeddings unavailable; falling back to local vectors")
            return None
    return normalize(np.array(out, dtype="float32"))


class DenseEncoder:
    """Dense text vectors: Gemini embeddings if configured+available, else LSA."""

    def __init__(self, cfg: dict, cache_dir: Path, corpus: list[str]):
        ecfg = cfg.get("classification", {}).get("embeddings", {})
        self.provider = ecfg.get("provider", "auto")
        self.model = ecfg.get("gemini_model", "gemini-embedding-001")
        self.dim = int(ecfg.get("dim", 256))
        self.cache = EmbeddingCache(Path(cache_dir) / "embeddings.jsonl")
        self.sparse = SparseVectorizer().fit(corpus)
        n_comp = max(2, min(256, self.sparse.transform(corpus[:1]).shape[1] - 1, len(corpus) - 1))
        self.svd = TruncatedSVD(n_components=n_comp, random_state=0).fit(self.sparse.transform(corpus))
        self.mode = "lsa"
        if self.provider in ("auto", "gemini"):
            probe = gemini_embed(["probe"], self.model, self.dim, self.cache)
            if probe is not None:
                self.mode = "gemini"
        log.info("dense encoder: %s", self.mode)

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.mode == "gemini":
            v = gemini_embed(texts, self.model, self.dim, self.cache)
            if v is not None:
                return v
            self.mode = "lsa"
        return normalize(self.svd.transform(self.sparse.transform(texts))).astype("float32")
