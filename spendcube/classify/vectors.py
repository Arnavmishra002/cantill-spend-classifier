"""Text vectors for retrieval and supervised learning (all local, no API).

Sparse: TF-IDF on words (1-2 grams) + character 3-5 grams, robust to the
abbreviations and OCR noise in procurement text ('ANCHR', 'BRG', 'C0NVEYOR').
Dense:  LSA (TruncatedSVD of the TF-IDF matrix), L2-normalised, searched with
FAISS (inner product) for k-nearest labelled examples. Numpy brute force is used
when FAISS is not installed; results are identical.
"""
from __future__ import annotations

import logging

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from ..textnorm import norm_key

log = logging.getLogger(__name__)

try:  # optional dependency
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


class DenseEncoder:
    """LSA vectors: TF-IDF (fit on the company's own lines) reduced with SVD."""

    def __init__(self, cfg: dict, corpus: list[str]):
        dim = int(cfg.get("classification", {}).get("embeddings", {}).get("dim", 256))
        self.sparse = SparseVectorizer().fit(corpus)
        X = self.sparse.transform(corpus)
        n_comp = max(2, min(dim, X.shape[1] - 1, len(corpus) - 1))
        self.svd = TruncatedSVD(n_components=n_comp, random_state=0).fit(X)
        self.mode = "lsa"

    def encode(self, texts: list[str]) -> np.ndarray:
        return normalize(self.svd.transform(self.sparse.transform(texts))).astype("float32")
