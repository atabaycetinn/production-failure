"""Retrievers. Baseline: dependency-free BM25 with metadata filters.

The ``Retriever`` protocol is the seam for Phase 1 (hybrid BM25 + dense + rerank).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Protocol

from ..models import DocChunk, RetrievedChunk

_TOKEN = re.compile(r"[a-z0-9][a-z0-9_.\-]*")
_STOP = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was were will with".split()
)


def tokenize(text: str) -> list[str]:
    toks = []
    for t in _TOKEN.findall(text.lower()):
        t = t.strip(".-")
        if t and t not in _STOP:
            toks.append(t)
            # also index sub-parts of identifiers like amount_log_scaled / fraud-preproc
            if "_" in t or "-" in t:
                toks.extend(p for p in re.split(r"[_\-]", t) if p and p not in _STOP)
    return toks


class Retriever(Protocol):
    def search(
        self, query: str, top_k: int = 5, doc_types: list[str] | None = None
    ) -> list[RetrievedChunk]: ...


class BM25Retriever:
    def __init__(self, chunks: list[DocChunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1, self.b = k1, b
        self._tfs: list[Counter[str]] = []
        self._lens: list[int] = []
        df: Counter[str] = Counter()
        for c in chunks:
            toks = tokenize(f"{c.title} {c.section} {c.text}")
            tf = Counter(toks)
            self._tfs.append(tf)
            self._lens.append(len(toks))
            df.update(tf.keys())
        n = max(len(chunks), 1)
        self._avgdl = sum(self._lens) / n if self._lens else 0.0
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def search(self, query: str, top_k: int = 5, doc_types: list[str] | None = None) -> list[RetrievedChunk]:
        q = tokenize(query)
        scored: list[RetrievedChunk] = []
        for i, chunk in enumerate(self.chunks):
            if doc_types and chunk.doc_type not in doc_types:
                continue
            tf, dl = self._tfs[i], self._lens[i]
            s = 0.0
            for t in q:
                f = tf.get(t)
                if not f:
                    continue
                s += self._idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self._avgdl))
            if s > 0:
                scored.append(RetrievedChunk(chunk=chunk, score=round(s, 4)))
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]
