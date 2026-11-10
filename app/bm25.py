# ─────────────────────────────────────────────────────────────────────────────
# app/bm25.py  —  High-Performance, Zero-Dependency Okapi BM25 Implementation
#
# Implements the standard Okapi BM25 ranking algorithm (Robertson et al.):
#   IDF(q) = ln((N - n(q) + 0.5) / (n(q) + 0.5) + 1)
#   Score(D, Q) = SUM_q [ IDF(q) * (f(q, D) * (k1 + 1)) / (f(q, D) + k1 * (1 - b + b * (|D| / avgdl))) ]
# ─────────────────────────────────────────────────────────────────────────────

import math
import re
from collections import Counter
from typing import Optional


def tokenize(text: str) -> list[str]:
    """
    Standard alphanumeric tokenization for lexical search:
    - Lowercase all characters
    - Split on non-alphanumeric boundaries
    - Retain tokens with length >= 2 (ignores single-character noise)
    """
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [t for t in tokens if len(t) >= 2]


class BM25Okapi:
    """
    In-memory Okapi BM25 index for sparse/lexical retrieval.
    """

    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75):
        """
        Args:
            corpus: List of document chunk strings.
            k1: Term frequency saturation parameter (default: 1.5).
            b: Document length normalization parameter (default: 0.75).
        """
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_lengths: list[int] = []
        self.doc_freqs: list[Counter] = []
        self.nd: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self.avgdl: float = 0.0

        if corpus:
            self._build_index(corpus)

    def _build_index(self, corpus: list[str]) -> None:
        """Tokenize documents, compute inverted document frequencies and avg length."""
        total_length = 0
        self.doc_lengths = []
        self.doc_freqs = []
        self.nd = {}

        for doc in corpus:
            tokens = tokenize(doc)
            length = len(tokens)
            self.doc_lengths.append(length)
            total_length += length

            tf = Counter(tokens)
            self.doc_freqs.append(tf)

            for term in tf.keys():
                self.nd[term] = self.nd.get(term, 0) + 1

        self.avgdl = total_length / self.corpus_size if self.corpus_size > 0 else 0.0

        # Precompute IDF values for all terms in the vocabulary
        self.idf = {}
        for term, freq in self.nd.items():
            # Standard Okapi IDF with +1 smoothing to avoid negative weights
            self.idf[term] = math.log(
                (self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0
            )

    def get_scores(self, query: str) -> list[float]:
        """
        Compute BM25 scores for all documents in the corpus for the given query.
        """
        query_tokens = tokenize(query)
        scores = [0.0] * self.corpus_size

        if not query_tokens or self.corpus_size == 0 or self.avgdl == 0.0:
            return scores

        for token in query_tokens:
            if token not in self.idf:
                continue

            idf = self.idf[token]
            for doc_idx, tf_counter in enumerate(self.doc_freqs):
                tf = tf_counter.get(token, 0)
                if tf == 0:
                    continue

                doc_len = self.doc_lengths[doc_idx]
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
                scores[doc_idx] += idf * (numerator / denominator)

        return scores

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """
        Retrieve top-k documents sorted by BM25 score.

        Returns:
            List of (doc_idx, score) tuples for docs with score > 0.
        """
        scores = self.get_scores(query)
        # Filter to documents that matched at least one term
        scored_docs = [
            (idx, score) for idx, score in enumerate(scores) if score > 0.0
        ]
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs[:top_k]
