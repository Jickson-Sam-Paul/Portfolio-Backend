from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

PROFILE_FILE = Path("app/data/profile.txt")
TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9+#\-.]*")


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    score: float
    text: str


class PortfolioRetriever:
    def __init__(
        self,
        profile_file: Path = PROFILE_FILE,
        max_chunk_chars: int = 700,
        min_score: float = 0.05,
    ):
        self.profile_file = profile_file
        self.max_chunk_chars = max_chunk_chars
        self.min_score = min_score
        self._chunks: List[Dict[str, str]] = []
        self._idf: Dict[str, float] = {}
        self._chunk_vectors: List[Dict[str, float]] = []
        self._chunk_norms: List[float] = []
        self._inverted_index: Dict[str, List[int]] = defaultdict(list)
        self._index_built = False

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        self._ensure_index()

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        query_counts = Counter(query_tokens)
        query_vector = {}
        for token, count in query_counts.items():
            idf = self._idf.get(token)
            if idf is None:
                continue
            query_vector[token] = float(count) * idf

        if not query_vector:
            return []

        query_norm = self._l2_norm(query_vector)
        if query_norm == 0.0:
            return []

        candidate_indices = set()
        for token in query_vector:
            candidate_indices.update(self._inverted_index.get(token, []))

        if not candidate_indices:
            return []

        scored: List[RetrievedChunk] = []
        for idx in candidate_indices:
            dot = 0.0
            chunk_vector = self._chunk_vectors[idx]
            for token, query_weight in query_vector.items():
                chunk_weight = chunk_vector.get(token)
                if chunk_weight is not None:
                    dot += chunk_weight * query_weight

            chunk_norm = self._chunk_norms[idx]
            if chunk_norm == 0.0:
                continue

            score = dot / (chunk_norm * query_norm)
            if score < self.min_score:
                continue

            chunk = self._chunks[idx]
            scored.append(
                RetrievedChunk(
                    chunk_id=chunk["id"],
                    score=score,
                    text=chunk["text"],
                )
            )

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    def _ensure_index(self) -> None:
        if self._index_built:
            return

        raw_text = self.profile_file.read_text(encoding="utf-8").strip()
        self._chunks = self._chunk_text(raw_text)
        tokenized_chunks = [self._tokenize(chunk["text"]) for chunk in self._chunks]

        num_docs = len(tokenized_chunks)
        document_frequency = Counter()
        for tokens in tokenized_chunks:
            document_frequency.update(set(tokens))

        self._idf = {
            token: math.log((1 + num_docs) / (1 + frequency)) + 1.0
            for token, frequency in document_frequency.items()
        }

        self._chunk_vectors = []
        self._chunk_norms = []
        self._inverted_index = defaultdict(list)
        for idx, tokens in enumerate(tokenized_chunks):
            counts = Counter(tokens)
            vector = {
                token: float(count) * self._idf[token]
                for token, count in counts.items()
                if token in self._idf
            }

            self._chunk_vectors.append(vector)
            self._chunk_norms.append(self._l2_norm(vector))

            for token in vector:
                self._inverted_index[token].append(idx)

        self._index_built = True

    def _chunk_text(self, text: str) -> List[Dict[str, str]]:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: List[Dict[str, str]] = []
        current: List[str] = []
        current_len = 0
        counter = 0

        def flush() -> None:
            nonlocal current, current_len, counter
            if not current:
                return
            chunk_text = "\n\n".join(current).strip()
            if not chunk_text:
                return
            counter += 1
            chunks.append({"id": f"chunk-{counter}", "text": chunk_text})
            current = []
            current_len = 0

        for paragraph in paragraphs:
            paragraph_len = len(paragraph)
            if paragraph_len > self.max_chunk_chars:
                flush()
                sentences = self._split_sentences(paragraph)
                sentence_bucket: List[str] = []
                bucket_len = 0
                for sentence in sentences:
                    sentence_len = len(sentence)
                    proposed_len = bucket_len + sentence_len + (1 if sentence_bucket else 0)
                    if sentence_bucket and proposed_len > self.max_chunk_chars:
                        counter += 1
                        chunks.append(
                            {"id": f"chunk-{counter}", "text": " ".join(sentence_bucket)}
                        )
                        sentence_bucket = [sentence]
                        bucket_len = sentence_len
                    else:
                        sentence_bucket.append(sentence)
                        bucket_len = proposed_len
                if sentence_bucket:
                    counter += 1
                    chunks.append(
                        {"id": f"chunk-{counter}", "text": " ".join(sentence_bucket)}
                    )
                continue

            proposed_len = current_len + paragraph_len + (2 if current else 0)
            if current and proposed_len > self.max_chunk_chars:
                flush()

            current.append(paragraph)
            current_len += paragraph_len + (2 if len(current) > 1 else 0)

        flush()
        return chunks

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [part.strip() for part in parts if part.strip()]

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        lowered = text.lower()
        tokens = TOKEN_PATTERN.findall(lowered)
        stopwords = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "by",
            "for",
            "from",
            "has",
            "have",
            "i",
            "in",
            "is",
            "it",
            "my",
            "of",
            "on",
            "or",
            "that",
            "the",
            "to",
            "was",
            "with",
        }
        return [token for token in tokens if token not in stopwords and len(token) > 1]

    @staticmethod
    def _l2_norm(vector: Dict[str, float]) -> float:
        return math.sqrt(sum(weight * weight for weight in vector.values()))


_retriever: PortfolioRetriever | None = None


def get_retriever() -> PortfolioRetriever:
    global _retriever
    if _retriever is None:
        _retriever = PortfolioRetriever()
    return _retriever

