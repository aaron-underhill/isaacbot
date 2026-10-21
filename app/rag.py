import json
from pathlib import Path

import numpy as np
from openai import OpenAI

from config import (
    EMBEDDING_MODEL,
    INDEX_PATH,
    MAX_RESULTS_PER_SESSION,
    METADATA_PATH,
    TOP_K_MEMORIES,
)


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)

    if norm == 0:
        return vector

    return vector / norm


def load_jsonl(path: Path) -> list[dict]:
    rows = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


class LocalRAGIndex:
    """
    Tiny local vector database for IsaacBot V0.1.

    Embeddings live on disk as a NumPy matrix. Metadata/transcripts live in
    JSONL. Querying is cosine similarity, implemented as a dot product because
    both stored vectors and the query vector are normalized.
    """

    def __init__(
        self,
        client: OpenAI,
        index_path: Path = INDEX_PATH,
        metadata_path: Path = METADATA_PATH,
    ):
        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                f"RAG index not found in {index_path.parent}. Run:\n"
                "    python app/build_rag_index.py\n"
                "first."
            )

        self.client = client

        data = np.load(index_path)
        self.embeddings = data["embeddings"].astype(np.float32)
        self.metadata = load_jsonl(metadata_path)

        if len(self.metadata) != self.embeddings.shape[0]:
            raise RuntimeError(
                "RAG index and metadata have different row counts. "
                "Rebuild the index."
            )

    def embed_query(self, query: str) -> np.ndarray:
        response = self.client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=query,
        )

        vector = np.asarray(
            response.data[0].embedding,
            dtype=np.float32,
        )

        return normalize(vector)

    def search(
        self,
        query: str,
        top_k: int = TOP_K_MEMORIES,
        max_per_session: int = MAX_RESULTS_PER_SESSION,
    ) -> list[dict]:
        query_vector = self.embed_query(query)

        # Stored vectors are normalized, so dot product == cosine similarity.
        scores = self.embeddings @ query_vector

        # Pull more than top_k first because overlapping windows from the same
        # historical session can otherwise dominate the result list.
        candidate_count = min(
            len(scores),
            max(top_k * 10, top_k),
        )

        candidate_indices = np.argpartition(
            scores,
            -candidate_count,
        )[-candidate_count:]

        candidate_indices = candidate_indices[
            np.argsort(scores[candidate_indices])[::-1]
        ]

        results = []
        session_counts = {}

        for index in candidate_indices:
            meta = self.metadata[int(index)]
            session_id = meta["session_id"]

            already_used = session_counts.get(session_id, 0)

            if already_used >= max_per_session:
                continue

            results.append({
                "score": float(scores[index]),
                "metadata": meta,
            })

            session_counts[session_id] = already_used + 1

            if len(results) >= top_k:
                break

        return results
