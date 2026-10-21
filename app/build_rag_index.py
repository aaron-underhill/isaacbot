"""
Build the local RAG index from training conversation windows.

Input:  <data dir>/train_windows.jsonl
        One JSON object per line with at least: window_id, session_id, text.
        Only training-split windows belong here, so validation and test
        conversations never leak into retrieval.

Output: <data dir>/rag/rag_embeddings.npz   (normalized float32 matrix)
        <data dir>/rag/rag_metadata.jsonl   (one row per embedding, same order)

Usage:
    python app/build_rag_index.py
"""

import json

import numpy as np
from openai import OpenAI

from config import (
    EMBED_BATCH_SIZE,
    EMBEDDING_MODEL,
    INDEX_PATH,
    METADATA_PATH,
    MIN_TURNS_FOR_RAG,
    RAG_DIR,
    WINDOWS_PATH,
)
from rag import load_jsonl


def turn_count(window: dict) -> int:
    turn_ids = window.get("turn_ids")

    if turn_ids is not None:
        return len(turn_ids)

    return len([line for line in window["text"].splitlines() if line.strip()])


def main() -> None:
    windows = [
        window
        for window in load_jsonl(WINDOWS_PATH)
        if turn_count(window) >= MIN_TURNS_FOR_RAG
    ]

    if not windows:
        raise SystemExit(f"No usable windows found in {WINDOWS_PATH}")

    client = OpenAI()
    vectors = []

    for start in range(0, len(windows), EMBED_BATCH_SIZE):
        batch = windows[start:start + EMBED_BATCH_SIZE]

        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[window["text"] for window in batch],
        )

        vectors.extend(item.embedding for item in response.data)

        print(f"Embedded {min(start + EMBED_BATCH_SIZE, len(windows))}/{len(windows)}")

    matrix = np.asarray(vectors, dtype=np.float32)

    # Normalize rows so cosine similarity becomes a dot product at query time.
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms

    RAG_DIR.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(INDEX_PATH, embeddings=matrix)

    with METADATA_PATH.open("w", encoding="utf-8") as file:
        for rag_id, window in enumerate(windows):
            row = {"rag_id": rag_id, **window}
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {matrix.shape[0]} x {matrix.shape[1]} index to {RAG_DIR}")


if __name__ == "__main__":
    main()
