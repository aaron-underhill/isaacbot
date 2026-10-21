import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI


# =========================================================
# PATHS / IMPORTS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = PROJECT_ROOT / "app"

# app/rag.py and app/persona.py import config directly.
sys.path.insert(0, str(APP_DIR))

from config import (  # noqa: E402
    CHAT_MODEL,
    EMBEDDING_MODEL,
    MAX_OUTPUT_TOKENS,
    TOP_K_MEMORIES,
)
from persona import build_instructions  # noqa: E402
from rag import LocalRAGIndex  # noqa: E402


VALIDATION_SESSIONS_PATH = (
    PROJECT_ROOT / "data" / "splits" / "validation_sessions.jsonl"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "evaluation"
PREDICTIONS_PATH = OUTPUT_DIR / "validation_predictions_v0.1.jsonl"
SUMMARY_PATH = OUTPUT_DIR / "validation_summary_v0.1.json"

MAX_CASES = 60
RANDOM_SEED = 42
MAX_CONTEXT_TURNS = 8
EMBED_BATCH_PAIRS = 50


# =========================================================
# TEXT HELPERS
# =========================================================

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?")
EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U00002600-\U000026FF"
    "]"
)


def load_jsonl(path):
    rows = []

    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8") as file:
        for n, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON in {path} line {n}: {exc}"
                ) from exc

    return rows


def write_jsonl(rows, path):
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            json.dump(row, file, ensure_ascii=False)
            file.write("\n")


def words(text):
    return WORD_RE.findall(text or "")


def contains_emoji(text):
    return bool(EMOJI_RE.search(text or ""))


def is_all_caps(text):
    letters = [c for c in (text or "") if c.isalpha()]
    return bool(letters) and all(c.isupper() for c in letters)


def cosine(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    an = np.linalg.norm(a)
    bn = np.linalg.norm(b)

    if an == 0 or bn == 0:
        return None

    return float(np.dot(a, b) / (an * bn))


# =========================================================
# BUILD VALIDATION CASES
# =========================================================

def turn_to_role(turn):
    if turn["speaker"] == "Aaron":
        return "user"

    if turn["speaker"] == "Isaac":
        return "assistant"

    raise ValueError(f"Unknown speaker: {turn['speaker']}")


def build_cases(sessions):
    cases = []

    for session in sessions:
        turns = session.get("turns", [])

        for target_index, target_turn in enumerate(turns):
            if target_turn.get("speaker") != "Isaac":
                continue

            if target_index == 0:
                continue

            if turns[target_index - 1].get("speaker") != "Aaron":
                continue

            start = max(0, target_index - MAX_CONTEXT_TURNS)
            context_turns = turns[start:target_index]

            if not context_turns:
                continue

            if context_turns[-1].get("speaker") != "Aaron":
                continue

            real = (target_turn.get("text") or "").strip()

            if not real:
                continue

            target_turn_id = target_turn.get("turn_id")

            cases.append({
                "case_key": f"{session['session_id']}:{target_turn_id}",
                "session_id": session["session_id"],
                "start_time": session.get("start_time"),
                "target_turn_id": target_turn_id,
                "context_turns": context_turns,
                "real_isaac": real,
            })

    return cases


def sample_cases(cases):
    rng = random.Random(RANDOM_SEED)
    rows = list(cases)
    rng.shuffle(rows)

    if MAX_CASES is not None:
        rows = rows[:MAX_CASES]

    return rows


# =========================================================
# ISAACBOT PREDICTION
# =========================================================

def build_retrieval_query(context_turns):
    recent = context_turns[-6:]

    return "\n".join(
        f"{turn['speaker']}: {turn.get('text', '')}"
        for turn in recent
    )


def build_model_input(context_turns):
    return [
        {
            "role": turn_to_role(turn),
            "content": turn.get("text", ""),
        }
        for turn in context_turns
    ]


def predict_case(client, rag, case):
    sources = rag.search(
        build_retrieval_query(case["context_turns"]),
        top_k=TOP_K_MEMORIES,
    )

    response = client.responses.create(
        model=CHAT_MODEL,
        instructions=build_instructions(sources),
        input=build_model_input(case["context_turns"]),
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )

    # A model call can occasionally return no text. Keep that as an empty
    # prediction instead of crashing the embedding stage later.
    predicted = (response.output_text or "").strip()

    return predicted, sources


# =========================================================
# METRICS
# =========================================================

def add_surface_metrics(row):
    real = row["real_isaac"]
    bot = row["isaacbot"]

    rw = len(words(real))
    bw = len(words(bot))

    row.setdefault("metrics", {})

    row["metrics"].update({
        "real_word_count": rw,
        "bot_word_count": bw,
        "absolute_word_count_error": abs(rw - bw),
        "empty_bot_response": not bool(bot.strip()),
        "real_has_emoji": contains_emoji(real),
        "bot_has_emoji": contains_emoji(bot),
        "emoji_presence_match": contains_emoji(real) == contains_emoji(bot),
        "real_has_question": "?" in real,
        "bot_has_question": "?" in bot,
        "question_presence_match": ("?" in real) == ("?" in bot),
        "real_all_caps": is_all_caps(real),
        "bot_all_caps": is_all_caps(bot),
        "all_caps_match": is_all_caps(real) == is_all_caps(bot),
    })


def embed_pairs(client, rows):
    """
    Compute semantic similarity only for non-empty real/bot pairs.
    Empty predictions get semantic_similarity=None instead of causing
    the OpenAI embeddings endpoint to reject the entire batch.
    """

    valid_rows = []

    for row in rows:
        real = (row.get("real_isaac") or "").strip()
        bot = (row.get("isaacbot") or "").strip()

        if not real or not bot:
            row.setdefault("metrics", {})["semantic_similarity"] = None
            continue

        valid_rows.append(row)

    for start in range(0, len(valid_rows), EMBED_BATCH_PAIRS):
        batch_rows = valid_rows[start:start + EMBED_BATCH_PAIRS]
        texts = []

        for row in batch_rows:
            texts.extend([
                row["real_isaac"],
                row["isaacbot"],
            ])

        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts,
        )

        vectors = [item.embedding for item in response.data]

        for i, row in enumerate(batch_rows):
            row["metrics"]["semantic_similarity"] = cosine(
                vectors[i * 2],
                vectors[i * 2 + 1],
            )


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def summarize(rows):
    return {
        "cases": len(rows),
        "chat_model": CHAT_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "top_k_memories": TOP_K_MEMORIES,
        "max_context_turns": MAX_CONTEXT_TURNS,
        "random_seed": RANDOM_SEED,
        "metrics": {
            "mean_semantic_similarity": mean([
                r["metrics"].get("semantic_similarity")
                for r in rows
            ]),
            "mean_real_word_count": mean([
                r["metrics"]["real_word_count"] for r in rows
            ]),
            "mean_bot_word_count": mean([
                r["metrics"]["bot_word_count"] for r in rows
            ]),
            "mean_absolute_word_count_error": mean([
                r["metrics"]["absolute_word_count_error"] for r in rows
            ]),
            "empty_response_rate": mean([
                1.0 if r["metrics"]["empty_bot_response"] else 0.0
                for r in rows
            ]),
            "emoji_presence_accuracy": mean([
                1.0 if r["metrics"]["emoji_presence_match"] else 0.0
                for r in rows
            ]),
            "question_presence_accuracy": mean([
                1.0 if r["metrics"]["question_presence_match"] else 0.0
                for r in rows
            ]),
            "all_caps_accuracy": mean([
                1.0 if r["metrics"]["all_caps_match"] else 0.0
                for r in rows
            ]),
            "real_emoji_rate": mean([
                1.0 if r["metrics"]["real_has_emoji"] else 0.0
                for r in rows
            ]),
            "bot_emoji_rate": mean([
                1.0 if r["metrics"]["bot_has_emoji"] else 0.0
                for r in rows
            ]),
        },
    }


# =========================================================
# CHECKPOINT / RESUME
# =========================================================

def build_prediction_row(number, case, predicted, sources):
    row = {
        "case_id": number,
        "case_key": case["case_key"],
        "session_id": case["session_id"],
        "target_turn_id": case["target_turn_id"],
        "start_time": case["start_time"],
        "context": [
            {
                "speaker": t["speaker"],
                "text": t.get("text", ""),
                "turn_id": t.get("turn_id"),
            }
            for t in case["context_turns"]
        ],
        "real_isaac": case["real_isaac"],
        "isaacbot": predicted,
        "retrieved": [
            {
                "score": source["score"],
                "session_id": source["metadata"]["session_id"],
                "window_id": source["metadata"]["window_id"],
            }
            for source in sources
        ],
        "metrics": {},
    }

    add_surface_metrics(row)
    return row


# =========================================================
# MAIN
# =========================================================

def main():
    load_dotenv(PROJECT_ROOT / ".env")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not loaded.")

    sessions = load_jsonl(VALIDATION_SESSIONS_PATH)
    cases = sample_cases(build_cases(sessions))

    if not cases:
        raise RuntimeError("No validation cases found.")

    client = OpenAI()
    rag = LocalRAGIndex(client)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Resume from prior successful checkpoints if this fixed script is rerun.
    existing_rows = load_jsonl(PREDICTIONS_PATH)
    existing_by_key = {
        row.get("case_key"): row
        for row in existing_rows
        if row.get("case_key")
    }

    rows = []

    print()
    print(f"Validation cases selected: {len(cases):,}")
    print(f"Model:                     {CHAT_MODEL}")
    print(f"RAG top-k:                 {TOP_K_MEMORIES}")
    print(f"Existing checkpoints:      {len(existing_by_key):,}")
    print()

    for number, case in enumerate(cases, start=1):
        if case["case_key"] in existing_by_key:
            row = existing_by_key[case["case_key"]]
            row["case_id"] = number
            rows.append(row)
        else:
            predicted, sources = predict_case(client, rag, case)
            row = build_prediction_row(
                number,
                case,
                predicted,
                sources,
            )
            rows.append(row)

            # Save immediately so a later metric/API error does not lose
            # already-generated model responses.
            write_jsonl(rows, PREDICTIONS_PATH)

        print(
            f"\rPrepared {number:,}/{len(cases):,}",
            end="",
            flush=True,
        )

    print()
    print("Calculating semantic similarities...")

    embed_pairs(client, rows)

    # Save again now that semantic metrics have been added.
    write_jsonl(rows, PREDICTIONS_PATH)

    summary = summarize(rows)

    with SUMMARY_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    m = summary["metrics"]

    print()
    print("Validation complete.")
    print(f"Predictions: {PREDICTIONS_PATH}")
    print(f"Summary:     {SUMMARY_PATH}")
    print()

    semantic = m["mean_semantic_similarity"]
    semantic_text = f"{semantic:.3f}" if semantic is not None else "N/A"

    print(f"Mean semantic similarity: {semantic_text}")
    print(
        f"Mean words — real / bot:   "
        f"{m['mean_real_word_count']:.2f} / {m['mean_bot_word_count']:.2f}"
    )
    print(
        f"Empty bot response rate:    "
        f"{m['empty_response_rate']:.1%}"
    )
    print(
        f"Emoji rate — real / bot:    "
        f"{m['real_emoji_rate']:.1%} / {m['bot_emoji_rate']:.1%}"
    )
    print(
        f"Question-presence accuracy: "
        f"{m['question_presence_accuracy']:.1%}"
    )
    print(
        f"ALL-CAPS accuracy:          "
        f"{m['all_caps_accuracy']:.1%}"
    )


if __name__ == "__main__":
    main()
