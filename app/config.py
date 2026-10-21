import os
from pathlib import Path

from dotenv import load_dotenv

APP_VERSION = "0.1.0"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env")


# ---------------------------------------------------------------------------
# Data locations
#
# The public repository ships only a small synthetic dataset in data/sample.
# Point ISAACBOT_DATA_DIR at a private folder to run on real conversations;
# that folder should never be committed.
# ---------------------------------------------------------------------------

DATA_DIR = Path(
    os.getenv("ISAACBOT_DATA_DIR", PROJECT_ROOT / "data" / "sample")
)

WINDOWS_PATH = DATA_DIR / "train_windows.jsonl"

PERSONA_PATH = DATA_DIR / "persona.md"

RAG_DIR = DATA_DIR / "rag"

INDEX_PATH = RAG_DIR / "rag_embeddings.npz"

METADATA_PATH = RAG_DIR / "rag_metadata.jsonl"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

CHAT_MODEL = os.getenv(
    "ISAACBOT_MODEL",
    "gpt-6-luna",
)

EMBEDDING_MODEL = os.getenv(
    "ISAACBOT_EMBEDDING_MODEL",
    "text-embedding-3-small",
)


# ---------------------------------------------------------------------------
# Retrieval and generation settings
# ---------------------------------------------------------------------------

MIN_TURNS_FOR_RAG = 2
EMBED_BATCH_SIZE = 64
TOP_K_MEMORIES = 5
MAX_RESULTS_PER_SESSION = 1

MAX_CHAT_HISTORY_MESSAGES = 12
MAX_OUTPUT_TOKENS = 300


# Showing retrieved conversations in the sidebar is useful for debugging but
# exposes the underlying corpus to anyone using the app. Off unless enabled.
SHOW_RETRIEVED_MEMORIES = (
    os.getenv("ISAACBOT_SHOW_MEMORIES", "false").lower() == "true"
)


# Display names used in the prompt and UI.
PERSONA_NAME = os.getenv("ISAACBOT_PERSONA_NAME", "Sam")
USER_NAME = os.getenv("ISAACBOT_USER_NAME", "Alex")
