import os

from dotenv import load_dotenv
from openai import OpenAI

from config import (
    CHAT_MODEL,
    MAX_CHAT_HISTORY_MESSAGES,
    MAX_OUTPUT_TOKENS,
    TOP_K_MEMORIES,
)
from persona import build_instructions
from rag import LocalRAGIndex


def build_retrieval_query(
    user_text: str,
    history: list[dict],
) -> str:
    """
    Retrieval works better when a vague current message can borrow a little
    context from the immediately preceding conversation.

    Example:
        Aaron: "she finally replied"
    makes more sense when retrieval also sees what "she" refers to.
    """
    recent = history[-6:]

    lines = [
        f"{item['role'].upper()}: {item['content']}"
        for item in recent
    ]

    lines.append(f"USER: {user_text}")

    return "\n".join(lines)


def print_sources(results: list[dict]):
    if not results:
        print("\nNo retrieved memories yet.\n")
        return

    print("\n--- Retrieved historical examples ---")

    for number, result in enumerate(results, start=1):
        meta = result["metadata"]

        print(
            f"\n[{number}] "
            f"score={result['score']:.3f} "
            f"session={meta['session_id']} "
            f"window={meta['window_id']}"
        )

        print(meta["text"])

    print("\n-------------------------------------\n")


def main():
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set.\n"
            "Create a .env file in the project root, or export it in your shell."
        )

    client = OpenAI()
    rag = LocalRAGIndex(client)

    history = []
    last_sources = []

    print()
    print("IsaacBot v0.1")
    print(f"Model: {CHAT_MODEL}")
    print("Commands: :quit  :clear  :sources")
    print()

    while True:
        try:
            user_text = input("Aaron: ").strip()

        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break

        if not user_text:
            continue

        command = user_text.lower()

        if command in {":quit", ":exit"}:
            print("bye")
            break

        if command == ":clear":
            history.clear()
            last_sources = []
            print("conversation cleared\n")
            continue

        if command == ":sources":
            print_sources(last_sources)
            continue

        retrieval_query = build_retrieval_query(
            user_text,
            history,
        )

        last_sources = rag.search(
            retrieval_query,
            top_k=TOP_K_MEMORIES,
        )

        instructions = build_instructions(
            last_sources
        )

        current_input = (
            history[-MAX_CHAT_HISTORY_MESSAGES:]
            + [{
                "role": "user",
                "content": user_text,
            }]
        )

        try:
            response = client.responses.create(
                model=CHAT_MODEL,
                instructions=instructions,
                input=current_input,
                max_output_tokens=MAX_OUTPUT_TOKENS,
            )

        except Exception as exc:
            print(f"\nAPI error: {exc}\n")
            continue

        reply = (response.output_text or "").strip()

        if not reply:
            reply = "(no text response)"

        print(f"Isaac: {reply}\n")

        history.append({
            "role": "user",
            "content": user_text,
        })

        history.append({
            "role": "assistant",
            "content": reply,
        })


if __name__ == "__main__":
    main()
