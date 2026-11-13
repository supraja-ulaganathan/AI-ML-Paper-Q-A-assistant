import warnings
warnings.filterwarnings("ignore")      # hides the Python 3.9 FutureWarning messages

import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

from search import search

logging.getLogger("google_genai").setLevel(logging.ERROR)
load_dotenv(Path(__file__).parent / ".env")

VERBOSE = False        # set True to see progress and error messages when debugging
SHOW_SOURCES = True    # set False to print the answer only
MIN_SCORE = 0.35
NOT_FOUND = "I could not find this in the provided papers."
RETRYABLE = ("503", "504", "429", "UNAVAILABLE", "DEADLINE", "RESOURCE_EXHAUSTED")

# The one that worked is first. If a name stops working, replace it with a current one from AI Studio.
MODELS = ["gemini-flash-lite-latest", "gemini-flash-latest", "gemini-3.8-flash"]

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"],
    http_options=types.HttpOptions(timeout=90000),   # 90 seconds
)

PROMPT = """You are a research assistant answering questions about a collection of arXiv papers.
Use ONLY the numbered excerpts below.
- Cite sources like [1] or [2] right after the claims they support.
- If the excerpts answer only part of the question, answer that part and say what is missing.
- Only if the excerpts contain nothing relevant, reply exactly: {not_found}
- Excerpts may mention other works in their text or reference lists. Do not present those as papers in this collection.
- Write plain text. Do not use LaTeX or dollar signs.
- Do not use outside knowledge.

Excerpts:
{context}

Question: {question}

Answer:"""


def log(msg):
    if VERBOSE:
        print(msg)


def get_text(resp):
    """Join only the text parts of the reply (avoids the 'non-text parts' warning)."""
    try:
        parts = resp.candidates[0].content.parts
        return "".join(p.text for p in parts if getattr(p, "text", None)).strip()
    except Exception:
        return (resp.text or "").strip()


def llm_text(prompt):
    """Plain LLM call with model fallback. Returns text, or None if every model failed."""
    for model_name in MODELS:
        for attempt in range(2):
            try:
                text = get_text(client.models.generate_content(model=model_name, contents=prompt))
                if text:
                    return text
            except Exception as e:
                if not any(c in str(e) for c in RETRYABLE):
                    break
                time.sleep(4 * (attempt + 1))
    return None


def answer(question, k=8):
    log("Searching papers...")
    hits = [h for h in search(question, k=k) if h["score"] >= MIN_SCORE]
    if not hits:
        return {"answer": NOT_FOUND, "sources": []}

    context = "\n\n".join(
        f'[{i}] ({h["title"]}, page {h["page"]}) {h["text"]}'
        for i, h in enumerate(hits, 1)
    )
    prompt = PROMPT.format(not_found=NOT_FOUND, context=context, question=question)

    log(f"Found {len(hits)} passages. Calling the LLM...")
    for model_name in MODELS:
        for attempt in range(2):
            try:
                resp = client.models.generate_content(model=model_name, contents=prompt)
                text = get_text(resp)
                if not text:
                    continue                          # empty reply: retry or try the next model
                if text.startswith(NOT_FOUND[:30]):   # refusal: show no sources
                    return {"answer": NOT_FOUND, "sources": []}
                log(f"(answered by {model_name})")
                return {"answer": text, "sources": hits}
            except Exception as e:
                msg = str(e)
                log(f"[{model_name}] try {attempt + 1}/2 failed: {msg[:110]}")
                if not any(c in msg for c in RETRYABLE):
                    break
                time.sleep(4 * (attempt + 1))
    return {"answer": "The LLM service is busy right now. Please try again in a few minutes.", "sources": hits}


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What is retrieval augmented generation?"
    out = answer(q)
    print(f"\n{out['answer']}\n")
    if SHOW_SOURCES and out["sources"]:
        print("Sources:")
        for i, s in enumerate(out["sources"], 1):
            print(f'[{i}] {s["title"]} (page {s["page"]})')