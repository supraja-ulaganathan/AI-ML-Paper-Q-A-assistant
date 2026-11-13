import warnings
warnings.filterwarnings("ignore")

import json
import re
import sys
from pathlib import Path

from answer import llm_text

PROCESSED = Path("data/processed")


def load_paper(paper_id):
    return json.loads((PROCESSED / f"{paper_id}.json").read_text(encoding="utf-8"))


def get_abstract(paper, max_chars=2500):
    """Take the text after the word 'Abstract', or the start of page 1 if not found."""
    first_page = paper["pages"][0]["text"]
    m = re.search(r"abstract", first_page, flags=re.IGNORECASE)
    start = m.end() if m else 0
    return first_page[start:start + max_chars].strip()


def summarize_paper(paper_id):
    text = get_abstract(load_paper(paper_id))
    out = llm_text(
        "Summarize this research paper abstract in 3 plain sentences. "
        "Use only the text given. No LaTeX.\n\n" + text
    )
    return out or "The summary service is busy right now. Please try again in a minute."


def keywords_for_paper(paper_id, top_n=8):
    text = get_abstract(load_paper(paper_id), max_chars=3500)
    out = llm_text(
        f"List the {top_n} most important keyphrases of this paper abstract. "
        "Reply with one comma-separated line only, no numbering, no extra words.\n\n" + text
    )
    if not out:
        return []
    return [k.strip() for k in out.replace("\n", ",").split(",") if k.strip()][:top_n]


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "list":
        for f in sorted(PROCESSED.glob("*.json")):
            p = json.loads(f.read_text(encoding="utf-8"))
            print(f'{p["id"]}  {p.get("title", "")[:80]}')
        print("\nUsage: python nlp_tools.py <paper_id>")
    else:
        pid = sys.argv[1]
        paper = load_paper(pid)
        print(f"\nTitle: {paper.get('title', pid)}\n")
        print("Keywords:", ", ".join(keywords_for_paper(pid)))
        print("\nSummary:", summarize_paper(pid))