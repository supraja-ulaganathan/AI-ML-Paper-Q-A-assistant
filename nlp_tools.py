import warnings
warnings.filterwarnings("ignore")

import json
import re
import sys
from pathlib import Path

from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

PROCESSED = Path("data/processed")
SUM_MODEL = "sshleifer/distilbart-cnn-12-6"   # summarization model (about 1.2 GB download on first use)

_kw_model = None
_sum_tok = None
_sum_model = None

EXTRA_STOP = {
    "arxiv", "preprint", "et", "al", "fig", "figure", "table", "paper", "propose",
    "proposed", "show", "shows", "also", "however", "work", "approach", "method",
    "methods", "results", "use", "using", "used", "novel", "present", "presents",
}


def load_paper(paper_id):
    return json.loads((PROCESSED / f"{paper_id}.json").read_text(encoding="utf-8"))


def get_abstract(paper, max_chars=2500):
    """Take the text after the word 'Abstract', or the start of page 1 if not found."""
    first_page = paper["pages"][0]["text"]
    m = re.search(r"abstract", first_page, flags=re.IGNORECASE)
    start = m.end() if m else 0
    return first_page[start:start + max_chars].strip()


def summarize_paper(paper_id):
    global _sum_tok, _sum_model
    if _sum_model is None:
        _sum_tok = AutoTokenizer.from_pretrained(SUM_MODEL)
        _sum_model = AutoModelForSeq2SeqLM.from_pretrained(SUM_MODEL)
    text = get_abstract(load_paper(paper_id))
    inputs = _sum_tok(text, return_tensors="pt", truncation=True, max_length=1024)
    ids = _sum_model.generate(
        **inputs, max_length=130, min_length=40, num_beams=4, early_stopping=True
    )
    summary = _sum_tok.decode(ids[0], skip_special_tokens=True).strip()
    return re.sub(r"\s+([.,;:])", r"\1", summary)      # remove the space before punctuation


def clean_for_keywords(text):
    text = re.sub(r"arxiv:\S+", " ", text, flags=re.IGNORECASE)   # arXiv stamps
    text = re.sub(r"https?://\S+|\S+@\S+", " ", text)              # links and emails
    return re.sub(r"\s{2,}", " ", text)


def keywords_for_paper(paper_id, top_n=8):
    global _kw_model
    if _kw_model is None:
        _kw_model = KeyBERT(model=SentenceTransformer("all-MiniLM-L6-v2"))
    paper = load_paper(paper_id)
    text = clean_for_keywords(get_abstract(paper, max_chars=3500))   # abstract only
    vec = CountVectorizer(
        ngram_range=(1, 2),
        stop_words=list(ENGLISH_STOP_WORDS | EXTRA_STOP),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b",             # letters only, no numbers
    )
    kws = _kw_model.extract_keywords(
        text, vectorizer=vec, use_mmr=True, diversity=0.5, top_n=top_n
    )
    return [k for k, _ in kws]


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