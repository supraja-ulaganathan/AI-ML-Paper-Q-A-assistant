import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from search import search, collection

MIN_SCORE = 0.35      # same value as in answer.py
K = 8                 # same number of passages that answer.py sends to the LLM

data = json.loads(Path("eval_questions.json").read_text(encoding="utf-8"))
items = data["in_scope"]
off = data.get("off_topic", [])

# Baseline: plain keyword search (TF-IDF) over the same chunks
store = collection.get(include=["documents", "metadatas"])
docs, metas = store["documents"], store["metadatas"]
vec = TfidfVectorizer(stop_words="english", sublinear_tf=True)
X = vec.fit_transform(docs)


def keyword_urls(q, k=K, per_paper=2):
    sims = cosine_similarity(vec.transform([q]), X).ravel()
    out, counts = [], {}
    for i in sims.argsort()[::-1]:
        url = metas[i]["url"]
        if counts.get(url, 0) >= per_paper:
            continue
        counts[url] = counts.get(url, 0) + 1
        out.append(url)
        if len(out) == k:
            break
    return out


def semantic_urls(q, k=K):
    return [r["url"] for r in search(q, k=k)]


def rank_of(urls, paper_ids):
    for i, u in enumerate(urls, 1):
        if any(pid in u for pid in paper_ids):
            return i
    return None


def score(fn):
    ranks = [rank_of(fn(it["q"]), it["paper_ids"]) for it in items]
    n = len(ranks)
    hit = lambda k: sum(1 for r in ranks if r and r <= k) / n
    mrr = sum(1 / r for r in ranks if r) / n
    return hit(1), hit(3), hit(5), hit(8), mrr, ranks


print(f"\nIn-scope questions: {len(items)}\n")
print(f'{"Method":<22}{"Hit@1":>8}{"Hit@3":>8}{"Hit@5":>8}{"Hit@8":>8}{"MRR":>8}')
results = {}
for name, fn in [("Keyword (TF-IDF)", keyword_urls), ("Semantic (yours)", semantic_urls)]:
    h1, h3, h5, h8, mrr, ranks = score(fn)
    results[name] = ranks
    print(f"{name:<22}{h1:>8.0%}{h3:>8.0%}{h5:>8.0%}{h8:>8.0%}{mrr:>8.2f}")

print("\nQuestions your semantic search missed in the top 8:")
for it, r in zip(items, results["Semantic (yours)"]):
    if r is None:
        print("  -", it["q"], f'(expected {it["paper_ids"]})')


def top_score(q):
    r = search(q, k=1)
    return r[0]["score"] if r else 0.0


print("\nRefusal check (based on the similarity threshold):")
refused_off = [q for q in off if top_score(q) < MIN_SCORE]
for q in off:
    if q not in refused_off:
        print(f"  Off-topic NOT refused (score {top_score(q)}): {q}")
print(f"Off-topic questions refused: {len(refused_off)}/{len(off)}")

wrong = [it["q"] for it in items if top_score(it["q"]) < MIN_SCORE]
print(f"In-scope questions wrongly refused: {len(wrong)}/{len(items)}")