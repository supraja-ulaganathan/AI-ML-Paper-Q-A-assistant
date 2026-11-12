import sys

import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("papers")


def search(query, k=5, per_paper=2):
    emb = model.encode([query], normalize_embeddings=True).tolist()
    res = collection.query(query_embeddings=emb, n_results=k * 6)
    results, counts = [], {}
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        pid = meta["paper_id"]
        if counts.get(pid, 0) >= per_paper:
            continue
        counts[pid] = counts.get(pid, 0) + 1
        results.append({
            "text": doc,
            "title": meta["title"],
            "page": meta["page"],
            "url": meta["url"],
            "score": round(1 - dist, 3),
        })
        if len(results) == k:
            break
    return results


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "What is retrieval augmented generation?"
    print(f"\nQuestion: {query}\n")
    for i, r in enumerate(search(query), 1):
        print(f'{i}. {r["title"]} (page {r["page"]}) score={r["score"]}')
        print(f'   {r["text"][:300]}...\n')