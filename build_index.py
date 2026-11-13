import json
import re
from pathlib import Path

import chromadb
from embedder import embed

PROCESSED = Path("data/processed")
CHUNK_WORDS = 200   # chunk size
OVERLAP = 40        # words shared between neighbouring chunks


def chunk_words(text, size=CHUNK_WORDS, overlap=OVERLAP):
    words = text.split()
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        piece = words[start:start + size]
        if len(piece) >= 30:                 # skip tiny fragments
            chunks.append(" ".join(piece))
        if start + size >= len(words):
            break
    return chunks


def is_reference_chunk(text):
    """Detect bibliography text: many years, 'arXiv' or 'pp.' mentions."""
    years = len(re.findall(r"\b(?:19|20)\d{2}\b", text))
    return years >= 5 or text.count("arXiv") >= 2 or text.count("pp.") >= 2


ids, docs, metas = [], [], []
skipped = 0
for f in sorted(PROCESSED.glob("*.json")):
    paper = json.loads(f.read_text(encoding="utf-8"))
    for page in paper["pages"]:
        for j, chunk in enumerate(chunk_words(page["text"])):
            if is_reference_chunk(chunk):    # skip bibliography chunks
                skipped += 1
                continue
            ids.append(f'{paper["id"]}_p{page["page"]}_c{j}')
            docs.append(chunk)
            metas.append({
                "paper_id": paper["id"],
                "title": paper.get("title", paper["id"]),
                "page": page["page"],
                "url": paper.get("url", ""),
            })

print(f"Total chunks kept: {len(docs)} (skipped {skipped} reference chunks)")

embeddings = []
for i in range(0, len(docs), 256):
    embeddings += embed(docs[i:i + 256])
    print(f"Embedded {min(i + 256, len(docs))}/{len(docs)}")

client = chromadb.PersistentClient(path="chroma_db")
try:
    client.delete_collection("papers")      # rebuild from scratch each run
except Exception:
    pass
collection = client.create_collection("papers", metadata={"hnsw:space": "cosine"})

BATCH = 500
for i in range(0, len(docs), BATCH):
    collection.add(
        ids=ids[i:i + BATCH],
        documents=docs[i:i + BATCH],
        embeddings=embeddings[i:i + BATCH],
        metadatas=metas[i:i + BATCH],
    )

print(f"Done. Stored {collection.count()} chunks in chroma_db/")