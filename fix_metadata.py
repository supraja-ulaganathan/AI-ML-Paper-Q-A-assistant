import json
from pathlib import Path

import arxiv

PAPERS = Path("data/papers")
META = Path("data/metadata.json")
meta = json.loads(META.read_text(encoding="utf-8")) if META.exists() else {}

ids = [p.stem for p in PAPERS.glob("*.pdf")]
missing = [i for i in ids if i not in meta or meta[i].get("title", i) == i]
print("Missing titles for:", missing)

if missing:
    for r in arxiv.Client().results(arxiv.Search(id_list=missing)):
        pid = r.get_short_id().replace("/", "_")
        meta[pid] = {
            "id": pid,
            "title": r.title,
            "authors": [a.name for a in r.authors][:5],
            "year": r.published.year,
            "url": r.entry_id,
        }
        print("Fixed:", pid, "-", r.title)

META.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")