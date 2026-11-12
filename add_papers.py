import json
from pathlib import Path
import arxiv

IDS = ["2005.11401", "2312.10997"]
meta_path = Path("data/metadata.json")
metadata = json.loads(meta_path.read_text(encoding="utf-8"))

client = arxiv.Client()
for r in client.results(arxiv.Search(id_list=IDS)):
    pid = r.get_short_id().replace("/", "_")
    r.download_pdf(dirpath="data/papers", filename=f"{pid}.pdf")
    metadata[pid] = {
        "id": pid,
        "title": r.title,
        "authors": [a.name for a in r.authors][:5],
        "year": r.published.year,
        "url": r.entry_id,
    }
    print("Added:", r.title)

meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")