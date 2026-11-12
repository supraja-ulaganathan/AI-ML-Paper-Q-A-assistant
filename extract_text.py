import json
import re
from pathlib import Path
import fitz  # PyMuPDF

PAPERS_DIR = Path("data/papers")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

meta = json.loads(Path("data/metadata.json").read_text(encoding="utf-8"))


def clean(text: str) -> str:
    text = re.sub(r"-\n(\w)", r"\1", text)   # rejoin words split across lines
    text = re.sub(r"\s*\n\s*", " ", text)     # line breaks -> spaces
    return re.sub(r"\s{2,}", " ", text).strip()


for pdf in sorted(PAPERS_DIR.glob("*.pdf")):
    pid = pdf.stem
    pages = []
    with fitz.open(pdf) as doc:
        for i, page in enumerate(doc, start=1):
            t = clean(page.get_text())
            if t:
                pages.append({"page": i, "text": t})

    total_chars = sum(len(p["text"]) for p in pages)
    info = meta.get(pid, {"id": pid, "title": pid})
    record = {**info, "pages": pages}
    (OUT_DIR / f"{pid}.json").write_text(
        json.dumps(record, ensure_ascii=False), encoding="utf-8"
    )

    flag = "  <-- check this one" if total_chars < 5000 else ""
    print(f"{pid}: {len(pages)} pages, {total_chars} characters{flag}")