"""Download the 21 arXiv papers used by the assistant and build the search index.

Usage:
    python setup_data.py            # build only if the index does not exist yet
    python setup_data.py --force    # rebuild everything
"""
import json
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).parent

PAPER_IDS = [
    "1912.02145v1", "2005.11401v4", "2108.06279v2", "2204.07496v4", "2205.02303v1",
    "2210.15133v1", "2301.05508v1", "2309.15217v2", "2310.00259v3", "2311.02956v1",
    "2312.10997v5", "2402.12317v2", "2406.18960v1", "2409.11353v3", "2411.18583v1",
    "2502.00306v2", "2503.21676v2", "2506.22486v1", "2508.09534v1", "2601.07528v2",
    "2611.08228v1",
]


def index_exists():
    db = BASE / "chroma_db"
    return db.exists() and any(db.iterdir())


def download_papers():
    import arxiv

    papers_dir = BASE / "data" / "papers"
    papers_dir.mkdir(parents=True, exist_ok=True)
    meta = {}

    client = arxiv.Client(delay_seconds=3, num_retries=3)
    for r in client.results(arxiv.Search(id_list=PAPER_IDS)):
        pid = r.get_short_id().replace("/", "_")
        pdf = papers_dir / f"{pid}.pdf"
        if not pdf.exists():
            r.download_pdf(dirpath=str(papers_dir), filename=f"{pid}.pdf")
        meta[pid] = {
            "id": pid,
            "title": r.title,
            "authors": [a.name for a in r.authors][:5],
            "year": r.published.year,
            "url": r.entry_id,
        }
        print("Got:", r.title)

    (BASE / "data" / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def build(force=False):
    download_papers()
    args = ["--force"] if force else []
    subprocess.run([sys.executable, "extract_text.py", *args], check=True, cwd=BASE)
    subprocess.run([sys.executable, "build_index.py"], check=True, cwd=BASE)
    print("Setup complete.")


def ensure_data(force=False):
    """Build the data only when it is missing (safe to call on every app start)."""
    if index_exists() and not force:
        return
    build(force=force)


if __name__ == "__main__":
    ensure_data(force="--force" in sys.argv)
    print("Now run: python app.py")