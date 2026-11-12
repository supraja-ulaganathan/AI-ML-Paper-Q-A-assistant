import sys

from search import search

q = " ".join(sys.argv[1:])
for i, h in enumerate(search(q, k=8), 1):
    print(f'[{i}] {h["title"]} (page {h["page"]}) score={h["score"]}')
    print("    ", h["text"][:400], "\n")