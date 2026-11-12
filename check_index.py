from collections import Counter

import chromadb

BLOCKED = {"2506.06962", "2510.18439", "2504.17204", "2504.13684",
           "1407.5416", "1706.06749", "2402.11651"}

col = chromadb.PersistentClient(path="chroma_db").get_collection("papers")
metas = col.get(include=["metadatas"])["metadatas"]

count, title = Counter(), {}
for m in metas:
    count[m["paper_id"]] += 1
    title[m["paper_id"]] = m["title"]

print(f"{len(count)} papers, {sum(count.values())} chunks\n")
for pid in sorted(count):
    flag = ""
    if title[pid] == pid:
        flag += "  <-- NO TITLE"
    if pid.split("v")[0] in BLOCKED:
        flag += "  <-- SHOULD BE DELETED"
    print(f"{pid}  {count[pid]:4d} chunks  {title[pid][:70]}{flag}")