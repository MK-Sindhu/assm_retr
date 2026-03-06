# build_mate_vocab.py
import json
from pathlib import Path
from collections import Counter

INP = Path("assembly_filter_out/assembly_graphs_v2.jsonl")
OUT = Path("assembly_filter_out/mate_type_vocab_v2.json")

cnt = Counter()
with INP.open() as f:
    for line in f:
        r = json.loads(line)
        for d in r.get("edge_mate_type_counts", []):
            if isinstance(d, dict):
                for k, v in d.items():
                    if isinstance(k, str):
                        cnt[k] += int(v) if isinstance(v, int) else 1

types = [t for t, _ in cnt.most_common()]
vocab = {t: i for i, t in enumerate(types)}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({"vocab": vocab, "counts": cnt}, indent=2, default=int))

print("Saved:", OUT)
print("Num mate types:", len(vocab))
print("Top 20:", types[:20])