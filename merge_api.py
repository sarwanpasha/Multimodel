#!/usr/bin/env python3
"""Merge harness shards shards/<OUT>.s<k>.json into OUT; ds_index is mapped back
to the unsharded dataset order (shard-local i -> i*K + k). Writes nothing if a
shard is missing."""
import json, os, sys

out, K = sys.argv[1], int(sys.argv[2])
base = out[:-5] if out.endswith(".json") else out
parts = []
for k in range(K):
    f = f"shards/{base}.s{k}.json"
    if not os.path.exists(f):
        print("missing", f)
        sys.exit(1)
    parts.append(json.load(open(f)))
listkey = next(k for k in ("steps", "items", "results") if k in parts[0])
rows = []
for k, j in enumerate(parts):
    for r in j[listkey]:
        if isinstance(r, dict) and "ds_index" in r:
            r["shard"] = k
            r["ds_index_local"] = r["ds_index"]
            r["ds_index"] = r["ds_index"] * K + k
        rows.append(r)
merged = {kk: v for kk, v in parts[0].items() if kk != listkey}
merged["shard_summaries"] = [j.get("summary") for j in parts]
merged["nshards"] = K
merged[listkey] = rows
json.dump(merged, open(out, "w"))
print(f"merged {K} shards -> {out}: {len(rows)} rows "
      f"({sum(1 for r in rows if not r.get('skipped'))} kept)")
