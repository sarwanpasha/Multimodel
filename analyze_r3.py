"""
Third-report analyses:
  (A) digest of the coordinate-channel and replicate web runs
  (B) family agreement across two runs of the same two-channel condition
  (C) page-clustered bootstrap CI for the DocVQA joint-decisive rate
Usage: python analyze_r3.py > r3_results.txt
"""
import json, math, os, random


def wilson(k, n):
    if n == 0:
        return None
    p, z = k / n, 1.96
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round((c - h) / d, 4), round((c + h) / d, 4)]


def load(p):
    d = json.load(open(p))
    rs = d.get("items") or d.get("steps") or d.get("trials") or d.get("results") or []
    return d, rs


def mean(rs, k):
    v = [r[k] for r in rs if k in r and r[k] is not None]
    return round(sum(v) / len(v), 4) if v else None


def fam(r):
    """canonical hashable form of the minimal decisive family"""
    f = r.get("minimal_decisive") or r.get("minimal_sets") or r.get("Dmin") or []
    return frozenset(frozenset(s) for s in f)


def digest(path):
    d, rs = load(path)
    cfg = d.get("args") or {}
    keep = [r for r in rs if not r.get("skipped")]
    dec = [r for r in keep if r.get("flipped")]
    jd = sum(r.get("joint_decisive", 0) for r in dec)
    print("\n## %s" % os.path.basename(path))
    print("   model=%s read=%s ch=%s struct_style=%s ocr_dev=%s"
          % (cfg.get("model"), cfg.get("read_mode"), cfg.get("channels"),
             cfg.get("struct_style"), cfg.get("ocr_device")))
    print("   n_kept=%d n_dec=%d flip=%s" % (len(keep), len(dec), mean(keep, "flipped")))
    print("   joint=%s CI=%s" % (round(jd / len(dec), 4) if dec else None,
                                 wilson(jd, len(dec))))
    print("   mean|Dmin|=%s min_size=%s unionD=F %s"
          % (mean(dec, "n_minimal_decisive"), mean(dec, "min_decisive_size"),
             mean(dec, "union_equals_injected")))
    for m in ("coupled", "indep", "anomaly"):
        print("   %-8s vs Dmin=%s vs F=%s" % (m, mean(dec, m + "_chan_ok"),
                                              mean(dec, m + "_injectionset_ok")
                                              or mean(dec, m + "_inj_ok")))
    h = {}
    for r in dec:
        n = r.get("n_minimal_decisive")
        h[n] = h.get(n, 0) + 1
    print("   |Dmin| hist: %s" % dict(sorted(h.items())))
    if any("shapley_top1_in_minimal" in r for r in dec):
        print("   shapley top-1 in Dmin=%s" % mean(dec, "shapley_top1_in_minimal"))
    if any("monotone_ok" in r for r in dec):
        ok = sum(1 for r in dec if r.get("monotone_ok"))
        print("   monotone %d/%d" % (ok, len(dec)))


def agreement(pa, pb):
    _, ra = load(pa)
    _, rb = load(pb)
    A = {r["ds_index"]: r for r in ra if "ds_index" in r}
    B = {r["ds_index"]: r for r in rb if "ds_index" in r}
    both = sorted(set(A) & set(B))
    same_keep = sum(1 for i in both
                    if bool(A[i].get("skipped")) == bool(B[i].get("skipped")))
    kk = [i for i in both if not A[i].get("skipped") and not B[i].get("skipped")]
    same_flip = sum(1 for i in kk
                    if bool(A[i].get("flipped")) == bool(B[i].get("flipped")))
    dd = [i for i in kk if A[i].get("flipped") and B[i].get("flipped")]
    same_fam = sum(1 for i in dd if fam(A[i]) == fam(B[i]))
    same_size = sum(1 for i in dd
                    if A[i].get("n_minimal_decisive") == B[i].get("n_minimal_decisive"))
    same_joint = sum(1 for i in dd
                     if bool(A[i].get("joint_decisive")) == bool(B[i].get("joint_decisive")))
    print("\n## agreement %s vs %s" % (os.path.basename(pa), os.path.basename(pb)))
    print("   common items=%d  same keep/skip=%d (%.3f)"
          % (len(both), same_keep, same_keep / len(both)))
    print("   both kept=%d  same flip=%d (%.3f)"
          % (len(kk), same_flip, same_flip / max(len(kk), 1)))
    print("   both decisive=%d  identical family=%d (%.3f)  same |Dmin|=%d (%.3f)"
          % (len(dd), same_fam, same_fam / max(len(dd), 1),
             same_size, same_size / max(len(dd), 1)))
    print("   same joint flag=%d (%.3f)" % (same_joint, same_joint / max(len(dd), 1)))
    ja = sum(1 for i in dd if A[i].get("joint_decisive"))
    jb = sum(1 for i in dd if B[i].get("joint_decisive"))
    print("   paired joint rate: A=%d/%d (%.4f)  B=%d/%d (%.4f)"
          % (ja, len(dd), ja / max(len(dd), 1), jb, len(dd), jb / max(len(dd), 1)))


def cluster_boot(path, pagemap, B=20000, seed=0):
    _, rs = load(path)
    pm = json.load(open(pagemap))["map"]
    dec = [r for r in rs if not r.get("skipped") and r.get("flipped")]
    by = {}
    miss = 0
    for r in dec:
        k = pm.get(str(r.get("ds_index")))
        if k is None:
            miss += 1
            k = "__unmapped_%d" % r.get("ds_index", 0)
        by.setdefault(k, []).append(1 if r.get("joint_decisive") else 0)
    docs = list(by.values())
    n = sum(len(v) for v in docs)
    k = sum(sum(v) for v in docs)
    rng = random.Random(seed)
    est = []
    for _ in range(B):
        num = den = 0
        for _ in range(len(docs)):
            g = docs[rng.randrange(len(docs))]
            num += sum(g)
            den += len(g)
        est.append(num / den)
    est.sort()
    lo, hi = est[int(0.025 * B)], est[int(0.975 * B)]
    sizes = {}
    for v in docs:
        sizes[len(v)] = sizes.get(len(v), 0) + 1
    print("\n## page-clustered bootstrap  %s" % os.path.basename(path))
    print("   decisive items=%d  distinct documents=%d  unmapped=%d" % (n, len(docs), miss))
    print("   items per document: %s" % dict(sorted(sizes.items())))
    print("   joint rate=%.4f  Wilson (item-level)=%s" % (k / n, wilson(k, n)))
    print("   cluster bootstrap 95%% CI=[%.4f, %.4f]  (B=%d)" % (lo, hi, B))
    multi = [v for v in docs if len(v) > 1]
    if multi:
        agree = sum(1 for v in multi if len(set(v)) == 1)
        print("   documents with >1 decisive item=%d, of which unanimous=%d (%.3f)"
              % (len(multi), agree, agree / len(multi)))


if __name__ == "__main__":
    for f in ("m2w_s4_coords.json", "m2w_s4_base_r.json"):
        if os.path.exists(f):
            digest(f)
    if os.path.exists("m2w_s4_base_r.json"):
        agreement("m2w_s4_base.json", "m2w_s4_base_r.json")
    if os.path.exists("m2w_s4_3ch_r.json"):
        agreement("m2w_s4_3ch_c.json", "m2w_s4_3ch_r.json")
    if os.path.exists("m2w_s4_coords.json"):
        agreement("m2w_s4_base.json", "m2w_s4_coords.json")
    if os.path.exists("docvqa_v3_base_r.json"):
        digest("docvqa_v3_base_r.json")
        agreement("docvqa_v3_base.json", "docvqa_v3_base_r.json")
    if os.path.exists("docvqa_v3_idefics3.json"):
        digest("docvqa_v3_idefics3.json")
    if os.path.exists("m2w_s4_idefics3.json"):
        digest("m2w_s4_idefics3.json")
    if os.path.exists("docvqa_pagemap.json"):
        cluster_boot("docvqa_v3_base.json", "docvqa_pagemap.json")
