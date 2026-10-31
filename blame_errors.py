#!/usr/bin/env python3
"""
Blame-error analysis over existing result files (no model calls).

For every decisive step that records a minimal decisive family D* and a
method's blamed set P (coupled_pred, indep_pred, anomaly_pred, or the
Shapley top-1 channel), classify the blame:

  exact        P is itself a minimal repair (P in D*)
  sufficient   restoring P repairs the step (P contains some R in D*)
  wrongful     P names a channel that lies in no minimal repair (P - U(D*))
  under        P is smaller than every minimal repair (|P| < min |R|):
               a single culprit named for a failure that needed two
  missed       restoring P does not repair the step (not sufficient)
  vs_injected  P equals the injected set F (the benchmark's answer key)

usage: python blame_errors.py OUT.json file1.json [file2.json ...]
"""
import json, math, sys

METHODS = ["coupled", "indep", "anomaly", "shapley_top1"]


def wilson(k, n):
    if n == 0:
        return [None, None]
    z = 1.96
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def rows_of(j):
    for k in ("steps", "items", "results", "trials", "rows"):
        if isinstance(j.get(k), list):
            return j[k]
    return []


def main():
    out, files = sys.argv[1], sys.argv[2:]
    report = {}
    for f in files:
        try:
            j = json.load(open(f))
        except Exception as e:
            report[f] = {"error": str(e)[:100]}
            continue
        rows = [r for r in rows_of(j)
                if isinstance(r, dict) and not r.get("skipped")
                and r.get("minimal_decisive")]
        if not rows:
            report[f] = {"n": 0}
            continue
        res = {"n_decisive": len(rows)}
        for m in METHODS:
            ps = []
            for r in rows:
                if m == "shapley_top1":
                    v = r.get("shapley_top1")
                    P = {v} if v else None
                else:
                    v = r.get(f"{m}_pred")
                    if isinstance(v, list) and v and isinstance(v[0], list):
                        # a predicted family of repair sets: blame = its union
                        P = set().union(*map(set, v))
                    else:
                        P = set(v) if isinstance(v, list) else None
                if P is None:
                    continue
                try:
                    D = [set(R) for R in r["minimal_decisive"]]
                except TypeError:
                    continue   # nested layout (older chart files): not comparable
                U = set().union(*D)
                F = set(r.get("injected") or U)
                ps.append({
                    "exact": int(any(P == R for R in D)),
                    "sufficient": int(any(R <= P for R in D)),
                    "wrongful": int(bool(P - U)),
                    "under": int(len(P) < min(len(R) for R in D)),
                    "missed": int(not any(R <= P for R in D)),
                    "vs_injected": int(P == F),
                    "names_innocent_injected": int(bool(P & (F - U))),
                })
            if not ps:
                continue
            n = len(ps)
            res[m] = {"n": n}
            for k in ps[0]:
                c = sum(p[k] for p in ps)
                res[m][k] = {"rate": round(c / n, 3), "ci": wilson(c, n)}
        report[f] = res
    json.dump(report, open(out, "w"), indent=1)
    # compact table
    print("file | method | n | exact | suff | wrongful | under | missed | =F")
    for f, res in report.items():
        for m in METHODS:
            if m in res:
                r = res[m]
                print(f"{f} | {m} | {r['n']} | " + " | ".join(
                    f"{r[k]['rate']:.3f}" for k in
                    ("exact", "sufficient", "wrongful", "under", "missed",
                     "vs_injected")))


if __name__ == "__main__":
    main()
