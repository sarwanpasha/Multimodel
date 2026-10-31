#!/usr/bin/env python3
"""
Summarise sampling_stability shards.

usage: python analyze_stability.py OUT.json shard1.json shard2.json ...

Per step: recovery probability p(R) for every restoration R. An "audit" draws
one sample per restoration, checks that the all-faulted draw fails, and reads
off D* (minimal recovering restorations), the joint flag (no singleton
recovers but the full restoration does) and single-site blame (the set of
singleton restorations that recover). Two independent audits are simulated by
Monte Carlo and their disagreement rates reported, together with how often a
single sampled audit matches the greedy audit, and how often majority votes
over 5 replicates per restoration (two disjoint halves of the 10 draws)
disagree with each other.
"""
import itertools, json, random, sys
import numpy as np


def subsets(F):
    F = sorted(F)
    return [frozenset(c) for r in range(1, len(F) + 1)
            for c in itertools.combinations(F, r)]


def key(S):
    return ",".join(sorted(S)) if S else "none"


def dstar(rec, F):
    win = [R for R in subsets(F) if rec[key(R)]]
    return frozenset(R for R in win if not any(Q < R for Q in win))


def joint(rec, F):
    return int(rec[key(frozenset(F))] == 1 and
               not any(rec[key(frozenset([c]))] for c in F))


def singles(rec, F):
    return frozenset(c for c in F if rec[key(frozenset([c]))])


def boot(vals, B=2000, seed=0):
    v = np.asarray(vals, float)
    if len(v) == 0:
        return [None, None]
    r = np.random.default_rng(seed)
    m = [v[r.integers(0, len(v), len(v))].mean() for _ in range(B)]
    return [round(float(np.percentile(m, 2.5)), 3),
            round(float(np.percentile(m, 97.5)), 3)]


def main():
    out, files = sys.argv[1], sys.argv[2:]
    steps = []
    for f in files:
        steps += [s for s in json.load(open(f))["steps"] if not s["skipped"]]
    rnd = random.Random(0)
    per = []
    for s in steps:
        F = s["injected"]
        gt = s["gt"]
        hit = {k: [int(p == gt) for p in v] for k, v in s["sampled_preds"].items()}
        R = len(hit["none"])
        p = {k: float(np.mean(v)) for k, v in hit.items()}
        g = s["greedy_recovers"]
        gD, gJ, gS = dstar(g, F), joint(g, F), singles(g, F)

        def draw():
            while True:
                rec = {k: v[rnd.randrange(R)] for k, v in hit.items()}
                if rec["none"] == 0 or all(v == 1 for v in hit["none"]):
                    return rec
        trials = 400
        dD = dJ = dS = mD = mJ = 0
        for _ in range(trials):
            a, b = draw(), draw()
            dD += dstar(a, F) != dstar(b, F)
            dJ += joint(a, F) != joint(b, F)
            dS += singles(a, F) != singles(b, F)
            mD += dstar(a, F) == gD
            mJ += joint(a, F) == gJ
        half = R // 2
        maj = lambda lo, hi: {k: int(np.mean(v[lo:hi]) > 0.5) for k, v in hit.items()}
        A, B = maj(0, half), maj(half, R)
        per.append({
            "n_ch": len(F),
            "p_none": p["none"], "p_full": p[key(frozenset(F))],
            "frac_restorations_uncertain": float(np.mean(
                [0.1 <= p[key(Rs)] <= 0.9 for Rs in subsets(F)])),
            "pair_disagree_dstar": dD / trials,
            "pair_disagree_joint": dJ / trials,
            "pair_disagree_single": dS / trials,
            "single_matches_greedy_dstar": mD / trials,
            "single_matches_greedy_joint": mJ / trials,
            "maj5_disagree_dstar": int(dstar(A, F) != dstar(B, F)),
            "maj5_disagree_joint": int(joint(A, F) != joint(B, F)),
            "maj10_matches_greedy_dstar": int(dstar(maj(0, R), F) == gD),
            "greedy_joint": gJ,
            "maj10_joint": joint(maj(0, R), F),
        })
    summ = {}
    for nch in sorted({r["n_ch"] for r in per}):
        rows = [r for r in per if r["n_ch"] == nch]
        d = {"n_steps": len(rows)}
        for k in rows[0]:
            if k == "n_ch":
                continue
            vals = [r[k] for r in rows]
            d[k] = {"mean": round(float(np.mean(vals)), 3), "ci": boot(vals)}
        summ[f"{nch}ch"] = d
    json.dump({"summary": summ, "per_step": per}, open(out, "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
