#!/usr/bin/env python3
"""
Merge sharded percept_vlm_v2 runs into one result file.

Why shards: short jobs are easier to schedule on a shared GPU cluster than one
long job, and a crashed shard can be rerun on its own.

Why this is exact rather than an approximation: `run_trial` is called with
`seed=args.seed + i` for i in range(trials), so shards at
`--seed 0 --trials 50`, `--seed 50 --trials 50`, ... generate precisely the same
episode sequence, in the same order, as a single `--seed 0 --trials 200` run.
Concatenating their trial records and recomputing the summary reproduces the
unsharded result exactly.

The summary formulas below are duplicated from percept_vlm_v2.main rather than
imported, so merging does not require loading torch on a machine without a GPU.

Usage:
  python merge_shards.py --out merged.json shard0.json shard1.json ...
"""
import argparse, json, sys
import numpy as np


def wilson(k, n):
    if n == 0:
        return None
    p, z = k / n, 1.96
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return [(c - h) / d, (c + h) / d]


def summarize(results, cfg):
    nonskip = [r for r in results if not r.get("skipped")]
    decisive = [r for r in nonskip if r.get("flipped")]
    masked = [r for r in nonskip if not r.get("flipped")]

    def mean(rows, k):
        vals = [r[k] for r in rows if k in r]
        return float(np.mean(vals)) if vals else None

    nd = len(decisive)
    return {
        "model": cfg.get("model"), "read_mode": cfg.get("read_mode"),
        "fault_kind": cfg.get("fault_kind"),
        "restore_semantics": cfg.get("restore_semantics"),
        "seed": cfg.get("seed"),
        "n_trials": len(results),
        "n_skipped": sum(1 for r in results if r.get("skipped")),
        "n_nonskipped": len(nonskip),
        "n_decisive": nd,
        "n_outcome_masked": len(masked),
        "flip_rate": len(decisive) / max(1, len(nonskip)),
        "flip_rate_ci95": wilson(len(decisive), max(1, len(nonskip))),
        "true_masking_rate": mean(decisive, "true_masking"),
        "true_masking_ci95": wilson(
            sum(r.get("true_masking", 0) for r in decisive), nd),
        "mean_n_minimal_decisive": mean(decisive, "n_minimal_decisive"),
        "joint_restoration_recovers_rate": mean(
            decisive, "joint_restoration_recovers"),
        "coupled_step_acc": mean(decisive, "coupled_step_ok"),
        "coupled_chan_acc": mean(decisive, "coupled_chan_ok"),
        "coupled_chan_ci95": wilson(
            sum(r.get("coupled_chan_ok", 0) for r in decisive), nd),
        "anomaly_step_acc": mean(decisive, "anomaly_step_ok"),
        "anomaly_chan_acc": mean(decisive, "anomaly_chan_ok"),
        "anomaly_chan_ci95": wilson(
            sum(r.get("anomaly_chan_ok", 0) for r in decisive), nd),
        "indep_step_acc": mean(decisive, "indep_step_ok"),
        "indep_chan_acc": mean(decisive, "indep_chan_ok"),
        "indep_chan_ci95": wilson(
            sum(r.get("indep_chan_ok", 0) for r in decisive), nd),
        "coupled_injectionset_acc": mean(decisive, "coupled_inj_ok"),
        "anomaly_injectionset_acc": mean(decisive, "anomaly_inj_ok"),
        "indep_injectionset_acc": mean(decisive, "indep_inj_ok"),
        "prefilter_kept_fraction": mean(nonskip, "prefilter_kept_fraction"),
        "injected_in_candidates_rate": mean(nonskip, "injected_in_candidates"),
        "mean_vlm_calls": mean(nonskip, "vlm_calls"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--require", type=int, default=0,
                    help="if >0, fail unless exactly this many shards are given "
                         "and all exist, so a partial merge is never mistaken "
                         "for a complete run")
    ap.add_argument("shards", nargs="+")
    args = ap.parse_args()

    loaded = []
    for p in args.shards:
        try:
            with open(p) as f:
                loaded.append((p, json.load(f)))
        except FileNotFoundError:
            print(f"missing shard: {p}", file=sys.stderr)

    if args.require and len(loaded) != args.require:
        print(f"REFUSING to merge: have {len(loaded)} of {args.require} shards",
              file=sys.stderr)
        return 2
    if not loaded:
        print("no shards loaded", file=sys.stderr)
        return 2

    keys = ("model", "read_mode", "fault_kind", "restore_semantics", "n_steps")
    ref = {k: loaded[0][1]["config"].get(k) for k in keys}
    for p, d in loaded[1:]:
        got = {k: d["config"].get(k) for k in keys}
        if got != ref:
            print(f"condition mismatch in {p}:\n  expected {ref}\n  got      {got}",
                  file=sys.stderr)
            return 2

    spans = []
    for p, d in loaded:
        s, t = d["config"]["seed"], d["config"]["trials"]
        spans.append((s, s + t, p))
    spans.sort()
    for (a0, a1, pa), (b0, b1, pb) in zip(spans, spans[1:]):
        if a1 > b0:
            print(f"overlapping seed ranges: {pa} [{a0},{a1}) and "
                  f"{pb} [{b0},{b1})", file=sys.stderr)
            return 2

    by_path = {p: d for p, d in loaded}
    results = []
    for _, _, p in spans:
        results.extend(by_path[p]["trials"])

    cfg = dict(loaded[0][1]["config"])
    cfg["seed"] = spans[0][0]
    cfg["trials"] = sum(b - a for a, b, _ in spans)
    cfg["merged_from"] = [p for _, _, p in spans]

    summary = summarize(results, cfg)
    print(f"merged {len(loaded)} shards -> {cfg['trials']} trials "
          f"(seeds {spans[0][0]}..{spans[-1][1] - 1})")
    for k, v in summary.items():
        print(f"  {k:34s}: {v}")
    with open(args.out, "w") as f:
        json.dump({"config": cfg, "summary": summary, "trials": results},
                  f, indent=2)
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
