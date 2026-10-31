#!/usr/bin/env python3
"""
PERCEPT-ATTRIB v2 -- corrected restoration semantics + identifiable ground truth
===============================================================================

Two changes relative to `percept_vlm_pilot.py`, both of which change what the
numbers mean:

(1) PER-CHANNEL RESTORATION.  v1's `run_agent` restored a step only if *every*
    faulted channel at that step was in the restore set:

        restore = all((s["t"], c) in restore_set for c in chans)

    Under a double fault this makes a single-channel restoration a no-op: the run
    is identical to the fully-faulted run.  Consequently "no single restoration
    recovers the outcome" (masking) and "the joint restoration recovers it" were
    true by construction rather than by measurement, and coupled attribution was
    guaranteed to return exactly the injected channel set.  v2 restores channels
    individually, so a single-channel restoration on a double fault is a real
    intervention whose outcome is measured, not assumed.

    `--restore_semantics atomic` reproduces v1 exactly, so the contribution of the
    bug to the published numbers can be quantified rather than hand-waved.

(2) GROUND TRUTH = MINIMAL DECISIVE SET, NOT INJECTION SET.  v1 scored attribution
    against the set of channels the harness corrupted.  Theorem 1 does not claim
    that set is identifiable; it claims the *decisive set* D1 and the minimal
    decisive restoration sets are.  When a channel is corrupted but the agent never
    consults it (or a co-present channel already carries the value), that channel is
    not causally responsible and no counterfactual method can, or should, name it.
    v2 computes the minimal decisive restoration sets by exhaustive intervention and
    scores against them, and reports injection-set recovery separately and clearly
    labelled as the non-identifiable quantity it is.

Everything else (episode construction, chart rendering, fault injection, the VLM
wrapper, the Stage-1 prefilter) is imported unchanged from the v1 module so the two
are directly comparable.
"""
import argparse, itertools, json, random
import numpy as np

from percept_vlm_pilot import (
    VLMReader, build_episode, deface_chart_image, channel_disagreement,
)

CHANNELS = ("visual", "struct")


# --------------------------- agent with per-channel restoration ------------- #

def read_step(reader, step, faulted_channels, cache, read_mode):
    """One agent read with an explicit set of currently-faulted channels.

    `faulted_channels` is the set remaining faulty AFTER restoration has been
    applied, so partial restoration is expressible.  Reads are deterministic
    (greedy decoding, fixed inputs), so the cache is exact, and re-execution under
    an intervention IS the coupled counterfactual: there is no stochastic
    background to hold fixed.
    """
    img = step["image"]
    struct = step["struct"]
    faulted = set(faulted_channels)

    if "struct" in faulted:
        bad = dict(zip(step["labels"], step["values"]))
        bad[step["target_label"]] = 999
        struct = "; ".join(f"{k}={v}" for k, v in bad.items())
    if "visual" in faulted:
        img = deface_chart_image(img, step["target_px"])

    fault_key = ",".join(sorted(faulted)) if faulted else "clean"
    key = (step["t"], fault_key, read_mode)
    if cache is not None and key in cache:
        return cache[key]

    if read_mode == "fused":
        val = reader.read_value(img, struct, step["target_label"],
                                use_image=True, use_struct=True)
    elif read_mode == "struct_primary":
        val = reader.read_value(img, struct, step["target_label"],
                                use_image=False, use_struct=True)
        if val is None:
            val = reader.read_value(img, struct, step["target_label"],
                                    use_image=True, use_struct=False)
    else:  # image_primary
        val = reader.read_value(img, struct, step["target_label"],
                                use_image=True, use_struct=False)
        if val is None:
            val = reader.read_value(img, struct, step["target_label"],
                                    use_image=False, use_struct=True)

    if cache is not None:
        cache[key] = val
    return val


def run_agent(reader, steps, faults, restores=(), cache=None,
              read_mode="image_primary", semantics="per_channel"):
    """Roll out the episode under a set of (step, channel) restorations."""
    fault_map = {f["step"]: set(f["channels"]) for f in faults}
    restore_set = set(restores)
    total, reads = 0, []
    for s in steps:
        injected = fault_map.get(s["t"], set())
        if semantics == "atomic":
            # v1 behaviour: all-or-nothing, kept for quantifying the difference
            fully = injected and all((s["t"], c) in restore_set for c in injected)
            effective = set() if fully else set(injected)
        else:
            effective = {c for c in injected if (s["t"], c) not in restore_set}
        v = read_step(reader, s, effective, cache, read_mode)
        if v is None:
            v = 0
        reads.append(v)
        total += v
    return total, reads


# --------------------------- identifiable ground truth ---------------------- #

def all_restoration_sets(steps):
    """Every nonempty (step, channel-subset) restoration, smallest subsets first."""
    out = []
    for size in (1, 2):
        for s in steps:
            for combo in itertools.combinations(CHANNELS, size):
                out.append(frozenset((s["t"], c) for c in combo))
    return out


def minimal_decisive_sets(reader, steps, faults, true_answer, cache, read_mode,
                          semantics="per_channel"):
    """Exhaustively compute PFE(I) for every candidate restoration set I and return
    the MINIMAL decisive ones: those with PFE(I)=1 that contain no decisive proper
    subset.  This is the quantity Theorem 1 says is identifiable.  Cost is bounded
    (n_steps x 3 rollouts) and reads are cached, so it is cheap in VLM calls.
    """
    decisive = []
    for I in all_restoration_sets(steps):
        pred, _ = run_agent(reader, steps, faults, restores=I, cache=cache,
                            read_mode=read_mode, semantics=semantics)
        if pred == true_answer:
            decisive.append(I)
    minimal = [I for I in decisive
               if not any(J < I for J in decisive)]
    return minimal, decisive


def as_step_channels(I):
    """A restoration set -> (step, frozenset(channels)) when it lives at one step."""
    steps = {t for t, _ in I}
    if len(steps) != 1:
        return None
    t = steps.pop()
    return (t, frozenset(c for _, c in I))


# --------------------------- attribution methods ---------------------------- #

def coupled_attribution(reader, steps, faults, candidates, true_answer, cache,
                        read_mode, semantics="per_channel"):
    """Ours: earliest-first single-channel restorations, then joint restoration at a
    step if no single one is decisive (Prop. 3 regime)."""
    for t in sorted(candidates):
        for m in CHANNELS:
            pred, _ = run_agent(reader, steps, faults, restores=[(t, m)],
                                cache=cache, read_mode=read_mode,
                                semantics=semantics)
            if pred == true_answer:
                return (t, frozenset({m}))
    for t in sorted(candidates):
        pred, _ = run_agent(reader, steps, faults,
                            restores=[(t, "visual"), (t, "struct")],
                            cache=cache, read_mode=read_mode, semantics=semantics)
        if pred == true_answer:
            return (t, frozenset(CHANNELS))
    return (None, None)


def anomaly_attribution(reader, steps, faults, cache):
    """Baseline A: largest cross-channel disagreement, no counterfactual check.
    Can name a step but not a channel; guesses visual, as in v1."""
    best_t, best_score = None, -1
    fault_map = {f["step"]: f for f in faults}
    for s in steps:
        d = channel_disagreement(reader, s, fault_map.get(s["t"]), cache)
        if d > best_score:
            best_score, best_t = d, s["t"]
    return (best_t, frozenset({"visual"}))


def independent_ablation_attribution(reader, steps, faults, candidates,
                                     true_answer, cache, read_mode,
                                     semantics="per_channel"):
    """Baseline B (FAMAS-style): single restorations only, no earliest-first
    ordering, no joint pass."""
    recoveries = []
    for t in sorted(candidates):
        for m in CHANNELS:
            pred, _ = run_agent(reader, steps, faults, restores=[(t, m)],
                                cache=cache, read_mode=read_mode,
                                semantics=semantics)
            if pred == true_answer:
                recoveries.append((t, m))
    if not recoveries:
        at, _ = anomaly_attribution(reader, steps, faults, cache)
        return (at, frozenset({"visual"}))
    t, m = recoveries[-1]
    return (t, frozenset({m}))


# --------------------------- one trial -------------------------------------- #

def run_trial(reader, n_steps, seed, read_mode, fault_kind, semantics):
    steps, true_answer = build_episode(n_steps, seed)
    cache = {}

    clean_pred, _ = run_agent(reader, steps, faults=[], cache=cache,
                              read_mode=read_mode, semantics=semantics)
    if clean_pred != true_answer:
        return {"skipped": True}

    rng = random.Random(seed * 31 + 1)
    inj_step = rng.randint(0, n_steps - 1)
    if fault_kind == "double":
        inj_channels = list(CHANNELS)
    else:
        inj_channels = [rng.choice(list(CHANNELS))]
    faults = [{"step": inj_step, "channels": list(inj_channels)}]
    inj_set = frozenset(inj_channels)

    broken_pred, _ = run_agent(reader, steps, faults=faults, cache=cache,
                               read_mode=read_mode, semantics=semantics)
    flipped = (broken_pred != true_answer)

    # Stage-1 prefilter (isolated per-channel reads; independent of read_mode)
    fault_map = {f["step"]: f for f in faults}
    cand_scores = [(s["t"], channel_disagreement(reader, s, fault_map.get(s["t"]),
                                                 cache)) for s in steps]
    scores = np.array([c for _, c in cand_scores], dtype=float)
    thr = scores.mean() + 0.5 * scores.std()
    candidates = [t for (t, c) in cand_scores if c > thr] or list(range(n_steps))

    base = {
        "skipped": False, "flipped": flipped, "read_mode": read_mode,
        "fault_kind": fault_kind, "semantics": semantics,
        "inj_step": inj_step, "inj_channels": sorted(inj_channels),
        "prefilter_kept_fraction": len(candidates) / n_steps,
        "injected_in_candidates": inj_step in candidates,
    }

    if not flipped:
        base["redundancy_masked"] = True
        base["vlm_calls"] = len(cache)
        return base

    # ---- identifiable ground truth -------------------------------------- #
    minimal, decisive = minimal_decisive_sets(reader, steps, faults, true_answer,
                                              cache, read_mode, semantics)
    minimal_sc = [as_step_channels(I) for I in minimal]
    minimal_sc = [x for x in minimal_sc if x is not None]

    # Does any SINGLE-channel restoration at the injected step recover?  This is
    # the masking question Prop. 3 actually asks, now measured rather than assumed.
    single_recovers = []
    for m in CHANNELS:
        pred, _ = run_agent(reader, steps, faults, restores=[(inj_step, m)],
                            cache=cache, read_mode=read_mode, semantics=semantics)
        if pred == true_answer:
            single_recovers.append(m)
    joint_pred, _ = run_agent(reader, steps, faults,
                              restores=[(inj_step, "visual"),
                                        (inj_step, "struct")],
                              cache=cache, read_mode=read_mode,
                              semantics=semantics)

    base["n_minimal_decisive"] = len(minimal_sc)
    base["minimal_decisive"] = [[t, sorted(ms)] for (t, ms) in minimal_sc]
    base["single_channel_recovers"] = sorted(single_recovers)
    base["joint_restoration_recovers"] = int(joint_pred == true_answer)
    # genuine Prop-3 masking: both channels faulted, neither alone recovers,
    # the pair does.
    base["true_masking"] = int(fault_kind == "double"
                               and not single_recovers
                               and joint_pred == true_answer)

    # ---- attribution ------------------------------------------------------ #
    ct, cm = coupled_attribution(reader, steps, faults, candidates, true_answer,
                                 cache, read_mode, semantics)
    at, am = anomaly_attribution(reader, steps, faults, cache)
    it, im = independent_ablation_attribution(reader, steps, faults, candidates,
                                              true_answer, cache, read_mode,
                                              semantics)

    gt = set(minimal_sc)

    def score(t, m):
        if t is None or m is None:
            return 0, 0
        step_ok = int(any(t == gt_t for gt_t, _ in gt))
        set_ok = int((t, m) in gt)
        return step_ok, set_ok

    base["redundancy_masked"] = False
    for name, (t, m) in (("coupled", (ct, cm)), ("anomaly", (at, am)),
                         ("indep", (it, im))):
        s_ok, c_ok = score(t, m)
        base[f"{name}_step_ok"] = s_ok
        base[f"{name}_chan_ok"] = c_ok
        # secondary, NON-identifiable metric: did it name the injected set?
        base[f"{name}_inj_ok"] = int(t == inj_step and m == inj_set)
    base["vlm_calls"] = len(cache)
    return base


# --------------------------- main ------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--n_steps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--read_mode", default="fused",
                    choices=["image_primary", "struct_primary", "fused"])
    ap.add_argument("--fault_kind", default="double", choices=["single", "double"])
    ap.add_argument("--restore_semantics", default="per_channel",
                    choices=["per_channel", "atomic"],
                    help="'atomic' reproduces the v1 all-or-nothing restoration "
                         "so its contribution to the v1 numbers can be measured.")
    ap.add_argument("--out", default="vlm_v2.json")
    args = ap.parse_args()

    reader = VLMReader(args.model)
    results = []
    for i in range(args.trials):
        r = run_trial(reader, args.n_steps, seed=args.seed + i,
                      read_mode=args.read_mode, fault_kind=args.fault_kind,
                      semantics=args.restore_semantics)
        results.append(r)
        print(f"trial {i:4d}: {json.dumps(r)}", flush=True)

    nonskip = [r for r in results if not r.get("skipped")]
    decisive = [r for r in nonskip if r.get("flipped")]
    masked = [r for r in nonskip if not r.get("flipped")]

    def mean(rows, k):
        vals = [r[k] for r in rows if k in r]
        return float(np.mean(vals)) if vals else None

    def wilson(k, n):
        if n == 0:
            return None
        p, z = k / n, 1.96
        d = 1 + z * z / n
        c = p + z * z / (2 * n)
        h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
        return [(c - h) / d, (c + h) / d]

    nd = len(decisive)
    summary = {
        "model": args.model, "read_mode": args.read_mode,
        "fault_kind": args.fault_kind, "restore_semantics": args.restore_semantics,
        "seed": args.seed,
        "n_trials": args.trials,
        "n_skipped": sum(1 for r in results if r.get("skipped")),
        "n_nonskipped": len(nonskip),
        "n_decisive": nd,
        "n_outcome_masked": len(masked),
        "flip_rate": len(decisive) / max(1, len(nonskip)),
        "flip_rate_ci95": wilson(len(decisive), max(1, len(nonskip))),
        "true_masking_rate": mean(decisive, "true_masking"),
        "true_masking_ci95": wilson(sum(r.get("true_masking", 0) for r in decisive), nd),
        "mean_n_minimal_decisive": mean(decisive, "n_minimal_decisive"),
        "joint_restoration_recovers_rate": mean(decisive, "joint_restoration_recovers"),
        "coupled_step_acc": mean(decisive, "coupled_step_ok"),
        "coupled_chan_acc": mean(decisive, "coupled_chan_ok"),
        "coupled_chan_ci95": wilson(sum(r.get("coupled_chan_ok", 0) for r in decisive), nd),
        "anomaly_step_acc": mean(decisive, "anomaly_step_ok"),
        "anomaly_chan_acc": mean(decisive, "anomaly_chan_ok"),
        "anomaly_chan_ci95": wilson(sum(r.get("anomaly_chan_ok", 0) for r in decisive), nd),
        "indep_step_acc": mean(decisive, "indep_step_ok"),
        "indep_chan_acc": mean(decisive, "indep_chan_ok"),
        "indep_chan_ci95": wilson(sum(r.get("indep_chan_ok", 0) for r in decisive), nd),
        "coupled_injectionset_acc": mean(decisive, "coupled_inj_ok"),
        "anomaly_injectionset_acc": mean(decisive, "anomaly_inj_ok"),
        "indep_injectionset_acc": mean(decisive, "indep_inj_ok"),
        "prefilter_kept_fraction": mean(nonskip, "prefilter_kept_fraction"),
        "injected_in_candidates_rate": mean(nonskip, "injected_in_candidates"),
        "mean_vlm_calls": mean(nonskip, "vlm_calls"),
    }
    print("\n=== v2 SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k:34s}: {v}")
    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "summary": summary, "trials": results},
                  f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
