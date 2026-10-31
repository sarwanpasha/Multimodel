"""
Chart-reading study, stage 3: faults that span several steps, and alternative
fault constructions.
===============================================================================

Two extensions over percept_vlm_v2.py.

(1) MULTI-STEP FAULTS.  With --n_fault_steps 2 the episode is corrupted at two
    different steps, both channels at each, so the corrupted set F has four
    (step, channel) sites and the restoration lattice has 15 nonempty subsets.
    The task is a sum over per-step reads, so a repair that fixes only one of
    the two damaged steps leaves the total wrong: the minimal decisive family
    is forced to span steps.  This is what makes the earliest-decisive-step
    clause and the monotone-repair assumption testable rather than assumed; in
    the single-step, two-channel design every restoration chain is
    (empty) < (singleton) < (pair) with both endpoints pinned, so nothing about
    monotonicity can be observed.

(2) FAULT DESIGN.  Each channel has three alternative corruptions of the same
    perceptual site, so the measured structure can be compared across recipes
    with the agent, the episodes and the seeds held fixed.

      visual: paint999     white out the target bar's printed value and paint
                           "999" over it (the stage-2 fault).
              erase        white out the value and paint nothing.
              paint_other  paint another bar's value: a plausible misread.

      struct: decoy999     set the target's tabulated value to 999 (stage 2).
              plausible    set it to another bar's value.
              drop         delete the target's row from the table.

Ground truth is unchanged: the minimal decisive restoration family, computed by
running every restoration.

Usage
  python percept_vlm_v3.py --n_fault_steps 2 --trials 200 --out X.json
  python percept_vlm_v3.py --visual_fault erase --struct_fault drop --out Y.json
"""
import argparse, itertools, json, random
import numpy as np

from percept_vlm_pilot import (
    VLMReader, build_episode, channel_disagreement,
)

CHANNELS = ("visual", "struct")


# --------------------------- fault constructions ---------------------------- #

def struct_text(step, faulted, kind):
    """The tabular channel, corrupted or not."""
    table = dict(zip(step["labels"], step["values"]))
    if not faulted:
        return step["struct"]
    tgt = step["target_label"]
    others = [l for l in step["labels"] if l != tgt]
    if kind == "decoy999":
        table[tgt] = 999
    elif kind == "plausible":
        table[tgt] = table[others[0]] if others else 999
    elif kind == "drop":
        table.pop(tgt, None)
    else:
        raise ValueError(kind)
    return "; ".join(f"{k}={v}" for k, v in table.items())


def visual_image(step, faulted, kind):
    """The rendered channel, corrupted or not."""
    if not faulted:
        return step["image"]
    from PIL import ImageDraw
    img = step["image"].copy()
    draw = ImageDraw.Draw(img)
    px, py = int(step["target_px"][0]), int(step["target_px"][1])
    draw.rectangle([px - 16, py - 9, px + 16, py + 9], fill=(255, 255, 255))
    if kind == "paint999":
        draw.text((px - 11, py - 7), "999", fill=(0, 0, 0))
    elif kind == "erase":
        pass
    elif kind == "paint_other":
        others = [v for i, v in enumerate(step["values"])
                  if i != step["target_idx"]]
        val = others[0] if others else 999
        draw.text((px - 11, py - 7), str(val), fill=(0, 0, 0))
    else:
        raise ValueError(kind)
    return img


# --------------------------- agent ------------------------------------------ #

def read_step(reader, step, faulted, cache, read_mode, vf, sf):
    faulted = set(faulted)
    img = visual_image(step, "visual" in faulted, vf)
    struct = struct_text(step, "struct" in faulted, sf)

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
    else:
        val = reader.read_value(img, struct, step["target_label"],
                                use_image=True, use_struct=False)
        if val is None:
            val = reader.read_value(img, struct, step["target_label"],
                                    use_image=False, use_struct=True)

    if cache is not None:
        cache[key] = val
    return val


def run_agent(reader, steps, faults, restores, cache, read_mode, vf, sf,
              semantics="per_channel"):
    fault_map = {f["step"]: set(f["channels"]) for f in faults}
    restore_set = set(restores)
    total = 0
    for s in steps:
        injected = fault_map.get(s["t"], set())
        if semantics == "atomic":
            fully = injected and all((s["t"], c) in restore_set for c in injected)
            effective = set() if fully else set(injected)
        else:
            effective = {c for c in injected if (s["t"], c) not in restore_set}
        v = read_step(reader, s, effective, cache, read_mode, vf, sf)
        total += 0 if v is None else v
    return total


# --------------------------- ground truth ----------------------------------- #

def order_key(I, earliest_first=True):
    """Increasing size, then earliest step touched, then channel order."""
    steps = sorted({t for t, _ in I})
    chans = sorted(c for _, c in I)
    return (len(I), steps[0] if earliest_first else -steps[0], steps, chans)


def candidate_sets(sites, max_size=None):
    """Every nonempty subset of the given sites, smallest and earliest first."""
    out = []
    top = len(sites) if max_size is None else min(max_size, len(sites))
    for r in range(1, top + 1):
        for combo in itertools.combinations(sorted(sites), r):
            out.append(frozenset(combo))
    return sorted(out, key=order_key)


def all_sites(steps):
    """Every (step, channel) site in the episode, corrupted or not.  The
    attribution procedures search this lattice; they are not told where the
    faults are.  The ground truth is enumerated over the corrupted sites only,
    which loses nothing: repairing an uncorrupted site is a no-op here, so
    PFE(I) = PFE(I intersect F) holds by construction, and every minimal set
    is a subset of F (Lemma 1)."""
    return frozenset((s["t"], c) for s in steps for c in CHANNELS)


def minimal_decisive(reader, steps, faults, sites, answer, cache, read_mode,
                     vf, sf, semantics):
    decisive = []
    for I in candidate_sets(sites):
        if run_agent(reader, steps, faults, I, cache, read_mode, vf, sf,
                     semantics) == answer:
            decisive.append(I)
    minimal = [I for I in decisive if not any(J < I for J in decisive)]
    return minimal, decisive


# --------------------------- attribution methods ---------------------------- #

def coupled_attribution(reader, steps, faults, search, answer, cache, read_mode,
                        vf, sf, semantics, max_size=4):
    """Increasing size, earliest step first, over the whole site lattice; the
    first decisive set wins.  Returns (set, rollouts used)."""
    n = 0
    for I in candidate_sets(search, max_size):
        n += 1
        if run_agent(reader, steps, faults, I, cache, read_mode, vf, sf,
                     semantics) == answer:
            return I, n
    return frozenset(), n


def independent_ablation(reader, steps, faults, search, answer, cache, read_mode,
                         vf, sf, semantics):
    """Singleton restorations only, over the whole lattice: the replay /
    ablation analog, which cannot express a joint repair."""
    hits = []
    for site in sorted(search):
        if run_agent(reader, steps, faults, [site], cache, read_mode, vf, sf,
                     semantics) == answer:
            hits.append(site)
    return frozenset({hits[-1]}) if hits else frozenset()


def anomaly_attribution(reader, steps, faults):
    """Cross-channel disagreement, no counterfactual check.  Flags every step
    whose disagreement exceeds mean + 0.5 sd and guesses the visual channel at
    each, so it can name several steps without being told how many are faulty;
    if none clears the threshold it names the single loudest step."""
    cache = {}          # private: this baseline keys its cache differently
    fault_map = {f["step"]: f for f in faults}
    scored = [(s["t"], channel_disagreement(reader, s, fault_map.get(s["t"]),
                                            cache)) for s in steps]
    v = np.array([d for _, d in scored], dtype=float)
    thr = v.mean() + 0.5 * v.std()
    flagged = [t for t, d in scored if d > thr]
    if not flagged:
        flagged = [max(scored, key=lambda x: x[1])[0]]
    return frozenset((t, "visual") for t in flagged)


def greedy_attribution(reader, steps, faults, search, answer, cache, read_mode,
                       vf, sf, semantics, max_size=4):
    """A stronger non-exhaustive baseline for multi-site faults: repeatedly add
    the site whose repair most reduces the distance between the reported total
    and the true one, stopping when the outcome is restored.  Uses the numeric
    structure of this task, which the categorical settings do not have."""
    chosen, n = set(), 0
    for _ in range(max_size):
        best, best_gap = None, None
        for site in sorted(search):
            if site in chosen:
                continue
            tot = run_agent(reader, steps, faults, chosen | {site}, cache,
                            read_mode, vf, sf, semantics)
            n += 1
            gap = abs(tot - answer)
            if best_gap is None or gap < best_gap:
                best, best_gap = site, gap
        if best is None:
            break
        chosen.add(best)
        if best_gap == 0:
            return frozenset(chosen), n
    return frozenset(chosen), n


# --------------------------- one trial -------------------------------------- #

def run_trial(reader, args, seed):
    steps, answer = build_episode(args.n_steps, seed)
    cache = {}
    vf, sf, sem = args.visual_fault, args.struct_fault, args.restore_semantics

    if run_agent(reader, steps, [], [], cache, args.read_mode, vf, sf,
                 sem) != answer:
        return {"skipped": True, "reason": "clean_wrong"}

    rng = random.Random(seed * 31 + 1)
    inj_steps = sorted(rng.sample(range(args.n_steps), args.n_fault_steps))
    chans = (list(CHANNELS) if args.fault_kind == "double"
             else [rng.choice(list(CHANNELS))])
    faults = [{"step": t, "channels": list(chans)} for t in inj_steps]
    sites = frozenset((t, c) for t in inj_steps for c in chans)

    flipped = run_agent(reader, steps, faults, [], cache, args.read_mode, vf,
                        sf, sem) != answer
    base = {"skipped": False, "flipped": flipped, "read_mode": args.read_mode,
            "fault_kind": args.fault_kind, "semantics": sem,
            "visual_fault": vf, "struct_fault": sf,
            "n_fault_steps": args.n_fault_steps,
            "inj_steps": inj_steps, "inj_channels": sorted(chans),
            "injected_sites": sorted([list(x) for x in sites])}
    if not flipped:
        base["outcome_masked"] = True
        base["vlm_calls"] = len(cache)
        return base

    minimal, decisive = minimal_decisive(reader, steps, faults, sites, answer,
                                         cache, args.read_mode, vf, sf, sem)
    gt = set(minimal)
    dset = set(decisive)
    base["decisive_sets"] = [sorted([list(x) for x in I]) for I in decisive]
    base["monotone_ok"] = int(all(J in dset
                                  for I in dset
                                  for J in candidate_sets(sites)
                                  if I < J))
    base["outcome_masked"] = False
    base["n_minimal_decisive"] = len(minimal)
    base["minimal_decisive"] = [sorted([list(x) for x in I]) for I in minimal]
    base["min_decisive_size"] = min(len(I) for I in minimal) if minimal else None
    base["union_equals_injected"] = int(
        (set().union(*minimal) == set(sites)) if minimal else False)
    base["joint_decisive"] = int(len(minimal) == 1
                                 and frozenset(sites) in gt
                                 and len(sites) > 1)

    # Theorem 2's earliest-decisive-step clause, measured.
    earliest = inj_steps[0]
    base["earliest_injected_step"] = earliest
    base["all_minimal_touch_earliest"] = int(
        all(any(t == earliest for t, _ in I) for I in minimal)) if minimal else 0
    base["any_minimal_skips_a_faulted_step"] = int(
        any({t for t, _ in I} != set(inj_steps) for I in minimal)) if minimal else 0
    base["minimal_step_spans"] = sorted(
        {len({t for t, _ in I}) for I in minimal}) if minimal else []

    search = all_sites(steps)
    calls_before = len(cache)
    cpl, n_roll = coupled_attribution(reader, steps, faults, search, answer,
                                      cache, args.read_mode, vf, sf, sem)
    base["coupled_rollouts"] = n_roll
    base["coupled_new_calls"] = len(cache) - calls_before
    base["search_lattice_size"] = 2 ** len(search) - 1
    ind = independent_ablation(reader, steps, faults, search, answer, cache,
                               args.read_mode, vf, sf, sem)
    ano = anomaly_attribution(reader, steps, faults)
    grd, n_grd = greedy_attribution(reader, steps, faults, search, answer,
                                    cache, args.read_mode, vf, sf, sem)
    base["greedy_rollouts"] = n_grd
    for name, pred in (("coupled", cpl), ("anomaly", ano), ("indep", ind),
                       ("greedy", grd)):
        base[f"{name}_chan_ok"] = int(pred in gt)
        base[f"{name}_inj_ok"] = int(pred == frozenset(sites))
        base[f"{name}_step_ok"] = int(
            bool(pred) and {t for t, _ in pred} <= set(inj_steps))
        base[f"{name}_pred"] = sorted([list(x) for x in pred])
    base["vlm_calls"] = len(cache)
    return base


# --------------------------- main ------------------------------------------- #

def wilson(k, n):
    if n == 0:
        return None
    p, z = k / n, 1.96
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return [(c - h) / d, (c + h) / d]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--n_steps", type=int, default=5)
    ap.add_argument("--n_fault_steps", type=int, default=1, choices=[1, 2])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--read_mode", default="fused",
                    choices=["image_primary", "struct_primary", "fused"])
    ap.add_argument("--fault_kind", default="double", choices=["single", "double"])
    ap.add_argument("--visual_fault", default="paint999",
                    choices=["paint999", "erase", "paint_other"])
    ap.add_argument("--struct_fault", default="decoy999",
                    choices=["decoy999", "plausible", "drop"])
    ap.add_argument("--restore_semantics", default="per_channel",
                    choices=["per_channel", "atomic"])
    ap.add_argument("--out", default="vlm_v3.json")
    args = ap.parse_args()

    reader = VLMReader(args.model, device=args.device)

    results = []
    for i in range(args.trials):
        try:
            r = run_trial(reader, args, seed=args.seed + i)
        except Exception as e:
            r = {"skipped": True, "reason": f"error:{type(e).__name__}"}
        r["trial_seed"] = args.seed + i
        results.append(r)
        if (i + 1) % 25 == 0:
            print(f"  trial {i + 1}/{args.trials}", flush=True)

    nonskip = [r for r in results if not r.get("skipped")]
    dec = [r for r in nonskip if r.get("flipped")]

    def mean(rows, k):
        v = [r[k] for r in rows if k in r and r[k] is not None]
        return float(np.mean(v)) if v else None

    summary = {
        "n_trials": len(results), "n_kept": len(nonskip), "n_decisive": len(dec),
        "flip_rate": mean(nonskip, "flipped"),
        "joint_decisive_rate": mean(dec, "joint_decisive"),
        "joint_decisive_ci": wilson(sum(r.get("joint_decisive", 0) for r in dec),
                                    len(dec)),
        "mean_n_minimal": mean(dec, "n_minimal_decisive"),
        "mean_min_size": mean(dec, "min_decisive_size"),
        "union_equals_injected": mean(dec, "union_equals_injected"),
        "all_minimal_touch_earliest": mean(dec, "all_minimal_touch_earliest"),
        "any_minimal_skips_a_faulted_step": mean(
            dec, "any_minimal_skips_a_faulted_step"),
        "mean_coupled_rollouts": mean(dec, "coupled_rollouts"),
        "mean_coupled_new_calls": mean(dec, "coupled_new_calls"),
        "mean_vlm_calls": mean(dec, "vlm_calls"),
    }
    for m in ("coupled", "anomaly", "indep", "greedy"):
        summary[f"{m}_chan_acc"] = mean(dec, f"{m}_chan_ok")
        summary[f"{m}_inj_acc"] = mean(dec, f"{m}_inj_ok")
        summary[f"{m}_step_acc"] = mean(dec, f"{m}_step_ok")

    out = {"args": vars(args), "summary": summary, "trials": results}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
