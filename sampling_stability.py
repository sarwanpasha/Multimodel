#!/usr/bin/env python3
"""
Sampling stability of the counterfactual audit (injected web condition).

For each step that the greedy reader gets right when clean and wrong when every
channel is faulted, replay every restoration (every subset R of the injected
set F, from nothing restored to everything restored) R_REP times with
temperature sampling. The raw predictions are stored so that
analyze_stability.py can estimate per-restoration recovery probabilities and
how often two independent sampled audits disagree on the minimal decisive
family, the joint flag, or single-site blame.

Item construction is identical to m2w_stage4 (same build_item, same faults);
each step gets its own rng so shards are independent and reproducible.
"""
import argparse, itertools, json, random, time
from types import SimpleNamespace

import m2w_stage4 as M


def all_subsets(ch):
    ch = sorted(ch)
    return [frozenset(c) for r in range(len(ch) + 1)
            for c in itertools.combinations(ch, r)]


def key(S):
    return ",".join(sorted(S)) if S else "none"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--channels", type=int, default=2, choices=[2, 3])
    ap.add_argument("--n_steps", type=int, default=40,
                    help="decisive steps to keep in this shard")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--max_tries", type=int, default=2000)
    ap.add_argument("--split", default="test_task")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    M._OCR_GPU = False
    margs = SimpleNamespace(
        split=args.split, model=args.model, device="cuda", n_steps=0,
        k_negatives=9, max_side=1280, seed=args.seed, channels=args.channels,
        read_mode="fused", fault_kind="all", visual_fault="blur_grey",
        struct_fault="decoy_swap", ocr_fault="decoy_swap",
        struct_style="plain", ocr_device="cpu", gate_only=False)
    channels = (("visual", "struct", "ocr") if args.channels == 3
                else ("visual", "struct"))
    F = frozenset(channels)

    ds = load_dataset("osunlp/Multimodal-Mind2Web", split=args.split)
    idxs = list(range(len(ds)))
    random.Random(args.seed).shuffle(idxs)
    idxs = idxs[args.shard::args.nshards][:args.max_tries]

    reader = M.Reader(args.model, device="cuda")
    gen0 = reader.model.generate
    state = {"sample": False}

    def generate(*a, **kw):
        if state["sample"]:
            # Qwen2.5-VL ships top_k=1/top_p~0 in its generation config, which
            # would make "sampling" greedy; switch those filters off.
            kw.update(do_sample=True, temperature=args.temp, top_p=1.0,
                      top_k=0)
        return gen0(*a, **kw)
    reader.model.generate = generate

    rows, kept, t0 = [], 0, time.time()
    for i in idxs:
        if kept >= args.n_steps:
            break
        try:
            rng = random.Random(args.seed * 1000003 + i)
            item, why = M.build_item(ds[i], margs, rng)
            if item is None:
                rows.append({"ds_index": i, "skipped": True, "reason": why})
                continue
            state["sample"] = False
            cache = {}
            gt = item["gt"]
            if M.run_under(reader, item, set(), cache, "fused") != gt:
                rows.append({"ds_index": i, "skipped": True,
                             "reason": "clean_wrong"})
                continue
            if M.run_under(reader, item, set(F), cache, "fused") == gt:
                rows.append({"ds_index": i, "skipped": True,
                             "reason": "not_flipped"})
                continue
            greedy = {key(R): int(M.run_under(reader, item, set(F - R), cache,
                                              "fused") == gt)
                      for R in all_subsets(F)}
            state["sample"] = True
            sampled = {}
            for si, R in enumerate(all_subsets(F)):
                preds = []
                for r in range(args.reps):
                    torch.manual_seed(args.seed * 7919 + i * 131 + si * 17 + r)
                    preds.append(M.run_under(reader, item, set(F - R), {},
                                             "fused"))
                sampled[key(R)] = preds
            state["sample"] = False
            rows.append({"ds_index": i, "skipped": False, "gt": gt,
                         "n_choices": item["n"], "injected": sorted(F),
                         "greedy_recovers": greedy, "sampled_preds": sampled})
            kept += 1
            print(f"  kept={kept} tried={len(rows)} "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
        except Exception as e:
            state["sample"] = False
            rows.append({"ds_index": i, "skipped": True,
                         "reason": f"error:{type(e).__name__}: {str(e)[:180]}"})
    json.dump({"args": vars(args), "steps": rows}, open(args.out, "w"))
    print(f"wrote {args.out}: kept={kept} of {len(rows)} tried", flush=True)


if __name__ == "__main__":
    main()
