#!/usr/bin/env python3
"""
PERCEPT-ATTRIB — Real-VLM Document-Agent Pilot
==============================================
Validates perception-level fault attribution with a REAL open vision-language
model doing REAL perception, on PUBLIC, self-generated data (no dataset download,
no license issues). The model genuinely reads chart/table images; faults and
attribution are real, not simulated.

Setting: a multi-step document-reading agent answers a multi-hop arithmetic
question over a small series of charts. Two co-present perception channels
describe each chart:
    - VISUAL channel  : the rendered PNG (model reads values off the image)
    - STRUCT channel  : the underlying data as text (DOM/table analog)
At each step the agent reads one value via the fused channels and accumulates a
running total; the final answer is correct iff every read was correct.

We inject ONE single-channel perception fault at ONE step (either corrupt the
struct text value, or visually deface the chart so the model misreads it), verify
it flips the answer correct->wrong, then attribute via:
    (ours)      coupled counterfactual restoration, earliest-first, fixed inputs
    (baseline)  per-step self-consistency anomaly (which step's two channels or
                repeated reads disagree most) -- the LLM-as-judge analog

Ground truth is exact because WE generate the chart values, so the (step,channel)
of the injected fault is known -> deterministic labels (the property the concept
doc relies on).

Model: Qwen/Qwen2-VL-2B-Instruct (Apache-2.0, ~4GB). Runs on a single A40/A100.

AIR-GAP NOTE: if compute nodes have no internet, pre-download on a machine with internet access:
    python -c "from huggingface_hub import snapshot_download; \
        snapshot_download('Qwen/Qwen2-VL-2B-Instruct')"
then set HF_HUB_OFFLINE=1 in the job environment.
"""

import argparse
import io
import json
import os
import random
import numpy as np

# Headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


# --------------------------- data generation -------------------------------- #

def make_chart(values, labels, title, seed):
    """Render a bar chart PNG from known values. Returns (PIL.Image, label_xy)
    where label_xy[i] is the (px, py) pixel position of bar i's value label, so a
    visual fault can be placed exactly on the target bar's printed number."""
    rng = random.Random(seed)
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    colors = [(rng.random() * 0.5 + 0.2, rng.random() * 0.5 + 0.2,
               rng.random() * 0.6 + 0.3) for _ in values]
    bars = ax.bar(labels, values, color=colors)
    ax.set_title(title, fontsize=11)
    ymax = max(values) * 1.25
    ax.set_ylim(0, ymax)
    label_data = []  # (x_data, y_data) of each printed value
    for i, (b, v) in enumerate(zip(bars, values)):
        x_data = b.get_x() + b.get_width() / 2
        y_data = v + max(values) * 0.02
        ax.text(x_data, y_data, str(v), ha="center", fontsize=10)
        label_data.append((x_data, y_data))
    ax.tick_params(axis="x", labelsize=9)
    fig.canvas.draw()  # finalize transforms before querying pixel coords

    # convert each label's data coords -> display (pixel) coords. Matplotlib's
    # origin is bottom-left; PIL's is top-left, so flip y by the figure height.
    H_px = int(fig.bbox.height)
    label_xy = []
    for (xd, yd) in label_data:
        px, py = ax.transData.transform((xd, yd))
        label_xy.append((float(px), float(H_px - py)))

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB"), label_xy


def deface_chart_image(img, label_px):
    """
    Visually corrupt ONE bar's printed value so the model misreads it (a real
    visual perception fault). `label_px` is the exact (px, py) of the target bar's
    value label. We white-out that number and paint a misleading one over it.
    """
    img = img.copy()
    draw = ImageDraw.Draw(img)
    px, py = int(label_px[0]), int(label_px[1])
    draw.rectangle([px - 16, py - 9, px + 16, py + 9], fill=(255, 255, 255))
    draw.text((px - 11, py - 7), "999", fill=(0, 0, 0))
    return img


def build_episode(n_steps, seed):
    """
    One multi-step episode. Each step has a chart; the task reads a designated
    bar's value from each chart and sums them. Ground-truth answer = sum of reads.
    Returns list of step dicts with image, struct text, true value, target label.
    """
    rng = random.Random(seed)
    steps = []
    for t in range(n_steps):
        n_bars = rng.randint(3, 4)
        labels = [chr(ord("A") + i) for i in range(n_bars)]
        values = [rng.randint(10, 90) for _ in range(n_bars)]
        target = rng.randint(0, n_bars - 1)
        img, label_xy = make_chart(values, labels, f"Chart {t+1}", seed * 100 + t)
        struct = "; ".join(f"{labels[i]}={values[i]}" for i in range(n_bars))
        steps.append({
            "t": t,
            "image": img,
            "struct": struct,
            "labels": labels,
            "values": values,
            "target_idx": target,
            "target_label": labels[target],
            "true_value": values[target],
            "n_bars": n_bars,
            "target_px": label_xy[target],
        })
    answer = sum(s["true_value"] for s in steps)
    return steps, answer


# --------------------------- VLM perception --------------------------------- #

class VLMReader:
    """Wraps Qwen2-VL to read a single bar value from (image, struct) channels."""

    def __init__(self, model_id, device="cuda"):
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        # transformers >=4.5x routes AutoProcessor through a video processor that
        # pulls in torchvision; we only use still images. use_fast=True selects the
        # fast image processor and avoids the slow/video backend dependency.
        self.processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype="auto", device_map=device)
        self.device = device

    def read_value(self, image, struct_text, target_label, use_image=True,
                   use_struct=True):
        """
        Ask the model for the value of `target_label`, giving it whichever channels
        are enabled. Returns an int (best-effort parse) or None.
        """
        import torch
        content = []
        prompt = (f"What is the numeric value of bar '{target_label}'? "
                  f"Answer with ONLY the integer.")
        if use_image:
            content.append({"type": "image", "image": image})
        ctx = ""
        if use_struct:
            ctx = f"\nThe chart's underlying data is: {struct_text}."
        content.append({"type": "text", "text": prompt + ctx})

        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs = [image] if use_image else None
        inputs = self.processor(text=[text], images=image_inputs,
                                padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=12, do_sample=False)
        gen = out[:, inputs.input_ids.shape[1]:]
        ans = self.processor.batch_decode(gen, skip_special_tokens=True)[0]
        return _parse_int(ans)


def _parse_int(s):
    # Take the FIRST standalone integer in the response. Gluing all digits together
    # (the old behavior) corrupted answers like "75 (out of 100)" -> 75100, which
    # inflated the skip rate especially for larger, more verbose models.
    import re
    m = re.search(r"-?\d+", s)
    if m is None:
        return None
    try:
        return int(m.group())
    except ValueError:
        return None


# --------------------------- agent + faults --------------------------------- #

def agent_read_step(reader, step, fault=None, restore=False, cache=None,
                    read_mode="image_primary"):
    """
    The agent's read at one step under a chosen channel-fusion policy:
      - 'image_primary'  : read from the image; fall back to struct ONLY if the
                           image read is missing. A visual fault then has no clean
                           channel to hide behind -> single-channel faults become
                           decisive (high flip rate). This is the realistic GUI/
                           document-agent default.
      - 'struct_primary' : symmetric; struct leads, image is fallback.
      - 'fused'          : model sees BOTH channels at once. A single-channel fault
                           is usually masked by the clean channel -> this is the
                           REDUNDANCY regime (Prop. 3), kept as a labeled arm.

    A fault corrupts exactly one channel (or, for double faults, both) unless the
    channel is restored.  cache memoizes deterministic reads -> fewer VLM calls.
    """
    img = step["image"]
    struct = step["struct"]

    # which channels are faulted right now (supports single OR double faults)
    faulted = set()
    if fault and not restore:
        faulted = set(fault["channels"]) if "channels" in fault else {fault["channel"]}

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
    else:  # image_primary (default)
        val = reader.read_value(img, struct, step["target_label"],
                                use_image=True, use_struct=False)
        if val is None:
            val = reader.read_value(img, struct, step["target_label"],
                                    use_image=False, use_struct=True)

    if cache is not None:
        cache[key] = val
    return val


def run_agent(reader, steps, faults, restores=(), cache=None,
              read_mode="image_primary"):
    """Run the full multi-step agent; return (predicted_sum, per_step_reads).
    `restores` is a set of (step, channel) pairs to restore to veridical."""
    total = 0
    reads = []
    fault_map = {f["step"]: f for f in faults}
    restore_set = set(restores)
    for s in steps:
        f = fault_map.get(s["t"])
        # a fault is 'restored' only if ALL its faulted channels are in restore_set
        restore = False
        if f:
            chans = set(f["channels"]) if "channels" in f else {f["channel"]}
            restore = all((s["t"], c) in restore_set for c in chans)
        v = agent_read_step(reader, s, fault=f, restore=restore, cache=cache,
                            read_mode=read_mode)
        if v is None:
            v = 0
        reads.append(v)
        total += v
    return total, reads


def channel_disagreement(reader, step, fault, cache):
    """
    Stage-1 prefilter signal: do the two channels, read SEPARATELY, disagree?
    Large disagreement flags a candidate fault step. (Reads each channel in
    isolation, so this is independent of the agent's read_mode.)
    """
    img = step["image"]; struct = step["struct"]
    faulted = set()
    if fault:
        faulted = set(fault["channels"]) if "channels" in fault else {fault["channel"]}
    if "struct" in faulted:
        bad = dict(zip(step["labels"], step["values"]))
        bad[step["target_label"]] = 999
        struct = "; ".join(f"{k}={v}" for k, v in bad.items())
    if "visual" in faulted:
        img = deface_chart_image(img, step["target_px"])

    fkey = ",".join(sorted(faulted)) if faulted else "clean"
    kI = (step["t"], fkey, "img"); kS = (step["t"], fkey, "struct")
    if cache is not None and kI in cache:
        vi = cache[kI]
    else:
        vi = reader.read_value(img, struct, step["target_label"],
                               use_image=True, use_struct=False)
        if cache is not None: cache[kI] = vi
    if cache is not None and kS in cache:
        vs = cache[kS]
    else:
        vs = reader.read_value(img, struct, step["target_label"],
                               use_image=False, use_struct=True)
        if cache is not None: cache[kS] = vs
    if vi is None or vs is None:
        return 9999  # missing read is itself suspicious
    return abs(vi - vs)


# --------------------------- attribution ------------------------------------ #

def coupled_attribution(reader, steps, faults, candidates, true_answer, cache,
                        read_mode="image_primary"):
    """
    Ours: earliest-first restoration. Because VLM greedy decoding is deterministic
    and our inputs are fixed, restoration IS the coupled counterfactual (no
    stochastic U to control). First try each SINGLE-channel restoration; if none
    flips the outcome (the redundancy regime), try the JOINT restoration of both
    channels at a step. Returns (step, frozenset_of_channels) or (None, None).
    """
    # pass 1: single-channel restorations, earliest-first (Theorem 1/2)
    for t in sorted(candidates):
        for m in ("visual", "struct"):
            pred, _ = run_agent(reader, steps, faults, restores=[(t, m)],
                                cache=cache, read_mode=read_mode)
            if pred == true_answer:
                return (t, frozenset({m}))
    # pass 2: joint restoration at a step (Prop. 3 redundancy / double-fault)
    for t in sorted(candidates):
        pred, _ = run_agent(reader, steps, faults,
                            restores=[(t, "visual"), (t, "struct")],
                            cache=cache, read_mode=read_mode)
        if pred == true_answer:
            return (t, frozenset({"visual", "struct"}))
    return (None, None)


def anomaly_attribution(reader, steps, faults, cache):
    """
    Baseline A (anomaly / self-consistency, LLM-as-judge analog): attribute to the
    step with the largest cross-channel disagreement, WITHOUT counterfactual
    verification. Cannot identify WHICH channel is at fault from disagreement alone.
    """
    best_t, best_score = None, -1
    for s in steps:
        f = next((x for x in faults if x["step"] == s["t"]), None)
        d = channel_disagreement(reader, s, f, cache)
        if d > best_score:
            best_score, best_t = d, s["t"]
    return (best_t, frozenset({"visual"}))  # baseline guesses the channel


def independent_ablation_attribution(reader, steps, faults, candidates,
                                     true_answer, cache, read_mode="image_primary"):
    """
    Baseline B (independent ablation, FAMAS-style). Restores each (step, channel)
    INDEPENDENTLY and records whether that single restoration recovers the answer.
    Attributes to the single best-scoring (step, channel). Unlike ours, it (i) does
    NOT order earliest-first (ties broken arbitrarily -> mislocates under
    propagation) and (ii) NEVER tries a JOINT restoration -> structurally cannot
    attribute a double-channel fault, where no single-channel restoration helps.

    Returns (step, frozenset_of_channels) or (None, None). We make its scoring
    charitable (it sees the same restorations ours does) so the comparison is fair;
    the gap comes purely from missing ordering + joint reasoning.
    """
    recoveries = []  # (step, channel) that individually recover the answer
    for t in sorted(candidates):
        for m in ("visual", "struct"):
            pred, _ = run_agent(reader, steps, faults, restores=[(t, m)],
                                cache=cache, read_mode=read_mode)
            if pred == true_answer:
                recoveries.append((t, m))
    if not recoveries:
        # no single-channel restoration helps (e.g., a double fault). The baseline
        # has no joint-restoration pass, so it cannot attribute -> picks its single
        # best disagreement step+channel as a last resort (and will be wrong on the
        # channel for joint faults).
        at, _ = anomaly_attribution(reader, steps, faults, cache)
        return (at, frozenset({"visual"}))
    # no earliest-first preference: pick arbitrarily (last one found), modeling a
    # method without causal ordering. On single faults this often still lands right;
    # the separation shows up under propagation / multi-recovery.
    t, m = recoveries[-1]
    return (t, frozenset({m}))


# --------------------------- experiment ------------------------------------- #

def run_trial(reader, n_steps, seed, read_mode="image_primary", fault_kind="single"):
    """
    One trial. fault_kind: 'single' (one channel) or 'double' (both channels at the
    injected step -> guaranteed-decisive condition). For single faults that do NOT
    flip the answer, we DO NOT discard the trial: we record it as a redundancy/
    masking case (the clean co-present channel absorbed the fault, Prop. 3), which
    is itself informative data.
    """
    steps, true_answer = build_episode(n_steps, seed)
    cache = {}

    # clean run must be correct, else skip (model can't read it cleanly under this mode)
    clean_pred, _ = run_agent(reader, steps, faults=[], cache=cache, read_mode=read_mode)
    if clean_pred != true_answer:
        return {"skipped": True}

    rng = random.Random(seed * 31 + 1)
    inj_step = rng.randint(0, n_steps - 1)
    if fault_kind == "double":
        inj_channels = ["visual", "struct"]
        fault = {"step": inj_step, "channels": inj_channels}
    else:
        inj_channels = [rng.choice(["visual", "struct"])]
        fault = {"step": inj_step, "channel": inj_channels[0]}
    faults = [fault]
    inj_set = frozenset(inj_channels)

    broken_pred, _ = run_agent(reader, steps, faults=faults, cache=cache,
                               read_mode=read_mode)
    flipped = (broken_pred != true_answer)

    # Stage 1 prefilter (independent of read_mode; uses isolated channel reads)
    cand_scores = []
    for s in steps:
        f = next((x for x in faults if x["step"] == s["t"]), None)
        cand_scores.append((s["t"], channel_disagreement(reader, s, f, cache)))
    scores = np.array([c for _, c in cand_scores], dtype=float)
    thr = scores.mean() + 0.5 * scores.std()
    candidates = [t for (t, c) in cand_scores if c > thr] or list(range(n_steps))

    base = {
        "skipped": False,
        "flipped": flipped,
        "read_mode": read_mode,
        "fault_kind": fault_kind,
        "inj_step": inj_step,
        "inj_channels": sorted(inj_channels),
        "prefilter_kept_fraction": len(candidates) / n_steps,
        "injected_in_candidates": inj_step in candidates,
    }

    if not flipped:
        # REDUNDANCY/MASKING case: the fault did not change the outcome because a
        # clean co-present channel carried the correct value. Record it as such; do
        # not attribute (there is no outcome failure to attribute).
        base["redundancy_masked"] = True
        base["vlm_calls"] = len(cache)
        return base

    # decisive case: attribute
    ct, cm = coupled_attribution(reader, steps, faults, candidates, true_answer,
                                 cache, read_mode=read_mode)
    at, am = anomaly_attribution(reader, steps, faults, cache)
    it, im = independent_ablation_attribution(reader, steps, faults, candidates,
                                              true_answer, cache, read_mode=read_mode)

    base["redundancy_masked"] = False
    base["coupled_step_ok"] = int(ct == inj_step)
    # channel-correct: the restored channel-set matches the injected channel-set
    base["coupled_chan_ok"] = int(ct == inj_step and cm == inj_set)
    base["anomaly_step_ok"] = int(at == inj_step)
    base["anomaly_chan_ok"] = int(at == inj_step and am == inj_set)
    base["indep_step_ok"] = int(it == inj_step)
    base["indep_chan_ok"] = int(it == inj_step and im == inj_set)
    base["vlm_calls"] = len(cache)
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--trials", type=int, default=300,
                    help="trials per condition. Decisive-trial count depends on "
                         "flip rate, so 300+ is recommended for stable accuracy.")
    ap.add_argument("--n_steps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="vlm_results.json")
    ap.add_argument("--read_mode", default="image_primary",
                    choices=["image_primary", "struct_primary", "fused"],
                    help="agent channel-fusion policy. image_primary makes single "
                         "visual faults decisive; fused is the redundancy regime.")
    ap.add_argument("--fault_kind", default="single", choices=["single", "double"],
                    help="single-channel fault, or double (both channels) for a "
                         "guaranteed-decisive condition.")
    ap.add_argument("--dry_run", action="store_true",
                    help="generate data + check pipeline WITHOUT loading the VLM")
    args = ap.parse_args()

    if args.dry_run:
        print("[dry-run] building episodes and injecting faults, no VLM loaded")
        _dry_run(args)
        return

    reader = VLMReader(args.model)
    results = []
    n_skip = 0
    for i in range(args.trials):
        r = run_trial(reader, args.n_steps, seed=args.seed + i,
                      read_mode=args.read_mode, fault_kind=args.fault_kind)
        results.append(r)
        if r.get("skipped"):
            n_skip += 1
        print(f"trial {i:3d}: {r}")

    nonskip = [r for r in results if not r.get("skipped")]
    decisive = [r for r in nonskip if r.get("flipped")]
    masked = [r for r in nonskip if not r.get("flipped")]

    def acc(rows, k):
        return float(np.mean([r[k] for r in rows])) if rows else None

    summary = {
        "n_trials": args.trials,
        "n_skipped": n_skip,
        "n_nonskipped": len(nonskip),
        "n_decisive_flipped": len(decisive),
        "n_redundancy_masked": len(masked),
        "flip_rate_among_nonskipped": (len(decisive) / max(1, len(nonskip))),
        "redundancy_masking_rate": (len(masked) / max(1, len(nonskip))),
        # prefilter quality measured over ALL non-skipped trials
        "prefilter_kept_fraction": acc(nonskip, "prefilter_kept_fraction"),
        "injected_in_candidates_rate": acc(nonskip, "injected_in_candidates"),
        # attribution accuracy measured ONLY on decisive (flipped) trials
        "coupled_step_acc": acc(decisive, "coupled_step_ok"),
        "coupled_chan_acc": acc(decisive, "coupled_chan_ok"),
        "anomaly_step_acc": acc(decisive, "anomaly_step_ok"),
        "anomaly_chan_acc": acc(decisive, "anomaly_chan_ok"),
        "indep_ablation_step_acc": acc(decisive, "indep_step_ok"),
        "indep_ablation_chan_acc": acc(decisive, "indep_chan_ok"),
        "mean_vlm_calls": acc(decisive, "vlm_calls"),
    }
    print("\n=== SUMMARY ===")
    print(f"  read_mode={args.read_mode}  fault_kind={args.fault_kind}")
    for k, v in summary.items():
        print(f"  {k:32s}: {v}")
    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "summary": summary, "trials": results}, f,
                  indent=2)
    print(f"\nWrote {args.out}")


class _OracleReader:
    """A perfect reader for dry-run pipeline testing (no GPU/model)."""
    def read_value(self, image, struct_text, target_label, use_image=True,
                   use_struct=True):
        # parse the struct text if available; else can't read (simulates image-only)
        if use_struct and struct_text:
            for kv in struct_text.split(";"):
                k, _, v = kv.strip().partition("=")
                if k.strip() == target_label:
                    return int(v)
        # image-only: oracle "reads" perfectly via a side channel we won't have at
        # real inference; for dry-run we just return None to mimic uncertainty
        return None


def _dry_run(args):
    reader = _OracleReader()
    ok = 0
    for i in range(min(5, args.trials)):
        steps, ans = build_episode(args.n_steps, args.seed + i)
        pred, reads = run_agent(reader, steps, faults=[], cache={})
        print(f"  episode {i}: true_answer={ans} struct_only_pred={pred} reads={reads}")
        ok += int(pred == ans)
    print(f"[dry-run] struct-channel pipeline consistent on {ok}/"
          f"{min(5, args.trials)} episodes (image channel needs the real VLM)")


if __name__ == "__main__":
    main()
