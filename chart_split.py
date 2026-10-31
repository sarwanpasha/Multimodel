"""
Split-information charts: independent sufficiency as a controlled variable.
==========================================================================

The three settings of the main study differ on many axes at once, so the
cross-setting spread in the joint-decisive rate cannot on its own be attributed
to whether each channel carries an independently sufficient copy of the answer.
This condition varies that one property and holds everything else fixed.

Both modes use the same generator, the same 5-step sum-of-reads task, the same
seeds, the same reader, the same bar counts and value distribution, the same
fault placement and the same restoration semantics.  They differ only in how
the target is named:

  dup     the x axis is labelled with the item names and the table gives
          "A=57; B=23; ...".  Either channel alone answers the question.
          This reproduces the chart setting of the main study.

  split   the x axis is labelled with opaque bar codes and the table gives
          "A -> K4; B -> M2; ...".  The image carries the values but not which
          bar is the target; the table carries the identity but no values.
          Neither channel alone answers the question.

Faults, at one or two steps, corrupt both channels: the visual fault paints 999
over the target bar's printed value, and the structured fault points the target
name at another bar (split) or gives it a wrong value (dup).

Usage
  python chart_split.py --info split --trials 200 --out chart_split.json
  python chart_split.py --info dup   --trials 200 --out chart_dup.json
  python chart_split.py --info split --n_fault_steps 2 --out chart_split2.json
"""
import argparse, io, itertools, json, random, re
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

CHANNELS = ("visual", "struct")
CODE_LETTERS = "KMRTVWZ"


# --------------------------- rendering -------------------------------------- #

def make_chart(values, axis_labels, title, seed):
    """Bar chart with the value printed above each bar.  Returns the image and
    the pixel position of each printed value, so a fault can be placed exactly."""
    rng = random.Random(seed)
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    colors = [(rng.random() * 0.5 + 0.2, rng.random() * 0.5 + 0.2,
               rng.random() * 0.6 + 0.3) for _ in values]
    bars = ax.bar(axis_labels, values, color=colors)
    ax.set_title(title, fontsize=11)
    ax.set_ylim(0, max(values) * 1.25)
    pos = []
    for b, v in zip(bars, values):
        xd = b.get_x() + b.get_width() / 2
        yd = v + max(values) * 0.02
        ax.text(xd, yd, str(v), ha="center", fontsize=10)
        pos.append((xd, yd))
    ax.tick_params(axis="x", labelsize=9)
    fig.canvas.draw()
    h = int(fig.bbox.height)
    px = []
    for (xd, yd) in pos:
        x, y = ax.transData.transform((xd, yd))
        px.append((float(x), float(h - y)))
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB"), px


def build_episode(n_steps, seed, info):
    rng = random.Random(seed)
    steps = []
    for t in range(n_steps):
        n_bars = rng.randint(3, 4)
        names = [chr(ord("A") + i) for i in range(n_bars)]
        values = [rng.randint(10, 90) for _ in range(n_bars)]
        codes = []
        while len(codes) < n_bars:
            c = rng.choice(CODE_LETTERS) + str(rng.randint(1, 9))
            if c not in codes:
                codes.append(c)
        target = rng.randint(0, n_bars - 1)
        axis = names if info == "dup" else codes
        img, px = make_chart(values, axis, "Chart %d" % (t + 1),
                             seed * 100 + t)
        steps.append({"t": t, "image": img, "names": names, "codes": codes,
                      "values": values, "target_idx": target,
                      "target_name": names[target],
                      "true_value": values[target], "n_bars": n_bars,
                      "target_px": px[target]})
    return steps, sum(s["true_value"] for s in steps)


def struct_text(step, faulted, info, rng_seed):
    names, codes, vals = step["names"], step["codes"], step["values"]
    tgt = step["target_idx"]
    others = [i for i in range(step["n_bars"]) if i != tgt]
    j = random.Random(rng_seed).choice(others) if others else tgt
    if info == "dup":
        v = list(vals)
        if faulted:
            v[tgt] = 999
        return "; ".join("%s=%d" % (n, x) for n, x in zip(names, v))
    c = list(codes)
    if faulted:
        c[tgt] = codes[j]
    return "; ".join("%s -> bar %s" % (n, x) for n, x in zip(names, c))


def visual_image(step, faulted):
    if not faulted:
        return step["image"]
    img = step["image"].copy()
    d = ImageDraw.Draw(img)
    x, y = int(step["target_px"][0]), int(step["target_px"][1])
    d.rectangle([x - 16, y - 9, x + 16, y + 9], fill=(255, 255, 255))
    d.text((x - 11, y - 7), "999", fill=(0, 0, 0))
    return img


# --------------------------- reader ----------------------------------------- #

def _cast_inputs(inputs, model):
    try:
        dt = next(model.parameters()).dtype
    except Exception:
        return inputs
    for k in list(inputs.keys()):
        v = inputs[k]
        if hasattr(v, "is_floating_point") and v.is_floating_point() and v.dtype != dt:
            inputs[k] = v.to(dt)
    return inputs


class Reader:
    def __init__(self, model_id, device="cuda"):
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
        _tk = getattr(self.processor, "tokenizer", None)
        if _tk is not None and getattr(_tk, "pad_token", None) is None:
            _tk.pad_token = _tk.eos_token or _tk.unk_token
        mid = model_id.lower()
        if "qwen2.5-vl" in mid or "qwen2_5" in mid:
            from transformers import Qwen2_5_VLForConditionalGeneration as cls
        elif "qwen2-vl" in mid:
            from transformers import Qwen2VLForConditionalGeneration as cls
        else:
            from transformers import AutoModelForImageTextToText as cls
        self.model = cls.from_pretrained(model_id, torch_dtype="auto",
                                         device_map=device)
        self.device = "cuda" if device == "auto" else device

    def read_value(self, image, struct, target_name, info,
                   use_image=True, use_struct=True):
        import torch
        parts = ["You are reading a bar chart."]
        if use_image:
            parts.append("The chart is in the image. Each bar is labelled on "
                         "the x axis and its value is printed above it.")
        if use_struct:
            parts.append("A table gives each item's value."
                         if info == "dup" else
                         "A table gives which labelled bar belongs to each item.")
        parts.append('What is the value for item "%s"?' % target_name)
        prompt = " ".join(parts)
        if use_struct:
            prompt += "\n\n" + struct
        prompt += "\n\nAnswer with ONLY the number."
        content = ([{"type": "image"}] if use_image else []) + [
            {"type": "text", "text": prompt}]
        chat = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False, add_generation_prompt=True)
        try:
            inputs = self.processor(text=[chat],
                                    images=[image] if use_image else None,
                                    padding=True,
                                    return_tensors="pt").to(self.device)
        except (ValueError, TypeError):
            inputs = self.processor(text=[chat],
                                    images=[[image]] if use_image else None,
                                    padding=True,
                                    return_tensors="pt").to(self.device)
        inputs = _cast_inputs(inputs, self.model)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=8,
                                      do_sample=False)
        gen = out[:, inputs.input_ids.shape[1]:]
        ans = self.processor.batch_decode(gen, skip_special_tokens=True)[0]
        m = re.search(r"-?\d+", ans)
        return int(m.group()) if m else None


# --------------------------- agent ------------------------------------------ #

def read_step(reader, step, faulted, cache, read_mode, info, seed):
    faulted = set(faulted)
    key = (step["t"], ",".join(sorted(faulted)) or "clean", read_mode)
    if key in cache:
        return cache[key]
    img = visual_image(step, "visual" in faulted)
    st = struct_text(step, "struct" in faulted, info, seed + step["t"])
    use_image = read_mode in ("fused", "image_primary")
    use_struct = read_mode in ("fused", "struct_primary")
    val = reader.read_value(img, st, step["target_name"], info,
                            use_image=use_image, use_struct=use_struct)
    cache[key] = val
    return val


def run_agent(reader, steps, faults, restores, cache, read_mode, info, seed):
    fmap = {f["step"]: set(f["channels"]) for f in faults}
    rest = set(restores)
    total = 0
    for s in steps:
        f = set(fmap.get(s["t"], ()))
        f -= {c for (t, c) in rest if t == s["t"]}
        v = read_step(reader, s, f, cache, read_mode, info, seed)
        if v is None:
            return None
        total += v
    return total


def candidate_sets(sites):
    sites = sorted(sites)
    out = []
    for r in range(1, len(sites) + 1):
        for combo in itertools.combinations(sites, r):
            out.append(frozenset(combo))
    return out


def minimal_decisive(reader, steps, faults, sites, answer, cache, read_mode,
                     info, seed):
    dec = []
    for I in candidate_sets(sites):
        if run_agent(reader, steps, faults, I, cache, read_mode, info,
                     seed) == answer:
            dec.append(I)
    return [I for I in dec if not any(J < I for J in dec)], dec


def first_decisive(reader, steps, faults, sites, answer, cache, read_mode,
                   info, seed):
    for I in candidate_sets(sites):
        if run_agent(reader, steps, faults, I, cache, read_mode, info,
                     seed) == answer:
            return I
    return frozenset()


def single_site_hits(reader, steps, faults, sites, answer, cache, read_mode,
                     info, seed):
    return [s for s in sorted(sites)
            if run_agent(reader, steps, faults, [s], cache, read_mode, info,
                         seed) == answer]


# --------------------------- one trial -------------------------------------- #

def run_trial(reader, args, seed):
    steps, answer = build_episode(args.n_steps, seed, args.info)
    cache = {}
    if run_agent(reader, steps, [], [], cache, args.read_mode, args.info,
                 seed) != answer:
        return {"skipped": True, "reason": "clean_wrong"}

    rng = random.Random(seed * 31 + 1)
    inj_steps = sorted(rng.sample(range(args.n_steps), args.n_fault_steps))
    faults = [{"step": t, "channels": list(CHANNELS)} for t in inj_steps]
    sites = frozenset((t, c) for t in inj_steps for c in CHANNELS)

    flipped = run_agent(reader, steps, faults, [], cache, args.read_mode,
                        args.info, seed) != answer
    base = {"skipped": False, "flipped": flipped, "info": args.info,
            "n_fault_steps": args.n_fault_steps, "inj_steps": inj_steps,
            "injected_sites": sorted([list(x) for x in sites])}
    if not flipped:
        base["outcome_masked"] = True
        base["vlm_calls"] = len(cache)
        return base

    minimal, decisive = minimal_decisive(reader, steps, faults, sites, answer,
                                         cache, args.read_mode, args.info, seed)
    gt, dset = set(minimal), set(decisive)
    base["outcome_masked"] = False
    base["monotone_ok"] = int(all(J in dset for I in dset
                                  for J in candidate_sets(sites) if I < J))
    base["n_minimal_decisive"] = len(minimal)
    base["minimal_decisive"] = [sorted([list(x) for x in I]) for I in minimal]
    base["min_decisive_size"] = min(len(I) for I in minimal) if minimal else None
    singles = single_site_hits(reader, steps, faults, sites, answer, cache,
                               args.read_mode, args.info, seed)
    base["single_site_recovers"] = [list(s) for s in singles]
    base["no_single_repair"] = int(not singles)
    base["joint_decisive"] = int(len(minimal) == 1
                                 and frozenset(sites) in gt
                                 and len(sites) > 1)
    base["union_equals_injected"] = int(
        set().union(*minimal) == set(sites) if minimal else False)

    before = len(cache)
    cpl = first_decisive(reader, steps, faults, sites, answer, cache,
                         args.read_mode, args.info, seed)
    base["coupled_calls"] = len(cache) - before
    ind = frozenset({singles[-1]}) if singles else frozenset()
    for name, pred in (("coupled", cpl), ("indep", ind)):
        base[name + "_chan_ok"] = int(pred in gt)
        base[name + "_injectionset_ok"] = int(pred == frozenset(sites))
        base[name + "_pred"] = sorted([list(x) for x in pred])
    base["vlm_calls"] = len(cache)
    return base


def wilson(k, n):
    if n == 0:
        return None
    p, z = k / n, 1.96
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return [round((c - h) / d, 4), round((c + h) / d, 4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--n_steps", type=int, default=5)
    ap.add_argument("--n_fault_steps", type=int, default=1, choices=[1, 2])
    ap.add_argument("--info", default="split", choices=["split", "dup"])
    ap.add_argument("--read_mode", default="fused")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="chart_split.json")
    args = ap.parse_args()

    reader = Reader(args.model, device=args.device)
    rows = []
    for i in range(args.trials):
        try:
            r = run_trial(reader, args, seed=args.seed + i)
        except Exception as e:
            r = {"skipped": True,
                 "reason": "error:%s: %s" % (type(e).__name__, str(e)[:180])}
        r["trial"] = i
        rows.append(r)
        if i % 25 == 0:
            print("  [%d] kept=%d" % (i, sum(1 for x in rows
                                             if not x.get("skipped"))),
                  flush=True)

    keep = [r for r in rows if not r.get("skipped")]
    dec = [r for r in keep if r.get("flipped")]

    def mean(rs, k):
        v = [r[k] for r in rs if k in r and r[k] is not None]
        return float(np.mean(v)) if v else None

    fam, msz = {}, {}
    for r in dec:
        fam[r.get("n_minimal_decisive")] = fam.get(r.get("n_minimal_decisive"), 0) + 1
        msz[r.get("min_decisive_size")] = msz.get(r.get("min_decisive_size"), 0) + 1
    skips = {}
    for r in rows:
        if r.get("skipped"):
            skips[r.get("reason", "?")] = skips.get(r.get("reason", "?"), 0) + 1

    summary = {
        "model": args.model, "info": args.info,
        "n_fault_steps": args.n_fault_steps,
        "n_trials": len(rows), "n_kept": len(keep), "skip_reasons": skips,
        "clean_accuracy": len(keep) / max(1, len(rows)),
        "n_decisive": len(dec), "flip_rate": mean(keep, "flipped"),
        "joint_decisive_rate": mean(dec, "joint_decisive"),
        "joint_decisive_ci": wilson(
            sum(r.get("joint_decisive", 0) for r in dec), max(1, len(dec))),
        "no_single_repair_rate": mean(dec, "no_single_repair"),
        "no_single_repair_ci": wilson(
            sum(r.get("no_single_repair", 0) for r in dec), max(1, len(dec))),
        "mean_n_minimal": mean(dec, "n_minimal_decisive"),
        "mean_min_size": mean(dec, "min_decisive_size"),
        "family_size_hist": dict(sorted(fam.items(), key=lambda kv: str(kv[0]))),
        "min_size_hist": dict(sorted(msz.items(), key=lambda kv: str(kv[0]))),
        "union_equals_injected": mean(dec, "union_equals_injected"),
        "monotone_rate": mean(dec, "monotone_ok"),
        "coupled_chan_acc": mean(dec, "coupled_chan_ok"),
        "indep_chan_acc": mean(dec, "indep_chan_ok"),
        "coupled_injectionset_acc": mean(dec, "coupled_injectionset_ok"),
        "mean_coupled_calls": mean(dec, "coupled_calls"),
        "mean_vlm_calls": mean(keep, "vlm_calls"),
    }
    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "summary": summary, "trials": rows},
                  f, indent=2, default=str)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
