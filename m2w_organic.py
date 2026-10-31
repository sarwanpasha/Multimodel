"""
Organic perception failures on the web: no injected fault.
==========================================================

Every other web condition in this study corrupts a channel on purpose and then
asks which repairs recover the outcome.  This one injects nothing.  It takes the
steps the agent gets wrong on its own, under exactly the presentation used
elsewhere in this study, and asks the same question of the losses that the
perception pipeline introduces by itself:

  visual   the screenshot is cropped to the candidate region and downscaled to
           the working resolution (--lo_side).  Restoring it means rendering the
           same crop at --hi_side, so small glyphs and thin outlines survive.

  struct   the benchmark's cleaned element description is a tag, a role and one
           text attribute, truncated to 80 characters.  Restoring it means the
           full attribute record, untruncated.

  ocr      an OCR module reads the text inside each candidate box off the
           downscaled crop and makes real recognition errors.  Restoring it
           means the element's own text as the page reports it.

None of these is a corruption we designed: they are the lossy steps any agent of
this shape performs.  The factual reading is all three at their deployed
fidelity, and a restoration set is a set of channels raised to full fidelity, so
Definition 1 applies verbatim with F = the full channel set.  A step is decisive
when raising every channel recovers it, which is the checkable form of
Assumption 3 (faithfulness): failures no perception upgrade repairs are counted
and reported, not discarded.

Usage
  python m2w_organic.py --n_steps 150 --channels 3 --out m2w_organic.json
"""
import argparse, json, random, re
import numpy as np

import m2w_stage4 as M


def parse_full(cstr):
    """(untruncated attribute record, the element's own text)."""
    try:
        c = json.loads(cstr) if isinstance(cstr, str) else cstr
        attrs = c.get("attributes")
        if isinstance(attrs, str):
            attrs = json.loads(attrs)
        attrs = attrs if isinstance(attrs, dict) else {}
        tag = c.get("tag", "?")
        parts = []
        for k in ("role", "text", "aria_label", "value", "placeholder",
                  "title", "alt", "name", "type", "id", "class", "href"):
            v = attrs.get(k)
            if v:
                vv = re.sub(r"\s+", " ", str(v)).strip()[:60]
                if vv:
                    parts.append(k + "=" + vv)
        full = re.sub(r"\s+", " ", "<" + str(tag) + "> " + " ".join(parts))
        full = full.strip()[:240]
        text = (attrs.get("text") or attrs.get("aria_label") or
                attrs.get("value") or attrs.get("placeholder") or "")
        text = re.sub(r"\s+", " ", str(text)).strip()[:60]
        return (full if full else None), (text if text else "(no text)")
    except Exception:
        return None, None


def build_candidates(step, k_negatives, rng):
    """M.build_candidates, keeping the raw candidate record as a fourth field.

    The random draws are made in the same order and number as there, so with the
    same seed this condition runs on the same steps and the same candidate sets
    as the injected conditions.
    """
    pos = step.get("pos_candidates") or []
    neg = step.get("neg_candidates") or []
    if not pos:
        return None, None
    gd, gid, gb = M.parse_candidate(pos[0])
    if gd is None or gb is None:
        return None, None
    chosen = [(gd, gid, gb, pos[0])]
    for n in (rng.sample(neg, min(k_negatives * 3, len(neg))) if neg else []):
        d, nid, b = M.parse_candidate(n)
        if d is not None and b is not None and nid != gid:
            chosen.append((d, nid, b, n))
        if len(chosen) >= k_negatives + 1:
            break
    if len(chosen) < 2:
        return None, None
    rng.shuffle(chosen)
    gt = next(i for i, c in enumerate(chosen) if c[1] == gid)
    return chosen, gt


def build_item(step, args, rng):
    channels = (("visual", "struct", "ocr") if args.channels == 3
                else ("visual", "struct"))
    chosen, gt = build_candidates(step, args.k_negatives, rng)
    if chosen is None:
        return None, "no_candidates"
    full_img = step.get("screenshot")
    if full_img is None:
        return None, "no_screenshot"
    boxes = [c[2] for c in chosen]

    crop_lo, s_lo, o_lo = M.crop_window(full_img, boxes, max_side=args.lo_side)
    crop_hi, s_hi, o_hi = M.crop_window(full_img, boxes, max_side=args.hi_side)
    if crop_lo is None or crop_hi is None:
        return None, "degenerate_crop"
    boxes_lo = [M.to_crop_coords(b, s_lo, o_lo) for b in boxes]
    boxes_hi = [M.to_crop_coords(b, s_hi, o_hi) for b in boxes]

    descs_short = [c[0] for c in chosen]
    rich = [parse_full(c[3]) for c in chosen]
    descs_full = [(f if f else d) for (f, _), d in zip(rich, descs_short)]
    dom_text = [(t if t else "(no text)") for (_, t) in rich]

    item = {
        "img_bad": M.mark_candidates(crop_lo, boxes_lo),
        "img_ok": M.mark_candidates(crop_hi, boxes_hi),
        "listing_bad": M.render_listing(descs_short),
        "listing_ok": M.render_listing(descs_full),
        "has_struct": True, "has_ocr": "ocr" in channels,
        "gt": gt, "n": len(chosen),
        "notes": {"visual": "downscale_%d_to_%d" % (args.hi_side, args.lo_side),
                  "struct": "truncated_80_to_full",
                  "ocr": "easyocr_to_dom_text" if "ocr" in channels else None},
    }
    if "ocr" in channels:
        lines = M.ocr_per_box(crop_lo, boxes_lo)
        if lines is None:
            return None, "ocr_failed"
        item["ocr_bad"] = M.render_listing(lines)
        item["ocr_ok"] = M.render_listing(dom_text)
        item["ocr_agrees"] = int(lines[gt].strip().lower()
                                 == dom_text[gt].strip().lower())
    else:
        item["ocr_bad"] = item["ocr_ok"] = ""
        item["ocr_agrees"] = None
    item["struct_agrees"] = int(descs_short[gt] == descs_full[gt])
    return item, None


def run_step(reader, step, args, rng):
    channels = (("visual", "struct", "ocr") if args.channels == 3
                else ("visual", "struct"))
    item, why = build_item(step, args, rng)
    if item is None:
        return {"skipped": True, "reason": why}
    cache = {}
    injected = set(channels)

    # The factual reading: every channel at its deployed fidelity.
    if M.run_under(reader, item, injected, cache, args.read_mode) == item["gt"]:
        return {"skipped": True, "reason": "agent_succeeded"}

    # Decisive iff raising every channel to full fidelity recovers the step.
    flipped = M.run_under(reader, item, set(), cache,
                          args.read_mode) == item["gt"]
    base = {"skipped": False, "gate": False, "organic": True,
            "flipped": flipped, "injected": sorted(injected),
            "n_choices": item["n"], "n_channels": len(channels),
            "notes": item["notes"], "lo_side": args.lo_side,
            "hi_side": args.hi_side,
            "ocr_agrees_with_dom": item["ocr_agrees"],
            "struct_untruncated_equal": item["struct_agrees"]}
    if not flipped:
        base["outcome_masked"] = True
        base["vlm_calls"] = len(cache)
        return base

    minimal, decisive = M.minimal_decisive(reader, item, injected, cache,
                                           args.read_mode)
    gt_sets, dset = set(minimal), set(decisive)
    base["decisive_sets"] = [sorted(R) for R in decisive]
    base["monotone_ok"] = int(all(J in dset
                                  for I in dset
                                  for J in M.nonempty_subsets(sorted(injected))
                                  if I < J))
    singles = [m for m in sorted(injected)
               if M.run_under(reader, item, set(injected) - {m}, cache,
                              args.read_mode) == item["gt"]]
    base["outcome_masked"] = False
    base["n_minimal_decisive"] = len(minimal)
    base["minimal_decisive"] = [sorted(R) for R in minimal]
    base["min_decisive_size"] = min(len(R) for R in minimal) if minimal else None
    base["single_channel_recovers"] = singles
    base["no_single_repair"] = int(not singles)
    base["joint_decisive"] = int(frozenset(injected) in gt_sets
                                 and len(minimal) == 1
                                 and len(injected) > 1)
    base["union_equals_injected"] = int(
        set().union(*minimal) == set(injected) if minimal else False)

    before = len(cache)
    cpl = M.coupled_attribute(reader, item, injected, cache, args.read_mode,
                              channels)
    base["coupled_calls"] = len(cache) - before
    ind = M.independent_attribute(reader, item, injected, cache,
                                  args.read_mode, channels)
    ano = M.anomaly_attribute(reader, item, injected, cache, channels)
    for name, pred in (("coupled", cpl), ("anomaly", ano), ("indep", ind)):
        base[name + "_chan_ok"] = int(pred in gt_sets)
        base[name + "_injectionset_ok"] = int(pred == frozenset(injected))
        base[name + "_pred"] = sorted(pred)

    if len(injected) >= 3:
        phi = M.shapley(reader, item, injected, cache, args.read_mode, channels)
        base["shapley"] = {k: round(v, 4) for k, v in phi.items()}
        top = max(phi, key=lambda k: phi[k]) if phi else None
        base["shapley_top1"] = top
        base["shapley_top1_in_minimal"] = int(
            frozenset({top}) in gt_sets) if top else 0
    base["vlm_calls"] = len(cache)
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test_task")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n_steps", type=int, default=150,
                    help="number of organic failures to collect")
    ap.add_argument("--k_negatives", type=int, default=9)
    ap.add_argument("--lo_side", type=int, default=1280,
                    help="working resolution: the deployed visual channel")
    ap.add_argument("--hi_side", type=int, default=2560,
                    help="full fidelity: the restored visual channel")
    ap.add_argument("--max_seen", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--channels", type=int, default=3, choices=[2, 3])
    ap.add_argument("--read_mode", default="fused")
    ap.add_argument("--ocr_device", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--out", default="m2w_organic.json")
    args = ap.parse_args()

    M._OCR_GPU = (args.ocr_device == "gpu")

    from datasets import load_dataset
    ds = load_dataset("osunlp/Multimodal-Mind2Web", split=args.split)
    rng = random.Random(args.seed)
    reader = M.Reader(args.model, device=args.device)

    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    results, kept, seen = [], 0, 0
    for i in idxs:
        if kept >= args.n_steps or seen >= args.max_seen:
            break
        seen += 1
        try:
            r = run_step(reader, ds[i], args, rng)
        except Exception as e:
            r = {"skipped": True,
                 "reason": "error:%s: %s" % (type(e).__name__, str(e)[:180])}
        r["ds_index"] = int(i)
        results.append(r)
        if not r.get("skipped"):
            kept += 1
        if seen % 25 == 0:
            print("  [%d] failures=%d" % (seen, kept), flush=True)

    nonskip = [r for r in results if not r.get("skipped")]
    dec = [r for r in nonskip if r.get("flipped")]
    skip_reasons = {}
    for r in results:
        if r.get("skipped"):
            k = r.get("reason", "?")
            skip_reasons[k] = skip_reasons.get(k, 0) + 1

    def mean(rows, k):
        v = [r[k] for r in rows if k in r and r[k] is not None]
        return float(np.mean(v)) if v else None

    sizes = {}
    for r in dec:
        n = r.get("min_decisive_size")
        sizes[n] = sizes.get(n, 0) + 1
    fam = {}
    for r in dec:
        n = r.get("n_minimal_decisive")
        fam[n] = fam.get(n, 0) + 1

    summary = {
        "model": args.model, "channels": args.channels,
        "lo_side": args.lo_side, "hi_side": args.hi_side,
        "n_seen": seen, "n_organic_failures": len(nonskip),
        "skip_reasons": skip_reasons,
        "n_repairable": len(dec),
        "repairable_rate": mean(nonskip, "flipped"),
        "repairable_ci": M.wilson(len(dec), max(1, len(nonskip))),
        "no_single_repair_rate": mean(dec, "no_single_repair"),
        "no_single_repair_ci": M.wilson(
            sum(r.get("no_single_repair", 0) for r in dec), max(1, len(dec))),
        "joint_decisive_rate": mean(dec, "joint_decisive"),
        "joint_decisive_ci": M.wilson(
            sum(r.get("joint_decisive", 0) for r in dec), max(1, len(dec))),
        "mean_n_minimal": mean(dec, "n_minimal_decisive"),
        "mean_min_size": mean(dec, "min_decisive_size"),
        "min_size_hist": dict(sorted(sizes.items(), key=lambda kv: str(kv[0]))),
        "family_size_hist": dict(sorted(fam.items(), key=lambda kv: str(kv[0]))),
        "monotone_rate": mean(dec, "monotone_ok"),
        "ocr_agrees_with_dom": mean(nonskip, "ocr_agrees_with_dom"),
        "struct_untruncated_equal": mean(nonskip, "struct_untruncated_equal"),
        "mean_coupled_calls": mean(dec, "coupled_calls"),
    }
    for m in ("coupled", "anomaly", "indep"):
        summary[m + "_chan_acc"] = mean(dec, m + "_chan_ok")
    if args.channels == 3:
        summary["shapley_top1_in_minimal"] = mean(dec, "shapley_top1_in_minimal")

    out = {"args": vars(args), "summary": summary, "steps": results}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
