"""
Organic perception failures on document pages: no injected fault.
=================================================================

The companion of m2w_organic.py for the DocVQA setting.  Nothing is corrupted.
The factual reading of each channel is the one the agent actually receives, and
the veridical reading is the same channel at full fidelity:

  visual   the page rendered with its longer side at --lo_side, the working
           resolution.  Restoring it renders the same page at --hi_side, so
           small print survives.

  struct   the OCR transcript read off the working-resolution page, with the
           recognition errors that resolution causes.  Restoring it is the
           transcript read off the full-resolution page.

Neither upgrade consults the gold answer, so nothing leaks: both readings are
functions of the page alone.  A kept item is one the agent gets wrong with both
channels at their deployed fidelity, and it is decisive when raising both
recovers it, so Assumption 3 is checked rather than assumed.

Usage
  python docvqa_organic.py --n_items 150 --out docvqa_organic.json
"""
import argparse, json, random
import numpy as np

import docvqa_v2 as D
import docvqa_v3 as V

CHANNELS = ("visual", "struct")

_OCR = None


def _cpu_ocr(gpu=False):
    """EasyOCR on CPU: reproducible across node types, which matters because the
    transcript decides which items are kept."""
    global _OCR
    if _OCR is None:
        import easyocr
        _OCR = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _OCR


def rescale(img, side):
    w, h = img.size
    m = max(w, h)
    if m == side or m == 0:
        return img
    s = side / float(m)
    return img.resize((max(1, int(w * s)), max(1, int(h * s))))


def run_item(reader, raw, args, ds_index):
    img = raw["image"].convert("RGB")
    question = raw["question"]
    golds = raw["answers"] if isinstance(raw["answers"], list) else [raw["answers"]]
    golds = [g for g in golds if str(g).strip()]
    meta = {"ds_index": int(ds_index),
            "qid": raw.get("questionId") or raw.get("question_id")}
    if not golds:
        return dict(meta, skipped=True, reason="no_gold")

    img_hi = rescale(img, args.hi_side)
    img_lo = rescale(img, args.lo_side)
    toks_hi = D.ocr_tokens(img_hi)
    toks_lo = D.ocr_tokens(img_lo)
    if not toks_hi or not toks_lo:
        return dict(meta, skipped=True, reason="ocr_empty")
    txt_hi, txt_lo = D.transcript(toks_hi), D.transcript(toks_lo)

    item = {"img": img_hi, "img_bad": img_lo,
            "txt": txt_hi, "txt_bad": txt_lo,
            "question": question, "golds": golds}
    cache = {}
    injected = set(CHANNELS)

    factual = D.run_under(reader, item, injected, cache, args.read_mode)
    if D.correct(factual, golds):
        return dict(meta, skipped=True, reason="agent_succeeded")

    flipped = D.correct(D.run_under(reader, item, set(), cache, args.read_mode),
                        golds)
    base = dict(meta, skipped=False, organic=True, flipped=flipped,
                read_mode=args.read_mode, injected=sorted(injected),
                lo_side=args.lo_side, hi_side=args.hi_side,
                n_tokens_lo=len(toks_lo), n_tokens_hi=len(toks_hi),
                transcripts_equal=int(txt_lo.strip() == txt_hi.strip()))
    if not flipped:
        base["outcome_masked"] = True
        base["vlm_calls"] = len(cache)
        return base

    minimal, _ = D.minimal_decisive(reader, item, injected, cache,
                                    args.read_mode)
    gt = set(minimal)
    singles = [m for m in CHANNELS
               if D.correct(D.run_under(reader, item, set(injected) - {m},
                                        cache, args.read_mode), golds)]
    base["outcome_masked"] = False
    base["n_minimal_decisive"] = len(minimal)
    base["minimal_decisive"] = [sorted(R) for R in minimal]
    base["min_decisive_size"] = min(len(R) for R in minimal) if minimal else None
    base["single_channel_recovers"] = singles
    base["no_single_repair"] = int(not singles)
    base["joint_decisive"] = int(not singles and frozenset(CHANNELS) in gt)
    base["union_equals_injected"] = int(
        (set().union(*minimal) == set(injected)) if minimal else False)
    for name, pred in (
        ("coupled", D.coupled_attribute(reader, item, injected, cache,
                                        args.read_mode)),
        ("anomaly", D.anomaly_attribute(reader, item, injected, cache)),
        ("indep", D.independent_attribute(reader, item, injected, cache,
                                          args.read_mode)),
    ):
        base[name + "_chan_ok"] = int(pred in gt)
        base[name + "_injectionset_ok"] = int(pred == frozenset(injected))
        base[name + "_pred"] = sorted(pred)
    base["vlm_calls"] = len(cache)
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n_items", type=int, default=150,
                    help="number of organic failures to collect")
    ap.add_argument("--lo_side", type=int, default=1000)
    ap.add_argument("--hi_side", type=int, default=2000)
    ap.add_argument("--max_seen", type=int, default=1200)
    ap.add_argument("--read_mode", default="fused")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="docvqa_organic.json")
    args = ap.parse_args()

    D.get_ocr = _cpu_ocr

    from datasets import load_dataset
    print("loading DocVQA validation split ...", flush=True)
    ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation")
    ds = ds.shuffle(seed=args.seed)

    if not args.model.lower().startswith("qwen/"):
        D.ask = V.ask_generic
    reader = V.Reader(args.model, device=args.device)

    results, kept, seen = [], 0, 0
    for i in range(len(ds)):
        if kept >= args.n_items or seen >= args.max_seen:
            break
        seen += 1
        try:
            r = run_item(reader, ds[i], args, ds_index=i)
        except Exception as e:
            r = {"ds_index": int(i), "skipped": True,
                 "reason": "error:%s: %s" % (type(e).__name__, str(e)[:180])}
        results.append(r)
        if not r.get("skipped"):
            kept += 1
        if seen % 25 == 0:
            print("  [%d] failures=%d" % (seen, kept), flush=True)

    nonskip = [r for r in results if not r.get("skipped")]
    dec = [r for r in nonskip if r.get("flipped")]
    skips = {}
    for r in results:
        if r.get("skipped"):
            k = r.get("reason", "?")
            skips[k] = skips.get(k, 0) + 1

    def mean(rows, k):
        v = [r[k] for r in rows if k in r and r[k] is not None]
        return float(np.mean(v)) if v else None

    fam, msz = {}, {}
    for r in dec:
        fam[r.get("n_minimal_decisive")] = fam.get(r.get("n_minimal_decisive"), 0) + 1
        msz[r.get("min_decisive_size")] = msz.get(r.get("min_decisive_size"), 0) + 1

    summary = {
        "model": args.model, "lo_side": args.lo_side, "hi_side": args.hi_side,
        "n_seen": seen, "n_organic_failures": len(nonskip),
        "skip_reasons": skips,
        "n_repairable": len(dec),
        "repairable_rate": mean(nonskip, "flipped"),
        "repairable_ci": D.wilson(len(dec), max(1, len(nonskip))),
        "no_single_repair_rate": mean(dec, "no_single_repair"),
        "no_single_repair_ci": D.wilson(
            sum(r.get("no_single_repair", 0) for r in dec), max(1, len(dec))),
        "joint_decisive_rate": mean(dec, "joint_decisive"),
        "joint_decisive_ci": D.wilson(
            sum(r.get("joint_decisive", 0) for r in dec), max(1, len(dec))),
        "mean_n_minimal": mean(dec, "n_minimal_decisive"),
        "mean_min_size": mean(dec, "min_decisive_size"),
        "family_size_hist": dict(sorted(fam.items(), key=lambda kv: str(kv[0]))),
        "min_size_hist": dict(sorted(msz.items(), key=lambda kv: str(kv[0]))),
        "union_equals_injected": mean(dec, "union_equals_injected"),
        "transcripts_equal": mean(nonskip, "transcripts_equal"),
        "mean_tokens_lo": mean(nonskip, "n_tokens_lo"),
        "mean_tokens_hi": mean(nonskip, "n_tokens_hi"),
    }
    for m in ("coupled", "anomaly", "indep"):
        summary[m + "_chan_acc"] = mean(dec, m + "_chan_ok")

    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "summary": summary, "items": results},
                  f, indent=2, default=str)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
