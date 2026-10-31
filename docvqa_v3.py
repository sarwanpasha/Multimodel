"""
DocVQA stage 3: fault-design sensitivity, matched item ids, larger models.
=========================================================================

Same agent, same pages, same ground-truth procedure as docvqa_v2.py.  Three
things change.

(1) FAULT DESIGN.  The visual and textual corruptions of the located answer
    each have three alternative constructions, so the measured structure can be
    compared across recipes with everything else held fixed.

      visual: occlude    blur the answer region, then paint a grey box over it
                         (the stage-2 fault).
              blur_only  blur the answer region and paint nothing.
              whiteout   paint a white box, as if the region were never printed.

      struct: decoy      replace the answer tokens with a same-page decoy token
                         (the stage-2 fault).
              delete     remove the answer tokens from the transcript.
              swap       exchange the answer tokens with the decoy token in
                         place, a transposition rather than a substitution.

(2) IDENTIFIED ITEMS.  Every record carries the dataset index and, when the
    split provides one, the question id, so runs with different models can be
    compared on the pages both of them read correctly rather than on whatever
    each happened to keep.

(3) --device auto, so a model too large for one GPU can be sharded.

Usage
  python docvqa_v3.py --visual_fault whiteout --struct_fault delete --out X.json
  python docvqa_v3.py --model Qwen/Qwen2.5-VL-32B-Instruct --device auto --out Y.json
"""
import argparse, json, random
import numpy as np

import docvqa_v2 as D

CHANNELS = D.CHANNELS


def ask_generic(reader, img, text_channel, question, read_mode, cache, key):
    """Same contract as docvqa_v2.ask, but the chat template gets a bare image
    placeholder rather than a Qwen-specific one, so non-Qwen readers work."""
    import torch
    if cache is not None and key in cache:
        return cache[key]
    use_image = read_mode in ("fused", "image_primary")
    use_struct = read_mode in ("fused", "struct_primary")
    prompt = f"This is a document page. {question}"
    if use_struct:
        prompt += "\n\nOCR transcript of the page:\n" + text_channel
    prompt += "\n\nAnswer with ONLY the value, no explanation."
    content = ([{"type": "image"}] if use_image else []) + [
        {"type": "text", "text": prompt}]
    chat = reader.processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False, add_generation_prompt=True)
    images = [img] if use_image else None
    try:
        inputs = reader.processor(text=[chat], images=images, padding=True,
                                  return_tensors="pt").to(reader.device)
    except (ValueError, TypeError):
        inputs = reader.processor(text=[chat], images=[[img]] if use_image else None,
                                  padding=True,
                                  return_tensors="pt").to(reader.device)
    inputs = _cast_inputs(inputs, reader.model)
    with torch.no_grad():
        out = reader.model.generate(**inputs, max_new_tokens=24, do_sample=False)
    gen = out[:, inputs.input_ids.shape[1]:]
    ans = reader.processor.batch_decode(gen, skip_special_tokens=True)[0].strip()
    if cache is not None:
        cache[key] = ans
    return ans


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
    """Qwen2-VL or Qwen2.5-VL, optionally sharded across GPUs.

    docvqa_v2.ask() only touches .processor, .model and .device, so this is a
    drop-in replacement for percept_vlm_pilot.VLMReader that also covers the
    2.5 family and models too large for a single card.
    """

    def __init__(self, model_id, device="cuda"):
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
        _tk = getattr(self.processor, "tokenizer", None)
        if _tk is not None and getattr(_tk, "pad_token", None) is None:
            _tk.pad_token = _tk.eos_token or _tk.unk_token
        mid = model_id.lower()
        cls = None
        if "qwen2.5-vl" in mid or "qwen2_5" in mid:
            from transformers import Qwen2_5_VLForConditionalGeneration as cls
        elif "qwen2-vl" in mid:
            from transformers import Qwen2VLForConditionalGeneration as cls
        else:
            from transformers import AutoModelForImageTextToText as cls
        self.model = cls.from_pretrained(model_id, torch_dtype="auto",
                                         device_map=device)
        self.device = "cuda" if device == "auto" else device


# --------------------------- fault constructions ---------------------------- #

def deface(img, bbox, kind, pad=6):
    from PIL import ImageDraw, ImageFilter
    im = img.copy()
    x0, y0, x1, y1 = bbox
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(im.size[0], x1 + pad), min(im.size[1], y1 + pad)
    if x1 <= x0 or y1 <= y0:
        return im
    if kind in ("occlude", "blur_only"):
        crop = im.crop((x0, y0, x1, y1)).filter(ImageFilter.GaussianBlur(8))
        im.paste(crop, (x0, y0))
    if kind == "occlude":
        ImageDraw.Draw(im).rectangle([x0, y0, x1, y1], fill=(120, 120, 120))
    elif kind == "whiteout":
        ImageDraw.Draw(im).rectangle([x0, y0, x1, y1], fill=(255, 255, 255))
    elif kind != "blur_only":
        raise ValueError(kind)
    return im


def corrupt(tokens, idxs, decoy, kind):
    idxs = set(idxs)
    toks = [t for t, _ in tokens]
    if kind == "decoy":
        out = []
        for i, t in enumerate(toks):
            if i in idxs:
                if i == min(idxs):
                    out.append(decoy)
            else:
                out.append(t)
        return " ".join(out)
    if kind == "delete":
        return " ".join(t for i, t in enumerate(toks) if i not in idxs)
    if kind == "swap":
        j = next((i for i, t in enumerate(toks)
                  if t == decoy and i not in idxs), None)
        answer = " ".join(toks[i] for i in sorted(idxs))
        out = []
        for i, t in enumerate(toks):
            if i in idxs:
                if i == min(idxs):
                    out.append(decoy)
            elif i == j:
                out.append(answer)
            else:
                out.append(t)
        return " ".join(out)
    raise ValueError(kind)


# --------------------------- one item --------------------------------------- #

def run_item(reader, raw, args, seed, ds_index):
    img = raw["image"].convert("RGB")
    question = raw["question"]
    golds = raw["answers"] if isinstance(raw["answers"], list) else [raw["answers"]]
    golds = [g for g in golds if str(g).strip()]
    meta = {"ds_index": int(ds_index),
            "qid": raw.get("questionId") or raw.get("question_id")}
    if not golds:
        return dict(meta, skipped=True, reason="no_gold")

    toks = D.ocr_tokens(img)
    if not toks:
        return dict(meta, skipped=True, reason="ocr_empty")
    idxs, bbox = D.locate_answer(toks, golds)
    if idxs is None:
        return dict(meta, skipped=True, reason="answer_not_locatable")

    rng = random.Random(seed)
    decoy = D.pick_decoy(toks, idxs, golds, rng)
    item = {
        "img": img,
        "img_bad": deface(img, bbox, args.visual_fault),
        "txt": D.transcript(toks),
        "txt_bad": corrupt(toks, idxs, decoy, args.struct_fault),
        "question": question,
        "golds": golds,
    }
    cache = {}

    clean = D.run_under(reader, item, set(), cache, args.read_mode)
    if not D.correct(clean, golds):
        return dict(meta, skipped=True, reason="clean_wrong")

    injected = ({"visual", "struct"} if args.fault_kind == "double"
                else {args.fault_kind.split("_")[1]})
    broken = D.run_under(reader, item, injected, cache, args.read_mode)
    flipped = not D.correct(broken, golds)

    base = dict(meta, skipped=False, flipped=flipped, read_mode=args.read_mode,
                fault_kind=args.fault_kind, injected=sorted(injected),
                decoy=decoy, visual_fault=args.visual_fault,
                struct_fault=args.struct_fault)
    if not flipped:
        base["outcome_masked"] = True
        base["vlm_calls"] = len(cache)
        return base

    minimal, _ = D.minimal_decisive(reader, item, injected, cache, args.read_mode)
    gt = set(minimal)
    singles = [m for m in CHANNELS
               if D.correct(D.run_under(reader, item, set(injected) - {m},
                                        cache, args.read_mode), golds)]
    base["outcome_masked"] = False
    base["n_minimal_decisive"] = len(minimal)
    base["minimal_decisive"] = [sorted(R) for R in minimal]
    base["single_channel_recovers"] = singles
    base["joint_decisive"] = int(len(injected) == 2 and not singles
                                 and frozenset(CHANNELS) in gt)
    base["union_equals_injected"] = int(
        (set().union(*minimal) == set(injected)) if minimal else False)

    for name, pred in (
        ("coupled", D.coupled_attribute(reader, item, injected, cache,
                                        args.read_mode)),
        ("anomaly", D.anomaly_attribute(reader, item, injected, cache)),
        ("indep", D.independent_attribute(reader, item, injected, cache,
                                          args.read_mode)),
    ):
        base[f"{name}_chan_ok"] = int(pred in gt)
        base[f"{name}_injectionset_ok"] = int(pred == frozenset(injected))
        base[f"{name}_pred"] = sorted(pred)
    base["vlm_calls"] = len(cache)
    return base


# --------------------------- main ------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n_items", type=int, default=300)
    ap.add_argument("--fault_kind", default="double",
                    choices=["double", "single_visual", "single_struct"])
    ap.add_argument("--read_mode", default="fused",
                    choices=["fused", "image_primary", "struct_primary"])
    ap.add_argument("--visual_fault", default="occlude",
                    choices=["occlude", "blur_only", "whiteout"])
    ap.add_argument("--struct_fault", default="decoy",
                    choices=["decoy", "delete", "swap"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="docvqa_v3.json")
    args = ap.parse_args()

    from datasets import load_dataset

    print("loading DocVQA validation split ...", flush=True)
    ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation")
    ds = ds.shuffle(seed=args.seed)

    if not args.model.lower().startswith("qwen/"):
        D.ask = ask_generic          # non-Qwen readers need the bare placeholder
    reader = Reader(args.model, device=args.device)
    results, kept = [], 0
    for i in range(len(ds)):
        if kept >= args.n_items:
            break
        try:
            r = run_item(reader, ds[i], args, seed=args.seed * 1000 + i,
                         ds_index=i)
        except Exception as e:
            r = {"ds_index": int(i), "skipped": True,
                 "reason": f"error:{type(e).__name__}: {str(e)[:180]}"}
        results.append(r)
        if not r.get("skipped"):
            kept += 1
        if i % 25 == 0:
            print(f"  [{i}] kept={kept}", flush=True)

    nonskip = [r for r in results if not r.get("skipped")]
    decisive = [r for r in nonskip if r.get("flipped")]
    nd = len(decisive)

    def mean(rows, k):
        v = [r[k] for r in rows if k in r and r[k] is not None]
        return float(np.mean(v)) if v else None

    skip_reasons = {}
    for r in results:
        if r.get("skipped"):
            k = r.get("reason", "?")
            skip_reasons[k] = skip_reasons.get(k, 0) + 1

    summary = {
        "model": args.model, "read_mode": args.read_mode,
        "fault_kind": args.fault_kind,
        "visual_fault": args.visual_fault, "struct_fault": args.struct_fault,
        "n_seen": len(results), "n_nonskipped": len(nonskip),
        "skip_reasons": skip_reasons, "n_decisive": nd,
        "flip_rate": nd / max(1, len(nonskip)),
        "flip_rate_ci95": D.wilson(nd, max(1, len(nonskip))),
        "joint_decisive_rate": mean(decisive, "joint_decisive"),
        "joint_decisive_ci95": D.wilson(
            sum(r.get("joint_decisive", 0) for r in decisive), max(1, nd)),
        "mean_n_minimal_decisive": mean(decisive, "n_minimal_decisive"),
        "union_equals_injected": mean(decisive, "union_equals_injected"),
        "coupled_chan_acc": mean(decisive, "coupled_chan_ok"),
        "anomaly_chan_acc": mean(decisive, "anomaly_chan_ok"),
        "indep_chan_acc": mean(decisive, "indep_chan_ok"),
        "coupled_injectionset_acc": mean(decisive, "coupled_injectionset_ok"),
        "anomaly_injectionset_acc": mean(decisive, "anomaly_injectionset_ok"),
        "indep_injectionset_acc": mean(decisive, "indep_injectionset_ok"),
        "mean_vlm_calls": mean(nonskip, "vlm_calls"),
    }
    print("\n=== DocVQA v3 SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k:30s}: {v}")
    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "summary": summary,
                   "items": results}, f, indent=2, default=str)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
