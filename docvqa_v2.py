#!/usr/bin/env python3
"""
DocVQA two-channel attribution -- real agent, real interventions
================================================================

Replaces `cross_task_docvqa.py`, which could not support a causal claim:

  * it never showed the OCR text to the model, so there was no second channel in
    the agent's input; the "outcome" was a hand-written boolean formula over
    substring checks rather than an agent run;
  * `coupled_attribute` returned a singleton on every reachable path (the joint
    branch was dead code), while ground truth was the *injection* set, so all
    three methods scored 0.0 by construction;
  * the visual fault defaced a random 40% rectangle, which usually misses the
    answer entirely.

This version:

  1. Builds a genuine two-channel agent over a real document page.  VISUAL is the
     page image; STRUCT is the OCR transcript, placed in the prompt next to the
     image.  `--read_mode fused` shows both, so both are actually consulted.
  2. LOCALISES the answer with OCR word boxes, so the visual fault occludes the
     region the answer is actually in.  Items whose answer cannot be located are
     skipped, and the skip rate is reported rather than hidden.
  3. Obtains ground truth by RUNNING restorations: the minimal decisive
     restoration set, computed over {visual}, {struct}, {visual,struct}.  With
     greedy decoding and fixed inputs, re-execution under an intervention is the
     coupled counterfactual exactly.
  4. Reports the REGIME MIX on real documents: how often an injected fault is
     masked, single-decisive, or joint-decisive.  The joint-decisive rate is the
     real-data base rate for the Prop-3 regime, which would otherwise only be asserted
     rather than measures (the FUNSD study found zero failure-causing faults).

Scoring uses ANLS >= 0.5, the DocVQA standard, not substring containment.
"""
import argparse, json, random, re
import numpy as np

CHANNELS = ("visual", "struct")
_OCR = None


# --------------------------- OCR + scoring ---------------------------------- #

def get_ocr(gpu=True):
    global _OCR
    if _OCR is None:
        import easyocr
        _OCR = easyocr.Reader(["en"], gpu=gpu, verbose=False)
    return _OCR


def norm(s):
    return re.sub(r"\s+", " ", str(s).strip().lower())


def levenshtein(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def anls(pred, golds, tau=0.5):
    """DocVQA's Average Normalized Levenshtein Similarity against a gold set."""
    p = norm(pred)
    best = 0.0
    for g in golds:
        g = norm(g)
        if not p and not g:
            s = 1.0
        else:
            d = levenshtein(p, g)
            s = 1.0 - d / max(len(p), len(g), 1)
        best = max(best, s)
    return best if best >= tau else 0.0


def correct(pred, golds, tau=0.5):
    return anls(pred, golds, tau) > 0.0


# --------------------------- channels --------------------------------------- #

def ocr_tokens(img):
    """OCR with boxes: list of (text, (x0,y0,x1,y1))."""
    rdr = get_ocr()
    arr = np.asarray(img.convert("RGB"))
    try:
        raw = rdr.readtext(arr, detail=1)
    except Exception:
        return []
    out = []
    for box, text, _conf in raw:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        out.append((text, (int(min(xs)), int(min(ys)),
                           int(max(xs)), int(max(ys)))))
    return out


def transcript(tokens):
    return " ".join(t for t, _ in tokens)


def locate_answer(tokens, golds):
    """Find the OCR token(s) matching the gold answer. Returns (indices, bbox) or
    (None, None) when the answer is not locatable, in which case the item is
    skipped: a visual fault we cannot aim is not a perception fault we can study.
    """
    gold_norms = [norm(g) for g in golds]
    # 1) single-token exact/near match
    best_i, best_s = None, 0.0
    for i, (text, _) in enumerate(tokens):
        s = anls(text, golds, tau=0.0)
        if s > best_s:
            best_i, best_s = i, s
    if best_i is not None and best_s >= 0.8:
        return [best_i], tokens[best_i][1]
    # 2) contiguous span match for multi-word answers
    for span in (2, 3, 4):
        for i in range(0, max(0, len(tokens) - span + 1)):
            joined = " ".join(t for t, _ in tokens[i:i + span])
            if anls(joined, golds, tau=0.0) >= 0.8:
                xs0 = min(tokens[k][1][0] for k in range(i, i + span))
                ys0 = min(tokens[k][1][1] for k in range(i, i + span))
                xs1 = max(tokens[k][1][2] for k in range(i, i + span))
                ys1 = max(tokens[k][1][3] for k in range(i, i + span))
                return list(range(i, i + span)), (xs0, ys0, xs1, ys1)
    return None, None


def deface_region(img, bbox, pad=6):
    """Visual fault: occlude the answer region so the image no longer carries it.
    Targeted at the located answer, not a random rectangle."""
    from PIL import ImageDraw, ImageFilter
    im = img.copy()
    x0, y0, x1, y1 = bbox
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(im.size[0], x1 + pad), min(im.size[1], y1 + pad)
    if x1 <= x0 or y1 <= y0:
        return im
    crop = im.crop((x0, y0, x1, y1)).filter(ImageFilter.GaussianBlur(8))
    im.paste(crop, (x0, y0))
    d = ImageDraw.Draw(im)
    d.rectangle([x0, y0, x1, y1], fill=(120, 120, 120))
    return im


def corrupt_transcript(tokens, idxs, decoy):
    """Struct fault: replace the answer token(s) in the transcript with a decoy
    drawn from elsewhere on the same page, so the corrupted channel is plausible
    rather than obviously broken."""
    out = []
    idxs = set(idxs)
    for i, (t, _) in enumerate(tokens):
        if i in idxs:
            if i == min(idxs):
                out.append(decoy)
        else:
            out.append(t)
    return " ".join(out)


def pick_decoy(tokens, idxs, golds, rng):
    """A same-page token that is not the answer: a realistic OCR confusion."""
    cands = [t for i, (t, _) in enumerate(tokens)
             if i not in set(idxs) and t.strip()
             and anls(t, golds, tau=0.0) < 0.5]
    numeric = [t for t in cands if any(ch.isdigit() for ch in t)]
    gold_has_digit = any(ch.isdigit() for g in golds for ch in str(g))
    pool = numeric if (gold_has_digit and numeric) else cands
    return rng.choice(pool) if pool else "xxxx"


# --------------------------- agent ------------------------------------------ #

def ask(reader, img, text_channel, question, read_mode, cache, key):
    """One agent answer under a channel configuration. Greedy decoding, so the
    cache is exact and re-execution under an intervention is the counterfactual."""
    import torch
    if cache is not None and key in cache:
        return cache[key]

    use_image = read_mode in ("fused", "image_primary")
    use_struct = read_mode in ("fused", "struct_primary")

    prompt = f"This is a document page. {question}"
    if use_struct:
        prompt += ("\n\nOCR transcript of the page:\n" + text_channel)
    prompt += "\n\nAnswer with ONLY the value, no explanation."

    content = []
    if use_image:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    chat = reader.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    images = [img] if use_image else None
    inputs = reader.processor(text=[chat], images=images, padding=True,
                              return_tensors="pt").to(reader.device)
    with torch.no_grad():
        out = reader.model.generate(**inputs, max_new_tokens=24, do_sample=False)
    gen = out[:, inputs.input_ids.shape[1]:]
    ans = reader.processor.batch_decode(gen, skip_special_tokens=True)[0].strip()
    if cache is not None:
        cache[key] = ans
    return ans


def run_under(reader, item, faulted, cache, read_mode):
    """Run the agent with a given set of channels left faulty."""
    img = item["img_bad"] if "visual" in faulted else item["img"]
    txt = item["txt_bad"] if "struct" in faulted else item["txt"]
    key = (",".join(sorted(faulted)) if faulted else "clean", read_mode)
    return ask(reader, img, txt, item["question"], read_mode, cache, key)


# --------------------------- attribution ------------------------------------ #

def subsets():
    return [frozenset({"visual"}), frozenset({"struct"}),
            frozenset({"visual", "struct"})]


def minimal_decisive(reader, item, injected, cache, read_mode):
    """Minimal decisive restoration sets, by running every restoration."""
    dec = []
    for R in subsets():
        remaining = set(injected) - set(R)
        ans = run_under(reader, item, remaining, cache, read_mode)
        if correct(ans, item["golds"]):
            dec.append(R)
    minimal = [R for R in dec if not any(D < R for D in dec)]
    return minimal, dec


def coupled_attribute(reader, item, injected, cache, read_mode):
    """Ours: single-channel restorations first, joint restoration as the fallback
    when no single one is decisive (the Prop-3 regime)."""
    for m in CHANNELS:
        remaining = set(injected) - {m}
        if correct(run_under(reader, item, remaining, cache, read_mode),
                   item["golds"]):
            return frozenset({m})
    remaining = set(injected) - set(CHANNELS)
    if correct(run_under(reader, item, remaining, cache, read_mode),
               item["golds"]):
        return frozenset(CHANNELS)
    return frozenset()


def anomaly_attribute(reader, item, injected, cache):
    """Baseline A, strengthened relative to v1's 'always guess visual': read each
    channel in isolation and name the one whose isolated answer disagrees with the
    gold-free consensus. No counterfactual verification, and it can only ever name
    a single channel."""
    img_only = run_under(reader, item, injected, cache, "image_primary")
    txt_only = run_under(reader, item, injected, cache, "struct_primary")
    if norm(img_only) == norm(txt_only):
        return frozenset({"visual"})  # no signal; falls back to the prior
    fused = run_under(reader, item, injected, cache, "fused")
    if norm(img_only) != norm(fused):
        return frozenset({"visual"})
    if norm(txt_only) != norm(fused):
        return frozenset({"struct"})
    return frozenset({"visual"})


def independent_attribute(reader, item, injected, cache, read_mode):
    """Baseline B: single restorations only, no joint pass."""
    hits = []
    for m in CHANNELS:
        remaining = set(injected) - {m}
        if correct(run_under(reader, item, remaining, cache, read_mode),
                   item["golds"]):
            hits.append(m)
    if not hits:
        return frozenset()
    return frozenset({hits[-1]})


# --------------------------- one item --------------------------------------- #

def run_item(reader, raw, fault_kind, read_mode, seed):
    img = raw["image"].convert("RGB")
    question = raw["question"]
    golds = raw["answers"] if isinstance(raw["answers"], list) else [raw["answers"]]
    golds = [g for g in golds if str(g).strip()]
    if not golds:
        return {"skipped": True, "reason": "no_gold"}

    toks = ocr_tokens(img)
    if not toks:
        return {"skipped": True, "reason": "ocr_empty"}
    idxs, bbox = locate_answer(toks, golds)
    if idxs is None:
        return {"skipped": True, "reason": "answer_not_locatable"}

    rng = random.Random(seed)
    txt = transcript(toks)
    decoy = pick_decoy(toks, idxs, golds, rng)
    item = {
        "img": img,
        "img_bad": deface_region(img, bbox),
        "txt": txt,
        "txt_bad": corrupt_transcript(toks, idxs, decoy),
        "question": question,
        "golds": golds,
    }
    cache = {}

    clean = run_under(reader, item, set(), cache, read_mode)
    if not correct(clean, golds):
        return {"skipped": True, "reason": "clean_wrong"}

    injected = ({"visual", "struct"} if fault_kind == "double"
                else {fault_kind.split("_")[1]})
    broken = run_under(reader, item, injected, cache, read_mode)
    flipped = not correct(broken, golds)

    base = {
        "skipped": False, "flipped": flipped, "read_mode": read_mode,
        "fault_kind": fault_kind, "injected": sorted(injected),
        "decoy": decoy,
    }
    if not flipped:
        base["outcome_masked"] = True
        base["vlm_calls"] = len(cache)
        return base

    minimal, _ = minimal_decisive(reader, item, injected, cache, read_mode)
    gt = set(minimal)
    single_recovers = [m for m in CHANNELS
                       if correct(run_under(reader, item,
                                            set(injected) - {m}, cache, read_mode),
                                  golds)]
    base["outcome_masked"] = False
    base["n_minimal_decisive"] = len(minimal)
    base["minimal_decisive"] = [sorted(R) for R in minimal]
    base["single_channel_recovers"] = single_recovers
    base["joint_decisive"] = int(len(injected) == 2 and not single_recovers
                                 and frozenset(CHANNELS) in gt)

    cpl = coupled_attribute(reader, item, injected, cache, read_mode)
    ano = anomaly_attribute(reader, item, injected, cache)
    ind = independent_attribute(reader, item, injected, cache, read_mode)
    for name, pred in (("coupled", cpl), ("anomaly", ano), ("indep", ind)):
        base[f"{name}_chan_ok"] = int(pred in gt)
        base[f"{name}_injectionset_ok"] = int(pred == frozenset(injected))
        base[f"{name}_pred"] = sorted(pred)
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
    ap.add_argument("--n_items", type=int, default=200)
    ap.add_argument("--fault_kind", default="double",
                    choices=["single_visual", "single_struct", "double"])
    ap.add_argument("--read_mode", default="fused",
                    choices=["fused", "image_primary", "struct_primary"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="docvqa_v2.json")
    args = ap.parse_args()

    from datasets import load_dataset
    from percept_vlm_pilot import VLMReader

    print("loading DocVQA validation split ...", flush=True)
    ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation")
    ds = ds.shuffle(seed=args.seed)

    reader = VLMReader(args.model)
    results, kept = [], 0
    for i in range(len(ds)):
        if kept >= args.n_items:
            break
        r = run_item(reader, ds[i], args.fault_kind, args.read_mode,
                     seed=args.seed * 1000 + i)
        results.append(r)
        if not r.get("skipped"):
            kept += 1
        if i % 25 == 0:
            print(f"  [{i}] kept={kept}", flush=True)

    nonskip = [r for r in results if not r.get("skipped")]
    decisive = [r for r in nonskip if r.get("flipped")]
    nd = len(decisive)

    def mean(rows, k):
        v = [r[k] for r in rows if k in r]
        return float(np.mean(v)) if v else None

    skip_reasons = {}
    for r in results:
        if r.get("skipped"):
            skip_reasons[r.get("reason", "?")] = \
                skip_reasons.get(r.get("reason", "?"), 0) + 1

    summary = {
        "model": args.model, "read_mode": args.read_mode,
        "fault_kind": args.fault_kind, "seed": args.seed,
        "n_seen": len(results), "n_nonskipped": len(nonskip),
        "skip_reasons": skip_reasons,
        "n_decisive": nd, "n_outcome_masked": len(nonskip) - nd,
        "flip_rate": nd / max(1, len(nonskip)),
        "flip_rate_ci95": wilson(nd, max(1, len(nonskip))),
        "joint_decisive_rate": mean(decisive, "joint_decisive"),
        "joint_decisive_ci95": wilson(
            sum(r.get("joint_decisive", 0) for r in decisive), nd),
        "mean_n_minimal_decisive": mean(decisive, "n_minimal_decisive"),
        "coupled_chan_acc": mean(decisive, "coupled_chan_ok"),
        "coupled_chan_ci95": wilson(
            sum(r.get("coupled_chan_ok", 0) for r in decisive), nd),
        "anomaly_chan_acc": mean(decisive, "anomaly_chan_ok"),
        "anomaly_chan_ci95": wilson(
            sum(r.get("anomaly_chan_ok", 0) for r in decisive), nd),
        "indep_chan_acc": mean(decisive, "indep_chan_ok"),
        "indep_chan_ci95": wilson(
            sum(r.get("indep_chan_ok", 0) for r in decisive), nd),
        "coupled_injectionset_acc": mean(decisive, "coupled_injectionset_ok"),
        "anomaly_injectionset_acc": mean(decisive, "anomaly_injectionset_ok"),
        "indep_injectionset_acc": mean(decisive, "indep_injectionset_ok"),
        "mean_vlm_calls": mean(nonskip, "vlm_calls"),
    }
    print("\n=== DocVQA v2 SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k:30s}: {v}")
    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "summary": summary,
                   "items": results}, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
