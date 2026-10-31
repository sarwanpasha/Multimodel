"""
Mind2Web stage-4: fault-design sensitivity and a genuine three-channel setting.
========================================================================

Two extensions over m2w_stage3.py, both aimed at the same question: is the
measured regime (how often the minimal decisive restoration set is the whole
injected set) a property of the agent, or an artifact of how we chose to break
the channels?

(1) FAULT DESIGN.  Each channel now has several alternative corruptions that
    are equally faithful to a real failure mode.  Re-running the same agent on
    the same steps under a different corruption recipe, with everything else
    fixed, isolates the contribution of the recipe.

      visual: blur_grey   the target's box interior is blurred and greyed,
                          its number kept (the stage-3 fault).
              remove_mark the target's box and number are not drawn at all,
                          so the screenshot carries no mark for the answer.
              swap_marks  the target's number and one distractor's number are
                          exchanged on the image: a mark-assignment error.

      struct: decoy_swap  the target's description is replaced by a copy of a
                          distractor's description (the stage-3 fault).
              stale_swap  the target's and a distractor's descriptions are
                          exchanged, as in a DOM that is one render behind.
              blank_desc  the target's description is replaced by a generic
                          uninformative placeholder: deletion, not deceit.

(2) A THIRD CHANNEL.  With --channels 3 the agent additionally reads an OCR
    transcript of each candidate box.  The OCR is computed once from the CLEAN
    crop and its fault is injected separately, which is the right model: an OCR
    module can misread a region that is rendered correctly, so its error is a
    distinct site, not a deterministic function of the screenshot fault.  With
    three co-present channels the minimal decisive family is drawn from seven
    nonempty subsets rather than three, so it can be non-degenerate, exact
    Shapley values over channels are computable, and the increasing-size search
    is a real search rather than an enumeration of three cases.

Ground truth is unchanged: the minimal decisive restoration family, computed by
actually running every restoration.

Usage
  python m2w_stage4.py --channels 2 --visual_fault remove_mark --out X.json
  python m2w_stage4.py --channels 3 --n_steps 150 --out Y.json
"""
import argparse, itertools, json, random, re
import numpy as np

# --------------------------- model ------------------------------------------ #


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
    """Qwen2-VL or Qwen2.5-VL, chosen by model id."""

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

    def choose(self, img, listing, ocr_text, n_choices,
               use_image, use_struct, use_ocr):
        import torch
        parts = ["You are a web agent."]
        if use_image:
            parts.append("The screenshot shows candidate elements outlined "
                         f"and numbered 0-{n_choices - 1}.")
        if use_struct:
            parts.append("A list of the candidates' page descriptions is "
                         "given below.")
        if use_ocr:
            parts.append("The text read from inside each candidate's box by "
                         "an OCR module is also given below.")
        parts.append("Which element should be acted on next to progress the "
                     "task?")
        prompt = " ".join(parts)
        if use_struct:
            prompt += f"\n\nCandidates:\n{listing}"
        if use_ocr:
            prompt += f"\n\nOCR text per candidate:\n{ocr_text}"
        prompt += "\n\nAnswer with ONLY the number."

        content = ([{"type": "image"}] if use_image else []) + [
            {"type": "text", "text": prompt}]
        chat = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False, add_generation_prompt=True)
        try:
            inputs = self.processor(text=[chat],
                                    images=[img] if use_image else None,
                                    padding=True,
                                    return_tensors="pt").to(self.device)
        except (ValueError, TypeError):
            inputs = self.processor(text=[chat],
                                    images=[[img]] if use_image else None,
                                    padding=True,
                                    return_tensors="pt").to(self.device)
        inputs = _cast_inputs(inputs, self.model)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=8, do_sample=False)
        gen = out[:, inputs.input_ids.shape[1]:]
        ans = self.processor.batch_decode(gen, skip_special_tokens=True)[0]
        m = re.search(r"\d+", ans)
        if m is None:
            return None
        i = int(m.group())
        return i if 0 <= i < n_choices else None


# --------------------------- candidates + geometry -------------------------- #

def parse_candidate(cstr):
    """-> (description, backend_node_id, bbox or None)."""
    try:
        c = json.loads(cstr) if isinstance(cstr, str) else cstr
        attrs = c.get("attributes")
        if isinstance(attrs, str):
            attrs = json.loads(attrs)
        attrs = attrs if isinstance(attrs, dict) else {}
        tag = c.get("tag", "?")
        text = (attrs.get("text") or attrs.get("aria_label") or
                attrs.get("value") or attrs.get("placeholder") or "")
        role = attrs.get("role", "")
        desc = re.sub(r"\s+", " ", f"<{tag}> {role} {text}".strip())[:80]
        bbox = None
        rect = attrs.get("bounding_box_rect")
        if rect:
            try:
                x, y, w, h = [float(v) for v in str(rect).split(",")[:4]]
                if w > 0 and h > 0:
                    bbox = (x, y, x + w, y + h)
            except Exception:
                bbox = None
        return desc, c.get("backend_node_id"), bbox
    except Exception:
        return None, None, None


def build_candidates(step, k_negatives, rng):
    pos = step.get("pos_candidates") or []
    neg = step.get("neg_candidates") or []
    if not pos:
        return None, None
    gd, gid, gb = parse_candidate(pos[0])
    if gd is None or gb is None:
        return None, None
    chosen = [(gd, gid, gb)]
    for n in (rng.sample(neg, min(k_negatives * 3, len(neg))) if neg else []):
        d, nid, b = parse_candidate(n)
        if d is not None and b is not None and nid != gid:
            chosen.append((d, nid, b))
        if len(chosen) >= k_negatives + 1:
            break
    if len(chosen) < 2:
        return None, None
    rng.shuffle(chosen)
    gt = next(i for i, (_, nid, _) in enumerate(chosen) if nid == gid)
    return chosen, gt


def crop_window(img, boxes, pad=60, max_side=1280):
    W, H = img.size
    x0 = max(0, min(b[0] for b in boxes) - pad)
    y0 = max(0, min(b[1] for b in boxes) - pad)
    x1 = min(W, max(b[2] for b in boxes) + pad)
    y1 = min(H, max(b[3] for b in boxes) + pad)
    if x1 - x0 < 40 or y1 - y0 < 40:
        return None, None, None
    crop = img.crop((int(x0), int(y0), int(x1), int(y1)))
    cw, ch = crop.size
    scale = min(1.0, max_side / max(cw, ch))
    if scale < 1.0:
        crop = crop.resize((max(1, int(cw * scale)), max(1, int(ch * scale))))
    return crop, scale, (x0, y0)


def to_crop_coords(bbox, scale, offset):
    ox, oy = offset
    return tuple((v - o) * scale for v, o in zip(bbox, (ox, oy, ox, oy)))


# --------------------------- visual channel --------------------------------- #

def mark_candidates(img, boxes_c, numbers=None, blur_idx=None, drop_idx=None):
    """Set-of-mark rendering.

    numbers[i] is the label drawn on box i (identity by default).  blur_idx is
    blurred and greyed inside while keeping its number.  drop_idx gets neither
    outline nor number.
    """
    from PIL import ImageDraw, ImageFilter
    im = img.convert("RGB").copy()
    n = len(boxes_c)
    numbers = list(range(n)) if numbers is None else list(numbers)
    if blur_idx is not None:
        x0, y0, x1, y1 = [int(v) for v in boxes_c[blur_idx]]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(im.size[0], x1), min(im.size[1], y1)
        if x1 > x0 and y1 > y0:
            patch = im.crop((x0, y0, x1, y1)).filter(ImageFilter.GaussianBlur(9))
            im.paste(patch, (x0, y0))
            ov = ImageDraw.Draw(im, "RGBA")
            ov.rectangle([x0, y0, x1, y1], fill=(128, 128, 128, 190))
    d = ImageDraw.Draw(im)
    for i, b in enumerate(boxes_c):
        if drop_idx is not None and i == drop_idx:
            continue
        x0, y0, x1, y1 = [int(v) for v in b]
        d.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=2)
        ty = max(0, y0 - 14)
        lab = str(numbers[i])
        d.rectangle([x0, ty, x0 + 8 + 10 * len(lab), ty + 14], fill=(255, 0, 0))
        d.text((x0 + 4, ty + 2), lab, fill=(255, 255, 255))
    return im


def make_visual(crop, boxes_c, gt, kind, rng):
    """Return (clean image, faulted image, note)."""
    ok = mark_candidates(crop, boxes_c)
    n = len(boxes_c)
    others = [i for i in range(n) if i != gt]
    if kind == "blur_grey":
        return ok, mark_candidates(crop, boxes_c, blur_idx=gt), "blur_grey"
    if kind == "remove_mark":
        return ok, mark_candidates(crop, boxes_c, drop_idx=gt), "remove_mark"
    if kind == "swap_marks":
        j = rng.choice(others)
        nums = list(range(n))
        nums[gt], nums[j] = nums[j], nums[gt]
        return ok, mark_candidates(crop, boxes_c, numbers=nums), f"swap_marks:{j}"
    raise ValueError(kind)


# --------------------------- text channels ---------------------------------- #

def render_listing(descs, boxes=None):
    """One line per candidate.  Given box coordinates the line also carries the element's
    box in crop pixels, so the structured channel can locate the target without
    resolving a numbered mark in the image."""
    if boxes is None:
        return "\n".join(f"{i}: {d}" for i, d in enumerate(descs))
    out = []
    for i, (d, b) in enumerate(zip(descs, boxes)):
        x0, y0, x1, y1 = [int(v) for v in b]
        out.append(f"{i}: {d} [box {x0},{y0},{x1},{y1}]")
    return "\n".join(out)


def make_struct(descs, gt, kind, rng, boxes=None):
    """Return (clean listing, faulted listing, note)."""
    ok = render_listing(descs, boxes)
    n = len(descs)
    others = [i for i in range(n) if i != gt]
    bad = list(descs)
    if kind == "decoy_swap":
        j = rng.choice(others)
        bad[gt] = descs[j]
        note = f"decoy_swap:{j}"
    elif kind == "stale_swap":
        j = rng.choice(others)
        bad[gt], bad[j] = descs[j], descs[gt]
        note = f"stale_swap:{j}"
    elif kind == "blank_desc":
        bad[gt] = "<div>"
        note = "blank_desc"
    else:
        raise ValueError(kind)
    return ok, render_listing(bad, boxes), note


def make_ocr(ocr_lines, gt, kind, rng):
    ok = render_listing(ocr_lines)
    n = len(ocr_lines)
    others = [i for i in range(n) if i != gt]
    bad = list(ocr_lines)
    if kind == "decoy_swap":
        j = rng.choice(others)
        bad[gt] = ocr_lines[j]
        note = f"decoy_swap:{j}"
    elif kind == "blank_desc":
        bad[gt] = "(no text)"
        note = "blank_desc"
    else:
        raise ValueError(kind)
    return ok, render_listing(bad), note


_OCR = None
_OCR_GPU = False        # CPU by default: easyocr on GPU is not bit-reproducible
                        # across card types, and the OCR text decides which
                        # steps are usable, so a GPU reader makes the kept set
                        # depend on which node the job landed on.


def get_ocr():
    global _OCR
    if _OCR is None:
        import easyocr
        _OCR = easyocr.Reader(["en"], gpu=_OCR_GPU, verbose=False)
    return _OCR


def ocr_per_box(crop, boxes_c, max_chars=60):
    """Text read inside each candidate box, from the CLEAN crop."""
    rdr = get_ocr()
    arr = np.asarray(crop.convert("RGB"))
    try:
        raw = rdr.readtext(arr, detail=1)
    except Exception:
        return None
    toks = []
    for box, text, _c in raw:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        toks.append((text, ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0)))
    lines = []
    for (x0, y0, x1, y1) in boxes_c:
        inside = [t for t, (cx, cy) in toks if x0 <= cx <= x1 and y0 <= cy <= y1]
        s = re.sub(r"\s+", " ", " ".join(inside)).strip()[:max_chars]
        lines.append(s if s else "(no text)")
    return lines


# --------------------------- agent under interventions ---------------------- #

def run_under(reader, item, faulted, cache, read_mode):
    key = (",".join(sorted(faulted)) if faulted else "clean", read_mode)
    if key in cache:
        return cache[key]
    img = item["img_bad"] if "visual" in faulted else item["img_ok"]
    listing = item["listing_bad"] if "struct" in faulted else item["listing_ok"]
    ocr = item["ocr_bad"] if "ocr" in faulted else item["ocr_ok"]
    use_image = read_mode in ("fused", "image_primary")
    use_struct = (read_mode in ("fused", "struct_primary")) and item["has_struct"]
    use_ocr = (read_mode in ("fused", "ocr_primary")) and item["has_ocr"]
    pred = reader.choose(img, listing, ocr, item["n"],
                         use_image, use_struct, use_ocr)
    cache[key] = pred
    return pred


def nonempty_subsets(channels):
    """All nonempty subsets, smallest first, deterministic order."""
    out = []
    for r in range(1, len(channels) + 1):
        for combo in itertools.combinations(channels, r):
            out.append(frozenset(combo))
    return out


def minimal_decisive(reader, item, injected, cache, read_mode):
    dec = []
    for R in nonempty_subsets(sorted(injected)):
        remaining = set(injected) - set(R)
        if run_under(reader, item, remaining, cache, read_mode) == item["gt"]:
            dec.append(R)
    return [R for R in dec if not any(D < R for D in dec)], dec


def coupled_attribute(reader, item, injected, cache, read_mode, channels):
    """Increasing-size search over the lattice; first decisive set wins."""
    for R in nonempty_subsets(sorted(injected)):
        if run_under(reader, item, set(injected) - set(R), cache,
                     read_mode) == item["gt"]:
            return R
    return frozenset()


def independent_attribute(reader, item, injected, cache, read_mode, channels):
    """Singleton restorations only, no joint pass (the replay/ablation analog)."""
    hits = [m for m in channels
            if run_under(reader, item, set(injected) - {m}, cache,
                         read_mode) == item["gt"]]
    return frozenset({hits[-1]}) if hits else frozenset()


def anomaly_attribute(reader, item, injected, cache, channels):
    """Cross-channel disagreement, no counterfactual verification."""
    modes = {"visual": "image_primary", "struct": "struct_primary",
             "ocr": "ocr_primary"}
    fused = run_under(reader, item, injected, cache, "fused")
    iso = {c: run_under(reader, item, injected, cache, modes[c])
           for c in channels}
    odd = [c for c in channels if iso[c] != fused]
    if len(odd) == 1:
        return frozenset(odd)
    vals = list(iso.values())
    for c in channels:
        if vals.count(iso[c]) == 1:
            return frozenset({c})
    return frozenset({"visual"})


def shapley(reader, item, injected, cache, read_mode, channels):
    """Exact Shapley value of each channel in the 0/1 game v(S) = 1 iff
    restoring S recovers the outcome.  Only meaningful for >= 3 channels."""
    chans = sorted(injected)
    k = len(chans)
    if k == 0:
        return {}
    v = {}
    for r in range(0, k + 1):
        for combo in itertools.combinations(chans, r):
            S = frozenset(combo)
            v[S] = int(run_under(reader, item, set(injected) - set(S), cache,
                                 read_mode) == item["gt"])
    from math import factorial
    phi = {}
    for c in chans:
        tot = 0.0
        rest = [x for x in chans if x != c]
        for r in range(0, k):
            for combo in itertools.combinations(rest, r):
                S = frozenset(combo)
                w = factorial(r) * factorial(k - r - 1) / factorial(k)
                tot += w * (v[S | {c}] - v[S])
        phi[c] = tot
    return phi


# --------------------------- one step --------------------------------------- #

def build_item(step, args, rng):
    channels = (("visual", "struct", "ocr") if args.channels == 3
                else ("visual", "struct"))
    choices, gt = build_candidates(step, args.k_negatives, rng)
    if choices is None:
        return None, "no_candidates"
    full = step.get("screenshot")
    if full is None:
        return None, "no_screenshot"
    boxes = [b for (_, _, b) in choices]
    crop, scale, off = crop_window(full, boxes, max_side=args.max_side)
    if crop is None:
        return None, "degenerate_crop"
    boxes_c = [to_crop_coords(b, scale, off) for b in boxes]
    descs = [d for (d, _, _) in choices]

    img_ok, img_bad, vnote = make_visual(crop, boxes_c, gt,
                                         args.visual_fault, rng)
    l_ok, l_bad, snote = make_struct(
        descs, gt, args.struct_fault, rng,
        boxes=boxes_c if args.struct_style == "with_coords" else None)

    if "ocr" in channels:
        lines = ocr_per_box(crop, boxes_c)
        if lines is None:
            return None, "ocr_failed"
        if lines[gt] == "(no text)":
            return None, "ocr_empty_target"
        o_ok, o_bad, onote = make_ocr(lines, gt, args.ocr_fault, rng)
    else:
        o_ok, o_bad, onote = "", "", None

    item = {
        "img_ok": img_ok, "img_bad": img_bad,
        "listing_ok": l_ok, "listing_bad": l_bad,
        "ocr_ok": o_ok, "ocr_bad": o_bad,
        "has_struct": True, "has_ocr": "ocr" in channels,
        "gt": gt, "n": len(choices),
        "notes": {"visual": vnote, "struct": snote, "ocr": onote},
    }
    return item, None


def run_step(reader, step, args, rng):
    channels = (("visual", "struct", "ocr") if args.channels == 3
                else ("visual", "struct"))
    item, why = build_item(step, args, rng)
    if item is None:
        return {"skipped": True, "reason": why}
    cache = {}

    if args.gate_only:
        g = {"skipped": False, "gate": True,
             "fused_ok": int(run_under(reader, item, set(), cache, "fused")
                             == item["gt"]),
             "image_ok": int(run_under(reader, item, set(), cache,
                                       "image_primary") == item["gt"]),
             "struct_ok": int(run_under(reader, item, set(), cache,
                                        "struct_primary") == item["gt"])}
        if "ocr" in channels:
            g["ocr_ok"] = int(run_under(reader, item, set(), cache,
                                        "ocr_primary") == item["gt"])
        return g

    if run_under(reader, item, set(), cache, args.read_mode) != item["gt"]:
        return {"skipped": True, "reason": "clean_wrong"}

    if args.fault_kind == "all":
        injected = set(channels)
    else:
        injected = {args.fault_kind.split("_", 1)[1]}
    flipped = run_under(reader, item, injected, cache,
                        args.read_mode) != item["gt"]
    base = {"skipped": False, "gate": False, "flipped": flipped,
            "injected": sorted(injected), "n_choices": item["n"],
            "n_channels": len(channels), "notes": item["notes"],
            "struct_style": args.struct_style,
            "visual_fault": args.visual_fault,
            "struct_fault": args.struct_fault,
            "ocr_fault": args.ocr_fault if "ocr" in channels else None}
    if not flipped:
        base["outcome_masked"] = True
        base["vlm_calls"] = len(cache)
        return base

    minimal, decisive = minimal_decisive(reader, item, injected, cache,
                                         args.read_mode)
    gt_sets = set(minimal)
    dset = set(decisive)
    base["decisive_sets"] = [sorted(R) for R in decisive]
    base["monotone_ok"] = int(all(J in dset
                                  for I in dset
                                  for J in nonempty_subsets(sorted(injected))
                                  if I < J))
    singles = [m for m in sorted(injected)
               if run_under(reader, item, set(injected) - {m}, cache,
                            args.read_mode) == item["gt"]]
    base["outcome_masked"] = False
    base["n_minimal_decisive"] = len(minimal)
    base["minimal_decisive"] = [sorted(R) for R in minimal]
    base["min_decisive_size"] = min(len(R) for R in minimal) if minimal else None
    base["single_channel_recovers"] = singles
    base["joint_decisive"] = int(frozenset(injected) in gt_sets
                                 and len(minimal) == 1
                                 and len(injected) > 1)
    base["union_equals_injected"] = int(
        set().union(*minimal) == set(injected) if minimal else False)

    calls_before = len(cache)
    cpl = coupled_attribute(reader, item, injected, cache, args.read_mode,
                            channels)
    base["coupled_calls"] = len(cache) - calls_before
    ind = independent_attribute(reader, item, injected, cache, args.read_mode,
                                channels)
    ano = anomaly_attribute(reader, item, injected, cache, channels)
    for name, pred in (("coupled", cpl), ("anomaly", ano), ("indep", ind)):
        base[f"{name}_chan_ok"] = int(pred in gt_sets)
        base[f"{name}_injectionset_ok"] = int(pred == frozenset(injected))
        base[f"{name}_pred"] = sorted(pred)

    if len(injected) >= 3:
        phi = shapley(reader, item, injected, cache, args.read_mode, channels)
        base["shapley"] = {k: round(v, 4) for k, v in phi.items()}
        top = max(phi, key=lambda k: phi[k]) if phi else None
        base["shapley_top1"] = top
        base["shapley_top1_in_minimal"] = int(
            frozenset({top}) in gt_sets) if top else 0
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
    ap.add_argument("--split", default="test_task")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n_steps", type=int, default=200)
    ap.add_argument("--k_negatives", type=int, default=9)
    ap.add_argument("--max_side", type=int, default=1280)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--channels", type=int, default=2, choices=[2, 3])
    ap.add_argument("--read_mode", default="fused",
                    choices=["fused", "image_primary", "struct_primary",
                             "ocr_primary"])
    ap.add_argument("--fault_kind", default="all",
                    choices=["all", "single_visual", "single_struct",
                             "single_ocr"])
    ap.add_argument("--visual_fault", default="blur_grey",
                    choices=["blur_grey", "remove_mark", "swap_marks"])
    ap.add_argument("--struct_fault", default="decoy_swap",
                    choices=["decoy_swap", "stale_swap", "blank_desc"])
    ap.add_argument("--ocr_fault", default="decoy_swap",
                    choices=["decoy_swap", "blank_desc"])
    ap.add_argument("--struct_style", default="plain",
                    choices=["plain", "with_coords"])
    ap.add_argument("--ocr_device", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--gate_only", action="store_true")
    ap.add_argument("--out", default="m2w_stage4.json")
    args = ap.parse_args()

    global _OCR_GPU
    _OCR_GPU = (args.ocr_device == "gpu")

    from datasets import load_dataset
    ds = load_dataset("osunlp/Multimodal-Mind2Web", split=args.split)
    rng = random.Random(args.seed)
    reader = Reader(args.model, device=args.device)

    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    results, kept = [], 0
    for i in idxs:
        if kept >= args.n_steps:
            break
        try:
            r = run_step(reader, ds[i], args, rng)
        except Exception as e:
            r = {"skipped": True, "reason": f"error:{type(e).__name__}: {str(e)[:180]}"}
        r["ds_index"] = int(i)
        results.append(r)
        if not r.get("skipped"):
            kept += 1
        if kept and kept % 25 == 0 and not r.get("skipped"):
            print(f"  kept={kept}", flush=True)

    nonskip = [r for r in results if not r.get("skipped")]
    skip_reasons = {}
    for r in results:
        if r.get("skipped"):
            k = r.get("reason", "?")
            skip_reasons[k] = skip_reasons.get(k, 0) + 1

    def mean(rows, k):
        v = [r[k] for r in rows if k in r]
        return float(np.mean(v)) if v else None

    summary = {"n_kept": len(nonskip), "skip_reasons": skip_reasons,
               "chance": 1.0 / (args.k_negatives + 1)}
    if args.gate_only:
        for k in ("fused_ok", "image_ok", "struct_ok", "ocr_ok"):
            summary[k] = mean(nonskip, k)
    else:
        dec = [r for r in nonskip if r.get("flipped")]
        summary["flip_rate"] = mean(nonskip, "flipped")
        summary["n_decisive"] = len(dec)
        summary["joint_decisive_rate"] = mean(dec, "joint_decisive")
        summary["joint_decisive_ci"] = wilson(
            sum(r.get("joint_decisive", 0) for r in dec), len(dec))
        summary["mean_n_minimal"] = mean(dec, "n_minimal_decisive")
        summary["mean_min_size"] = mean(dec, "min_decisive_size")
        summary["union_equals_injected"] = mean(dec, "union_equals_injected")
        summary["mean_coupled_calls"] = mean(dec, "coupled_calls")
        for m in ("coupled", "anomaly", "indep"):
            summary[f"{m}_chan_acc"] = mean(dec, f"{m}_chan_ok")
            summary[f"{m}_injectionset_acc"] = mean(dec, f"{m}_injectionset_ok")
        if args.channels == 3 and args.fault_kind == "all":
            summary["shapley_top1_in_minimal"] = mean(dec,
                                                      "shapley_top1_in_minimal")

    out = {"args": vars(args), "summary": summary, "steps": results}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
