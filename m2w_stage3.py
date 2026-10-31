#!/usr/bin/env python3
"""
Multimodal-Mind2Web Stage 3: two-channel fault injection and attribution
========================================================================

Stage 2 (`m2w_stage2_baseline.py`) gated at fused accuracy 0.205 (2B) / 0.385
(7B) against a 0.10 chance baseline, i.e. below its own 0.40 threshold, and
concluded the VLM was too weak for an injection study.  Two things in that setup
were doing most of the damage, and both are fixable without a bigger model:

  (1) NO CROPPING.  Mind2Web screenshots are full-page renders, often
      1280x5000+.  Downscaling the whole page to 1024 on the long side makes
      element text physically unreadable, so the "visual channel" carried almost
      no usable signal.  Image-only accuracy of 0.100 at 7B is exactly chance,
      which is the signature of an unreadable image rather than a weak model.
      We instead crop to the region spanned by the candidate elements, using the
      `bounding_box_rect` attribute, and only then downscale.

  (2) NO VISUAL GROUNDING OF THE CHOICES.  The candidate list was numbered in
      text, but nothing in the image said which region each number referred to,
      so the model could not use the screenshot to discriminate candidates even
      in principle.  We draw a numbered box around each candidate (set-of-mark
      style, as in SeeAct), which is what makes the visual channel a real channel.

With both channels genuinely present, faults become injectable:

  VISUAL fault: blur and grey out the target element's own box, leaving its
                number in place, so the image no longer supports the right choice.
  STRUCT fault: replace the target's textual description with a decoy drawn from
                a negative candidate, so the text no longer identifies it.

Ground truth is the minimal decisive restoration set, computed by running the
restorations, exactly as in the chart and DocVQA studies.

`--gate_only` measures clean accuracy for fused / image-only / struct-only and
stops, so the gate can be re-checked cheaply before spending time on injection.
"""
import argparse, json, random, re
import numpy as np

CHANNELS = ("visual", "struct")


# --------------------------- model ------------------------------------------ #

class Reader:
    """Qwen2-VL or Qwen2.5-VL, chosen by model id."""

    def __init__(self, model_id, device="cuda"):
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
        cls = None
        if "Qwen2.5-VL" in model_id or "Qwen2_5" in model_id:
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration as cls
            except ImportError:
                cls = None
        if cls is None:
            try:
                from transformers import Qwen2VLForConditionalGeneration as cls
            except ImportError:
                from transformers import AutoModelForImageTextToText as cls
        self.model = cls.from_pretrained(model_id, torch_dtype="auto",
                                         device_map=device)
        self.device = device

    def choose(self, img, listing, n_choices, use_image, use_struct):
        import torch
        if use_image and use_struct:
            prompt = ("You are a web agent. The screenshot shows candidate "
                      "elements outlined and numbered. Using BOTH the screenshot "
                      "and the list below, which element should be acted on next "
                      f"to progress the task?\n\nCandidates:\n{listing}\n\n"
                      "Answer with ONLY the number.")
        elif use_image:
            prompt = ("You are a web agent. The screenshot shows candidate "
                      f"elements outlined and numbered 0-{n_choices - 1}. Which "
                      "element should be acted on next to progress the task?\n\n"
                      "Answer with ONLY the number.")
        else:
            prompt = ("You are a web agent. Given the list of candidate elements "
                      "below, which element should be acted on next to progress "
                      f"the task?\n\nCandidates:\n{listing}\n\n"
                      "Answer with ONLY the number.")
        content = []
        if use_image:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt})
        chat = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[chat], images=[img] if use_image else None,
                                padding=True, return_tensors="pt").to(self.device)
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
        return None, None          # need the target's box to aim a visual fault
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
    """Crop to the region spanned by the candidate boxes, then downscale. Returns
    (cropped image, scale, offset) so boxes can be mapped into crop coordinates."""
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
    return tuple((v - o) * scale
                 for v, o in zip(bbox, (ox, oy, ox, oy)))


def mark_candidates(img, boxes_c, faulted_idx=None):
    """Set-of-mark: outline and number each candidate. If `faulted_idx` is given,
    that element's interior is blurred and greyed (the visual fault) while its
    number is kept, so the choice remains expressible but visually unsupported."""
    from PIL import ImageDraw, ImageFilter
    im = img.convert("RGB").copy()
    if faulted_idx is not None:
        x0, y0, x1, y1 = [int(v) for v in boxes_c[faulted_idx]]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(im.size[0], x1), min(im.size[1], y1)
        if x1 > x0 and y1 > y0:
            patch = im.crop((x0, y0, x1, y1)).filter(ImageFilter.GaussianBlur(9))
            im.paste(patch, (x0, y0))
            ov = ImageDraw.Draw(im, "RGBA")
            ov.rectangle([x0, y0, x1, y1], fill=(128, 128, 128, 190))
    d = ImageDraw.Draw(im)
    for i, b in enumerate(boxes_c):
        x0, y0, x1, y1 = [int(v) for v in b]
        d.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=2)
        ty = max(0, y0 - 14)
        d.rectangle([x0, ty, x0 + 18, ty + 14], fill=(255, 0, 0))
        d.text((x0 + 4, ty + 2), str(i), fill=(255, 255, 255))
    return im


def listing_text(choices, faulted_idx=None, decoy=None):
    lines = []
    for i, (desc, _, _) in enumerate(choices):
        d = decoy if (faulted_idx is not None and i == faulted_idx) else desc
        lines.append(f"{i}: {d}")
    return "\n".join(lines)


# --------------------------- agent under interventions ---------------------- #

def run_under(reader, item, faulted, cache, read_mode):
    key = (",".join(sorted(faulted)) if faulted else "clean", read_mode)
    if key in cache:
        return cache[key]
    img = item["img_bad"] if "visual" in faulted else item["img_ok"]
    listing = (item["listing_bad"] if "struct" in faulted
               else item["listing_ok"])
    use_image = read_mode in ("fused", "image_primary")
    use_struct = read_mode in ("fused", "struct_primary")
    pred = reader.choose(img, listing, item["n"], use_image, use_struct)
    cache[key] = pred
    return pred


def subsets():
    return [frozenset({"visual"}), frozenset({"struct"}),
            frozenset({"visual", "struct"})]


def minimal_decisive(reader, item, injected, cache, read_mode):
    dec = []
    for R in subsets():
        remaining = set(injected) - set(R)
        if run_under(reader, item, remaining, cache, read_mode) == item["gt"]:
            dec.append(R)
    return [R for R in dec if not any(D < R for D in dec)], dec


def coupled_attribute(reader, item, injected, cache, read_mode):
    for m in CHANNELS:
        if run_under(reader, item, set(injected) - {m}, cache,
                     read_mode) == item["gt"]:
            return frozenset({m})
    if run_under(reader, item, set(injected) - set(CHANNELS), cache,
                 read_mode) == item["gt"]:
        return frozenset(CHANNELS)
    return frozenset()


def anomaly_attribute(reader, item, injected, cache):
    """Cross-channel disagreement, no counterfactual verification, single channel
    only. Names the isolated channel that disagrees with the fused choice."""
    img_only = run_under(reader, item, injected, cache, "image_primary")
    txt_only = run_under(reader, item, injected, cache, "struct_primary")
    fused = run_under(reader, item, injected, cache, "fused")
    if img_only == txt_only:
        return frozenset({"visual"})
    if img_only != fused:
        return frozenset({"visual"})
    if txt_only != fused:
        return frozenset({"struct"})
    return frozenset({"visual"})


def independent_attribute(reader, item, injected, cache, read_mode):
    hits = [m for m in CHANNELS
            if run_under(reader, item, set(injected) - {m}, cache,
                         read_mode) == item["gt"]]
    return frozenset({hits[-1]}) if hits else frozenset()


# --------------------------- one step --------------------------------------- #

def build_item(step, k_negatives, max_side, rng):
    choices, gt = build_candidates(step, k_negatives, rng)
    if choices is None:
        return None, "no_candidates"
    try:
        full = step["screenshot"]
    except Exception:
        return None, "no_screenshot"
    if full is None:
        return None, "no_screenshot"
    boxes = [b for (_, _, b) in choices]
    crop, scale, off = crop_window(full, boxes, max_side=max_side)
    if crop is None:
        return None, "degenerate_crop"
    boxes_c = [to_crop_coords(b, scale, off) for b in boxes]
    others = [d for i, (d, _, _) in enumerate(choices) if i != gt]
    decoy = rng.choice(others) if others else "<div>"
    item = {
        "img_ok": mark_candidates(crop, boxes_c, faulted_idx=None),
        "img_bad": mark_candidates(crop, boxes_c, faulted_idx=gt),
        "listing_ok": listing_text(choices),
        "listing_bad": listing_text(choices, faulted_idx=gt, decoy=decoy),
        "gt": gt, "n": len(choices), "decoy": decoy,
    }
    return item, None


def run_step(reader, step, args, rng):
    item, why = build_item(step, args.k_negatives, args.max_side, rng)
    if item is None:
        return {"skipped": True, "reason": why}
    cache = {}

    if args.gate_only:
        return {
            "skipped": False, "gate": True,
            "fused_ok": int(run_under(reader, item, set(), cache, "fused") == item["gt"]),
            "image_ok": int(run_under(reader, item, set(), cache, "image_primary") == item["gt"]),
            "struct_ok": int(run_under(reader, item, set(), cache, "struct_primary") == item["gt"]),
        }

    if run_under(reader, item, set(), cache, args.read_mode) != item["gt"]:
        return {"skipped": True, "reason": "clean_wrong"}

    injected = ({"visual", "struct"} if args.fault_kind == "double"
                else {args.fault_kind.split("_")[1]})
    flipped = run_under(reader, item, injected, cache,
                        args.read_mode) != item["gt"]
    base = {"skipped": False, "gate": False, "flipped": flipped,
            "injected": sorted(injected), "n_choices": item["n"]}
    if not flipped:
        base["outcome_masked"] = True
        base["vlm_calls"] = len(cache)
        return base

    minimal, _ = minimal_decisive(reader, item, injected, cache, args.read_mode)
    gt_sets = set(minimal)
    singles = [m for m in CHANNELS
               if run_under(reader, item, set(injected) - {m}, cache,
                            args.read_mode) == item["gt"]]
    base["outcome_masked"] = False
    base["n_minimal_decisive"] = len(minimal)
    base["minimal_decisive"] = [sorted(R) for R in minimal]
    base["single_channel_recovers"] = singles
    base["joint_decisive"] = int(len(injected) == 2 and not singles
                                 and frozenset(CHANNELS) in gt_sets)
    for name, pred in (
        ("coupled", coupled_attribute(reader, item, injected, cache, args.read_mode)),
        ("anomaly", anomaly_attribute(reader, item, injected, cache)),
        ("indep", independent_attribute(reader, item, injected, cache, args.read_mode)),
    ):
        base[f"{name}_chan_ok"] = int(pred in gt_sets)
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
    ap.add_argument("--split", default="test_task")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--n_steps", type=int, default=200)
    ap.add_argument("--k_negatives", type=int, default=9)
    ap.add_argument("--max_side", type=int, default=1280)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--read_mode", default="fused",
                    choices=["fused", "image_primary", "struct_primary"])
    ap.add_argument("--fault_kind", default="double",
                    choices=["single_visual", "single_struct", "double"])
    ap.add_argument("--gate_only", action="store_true")
    ap.add_argument("--out", default="m2w_stage3.json")
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("osunlp/Multimodal-Mind2Web", split=args.split)
    rng = random.Random(args.seed)
    reader = Reader(args.model)

    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    results, kept = [], 0
    for i in idxs:
        if kept >= args.n_steps:
            break
        try:
            r = run_step(reader, ds[i], args, rng)
        except Exception as e:
            r = {"skipped": True, "reason": f"error:{type(e).__name__}"}
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

    chance = 1.0 / (args.k_negatives + 1)
    if args.gate_only:
        n = len(nonskip)
        summary = {
            "model": args.model, "split": args.split, "mode": "gate",
            "n_steps": n, "chance": chance, "skip_reasons": skip_reasons,
            "fused_acc": mean(nonskip, "fused_ok"),
            "fused_ci95": wilson(sum(r["fused_ok"] for r in nonskip), n),
            "image_only_acc": mean(nonskip, "image_ok"),
            "image_ci95": wilson(sum(r["image_ok"] for r in nonskip), n),
            "struct_only_acc": mean(nonskip, "struct_ok"),
            "struct_ci95": wilson(sum(r["struct_ok"] for r in nonskip), n),
        }
    else:
        decisive = [r for r in nonskip if r.get("flipped")]
        nd = len(decisive)
        summary = {
            "model": args.model, "split": args.split, "mode": "attribution",
            "read_mode": args.read_mode, "fault_kind": args.fault_kind,
            "chance": chance, "skip_reasons": skip_reasons,
            "n_nonskipped": len(nonskip), "n_decisive": nd,
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

    print("\n=== M2W STAGE 3 SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k:28s}: {v}")
    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "summary": summary,
                   "steps": results}, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
