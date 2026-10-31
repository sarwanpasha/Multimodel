#!/usr/bin/env python3
"""
Run an existing condition script (m2w_stage4, m2w_organic, docvqa_v3,
docvqa_organic) with three optional substitutions, leaving the condition code
itself untouched:

  --api            the reader is DeepSeek's hosted model (OpenAI-compatible
                   chat completions, image input via base64 data URLs),
                   prompted with exactly the text the local readers receive;
  --data cord      the DocVQA stream is replaced by CORD-v2 receipts, asking
                   for the total amount, scored by exact amount match;
  --shard/--nshards  the dataset is restricted to every K-th example so K
                   processes can run side by side; ds_index in the output is
                   shard-local and merge_api.py maps it back.

Usage:
  python harness.py --target docvqa_v3 [--api] [--data cord] \
      [--shard k --nshards K] -- <arguments of the target script>
"""
import argparse, base64, io, json, os, random, re, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--target", required=True,
                choices=["m2w_stage4", "m2w_organic", "docvqa_v3",
                         "docvqa_organic"])
ap.add_argument("--api", action="store_true")
ap.add_argument("--api_model", default="deepseek-flash")
ap.add_argument("--api_temp", type=float, default=0.0)
ap.add_argument("--api_thinking", default="disabled", choices=["disabled", "enabled"])
ap.add_argument("--data", default="default", choices=["default", "cord"])
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("rest", nargs=argparse.REMAINDER)
H = ap.parse_args()
rest = H.rest[1:] if H.rest and H.rest[0] == "--" else H.rest
sys.argv = [H.target + ".py"] + rest

# --------------------------------------------------------------- sharding --- #

class Shard:
    """Every K-th example of a dataset, starting at k."""
    def __init__(self, base, k, K):
        self.base, self.k, self.K = base, k, K
    def __len__(self):
        n = len(self.base)
        return max(0, (n - self.k + self.K - 1) // self.K)
    def __getitem__(self, i):
        return self.base[i * self.K + self.k]
    def shuffle(self, seed=None, **kw):
        return Shard(self.base.shuffle(seed=seed), self.k, self.K)


class CordQA:
    """CORD-v2 receipts as DocVQA-style rows asking for the total amount."""
    def __init__(self, rows):
        self.rows = rows
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        # rows hold (dataset, index, total); the image is decoded only when
        # an item is read, so K shard processes do not each hold 1000 images
        d, j, tot, qid = self.rows[i]
        return {"image": d[j]["image"],
                "question": "What is the total amount on this receipt?",
                "answers": [tot], "questionId": qid}
    def shuffle(self, seed=None, **kw):
        r = list(self.rows)
        random.Random(seed).shuffle(r)
        return CordQA(r)


def _cord_total(gt):
    try:
        g = json.loads(gt)["gt_parse"]
    except Exception:
        return None
    t = g.get("total")
    if isinstance(t, list):
        t = t[0] if t else None
    if not isinstance(t, dict):
        return None
    v = t.get("total_price")
    if isinstance(v, list):
        v = v[0] if v else None
    if not v or not re.search(r"\d", str(v)):
        return None
    return str(v).strip()


def amount_key(s):
    s = re.sub(r"[^\d.,]", "", str(s))
    s = re.sub(r"[.,]00$", "", s)   # 60,000.00 and 60,000 are the same
    return re.sub(r"\D", "", s)


import datasets as _hfds
_orig_load = _hfds.load_dataset


def _load(*a, **kw):
    if H.data == "cord":
        rows = []
        for split in ("train", "validation", "test"):
            d = _orig_load("naver-clova-ix/cord-v2", split=split)
            gts = d["ground_truth"]          # text column only, no images
            for j in range(len(d)):
                tot = _cord_total(gts[j])
                if tot is None:
                    continue
                rows.append((d, j, tot, f"cord-{split}-{j}"))
        print(f"CORD rows with a total: {len(rows)}", flush=True)
        base = CordQA(rows)
    else:
        base = _orig_load(*a, **kw)
    if H.nshards > 1:
        return Shard(base, H.shard, H.nshards)
    return base


_hfds.load_dataset = _load

# ------------------------------------------------------------ API reader --- #

_client = None


def _api():
    global _client
    if _client is None:
        from openai import OpenAI
        key = os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise SystemExit("DEEPSEEK_API_KEY is not set")
        _client = OpenAI(api_key=key, base_url="https://api.deepseek.com",
                         timeout=120)
    return _client


API_LOG = {"calls": 0, "errors": 0, "retries": 0, "empty": 0}


def api_chat(img, text, max_tokens=32):
    content = []
    if img is not None:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"}})
    content.append({"type": "text", "text": text})
    for attempt in range(8):
        try:
            r = _api().chat.completions.create(
                model=H.api_model,
                messages=[{"role": "user", "content": content}],
                temperature=H.api_temp,
                # deepseek-flash thinks by default and spends the token budget
                # on hidden reasoning; the local readers answer directly, so
                # thinking is off unless asked for.
                max_tokens=max_tokens if H.api_thinking == "disabled" else 8192,
                extra_body={"thinking": {"type": H.api_thinking}})
            API_LOG["calls"] += 1
            out = (r.choices[0].message.content or "").strip()
            if not out:
                API_LOG["empty"] += 1
            return out
        except Exception as e:
            API_LOG["retries"] += 1
            msg = str(e)
            if "401" in msg or "Authentication" in msg:
                raise
            time.sleep(min(60, 2 ** attempt + random.random()))
    API_LOG["errors"] += 1
    raise RuntimeError("api_failed")


class APIWebReader:
    """Same prompt as m2w_stage4.Reader.choose, answered by the API."""
    def __init__(self, model_id=None, device=None):
        self.model = None
    def choose(self, img, listing, ocr_text, n_choices,
               use_image, use_struct, use_ocr):
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
        ans = api_chat(img if use_image else None, prompt, max_tokens=16)
        m = re.search(r"\d+", ans)
        if m is None:
            return None
        i = int(m.group())
        return i if 0 <= i < n_choices else None


class APIDocReader:
    def __init__(self, model_id=None, device=None):
        self.model = None
        self.device = "cpu"


def api_ask(reader, img, text_channel, question, read_mode, cache, key):
    """Same prompt as docvqa_v3.ask_generic, answered by the API."""
    if cache is not None and key in cache:
        return cache[key]
    use_image = read_mode in ("fused", "image_primary")
    use_struct = read_mode in ("fused", "struct_primary")
    prompt = f"This is a document page. {question}"
    if use_struct:
        prompt += "\n\nOCR transcript of the page:\n" + text_channel
    prompt += "\n\nAnswer with ONLY the value, no explanation."
    ans = api_chat(img if use_image else None, prompt, max_tokens=32)
    if cache is not None:
        cache[key] = ans
    return ans


# ------------------------------------------------------ module patching --- #

if H.target in ("m2w_stage4", "m2w_organic"):
    import m2w_stage4 as M
    if H.api:
        M.Reader = APIWebReader
    T = M if H.target == "m2w_stage4" else __import__("m2w_organic")
else:
    import docvqa_v2 as D
    import docvqa_v3 as V
    if H.api:
        V.Reader = APIDocReader
        V.ask_generic = api_ask
        D.ask = api_ask
    if H.data == "cord":
        # Amounts are scored by exact value, not ANLS: 60,500 is not 60,000.
        def correct(pred, golds, tau=0.5):
            p = amount_key(pred)
            return bool(p) and any(p == amount_key(g) for g in golds)
        D.correct = correct
        V.D.correct = correct

        def pick_decoy(tokens, idxs, golds, rng):
            gk = {amount_key(g) for g in golds}
            nums = [t for t, _ in tokens
                    if len(amount_key(t)) >= 3 and amount_key(t) not in gk]
            if nums:
                return rng.choice(nums)
            g = str(golds[0])
            ds_ = [i for i, c in enumerate(g) if c.isdigit()]
            i = rng.choice(ds_[:-1] or ds_)
            return g[:i] + str((int(g[i]) + rng.randint(1, 8)) % 10) + g[i + 1:]
        D.pick_decoy = pick_decoy

        # A total printed twice (e.g. TOTAL and CASH) would survive a fault
        # aimed at one copy, so faults cover every copy of the amount.
        _CUR = {}
        _ocr0 = D.ocr_tokens
        def ocr_tokens(img):
            t = _ocr0(img)
            _CUR["toks"] = t
            return t
        D.ocr_tokens = ocr_tokens
        _def0 = V.deface
        def deface(img, bbox, kind, pad=6):
            im = _def0(img, bbox, kind, pad)
            gk = {amount_key(g) for g in _CUR.get("golds", [])}
            for t, bb in _CUR.get("toks", []):
                if amount_key(t) in gk and tuple(bb) != tuple(bbox):
                    im = _def0(im, bb, kind, pad)
            return im
        V.deface = deface
        _cor0 = V.corrupt
        def corrupt(tokens, idxs, decoy, kind):
            gk = {amount_key(g) for g in _CUR.get("golds", [])}
            more = {i for i, (t, _) in enumerate(tokens) if amount_key(t) in gk}
            return _cor0(tokens, set(idxs) | more, decoy, kind)
        V.corrupt = corrupt
        _run0 = V.run_item
        def run_item(reader, raw, *a, **kw):
            _CUR["golds"] = raw["answers"]
            _CUR["toks"] = []
            return _run0(reader, raw, *a, **kw)
        V.run_item = run_item
    T = V if H.target == "docvqa_v3" else __import__("docvqa_organic")
    if H.data == "cord" and H.target == "docvqa_organic":
        T.D.correct = D.correct

if __name__ == "__main__":
    print(f"harness target={H.target} api={H.api} data={H.data} "
          f"shard={H.shard}/{H.nshards} argv={sys.argv[1:]}", flush=True)
    T.main()
    print("API_LOG", json.dumps(API_LOG), flush=True)
