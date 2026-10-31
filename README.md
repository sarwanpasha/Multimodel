# Multimodel

Code for measuring how blame can be assigned when a multimodal agent fails.
An agent reads a task through several perception channels (an image, an OCR
transcript, a structured description). For every failure, the scripts restore
each subset of channels to a clean ("veridical") reading, rerun the agent with
greedy decoding, and record which restorations repair the failure. From these
rollouts they compute the family of minimal repairs and compare it with the set
of channels that were corrupted.

## Settings and scripts

| Setting | Script | Notes |
|---|---|---|
| Synthetic bar-chart episodes | `percept_vlm_pilot.py`, `percept_vlm_v2.py`, `percept_vlm_v3.py` | Shared reader and rendering in the pilot module |
| Charts with duplicated or split channels | `chart_split.py` | `--info dup` or `--info split` |
| DocVQA with an OCR channel | `docvqa_v2.py`, `docvqa_v3.py` | v3 adds non-Qwen readers |
| DocVQA natural (low vs high fidelity) | `docvqa_organic.py` | |
| Mind2Web web steps | `m2w_stage3.py`, `m2w_stage4.py` | 2 or 3 channels |
| Mind2Web natural (low vs high fidelity) | `m2w_organic.py` | |
| Receipts (CORD), API reader, sharding | `harness.py` | wraps the DocVQA and web scripts |
| Sampling stability | `sampling_stability.py`, `analyze_stability.py` | |
| Blame-error analysis | `blame_errors.py` | runs on saved result files |
| Utilities | `merge_shards.py`, `merge_api.py`, `analyze_r3.py` | merging shards, bootstrap summaries |

## Setup

Python 3.11 with PyTorch, Hugging Face `transformers` and `datasets`, EasyOCR, Pillow, NumPy, Matplotlib, and the OpenAI Python client (only for the API reader). See `requirements.txt`.

Datasets are downloaded from the Hugging Face Hub on first use: `lmms-lab/DocVQA`, `osunlp/Multimodal-Mind2Web`, and `naver-clova-ix/cord-v2`.

## Examples

```bash
# DocVQA, injected double fault, Qwen2-VL-2B
python docvqa_v3.py --model Qwen/Qwen2-VL-2B-Instruct --n_items 300 --out docvqa_2b.json

# Web steps with 3 channels, Qwen2.5-VL-7B
python m2w_stage4.py --model Qwen/Qwen2.5-VL-7B-Instruct --channels 3 --n_steps 300 --out web_3ch.json

# Natural failures on the web
python m2w_organic.py --model Qwen/Qwen2.5-VL-7B-Instruct --channels 3 --n_steps 300 --out web_natural.json

# Receipts (CORD) through the harness, shard 0 of 2
python harness.py --shard 0 --nshards 2 --target docvqa_v3 --data cord -- \
    --model Qwen/Qwen2-VL-2B-Instruct --n_items 150 --out receipts_2b.s0.json

# API reader (reads the key from the DEEPSEEK_API_KEY environment variable)
python harness.py --target m2w_stage4 --api -- --model deepseek-flash --channels 2 --n_steps 25 --out web_api.json

# Charts with split channels
python chart_split.py --info split --model Qwen/Qwen2-VL-7B-Instruct --trials 200 --out chart_split.json

# Blame errors over saved results
python blame_errors.py blame_errors.json docvqa_2b.json web_3ch.json web_natural.json
```

Each script writes a JSON file with its arguments, a summary, and one record per item, including the minimal repairs and the output of each blame procedure.

## License

MIT, see `LICENSE`.
