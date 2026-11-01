# elpis on an RX 7900 XTX

Running the "elpis" model pair (Qwen3.8-27B as EXL3 4.00bpw + DFlash2 speculative draft, both by
[r0b0tlab](https://huggingface.co/r0b0tlab)) on an AMD RX 7900 XTX through
[phoenixhaxor/exllamav3-rocm](https://github.com/phoenixhaxor/exllamav3-rocm) and its patched TabbyAPI.
This repo is the recipe, two small patches that were needed, the benchmark scripts and the raw numbers.

Short version: on a 7900 XTX it runs at **~177 tok/s on code and ~75 tok/s on prose** (thinking off),
**~1100 tok/s prefill**, 131k context, stable over a 30-minute soak. That is about 2x llama.cpp on the same
card for code, and about 70% of the CUDA [gildrb/elpis](https://github.com/gildrb/elpis) build on an RTX 3090.

## Results

Greedy (temperature 0), single stream, through the OpenAI-compatible API. tok/s = completion tokens divided by
the **whole request** wall time (prefill and HTTP included), so these are slightly below decode-only numbers.
Prompts are in [`bench/bench_api.py`](bench/bench_api.py): a Python LRU-cache class with tests (code), a word
problem (math), a 400-word story (prose).

| tok/s | llama.cpp, 7900 XTX ¹ | **elpis, 7900 XTX** ² | elpis CUDA, RTX 3090 ³ |
|---|---|---|---|
| code, thinking on | 78 | **106** | 140 |
| math, thinking on | 74 | **113** | 172 |
| prose, thinking on | 65 | **77** | 130 |
| code, thinking off | 82 | **177** | 241 |
| math, thinking off | 82 | **149** | 229 |
| prose, thinking off | 64 | **75** | 113 |
| prefill, 51k-token prompt | - | **1100** | 1356 |
| decode after a 51k prompt (server-side, thinking on) | - | **72** | - |

¹ llama.cpp Vulkan backend, a Qwen3.8-27B IQ3_S GGUF with an MTP draft. Not the same weights: included only as a
  rough "what this card did before" baseline. 512-token replies.
² This repo. 30-minute soak medians, 1024-token replies (raw data:
  [`results/7900xtx-exl3rocm_soak30m.jsonl`](results/7900xtx-exl3rocm_soak30m.jsonl)). Prefill and long-context
  decode from a separate 512-token run. Stock 339 W power limit.
³ gildrb/elpis docker build (commit 13e8b59) on an RTX 3090 at 300 W, same weights, same prompts, 512-token replies.

Speculative decoding is the whole story here: acceptance is high on code and math and low on prose (measured
~3.2 / ~4.2 / ~2.4 tokens emitted per verification round on code / math / prose with thinking on), and reasoning text drafts poorly, so
thinking-on numbers are lower. The gap to the CUDA build is the engine, not a misconfiguration: the ROCm port
verifies a 7-token chain per round, the CUDA elpis build ships a tree verifier and many CUDA kernel patches.
Our numbers match the port's own README (it reports 135 code / 72 prose with its default model pair).

**Soak (30 min, 198 requests, 191k tokens):** 0 errors, no drift (first five vs last five rounds within 0.5%).
Peaks: junction 81 °C, memory 95 °C, 334 W. Memory at 95 °C is close to GDDR6 throttling; it held flat, but
check yours if your case runs hot.

**Sampled decoding (greedy vs temperature):** the numbers above are greedy. Same card, same session, same prompts,
1024-token replies, 6 rounds interleaved (greedy / T=0.6 / T=1.0 back to back, so thermals hit all three equally),
108 requests, 0 errors. Sampling is real on this server (identical requests give different text) and `top_k` is
accepted. Median whole-request tok/s; "draft accept" is accepted / (accepted + rejected) draft tokens from the
server's usage counters, summed over the 6 runs. Raw data:
[`results/7900xtx-exl3rocm_sampled.jsonl`](results/7900xtx-exl3rocm_sampled.jsonl), script:
[`bench/sampled.py`](bench/sampled.py).

| tok/s (median of 6) | greedy | T=0.6, top_p 0.95, top_k 20 | T=1.0, top_p 0.95, top_k 20 |
|---|---|---|---|
| code, thinking on | 107 | 113 | **96** |
| math, thinking on | 114 | 114 | **99** |
| prose, thinking on | 77 | 86 | 85 |
| code, thinking off | 178 | 181 | 179 |
| math, thinking off | 150 | 152 | 149 |
| prose, thinking off | 76 | 70 | 69 |

Draft accept, thinking on: code 40% / 44% / 35%, math 44% / 44% / 37%, prose 24% / 30% / 28% (greedy / T=0.6 / T=1.0).
Thinking off: code 76-77% at all three, math 60-62%, prose 20-24%.

Reading it: at the recommended T=0.6 the speed is the same as greedy within run-to-run spread (code and math thinking
on: -0% to +6%; one run in six reached 131 on code). At T=1.0 the thinking-on code and math cases lose about 10-13%
because the draft's guesses are rejected more often. Thinking-off code and math are unchanged even at T=1.0 (those
tokens are low-entropy). Prose is mixed: thinking off loses about 8% when sampled, thinking on gains about 10%.
Caveats: greedy is deterministic, so its spread is near zero, while sampled runs generate different text each time
and the spread is the real variation; each reply is cut at 1024 tokens, so thinking-on rows measure the start of
the reasoning trace, not a full answer; n=6 per cell.

**Same-session comparison with llama.cpp (greedy and sampled):** the same `bench/sampled.py` run against llama.cpp on the
same card, same prompts, 1024-token replies, 6 rounds, 108 requests, 0 errors. llama.cpp is the Vulkan build serving
`Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` (a Qwen3.8-27B derivative at IQ3_S, about 3.55 bits/weight, MTP
draft, 1 slot), not the same weights as elpis (4.00 bpw EXL3), so this is "what the card did before", not a strict
A/B. Raw data:
[`results/llamacpp-vulkan-swift1.5_sampled.jsonl`](results/llamacpp-vulkan-swift1.5_sampled.jsonl).

| tok/s (median of 6) | llama.cpp greedy / T=0.6 / T=1.0 | elpis greedy / T=0.6 / T=1.0 | elpis vs llama.cpp |
|---|---|---|---|
| code, thinking on | 73 / 74 / 74 | 107 / 113 / 96 | 1.5x / 1.5x / 1.3x |
| math, thinking on | 74 / 74 / 72 | 114 / 114 / 99 | 1.6x / 1.5x / 1.4x |
| prose, thinking on | 70 / 69 / 67 | 77 / 86 / 85 | 1.1x / 1.3x / 1.3x |
| code, thinking off | 85 / 87 / 85 | 178 / 181 / 179 | 2.1x at all three |
| math, thinking off | 83 / 83 / 81 | 150 / 152 / 149 | 1.8x at all three |
| prose, thinking off | 64 / 63 / 60 | 76 / 70 / 69 | 1.2x / 1.1x / 1.2x |

llama.cpp barely moves with sampling (within 1-3 tok/s), so the sampled gap to elpis is about the same as the greedy
gap: thinking-on 1.3-1.6x on code and math (1.1-1.3x on prose), thinking-off 1.8-2.1x on code and math. The earlier
table's llama.cpp column (78 / 74 / 65 etc.) was a separate 512-token run on a different GGUF, so it differs by a few
tok/s from this one.

**Context:** 131072 tokens with Q8 KV. Peak VRAM 22.45 GiB with a 122k-token prompt; a needle-in-a-haystack
question at 122k was answered correctly ([`bench/needle.py`](bench/needle.py)). 192k does not fit with the
4.00bpw weights (the port's 192k profile is for its 3.5bpw model). There is no 3-bit KV cache on ROCm yet (the
attention kernels cover Q8/FP16), which is what lets the CUDA build reach 262k on a 24 GB card.

**Power limit:** raising it from 339 W to 402 W gave +2%. Not worth it.

## Setup

Tested on Ubuntu 24.04 in a Linux container with `/dev/kfd` and `/dev/dri/renderD*` passed through, host kernel
7.0 (Proxmox), Intel i9-12900K, 32 GB RAM. Software: ROCm 7.2.4 (AMD apt repo), Python 3.12, torch 2.13.0+rocm7.2,
exllamav3-rocm at `360c902`, TabbyAPI at `f07131c` (pinned by the port's installer).

```bash
git clone <this repo> elpis-on-7900xtx && cd elpis-on-7900xtx
sudo bash scripts/setup.sh /opt/elpis-xtx /models       # ROCm, venv, build (~10 min), TabbyAPI, config
bash scripts/download_models.sh /models                 # ~17.8 GB at pinned revisions, sha256-checked
sudo cp config/elpis-xtx.service /etc/systemd/system/ && sudo systemctl daemon-reload
sudo systemctl start elpis-xtx                           # cold load takes 3-4 minutes
```

TabbyAPI writes its API key to `/opt/elpis-xtx/tabbyAPI/api_tokens.yml` on first start. The server listens on
port 8096 (`/v1/chat/completions`, streaming, tools, images). Turn thinking off per request with
`"chat_template_kwargs": {"enable_thinking": false}`.

## The two patches

Both apply to exllamav3-rocm `360c902` and are applied by `setup.sh`.

1. [`0001-attention-host-rsqrtf.patch`](patches/0001-attention-host-rsqrtf.patch): the build fails with ROCm
   7.2.4 (clang 22) because `attention.cu` calls `rsqrtf()` in host code, and clang only declares a `__device__`
   version:
   ```
   attention.hip:910:44: error: no matching function for call to 'rsqrtf'
   note: candidate function not viable: call to __device__ function from __host__ function
   ```
   Replaced with `1.0f / sqrtf(dim)`, which is the same value. Delete any stale `attention.hip` before
   rebuilding, or the old hipified copy is compiled again.
2. [`0002-dflash2-codebook-weight-suffix.patch`](patches/0002-dflash2-codebook-weight-suffix.patch): r0b0tlab's
   DFlash2 draft stores the candidate-selector codebooks as `...predecessor_codebook.weight` /
   `...successor_codebook.weight` (nn.Embedding names); the port only looks for the bare names and the draft fails
   to load with a missing-tensor error. The loader now tries the bare name first, then `.weight`.

## Measuring

```bash
python3 bench/bench_api.py http://localhost:8096 $KEY qwen38-27b-exl3 mylabel --reps 2 --nothink --out out.json
python3 bench/bench_api.py http://localhost:8096 $KEY qwen38-27b-exl3 mylabel --long 32000      # prefill
python3 bench/needle.py http://localhost:8096 $KEY qwen38-27b-exl3 122000
python3 bench/soak.py http://localhost:8096 $KEY 30 soak.jsonl                                   # on the GPU host
```

Traps we hit: TabbyAPI's non-streaming responses carry no `usage`, so the scripts stream with
`stream_options.include_usage` and read `completion_tokens_per_sec` and the accepted/rejected prediction counts
from it. The CUDA elpis server rejects `top_k` with a 400, so the scripts do not send it.

## Credits and licenses

- Model: Qwen3.8-27B by the Qwen team. EXL3 quant and DFlash2 draft by r0b0tlab (Apache-2.0).
- [exllamav3](https://github.com/turboderp-org/exllamav3) by turboderp (MIT); ROCm port by phoenixhaxor (MIT).
- [TabbyAPI](https://github.com/theroyallab/tabbyAPI) by theroyallab.
- elpis CUDA build: [gildrb/elpis](https://github.com/gildrb/elpis) (MIT).
- Scripts and patches in this repo: MIT.
