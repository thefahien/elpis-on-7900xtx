# Benchmarks

All elpis numbers are from an RX 7900 XTX at the stock 339 W power limit, unless stated otherwise.

## Metric

tok/s = completion tokens divided by the **whole request** wall time. Prefill and HTTP are included, so the
numbers are slightly below decode-only numbers. Requests are single-stream through the OpenAI-compatible API.

## Prompts

The prompts are in [`bench/bench_api.py`](../bench/bench_api.py):

- code: a Python LRU-cache class with tests
- math: a word problem
- prose: a 400-word story

"Thinking on" and "thinking off" use the model's chat template, toggled per request.

## Greedy and sampled, elpis vs llama.cpp (same session)

7900 XTX, same session, 1024-token replies. Rounds were interleaved: greedy, T=0.6 and T=1.0 back to back, so
thermals affect all three equally. n=6 per cell. Each server got 108 requests, with 0 errors.
Sampling parameters: T=0.6, top_p 0.95, top_k 20 and T=1.0, top_p 0.95, top_k 20.
Values are median [min-max] whole-request tok/s.

| tok/s | elpis greedy | elpis T=0.6 | elpis T=1.0 | llama.cpp greedy | llama.cpp T=0.6 | llama.cpp T=1.0 |
|---|---|---|---|---|---|---|
| code, thinking on | 106.7 [107-107] | 113.2 [103-131] | 95.5 [82-119] | 72.7 [72-76] | 74.2 [66-76] | 73.8 [67-80] |
| math, thinking on | 114.0 [114-114] | 113.7 [109-121] | 99.2 [96-108] | 73.5 [73-74] | 73.9 [72-76] | 71.5 [71-72] |
| prose, thinking on | 77.2 [77-77] | 86.1 [79-100] | 84.5 [74-95] | 69.6 [69-70] | 68.8 [67-72] | 67.1 [67-69] |
| code, thinking off | 178.4 [178-179] | 180.7 [171-186] | 179.3 [174-184] | 85.0 [84-86] | 86.6 [86-88] | 84.7 [83-86] |
| math, thinking off | 149.5 [149-150] | 152.2 [140-157] | 148.7 [135-157] | 83.1 [82-84] | 82.7 [81-83] | 81.2 [79-83] |
| prose, thinking off | 75.8 [76-76] | 69.5 [68-72] | 69.1 [66-70] | 64.3 [62-64] | 62.9 [60-65] | 60.2 [59-61] |

Raw data: [`results/7900xtx-exl3rocm_sampled.jsonl`](../results/7900xtx-exl3rocm_sampled.jsonl),
[`results/llamacpp-vulkan-swift1.5_sampled.jsonl`](../results/llamacpp-vulkan-swift1.5_sampled.jsonl).
Script: [`bench/sampled.py`](../bench/sampled.py).

Reading it:

- Greedy is deterministic, so its spread is near zero. Sampled runs produce different text each time, so their
  spread is real variation.
- At T=0.6, elpis code and math are within 0% to +6% of greedy. Prose with thinking off is about 8% lower;
  prose with thinking on is about 11% higher.
- At T=1.0, elpis thinking-on code and math lose about 10-13%. Thinking-off code and math do not change.
- llama.cpp moves 1-3 tok/s across the three settings.
- Speed only. Quality under sampling was not measured.
- llama.cpp ran a different model (see [Reproducing](#reproducing)). This is not a strict A/B.

### Draft acceptance (elpis)

Same run as above. Accepted / (accepted + rejected) draft tokens from the server's usage counters, summed over
the 6 runs per cell.

| draft accept | greedy | T=0.6 | T=1.0 |
|---|---|---|---|
| code, thinking on | 40% | 44% | 35% |
| math, thinking on | 44% | 44% | 37% |
| prose, thinking on | 24% | 30% | 28% |
| code, thinking off | 76% | 77% | 77% |
| math, thinking off | 62% | 62% | 60% |
| prose, thinking off | 24% | 21% | 20% |

## 30-minute soak (elpis)

7900 XTX, 1024-token replies, greedy, one 30-minute session. 33 rounds per cell, 198 requests, 191,433 tokens,
0 errors. Whole-request tok/s.

| tok/s | median | min | max |
|---|---|---|---|
| code, thinking on | 106.2 | 105.8 | 107.2 |
| math, thinking on | 113.4 | 112.9 | 114.7 |
| prose, thinking on | 76.8 | 76.5 | 77.5 |
| code, thinking off | 177.2 | 175.8 | 177.8 |
| math, thinking off | 148.6 | 147.8 | 149.2 |
| prose, thinking off | 75.4 | 75.0 | 75.6 |

Peaks: junction 81 °C, memory 95 °C, 334 W. Memory at 95 °C is close to GDDR6 throttling. Check yours if your
case runs hot. Raw data: [`results/7900xtx-exl3rocm_soak30m.jsonl`](../results/7900xtx-exl3rocm_soak30m.jsonl).

## 512-token protocol: 7900 XTX vs RTX 3090 (and an older llama.cpp run)

512-token replies, greedy, `bench/bench_api.py`, best of 2 reps per cell. Separate sessions and machines.
Only 2 reps, so treat these as rough.

| tok/s | elpis, 7900 XTX | elpis CUDA, RTX 3090 | XTX / 3090 | old llama.cpp, 7900 XTX |
|---|---|---|---|---|
| code, thinking on | 88.7 | 140.7 | 63% | 78.1 |
| math, thinking on | 114.0 | 172.1 | 66% | 74.6 |
| prose, thinking on | 68.9 | 130.0 | 53% | 64.8 |
| code, thinking off | 170.3 | 241.3 | 71% | 83.0 |
| math, thinking off | 151.9 | 229.4 | 66% | 84.1 |
| prose, thinking off | 71.7 | 113.0 | 63% | 64.0 |

- 7900 XTX: stock 339 W. Thinking on from `results/7900xtx-exl3rocm_think_long51k.json`, thinking off from
  `results/7900xtx-exl3rocm_nothink.json`.
- RTX 3090: gildrb/elpis commit `13e8b59`, 300 W, same weights, same prompts.
- Old llama.cpp run: Vulkan backend with the base Qwen3.8-27B IQ3_S GGUF, not the Swift-1.5 model used in the
  same-session tables. Its numbers in `results/llamacpp-vulkan_*.json` are best of 2.

## Prefill and long-prompt decode (elpis)

7900 XTX, one 51,491-token prompt (`--long 50000`), from `results/7900xtx-exl3rocm_think_long51k.json`.

| | value |
|---|---|
| prefill, server-side | 1100.9 tok/s |
| prefill, whole request (46.93 s) | 1097.2 tok/s |
| decode after the prompt, server-side (512 tokens, 51,456 tokens cached) | 71.6 tok/s |
| decode after the prompt, whole request | 68.3 tok/s |

## Context (elpis)

- 131072 tokens with Q8 KV cache.
- Peak VRAM 22.45 GiB with a 122k-token prompt.
- One needle-in-a-haystack question at 122k was answered correctly ([`bench/needle.py`](../bench/needle.py)). n=1.
- 192k does not fit with the 4.00bpw weights. The port's 192k profile is for its 3.5bpw model.
- There is no 3-bit KV cache on ROCm yet; the attention kernels cover Q8/FP16. A 3-bit KV cache is what lets the
  CUDA build reach 262k on a 24 GB card.

## Power limit (elpis)

Raising the limit from 339 W to 402 W gave about +2%. Single short run: thinking on, 512-token replies, best of
2 reps each.

| tok/s | 339 W | 402 W |
|---|---|---|
| code, thinking on | 88.7 | 90.4 |
| math, thinking on | 114.0 | 116.5 |
| prose, thinking on | 68.9 | 70.4 |

Raw data: `results/7900xtx-exl3rocm_think_402W.json` vs `results/7900xtx-exl3rocm_think_long51k.json`.

## Speculative decoding

Most of the speed comes from the draft. Acceptance is high on code and math and low on prose. Reasoning text
drafts poorly, so thinking-on numbers are lower. The ROCm port verifies a 7-token chain per round. The CUDA elpis
build ships a tree verifier and more kernel patches. That is the likely explanation for the gap to the 3090; it
was not proven here.

## Reproducing

### Measuring commands (elpis)

```bash
KEY=$(awk '/^api_key:/{print $2}' /opt/elpis-xtx/tabbyAPI/api_tokens.yml)
python3 bench/bench_api.py http://localhost:8096 $KEY qwen38-27b-exl3 mylabel --reps 2 --nothink --out out.json
python3 bench/bench_api.py http://localhost:8096 $KEY qwen38-27b-exl3 mylabel --long 50000      # prefill
python3 bench/needle.py http://localhost:8096 $KEY qwen38-27b-exl3 122000
python3 bench/soak.py http://localhost:8096 $KEY 30 soak.jsonl                                   # on the GPU machine
```

### Script notes

- TabbyAPI's non-streaming responses carry no `usage`. The scripts stream with `stream_options.include_usage` and
  read `completion_tokens_per_sec` and the accepted/rejected prediction counts from it.
- `bench/sampled.py` sends `top_k` 20. TabbyAPI and llama-server accept it.
- `bench/bench_api.py`, `bench/soak.py` and `bench/needle.py` do not send `top_k`. The CUDA elpis server rejects
  `top_k` with HTTP 400. Do not point `sampled.py` at that server.

### llama.cpp baseline (same-session tables)

- Upstream ggml-org/llama.cpp, commit `b3c3b96`, Vulkan build.
- Model: `Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` by ukisai (IQ3_S, about 3.55 bits/weight, MTP draft).
- A vision projector was also loaded. The server was launched with thinking on by default
  (`--reasoning-effort xhigh` and `--chat-template-kwargs '{"enable_thinking":true,"reasoning_effort":"xhigh"}'`);
  the benchmark requests set `enable_thinking` per request for the thinking-off rows.
- Server flags (paths, alias, host and port omitted):

```bash
llama-server -m Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf --mmproj-offload -ngl 99 -ngld 99 -fa 1 \
  --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-n-min 1 -c 262144 --parallel 1 --context-shift \
  --cache-type-k q8_0 --cache-type-v q8_0 --temp 1.0 --cache-ram 12288 --n-predict 8192 \
  --reasoning-effort xhigh \
  --chat-template-kwargs '{"enable_thinking":true,"reasoning_effort":"xhigh"}'
```

## Raw data index

| file | contents |
|---|---|
| [`results/7900xtx-exl3rocm_sampled.jsonl`](../results/7900xtx-exl3rocm_sampled.jsonl) | elpis, 7900 XTX, greedy / T=0.6 / T=1.0, 1024-token replies, 6 rounds |
| [`results/llamacpp-vulkan-swift1.5_sampled.jsonl`](../results/llamacpp-vulkan-swift1.5_sampled.jsonl) | llama.cpp Vulkan, Swift-1.5 IQ3_S, same session and protocol |
| [`results/7900xtx-exl3rocm_soak30m.jsonl`](../results/7900xtx-exl3rocm_soak30m.jsonl) | elpis, 7900 XTX, 30-minute soak, 1024-token replies, greedy |
| [`results/7900xtx-exl3rocm_think_long51k.json`](../results/7900xtx-exl3rocm_think_long51k.json) | elpis, 7900 XTX, 512-token, thinking on, 339 W; 51k-token prefill run |
| [`results/7900xtx-exl3rocm_nothink.json`](../results/7900xtx-exl3rocm_nothink.json) | elpis, 7900 XTX, 512-token, thinking off |
| [`results/7900xtx-exl3rocm_think_402W.json`](../results/7900xtx-exl3rocm_think_402W.json) | elpis, 7900 XTX, 512-token, thinking on, 402 W |
| [`results/rtx3090-elpis-cuda_think.json`](../results/rtx3090-elpis-cuda_think.json) | gildrb/elpis CUDA, RTX 3090, 512-token, thinking on |
| [`results/rtx3090-elpis-cuda_nothink.json`](../results/rtx3090-elpis-cuda_nothink.json) | gildrb/elpis CUDA, RTX 3090, 512-token, thinking off |
| [`results/llamacpp-vulkan_think.json`](../results/llamacpp-vulkan_think.json) | old llama.cpp Vulkan run, base Qwen3.8-27B IQ3_S, 512-token, thinking on |
| [`results/llamacpp-vulkan_nothink.json`](../results/llamacpp-vulkan_nothink.json) | old llama.cpp Vulkan run, base Qwen3.8-27B IQ3_S, 512-token, thinking off |
