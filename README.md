# elpis on an RX 7900 XTX

This repo runs the "elpis" model pair on an AMD RX 7900 XTX. The pair is Qwen3.8-27B as EXL3 4.00bpw plus a
DFlash2 speculative draft, both by [r0b0tlab](https://huggingface.co/r0b0tlab). It uses
[phoenixhaxor/exllamav3-rocm](https://github.com/phoenixhaxor/exllamav3-rocm) and its patched TabbyAPI.
The repo contains the recipe, two small patches, the benchmark scripts and the raw numbers.

Short version:

- Against llama.cpp on the same card, in one session, with 1024-token greedy replies: about 2.1x on code and 1.8x on
  math with thinking off, about 1.5x on code and math with thinking on, and 1.1-1.2x on prose. llama.cpp ran a
  different, smaller model (see "Not like-for-like" below).
- Against the CUDA [gildrb/elpis](https://github.com/gildrb/elpis) build on an RTX 3090 (512-token replies, best of
  2 reps only): the 7900 XTX reaches 53-71% of the 3090's speed, depending on the prompt.

## Results

Same session, same card, 1024-token replies, greedy (temperature 0), single stream, median of 6 rounds.
tok/s = completion tokens / whole-request wall time. Definitions and all other tables: [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

| tok/s | llama.cpp (Vulkan) | elpis | speedup |
|---|---|---|---|
| code, thinking on | 72.7 | 106.7 | 1.47x |
| math, thinking on | 73.5 | 114.0 | 1.55x |
| prose, thinking on | 69.6 | 77.2 | 1.11x |
| code, thinking off | 85.0 | 178.4 | 2.10x |
| math, thinking off | 83.1 | 149.5 | 1.80x |
| prose, thinking off | 64.3 | 75.8 | 1.18x |

Raw data: [`results/7900xtx-exl3rocm_sampled.jsonl`](results/7900xtx-exl3rocm_sampled.jsonl),
[`results/llamacpp-vulkan-swift1.5_sampled.jsonl`](results/llamacpp-vulkan-swift1.5_sampled.jsonl).

- **Sampled decoding:** at T=0.6, elpis is within 0% to +6% of greedy on code and math. Prose with thinking off is
  about 8% lower (76 -> 70). At T=1.0, thinking-on code and math lose about 10-13%. llama.cpp moves 1-3 tok/s.
- **30-minute soak:** 198 requests, 191k tokens, 0 errors. Peaks: junction 81 °C, memory 95 °C, 334 W. Memory at
  95 °C is close to GDDR6 throttling.
- **Context:** 131072 tokens with Q8 KV. Peak VRAM 22.45 GiB at a 122k-token prompt. One needle-in-a-haystack
  question at 122k was answered correctly (n=1).
- **Prefill:** 1100 tok/s on a 51,491-token prompt (1100.9 server-side, 1097 whole-request). Decode after that
  prompt: 71.6 tok/s server-side (68.3 whole-request).

### Not like-for-like

- llama.cpp ran `Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` by ukisai: IQ3_S, about 3.55 bits/weight, MTP draft,
  1 slot. elpis runs 4.00 bpw EXL3. Different weights, different size.
- Only the llama.cpp Vulkan backend was tested, not HIP/ROCm.
- Output quality was not compared. Quality under sampling was not measured.
- Thinking-on rows are cut at the reply limit, so they measure the start of the reasoning trace.

### Known limitations

- 192k context does not fit with the 4.00bpw weights.
- No 3-bit KV cache on ROCm yet.
- Memory temperature reaches 95 °C under sustained load, near throttling.
- The ROCm port verifies a draft chain, not a tree.
- Thinking-on speeds depend on what the model writes.

## Setup

Tested on Ubuntu 24.04 in a Linux container with `/dev/kfd` and `/dev/dri/renderD*` passed through, host kernel
7.0 (Proxmox), Intel i9-12900K, 32 GB RAM. Software: ROCm 7.2.4 (AMD apt repo), Python 3.12, torch 2.13.0+rocm7.2,
exllamav3-rocm at `360c902`, TabbyAPI at `f07131c` (pinned by the port's installer).

```bash
git clone https://github.com/thefahien/elpis-on-7900xtx && cd elpis-on-7900xtx
sudo bash scripts/setup.sh /opt/elpis-xtx /models       # ROCm, venv, build (~10 min), TabbyAPI, config
bash scripts/download_models.sh /models                 # ~17.8 GB at pinned revisions, sha256-checked
sudo cp config/elpis-xtx.service /etc/systemd/system/ && sudo systemctl daemon-reload
sudo systemctl start elpis-xtx                           # cold load takes 3-4 minutes
```

TabbyAPI writes its API key on first start. The server listens on port 8096 (`/v1/chat/completions`, with
streaming). The model name is `qwen38-27b-exl3` (`model_name` in `config/config.yml`).

```bash
KEY=$(awk '/^api_key:/{print $2}' /opt/elpis-xtx/tabbyAPI/api_tokens.yml)
curl -s http://localhost:8096/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model": "qwen38-27b-exl3", "max_tokens": 32,
       "messages": [{"role": "user", "content": "Say hello."}],
       "chat_template_kwargs": {"enable_thinking": false}}'
```

Thinking is on by default. Turn it off per request with `"chat_template_kwargs": {"enable_thinking": false}`.

## Patches

Both apply to exllamav3-rocm `360c902` and are applied by `setup.sh`. Details: [patches/README.md](patches/README.md).

1. `0001-attention-host-rsqrtf.patch`: fixes a build error with ROCm 7.2.4 (clang 22).
2. `0002-dflash2-codebook-weight-suffix.patch`: lets r0b0tlab's DFlash2 draft load.

## Measuring

Commands, script notes and the llama.cpp baseline setup are in [docs/BENCHMARKS.md](docs/BENCHMARKS.md#reproducing).

## Credits and licenses

- Model: Qwen3.8-27B by the Qwen team. See its model card for the license.
- EXL3 quant and DFlash2 draft by [r0b0tlab](https://huggingface.co/r0b0tlab) (Apache-2.0).
- [exllamav3](https://github.com/turboderp-org/exllamav3) by turboderp (MIT).
- ROCm port: [phoenixhaxor/exllamav3-rocm](https://github.com/phoenixhaxor/exllamav3-rocm) (MIT).
- [TabbyAPI](https://github.com/theroyallab/tabbyAPI) by theroyallab (AGPL-3.0).
- CUDA build: [gildrb/elpis](https://github.com/gildrb/elpis) (MIT). Commit `13e8b59` was used for the RTX 3090
  numbers, with the 3090 at 300 W.
- Scripts and patches in this repo: MIT.
