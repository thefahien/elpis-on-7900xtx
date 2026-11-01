#!/usr/bin/env python3
"""Greedy vs sampled decoding, same session, interleaved so drift/thermals hit every config equally.
usage: sampled.py <base_url> <api_key> <reps> <out.jsonl> [--tokens 1024] [--model qwen38-27b-exl3]
Configs: greedy (T=0) | T=0.6 top_p=0.95 top_k=20 (Qwen3 thinking-mode recommendation)
         | T=1.0 top_p=0.95 top_k=20 (Qwen3.8 generation_config default). tok/s = completion tokens / whole request.
accepted/rejected = usage.completion_tokens_details prediction counters from the server (draft tokens).
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_api as B

CONFIGS = {
    "greedy": {"temperature": 0.0},
    "t0.6": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
    "t1.0": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
}
ap = argparse.ArgumentParser()
ap.add_argument("base"); ap.add_argument("key"); ap.add_argument("reps", type=int); ap.add_argument("out")
ap.add_argument("--tokens", type=int, default=1024); ap.add_argument("--model", default="qwen38-27b-exl3")
a = ap.parse_args()
B.call(a.base, a.key, a.model, [{"role": "user", "content": "Reply with OK."}], 8)  # warm
with open(a.out, "a") as f:
    for rep in range(a.reps):
        for think in (True, False):
            B.NOTHINK = not think
            for name, p in B.PROMPTS.items():
                for cname, params in CONFIGS.items():
                    B.SAMPLING = params
                    rec = {"t": round(time.time()), "rep": rep, "cfg": cname, "think": think, "prompt": name}
                    try:
                        dt, pt, ct, _, u = B.call(a.base, a.key, a.model, [{"role": "user", "content": p}], a.tokens)
                        d = u.get("completion_tokens_details") or {}
                        rec.update(ct=ct, s=round(dt, 2), tok_s=round(ct / dt, 1), server_tok_s=u.get("completion_tokens_per_sec"),
                                   accepted=d.get("accepted_prediction_tokens"), rejected=d.get("rejected_prediction_tokens"))
                    except Exception as e:
                        rec["error"] = str(e)[:200]
                    f.write(json.dumps(rec) + "\n"); f.flush()
                    print(rec, flush=True)
