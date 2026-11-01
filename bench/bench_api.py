#!/usr/bin/env python3
"""Same-prompt bench for any OpenAI-compatible endpoint (used for all numbers in the README).

usage: bench_api.py <base_url> <api_key> <model> <label> [--long 32000] [--reps 2] [--out f.json]
Greedy (temperature 0), streaming with include_usage; tok/s = completion tokens / whole-request wall
time (prefill + HTTP included). --nothink sends chat_template_kwargs.enable_thinking=false. --long N adds a ~N-token unique-prefix prompt and reports
TTFT-based prefill tok/s (max_tokens=1) plus decode tok/s after it.
"""
import argparse, json, random, string, time, urllib.request

PROMPTS = {
    "code": "Write a Python class LRUCache with get(key) and put(key, value) in O(1) using a dict and a "
            "doubly linked list, no OrderedDict. Include type hints, docstrings, and five pytest tests.",
    "math": "Solve step by step: A shop sells pens at 3 for $4 and notebooks at $2.50 each. Priya buys 18 pens "
            "and some notebooks, pays with two $50 notes and gets $43 change. How many notebooks did she buy? "
            "Then verify the answer by recomputing the total.",
    "prose": "Write a 400-word short story about a lighthouse keeper who discovers that the light has been "
             "signalling to someone on the other side of the sea for years.",
}


NOTHINK = False
SAMPLING = {"temperature": 0.0}   # soak/sampled.py override this; default stays greedy


def call(base, key, model, messages, max_tokens, timeout=900):
    """Streaming request with include_usage; returns wall s, prompt tok, completion tok, text, usage."""
    body = {"model": model, "messages": messages, "max_tokens": max_tokens,
            "stream": True, "stream_options": {"include_usage": True}, **SAMPLING}
    if NOTHINK:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    t0 = time.time()
    usage, parts = {}, []
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            for c in d.get("choices") or []:
                delta = c.get("delta") or {}
                parts.append((delta.get("content") or "") + (delta.get("reasoning_content") or ""))
    dt = time.time() - t0
    return dt, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), "".join(parts), usage


def filler(n_tokens):
    rnd = random.Random(time.time_ns())
    tag = "".join(rnd.choice(string.ascii_lowercase) for _ in range(12))
    names = ["user", "order", "cache", "config", "session", "item", "index", "buffer", "token", "event"]
    out, i = [f"# run {tag}\n"], 0
    while len(out) < n_tokens // 28:
        a, b = rnd.choice(names), rnd.choice(names)
        out.append(f"def {a}_{b}_{i}(x, y):\n    '''Update {a} with {b}.'''\n    if x > {i}:\n        return {a}(x) + {b}(y)\n    return None\n\n")
        i += 1
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base"); ap.add_argument("key"); ap.add_argument("model"); ap.add_argument("label")
    ap.add_argument("--reps", type=int, default=2); ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--long", type=int, default=0); ap.add_argument("--out", default=None)
    ap.add_argument("--nothink", action="store_true")
    a = ap.parse_args()
    global NOTHINK
    NOTHINK = a.nothink
    res = {"label": a.label, "base": a.base, "runs": {}}
    call(a.base, a.key, a.model, [{"role": "user", "content": "Reply with OK."}], 8)  # warm
    for name, p in PROMPTS.items():
        rs = []
        for _ in range(a.reps):
            dt, pt, ct, text, u = call(a.base, a.key, a.model, [{"role": "user", "content": p}], a.tokens)
            acc = (u.get("completion_tokens_details") or {})
            rs.append({"s": round(dt, 2), "ct": ct, "tok_s": round(ct / dt, 1),
                       "server_decode_tok_s": u.get("completion_tokens_per_sec"),
                       "accepted": acc.get("accepted_prediction_tokens"), "rejected": acc.get("rejected_prediction_tokens")})
        best = max(r["tok_s"] for r in rs)
        res["runs"][name] = {"reps": rs, "best": best, "sample": text[:300]}
        print(f"{a.label} {name}: " + ", ".join(
            f"{r['ct']} tok/{r['s']}s = {r['tok_s']} whole (server decode {r['server_decode_tok_s']})" for r in rs), flush=True)
    if a.long:
        body = filler(a.long) + "\n# Now write merge_sessions(a, b) that combines two session dicts, with a docstring and tests.\n"
        msgs = [{"role": "user", "content": body}]
        dt1, pt, _, _, u1 = call(a.base, a.key, a.model, msgs, 1)
        res["prefill"] = {"prompt_tokens": pt, "s": round(dt1, 2), "tok_s": round(pt / dt1, 1),
                          "server_prompt_tok_s": u1.get("prompt_tokens_per_sec")}
        print(f"{a.label} prefill: {pt} tok in {dt1:.1f}s = {pt / dt1:.0f} tok/s (server {u1.get('prompt_tokens_per_sec')})", flush=True)
        # same prompt again: prefix cache should hit, so this times decode after a long context
        dt2, pt2, ct2, _, u2 = call(a.base, a.key, a.model, msgs, a.tokens)
        res["long_decode"] = {"prompt_tokens": pt2, "s": round(dt2, 2), "ct": ct2, "tok_s_whole": round(ct2 / dt2, 1),
                              "server_decode_tok_s": u2.get("completion_tokens_per_sec"),
                              "cached": (u2.get("prompt_tokens_details") or {}).get("cached_tokens")}
        print(f"{a.label} decode@{pt2}: {ct2} tok in {dt2:.1f}s = {ct2 / dt2:.1f} whole, server decode "
              f"{u2.get('completion_tokens_per_sec')} (cached {res['long_decode']['cached']})", flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
