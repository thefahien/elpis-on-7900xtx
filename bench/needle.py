#!/usr/bin/env python3
"""Needle-in-haystack near the context limit. usage: needle.py <base> <key> <model> <n_tokens> [depth 0-1]"""
import json, random, sys, time, urllib.request

base, key, model, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
depth = float(sys.argv[5]) if len(sys.argv) > 5 else 0.37
rnd = random.Random(7)
words = "river stone lamp field copper window market thread harbor lantern orchard pencil signal valley".split()
code = f"{rnd.randint(100000, 999999)}"
sents, approx = [], 0
while approx < n:
    s = " ".join(rnd.choice(words) for _ in range(14)).capitalize() + "."
    sents.append(s); approx += 17
sents.insert(int(len(sents) * depth), f"The secret vault code for the harbor archive is {code}.")
prompt = " ".join(sents) + "\n\nQuestion: what is the secret vault code for the harbor archive? Reply with the number only."
body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 30, "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False}}
req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
t0 = time.time()
try:
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read())
except urllib.error.HTTPError as e:
    print("HTTP", e.code, e.read()[:400]); sys.exit(1)
dt = time.time() - t0
ans = d["choices"][0]["message"].get("content") or ""
u = d.get("usage") or {}
print(f"tokens~{approx} prompt_tokens={u.get('prompt_tokens')} {dt:.1f}s expected={code} got={ans.strip()!r} "
      f"{'PASS' if code in ans else 'FAIL'}")
