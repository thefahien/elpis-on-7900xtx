#!/usr/bin/env python3
"""Sustained soak: loop code/math/prose (thinking on + off) at 1024 tokens; log tok/s and card sensors.
Run on the GPU host (reads amdgpu hwmon from sysfs) or pass --no-sensors.
usage: soak.py <base_url> <api_key> <minutes> <out.jsonl> [--hwmon /sys/class/drm/card1/device/hwmon/hwmon7]
"""
import argparse, glob, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_api as B


def find_hwmon():
    for h in glob.glob("/sys/class/drm/card*/device/hwmon/hwmon*"):
        try:
            if open(f"{h}/name").read().strip() == "amdgpu":
                return h
        except OSError:
            pass
    return None


def sensors(h):
    rd = lambda f: int(open(f"{h}/{f}").read())
    return {"edge_c": rd("temp1_input") // 1000, "junction_c": rd("temp2_input") // 1000,
            "mem_c": rd("temp3_input") // 1000, "power_w": rd("power1_average") // 1_000_000}


ap = argparse.ArgumentParser()
ap.add_argument("base"); ap.add_argument("key"); ap.add_argument("minutes", type=float); ap.add_argument("out")
ap.add_argument("--model", default="qwen38-27b-exl3"); ap.add_argument("--hwmon", default=None)
ap.add_argument("--no-sensors", action="store_true")
a = ap.parse_args()
hw = None if a.no_sensors else (a.hwmon or find_hwmon())
t_end, n = time.time() + a.minutes * 60, 0
with open(a.out, "a") as f:
    while time.time() < t_end:
        for think in (True, False):
            B.NOTHINK = not think
            for name, p in B.PROMPTS.items():
                try:
                    dt, pt, ct, _, u = B.call(a.base, a.key, a.model, [{"role": "user", "content": p}], 1024)
                    rec = {"t": round(time.time()), "prompt": name, "think": think, "ct": ct, "s": round(dt, 2),
                           "tok_s": round(ct / dt, 1), "server_tok_s": u.get("completion_tokens_per_sec")}
                except Exception as e:
                    rec = {"t": round(time.time()), "prompt": name, "think": think, "error": str(e)[:200]}
                if hw:
                    try:
                        rec.update(sensors(hw))
                    except OSError as e:
                        rec["sensor_err"] = str(e)[:80]
                f.write(json.dumps(rec) + "\n"); f.flush()
                n += 1
print("requests", n)
