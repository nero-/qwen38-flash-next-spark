#!/usr/bin/env python3
"""Serve N concurrent near-max-context sessions; check capacity, retrieval and RAM on one Spark.

Each session has a unique leading line (no prefix-cache sharing) and three passkeys
at different depths. Passes only if every session completes without preemption,
retrieves its passkeys, and the Spark stays under the used-RAM limit.
--followup (phase 2, run after phase 1): send a new question on every session's
identical ~258K prefix at once. Passes only if the prefix cache still holds all
contexts (cache hits), all sessions decode concurrently, nothing is preempted,
and RAM stays under the limit: N resident max-context chat sessions.
Usage: capacity_check.py TAG [--followup] [--sessions 4] [--prompt-tokens 258000] [--limit-gib 118]
"""
import argparse, concurrent.futures, json, re, subprocess, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL = "qwen3.8-flash-next-4p89bpw"
p = argparse.ArgumentParser()
p.add_argument("tag")
p.add_argument("--sessions", type=int, default=4)
p.add_argument("--prompt-tokens", type=int, default=258000)
p.add_argument("--limit-gib", type=float, default=118)
p.add_argument("--followup", action="store_true")
a = p.parse_args()


def post(path, payload, timeout=3600):
    req = urllib.request.Request("http://127.0.0.1:8000" + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def metric(name):
    text = urllib.request.urlopen("http://127.0.0.1:8000/metrics", timeout=10).read().decode()
    return sum(float(l.split()[-1]) for l in text.splitlines() if l.startswith(name + "{") or l.startswith(name + " "))


tok = lambda s: post("/tokenize", {"model": MODEL, "prompt": s, "add_special_tokens": False})["tokens"]
filler = tok(" Ordinary archive entry: the warehouse ledger lists shelves, labels, crates and loading bays.\n")


QUESTION = ("\nReturn the three audit passkeys, labeled ALPHA, BETA, GAMMA. Copy them exactly; no explanation.")
FOLLOWUP = ("\nFrom the archive above, return only the GAMMA passkey, then describe the archive in three sentences.")
FRAME = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def build(i, question=QUESTION):
    keys = {name: f"{name.lower()}-{i}{n:06d}" for name, n in (("ALPHA", 731826), ("BETA", 492175), ("GAMMA", 863204))}
    head = tok(f"<|im_start|>user\nSession {i} archive, reference code S{i:02d}-{i * 7919 % 100000:05d}.\n")
    # The body is sized from the phase-1 tail so both phases share an identical prefix.
    body_len = a.prompt_tokens - len(head) - len(tok(QUESTION + FRAME))
    body = (filler * (a.prompt_tokens // len(filler) + 2))[:body_len]
    for depth, (name, key) in reversed(list(zip((0.05, 0.5, 0.95), keys.items()))):
        at = int(len(body) * depth)
        body[at:at] = tok(f"\nAudit passkey {name}: {key}\n")
    return head + body[:body_len] + tok(question + FRAME), keys


def session(i):
    prompt, keys = build(i, FOLLOWUP if a.followup else QUESTION)
    started = time.time()
    extra = dict(max_tokens=512, min_tokens=400) if a.followup else dict(max_tokens=96)
    data = post("/v1/completions", dict(model=MODEL, prompt=prompt, temperature=0, seed=i, **extra))
    text = data["choices"][0]["text"]
    wanted = [keys["GAMMA"]] if a.followup else list(keys.values())
    return dict(session=i, prompt_tokens=data["usage"]["prompt_tokens"], wall_s=round(time.time() - started, 1),
                completion_tokens=data["usage"]["completion_tokens"],
                retrieved=sum(k in text for k in wanted), wanted=len(wanted), text=text[:300])


samples = []


def sampler(stop):
    while not stop.is_set():
        text = urllib.request.urlopen("http://127.0.0.1:8000/metrics", timeout=10).read().decode()
        pick = lambda n: sum(float(l.split()[-1]) for l in text.splitlines() if l.startswith(n + "{"))
        samples.append(dict(running=pick("vllm:num_requests_running"), kv=pick("vllm:kv_cache_usage_perc")))
        stop.wait(1)


phase = "capacity-followup" if a.followup else "capacity"
dest = ROOT / "evidence" / f"{a.tag}-{phase}.json"
assert not dest.exists(), dest
watch = ["python3", str(ROOT / "watch_memory.py"), "--tag", f"{a.tag}-{phase}", "--seconds", "5400"]
watchers = [subprocess.Popen(watch, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)]
preempt0 = metric("vllm:num_preemptions_total")
hits0, queries0 = metric("vllm:prefix_cache_hits_total"), metric("vllm:prefix_cache_queries_total")
import threading
stop = threading.Event()
threading.Thread(target=sampler, args=(stop,), daemon=True).start()
started = time.time()
try:
    with concurrent.futures.ThreadPoolExecutor(a.sessions) as pool:
        results = list(pool.map(session, range(a.sessions)))
finally:
    stop.set()
    for w in watchers:
        w.terminate()
time.sleep(2)
peaks = {}
rows = (ROOT / "evidence" / f"{a.tag}-{phase}-memory.jsonl").read_text()
peaks["spark"] = round(max(json.loads(l)["used_bytes"] for l in rows.splitlines() if l.strip()) / 2**30, 2)
record = dict(tag=a.tag, sessions=a.sessions, prompt_tokens=a.prompt_tokens, wall_s=round(time.time() - started, 1),
              preemptions=metric("vllm:num_preemptions_total") - preempt0, peak_used_gib=peaks,
              prefix_cache_hit_rate=round((metric("vllm:prefix_cache_hits_total") - hits0)
                                          / max(1.0, metric("vllm:prefix_cache_queries_total") - queries0), 4),
              peak_running=max((x["running"] for x in samples), default=0),
              peak_kv_usage=round(max((x["kv"] for x in samples), default=0), 4), results=results)
record["passed"] = bool(record["preemptions"] == 0 and all(r["retrieved"] == r["wanted"] for r in results)
                        and max(peaks.values()) < a.limit_gib
                        and (not a.followup or (record["peak_running"] >= a.sessions
                                                and record["prefix_cache_hit_rate"] >= 0.9)))
dest.write_text(json.dumps(record, indent=1))
print(json.dumps({k: v for k, v in record.items() if k != "results"}), flush=True)
print(json.dumps([{k: r[k] for k in ("session", "prompt_tokens", "wall_s", "completion_tokens", "retrieved")}
                  for r in results]), flush=True)
