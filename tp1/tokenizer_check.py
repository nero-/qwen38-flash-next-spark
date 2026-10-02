#!/usr/bin/env python3
"""Measure tokenizer cost on the running server, and record token IDs for identity checks.

tokenizer_check.py TAG [--tokens 250000] [--repeats 3]
- /tokenize wall time on a long text (CPU tokenizer only, no GPU work)
- cold TTFT for the same text sent as a string (tokenization + prefill), unique prefix per repeat
- SHA-256 of the token IDs for the long text and for the frozen quality corpus, so two
  arms can be compared for byte-identical tokenization
"""
import argparse, hashlib, json, statistics, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL = "qwen3.8-flash-next-4p89bpw"
p = argparse.ArgumentParser()
p.add_argument("tag")
p.add_argument("--tokens", type=int, default=250000)
p.add_argument("--repeats", type=int, default=3)
a = p.parse_args()
dest = ROOT / "evidence" / f"{a.tag}-tokenizer.json"
assert not dest.exists(), dest


def post(path, payload, timeout=1800):
    req = urllib.request.Request("http://127.0.0.1:8000" + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def tokenize(text):
    started = time.perf_counter()
    ids = post("/tokenize", {"model": MODEL, "prompt": text, "add_special_tokens": False})["tokens"]
    return ids, time.perf_counter() - started


def ttft(text):
    body = json.dumps(dict(model=MODEL, prompt=text, max_tokens=1, temperature=0, stream=True)).encode()
    req = urllib.request.Request("http://127.0.0.1:8000/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            if line.startswith(b"data: ") and b'"text"' in line:
                return time.perf_counter() - started
    raise RuntimeError("no token streamed")


# Mixed prose, code, numbers and non-ASCII so the tokenizer sees varied merges.
corpus = json.loads((ROOT / "quality-data/corpus-v1.json").read_text())
base = corpus["wiki"][:400000] + "\n" + corpus["code"] + "\nÜnïcødé — 数据 データ 🚀 ±½ 2026-09-29\n"
ids, _ = tokenize(base)
text = base * (a.tokens // len(ids) + 1)
long_ids, _ = tokenize(text)
# Trim to the target length on a character boundary proportional to token count.
text = text[: int(len(text) * a.tokens / len(long_ids))]

rows = []
for i in range(a.repeats):
    unique = f"Run {a.tag} repeat {i} nonce {time.time_ns()}.\n" + text
    ids, tok_s = tokenize(unique)
    rows.append(dict(repeat=i, prompt_tokens=len(ids), tokenize_s=round(tok_s, 3), ttft_s=round(ttft(unique), 3)))
    print(json.dumps(rows[-1]), flush=True)

def detokenize(ids):
    return post("/detokenize", {"model": MODEL, "tokens": ids})["prompt"]


def stream_check(prompt):
    """Streamed text must equal /detokenize of the streamed token IDs (DecodeStream path)."""
    body = json.dumps(dict(model=MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=400,
                           temperature=0, stream=True, return_token_ids=True,
                           chat_template_kwargs={"enable_thinking": False})).encode()
    req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    text, ids = "", []
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            for choice in json.loads(line[6:]).get("choices", []):
                text += (choice.get("delta") or {}).get("content") or ""
                ids += choice.get("token_ids") or []
    decoded = detokenize(ids)
    # The final <|im_end|> is a token but not streamed content.
    for special in ("<|im_end|>", "<|endoftext|>"):
        decoded = decoded.removesuffix(special)
    return dict(tokens=len(ids), match=decoded.strip() == text.strip(), text_sha256=hashlib.sha256(text.encode()).hexdigest())


fixed_ids, _ = tokenize(text)
corpus_hashes, detok_hashes = {}, {}
for name in ("wiki", "code"):
    cids, _ = tokenize(corpus[name])
    corpus_hashes[name] = hashlib.sha256(json.dumps(cids).encode()).hexdigest()
    detok_hashes[name] = hashlib.sha256(detokenize(cids[:50000]).encode()).hexdigest()
streams = {name: stream_check(prompt) for name, prompt in {
    "code": "Write a Python function that parses ISO-8601 timestamps with time zones, with tests.",
    "cjk": "用中文和日本語各写一段关于东京塔的介绍，并加入一些表情符号 🚀✨。",
    "mixed": "List 20 Unicode characters with their code points, e.g. é U+00E9, then explain UTF-8 encoding.",
}.items()}
record = dict(tag=a.tag, rows=rows,
              tokenize_s_median=statistics.median(r["tokenize_s"] for r in rows),
              ttft_s_median=statistics.median(r["ttft_s"] for r in rows),
              long_text_ids_sha256=hashlib.sha256(json.dumps(fixed_ids).encode()).hexdigest(),
              long_text_tokens=len(fixed_ids), corpus_ids_sha256=corpus_hashes,
              corpus_detok_sha256=detok_hashes, stream_detok=streams)
dest.write_text(json.dumps(record, indent=1))
print(json.dumps({k: v for k, v in record.items() if k != "rows"}), flush=True)
