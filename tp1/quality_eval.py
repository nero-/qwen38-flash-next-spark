#!/usr/bin/env python3
"""Paired quality evidence for serving arms: teacher-forced NLL and greedy GSM8K.

run TAG      score the running server on the frozen corpus (quality-data/corpus-v1.json), then MMLU-Pro
mmlu TAG     only the 1,000-question MMLU-Pro subset (quality-data/mmlupro-v1.json)
compare A B  paired differences between two saved runs

NLL uses identical token IDs for every arm (same tokenizer), so per-token
log-probabilities can be compared directly. It exercises the prefill path.
GSM8K is greedy with thinking disabled and exercises decode, where recurrent
state precision accumulates. Neither is a full-distribution KLD against the
BF16 teacher; small differences must be read against the repeat-run floor.
"""
import concurrent.futures, hashlib, json, math, re, sys, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "evidence"
CORPUS = ROOT / "quality-data/corpus-v1.json"
CORPUS_SHA256 = "2e42176fbed761c0b22185b0280189eb433676c23f6d6f8c2914b51b491baf7a"
MODEL = "qwen3.8-flash-next-4p89bpw"
CHUNK, CHUNKS = 1024, {"wiki": 64, "code": 48}


def post(path, payload, timeout=600):
    req = urllib.request.Request("http://127.0.0.1:8000" + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def nll(corpus):
    out = {}
    for name, count in CHUNKS.items():
        ids = post("/tokenize", {"model": MODEL, "prompt": corpus[name], "add_special_tokens": False})["tokens"]
        assert len(ids) >= CHUNK * count, (name, len(ids))
        rows = []
        for i in range(count):
            chunk = ids[i * CHUNK:(i + 1) * CHUNK]
            data = post("/v1/completions", dict(model=MODEL, prompt=chunk, max_tokens=1, temperature=0,
                                                 echo=True, logprobs=1))
            lp = data["choices"][0]["logprobs"]["token_logprobs"][:CHUNK]
            assert lp[0] is None and all(x is not None for x in lp[1:])
            rows.append(lp[1:])
        flat = [x for r in rows for x in r]
        out[name] = dict(tokens=len(flat), mean_nll=-sum(flat) / len(flat), token_logprobs=rows,
                         ids_sha256=hashlib.sha256(json.dumps(ids[:CHUNK * count]).encode()).hexdigest())
    return out


def gsm8k(problems):
    def solve(p):
        prompt = p["question"] + "\nSolve step by step, then give the final answer on the last line as 'Answer: <number>'."
        data = post("/v1/chat/completions", dict(model=MODEL, messages=[{"role": "user", "content": prompt}],
                                                  temperature=0, max_tokens=1024, seed=0,
                                                  chat_template_kwargs={"enable_thinking": False}))
        text = data["choices"][0]["message"]["content"] or ""
        m = re.findall(r"Answer:\s*\$?\s*(-?[\d,]*\.?\d+)", text) or re.findall(r"-?[\d,]*\.?\d+", text)
        pred = m[-1].replace(",", "").rstrip(".") if m else None
        gold = p["answer"].split("####")[-1].strip().replace(",", "")
        try:
            ok = pred is not None and math.isclose(float(pred), float(gold))
        except ValueError:
            ok = False
        return dict(correct=ok, pred=pred, gold=gold, text=text,
                    finish=data["choices"][0]["finish_reason"], tokens=data["usage"]["completion_tokens"])
    with concurrent.futures.ThreadPoolExecutor(16) as pool:
        rows = list(pool.map(solve, problems))
    return dict(n=len(rows), accuracy=sum(r["correct"] for r in rows) / len(rows),
                truncated=sum(r["finish"] == "length" for r in rows), rows=rows)


MMLU = ROOT / "quality-data/mmlupro-v1.json"
MMLU_SHA256 = "169665a78e21f0acc227df0a21f1cf4f0cdd119f5bc211468c7973e979f152bd"


def mmlu(tag):
    """Direct-answer MMLU-Pro (1,000 stratified questions, 10 options, thinking off)."""
    dest = EVIDENCE / f"{tag}-mmlu.json"
    if dest.exists():
        raise SystemExit(f"Refusing to overwrite {dest}")
    raw = MMLU.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == MMLU_SHA256, "MMLU subset drift"
    letters = "ABCDEFGHIJ"

    def ask(q):
        options = "\n".join(f"{letters[i]}. {o}" for i, o in enumerate(q["options"]))
        prompt = (f"Answer the following multiple choice question. Reply with only the letter of the correct option.\n\n"
                  f"Question: {q['question']}\nOptions:\n{options}")
        data = post("/v1/chat/completions", dict(model=MODEL, messages=[{"role": "user", "content": prompt}],
                                                  temperature=0, max_tokens=8, seed=0,
                                                  chat_template_kwargs={"enable_thinking": False}))
        text = (data["choices"][0]["message"]["content"] or "").strip()
        m = re.match(r"\(?([A-J])\b", text)
        return dict(id=q["id"], category=q["category"], pred=m.group(1) if m else None, gold=q["answer"],
                    correct=bool(m) and m.group(1) == q["answer"])
    with concurrent.futures.ThreadPoolExecutor(16) as pool:
        rows = list(pool.map(ask, json.loads(raw)))
    record = dict(tag=tag, subset_sha256=MMLU_SHA256, n=len(rows),
                  accuracy=sum(r["correct"] for r in rows) / len(rows),
                  unparsed=sum(r["pred"] is None for r in rows), rows=rows)
    dest.write_text(json.dumps(record))
    print(json.dumps({k: v for k, v in record.items() if k != "rows"}), flush=True)


def run(tag):
    dest = EVIDENCE / f"{tag}-qeval.json"
    if dest.exists():
        raise SystemExit(f"Refusing to overwrite {dest}")
    raw = CORPUS.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == CORPUS_SHA256, "corpus drift"
    corpus = json.loads(raw)
    started = time.time()
    record = dict(tag=tag, corpus_sha256=CORPUS_SHA256, nll=nll(corpus))
    t_nll = time.time()
    record["gsm8k"] = gsm8k(corpus["gsm8k"])
    record.update(nll_s=t_nll - started, gsm8k_s=time.time() - t_nll)
    dest.write_text(json.dumps(record))
    print(json.dumps(summary(record)), flush=True)
    mmlu(tag)


def summary(r):
    return dict(tag=r["tag"], **{f"nll_{k}": round(v["mean_nll"], 5) for k, v in r["nll"].items()},
                gsm8k_acc=r["gsm8k"]["accuracy"], gsm8k_truncated=r["gsm8k"]["truncated"])


def compare(a_tag, b_tag):
    out = {}
    qa, qb = (EVIDENCE / f"{t}-qeval.json" for t in (a_tag, b_tag))
    # Screening arms may have only MMLU evidence; compare whatever both arms have.
    a, b = (json.loads(f.read_text()) if f.exists() else None for f in (qa, qb))
    if a and b:
        out.update(a=summary(a), b=summary(b))
    for name in (a["nll"] if a and b else []):
        assert a["nll"][name]["ids_sha256"] == b["nll"][name]["ids_sha256"]
        da = [x for r in a["nll"][name]["token_logprobs"] for x in r]
        db = [x for r in b["nll"][name]["token_logprobs"] for x in r]
        diffs = [y - x for x, y in zip(da, db)]
        mean = sum(diffs) / len(diffs)
        # Chunk-level standard error: tokens within a chunk are not independent.
        chunk = [sum(y - x for x, y in zip(ra, rb)) / len(ra)
                 for ra, rb in zip(a["nll"][name]["token_logprobs"], b["nll"][name]["token_logprobs"])]
        cm = sum(chunk) / len(chunk)
        se = (sum((c - cm) ** 2 for c in chunk) / (len(chunk) - 1) / len(chunk)) ** 0.5
        out[f"{name}_delta_nll_b_minus_a"] = round(-mean, 5)
        out[f"{name}_delta_nll_se"] = round(se, 5)
        out[f"{name}_mean_abs_logprob_diff"] = round(sum(map(abs, diffs)) / len(diffs), 5)
    if a and b:
        ra, rb = a["gsm8k"]["rows"], b["gsm8k"]["rows"]
        out["gsm8k_identical_text"] = sum(x["text"] == y["text"] for x, y in zip(ra, rb)) / len(ra)
        out["gsm8k_only_a_correct"] = sum(x["correct"] and not y["correct"] for x, y in zip(ra, rb))
        out["gsm8k_only_b_correct"] = sum(y["correct"] and not x["correct"] for x, y in zip(ra, rb))
    ma, mb = (EVIDENCE / f"{t}-mmlu.json" for t in (a_tag, b_tag))
    if ma.exists() and mb.exists():
        xa, xb = (json.loads(f.read_text()) for f in (ma, mb))
        pairs = list(zip(xa["rows"], xb["rows"]))
        assert all(x["id"] == y["id"] for x, y in pairs)
        only_a = sum(x["correct"] and not y["correct"] for x, y in pairs)
        only_b = sum(y["correct"] and not x["correct"] for x, y in pairs)
        # Exact two-sided McNemar test on the discordant pairs.
        n, k = only_a + only_b, min(only_a, only_b)
        p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
        out.update(mmlu_a=xa["accuracy"], mmlu_b=xb["accuracy"], mmlu_only_a_correct=only_a,
                   mmlu_only_b_correct=only_b, mmlu_mcnemar_p=round(p, 4))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    if sys.argv[1:2] == ["run"] and len(sys.argv) == 3:
        run(sys.argv[2])
    elif sys.argv[1:2] == ["mmlu"] and len(sys.argv) == 3:
        mmlu(sys.argv[2])
    elif sys.argv[1:2] == ["compare"] and len(sys.argv) == 4:
        compare(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit(__doc__)
