#!/usr/bin/env python3
"""Run a fixed evidence sequence per profile arm, then restore the selected default.

For each arm: coordinated profile switch (select_profile.py), readiness, smoke,
unmodified-LIL matrix + odd-concurrency cells (bench.py), and paired quality
evidence (quality_eval.py). A memory watcher runs for the whole arm. The previous
selection is restored even if an arm fails; failures are recorded, not retried.
"""
import argparse, json, subprocess, sys, time
from pathlib import Path

ROOT = Path.home() / "builds/qwen-tp2"
p = argparse.ArgumentParser()
p.add_argument("arms", nargs="+", help="profile names, e.g. hc-adaptive+cg4")
p.add_argument("--prefix", required=True, help="evidence tag prefix for this campaign")
p.add_argument("--cases", nargs="+", default=["matrix", "odd"])
p.add_argument("--quality", choices=["full", "mmlu", "none"], default="full")
p.add_argument("--restore", default="hc-adaptive")
a = p.parse_args()


def log(**kw):
    print(json.dumps(dict(time=time.strftime("%H:%M:%S"), **kw)), flush=True)


def current():
    return (ROOT / "selected-profile.txt").read_text().strip()


def switch(profile):
    if current() == profile and subprocess.run(["curl", "-fsS", "-m", "3", "http://127.0.0.1:8000/health"],
                                               capture_output=True).returncode == 0:
        return
    subprocess.run(["python3", str(ROOT / "select_profile.py"), profile], check=True, timeout=2400)


def smoke():
    payload = dict(model="qwen3.8-flash-next-4p89bpw", temperature=0, max_tokens=32,
                   messages=[{"role": "user", "content": "What is 17 times 23? Answer only the number."}],
                   chat_template_kwargs={"enable_thinking": False})
    out = subprocess.check_output(["curl", "-fsS", "-m", "180", "-H", "Content-Type: application/json",
                                   "-d", json.dumps(payload), "http://127.0.0.1:8000/v1/chat/completions"])
    answer = json.loads(out)["choices"][0]["message"]["content"].strip()
    assert answer == "391", answer


failures = []
try:
    for arm in a.arms:
        tag = f"{a.prefix}-{arm.replace('+', '_')}"
        watcher = subprocess.Popen(["python3", str(ROOT / "watch_memory.py"), "--tag", tag,
                                    "--seconds", "7200", "--guard", "--max-used-gib", "118"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            started = time.time()
            log(arm=arm, step="switch")
            switch(arm)
            log(arm=arm, step="ready", startup_s=round(time.time() - started))
            smoke()
            if a.quality == "full":
                subprocess.run(["python3", str(ROOT / "determinism_probe.py"), tag, "--lengths", "960", "2048"],
                               check=True, timeout=900)
            subprocess.run(["python3", str(ROOT / "bench.py"), tag, "--cases", *a.cases], check=True, timeout=5400)
            if a.quality != "none":
                subprocess.run(["python3", str(ROOT / "quality_eval.py"), "run" if a.quality == "full" else "mmlu", tag],
                               check=True, timeout=3600)
            log(arm=arm, step="done", tag=tag)
        except Exception as exc:  # record and continue with the next arm
            failures.append(arm)
            log(arm=arm, step="FAILED", error=repr(exc))
        finally:
            watcher.terminate()
finally:
    if current() != a.restore or failures:
        log(step="restore", profile=a.restore)
        # A failed switch has already started its own rollback; let it become ready
        # (idle metrics are required) before switching to the restore profile.
        subprocess.run(["bash", str(ROOT / "cluster.sh"), "wait"], check=False, timeout=1300)
        if current() != a.restore:
            subprocess.run(["python3", str(ROOT / "select_profile.py"), a.restore], check=False, timeout=2400)
    log(step="finished", failures=failures)
sys.exit(1 if failures else 0)
