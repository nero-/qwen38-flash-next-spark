#!/usr/bin/env python3
"""Run a fixed evidence sequence per profile arm on one Spark, then restore the prior selection.

campaign.py --prefix P [--keep] ARM[:cases[:q]] ...
  ARM    profile, e.g. tp1+cg4
  cases  comma-separated bench.py cases (default quick)
  q      also run quality_eval.py (NLL + GSM8K + MMLU-Pro); "m" runs MMLU-Pro only
Each arm: switch -> smoke -> bench -> optional quality. Tags are P-<profile with + as _>.
Progress is appended to evidence/<P>-campaign.jsonl. --keep leaves the last arm serving
instead of restoring the profile selected at launch.
"""
import argparse, json, subprocess, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument("--prefix", required=True)
p.add_argument("--keep", action="store_true")
p.add_argument("arms", nargs="+")
a = p.parse_args()
log_path = ROOT / "evidence" / f"{a.prefix}-campaign.jsonl"
log_path.parent.mkdir(exist_ok=True)


def log(**kw):
    kw["time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with log_path.open("a") as out:
        out.write(json.dumps(kw) + "\n")
    print(json.dumps(kw), flush=True)


def run(*cmd):
    return subprocess.run([str(c) for c in cmd], cwd=ROOT, check=True)


selection = ROOT / "selected-profile.txt"
original = selection.read_text().strip() if selection.exists() else None
try:
    for spec in a.arms:
        profile, cases, quality = (spec.split(":") + ["quick", ""])[:3]
        tag = f"{a.prefix}-{profile.replace('+', '_')}"
        log(arm=profile, step="switch")
        try:
            run("python3", "select_profile.py", profile)
            run("python3", "smoke.py")
            run("python3", "bench.py", tag, "--cases", *cases.split(","))
            if quality == "q":
                run("python3", "quality_eval.py", "run", tag)
            elif quality == "m":
                run("python3", "quality_eval.py", "mmlu", tag)
            log(arm=profile, step="done", tag=tag)
        except subprocess.CalledProcessError as exc:
            log(arm=profile, step="failed", error=str(exc))
finally:
    if original and not a.keep and (not selection.exists() or selection.read_text().strip() != original):
        log(step="restore", profile=original)
        subprocess.run(["python3", "select_profile.py", original], cwd=ROOT, check=False)
    log(step="campaign-complete")
