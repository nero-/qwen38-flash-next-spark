"""Keep the two selected completed LIL matrices and their associated evidence."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("root", type=Path)
parser.add_argument("--apply", action="store_true")
parser.add_argument("--tags", nargs=2, default=["kk-b12x-cu133", "eugr-20260923-r2"])
args = parser.parse_args()
root = args.root.resolve()
allowed = {Path.home() / "bench",
           Path("/Users/jg/Agent/Builds/qwen38-flash-next-spark/results")}
if root not in allowed:
    raise RuntimeError(f"Unexpected benchmark root: {root}")
trials = root / "tp1-opt-20260922"
tags = args.tags
if len(set(tags)) != 2 or any("/" in tag or not tag for tag in tags):
    raise ValueError("Select two distinct run tags")
for tag in tags:
    report = json.loads((trials / f"{tag}-matrix.json").read_text())
    cells = report["results"]
    assert {(c["context_tokens"], c["concurrency"]) for c in cells} == {
        (ctx, conc) for ctx in (8192, 32768, 65536) for conc in (1, 8)}
    assert all(c["num_errors"] == 0 and not c["failure_reason"] for c in cells)
    assert all(c.get("effective_concurrency") == c["concurrency"] for c in cells)
    assert not any(c.get(flag) for c in cells for flag in (
        "underfilled", "warmup_timed_out", "capacity_limited", "loop_detected"))

baseline_files = set()
if "instant-cu133-lil" in tags:
    baseline_files.update(f"instant-cu133-{suffix}" for suffix in (
        "launch.json", "launch.log", "server.log", "memory.jsonl"))
    baseline_files.add("loader-cu133-correctness.log")
keep = {root / "README.md"}
for path in trials.iterdir():
    if path.name.endswith(".resume.json"):
        continue
    if (any(path.name.startswith(tag + "-") for tag in tags)
            or path.name in baseline_files
            or path.name == "lil-matrix-summary.json"):
        keep.add(path)
files = [p for p in root.rglob("*") if p.is_file() or p.is_symlink()]
remove = sorted(p for p in files if p not in keep)
plan = {"root": str(root), "retained_runs": tags,
        "retained_files": sorted(str(p.relative_to(root)) for p in keep if p.exists()),
        "removed_files": [str(p.relative_to(root)) for p in remove],
        "removed_bytes": sum(p.lstat().st_size for p in remove), "applied": args.apply}
print(json.dumps(plan, indent=2), flush=True)
if args.apply:
    for path in remove:
        path.unlink()
    for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_dir() and not path.is_symlink() and not any(path.iterdir()):
            path.rmdir()
