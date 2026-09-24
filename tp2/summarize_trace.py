"""Report GPU kernel durations, without treating overlaps as wall-clock shares."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('trace', type=Path)
a = p.parse_args()
reader = gzip.open if a.trace.suffix == '.gz' else open
with reader(a.trace, 'rt') as f:
    data = json.load(f)
totals = defaultdict(float)
counts = defaultdict(int)
phases = [e for e in data['traceEvents'] if e.get('cat') == 'user_annotation'
    and e.get('name', '').startswith('execute_')]
runtime = {e['args']['correlation']: e for e in data['traceEvents']
    if e.get('cat') == 'cuda_runtime' and 'correlation' in e.get('args', {})}
phase_totals = defaultdict(lambda: defaultdict(float))
for event in data['traceEvents']:
    if event.get('ph') == 'X' and event.get('cat') == 'kernel':
        name = event['name']
        totals[name] += event.get('dur', 0)
        counts[name] += 1
        launch = runtime.get(event.get('args', {}).get('correlation'))
        phase = next((p['name'] for p in phases if launch and
            p['ts'] <= launch['ts'] <= p['ts'] + p['dur']), 'outside_execute_annotation')
        phase_totals[phase][name] += event.get('dur', 0) / 1000
total = sum(totals.values())
record = dict(trace=str(a.trace), total_kernel_ms=total / 1000,
    execution_phase_kernel_ms=phase_totals,
    scope='Summed GPU kernel durations; overlapping streams are double-counted. These are not critical-path or end-to-end wall-time fractions.',
    kernels=[dict(name=name, calls=counts[name], ms=duration / 1000,
        fraction_of_summed_kernel_time=duration / total if total else 0)
        for name, duration in sorted(totals.items(), key=lambda item: -item[1])])
print(json.dumps(record, indent=2))
