"""Summarize the final matched cable matrices without changing the LIL harness."""
import json
from pathlib import Path
folder = Path(__file__).resolve().parents[2] / 'results/tp2'
a, b = [json.loads((folder / f'cable{c}-final-matrix.json').read_text()) for c in (1, 2)]
rows = []
for old, new in zip(a['results'], b['results']):
    assert (old['context_tokens'], old['concurrency']) == (new['context_tokens'], new['concurrency'])
    for r in (old, new):
        assert not any(r[k] for k in ('num_errors', 'underfilled', 'warmup_timed_out', 'capacity_limited', 'loop_detected'))
    row = {'context': old['context_tokens'], 'concurrency': old['concurrency']}
    for k in ('aggregate_tps', 'server_steps_per_s', 'server_accept_len_effective'):
        row[k] = {'one_cable': old[k], 'two_cables': new[k], 'change_pct': (new[k] / old[k] - 1) * 100}
    rows.append(row)
result = {'decode': rows, 'prefill_one_cable': a['prefill'], 'prefill_two_cables': b['prefill']}
(folder / 'cables-model-summary.json').write_text(json.dumps(result, indent=2) + '\n')
for r in rows:
    print(r['context'], r['concurrency'], *(f"{k}: {v['one_cable']:.3f} -> {v['two_cables']:.3f} ({v['change_pct']:+.2f}%)" for k, v in r.items() if isinstance(v, dict)))
print('Prefill:', a['prefill'], '->', b['prefill'])
