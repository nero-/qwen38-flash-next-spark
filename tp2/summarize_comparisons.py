"""Summarize saved LIL comparisons without changing their measurements."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('directory', type=Path)
a = p.parse_args()
for path in sorted(a.directory.glob('*-matrix.json')):
    d = json.loads(path.read_text())
    print(json.dumps({'tag': path.stem.removesuffix('-matrix'),
        'decode': d.get('summary_table'),
        'speculation': [{k: row.get(k) for k in ('context_tokens', 'concurrency',
            'server_steps_per_s', 'server_spec_accept_length', 'num_errors')}
            for row in d.get('results', [])],
        'prefill': {k: {field: v.get(field) for field in
            ('tok_per_sec', 'prompt_tokens', 'samples')} for k, v in d.get('prefill', {}).items()}}))
for path in sorted(a.directory.glob('*-tetris.json')):
    d = json.loads(path.read_text()).get('selected_summary', {})
    print(json.dumps({'tag': path.stem, 'tok_s': d.get('gen_tok_s', {}).get('avg'),
        'elapsed_s': d.get('elapsed', {}).get('avg'),
        'hit_max_tokens': d.get('hit_max_tokens'), 'errors': d.get('errors')}))
