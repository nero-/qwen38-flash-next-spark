"""Capture a bounded engine trace; this is not a throughput benchmark."""
import argparse
import json
import time
import urllib.request
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('tag')
a = p.parse_args()
root = Path.home() / 'builds/qwen-tp2'
out = root / 'evidence' / f'{a.tag}-trace-request.json'
if out.exists():
    raise SystemExit(f'Refusing to overwrite {out}')

def post(path, payload=None):
    req = urllib.request.Request('http://127.0.0.1:8000/' + path,
        data=json.dumps(payload or {}).encode(),
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=300) as response:
        body = response.read()
        return json.loads(body) if body else None

payload = dict(model='qwen3.8-flash-next-4p89bpw', temperature=0, seed=42,
    max_tokens=128, chat_template_kwargs={'enable_thinking': False},
    messages=[dict(role='user', content=(
        'Trace document.\n' + 'The warehouse contains shelves and numbered boxes.\n' * 800
        + '\nWrite a Python function that totals quantities by product name. Include an example.'))])
started = time.time()
post('start_profile')
try:
    result = post('v1/chat/completions', payload)
finally:
    post('stop_profile')
out.write_text(json.dumps(dict(wall_s=time.time() - started, response=result,
    scope='Bounded profiling request; profiler overhead makes its timing unsuitable for throughput comparison.'), indent=2))
print(out)
