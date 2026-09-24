import json, subprocess, urllib.request
from pathlib import Path
root=Path.home()/'builds/qwen-tp2'
assert (root/'selected-profile.txt').read_text().strip()=='hc-adaptive'
metrics=urllib.request.urlopen('http://localhost:8000/metrics').read().decode()
assert all(float(l.split()[-1])==0 for l in metrics.splitlines() if l.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{')))
for i in (1,2):
 tag=f'hc-adaptive-repeat32-c8-{i}'
 dest=root/'evidence'/f'{tag}.json';log=dest.with_suffix('.log')
 assert not dest.exists() and not log.exists()
 cmd=[str(root/'.venv/bin/python'),'-u',str(root/'llm-inference-bench/llm_decode_bench.py'),'--host','127.0.0.1','--port','8000','--model','qwen3.8-flash-next-4p89bpw','--no-hw-monitor','--display-mode','plain','--no-resume','--skip-prefill','--contexts','32768','--concurrency','8','--duration','30','--max-tokens','2048','--output',str(dest)]
 dest.with_suffix('.command.json').write_text(json.dumps(cmd,indent=2))
 with log.open('x') as f: subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=600)
 j=json.loads(dest.read_text()); print(tag,j.get('summary_table'),flush=True)
