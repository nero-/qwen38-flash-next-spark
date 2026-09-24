"""Bounded HC A/B/A follow-up, using the unchanged official LIL CLI."""
import json,subprocess,urllib.request,time
from pathlib import Path
root=Path.home()/'builds/qwen-tp2';out=root/'evidence'
def switch(profile):
 subprocess.run(['python3',str(root/'select_profile.py'),profile],check=True,timeout=180)
 subprocess.run(['bash',str(root/'cluster.sh'),'wait'],check=True,timeout=1250)
def run(profile,i):
 assert (root/'selected-profile.txt').read_text().strip()==profile
 tag=f'hc-ab-{profile}-{i}';dest=out/f'{tag}.json';log=out/f'{tag}.log'
 assert not dest.exists() and not log.exists()
 cmd=[str(root/'.venv/bin/python'),'-u',str(root/'llm-inference-bench/llm_decode_bench.py'),'--host','127.0.0.1','--port','8000','--model','qwen3.8-flash-next-4p89bpw','--no-hw-monitor','--display-mode','plain','--no-resume','--skip-prefill','--contexts','32768','--concurrency','8','--duration','30','--max-tokens','2048','--output',str(dest)]
 (out/f'{tag}-command.json').write_text(json.dumps(cmd,indent=2))
 (out/f'{tag}-container.json').write_text(subprocess.check_output(['docker','inspect','qwen-tp2'],text=True))
 with log.open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=600)
 j=json.loads(dest.read_text());r=j['results'][0]
 assert not r.get('num_errors') and not r.get('underfilled') and not r.get('timeout_reason')
 print(json.dumps({'tag':tag,'summary':j['summary_table'],'steps_per_s':r['server_steps_per_s'],'accept_length':r['server_spec_accept_length']}),flush=True)
assert (root/'selected-profile.txt').read_text().strip()=='hc-adaptive'
try:
 switch('hc')
 run('hc',1);run('hc',2)
finally:
 switch('hc-adaptive')
run('hc-adaptive',1)
print('Comparison complete; adaptive service restored.',flush=True)
