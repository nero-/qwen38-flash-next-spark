"""Run fixed functional checks, a screening matrix, and the matched TP4 prompt."""
import argparse,json,subprocess,time,urllib.request
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('tag');p.add_argument('--full',action='store_true');p.add_argument('--quality-only',action='store_true');p.add_argument('--cases',nargs='+',choices=['quality','matrix','tetris'],default=['quality','matrix','tetris']);p.add_argument('--tetris-max-tokens',type=int,default=8192);p.add_argument('--tetris-runs',type=int,default=1);p.add_argument('--thinking',action='store_true');a=p.parse_args()
root=Path.home()/'builds/qwen-tp2';out=root/'evidence';py=str(root/'.venv/bin/python')
if 'quality' in a.cases: subprocess.run([py,str(root/'quality.py'),a.tag],check=True)
if a.quality_only:raise SystemExit()
common=[py,'-u',str(root/'llm-inference-bench/llm_decode_bench.py'),'--host','127.0.0.1','--port','8000','--model','qwen3.8-flash-next-4p89bpw','--no-hw-monitor','--display-mode','plain','--no-resume']
cases={'matrix': ['--contexts','8192,32768,65536' if a.full else '8192,32768','--concurrency','1,8' if a.full else '1','--duration','30','--max-tokens','2048','--standalone-prefill','--prefill-contexts','8k,32k,64k' if a.full else '8k,64k'],
 'tetris':['--completion-stats','--prompt-file',str(root/'tetris-tp4-prompt.txt'),'--profile-concurrency','1','--profile-runs',str(a.tetris_runs),'--max-tokens',str(a.tetris_max_tokens),'--reasoning-effort','none','--completion-stats-temperature','1','--completion-stats-seed','41','--completion-stats-correct-regex','','--completion-stats-save-text','--completion-stats-no-prefill-scout']}
if a.thinking:
    i=cases['tetris'].index('--reasoning-effort');del cases['tetris'][i:i+2]
for name,args in cases.items():
    if name not in a.cases: continue
    dest=out/f'{a.tag}-{name}.json';log=out/f'{a.tag}-{name}.log'
    if dest.exists() or log.exists():raise RuntimeError(f'Already exists: {dest}')
    cmd=common+args+['--output',str(dest)]
    (out/f'{a.tag}-{name}-command.json').write_text(json.dumps(cmd,indent=2))
    for rank in (0,1):
        remote=['ssh','-o','BatchMode=yes','10.100.168.1'] if rank else []
        info=json.loads(subprocess.check_output(remote+['docker','inspect','qwen-tp2']))[0]
        assert info['State']['Running']
        (out/f'{a.tag}-{name}-rank{rank}.json').write_text(json.dumps(info,indent=2))
    with log.open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=1200)
    d=json.loads(dest.read_text());summary=d.get('selected_summary') or d.get('summary_table')
    print(json.dumps(dict(tag=a.tag,case=name,summary=summary)),flush=True)
