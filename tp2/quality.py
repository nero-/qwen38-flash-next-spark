"""Small regression suite and teacher-forced likelihood probe; not a KLD eval."""
import argparse, json, math, re, time, urllib.request
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('tag');a=p.parse_args()
root=Path.home()/'builds/qwen-tp2/evidence';out=root/f'{a.tag}-quality.json'
if out.exists(): raise SystemExit(f'Refusing to overwrite {out}')
model='qwen3.8-flash-next-4p89bpw'
def call(path,payload):
    req=urllib.request.Request('http://127.0.0.1:8000/v1/'+path,data=json.dumps({'model':model,**payload}).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=240) as r:return json.load(r)
cases=[
 ('arithmetic','What is 17 times 23? Return only the number.',lambda s:s.strip()=='391'),
 ('sort','Sort these numbers ascending and return only a JSON array: 7, -3, 12, 0, 7.',lambda s:json.loads(s)==[-3,0,7,7,12]),
 ('unicode','Return exactly this text and nothing else: café — 東京 — 🐢',lambda s:s.strip()=='café — 東京 — 🐢'),
 ('json','Return only a JSON object with keys name and count, values Ada and integer 4.',lambda s:json.loads(s)=={'name':'Ada','count':4}),
 ('logic','There are three boxes: A holds a red ball, B holds a blue ball, C is empty. Swap A and B, then move the ball from A to C. Which box now holds the red ball? Return only A, B, or C.',lambda s:s.strip()=='B'),
 ('code','What does Python print for sum(i*i for i in range(5))? Return only the number.',lambda s:s.strip()=='30'),
]
results=[]
for name,prompt,check in cases:
    data=call('chat/completions',dict(messages=[{'role':'user','content':prompt}],temperature=0,seed=42,max_tokens=128,chat_template_kwargs={'enable_thinking':False}))
    text=data['choices'][0]['message']['content']
    try:passed=bool(check(text))
    except (ValueError,TypeError):passed=False
    results.append(dict(name=name,passed=passed,response=data))
tools=[{'type':'function','function':{'name':'get_weather','description':'Get weather for a city','parameters':{'type':'object','properties':{'city':{'type':'string'}},'required':['city']}}}]
data=call('chat/completions',dict(messages=[{'role':'user','content':'Use get_weather to look up the weather in Honolulu.'}],tools=tools,tool_choice='auto',temperature=0,seed=42,max_tokens=128,chat_template_kwargs={'enable_thinking':False}))
tc=data['choices'][0]['message'].get('tool_calls') or []
passed=bool(tc) and tc[0]['function']['name']=='get_weather'
try:passed=passed and json.loads(tc[0]['function']['arguments']).get('city','').lower()=='honolulu'
except (KeyError,ValueError):passed=False
results.append(dict(name='tool_call',passed=passed,response=data))
texts=[
 'Question: What is 17 times 23?\nAnswer: 391.\n',
 'def is_prime(n):\n    if n < 2:\n        return False\n    for d in range(2, int(n ** 0.5) + 1):\n        if n % d == 0:\n            return False\n    return True\n',
 'A falling Tetris piece moves down until a collision prevents the next move. It is then merged into the board. Full rows are removed and the rows above fall into the empty space.\n',
 'The results are {"name":"Ada","count":4,"items":[-3,0,7,7,12]}.\n',
 'For a right triangle, the square of the hypotenuse equals the sum of the squares of the two other sides.\n',
 'A hash table maps keys to values. Collisions can be resolved using chaining or open addressing. Resizing maintains an appropriate load factor.\n',
 'Bonjour, le café est ouvert. 日本語の文章と英語の文章を比較します。\n',
 'function rotate(matrix) { return matrix[0].map((_, i) => matrix.map(row => row[i]).reverse()); }\n',
]
likelihood=[]
for text in texts:
    data=call('completions',dict(prompt=text,max_tokens=1,temperature=0,seed=42,echo=True,logprobs=1))
    c=data['choices'][0];n=data['usage']['prompt_tokens'];lp=c['logprobs']
    likelihood.append(dict(prompt=text,prompt_tokens=n,tokens=lp['tokens'][:n],token_logprobs=lp['token_logprobs'][:n]))
record=dict(tag=a.tag,time=time.time(),passed=sum(r['passed'] for r in results),total=len(results),cases=results,likelihood=likelihood,scope='Functional regression checks and limited teacher-forced likelihood; not full-distribution KLD or comprehensive model accuracy.')
out.write_text(json.dumps(record,indent=2));print(json.dumps({k:v for k,v in record.items() if k not in ('cases','likelihood')}),flush=True)
