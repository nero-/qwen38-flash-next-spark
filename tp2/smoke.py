import json
from pathlib import Path
import urllib.request
payload = {'model':'qwen3.8-flash-next-4p89bpw', 'messages':[{'role':'user','content':'What is 17 times 23? Answer only the number.'}], 'temperature':0, 'max_tokens':32, 'chat_template_kwargs':{'enable_thinking':False}}
req=urllib.request.Request('http://127.0.0.1:8000/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
with urllib.request.urlopen(req,timeout=180) as r: result=json.load(r)
Path.home().joinpath('builds/qwen-tp2/evidence/smoke.json').write_text(json.dumps(result,indent=2))
answer=result['choices'][0]['message']['content'].strip()
print(answer)
assert answer=='391', result
