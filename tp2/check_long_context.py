"""Exercise the configured context boundary and check simple passkey retrieval."""
import json
import argparse
from pathlib import Path
import time
import urllib.request
from transformers import AutoTokenizer

root = Path("/tmp")
parser = argparse.ArgumentParser()
parser.add_argument("--tag", default="resident3")
args = parser.parse_args()
output = root / (args.tag + "-long-context.json")
if output.exists():
    raise RuntimeError(f"Refusing to overwrite {output}")
tokenizer = AutoTokenizer.from_pretrained(Path.home() / "builds/qwen-tp2/model")
marker = "DOCUMENT_INSERTION_POINT"
rendered = tokenizer.apply_chat_template(
    [{"role": "user", "content": marker + "\nReturn the three audit passkeys, labeled ALPHA, BETA, GAMMA. Copy them exactly; no explanation."}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False)
prefix, suffix = rendered.split(marker)
encode = lambda s: tokenizer.encode(s, add_special_tokens=False)
filler = encode(" Ordinary document entry: the warehouse has shelves, labels, boxes, and loading bays.\n")
body = (filler * 20000)[:261000]
for offset, label, key in reversed([(12000, "ALPHA", "cedar-731826"),
                                    (128000, "BETA", "quartz-492175"),
                                    (248000, "GAMMA", "harbor-863204")]):
    body[offset:offset] = encode(f"\nAudit passkey {label}: {key}\n")
left, right = encode(prefix), encode(suffix)
budget = 261888
body += (filler * 1000)
prompt = left + body[:budget - len(left) - len(right)] + right
assert len(prompt) == budget
payload = {"model": "qwen3.8-flash-next-4p89bpw", "prompt": prompt,
           "max_tokens": 128, "temperature": 0, "top_p": 1, "seed": 42}
request = urllib.request.Request("http://127.0.0.1:8000/v1/completions",
    data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
started = time.time()
with urllib.request.urlopen(request, timeout=900) as response:
    result = json.load(response)
text = result["choices"][0]["text"]
record = {"prompt_tokens_requested": budget, "wall_s": time.time() - started,
          "all_passkeys_retrieved": all(k in text for k in
              ("cedar-731826", "quartz-492175", "harbor-863204")), "response": result}
output.write_text(json.dumps(record, indent=2))
print(json.dumps(record), flush=True)
