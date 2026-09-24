"""Send five small synthetic images to verify the former image cap is gone."""
import argparse
import base64
import io
import json
from pathlib import Path
import urllib.request
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--tag", required=True)
args = parser.parse_args()
output = Path.home() / "bench/tp1-opt-20260922" / (args.tag + "-images.json")
if output.exists():
    raise RuntimeError(f"Refusing to overwrite {output}")
content = [{"type": "text", "text": "How many images are attached? Answer only the number."}]
for color in ("red", "green", "blue", "yellow", "magenta"):
    image = Image.new("RGB", (224, 224), color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    content.append({"type": "image_url", "image_url": {"url": url}})
payload = {"model": "qwen3.8-flash-next-4p89bpw",
           "messages": [{"role": "user", "content": content}],
           "max_tokens": 32, "temperature": 0,
           "chat_template_kwargs": {"enable_thinking": False}}
request = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",
    data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
with urllib.request.urlopen(request, timeout=180) as response:
    record = {"image_count": 5, "image_dimensions": [224, 224],
              "http_status": response.status, "response": json.load(response)}
output.write_text(json.dumps(record, indent=2))
print(json.dumps(record))
