"""POST /api/chat -> {content, total_ms, model} (resposta completa, sem streaming).

Padrao nativo @vercel/python (BaseHTTPRequestHandler, so stdlib, HTTP puro
via urllib). O frontend detecta pelo Content-Type: application/json = resposta
unica (aqui); text/event-stream = streaming (servidor local FastAPI).

Env na Vercel: NVIDIA_API_KEY (obrigatoria) e opcionalmente
NVIDIA_API_KEY_LLAMA / _DEEPSEEK / _QWEN / _GLM / etc.
"""
import json
import os
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler

BASE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_MODEL = "meta/llama-3.1-70b-instruct"
TIMEOUT = 55

REGISTRY = {
    "meta/llama-3.1-70b-instruct": {
        "key_env": "NVIDIA_API_KEY_LLAMA",
        "params": {"temperature": 0.7, "top_p": 0.95, "max_tokens": 4096},
    },
    "meta/llama-3.1-405b-instruct": {
        "key_env": "NVIDIA_API_KEY_LLAMA405",
        "params": {"temperature": 0.7, "top_p": 0.95, "max_tokens": 4096},
    },
    "deepseek-ai/deepseek-r1": {
        "key_env": "NVIDIA_API_KEY_DEEPSEEK",
        "params": {"temperature": 0.6, "top_p": 0.95, "max_tokens": 4096},
    },
    "qwen/qwen3.5-397b-a17b": {
        "key_env": "NVIDIA_API_KEY_QWEN",
        "params": {"temperature": 0.6, "top_p": 0.95, "max_tokens": 4096},
    },
    "z-ai/glm-5.2": {
        "key_env": "NVIDIA_API_KEY_GLM",
        "params": {"temperature": 1, "top_p": 1, "max_tokens": 16384},
    },
    "nvidia/nemotron-3-super-120b-instruct": {
        "key_env": "NVIDIA_API_KEY_NEMOTRON",
        "params": {"temperature": 0.7, "top_p": 0.95, "max_tokens": 4096},
    },
    "mistralai/mixtral-8x22b-instruct-v0.1": {
        "key_env": "NVIDIA_API_KEY_MISTRAL",
        "params": {"temperature": 0.7, "top_p": 0.95, "max_tokens": 4096},
    },
    "meta/llama-3.2-90b-vision-instruct": {
        "key_env": "NVIDIA_API_KEY_VISION",
        "params": {"temperature": 0.7, "top_p": 0.95, "max_tokens": 2048},
    },
}


def run_chat(req):
    model = (req.get("model") or DEFAULT_MODEL).strip()
    messages = req.get("messages") or []
    entry = REGISTRY.get(model, REGISTRY[DEFAULT_MODEL])
    key = os.environ.get(entry["key_env"]) or os.environ.get("NVIDIA_API_KEY")
    if not key:
        return {"error": "NVIDIA_API_KEY nao configurada na Vercel (Settings -> Environment Variables)."}

    payload = {"model": model, "messages": messages, "stream": False}
    payload.update(entry["params"])

    request = urllib.request.Request(
        BASE_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    start = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        return {"error": f"NVIDIA API HTTP {exc.code}: {detail}"}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}

    total_ms = int((time.monotonic() - start) * 1000)
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return {"error": f"resposta inesperada da NVIDIA API: {json.dumps(data)[:300]}"}
    return {"content": content, "total_ms": total_ms, "model": model}


class handler(BaseHTTPRequestHandler):
    def _send(self, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        try:
            length = int(self.headers.get("content-length", 0))
            req = json.loads(self.rfile.read(length) or b"{}")
            self._send(run_chat(req))
        except Exception as exc:
            self._send({"error": f"{type(exc).__name__}: {exc}"})

    def do_GET(self):
        self._send({"ok": True, "models": sorted(REGISTRY)})
