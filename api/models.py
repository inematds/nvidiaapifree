"""GET /api/models -> lista de modelos disponiveis no tier gratuito."""
import json
from http.server import BaseHTTPRequestHandler

MODELS = [
    {"id": "meta/llama-3.1-70b-instruct", "label": "Llama 3.1 70B", "provider": "Meta", "category": "llm"},
    {"id": "meta/llama-3.1-405b-instruct", "label": "Llama 3.1 405B", "provider": "Meta", "category": "llm"},
    {"id": "deepseek-ai/deepseek-r1", "label": "DeepSeek R1", "provider": "DeepSeek", "category": "llm"},
    {"id": "qwen/qwen3.5-397b-a17b", "label": "Qwen 3.5 397B", "provider": "Qwen", "category": "llm"},
    {"id": "z-ai/glm-5.2", "label": "GLM-5.2", "provider": "Z.ai", "category": "llm"},
    {"id": "nvidia/nemotron-3-super-120b-instruct", "label": "Nemotron 120B", "provider": "NVIDIA", "category": "llm"},
    {"id": "mistralai/mixtral-8x22b-instruct-v0.1", "label": "Mixtral 8x22B", "provider": "Mistral", "category": "llm"},
    {"id": "meta/llama-3.2-90b-vision-instruct", "label": "Llama 3.2 90B Vision", "provider": "Meta", "category": "vision"},
]


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps(MODELS, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
