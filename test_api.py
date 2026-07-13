from openai import OpenAI
from dotenv import load_dotenv
import os
import sys
from pathlib import Path

_USE_COLOR = sys.stdout.isatty() and os.getenv("NO_COLOR") is None
_RESET = "\033[0m" if _USE_COLOR else ""
_GREEN = "\033[32m" if _USE_COLOR else ""
_DIM = "\033[90m" if _USE_COLOR else ""

load_dotenv(Path(__file__).parent / ".env")

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"],
)

print(f"{_GREEN}Testing NVIDIA Free API...{_RESET}")
print(f"{_DIM}Model: meta/llama-3.1-70b-instruct{_RESET}\n")

completion = client.chat.completions.create(
    model="meta/llama-3.1-70b-instruct",
    messages=[{"role": "user", "content": "Diga apenas 'ok, funcionando' se voce recebeu esta mensagem."}],
    temperature=0.7,
    top_p=0.95,
    max_tokens=256,
    stream=True,
)

for chunk in completion:
    if not getattr(chunk, "choices", None):
        continue
    if len(chunk.choices) == 0 or getattr(chunk.choices[0], "delta", None) is None:
        continue
    delta = chunk.choices[0].delta
    if getattr(delta, "content", None) is not None:
        print(delta.content, end="")

print(f"\n\n{_GREEN}OK{_RESET}")
