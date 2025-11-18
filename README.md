# OpenAI Proxy (Ports 5101 / 5102)

A production-ready Python/Flask proxy that sits between your tooling (e.g. Xcode, LobeChat, Open-WebUI) and the official OpenAI REST API. The proxy normalises legacy `/v1/chat/completions` traffic, automatically upgrades selected calls to the newer `/v1/responses` endpoint, injects optional tooling (web search, reasoning summaries), and exposes a consistent surface that existing OpenAI-compatible clients can consume.

> **Status:** Actively maintained. Works with OpenAI's GPT‑4.1/GPT‑5.x family, reasoning models, vision inputs, and streaming responses.

---

## Highlights

- ⚡ **Dual-port forwarding** – Port `5101` behaves like a transparent `/v1/chat/completions` proxy, whereas port `5102` upgrades those calls to `/v1/responses` (with automatic SSE → Chat-completions translation, reasoning summaries, and streaming support).
- 🧠 **Reasoning & web search defaults** – When a client does not specify tools, the proxy adds OpenAI's `web_search` tool, enables medium-effort reasoning, and requests reasoning summaries so downstream apps can display them without extra work.
- 🖼️ **Vision & file input compatibility** – Chat-style payloads containing `image_url`, `file_id`, or base64 image data are converted into the shapes the Responses API expects (`input_image`, `input_file`, etc.).
- 📜 **Model list filtering** – `/v1/models` responses are filtered per port: port `5101` hides Codex-only models, port `5102` shows only Codex-capable models, matching typical IDE routing requirements.
- 📼 **Structured logging** – Pass `--log <path>` to capture every upstream/downstream request, response, and streaming chunk (logs are rotated on each start).
- 🐳 **Docker-first deployment** – Comes with a `Dockerfile` and `docker-compose.yml` for repeatable builds, plus a simple `python openai_proxy.py --port 5101` entry point for local testing.

---

## Architecture at a Glance

```
Client (Xcode / UI)  -->  Port 5101  -->  https://api.openai.com (compat mode)
                     -->  Port 5102  -->  https://api.openai.com (Responses mode)
```

1. Incoming request → detect listening port by `Host` header / WSGI env.
2. `/v1/models` is served locally with filtered results.
3. Port `5102` routes `/v1/chat/completions` to `/v1/responses`, reshaping payloads & SSE streams so the client still receives classic chat-completion chunks.
4. Optional logging captures both request/response bodies (including streaming chunks) for diagnostics.

---

## Quick Start

### 1. Clone & configure
```bash
git clone https://github.com/<your-org>/openai-proxy.git
cd openai-proxy
```

(Optional) Provide a curated model list that should appear at `/v1/models`:
```bash
cp models.example.json models.json
# Edit models.json to your needs
```

### 2. Run via Docker Compose
```bash
docker-compose up -d --build
```
This maps the container’s `5101` port to host `5101/5102` (see `docker-compose.yml`).

### 3. Or run locally with Python 3.11+
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python openai_proxy.py --port 5101
```
Add `--log proxy.log` to capture traffic (the log file is truncated on every start).

### 4. Point your client
```
Base URL : http://<proxy-host>:5101/v1
API Key  : Your real OpenAI API key (the proxy simply forwards it)
```
Use port `5102` if you want automatic `/v1/responses` upgrades with reasoning summaries.

---

## Configuration Guide

| Setting | Location | Description |
|--------|----------|-------------|
| `OPENAI_API_KEY` | Client environment | The proxy does **not** store or mint keys; you must provide a valid OpenAI key to your client as usual. |
| `models.json` | Repository root (copied into container) | Optional override for `/v1/models`. If absent or invalid, the internal fallback list is used. |
| `docker-compose.yml` | Root | Adjust host port mappings, resource limits, restart policy, etc. |
| `--log <path>` | CLI flag | Writes a structured log (requests, responses, streaming chunks). File is deleted on startup to avoid leaking stale data. |
| `--port <int>` | CLI flag | Listen on a custom port (defaults to `5101`). |

### Behaviour per Port

| Port | Traffic type | Special handling |
|------|--------------|------------------|
| 5101 | Classic `/v1/chat/completions` passthrough | Filters Codex models out of `/v1/models`. No reasoning/tool injection. |
| 5102 | `/v1/chat/completions` → `/v1/responses` bridge | Adds `web_search` tool & medium reasoning when missing, converts images/files to Responses-native formats, streams reasoning summaries back as chat chunks. |

---

## Logging & Observability

- Enable logging: `python openai_proxy.py --port 5102 --log proxy_5102.log`
- Each entry contains timestamp, port, direction (`upstream-request`, `downstream-response-chunk`, etc.), and the raw payload.
- Logs are plain text; redact secrets before sharing.

---

## Security & Compliance

1. **API Terms** – This project merely forwards traffic; you are still bound by OpenAI’s [Terms of Use](https://openai.com/policies/terms-of-use) and any third-party agreements.
2. **No affiliation** – “OpenAI” and product names (GPT‑4.1, GPT‑5, etc.) are trademarks of OpenAI. This proxy is an independent, unofficial tool.
3. **Data handling** – Request/response bodies may contain personal or proprietary information. Ensure logging is enabled only in secure environments and rotate keys regularly.
4. **Licensing** – Released under the MIT License (see below). Use at your own risk; there is no warranty of any kind.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `Invalid value: 'image_url'` | Client sent Chat-style `image_url` blocks before proxy update | Upgrade to the latest proxy version (image/file mapping now handled automatically). |
| Streaming hangs after reasoning output | Client waits for `[DONE]` but responses keep streaming | Ensure your client consumes SSE chunks and stops on `[DONE]`. The proxy emits reasoning summaries before completion. |
| `/v1/models` missing entries | Port filtering active | Use port `5101` for non-Codex lists, `5102` for Codex-only lists. |
| `401 Unauthorized` | Invalid OpenAI key or misconfigured Authorization header | Confirm your client still supplies the correct `Bearer` token (the proxy never injects keys). |

---

## Contributing

1. Fork & clone the repo.
2. Create a branch: `git checkout -b feature/xyz`.
3. Format/lint: `ruff check` (or your preferred tool) + `pytest` if you add tests.
4. Submit a PR with a clear description. Logs should not contain sensitive info.

Bug reports are welcome—please include relevant log excerpts (redacted) and reproduction steps.

---

## License

```
MIT License

Copyright (c) 2024-present

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

Happy building—and please respect OpenAI's usage policies when routing traffic through this proxy.
