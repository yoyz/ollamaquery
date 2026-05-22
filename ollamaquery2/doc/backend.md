# Backend Support Reference

ollamaquery2 supports three backends: **Ollama**, **Llama.cpp**, and **LM Studio**.
Each exposes a different API surface despite all serving LLMs.

## Quick Comparison

| Feature | Ollama | Llama.cpp | LM Studio |
|---------|--------|-----------|-----------|
| Chat endpoint | `/api/chat` | `/v1/chat/completions` | `/v1/chat/completions` |
| Model list | `/api/tags` | `/v1/models` | `/v1/models` |
| Model ID format | `qwen3.5:9b` | `Qwen3.5-9B-Q4_K_M.gguf` | `openai/gpt-oss-20b` |
| Context size | `/api/ps` or `/api/show` | `/slots` | None (use default) |
| Token counting | `/api/tokenize` | `/tokenize` | None (use estimation) |
| Thinking/reasoning field | `message.reasoning_content` | `delta.reasoning_content` | `delta.reasoning` |
| Image support | `message.images` array | `content` array with `image_url` | `content` array with `image_url` |
| SSE streaming | JSON lines, no prefix | `data: ` prefix + `[DONE]` | `data: ` prefix + `[DONE]` |
| Default port | 11434 | 8080 | 1234 |
| Server header | `ollama` | `llama.cpp` | (no reliable marker) |
| Detection method | GET `/` for `ollama` string | HEAD check for `llama.cpp` header | GET `/v1/models` for model list |
| Auto-discovery | HEAD check + host IP scan | HEAD check + host IP scan | GET `/v1/models` + host IP scan |

## Backend-Specific Details

### Ollama
- Each model is pulled and cached locally with a tag-based name (e.g. `qwen3.5:9b`).
- The `/api/chat` endpoint uses a custom JSON format (not OpenAI-compatible).
- Images are sent as a top-level `images` array on the user message dict.
- Context size is available via `/api/ps` (running models) or `/api/show` (any model).
- Token counting via `/api/tokenize` (model-specific).
- Server detection: GET to `/` and scan for `"ollama"` in the response body.
- Supports `/api/ps` to list loaded models with their context usage.

### Llama.cpp
- Models are loaded by filename (`Qwen3.5-9B-Q4_K_M.gguf`) — no tagging system.
- OpenAI-compatible `/v1/chat/completions` endpoint.
- SSE streaming with `data: ` prefix, ends with `data: [DONE]`.
- Thinking/reasoning comes via `delta.reasoning_content` in streaming chunks.
- Context size via `/slots` endpoint (returns array of slot info with `n_ctx`).
- Token counting via `/tokenize` (model-agnostic, counts subwords).
- Server detection: HEAD request checks for `Server: llama.cpp` header.

### LM Studio
- Models have vendor-prefixed IDs like `openai/gpt-oss-20b`.
- OpenAI-compatible `/v1/chat/completions` endpoint.
- SSE streaming with `data: ` prefix, ends with `data: [DONE]`.
- **Thinking/reasoning uses `delta.reasoning` instead of `delta.reasoning_content`** (key difference from llama.cpp).
- **No token counting endpoint** — falls back to `estimate_token_count()` heuristic.
- **No context size endpoint** — returns 0, uses default or `--contextsizeset`.
- **No `/slots` endpoint** — unlike llama.cpp.
- Context size must be set manually or the server's default applies.
- Inspection parameters: only supports `temperature`, `top_p`, `presence_penalty`, `repeat_penalty`. Does **not** support `top_k` or `min_p`.
- Server detection: GET to `/v1/models` and check for a non-empty `data` array.
- Image format: OpenAI-compatible `content` array with `{"type": "image_url", "image_url": {"url": "data:image/...;base64,..."}}` (same as llama.cpp).

## Architecture: 4 Streaming Helpers

A new backend requires implementing 4 methods in `ModelQuery`:

```
_build_stream_request()     → URL, payload, headers
_parse_chunk()              → (thought, content, is_final, usage)
_iter_stream_lines()        → yields decoded JSON per line
_update_context_tokens()    → updates ctx.current_context_tokens
```

Plus: auto-detection, model listing, reachability check, CLI argument.

## Known Pitfalls

### `reasoning` vs `reasoning_content`
Llama.cpp uses `delta.reasoning_content` in streaming and `message.reasoning_content` in non-streaming.
LM Studio uses `delta.reasoning` and `message.reasoning` instead.
Ollama uses `message.reasoning_content` (in `/api/chat` response).

The code in `run_agentic_query` handles this with fallback:
```python
thinking = msg.get('reasoning_content', '') or msg.get('reasoning', '')
```

### Response format differences
Ollama returns `{"message": {"content": "..."}}` at top level.
OpenAI-compatible backends (llama.cpp, LM Studio) return `{"choices": [{"message": {"content": "..."}}]}`.

### Model ID format
In Ollama, model IDs are short tags (`qwen3.5:9b`).
In Llama.cpp, model IDs are filenames (`Qwen3.5-9B-Q4_K_M.gguf`).
In LM Studio, model IDs have vendor prefixes (`openai/gpt-oss-20b`, `nvidia/nemotron-3-nano-4b`).

When switching models, use the exact ID format shown by `/listmodel`.

### Auto-detection order
Backend auto-detection follows this priority:
1. User-specified `-b` and `-H`
2. Saved backend configs (Most Recently Used, validated with HEAD/GET)
3. Auto-discovery on default ports (127.0.0.1): llamacpp (8080) → ollama (11434) → lmstudio (1234)
4. Auto-discovery on host IPs: same ports in same order
5. Fallback to defaults
