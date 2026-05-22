# Code Structure — ollamaquery2.py

## Overview

`ollamaquery2.py` (~4306 lines) is a single-file CLI tool for interacting with local LLM servers (Ollama and Llama.cpp). It supports interactive chat, batch processing, inline shell/URL/file commands, context tracking, agentic mode, and color themes.

## Architecture Diagram

```
main()
 ├── resolve_connection()        # Backend detection (saved → auto → default)
 ├── list_models_*()              # Model listing & info display
 └── ChatLoop.run()              # Interactive session
      ├── gather_user_input()    # Readline input with multiline support
      ├── process_inline_commands()
      │    ├── _process_file_inclusions()   # @filepath
      │    ├── _process_command_lines()     # !shell, /curl
      │    └── execute_os_command() / fetch_and_convert_url()
      ├── run_agentic_query()               # ReAct loop (if agentic mode)
      │    ├── parse_tool_calls()           # Multi-tool extraction (strict/lazy)
      │    ├── parse_tool_call()            # Single tool with format normalization
      │    │    └── _normalize_tool_json()  # OpenAI function-calling → internal
      │    ├── ToolRegistry.execute()       # Dispatch + alias mapping
      │    │    └── TOOL_ARG_ALIASES        # path→file, cmd→command, etc.
      │    └── query_stream()               # Final answer streaming
      └── query_stream()                    # Unified streaming loop (non-agentic)
           ├── _build_stream_request()      # Backend-specific URL + payload
           ├── _iter_stream_lines()         # SSE-aware line iteration
           ├── _parse_chunk()               # Backend-specific chunk parsing
           ├── _update_context_tokens()     # Context tracking per backend
           └── calculate_stats() / print_stats_display()
```

## Component Breakdown

### 1. Configuration Layer (lines 1–185)

- **Imports**: Standard library + optional (`yaml`, `readline`, `pty`)
- **Built-in prompts**: 7 presets (`default`, `coder`, `sysadmin`, `concise`, `doctor`, `teacher`, `politic`)
- **Themes**: 5 built-in color schemes with custom theme loading from `~/.ollamaquery/themes.json`
- **Constants**: Default hosts/ports, max context size (4M tokens)

### 2. Command Registry (lines 188–395)

A `COMMANDS` dict mapping command names to metadata (aliases, category, description, usage). Helper functions:
- `get_command_aliases()` — flat list for readline completion
- `get_commands_by_category()` — grouped display for `/help`
- `format_help_text()` — compact or detailed format
- `is_known_command()` / `get_command_by_alias()` — lookup

No handler functions are stored in the registry dispatching happens via `ChatLoop.run_handle_*()` methods.

### 3. Utility Functions (lines 403–775)

| Function | Purpose |
|----------|---------|
| `colorize()` / `get_theme()` | ANSI color output with readline-safe wrapping |
| `_request_with_retry()` | HTTP requests with 3 retries, 1s delay, no-4xx |
| `sanitize_shell_command()` | Blocks `;`, `|`, `&&`, `` ` ``, `$()` |
| `validate_shell_command_safety()` | Length limit + dangerous pattern blocklist |
| `execute_os_command()` | Shell execution with timeout, safety checks |
| `fetch_and_convert_url()` | URL to text via html2text/pandoc/lynx/fallback |
| `get_html_bytes()` | Fetch HTML bytes via curl/wget/urllib |
| `HTMLStripper` | HTML-to-text extraction (skips script/style) |
| `prepare_image_data()` | Base64 encode image files |
| `fetch_models_*()` | Model list from Ollama/Llama.cpp API |
| `get_*_context_size()` | Context window size from server |
| `get_message_token_count_*()` | Server-side tokenization |
| `estimate_token_count()` | Heuristic fallback (word count × 1.1) |
| `parse_size()` | Byte count → human-readable |
| `context_bar()` | `[████░░░░] NN%` visual bar |

### 4. CommandContext (lines 777–918)

**Singleton** holding all shared state:

- **Connection**: `base_url`, `backend`, `model`
- **Session**: `system_prompt`, `context_size`, `current_images`, `force_no_thinking`
- **Stats**: `total_queries`, `total_tokens_generated`, `total_time_spent`, `query_history`
- **Context tracking**: `context_window_size`, `current_context_tokens`
- **Debug**: `debug_mode`, `debug_manager`

Key methods:
- `reset()` — clears conversation state, preserves preferences
- `update_stats()` — accumulates after each query
- `calculate_context_tokens()` — estimates tokens in message list

### 5. DebugManager (lines 923–989)

Per-category debug levels (`off`/`basic`/`verbose`/`trace`) for 9 categories (`network`, `payload`, `response`, `stream`, `context`, `thinking`, `commands`, `urlfetch`, `all`).

Helper:
- `debug_log()` — timestamped debug output to stderr with optional JSON data

### 6. ModelQuery (lines 1032–1233)

Handles all LLM API communication. Contains:

#### Backend-agnostic methods:
- `query_stream()` — unified streaming loop (see below)
- `query_sync()` — non-streaming POST
- `build_request_payload()` — payload construction
- `calculate_stats()` — compute tokens/tps from response
- `print_stats_display()` — formatted stats to stderr

#### Unified Streaming System

Replaces the old `query_stream_ollama` + `query_stream_llamacpp` with a single loop + 4 pluggable helpers:

| Helper | Lines | Function |
|--------|-------|----------|
| `_build_stream_request()` | 15 | Returns `(url, payload, headers)` per backend |
| `_iter_stream_lines()` | 12 | Decodes lines, strips SSE `data: ` prefix for OpenAI-compatible APIs |
| `_parse_chunk()` | 30 | Extracts `(thought, content, is_final, usage)` from a chunk |
| `_update_context_tokens()` | 15 | Backend-specific context tracking after stream end |

To add a new backend (vLLM, LM Studio), add an `elif` branch to each of the 4 helpers.

#### Debug helpers:
- `_debug_request()` — logs outgoing request URL + payload
- `_debug_response_chunk()` — logs incoming chunks
- `_debug_final_stats()` — logs final usage
- `_mask_payload()` — replaces base64 images with size indicators for safe logging

### 7. ChatCompleter (lines 1640–1718)

Readline tab-completion for:
- **Models**: autocomplete names after `/switchmodel `
- **Paths**: for `/cwd`, `/ls`, `/image`, and inline `@` syntax
- **Commands**: all registered aliases

### 8. Input Handling (lines 1726–1903)

**`gather_user_input()`** (71 lines):
- Triple-quote `"""` multiline blocks
- Backslash `\` line continuation
- Double Ctrl+C to quit
- Double Ctrl+D to exit

**`process_inline_commands()`** — thin dispatcher calling:
- `_process_file_inclusions(text)` — scans for `@filepath` mentions, loads files with token counting
- `_process_command_lines(text)` — handles `!shell` and `/curl url` per line

### 9. ChatLoop (lines 2109–2737)

The interactive session manager. Uses a dispatcher pattern:

```
run()
 ├── run_init_session()          # Model list, readline setup, history
 ├── run_update_ollama_context() # Refresh context window size
 ├── run_display_context_bar()   # Visual context usage bar
 ├── run_build_prompt()          # Dynamic prompt prefix
 ├── gather_user_input()
 ├── run_handle_exit()           # /quit /exit
 ├── run_handle_help()           # /? /help
 ├── run_handle_stats()          # /stats /usage
 ├── run_handle_listmodel()      # /listmodel /listmodelall
 ├── run_handle_context_size()   # /contextsizeset
 ├── run_handle_clear()          # /clear
 ├── run_handle_image()          # /image
 ├── run_handle_dumpcontext()    # /dumpcontext
 ├── run_handle_debug()          # /debug
 ├── run_handle_thinking()       # /thinkingon /thinkingoff
 ├── run_handle_cwd()            # /cwd
 ├── run_handle_ls()             # /ls (safe: shlex + no shell=True)
 ├── run_handle_switchmodel()    # /switchmodel (pings model on switch)
 ├── run_handle_spawnshell()     # /spawnshell
 └── run_process_query()         # Regular query → inline processing → stream
```

Each handler returns `True` (exit loop), `False` (continue loop), or `None` (not handled).

### 10. Main & Entry Point (lines 2749–3455)

**Connection resolution** (`resolve_connection()`, 59 lines):
1. CLI override (`-H`)
2. Saved configs (`~/.ollamaquery.d/backends.json`)
3. Auto-detection (checks default ports, scans host IPs)
4. Defaults (env vars or hardcoded)

**Model selection** (`main()`, 281 lines):
- Explicit `-m` flag
- Loaded model from `/api/ps` (Ollama only)
- Hosted model from `/v1/models` (llama.cpp)
- Empty → user must `/listmodel` + `/switchmodel`

**Modes**: chat (`-c`), batch (`--input-dir`), single query (`-I`), listing (`-l`), model info (`--show`)

## Data Flow (User Query)

```
gather_user_input()
  → process_inline_commands()
    → _process_file_inclusions()    # @file → loaded as context
    → _process_command_lines()      # !cmd, /curl → executed
  → ChatLoop.run_process_query()
    → messages.append(user)
    → query_stream()
      → _build_stream_request()     # URL + payload per backend
      → _iter_stream_lines()        # SSE-aware streaming
      → _parse_chunk()              # thought/content/usage extraction
      → stdout write (real-time)    # displayed to user
      → _update_context_tokens()    # backend-specific context tracking
      → calculate_stats()           # tokens, tps
      → update_stats()              # cumulative stats
    → messages.append(assistant)
```

## Backend Integration Guide

To add a new backend (e.g., vLLM, LM Studio), implement 4 methods:

```python
def _build_stream_request(self, backend, messages, model, stream_enabled, context_size):
    if backend == "vllm":
        return url, payload, headers
    ...

def _parse_chunk(self, chunk, backend):
    if backend == "vllm":
        return thought, content, is_final, usage
    ...

def _iter_stream_lines(self, response, backend):
    if backend == "vllm":
        # SSE with data: prefix
        ...
    yield decoded

def _update_context_tokens(self, backend, aggregated_usage, messages):
    if backend == "vllm":
        self.ctx.current_context_tokens = ...
```

Each helper is under 15–30 lines. The streaming loop (~80 lines) is fully shared.

## Key Design Decisions

- **Singleton CommandContext**: Single source of truth for all state, avoids threading issues in single-user CLI
- **/switchmodel pings**: Pre-loads model into memory so the first real query doesn't pay loading cost
- **@file works mid-sentence**: Whitespace-delimited scanning allows natural language (e.g., `"summarize @file.txt"`)
- **Blocklist over allowlist**: For `!shell` commands; intentional tradeoff for power users
- **Single file**: Easier to deploy (no pip install, just download and run)
