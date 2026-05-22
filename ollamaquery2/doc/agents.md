# Agentic Mode

## Overview

Agentic mode enables the LLM to use tools (read/write files, execute Python, fetch URLs, etc.) in a ReAct loop: the model decides which tool to call, the code executes it, and the result feeds back into the conversation.

## Enabling

```
/agentic          → show status + available subcommands (like /debug)
/agentic on       → enable agentic mode
/agentic off      → disable agentic mode
/agentic full     → enable everything: mode + verbose + thinking + trace + auto-confirm
```

## Available Tools

| Tool | Destructive | Description |
|---|---|---|
| `fetch_url` | No | Fetch a URL and return its content as plain text |
| `read_file` | No | Read a file from disk (text, max 100KB) |
| `write_file` | Yes | Write text content to a file (with confirmation) |
| `list_directory` | No | List files/directories in a path |
| `glob` | No | Find files matching a glob pattern |
| `run_python` | Yes | Execute Python 3 code (inline or file, 30s timeout) |
| `diff` | No | Generate a unified diff between two files |
| `patch` | Yes | Apply a unified diff to a file in-place |

```
/listtool  → see all available tools
```

## Subcommands

| Command | Effect |
|---|---|
| `/agentic auto` | Toggle skip destructive tool confirmation prompts |
| `/agentic sandbox` | Toggle host/container executor |
| `/agentic verbose` | Show raw model responses during ReAct |
| `/agentic thinking` | Show model reasoning during ReAct |
| `/agentic trace` | Show full tool args and results |
| `/agentic log` | Toggle structured JSONL logging |
| `/agentic iterations <N>` | Set max ReAct loop iterations (default: 10) |
| `/agentic timeout <N>` | Set per-step timeout in seconds (default: 120) |

## Architecture

```
User query        
    │
    ▼
run_agentic_query()
    │
    ├── Build augmented prompt (system + tools + history)
    │
    └── ReAct loop (while True, max iterations):
        │
        ├── query_sync(model, messages)  → model responds
        │
        ├── parse_tool_call(response)    → JSON or plain text?
        │   ├── JSON tool call → confirm → execute → append observation → loop
        │   └── plain text    → break (this is the Final Answer)
        │
        └── Three-layer safety:
            ├── _call_with_timeout()  — per-step wall-clock timeout
            ├── _is_stuck()           — character-level repetition detection
            └── same-tool-loop        — identical tool+args twice → abort
    │
    └── Final stream: query_stream(messages) → user sees typing effect
```

## How Tools Work

1. The LLM responds with a JSON object: `{"tool": "name", "arguments": {...}}`
2. `ToolRegistry.execute()` validates the tool name, confirms if destructive, and dispatches to the handler
3. The result (`{"success": bool, "output": str, "error": str}`) is appended to the message history
4. The LLM sees the result and decides the next step

## Container Sandbox

Tools that spawn subprocesses (fetch_url, write_file, run_python, patch) can run inside a container:

```
/agentic sandbox  → toggle host ↔ container
```

Uses `podman` by default (or `docker` via `OLLAMAQUERY_CONTAINER_RT` env var).
The current working directory is bind-mounted to `/workspace` inside the container.

## Logging

Every agentic session is logged to `~/.ollamaquery.d/agentic/YYYYMMDD_HHMMSS.jsonl`:

```jsonl
{"type": "start", "timestamp": "...", "user_input": "..."}
{"type": "turn", "iteration": 1, "model_response": "...", "tool_call": {...}}
{"type": "result", "tool_name": "write_file", "tool_args": {...}, "result": {...}}
{"type": "final", "final_answer": "..."}
{"type": "end", "total_iterations": 3}
```

## Safety Features

| Mechanism | What it protects against |
|---|---|
| **Destructive tool confirmation** | Unwanted write/execute/patch operations |
| **Auto-confirm** | Power users can skip prompts (`/agentic auto`) |
| **Path traversal guard** | read/write/list reject paths outside CWD |
| **Step timeout (120s)** | Model thinking forever |
| **Stuck detection** | Model repeating itself in a loop |
| **Same-tool-loop detection** | Model calling same tool+args repeatedly |
| **Container sandbox** | Full OS-level isolation via podman/docker |
