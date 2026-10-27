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

Two targets: an image (a throwaway `--rm` container per command, CWD bind-mounted
to `/workspace`) or an already-running container (`--sandbox-container NAME`, which
execs into it instead — state, packages and mounts you set up are what the model
gets, and nothing is created or destroyed). Select from the CLI:

```
--sandbox-image IMAGE        (or OLLAMAQUERY_CONTAINER_IMAGE; default python:3.12-alpine)
--sandbox-container NAME     (or OLLAMAQUERY_CONTAINER_NAME; implies container mode)
--sandbox-runtime podman|docker  (or OLLAMAQUERY_CONTAINER_RT; default podman)
--sandbox-memory SIZE           (or OLLAMAQUERY_CONTAINER_MEMORY; default 1g)
--sandbox-cpus N                (or OLLAMAQUERY_CONTAINER_CPUS; default 1)
```

The throwaway container is capped at 1 CPU / 1 GiB by default (`--memory 1g
--cpus 1`) so a runaway command cannot consume the host; the ceiling is shown in
the banner, in `/agentic status` and in the `/agentic sandbox` message. With
`--sandbox-container` no ceiling is applied — that container keeps the limits of
whoever started it.

The image is probed once per session and pulled only when absent (a locally built
`localhost/...` image is never pulled: podman would treat `localhost/` as a
registry). With `--sandbox-container`, the runtime's mounts are inspected and the
mount containing the current CWD becomes the workdir, so relative paths hit your
files; without such a mount the container's own workdir is kept.

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
