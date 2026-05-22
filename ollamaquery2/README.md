# ollamaquery2

A feature-rich CLI for local LLMs via Ollama, Llama.cpp, and LM Studio backends. Single Python file, no pip install needed.

## Backends

| Backend | Default URL | Detection |
|---------|-------------|-----------|
| Ollama | `http://127.0.0.1:11434` | HEAD `/` + body scan |
| Llama.cpp | `http://127.0.0.1:8080` | HEAD `Server:` header |
| LM Studio | `http://127.0.0.1:1234` | GET `/v1/models` |

Auto-detects available backends. Set `OLLAMA_HOST`, `LLAMACPP_HOST` env vars to override.

## Quick Start

```bash
# Chat mode (auto-detects backend)
python3 ollamaquery2.py -c
```

### CLI Reference

```text
-c                  Interactive chat mode
-m <model>          Model name
-b <backend>        Backend: ollama, llamacpp, lmstudio
-H <url>            Backend URL override
-I "query"          Single-shot query
-i <file>           Input file for batch
-o <file>           Output file
--input-dir <dir>   Batch process directory
--output-dir <dir>  Output directory
-l                  List models
-la                 List models with capabilities
--show <model>      Show model info
--show-details      Show detailed model info
-P <profile>        Prompt profile (default/coder/sysadmin/...)
-p "text"           Custom system prompt
--theme <name>      Color theme
--no-color          Disable colors
--image <path>      Attach image(s)
--shell-timeout <s> Shell command timeout (default 5)
```

Terminal features (`readline`, `pty`) are optional — tab completion and
spawnshell are disabled when unavailable.

## Features

### Core
- Streaming responses with real-time token output
- Thinking/reasoning display (Detects `<think>` tags, `reasoning_content`, `reasoning` fields)
- Multiline input via `"""` blocks
- Tab completion (commands, models, file paths)
- Context bar showing live token usage
- Per-query stats (tokens, TPS, context usage)
- Color theming (default, minimal, emacs_dark, vim_dark, high_contrast + custom JSON)
- `NO_COLOR` env var support

### Inline Commands
- `! command` — Execute shell command, feed output to LLM
- `/curl url` — Fetch URL content into context
- `@filepath` — Include file content mid-sentence

### Interactive Commands
| Command | Description |
|---------|-------------|
| `/?` `/help` | Help menu |
| `/listmodel` | List available models |
| `/listmodelall` | List models with capabilities |
| `/switchmodel <name>` | Change model, preserve history |
| `/clear` | Wipe conversation |
| `/stats` `/usage` | Session statistics |
| `/thinkingon` `/thinkingoff` | Toggle reasoning display |
| `/debug` | Per-category debug levels (`network`, `payload`, `response`, `stream`, `context`, `thinking`, `commands`, `urlfetch`, `all`) |
| `/cwd <path>` | Change working directory |
| `/ls` | List files locally |
| `/image <path>` | Attach an image to conversation |
| `/spawnshell` | Drop into interactive shell (output sent to LLM on exit) |
| `/dumpcontext` | Save conversation to JSON |
| `/contextsizeset <N>` | Override context window |
| `/quit` `/exit` | End session (`Ctrl+C` twice / `Ctrl+D` also works) |

### Agentic Mode

`/agentic on` enables a ReAct loop with tool access:

| Tool | Description |
|------|-------------|
| `fetch_url` | Fetch URL to text |
| `read_file` | Read file (max 100KB) |
| `write_file` | Write file |
| `list_directory` | List directory contents |
| `glob` | Find files by pattern |
| `run_python` | Execute Python 3 code |
| `run_command` | Execute shell command (120s timeout) |
| `diff` | Unified diff between files |
| `patch` | Apply unified diff |
| `edit_file` | Precise string replacement |
| `apply_patch` | Apply diff with file headers |

Subcommands: `/agentic auto`, `sandbox`, `verbose`, `thinking`, `trace`, `log`, `lazytool`, `iterations <N>`, `timeout <N>`.

Safety features: destructive tool confirmation, path traversal guard, step timeout (default 120s), stuck detection, same-tool-loop abort, optional container sandbox (podman/docker).

### Agentic Example

```text
> /agentic on
Agentic mode enabled.

> /agentic auto
Auto-confirm: ON

> what is the weather in Grenoble today (2026-05-22) and what kernel am I running?
[Agentic] Step 1/50…
[Tool] fetch_url(url='https://api.open-meteo.com/v1/forecast?latitude=45.1885&longitude=5.7245&current_weather=true')

[Agentic] Step 2/50…
[Tool] run_command(command='uname -r')

**Weather in Grenoble today (2026-05-22):**
- Temperature: 26.9°C
- Wind speed: 7.7 km/h
- Wind direction: 319°
- Weather: Clear sky

**Your kernel:** 6.8.0-117-generic
```

The model autonomously decides which tools to call and in what order —
`fetch_url` for weather data via Open-Meteo API, `run_command` for the
local kernel check.

**Warning:** `/agentic auto` skips all destructive-tool confirmation prompts.
Only enable this if you trust the model's reliability — a compromised or
unpredictable model could read, write, or execute arbitrary commands without
oversight.

## Testing

```bash
python3 -m unittest discover tests -v
```

~130 unit tests (no backend needed) + ~60 integration tests (require live backend).

Test coverage: configuration, retry logic, theme system, command registry, shell safety, HTML parsing, token counting, debug manager, image handling, backend detection, model listing, basic queries, stats, command handlers, switch model, spawn shell, dump context, full workflow. Agentic: ReAct loop, tool execution, JSON normalization, multi-tool parsing, argument aliases, lazy mode, stuck detection, timeout, end-to-end DNS resolver pipeline.

## Requirements

- Python 3.x
- A running LLM backend (Ollama / Llama.cpp / LM Studio)
- Tested on Fedora 43 and Ubuntu 24

## System Prompt

Default prompt: accurate chatbot mirroring the user's language, formatting
responses with markdown. Built-in profiles (select via `-P` / `--profile`):

| Profile | Focus |
|---------|-------|
| `default` | General-purpose, language-mirroring chatbot |
| `coder` | Python/C++ specialist with system engineering knowledge |
| `sysadmin` | Linux admin, short and direct answers |
| `concise` | No filler, skip pleasantries |
| `doctor` | Medical info with disclaimer (not a licensed professional) |
| `teacher` | Patient educator with analogies and adaptable depth |
| `politic` | Neutral analyst, multiple perspectives, no endorsements |

```bash
python3 ollamaquery2.py -c -P coder
python3 ollamaquery2.py -c -p "You are a helpful assistant that speaks like a pirate."
```

In agentic mode, the prompt is replaced with a model-specific one built
from composable blocks (role, tool definitions, format, examples, rules).

## Environment

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama API URL |
| `LLAMACPP_HOST` | `http://127.0.0.1:8080` | Llama.cpp API URL |
| `OLLAMAQUERY_THEME` | `default` | Color theme name |
| `NO_COLOR` | (unset) | Disable colors |
