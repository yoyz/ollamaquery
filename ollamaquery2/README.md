# ollamaquery2

A feature-rich CLI for LLMs via local Ollama, Llama.cpp, LM Studio and Strata servers, plus cloud OpenAI-compatible APIs (Gemini, OpenCode Zen/Go, Mistral, DeepSeek). Single Python file, no pip install needed.

## Backends

| Backend        | Default URL                                                 | Detection / auth                        |
| -------------- | ----------------------------------------------------------- | --------------------------------------- |
| Ollama         | `http://127.0.0.1:11434`                                    | HEAD `/` + body scan                    |
| Llama.cpp      | `http://127.0.0.1:8080`                                     | HEAD `Server:` header                   |
| Strata         | `http://127.0.0.1:8080`                                     | GET `/health` → `"service": "strata"`   |
| LM Studio      | `http://127.0.0.1:1234`                                     | GET `/v1/models`                        |
| Gemini         | `https://generativelanguage.googleapis.com/v1beta/openai`   | `GEMINI_API_KEY`                        |
| OpenCode Zen   | `https://opencode.ai/zen`                                   | `OPENCODEZEN_API_KEY`                   |
| OpenCode Go    | `https://opencode.ai/zen/go`                                | `OPENCODEGO_API_KEY`                    |
| Mistral        | `https://api.mistral.ai`                                    | `MISTRAL_API_KEY`                       |
| DeepSeek       | `https://api.deepseek.com`                                  | `DEEPSEEK_API_KEY`                      |

Auto-detects available **local** backends. Cloud backends are selected with
`-b` and read their API key from the environment (`GEMINI_HOST`,
`OPENCODEGO_HOST`, etc. override the host).

Strata shares Llama.cpp's default port (8080) and OpenAI dialect but streams
reasoning via `delta.reasoning_content` and exposes Anthropic
`/v1/messages/count_tokens`; it is recognized by its unique `/health` marker.

## Quick Start

```bash
# Chat mode (auto-detects backend)
python3 ollamaquery2.py -c
```

### CLI Reference

```text
-c                  Interactive chat mode
-m <model>          Model name
-b <backend>        Backend: ollama, llamacpp, strata, lmstudio, gemini, opencodezen, opencodego, mistral, deepseek
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
| Command                      | Description                                                                                                                  |          |                                                    |        |                                        |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------- | -------- | -------------------------------------------------- | ------ | -------------------------------------- |
| `/?` `/help`                 | Help menu                                                                                                                    |          |                                                    |        |                                        |
| `/listmodel`                 | List available models                                                                                                        |          |                                                    |        |                                        |
| `/listmodelall`              | List models with capabilities                                                                                                |          |                                                    |        |                                        |
| `/switchmodel <name>`        | Change model, preserve history                                                                                               |          |                                                    |        |                                        |
| `/clear`                     | Wipe conversation                                                                                                            |          |                                                    |        |                                        |
| `/stats` `/usage`            | Session statistics                                                                                                           |          |                                                    |        |                                        |
| `/thinkingon` `/thinkingoff` | Toggle reasoning display                                                                                                     |          |                                                    |        |                                        |
| `/reasoning [off\            | on\                                                                                                                          | low\     | medium\                                            | high]` | Set reasoning effort (Qwen3.8/gpt-oss) |
| `/preservethinking [on\      | off\                                                                                                                         | status]` | Retain reasoning traces across turns (Qwen3.6/3.8) |        |                                        |
| `/debug`                     | Per-category debug levels (`network`, `payload`, `response`, `stream`, `context`, `thinking`, `commands`, `urlfetch`, `all`) |          |                                                    |        |                                        |
| `/cwd <path>`                | Change working directory                                                                                                     |          |                                                    |        |                                        |
| `/ls`                        | List files locally                                                                                                           |          |                                                    |        |                                        |
| `/image <path>`              | Attach an image to conversation                                                                                              |          |                                                    |        |                                        |
| `/spawnshell`                | Drop into interactive shell (output sent to LLM on exit)                                                                     |          |                                                    |        |                                        |
| `/dumpcontext`               | Save conversation to JSON                                                                                                    |          |                                                    |        |                                        |
| `/contextsizeset <N>`        | Override context window                                                                                                      |          |                                                    |        |                                        |
| `/quit` `/exit`              | End session (`Ctrl+C` twice / `Ctrl+D` also works)                                                                           |          |                                                    |        |                                        |

### Agentic Mode

`/agentic on` enables a ReAct loop with tool access:

| Tool                       | Description                                                                                            |
| -------------------------- | ------------------------------------------------------------------------------------------------------ |
| `fetch_url`                | Fetch URL to text                                                                                      |
| `read_file`                | Read file (max 100KB)                                                                                  |
| `write_file`               | Write file                                                                                             |
| `list_directory`           | List directory contents                                                                                |
| `glob`                     | Find files by pattern                                                                                  |
| `grep`                     | Search file contents with a regex (case-insensitive by default, optional path + include glob)          |
| `run_python`               | Execute Python 3 code                                                                                  |
| `run_command`              | Execute shell command (120s timeout)                                                                   |
| `diff`                     | Unified diff between files                                                                             |
| `patch`                    | Apply unified diff                                                                                     |
| `edit_file`                | Precise string replacement                                                                             |
| `apply_patch`              | Apply diff with file headers                                                                           |
| `kubernetes_cluster_query` | Read-only OpenShift/Kubernetes cluster consultation (online `oc`/`kubectl`, offline `omc` must-gather) |

Subcommands: `/agentic auto`, `plan [on|off|run]`, `sandbox`, `verbose`, `thinking`, `trace`, `log`, `lazytool`, `acl`, `iterations <N>`, `timeout <N>`.

`/agentic plan` enables read-only planner mode. The default (strict) tier restricts
the tool surface to `read_file`, `glob`, `grep`, `list_directory`, `fetch_url`,
`diff`, and `kubernetes_cluster_query`, with **no shell**, and confines every path
to the current working directory — requests outside it (e.g. `read_file /home/...`)
are refused outright, never prompted. `/agentic plan run` is the deeper tier: it
re-adds `run_command`, which always asks for confirmation (bypass-immune), for
shell-only inspection. In both tiers the model can only query, never modify.

Safety features: destructive tool confirmation, path ACL (see below), shell approval gate, step timeout (default 300s), stuck detection, same-tool-loop abort (after 10 consecutive identical calls), optional container sandbox (podman/docker).

### Access Control (Path ACL)

File tools (`read_file`, `write_file`, `edit_file`, `list_directory`) and
`run_command` operands are governed by a session-scoped Path ACL:

| Path                              | read   | write   |
| --------------------------------- | ------ | ------- |
| Current working directory         | allow  | allow   |
| `$HOME` (explicit `~` / absolute) | allow  | deny    |
| `$HOME` dotfiles (`~/.*`)         | ask    | ask     |
| `/proc`, `/sys`, `/etc`           | ask    | deny    |
| anything else outside CWD/`$HOME` | deny   | deny    |

- `ask` prompts per access and is never auto-approved (bypass-immune to
  `/agentic auto`); non-TTY runs deny and log it.
- Sensitive virtual files (`/proc/*/mem`, `environ`, `kcore`, `/proc/*/fd/*`) are
  hard-blocked regardless of rules.
- `run_command` is now path-aware: `cat /etc/passwd`, `head /sys/...` etc.
  require approval even though `cat`/`head` are otherwise allowed.

Inspect and influence the policy with `/agentic acl`:

```text
/agentic acl                    # list current rules
/agentic acl log                # recent allow/deny decisions
/agentic acl allow /etc read    # allow reads under /etc this session
/agentic acl deny /proc read    # deny /proc reads this session
/agentic acl ask /sys read      # restore prompting for /sys
/agentic acl remove /etc        # drop session rules for a path
/agentic acl reset              # restore defaults
```

Defaults apply per session; `reset` restores them. Longest path prefix wins,
and session rules override defaults.

**Not covered by the Path ACL** — these rely on the older CWD-containment
check plus the destructive-tool / shell approval gates instead:
- `run_python` **inline `code`** — the code is inlined into `python3 -c <code>`
  and cannot be scanned for operands, so e.g. `open('/etc/passwd')` inside
  Python is not ACL-checked (use the `file_path` form to get operand
  coverage).
- `diff` and `apply_patch` — paths embedded in the arguments / patch text are
  resolved with plain CWD containment only (no ask).
- `glob` — restricted to the working directory by design.

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

### Running Agentic in a Container Sandbox

`/agentic sandbox` toggles shell tool execution (`run_command`, `run_python`)
into a throwaway container (podman/docker, image
`python:3.12-alpine` by default — override with `OLLAMAQUERY_CONTAINER_RT` /
`OLLAMAQUERY_CONTAINER_IMAGE`). Your working directory is mounted read-write
at `/workspace`; the rest of the system is isolated.

```text
> /agentic on
Agentic mode enabled.

> /agentic sandbox
Executor mode: container

> what OS does this container run?
[Agentic] Step 1/50…
[Tool] run_command(command='cat /etc/os-release | head -1')

NAME="Alpine Linux"
```

Notes:
- Only `run_command` / `run_python` run inside the container; the file tools
  (`read_file`, `write_file`, ...) still operate on host paths and stay
  governed by the Path ACL.
- Container commands skip the shell approval gate and the path-ACL operand
  scan (the container is the sandbox — `cat /etc/passwd` reads the
  container's `/etc`).
- Requires a working podman/docker setup and network to pull the image on
  first use.

### MCP servers (tool providers)

Local [Model Context Protocol](https://modelcontextprotocol.io) servers can
contribute tools over stdio. Point at a config with `--mcp-config <path>`, or
place `.ollamaquery-mcp.json` in the working directory (or
`~/.ollamaquery.d/mcp.json`):

```json
{
  "version": 1,
  "servers": {
    "calc": {
      "transport": "stdio",
      "command": ["/absolute/path/to/mcp_test_server"],
      "tool_prefix": "calc"
    }
  }
}
```

```text
/mcp status                     # configured servers and connection state
/mcp tools                      # tools this build would expose
/mcp tools --plan               # only tools eligible for /agentic plan
/mcp connect calc               # start it now (lazy servers start on first call)
/mcp help                       # subcommands (also shown by bare /mcp)
```

Tools are namespaced `<prefix>__<tool>` (e.g. `calc__add`). Config paths expand
`~`/`${VAR}` and relative paths resolve against the config file's directory, so
a config works from any working directory. Only the **stdio**
transport and the **tools** capability are implemented; HTTP/OAuth,
prompts/resources, sampling/elicitation/roots and `list_changed` fail closed.
In plan mode an MCP tool is eligible only when the server explicitly declares
`readOnlyHint: true` — an absent annotation is never treated as safe.

See `doc/mcp.md` for the full field reference, the security model, and a guide
to building a local test server with the `Neumann-Labs/mcp-cpp` SDK.

## Security model

Trust boundaries at a glance; the sections above give the details.

- **Shell gate.** Per-component allow/ask/deny, `rm ~/` / `rm /` hard-blocked
  (bypass-immune), CWD leniency, and a path-ACL operand scan. The `s` and `a`
  approval answers are **session-scoped**; nothing is written to disk.
- **`/agentic auto` is host-bash trust.** It auto-approves ask-level shell
  commands and skips destructive-tool confirmations. Home/root hard-denies and
  Path-ACL `ask` rules still apply, but the model can otherwise execute on the
  host without a prompt.
- **Container sandbox is a different, looser boundary.** `run_command` /
  `run_python` execute inside the container and skip the host shell gate and
  ACL operand scan. The working directory is bind-mounted read-write; the file
  tools still run on the host and stay under the Path ACL.
- **Path ACL coverage.** File tools and shell operands are governed (see
  "Access Control" above). `run_python` inline code, `diff`/`apply_patch`
  contents, and `glob` are not scanned.
- **Network.** `fetch_url` and `/curl` accept `http://` and `https://` only —
  `file://`, `ftp://` and `data:` are rejected, so URL fetch cannot read local
  files. Private/link-local address blocking is **not** implemented; the client
  can reach whatever the host can.
- **On disk.** Sessions, readline history, agentic JSONL logs and spill files
  are created `0600` in `0700` directories. They can contain anything the model
  or a `!` command saw, including secrets.
- **Plan mode.** `/agentic plan` (strict) exposes inspection tools only, with
  no shell and every path confined to the CWD. `/agentic plan run` re-adds
  `run_command`, which always confirms.

## Testing

Run from this directory:

```bash
python3 -m unittest discover tests -v
```

Offline suites need no backend and no network: `test_acl`, `test_mcp`,
`test_compaction`, `test_compact_command`, `test_agentic_timeout`,
`test_tool_summarize`, `test_state_files`. The backend-gated suites skip
automatically when no server is reachable.

- Per-file matrix, env vars and contributor rules: `tests/README.md`
- Coverage map, release checklist and result interpretation: `doc/testing.md`

## Requirements

- Python 3.x
- A running LLM backend: a local Ollama / Llama.cpp / Strata / LM Studio server,
  or an API key for Gemini, OpenCode Zen/Go, Mistral or DeepSeek
- Tested on Fedora 43 and Ubuntu 24

## System Prompt

Default prompt: accurate chatbot mirroring the user's language, formatting
responses with markdown. Built-in profiles (select via `-P` / `--profile`):

| Profile    | Focus                                                      |
| ---------- | ---------------------------------------------------------- |
| `default`  | General-purpose, language-mirroring chatbot                |
| `coder`    | Python/C++ specialist with system engineering knowledge    |
| `sysadmin` | Linux admin, short and direct answers                      |
| `concise`  | No filler, skip pleasantries                               |
| `doctor`   | Medical info with disclaimer (not a licensed professional) |
| `teacher`  | Patient educator with analogies and adaptable depth        |
| `politic`  | Neutral analyst, multiple perspectives, no endorsements    |

```bash
python3 ollamaquery2.py -c -P coder
python3 ollamaquery2.py -c -p "You are a helpful assistant that speaks like a pirate."
```

In agentic mode, the prompt is replaced with a model-specific one built
from composable blocks (role, tool definitions, format, examples, rules).

## Environment

| Variable                 | Default                  | Description                        |
| ------------------------ | ------------------------ | ---------------------------------- |
| `OLLAMA_HOST`            | `http://127.0.0.1:11434` | Ollama API URL                     |
| `LLAMACPP_HOST`          | `http://127.0.0.1:8080`  | Llama.cpp API URL                  |
| `STRATA_HOST`            | `http://127.0.0.1:8080`  | Strata API URL                     |
| `OLLAMAQUERY_THEME`      | `default`                | Color theme name                   |
| `NO_COLOR`               | (unset)                  | Disable colors                     |
| `GEMINI_API_KEY`         | (unset)                  | Gemini cloud backend key           |
| `OPENCODEZEN_API_KEY`    | (unset)                  | OpenCode Zen cloud backend key     |
| `OPENCODEGO_API_KEY`     | (unset)                  | OpenCode Go cloud backend key      |
| `MISTRAL_API_KEY`        | (unset)                  | Mistral cloud backend key          |
| `DEEPSEEK_API_KEY`       | (unset)                  | DeepSeek cloud backend key         |
| `OLLAMAQUERY_MCP_CONFIG` | (unset)                  | MCP config path (see `doc/mcp.md`) |

## Similar Projects

See `doc/similar_project.md` for a detailed comparison with Aider, Open Interpreter, ShellGPT, and Fabric — the closest alternatives depending on whether you need agentic coding, pipeline CLIs, or sandboxed execution.

## References

| Reference                                                       | What it is                                                                                                                           |
| --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| [MCP specification](https://modelcontextprotocol.io)            | Protocol source of truth; the built-in MCP client targets revision `2025-11-25`                                                      |
| [Neumann-Labs/mcp-cpp](https://github.com/Neumann-Labs/mcp-cpp) | Spec-faithful C++20 MCP SDK. Used as a wire-format oracle; check its license before vendoring source                                 |
| `doc/mcp.md`                                                    | MCP configuration reference, security model, and local test-server guide                                                             |
| `doc/mcp-conformance.md`                                        | Conformance review of the built-in MCP client against mcp-cpp: findings, capability matrix, fix plan                                 |

## License

GPL-3.0. See the `LICENSE` file at the repository root.
