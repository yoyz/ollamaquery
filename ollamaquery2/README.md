# ollamaquery2

A feature-rich CLI for LLMs via local Ollama, Llama.cpp, LM Studio and Strata servers, plus cloud OpenAI-compatible APIs (Gemini, OpenCode Zen/Go, Mistral, DeepSeek) and MaaS gateways in front of vLLM. Single Python file, no pip install needed.

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
| MaaS           | (none — the `https` origin is deployment-specific)          | `MAAS_API_KEY` / `MAAS_API_KEY_FILE`    |

Auto-detects available **local** backends. Cloud backends are selected with
`-b` and read their API key from the environment (`GEMINI_HOST`,
`OPENCODEGO_HOST`, etc. override the host).

Prefer `--api-key-file <0600 file>` or the backend's env var: `-k/--api-key` is
visible in `argv` (`/proc/<pid>/cmdline`) and prints a warning when it is the
value actually used. Precedence is file > env var > `-k`.

Strata shares Llama.cpp's default port (8080) and OpenAI dialect but streams
reasoning via `delta.reasoning_content` and exposes Anthropic
`/v1/messages/count_tokens`; it is recognized by its unique `/health` marker.

MaaS is an OpenShift AI / KServe gateway in front of vLLM that routes only
`/v1/models` and `/v1/chat/completions`: the origin is supplied with `-H` or
`MAAS_HOST` (a trailing `/v1` is accepted and stripped), model ids are `publishers/<publisher>/models/<name>` (short names
are resolved against the listing), and the context window is not discoverable —
set it with `--context-size` or `/contextsizeset`. See `doc/backend_maas.md`.

## Quick Start

```bash
# Chat mode (auto-detects backend)
python3 ollamaquery2.py -c
```

### CLI Reference

```text
-c                  Interactive chat mode
-m <model>          Model name
-b <backend>        Backend: ollama, llamacpp, strata, lmstudio, gemini, opencodezen, opencodego, mistral, deepseek, maas
--context-size <n>  Context window override (needed for MaaS, which exposes none)
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
| `/promptshow`                | Show the outbound system prompt + tools, read-only (`system`, `tools [name]`, `json`, `save <path>`)                         |          |                                                    |        |                                        |
| `/contextsizeset <N>`        | Override context window                                                                                                      |          |                                                    |        |                                        |
| `/quit` `/exit`              | End session (`Ctrl+C` twice / `Ctrl+D` also works)                                                                           |          |                                                    |        |                                        |

Unknown `/commands` are **answered by the harness, never sent to the model**: a
typo like `/aaaaa` prints an error with the closest command names and a pointer
to `/help`, without spending a token. Paths (`/tmp/foo`) and prose are still sent
as queries; to send a line that *starts* with `/` to the model, type `//` first
(`//aaaaa` → the model receives `/aaaaa`).

### Sessions

Conversations are auto-saved on exit; `/save` names them so a later `/resume`
does not need a timestamp:

| Command             | Description                                                                 |
| ------------------- | --------------------------------------------------------------------------- |
| `/save`             | Save now — updates the session this conversation is already bound to        |
| `/save <name>`      | Save under `<name>`; saving again under the same name updates that session  |
| `/sessions [count]` | List saved sessions (label, model, messages, summary; `*` = active session) |
| `/resume <session>` | Load by id (exact or prefix) or by name (`/resume acl` is enough)           |
| `/rename <name>`    | Rename the session this conversation saved, keeping its id                  |

A name is normalized to a slug (`Réunion ACL` → `reunion-acl`), stored both as a
label and in the id (`20261113_152947_reunion-acl`, so transcripts are readable
in `~/.ollamaquery.d/sessions/`), and matched with accents, case and spacing
folded. Tab completion on `/resume ` offers every saved slug and id. The newest
50 sessions are kept (dir `0700`, files `0600`).

### Agentic Mode

`/agentic on` enables a ReAct loop with tool access:

| Tool                       | Description                                                                                            |
| -------------------------- | ------------------------------------------------------------------------------------------------------ |
| `fetch_url`                | Fetch URL to text                                                                                      |
| `read_file`                | Read file (max 100KB)                                                                                  |
| `write_file`               | Write file                                                                                             |
| `list_directory`           | List directory contents                                                                                |
| `glob`                     | Find files by pattern                                                                                  |
| `grep`                     | Search file contents with a regex (case-insensitive; recursive searches skip `.git`; capped output)    |
| `run_python`               | Execute Python 3 code                                                                                  |
| `run_command`              | Execute shell command (120s timeout)                                                                   |
| `diff`                     | Unified diff between files                                                                             |
| `patch`                    | Apply unified diff                                                                                     |
| `edit_file`                | Precise string replacement                                                                             |
| `apply_patch`              | Apply diff with file headers                                                                           |
| `kubernetes_cluster_query` | Read-only OpenShift/Kubernetes cluster consultation (online `oc`/`kubectl`, offline `omc` must-gather) |
| `git_inspect`              | Read-only git inspection: `status`, `diff`, `diff_staged`, `diff_head`, `log` (any ref, path-scoped, renames followed), `show`, `blame -L` (fixed argv, no shell) |
| `path_policy`              | Read-only: the session path map (allowed / needs confirmation / forbidden), or the verdict for one path — no filesystem access, no prompt |

Subcommands: `/agentic on|off|full`, `preset [read|exec|work|trusted]`, `plan [on|off|run]`, `sandbox [host|container|bwrap|status|net shared|none|reset]`, `auto`, `verbose`, `thinking`, `trace`, `log`, `lazytool`, `acl`, `scratch [path|list|clear]`, `iterations <N>`, `timeout <N>`, `logretention <days>`, `status`.

The model is steered to use the **dedicated file tools** (`read_file`, `write_file`,
`edit_file`, `grep`, `glob`, `list_directory`) instead of shell equivalents (`cat`,
`head`, `sed -i`, `echo >`, `find`): they are path-ACL checked, their output is not cut
off by the generic observation cap (`read_file` pages large files and returns a `next`
offset), and they always act on the host project — including in sandbox mode, where a
shell `cat` would read the container's copy. `run_command` stays for what it is good at:
compilers, builds, tests, binaries, system inspection.

**Your profile stays.** `-P teacher` (or a custom system prompt) is **composed** with
the agentic protocol rather than replaced: the persona leads (voice, language, depth)
and the protocol blocks follow (tools, ReAct), in one system message — the persona is
user configuration, the protocol is mechanics. `/agentic off` restores the persona
alone, and `/promptshow system` shows the composed prompt that is actually sent.

`/agentic plan` enables read-only planner mode. The default (strict) tier restricts
the tool surface to `read_file`, `glob`, `grep`, `list_directory`, `fetch_url`,
`diff`, `kubernetes_cluster_query`, `git_inspect`, and `path_policy`, with **no shell**, and
confines every path to the current working directory — requests outside it (e.g.
`read_file /home/...`) are refused outright, never prompted. `/agentic plan run`
is the deeper tier: it re-adds `run_command`, which always asks for confirmation
(bypass-immune), for shell-only inspection. In both tiers the model can only
query, never modify.

Safety features: destructive tool confirmation, path ACL (see below), shell approval gate, step timeout (default 300s), stuck detection, same-tool-loop abort (after 10 consecutive identical calls), and a rootless bwrap sandbox (default for agentic) or a container sandbox (podman/docker).

#### Presets — the stance, not just a toggle

An agentic session has four presets. A preset is a **quadruplet**: the intent put
in the prompt, the tool surface, the approval rule, and the sandbox mount policy.
The same resolver drives all four, so the prompt, the native `tools` schema, the
backstop and the `Executor` cannot disagree.

| Preset                       | Surface                                                                                      | Sandbox                               | Promise                                                   |
| ---------------------------- | -------------------------------------------------------------------------------------------- | ------------------------------------- | --------------------------------------------------------- |
| **`read`** (alias `plan`)    | inspection tools only, no shell                                                              | none                                  | no write, no code                                         |
| **`exec`**                   | read + `run_command`/`run_python`, **no** write tools, no `fetch_url`, no cluster tool        | bwrap, project **ro**, net **none**   | runs commands, cannot modify the project                  |
| **`work`** (default)         | every tool                                                                                   | bwrap, project **rw**, net **shared** | project writable; host binaries/config mounted **ro** (the real `$HOME` is not) |
| **`trusted`** (alias `full`) | every tool                                                                                   | **host**, auto-confirm                | everything allowed, only the hard denies                  |

```text
/agentic preset                 # list the presets and the active one
/agentic preset exec            # read + run, project read-only, no network
/agentic preset show            # the active quadruplet + the effective tool surface
/agentic preset work            # the daily stance (first `/agentic on` selects it)
/agentic plan                   # alias for the read preset
/agentic plan run               # adds run_command on the HOST, confirmed per call
/agentic full                   # alias for the trusted preset
```

`/agentic auto` keeps its historical meaning (toggle auto-confirm); the preset
alias `auto` is available as `/agentic preset auto` = `trusted`.

`/agentic on` applies **`work` on its first activation**, then resumes the last
stance: an `exec` or `trusted` session survives `/agentic off` → `/agentic on`
instead of silently becoming `work`. `/agentic plan run` is the deeper plan tier
(the resolver reports it as the internal `read-run` stance): it re-adds
`run_command` **on the host** with a confirmation for every call — plan mode
never runs a shell inside a sandbox, where the gate would be bypassed.

Three invariants: **shell writes** are bounded by the bwrap mounts (the kernel
returns EROFS), **file writes** stay governed by the Path ACL, and **host**
execution only happens in `trusted`, in `/agentic plan run` (one confirmed
command at a time), or with an explicit `--sandbox-engine host`.
File tools run in the harness, not in the sandbox; `exec` therefore removes the
write tools from the surface instead of pretending the mount protects them.
`kubernetes_cluster_query` also runs in the harness (it needs KUBECONFIG,
`~/.kube/config` or a must-gather under `$HOME`, none of which the sandbox
mounts) and can reach a cluster, so `exec` — which promises **no network** —
does not offer it; use `read` to inspect a cluster.


Each step's stream is also checked for **completeness**: a turn cut mid-generation
(dropped connection, `finish_reason: length` output cap, malformed frames) or a
native tool call whose arguments don't parse is *continued* with a bounded nudge
("continue where you stopped, do not start over") instead of the partial text
being taken as the final answer — and a step that genuinely ended without a tool
call says so explicitly. A corrupt native tool call is never executed with empty
arguments.

### Access Control (Path ACL)

File tools (`read_file`, `write_file`, `edit_file`, `list_directory`) and
`run_command` operands are governed by a session-scoped Path ACL:

| Path                              | read   | write   |
| --------------------------------- | ------ | ------- |
| Current working directory         | allow  | allow   |
| Session scratch dir (`/tmp/ollamaquery/<pid>`) | allow  | allow   |
| `$HOME` (explicit `~` / absolute) | allow  | deny    |
| `$HOME` dotfiles (`~/.*`)         | ask    | ask     |
| `/proc`, `/sys`, `/etc`           | ask    | deny    |
| `/tmp` (everything but the scratch dir) | deny | deny  |
| anything else outside CWD/`$HOME` | ask    | deny    |

- **Scratch dir.** Each session gets `/tmp/ollamaquery/pid-<pid>/` (mode
  `0700`) as the one writable place outside the project, so the model has a
  definite answer to "where do I put this build output / test binary?" instead
  of probing the filesystem until something is refused. It is removed at exit
  (only the leaf: the parent stays), `OLLAMAQUERY_KEEP_SCRATCH=1` keeps it, and
  `write_file`/`edit_file` there need no destructive-tool confirmation. The rest
  of `/tmp` is denied — it is world-writable and shared with every other process,
  so it cannot be bounded or cleaned up — and a denial names the scratch dir as
  the redirect. `/agentic scratch` prints the path, `list` shows its contents,
  `clear` empties it. The container sandbox mounts it at the same absolute path
  (a running container that cannot be re-mounted gets a one-time warning).

- `ask` prompts per access and is never auto-approved (bypass-immune to
  `/agentic auto`); non-TTY runs deny and log it.
- **What the model sees, and when.** The agentic system prompt states the
  *positive* half — the roots it may use, plus the grants made this session —
  and points at the `path_policy` tool for everything else. A refusal is then one
  line, not a wall of rules:
  `[path ACL denied] read /var/log/syslog — outside the project and $HOME, not granted; call path_policy for the allowed paths …`
  The full map (allowed / one-confirmation / forbidden / session grants, deny
  rules first, with the `/agentic acl allow <path> read` remedy) is returned by
  `path_policy`, which is also usable *before* touching a path: `path_policy`
  with `path` reports `ALLOW` / `ASK` / `DENY` for that path without accessing it
  and without triggering a confirmation prompt. The shared agentic rules block
  makes that reflex explicit: a denial means stop, call `path_policy` once, then
  use an allowed path — never reroute through `run_command`/`run_python`.
- Reads outside the project are `ask`, not `deny`, on purpose: the gate is a
  consent + audit layer, not a sandbox. The shell can read the same file anyway,
  so a silent deny only pushed the model onto the least-audited channel. The
  shared temp area is the deliberate exception: `/tmp` is world-writable, shared
  with every process, and impossible to clean up, so it is denied outright and
  the model is redirected to its own scratch dir.
- Sensitive virtual files (`/proc/*/mem`, `environ`, `kcore`, `/proc/*/fd/*`) are
  hard-blocked regardless of rules.
- `run_command` is now path-aware: `cat /etc/passwd`, `head /sys/...` etc.
  require approval even though `cat`/`head` are otherwise allowed. Absolute
  paths hidden inside a code blob are seen too — `python3 -c
  "open('/etc/passwd')"`, `sh -c 'cat /etc/passwd'`, `dd if=/etc/shadow` are
  caught, while prose that merely names a path (`git commit -m 'fix /etc
  handling'`) is not scanned. Shell variables follow bash: an **unset** simple
  variable expands to the empty string, so `echo x > $UNSET/tmp/x` is a
  `/tmp` write and is denied; a **defined** variable stays opaque (documented
  limit), and a single-quoted spelling is a literal name.
- Host and bwrap commands receive `TMPDIR` = the session scratch
  (`/tmp/ollamaquery/pid-N`), so build temporaries go there; the scratch is
  wiped at exit, and a hardcoded `/tmp/...` write is denied. Container
  commands keep the container's own temp dir.
- The shell gate follows `cd`/`pushd`/`popd`: a relative **write** target is
  resolved against the directory the shell is actually in, so
  `cd / && echo x > rel` is `/rel` (denied). A `cd` inside a pipeline, `&` or
  `$(...)`/`(...)` stays in its subshell, and an opaque `cd $VAR` refuses
  relative writes (fail-closed).

Inspect and influence the policy with `/agentic acl`:

```text
/agentic acl                    # list current rules
/agentic acl log                # recent allow/deny decisions
/agentic acl allow /etc read    # allow reads under /etc this session
/agentic acl deny /proc read    # deny /proc reads this session
/agentic acl ask /sys read      # restore prompting for /sys
/agentic acl remove /etc        # drop session rules for a path
/agentic acl reset              # restore defaults
/agentic scratch                # this session's writable temp dir (list|clear)
```

Defaults apply per session; `reset` restores them. Longest path prefix wins,
and session rules override defaults.

**Not covered by the Path ACL** — these rely on the older CWD-containment
check plus the destructive-tool / shell approval gates instead:
- `run_python` **inline `code`** — the code is written to a session-scratch
  file and run as `python3 <file>` (so it is not exposed on `/proc/<pid>/cmdline`),
  but its body is still opaque to the ACL, so e.g. `open('/etc/passwd')` inside
  Python is not ACL-checked (use the `file_path` form to get operand coverage).
- `diff` and `apply_patch` — paths embedded in the arguments / patch text are
  resolved with plain CWD containment only (no ask).

`glob` **is** covered: every match is realpath-confined (a project symlink
cannot list outside the project) and path-ACL checked; an `ask` match is simply
not listed.

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

### Sandbox engines: bwrap (default for agentic), container, host

`run_command` / `run_python` run in one of three engines. The `work` and `exec`
presets select **bwrap**; `trusted` selects **host**; `--sandbox-engine` is the
explicit override (CLI beats env). bwrap is rootless, daemon-free and starts
instantly — no image, no pull, no container runtime.

```text
> /agentic on
[Agentic preset: work — tools=all sandbox=bwrap approve=auto]
[Sandbox] bwrap (project rw, net shared, host ro; no cgroup cap (v1)]

> /agentic preset exec
[Agentic preset: exec — tools=9 read/exec sandbox=bwrap approve=auto]
[Sandbox] bwrap (project ro, net none, host ro; no cgroup cap (v1))
```

The mount policy **is** the contract in this mode (the shell gate is bypassed,
like a container):

| Path / object                                                             | Mount                                                                     | Why                                                                |
| ------------------------------------------------------------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| `/usr`, `/bin`, `/sbin`, `/lib`, `/lib64`, `/etc`, `/var`, `/opt`         | `--ro-bind-try`                                                           | host binaries and config                                           |
| `/run/systemd/resolve`, `/run/NetworkManager`                             | `--ro-bind-try`                                                           | resolver state only — the rest of `/run` (sockets!) is **not** mounted |
| `/tmp`, `/var/tmp`                                                        | `--tmpfs`                                                                 | disposable; the session scratch is re-bound inside                 |
| project (CWD)                                                             | `--bind` rw (work) or `--ro-bind` (exec)                                  | same absolute path as the host, so model paths stay coherent       |
| session scratch `/tmp/ollamaquery/pid-N`                                  | `--bind` rw                                                               | `/agentic scratch` keeps working                                   |
| `~/.ollamaquery.d/sandbox/pid-N/home`                                     | `--bind` rw                                                               | pip/npm caches; **the real `$HOME` is never mounted**              |
| `/dev`, `/proc`                                                           | `--dev`, `--proc`                                                         | the namespace's own                                                |
| network                                                                   | `--share-net` (work) or `--unshare-net` (exec)                            | builds/installs vs no DNS/no downloads                             |
| isolation                                                                 | `--unshare-all --die-with-parent --new-session --cap-drop ALL --clearenv` | namespaces, caps, minimal env                                      |

> **`net none` is absolute.** A read-only mount does not stop `connect(2)` on a
> Unix socket, so `/run` is *not* mounted whole (a sandboxed command could
> otherwise drive `podman.sock`/`dbus` — measured). And the harness itself has
> no side door under `net none`: `fetch_url` and `kubernetes_cluster_query` are
> removed from the tool surface, and a `curl` GET is no longer promoted to a
> host-side fetch. Restore the network when you need it:

```bash
/agentic sandbox net status   # current stance + where it comes from
/agentic sandbox net shared   # network back on, immediately
/agentic sandbox net none     # cut it again
/agentic sandbox net reset    # hand the decision back to the active preset
```

`shared`/`none` are explicit overrides (they survive preset changes, like a CLI
flag); `reset` clears the override. The `exec` preset refuses `shared` — switch
preset first — and plan mode refuses any change until it ends.

Four cases are **refused by construction** rather than silently weakened:
`/` as the project (it would mount the whole host), a read-only project that
contains the session scratch (`TMPDIR=$PWD/tmp`) or the sandbox HOME
(`CWD=$HOME`) — both are re-bound writable after the ro-bind — and, in `exec`,
any rw bind that overlaps the project. The preset banner prints the reason and
the command returns `returncode -1` instead of running somewhere else.

```bash
python3 ollamaquery2.py -c --sandbox-engine bwrap \
  --sandbox-bwrap-project ro --sandbox-bwrap-rw "$PWD/build" \
  --sandbox-bwrap-net none --sandbox-bwrap-bind /data:ro
# work/trusted accept the rw subtree; `exec` refuses it (its project is ro)

/agentic sandbox bwrap|container|host|status   # switch/inspect at runtime
/agentic sandbox reset                         # clear the engine override, back to the preset
/agentic sandbox net shared|none|reset|status  # network stance at runtime
```

Limits, stated rather than implied: **no cgroup cap** in v1 (a fork bomb or a
huge build is not contained; use the container for that), host binaries only
(no package installation), Linux with unprivileged user namespaces, no
dedicated seccomp profile, and shared network = the sandbox can exfiltrate.
The probe is **fail-closed**: if bwrap is missing or user namespaces are
refused, tool commands are refused with a clear message — never silently run on
the host. `--sandbox-engine host` (or `trusted`) is the explicit way to run
unsandboxed. Full detail: `doc/agentic_bwrap.md`.

### Running Agentic in a Container Sandbox

`/agentic sandbox` toggles shell tool execution (`run_command`, `run_python`)
into a container. Two flavours:

**A throwaway container per command** (default). Your working directory is
mounted read-write at `/workspace`, the rest of the system is isolated, and
each command gets a fresh `--rm` container — so nothing installed in one
command survives the next, only `/workspace` persists.

```text
> /agentic on
Agentic mode enabled.

> /agentic sandbox
[Executor mode: container — image docker.io/library/python:3.12-alpine, 1 cpu, 1g memory]

> what OS does this container run?
[Agentic] Step 1/50…
[Tool] run_command(command='cat /etc/os-release | head -1')

NAME="Alpine Linux"
```

**An already-running container** (`exec`). Whatever you prepared inside it —
compilers, packages, credentials, its own mounts and state — is what the model
uses, and nothing is created or destroyed by the tool:

```bash
# start a box you control, with your project mounted
podman run -d --name dev-box -v "$PWD":/src -w /src my-dev-image sleep infinity

# then point ollamaquery2 at it (implies container mode)
python3 ollamaquery2.py -c --sandbox-container dev-box
```

`--sandbox-container` also asks the runtime which of the container's mounts
contains your current directory and runs commands there (`-w`), so the model's
relative paths hit your files. With no matching mount it keeps the container's
own workdir and relative writes land inside the container.

Selection (CLI beats env beats default):

| Setting | CLI | Environment | Default |
|---|---|---|---|
| Image | `--sandbox-image IMAGE` | `OLLAMAQUERY_CONTAINER_IMAGE` | `docker.io/library/python:3.12-alpine` |
| Running container | `--sandbox-container NAME` | `OLLAMAQUERY_CONTAINER_NAME` | — |
| Runtime | `--sandbox-runtime podman\|docker` | `OLLAMAQUERY_CONTAINER_RT` | `podman` |
| Memory ceiling | `--sandbox-memory SIZE` | `OLLAMAQUERY_CONTAINER_MEMORY` | `1g` |
| CPU ceiling | `--sandbox-cpus N` | `OLLAMAQUERY_CONTAINER_CPUS` | `1` |

Passing `--sandbox-image` or `--sandbox-container` starts the session in
container mode (the startup banner says which target; `/agentic sandbox`
switches back to host). The image is probed once per session and pulled only
if it is absent — a locally built `localhost/...` image is never pulled (podman
would treat `localhost/` as a registry and fail).

Notes:
- Only `run_command` / `run_python` run inside the container; the file tools
  (`read_file`, `write_file`, ...) still operate on host paths and stay
  governed by the Path ACL. That is also how a file reaches the sandbox: write
  it on the host in the mounted directory.
- Container commands skip the shell approval gate and the path-ACL operand
  scan (the container is the sandbox — `cat /etc/passwd` reads the
  container's `/etc`). With `--sandbox-container` the model runs commands
  inside *your* container, so point it at a disposable one.
- The throwaway container is capped at **1 CPU and 1 GiB** by default
  (`--memory 1g --cpus 1`), so a runaway command cannot take the host down;
  raise it with `--sandbox-memory 4g --sandbox-cpus 2`. The ceiling is shown
  wherever the mode is (`/agentic status`, the startup banner, the
  `/agentic sandbox` message). With `--sandbox-container` no ceiling is
  applied — that container's limits are whoever started it.
- Network: with podman's default (pasta) the container shares the host's
  addresses but has its own loopback, so `127.0.0.1` inside is **not** the
  host's loopback.
- Requires a working podman/docker setup (and network only if the image is
  missing).

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

- **Shell gate.** Per-component allow/ask/deny, `rm ~/` / `rm /`,
  home/root `find -delete` and destructive `find -exec` / `xargs rm`
  hard-blocked (bypass-immune, `auto_confirm` included),
  effective-command normalization
  (wrappers, `VAR=value`, `/bin/rm`), CWD leniency and a path-ACL operand scan.
  Leniency only covers project paths: `sh` / `curl … | sh` ask, and a `cd`
  outside the project disables it; a `cd` into home/root/tmp (or an ancestor)
  followed by a delete is hard-denied. A wrapper with a command (`env sh -c …`)
  asks even though `env *` is allow-listed. **Nested commands are inspected:**
  a `-c`/`env -S` payload, a `find -exec` action and the command `xargs` will
  run are parsed, so `find . -exec rm -rf ~ \;` is a hard deny and a relative
  write inside `bash -c 'cd / && echo x > rel'` is denied by the path ACL (it
  resolves against the directory the payload actually runs in; an unknown
  directory refuses the write). `find -execdir`/`-okdir` run in each matched
  file's directory, not the `find` cwd, so their relative writes are refused
  (with `-L` a symlink can leave the project) and a destructive payload there
  is a hard deny; `-exec` keeps the inherited cwd. The inspection is bounded at 6 nesting levels,
  so a relative write buried deeper is a documented limit (absolute paths are
  still seen); see `doc/agentic_shell_validation.md` §2.2. The write-target
  table now covers the common allow-listed writers with a visible destination
  (`git clone`/`-C`, `tar -x -C`, `sort -o`, `awk -i inplace`, `perl -i`,
  `patch -d`, `npm`/`yarn`/`pnpm --prefix`, `unzip -d`, `gzip`, `rsync`,
  `pip --target`, `cargo --target-dir`, `go -o`, `make -C`), so
  `git clone … /var/lib/x` or `cd /var/lib && npm install` hit the same write
  deny as `cp f /var/lib/x`. GNU tar's old no-dash options (`tar xPf`,
  `tar xCf /var/lib …`) are normalized before grading. **User installs ask:**
  `pip install --user` (`~/.local`), `cargo install` (`~/.cargo`/`--root`) and
  `go install` (`~/go/bin`, via the `~/go` and `~/bin` `home_tools` rules,
  which yield to the session CWD). **Arbitrary script/build execution**
  (`python script.py`, `node script.js`, `make` recipes, `npm run`, cargo
  build scripts) can still write wherever its code decides — use bwrap
  (`work`/`exec`) or the explicit `trusted` stance; see `TODO_security.md`
  H6. **Host code launchers ask:** `python`/`python3`, `node`/`nodejs`,
  `java`, `javac`, `make`, `npm`, `pip`/`pip3`, `cargo`, `go` and `git` are
  `ask` (CWD leniency does not apply), with only the narrow Git read forms
  (`status`, `log`, `diff`, `show`, `ls-files`, `grep`, `rev-parse`,
  `tag -l`, `branch -a`, `remote -v`, `config --get`) allowed; bwrap skips the
  gate and `trusted` auto-confirms. The
  `s` and `a` approval answers are **session-scoped**; nothing is written to
  disk. Broader allow rules (`git -c`, `tar --checkpoint-action`, upload
  `curl`) remain allow-listed, and a `find -exec` with a harmless payload still
  runs without a prompt: the gate is consent + audit, not a sandbox.
- **`/agentic auto` is host-bash trust.** It auto-approves ask-level shell
  commands and skips destructive-tool confirmations. Home/root hard-denies and
  Path-ACL `ask` rules still apply, but the model can otherwise execute on the
  host without a prompt.
- **Container and bwrap sandboxes are a different, looser boundary.** `run_command` /
  `run_python` execute inside the sandbox and skip the host shell gate and ACL
  operand scan. Containers bind the project read-write; bwrap does too in `work`
  and read-only in `exec`. **`exec` bounds writes, not reads**: host binaries,
  libraries and config are mounted read-only, so `cat /etc/...` works inside the
  sandbox even though `read_file` on the same path is denied by the strict path
  rules. The file tools still run on the host and stay under the Path ACL
  (which is why `exec` removes the write tools from the surface).
  bwrap has no cgroup ceiling in v1; the throwaway container keeps `--memory`/`--cpus`.
- **Path ACL coverage.** File tools and shell operands are governed (see
  "Access Control" above). `run_python` inline code and `diff`/`apply_patch`
  contents are still not scanned (`glob` now is).
- **Network.** `fetch_url` and `/curl` accept `http://` and `https://` only —
  `file://`, `ftp://` and `data:` are rejected, so URL fetch cannot read local
  files. Private/link-local address blocking is **not** implemented; the client
  can reach whatever the host can.
- **On disk.** Sessions, readline history, agentic JSONL logs and spill files
  are created `0600` in `0700` directories. They can contain anything the model
  or a `!` command saw, including secrets. Agentic logs are pruned after 7 days
  by default (`/agentic logretention <days>`, or
  `OLLAMAQUERY_AGENTIC_LOG_RETENTION_DAYS`).
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
| `MAAS_HOST`              | (unset)                  | MaaS `https` origin (or pass `-H`) |
| `MAAS_API_KEY`           | (unset)                  | MaaS key (prefer a file)           |
| `MAAS_API_KEY_FILE`      | (unset)                  | Path to a 0600 file holding the key|
| `MAAS_MODEL_TTL`         | `300`                    | Seconds the model listing is cached|
| `OLLAMAQUERY_MCP_CONFIG` | (unset)                  | MCP config path (see `doc/mcp.md`) |

## Similar Projects

See `doc/similar_project.md` for a detailed comparison with Aider, Open Interpreter, ShellGPT, and Fabric — the closest alternatives depending on whether you need agentic coding, pipeline CLIs, or sandboxed execution.

## References

| Reference                                                       | What it is                                                                                                                          |
| --------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| [MCP specification](https://modelcontextprotocol.io)            | Protocol source of truth; the built-in MCP client targets revision `2025-11-25`                                                     |
| [Neumann-Labs/mcp-cpp](https://github.com/Neumann-Labs/mcp-cpp) | Spec-faithful C++20 MCP SDK. Used as a wire-format oracle; check its license before vendoring source                                |
| `doc/mcp.md`                                                    | MCP configuration reference, security model, and local test-server guide                                                            |
| `doc/mcp-conformance.md`                                        | Conformance review of the built-in MCP client against mcp-cpp: findings, capability matrix, fix plan                                |
| `doc/agentic_bwrap.md`                                          | Rootless bwrap execution engine: mount policy, measured limits, presets, CLI/env, test commands                                     |
| `doc/self_ai_review.md`                                         | Read-only self-AI review workflow: freeze a diff bundle, run the `read` preset, budgets/timeouts, verify the findings before acting |

## License

GPL-3.0. See the `LICENSE` file at the repository root.
