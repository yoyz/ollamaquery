# MCP analysis with a real-world application

How `ollamaquery2.py` talks to applications that already speak MCP — and how to add the
next one without writing application-specific code.

Measured 2026-10-22 against **WordCraft 0.9.0** (`2e6db0bd`), client build **v0.3.1**.
Everything labelled *measured* was observed on a live run; everything labelled *by design*
was read out of `ollamaquery2.py` and should be re-verified if that code moves.

---

## 0. TL;DR

1. `ollamaquery2.py` is an **MCP client**. There is no per-application integration, no
   per-application code path, and none should be added.
2. The only per-application artifact is a **JSON file**. Saying "use this app" means
   "point the client at this file":
   `python3 ollamaquery2.py --mcp-config ./app.mcp.json -c`.
3. Three topologies cover every application imaginable: **local stdio**, **stdio over
   ssh**, and **bridge to a running app**. All three end up as a child process on a pipe.
4. The one genuine gap in this build: **Streamable HTTP and SSE transports are parsed but
   refused**. Network-reachable MCP servers need a local shim until that is implemented.
5. Safety is a three-layer ladder (server filters → confirmation gate → read-only plan
   tier) and it is **fail-closed**: a tool that does not declare itself read-only is
   treated as destructive.

---

## 1. The mental model

MCP has two transport families, and they differ in *who is the process*:

```
family A — stdio (this build supports it)

  ollamaquery2 ──pipe(2)──▶ [stdin]  mcp-server-process  [stdout] ──pipe(2)──▶ ollamaquery2
   (parent, client)                    (child, server)                 (parent)

family B — Streamable HTTP / SSE (parse-able here, refused by this build)

  ollamaquery2 ──HTTP POST/SSE──▶ http://host:port/mcp
   (client)                        (server, remote)
```

Four layers, and every integration problem is a question about one of them:

| Layer     | What it is                                                 | Who supplies it    |
| --------- | ---------------------------------------------------------- | ------------------ |
| Client    | `ollamaquery2.py` — JSON-RPC, tool catalog, policy         | us                 |
| Transport | a pipe (stdio) or HTTP                                     | us / the server    |
| Bridge    | optional translator when the app does not speak MCP itself | the app, or a shim |
| App       | the thing with the data and the side effects               | the vendor         |

> **The common misconception.** "MCP is a protocol, so I point the client at the app's
> port." False for stdio: there is no port, no socket, and no loopback traffic between
> client and server — just two inherited file descriptors. A port is involved only when
> the app itself exposes one *and* a bridge translates it (WordCraft is exactly that
> case; see §8).

---

## 2. Three topologies that cover every application

| # | App ships                                      | Topology                                       | Client config is                     | Cost                  |
| --- | ---------------------------------------------- | ---------------------------------------------- | ------------------------------------ | --------------------- |
| A | an MCP server binary/command                   | client **spawns** it on a pipe                 | `command: [...]`                     | lowest                |
| B | an MCP server, but it must run on another host | `ssh host mcp-server` becomes the pipe peer    | `command: ["ssh", "host", "server"]` | one ssh dependency    |
| C | only a control API (HTTP, TCP, WS, CLI)        | a **bridge** process translates MCP → that API | `command: ["bridge", ...]`           | the bridge must exist |

The client cannot tell A, B and C apart. That is the whole design goal: **the client
speaks MCP to a child process, and how that child reaches the data is somebody else's
problem.**

Topology C is the one that looks like it needs per-app work. It does not, as long as
*the vendor or the community already wrote the bridge*. WordCraft ships one
(`wordcraft-cli mcp --connect`), so WordCraft is topology C **with zero code from us**.

---

## 3. Classifying a new application: five questions

Ask these in order. Each answer rules out whole categories.

| # | Question                                     | Answer   | Action                                                                                                              |
| --- | -------------------------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------- |
| 1 | Does it ship an MCP server at all?           | yes      | go to 2                                                                                                             |
|   |                                              | no       | it is topology C: is there a community bridge? If not, **do not integrate** — write the doc entry and stop (see §9) |
| 2 | stdio or HTTP?                               | stdio    | local (A) or ssh (B)                                                                                                |
|   |                                              | HTTP/SSE | this build refuses it; use a stdio shim (§5)                                                                        |
| 3 | Local or remote?                             | local    | `command` directly                                                                                                  |
|   |                                              | remote   | `ssh` the server, or forward the port to loopback                                                                   |
| 4 | Does it need credentials?                    | no       | nothing to do                                                                                                       |
|   |                                              | yes      | `env` with `${VAR}` expansion — never a literal secret in the JSON                                                  |
| 5 | Does it annotate its tools (`readOnlyHint`)? | yes      | plan mode and the confirmation gate work by themselves                                                              |
|   |                                              | no       | every tool lands in the confirmation gate; plan mode gets nothing (fail-closed)                                     |

Question 5 is the one that decides whether an app is *pleasant* or *tedious* to drive.
Measured: WordCraft declares `readOnlyHint` on all 16 tools, so 7 of them are silently
usable in plan mode and 9 are gated.

A GUI-only download is **not** a "no" to question 1: many apps ship their MCP server
inside the same artifact with no launcher for it (an AppImage, a `.app`, a tarball).
See §8.6 for how to get it out.

---

## 4. The config contract

### 4.1 Discovery — first hit wins

| Priority | Source                      | Notes                                                 |
| -------: | --------------------------- | ----------------------------------------------------- |
| 1        | `--mcp-config PATH`         | must exist; a typo is an error, not a silent fallback |
| 2        | `$OLLAMAQUERY_MCP_CONFIG`   | same                                                  |
| 3        | `./.ollamaquery-mcp.json`   | project-local, optional                               |
| 4        | `~/.ollamaquery.d/mcp.json` | user-wide, optional                                   |

`--no-mcp` skips loading entirely. The Claude/VS Code shape is also accepted: `command` +
`args` are folded into one argv, and `"type": "http" | "streamable-http" | "sse"` — or a
bare `url` with no `transport` — is mapped onto an HTTP transport and then **refused**: the
server is listed with a warning and never connected. An unrecognised `type` falls back to
`stdio`.

### 4.2 Keys that matter

From `MCPServerConfig`; defaults in parentheses. `[A]` `[B]` `[C]` mark the topologies
where a key is meaningful.

| Key                                                  | Topology | Meaning                                                                                    |
| ---------------------------------------------------- | -------- | ------------------------------------------------------------------------------------------ |
| `transport` (`stdio`)                                | A B C    | only `stdio` connects in this build                                                        |
| `command` (`[]`)                                     | A B C    | argv list — **not** a shell string; no pipes, no globbing                                  |
| `env` (`{}`)                                         | A B C    | child environment; `${VAR}` and `${VAR:-default}` expanded at load                         |
| `cwd` (`null`)                                       | A B C    | child working directory                                                                    |
| `enabled` (`true`)                                   | all      | `false` hides every tool of the server                                                     |
| `startup` (`lazy`)                                   | all      | `lazy` connects on first agentic/plan entry; `eager` at startup                            |
| `tool_prefix` (server name)                          | all      | model-visible name is `<prefix>__<tool>`                                                   |
| `expose_as` (`{}`)                                   | all      | per-tool rename, overriding the prefix for that tool                                       |
| `allowed_tools` (`[]`)                               | all      | allow-list; empty means "all"                                                              |
| `denied_tools` (`[]`)                                | all      | deny-list; **deny beats confirm**                                                          |
| `read_only_only` (`false`)                           | all      | drop every tool not explicitly `readOnlyHint: true`                                        |
| `confirm_tools` (`[]`)                               | all      | force the confirmation gate for named tools                                                |
| `init_timeout_seconds` (15)                          | A B C    | `initialize` deadline                                                                      |
| `list_timeout_seconds` (30)                          | A B C    | `tools/list` deadline                                                                      |
| `call_timeout_seconds` (120)                         | A B C    | `tools/call` deadline                                                                      |
| `max_result_chars` (100000)                          | all      | ceiling per result; the context window lowers it further                                   |
| `retries` (1), `backoff_seconds` (0.5)               | A B C    | reconnect attempts                                                                         |
| `protocol_version` (`2025-11-25`)                    | all      | requested version; must be in `MCP_PROTOCOL_VERSIONS`                                      |
| `url`, `headers`, `oauth`, `verify_tls`              | —        | only meaningful for HTTP transports, which this build refuses (`oauth` additionally warns) |
| `allow_sampling`, `allow_elicitation`, `allow_roots` | all      | accepted, not implemented; enabling them is a validation warning                           |

Accepted versions: `2026-07-28`, `2025-11-25` (preferred), `2025-06-18`, `2025-03-26`,
`2024-11-05`. A server that negotiates down is fine — measured: WordCraft answers
`2025-06-18` and the client accepts it.

### 4.3 Skeleton

```json
{
  "version": 1,
  "servers": {
    "<name>": {
      "transport": "stdio",
      "command": ["/abs/path/to/mcp-server", "--flag"],
      "env": { "API_KEY": "${SOME_ENV_VAR}" },
      "startup": "lazy",
      "tool_prefix": "<short>",
      "confirm_tools": ["…"],
      "call_timeout_seconds": 120.0
    }
  }
}
```

`env` values are expanded **at load time**, so the secret lives in the environment, not
in the file. Note the file is parsed once per session; rotating a variable requires a new
session (or `/mcp disconnect <name>` + `/mcp connect <name>`, which re-spawns the child).

---

## 5. What this build supports — and what it refuses

| Capability                         | Status              | Consequence                                                 |
| ---------------------------------- | ------------------- | ----------------------------------------------------------- |
| stdio transport                    | implemented         | topologies A, B, C                                          |
| `tools/list`, `tools/call`         | implemented         | the whole tool surface                                      |
| `initialize` + version negotiation | implemented         | 5 accepted versions                                         |
| Streamable HTTP / SSE              | **refused**         | remote servers need a shim (§5.1)                           |
| OAuth / bearer headers             | **refused**         | static secrets only, via `env`                              |
| resources, prompts, completion     | **not implemented** | WordCraft's two `wordcraft://` resources are simply ignored |
| sampling, elicitation, roots       | **refused**         | server-initiated features never fire                        |
| `list_changed` refresh, `reload`   | **not implemented** | restart the session to re-read a catalog                    |

Fail-closed is deliberate: refused means "the server is listed but never connected", and
it is visible in `/mcp status` rather than silently degrading.

### 5.1 Four legitimate ways to reach an HTTP-only server

| Approach                | Command                                                 | Trade-off                                           |
| ----------------------- | ------------------------------------------------------- | --------------------------------------------------- |
| ssh stdio (B)           | `["ssh","host","mcp-server"]`                           | needs key-based ssh; failure kills the tool surface |
| generic stdio↔HTTP shim | `["npx","-y","mcp-remote","https://host/mcp"]`          | extra runtime (node); one more link to debug        |
| vendor bridge (C)       | `["wordcraft-cli","mcp","--connect","127.0.0.1:10080"]` | best when it exists: no new dependency              |
| implement the transport | code in `ollamaquery2.py`                               | the real fix; see §9                                |

Do not "solve" this by piping HTTP through `socat` into stdio — the framing differs
(length-prefixed/SSE vs newline JSON) and it will die on the first large payload.

---

## 6. The policy ladder

Three layers, each of which can stop a call independently.

| Layer           | Mechanism                                                     | Who decides          | Defeat-able by                                 |
| --------------- | ------------------------------------------------------------- | -------------------- | ---------------------------------------------- |
| 1. Exposure     | `enabled`, `allowed_tools`, `denied_tools`, `read_only_only`  | config (the human)   | nothing — filtered tools never reach the model |
| 2. Confirmation | `confirm_tools` + every tool that is not explicitly read-only | the prompt (`[y/N]`) | `auto_confirm`, except for `ask`-level rules   |
| 3. Plan tier    | `/agentic plan` exposes only `plan_eligible` tools            | annotation + config  | nothing; no shell, CWD-confined                |

`plan_eligible` is exactly:

```
filtered_reason is None  and  readOnlyHint is True  and  destructiveHint is not True
```

Everything else — including a tool that declares **nothing** — is treated as
destructive. This matches the spec defaults (`readOnlyHint: false`, `destructiveHint:
true`), and it is why an unannotated server is safe but annoying.

Listing markers in `/mcp tools`:

| Marker | Meaning                             |
| ------ | ----------------------------------- |
| `R`    | declared read-only, not destructive |
| `+`    | declared `destructiveHint: false`   |
| `!`    | declared destructive                |
| `?`    | undeclared — treated as destructive |
| `x`    | filtered by config (reason shown)   |

### 6.1 Confinement is the server's job, not ours

Our filters choose *which tools* the model can call. They say nothing about *which
files* those tools touch. That must come from the server:

- Headless WordCraft: `--automation-read-root` / `--automation-write-root` — enforced by
  the app, for every command, on every thread.
- Live WordCraft app: **no confinement exists**; `--connect` even rejects the flags.
  The confirmation gate is the only leash.

So the honest rule: **an application that offers no confinement and declares no
annotations should be given `read_only_only: true` or not integrated at all.**

### 6.2 Non-negotiable checklist

1. Secret via `${VAR}`, never a literal in the JSON.
2. `denied_tools` for anything you would not want the model to run unattended.
3. `confirm_tools` for the one or two tools that can destroy work — usually the
   generic escape hatches (`execute`, `batch`, `run`).
4. Prefer a server with file roots over one without.
5. Re-read `/mcp tools --plan` after every application upgrade: catalogs change.

---

## 7. Runbook: adding an application

```sh
# 1. find the server command (docs, `--help`, `which <app>-mcp`)
<app>-mcp --help

# 2. prove it speaks MCP before touching the config
printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"probe","version":"0"}}}' \
  | <app>-mcp | head -c 400

# 3. add the JSON entry, then let the client classify it
python3 ollamaquery2.py --mcp-config ./app.mcp.json -c
/mcp status                 # connected? version? errors?
/mcp tools <app> --all      # every tool, every annotation, what got filtered
/mcp tools <app> --plan     # what read-only mode would see

# 4. exercise one harmless tool through the agentic gate
/agentic plan on
"<prompt that only needs a read-only tool>"
```

For step 3 without a model backend, use the harness in Appendix A: it drives the real
client code (`load_mcp_config` → `MCPManager` → `call`) with no chat loop and no LLM.

---

## 8. Case study: WordCraft 0.9.0

### 8.1 What the application exposes

| Port/command                                  | Speaks                                                               | Who connects        |
| --------------------------------------------- | -------------------------------------------------------------------- | ------------------- |
| `wordcraft --control 10080`                   | control protocol: raw TCP, newline-delimited JSON, `key` per request | the bridge only     |
| `wordcraft-cli mcp`                           | MCP over stdio, headless                                             | client (topology A) |
| `wordcraft-cli mcp --connect 127.0.0.1:10080` | MCP over stdio **bridged** to the app                                | client (topology C) |

The control port is **not** MCP and never was: no `initialize`, no `tools/list`, envelopes
`{"id":7,"ok":false,"error":"…"}`, and any unrecognised `method` is re-interpreted as a
command id. Probed directly: `initialize`, `ping`, `tools/list` all return
``unknown command ``. That is why the bridge exists, and why an HTTP client pointed at
10080 gets `HTTP/0.9` garbage — the server answers bad JSON with one error line and
closes the connection, by design, so that a web page cannot drive the editor.

### 8.2 Measured topology

```
ollamaquery2 ──pipe── [fd0/fd1] wordcraft-cli mcp ──TCP──▶ 127.0.0.1:10080
               parent            child          client      app listener
```

| Observation                                                     | Value                                                                         |
| --------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| child descriptors 0/1/2                                         | `pipe:[…]` — no socket on the client hop at all                               |
| child descriptors 3 and 4                                       | the same socket inode (one dup'd connection)                                  |
| TCP flow                                                        | `127.0.0.1:60066 → 127.0.0.1:10080`, ESTABLISHED **before** the first request |
| socket reused across `initialize` → `tools/list` → `tools/call` | yes (inode unchanged)                                                         |
| negotiated MCP version                                          | `2025-06-18` (requested `2025-11-25`), accepted                               |
| tools                                                           | 16 — 7 read-only, 9 mutating                                                  |
| `destructiveHint`                                               | absent on all 16 → the 9 mutating are gated by default                        |
| `openWorldHint`                                                 | `false` on all 16                                                             |

### 8.3 Two servers, on purpose

|                      | `wordcraft` (headless)         | `wordcraft-app` (bridge)                                                    |
| -------------------- | ------------------------------ | --------------------------------------------------------------------------- |
| Confinement          | `--automation-read/write-root` | **none available**                                                          |
| Credentials          | none                           | `WORDCRAFT_CONTROL_KEY` (new value per app launch)                          |
| Survives app restart | n/a                            | no                                                                          |
| Extra tools          | —                              | `screenshot`, `click`, `key`, `ui_inspect`                                  |
| Our gate             | none (the roots are the leash) | `confirm_tools: execute, batch, save_document, open_document, new_document` |

### 8.4 Worked result

13 MCP calls, all `ok`, verified by reading the document back rather than by trusting the
write:

| Artifact        | Result                                                                  |
| --------------- | ----------------------------------------------------------------------- |
| `report.docx`   | 7 069 bytes, 1 page, 88 words, 15 paragraphs                            |
| `report.pdf`    | 28 KB, real selectable text                                             |
| `report-p1.png` | 918 × 1188 at 1.5 px/pt                                                 |
| read-back       | `Title`, `Subtitle`, bold run at offsets 115–145, italic run, 3×3 table |

### 8.5 The cost nobody mentions

Each MCP server's schemas are copied into the outbound request (`tools` array or the
inline prompt block). Two WordCraft servers = 32 schemas ≈ 5–6 K tokens **before the
first user turn**, and `/tokencount` charges it. With a dozen servers configured, tool
definitions become the largest thing in the prompt. This is the strongest argument for
`enabled: false` on servers you are not using this session — see §9.

### 8.6 Getting the MCP server out of a GUI-only bundle

The application's own `--help` is the wrong place to look, and this is the single most
common reason people conclude "this app has no MCP server":

```
$ ./wordcraft-0.9.0-linux-x86_64.AppImage --help
wordcraft — the WordCraft desktop app
USAGE:
  wordcraft [--sample] [--control PORT] [FILE…]
```

That output is the **`AppRun` symlink**, which resolves to `usr/bin/wordcraft` — the GUI
entry point only. The packaging stages both binaries into one FHS tree and copies the
whole tree into the AppDir:

```sh
install -Dm755 "$BIN/wordcraft"     "$STAGE/usr/bin/wordcraft"
install -Dm755 "$BIN/wordcraft-cli" "$STAGE/usr/bin/wordcraft-cli"
cp -R "$STAGE" "$APPDIR"                       # the AppImage is this tree, squashed
ln -s usr/bin/wordcraft "$APPDIR/AppRun"       # …but only one door
```

So the server is present and unreachable. An AppImage is an ELF runtime stub with a
squashfs filesystem appended; both extractors work without FUSE and without a display.

**Recipe A — the runtime's own extractor (preferred).** It accepts a glob, so only the
binary is written:

```sh
chmod +x wordcraft-0.9.0-linux-x86_64.AppImage
./wordcraft-0.9.0-linux-x86_64.AppImage --appimage-extract 'usr/bin/wordcraft-cli'
install -Dm755 squashfs-root/usr/bin/wordcraft-cli ~/.local/bin/wordcraft-cli
~/.local/bin/wordcraft-cli --version          # wordcraft-cli 0.9.0
```

Measured: one file written (`squashfs-root/usr/bin/wordcraft-cli`, 76 375 984 bytes), mode
`0755` preserved by squashfs, `--version` and `--help` both work headless.

**Recipe B — `unsquashfs` with the embedded offset.** Use this when the AppImage cannot be
executed at all (no FUSE, foreign arch, `noexec` mount), because it never runs the stub:

```sh
OFF=$(./wordcraft-0.9.0-linux-x86_64.AppImage --appimage-offset)   # 944632 for 0.9.0 x86_64
unsquashfs -o "$OFF" -d out wordcraft-0.9.0-linux-x86_64.AppImage 'usr/bin/wordcraft-cli'
# out/usr/bin/wordcraft-cli
```

Measured: identical file, identical mode, same `--version`.

Practical notes:

- **The offset is per build, never hardcode it.** `--appimage-offset` needs the stub to
  run. Without that, locate the squashfs magic (`hsqs`) and verify: on the 0.9.0 x86_64
  image `grep -abo hsqs` returns **two** hits — `194183`, a false positive inside the ELF
  stub, and `944632`, the real superblock — so take the last hit and let `unsquashfs`
  confirm it.
- **glibc**: the AppImage is built against glibc 2.35 (Ubuntu 22.04) so it runs on that or
  newer; check with `ldd --version`. Newer is fine, older is not.
- **The extracted CLI needs no display**, which is what makes it usable as a stdio MCP
  server on a headless build host.
- **Sizes are normal**: 76 MB is the whole document engine statically linked, not bloat to
  trim.

The same "hidden second binary" situation recurs across packaging formats:

| Artifact            | How to reach the server binary                                         |
| ------------------- | ---------------------------------------------------------------------- |
| AppImage            | `--appimage-extract '<path>'`, or `unsquashfs -o $(--appimage-offset)` |
| `.deb`              | `dpkg -x pkg.deb out/` (no install, no root)                           |
| `.rpm`              | `rpm2cpio pkg.rpm \| cpio -idmv`                                       |
| Flatpak             | `flatpak run --command=<cli> <app-id> <args>`                          |
| macOS `.dmg`/`.app` | `Contents/MacOS/<binary>` inside the bundle                            |
| tarball             | usually already an FHS tree: `usr/bin/<cli>`                           |

---

## 9. Open questions for the next discussion

1. **HTTP transport.** The single biggest gap. Most real-world MCP servers are remote;
   today they all need a shim. Implementing Streamable HTTP (POST + optional SSE,
   `Mcp-Session-Id`, `MCP-Protocol-Version`) converts topologies B/C into "just a URL".
2. **Tool-schema budget.** 32 schemas is fine; 200 is not. Options: per-server lazy
   exposure driven by a task hint, a two-pass catalog (`meta` tools that expose the rest
   on demand), or a hard per-server token ceiling in the config.
3. **Discovery.** Nothing tells you an app speaks MCP. A curated table in this repo
   (app → topology → command → annotation quality) would be more useful than any clever
   auto-detection, because the annotation quality is what decides usability.
4. **Ergonomics.** Today: hand-edit JSON, restart, `/mcp status`. Wanted:
   `/mcp add <name> <command…>` for the session, and `--mcp-server` for a one-off. The
   config file should stay the durable artifact, but a session-scoped add would make
   trial-and-error much cheaper.
5. **Credential rotation.** `env` is expanded once at load. A rotating key (WordCraft's
   is per launch) forces a session restart, or a disconnect/connect pair. A
   `key_command` (run a command to fetch the secret at spawn) would fix this generically.
6. **Resources and prompts.** Cheap to add, and WordCraft already advertises
   `wordcraft://document` and `wordcraft://commands`. Whether the agent *needs* them
   when `get_text`/`inspect_document` exist is unproven — that is the experiment to run
   before implementing.
7. **Confinement contract.** We should state, per server, whether confining paths is the
   server's responsibility (`--automation-*-root`) or absent. An app without roots is
   not unusable, but it must be marked, so nobody assumes protection that is not there.

---

## Appendix A — verification harness (no TUI, no model)

`ollamaquery2.py` is importable, so a config can be validated against the real client
without launching a chat. This is the cheapest way to answer "did I write the JSON
right?":

```python
import os, sys
os.environ.setdefault("SOME_KEY", "…")
sys.path.insert(0, "/path/to/ollamaquery2")
import ollamaquery2 as oq

cfg = oq.load_mcp_config()                      # honours --mcp-config precedence
for name, server in cfg.servers.items():
    print(name, oq.validate_mcp_server(server))  # [] means valid

mgr = oq.MCPManager(cfg)
mgr.ensure_started_all()                        # never raises; fills mgr.errors
print(mgr.errors)                               # {} means every server connected
print(len(mgr.aggregate_tools()))               # model-visible tool count
print(sorted(mgr.plan_read_only_names()))       # what /agentic plan would expose
print(sorted(mgr.confirm_names()))              # what is force-gated
print(mgr.call("<prefix>__<tool>", {}))         # a real round trip
mgr.shutdown()
```

Measured output for the WordCraft pair:

```
config source : /home/ollama/.ollamaquery.d/mcp.json
  wordcraft      problems: none
  wordcraft-app  problems: none
exposed tools : 32 (16 + 16)
plan-eligible : 14 (7 per server)
confirm-gated : wca__batch, wca__execute, wca__new_document,
                wca__open_document, wca__save_document
```

## Appendix B — the reference config used above

```json
{
  "version": 1,
  "servers": {
    "wordcraft": {
      "transport": "stdio",
      "command": ["/home/ollama/.local/bin/wordcraft-cli", "mcp",
                  "--automation-read-root", "/home/ollama/wordcraft",
                  "--automation-write-root", "/home/ollama/wordcraft"],
      "enabled": true,
      "startup": "lazy",
      "tool_prefix": "wc",
      "call_timeout_seconds": 180.0
    },
    "wordcraft-app": {
      "transport": "stdio",
      "command": ["/home/ollama/.local/bin/wordcraft-cli", "mcp",
                  "--connect", "127.0.0.1:10080"],
      "env": { "WORDCRAFT_CONTROL_KEY": "${WORDCRAFT_CONTROL_KEY}" },
      "enabled": true,
      "startup": "lazy",
      "tool_prefix": "wca",
      "confirm_tools": ["execute", "batch", "save_document",
                        "open_document", "new_document"],
      "call_timeout_seconds": 180.0
    }
  }
}
```

Set `"enabled": false` on `wordcraft-app` for sessions that do not need the live window;
that halves the tool-schema cost (§8.5).
