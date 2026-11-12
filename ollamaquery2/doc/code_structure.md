# Code Structure — ollamaquery2.py

Snapshot: **2026-11-12**, `__version__ = "0.4.1"`, **20 546 lines**.
Regenerated from the section markers (`# ============= …`) and an AST pass, not
patched by hand — see §8. Numbers drift; regenerate rather than trust a diff.

> Historical note: the previous revision of this file described a ~4 300-line
> version and placed `ChatLoop` at lines 2109–2737. That document was stale by
> an order of magnitude; this one replaces it wholesale.

## 1. Overview

`ollamaquery2.py` is a single-file CLI for local and cloud LLM backends. The
header states the constraint that shapes everything else: *single file,
maximally portable* (`cp`/`scp` and run, zero install, no packaging). The class
count stays manageable by **extraction into module-level collaborators inside
the same file**, each with a thin "door" on its old owner so call sites and
tests never change.

Supported backends: Ollama, llama.cpp, Strata, LM Studio, Gemini, OpenCode Zen,
OpenCode Go, Mistral, DeepSeek, plus MCP servers as tool providers (stdio).

Modes: chat (`-c`), batch (`--input-dir`), single query (`-I`), listing (`-l`),
model info (`--show`).

## 2. Section map

The file is partitioned by three-line banner comments. Ranges below are
`start–end` of each banner block's content (end = next banner − 4).

| Lines       | Section                           | What lives there                                                                                                                                                                                                                           |
| ----------- | --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 13–69       | Imports & configuration           | Standard library only; no required third-party deps                                                                                                                                                                                        |
| 73–152      | Message schema (type annotations) | `ToolCallFunction`, `ToolCall`, `_MessageBase`, the four role messages, `Message` union. **Annotations only** — the runtime shape is plain dicts with private keys (`_tokens`, `_system_nudge`, …)                                         |
| 156–314     | Default configuration             | `__version__`, `DEFAULT_*` constants, `MAX_CONTEXT_SIZE`, backend defaults, inference/sampling registries                                                                                                                                  |
| 318–363     | Theme & color configuration       | 5 built-in themes, custom theme loading                                                                                                                                                                                                    |
| 367–764     | Command registry                  | `COMMANDS`, categories, aliases, `get_commands_by_category()`                                                                                                                                                                              |
| 768–844     | Retry utility                     | `_request_with_retry` (wraps `urlopen`; covers handshake, not body reading — by design)                                                                                                                                                    |
| 848–981     | Context window tracking           | `refresh_context_window_size`, per-backend window discovery, `-1` = unknown                                                                                                                                                                |
| 985–1891    | Utility functions                 | HTML stripping (`CoreHTMLStripper`), image prep, colorize, `_bounded_read`, `parse_size`, model listing helpers                                                                                                                            |
| 1895–2603   | Context compaction                | Thresholds, `sanitize_tool_pairing`, `compact_messages`, LLM summarization, memory budget/shrink                                                                                                                                           |
| 2607–2654   | Private file modes                | `_PRIVATE_DIR_MODE` 0700 / `_PRIVATE_FILE_MODE` 0600, opening helpers, best-effort chmod of old transcripts                                                                                                                                |
| 2658–2767   | Session persistence               | `SessionManager`: save/list/load under `~/.ollamaquery.d/sessions/`                                                                                                                                                                        |
| 2771–3360   | Composable agentic system prompt  | Role/format/example/rules blocks, style registry, `get_agentic_prompt`, persona composition                                                                                                                                                |
| 3364–4364   | Shell approval gate               | Tokenizer, breakdown, home/root hard block, CWD leniency, path-ACL operand scan, `check_shell_approval`                                                                                                                                    |
| 4368–4679   | Executor (container/host)         | `Executor`: host (`shell=True` behind the gate) or podman/docker image/`exec`, resource ceiling, mount→workdir mapping                                                                                                                     |
| 4683–5193   | Agentic tool definitions          | `DESTRUCTIVE_TOOLS`, `PLAN_MODE_TOOLS`, `PLAN_MODE_RUN_TOOLS`, `plan_active_tools`, the JSON schemas of the 14 tools                                                                                                                       |
| 5197–5352   | Session scratch dir               | `/tmp/ollamaquery/pid-<n>` creation, ownership/mode checks, dead-PID sweep, leaf-only cleanup                                                                                                                                              |
| 5356–5512   | Agentic tool handlers             | Shared helpers used by the handlers (binary sniffing, grep helpers, output caps)                                                                                                                                                           |
| 5516–8362   | Path ACL                          | `PathAcl` (rules, `evaluate`, `prompt`, `describe`), `_path_acl_decision`, `_is_blocked_system_file`, `_shell_acl_scan`                                                                                                                    |
| 8366–8505   | "Tool registry" marker            | **Misplaced banner** — only `_is_harmless_get_fetch` lives here; `ToolRegistry` is at 10 263 (see next row)                                                                                                                                |
| 8509–10589  | MCP client + `ToolRegistry`       | `MCPServerConfig`/`MCPConfig`, config discovery/validation, JSON-RPC framing, `StdioMCPTransport`, `MCPConnection`, `MCPClient`, `MCPManager`; then `ToolRegistry` (10 263) — the registry merges MCP tools, so it must follow the manager |
| 10593–10691 | Agentic session logger            | `AgenticLogger`: JSONL turns, retention cleanup, best-effort by contract                                                                                                                                                                   |
| 10695–10819 | Debugging                         | `DebugManager` (per-category levels) + `debug_log`                                                                                                                                                                                         |
| 10823–11165 | Request builder                   | `RequestBuilder`: payload shaping, output cap, images, per-backend inference/reasoning params, URL, stream tuple                                                                                                                           |
| 11169–11413 | Stats calculator                  | `StatsCalculator`: prefill/think/answer split, hybrid server+client timing, the stats line                                                                                                                                                 |
| 11417–11717 | Stream parser                     | `StreamParser`: per-backend chunk shapes, SSE vs NDJSON, tool-call accumulation, sync aggregation                                                                                                                                          |
| 11721–12420 | Model query                       | `ModelQuery`: transport, retries, display, the thin doors onto the three collaborators above                                                                                                                                               |
| 12424–12712 | Command handling                  | `ChatCompleter` (tab completion)                                                                                                                                                                                                           |
| 12716–13626 | Input handling                    | `gather_user_input`, inline commands (`!`, `/curl`, `@file`), `_ExitRequested`, readline/history                                                                                                                                           |
| 13630–14026 | Tool-call parsing                 | Balanced-JSON primitives + `ToolCallParser` (strict vs lazy policy, 4-pass extraction)                                                                                                                                                     |
| 14030–14336 | Agentic timeout policy            | `TimeoutEvent` (frozen), `AgenticTimeoutPolicy` (states C / A / B, nudge, ceiling doubling)                                                                                                                                                |
| 14340–14549 | Agentic step feedback             | `StepFeedback`: live thinking, heartbeat dots, idle watchdog, accumulated content for timeout analysis                                                                                                                                     |
| 14553–14825 | Agentic tool execution            | `ToolExecution`: run, cap observation, same-tool guard, message append (native vs inline)                                                                                                                                                  |
| 14829–15200 | Agentic finalizer                 | `AgenticFinalizer`: stream the final answer **or** run the tool call it carried and re-enter (bounded)                                                                                                                                     |
| 15204–15999 | Agentic runner                    | `ReActState` + `AgenticRunner`: `run` (skeleton), `_start_query`, `_run_step`, `call_step`, presentation, partial-write recovery                                                                                                           |
| 16003–16490 | Outbound wire                     | `WireView`: `system_and_tools`, context bar, overhead, `/promptshow`, nudges, force-compact                                                                                                                                                |
| 16494–16936 | Conversation store                | `ConversationStore`: message list, session persistence, compaction orchestration                                                                                                                                                           |
| 16940–19242 | Chat loop                         | `ChatLoop`: dispatcher + 27 `run_handle_*`, query path, agentic wiring, doors                                                                                                                                                              |
| 19246–19908 | Main entry point                  | Backend probes/checks (`check_*`), `resolve_connection`, `auto_detect_backend`, `main`, listing                                                                                                                                            |
| 19912–20546 | Argument parser                   | `argparse` surface (backend, sandbox, MCP, ACL, theme, batch)                                                                                                                                                                              |

## 3. Classes and their spans

Class-body AST spans (`end_lineno − lineno + 1`); `defs` counts class-level
functions including property accessors.

| Class                           | Lines       | Size | Defs | Role                                                                                    |
| ------------------------------- | ----------- | ---- | ---- | --------------------------------------------------------------------------------------- |
| `CommandContext`                | 1402–1892   | 491  | 24   | State singleton: connection, session, token counting, stats, agentic flags, `path_acl`  |
| `SessionManager`                | 2667–2768   | 102  | 7    | Save/list/load sessions                                                                 |
| `Executor`                      | 4372–4680   | 309  | 13   | Host/container command runner (gate-aware; `exec` reuses a running container)           |
| `PathAcl`                       | 5551–5851   | 301  | 10   | Session-scoped allow/ask/deny policy (read/write, longest-prefix, ask is bypass-immune) |
| `MCPServerConfig` / `MCPConfig` | 8722–8787   | 64   | 4    | Parsed MCP configuration                                                                |
| `MCPToolEntry`                  | 9235–9287   | 53   | 4    | One catalog entry + plan-eligibility predicates                                         |
| `StdioMCPTransport`             | 9512–9655   | 144  | 10   | argv, no shell, own process group, stderr tail, framed writes                           |
| `InMemoryMCPTransport`          | 9658–9708   | 51   | 7    | Test transport                                                                          |
| `MCPConnection`                 | 9713–9863   | 151  | 7    | Reader thread, id→Event multiplexer, cancellation on timeout                            |
| `MCPClient`                     | 9881–10023  | 143  | 8    | initialize → initialized → paginated `tools/list` → `tools/call`                        |
| `MCPManager`                    | 10028–10261 | 234  | 19   | Lazy/eager startup, catalog aggregation, routing, policy sets                           |
| `ToolRegistry`                  | 10263–10590 | 328  | 11   | Dispatch, aliases, confirmation, plan surface, prompt block                             |
| `AgenticLogger`                 | 10597–10692 | 96   | 5    | JSONL session log, retention, fail-soft                                                 |
| `DebugManager`                  | 10699–10783 | 85   | 6    | Per-category debug levels                                                               |
| `RequestBuilder`                | 10827–11167 | 341  | 13   | Outbound payload shaping (backend is an explicit argument)                              |
| `StatsCalculator`               | 11173–11415 | 243  | 6    | Timings, token split, stats line                                                        |
| `StreamParser`                  | 11421–11719 | 299  | 7    | Chunk shapes + sync aggregation (late-bound line provider)                              |
| `ModelQuery`                    | 11725–12421 | 697  | 37   | Transport, retries, display, doors onto the three collaborators                         |
| `ChatCompleter`                 | 12470–12713 | 244  | 10   | Tab completion (commands, paths, models, subcommands, MCP servers)                      |
| `CoreHTMLStripper`              | 13548–13624 | 77   | 6    | stdlib HTMLParser subclass for `/curl` and `fetch_url`                                  |
| `ToolCallParser`                | 13801–14028 | 228  | 9    | Strict/lazy tool-call extraction                                                        |
| `TimeoutEvent`                  | 14055–14071 | 17   | 0    | Frozen facts of one timed-out step                                                      |
| `AgenticTimeoutPolicy`          | 14074–14338 | 265  | 8    | 3-state escalation + nudge + ceiling                                                    |
| `StepFeedback`                  | 14357–14551 | 195  | 8    | Per-step streaming feedback                                                             |
| `ToolExecution`                 | 14566–14827 | 262  | 8    | Tool run, observation cap, same-tool guard, message append                              |
| `AgenticFinalizer`              | 14833–15202 | 370  | 8    | Post-loop phase + bounded streaming re-entry                                            |
| `ReActState`                    | 15219–15258 | 40   | 1    | One query's mutable state (replaces nine threaded locals)                               |
| `AgenticRunner`                 | 15261–16001 | 741  | 22   | The step loop                                                                           |
| `WireView`                      | 16007–16491 | 485  | 19   | Outbound request accounting + `/promptshow`                                             |
| `ConversationStore`             | 16498–16937 | 440  | 20   | Messages, session, compaction                                                           |
| `ChatLoop`                      | 16944–19243 | 2300 | 113  | Command dispatch, query paths, doors onto every collaborator                            |

## 4. Architecture

```
main()                                    19246+
 ├── resolve_connection()                  CLI > saved MRU > auto-detect > default
 ├── list_models_*()                       no context needed
 └── ChatLoop.run()                       16940
      ├── CommandContext (singleton)      1402   all shared state
      ├── ChatCompleter                   12470  readline completion
      ├── ModelQuery                      11725
      │    ├── RequestBuilder             10827  payload/URL shaping
      │    ├── StreamParser               11421  chunk shapes, SSE vs NDJSON
      │    └── StatsCalculator            11173  timings + stats line
      ├── WireView                        16007  what the next request costs
      ├── ConversationStore               16498  messages + sessions + compact
      └── agentic chain
           ├── ToolCallParser             13801
           ├── AgenticRunner              15261  step loop
           │    ├── StepFeedback          14357  streaming feedback
           │    ├── ToolExecution         14566  run + observe
           │    └── AgenticTimeoutPolicy  14074  escalation
           ├── AgenticFinalizer           14833  final answer / re-entry
           └── ToolRegistry               10263  dispatch
                ├── Executor              4372   host / container
                ├── PathAcl               5551   read/write policy
                └── MCPManager            10028  external tools
```

Every extracted collaborator keeps **thin doors** on its old owner (e.g.
`ChatLoop._execute_tool_calls` → `ToolExecution.run_calls`,
`ModelQuery.build_request_payload` → `RequestBuilder.build_request_payload`).
That is what lets the 1 181 tests stub the old surface and still intercept.

## 5. Data flow

**Chat query**

```
gather_user_input()
 → process_inline_commands()          @file, !cmd, /curl
 → ChatLoop.run_process_query()
     → ConversationStore.maybe_auto_compact_chat()
     → WireView.system_and_tools()    mode-aware prompt + native tools
     → WireView.force_compact_if_no_output_room()
     → ModelQuery.query_stream()
         → RequestBuilder.build_request_payload()   backend-specific payload
         → _iter_stream_lines()                     SSE `data:` vs raw NDJSON
         → StreamParser._parse_chunk()               thought/content/tool_calls/usage
         → stdout (live) + stats line
         → WireView.refresh_context_bar()            outbound request, exact
     → ConversationStore.append(assistant)
```

**Agentic query**

```
ChatLoop.run_agentic_query()
 → AgenticRunner.run(host, input)
     → _start_query()                inline cmds, seed messages, ReActState
     → while: _run_step()
         → step_header()             auto-compact + near-full nudge
         → call_step()               query_sync_stream under call_with_timeout
         → timeout?  AgenticTimeoutPolicy.on_timeout()  (C / A / B)
         → ToolExecution.parse_calls() / run_calls() / append_messages()
         → stuck detection, iteration budget
     → AgenticFinalizer.finalize_agentic_query()   stream answer or re-enter
     → ConversationStore.persist_agentic_history()
```

## 6. Backend integration guide

Adding a backend currently touches four seams (all reached through doors on
`ModelQuery`, so tests can stub them):

1. **Request shape** — `RequestBuilder.build_request_payload()` /
   `_get_chat_url()`: URL, payload fields, `tools` array, size/output-cap
   options, inference and reasoning parameters.
2. **Line framing** — `StreamParser._iter_stream_lines()`: SSE (`data:` prefix,
   `[DONE]`) vs raw NDJSON.
3. **Chunk shape** — `StreamParser._parse_chunk()`: where
   `thought` / `content` / `is_final` / `usage` come from, and how tool calls
   accumulate (Ollama sends whole objects, OpenAI-family sends deltas by index).
4. **Token counting / window** — `CommandContext.get_message_token_count_*`,
   `count_wire_tokens`, `refresh_context_window_size`, and the
   `_extract_usage_timings` fields for the stats split.

`_aggregate_sync_stream()` reads lines through a **late-bound provider**, so a
test that replaces `ModelQuery._iter_stream_lines` still intercepts the parser —
do not bind the method at construction.

## 7. Key design decisions (short form)

- **Single file** — deployability over module hygiene; extractions happen inside
  the file with doors.
- **State singleton** (`CommandContext`) — one source of truth, no threading
  in a single-user CLI.
- **One definition of "context used"** — `CommandContext.outbound_tokens()`
  (wire system prompt + conversation + tools + framing); the bar, `/stats` and
  `/tokencount` all read it.
- **Safety as steering, not sandboxing** — the shell gate and Path ACL are
  consent + audit layers; `/agentic sandbox` is the real isolation boundary.
- **Ask is bypass-immune** — `auto_confirm` never approves an `ask` rule.
- **Plan mode is layered** — prompt block + native `tools` schema + execution
  backstop all read `plan_active_tools()`, so they cannot disagree.
- **Best-effort side channels** — logging, MCP spill notes, and debug output
  fail soft; they must never abort a working query.

Long form, with the reasoning and the bugs that motivated each rule:
`AGENTS.md` (local-only).

## 8. Regeneration script

```bash
python3 - <<'PY'
import ast, pathlib
src = pathlib.Path('ollamaquery2.py').read_text()
print('LINES', len(src.splitlines()))
for n in ast.parse(src).body:
    if isinstance(n, ast.ClassDef):
        defs = [m for m in n.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
        print(f"{n.lineno:6d}-{n.end_lineno:6d} ({n.end_lineno-n.lineno+1:5d} L, "
              f"{len(defs):2d} defs) {n.name}")
PY
```

Section map: every `# ============` triple; content runs to the next banner − 4.
