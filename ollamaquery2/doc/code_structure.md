# Code Structure — ollamaquery2.py

Snapshot: **2026-11-23**, `__version__ = "0.5.1"`, **27 053 lines**.
Regenerated from the section markers (`# ============= …`) and an AST pass, not
patched by hand — see §8. Numbers drift; regenerate rather than trust a diff.

> The line ranges in §2 and §4 still predate the `0.4.3` shell-gate pass
> (C1/C2), which added ~340 lines in the shell section: everything after the
> shell gate is shifted, and the header above is the only number kept current.
> Re-run §8 (and the section-marker pass) to refresh them. The D1 pass
> (2026-11-13) added ~50 lines before `ModelQuery` (`stream_is_incomplete` +
> `stream_incomplete_reason`), so ranges after the stream section shift again.
> The `0.4.4` pass (2026-11-19) adds 86 lines over the `0.4.3` snapshot
> (`_anthropic_messages` on `CommandContext`, token stamping in
> `ToolExecution`, the `/agentic status` column width), so every range after
> those points shifts again. The MaaS pass (same day, unreleased) adds ~590
> lines in a new section right after the retry utility (the gateway core) plus
> the per-backend wiring in the request/listing paths; every range below the
> utility section is shifted once more. The `0.4.5` pass (2026-11-19) is net
> **+11 lines** over `0.4.4` (a sampling entry withdrawn, the reasoning-only
> nudge, two parser fixes and the SSE keep-alive filter added). The `0.5.0`
> pass (2026-11-22) adds ~700 lines: the bwrap engine (`Executor`, the
> sandbox-home lifecycle, the probe), the preset resolver
> (`AgenticPreset`/`resolve_preset`/`agentic_active_tools`), the `/agentic
> preset` and `sandbox` surfaces, and the CLI/env wiring. The `0.5.1` pass
> (2026-11-23) is the post-0.5.0 security batch — H0/H1/H3/H6 in the shell
> gate (`_shell_access_plan`, effective cwd, TMPDIR, nested payloads, host
> code launchers), `SessionAccess`/`/agentic files` (A1), A2 retention and
> `--api-key-file` — so every range below the shell gate shifted once more.
> **§3 was regenerated from the AST for 0.5.1**; §2 and §4 still carry the
> older ranges and need the section-marker pass (the names and descriptions
> remain accurate).

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
OpenCode Go, Mistral, DeepSeek, MaaS (OpenShift AI / KServe in front of vLLM),
plus MCP servers as tool providers (stdio).

Modes: chat (`-c`), batch (`--input-dir`), single query (`-I`), listing (`-l`),
model info (`--show`).

## 2. Section map

The file is partitioned by three-line banner comments. Ranges below are
`start–end` of each banner block's content (end = next banner − 4).

| Lines       | Section                           | What lives there                                                                                                                                                                                                                          |
| ----------- | --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 13–69       | Imports & configuration           | Standard library only; no required third-party deps                                                                                                                                                                                       |
| 73–152      | Message schema (type annotations) | `ToolCallFunction`, `ToolCall`, `_MessageBase`, the four role messages, `Message` union. **Annotations only** — the runtime shape is plain dicts with private keys (`_tokens`, `_system_nudge`, …)                                        |
| 156–314     | Default configuration             | `__version__`, `DEFAULT_*` constants, `MAX_CONTEXT_SIZE`, backend defaults, inference/sampling registries                                                                                                                                 |
| 318–363     | Theme & color configuration       | 5 built-in themes, custom theme loading                                                                                                                                                                                                   |
| 367–826     | Command registry                  | `COMMANDS`, categories, aliases, `get_commands_by_category()`, unknown-command detection/escape (`command_word`, `unescape_command_line`)                                                                                                 |
| 830–906     | Retry utility                     | `_request_with_retry` (wraps `urlopen`; covers handshake, not body reading — by design)                                                                                                                                                   |
| 910–1043    | Context window tracking           | `refresh_context_window_size`, per-backend window discovery, `-1` = unknown                                                                                                                                                               |
| 1047–1953   | Utility functions                 | HTML stripping (`CoreHTMLStripper`), image prep, colorize, `_bounded_read`, `parse_size`, model listing helpers                                                                                                                           |
| 1957–2665   | Context compaction                | Thresholds, `sanitize_tool_pairing`, `compact_messages`, LLM summarization, memory budget/shrink                                                                                                                                          |
| 2669–2716   | Private file modes                | `_PRIVATE_DIR_MODE` 0700 / `_PRIVATE_FILE_MODE` 0600, opening helpers, best-effort chmod of old transcripts                                                                                                                               |
| 2720–2981   | Session persistence               | `_session_slug` + `SessionManager`: save/list/resolve/load under `~/.ollamaquery.d/sessions/`; a name is one slot (re-saving updates it), `/resume` matches ids or names                                                                  |
| 2985–3574   | Composable agentic system prompt  | Role/format/example/rules blocks, style registry, `get_agentic_prompt`, persona composition                                                                                                                                               |
| 3578–4578   | Shell approval gate               | Tokenizer, breakdown, home/root hard block, CWD leniency, path-ACL operand scan, `check_shell_approval`                                                                                                                                   |
| 4582–4893   | Executor (container/host)         | `Executor`: host (`shell=True` behind the gate) or podman/docker image/`exec`, resource ceiling, mount→workdir mapping                                                                                                                    |
| 4897–5407   | Agentic tool definitions          | `DESTRUCTIVE_TOOLS`, `PLAN_MODE_TOOLS`, `PLAN_MODE_RUN_TOOLS`, `plan_active_tools`, the JSON schemas of the 14 tools                                                                                                                      |
| 5411–5566   | Session scratch dir               | `/tmp/ollamaquery/pid-<n>` creation, ownership/mode checks, dead-PID sweep, leaf-only cleanup                                                                                                                                             |
| 5570–5726   | Agentic tool handlers             | Shared helpers used by the handlers (binary sniffing, grep helpers, output caps)                                                                                                                                                          |
| 5730–8576   | Path ACL                          | `PathAcl` (rules, `evaluate`, `prompt`, `describe`), `_path_acl_decision`, `_is_blocked_system_file`, `_shell_acl_scan`                                                                                                                   |
| 8580–8719   | "Tool registry" marker            | **Misplaced banner** — only `_is_harmless_get_fetch` lives here; `ToolRegistry` is at 10422 (see next row)                                                                                                                                |
| 8723–10804  | MCP client + `ToolRegistry`       | `MCPServerConfig`/`MCPConfig`, config discovery/validation, JSON-RPC framing, `StdioMCPTransport`, `MCPConnection`, `MCPClient`, `MCPManager`; then `ToolRegistry` (10422) — the registry merges MCP tools, so it must follow the manager |
| 10807–10905 | Agentic session logger            | `AgenticLogger`: JSONL turns, retention cleanup, best-effort by contract                                                                                                                                                                  |
| 10909–11033 | Debugging                         | `DebugManager` (per-category levels) + `debug_log`                                                                                                                                                                                        |
| 11037–11379 | Request builder                   | `RequestBuilder`: payload shaping, output cap, images, per-backend inference/reasoning params, URL, stream tuple                                                                                                                          |
| 11383–11627 | Stats calculator                  | `StatsCalculator`: prefill/think/answer split, hybrid server+client timing, the stats line                                                                                                                                                |
| 11631–11985 | Stream parser                     | `StreamParser`: per-backend chunk shapes, SSE vs NDJSON, tool-call accumulation, sync aggregation, stream-completeness facts (`saw_final`/`finish_reason`/`bad_lines`)                                                                    |
| 11989–12698 | Model query                       | `ModelQuery`: transport, retries, display, the thin doors onto the three collaborators above                                                                                                                                              |
| 12702–13037 | Command handling                  | `ChatCompleter` (tab completion, incl. saved-session handles for `/resume`)                                                                                                                                                               |
| 13041–13951 | Input handling                    | `gather_user_input`, inline commands (`!`, `/curl`, `@file`), `_ExitRequested`, readline/history                                                                                                                                          |
| 13955–14351 | Tool-call parsing                 | Balanced-JSON primitives + `ToolCallParser` (strict vs lazy policy, 4-pass extraction)                                                                                                                                                    |
| 14355–14661 | Agentic timeout policy            | `TimeoutEvent` (frozen), `AgenticTimeoutPolicy` (states C / A / B, nudge, ceiling doubling)                                                                                                                                               |
| 14665–14874 | Agentic step feedback             | `StepFeedback`: live thinking, token-driven dots, idle wait mark (`Zz`), accumulated content for timeout analysis                                                                                                                                    |
| 14878–15163 | Agentic tool execution            | `ToolExecution`: run, cap observation, same-tool guard, message append (native vs inline)                                                                                                                                                 |
| 15167–15538 | Agentic finalizer                 | `AgenticFinalizer`: stream the final answer **or** run the tool call it carried and re-enter (bounded)                                                                                                                                    |
| 15542–16463 | Agentic runner                    | `ReActState` + `AgenticRunner`: `run` (skeleton), `_start_query`, `_run_step`, `call_step`, presentation, partial-write recovery, bounded continuation nudges (`cut`/`length`/`badstream`/`unparsed`)                                     |
| 16467–16954 | Outbound wire                     | `WireView`: `system_and_tools`, context bar, overhead, `/promptshow`, nudges, force-compact                                                                                                                                               |
| 16958–17431 | Conversation store                | `ConversationStore`: message list, session persistence (save/rename/resume binding, `active_id`/`active_name`), compaction orchestration                                                                                                  |
| 17435–19867 | Chat loop                         | `ChatLoop`: dispatcher + 28 `run_handle_*`, query path, agentic wiring, doors                                                                                                                                                             |
| 19871–20533 | Main entry point                  | Backend probes/checks (`check_*`), `resolve_connection`, `auto_detect_backend`, `main`, listing                                                                                                                                           |
| 20537–21171 | Argument parser                   | `argparse` surface (backend, sandbox, MCP, ACL, theme, batch)                                                                                                                                                                             |

## 3. Classes and their spans

Class-body AST spans (`end_lineno − lineno + 1`); `defs` counts class-level
functions including property accessors.

| Class                           | Lines       | Size | Defs | Role                                                                                    |
| ------------------------------- | ----------- | ---- | ---- | --------------------------------------------------------------------------------------- |
| `CommandContext`                | 2491–3066   | 576  | 25   | State singleton: connection, session, token counting, stats, agentic flags, `path_acl`  |
| `SessionAccess`                 | 2190–2488   | 299  | 6    | Session access inventory (A1): paths/commands, bwrap tag, JSONL `access`                |
| `SessionManager`                | 3936–4166   | 231  | 12   | Save/list/resolve/load sessions; names are slots, ids are stable                        |
| `Executor`                      | 8654–9226   | 573  | 20   | Host/container command runner (gate-aware; `exec` reuses a running container)           |
| `PathAcl`                       | 10277–10604 | 328  | 10   | Session-scoped allow/ask/deny policy (read/write, longest-prefix, ask is bypass-immune) |
| `MCPServerConfig` / `MCPConfig` | 13578–13630 | 53   | 2    | Parsed MCP configuration                                                                |
| `MCPToolEntry`                  | 14091–14143 | 53   | 4    | One catalog entry + plan-eligibility predicates                                         |
| `StdioMCPTransport`             | 14368–14511 | 144  | 10   | argv, no shell, own process group, stderr tail, framed writes                           |
| `InMemoryMCPTransport`          | 14514–14564 | 51   | 7    | Test transport                                                                          |
| `MCPConnection`                 | 14569–14719 | 151  | 7    | Reader thread, id→Event multiplexer, cancellation on timeout                            |
| `MCPClient`                     | 14737–14879 | 143  | 8    | initialize → initialized → paginated `tools/list` → `tools/call`                        |
| `MCPManager`                    | 14884–15117 | 234  | 19   | Lazy/eager startup, catalog aggregation, routing, policy sets                           |
| `ToolRegistry`                  | 15119–15527 | 409  | 14   | Dispatch, aliases, confirmation, plan surface, prompt block                             |
| `AgenticLogger`                 | 15569–15671 | 103  | 5    | JSONL session log, retention, fail-soft                                                 |
| `DebugManager`                  | 15678–15762 | 85   | 6    | Per-category debug levels                                                               |
| `RequestBuilder`                | 15806–16161 | 356  | 13   | Outbound payload shaping (backend is an explicit argument)                              |
| `StatsCalculator`               | 16167–16409 | 243  | 6    | Timings, token split, stats line                                                        |
| `StreamParser`                  | 16415–16785 | 371  | 9    | Chunk shapes + sync aggregation (late-bound line provider)                              |
| `ModelQuery`                    | 16839–17591 | 753  | 39   | Transport, retries, display, doors onto the three collaborators                         |
| `ChatCompleter`                 | 17651–17951 | 301  | 12   | Tab completion (commands, paths, models, subcommands, MCP servers, session handles)     |
| `CoreHTMLStripper`              | 18785–18861 | 77   | 6    | stdlib HTMLParser subclass for `/curl` and `fetch_url`                                  |
| `ToolCallParser`                | 19067–19294 | 228  | 9    | Strict/lazy tool-call extraction                                                        |
| `TimeoutEvent`                  | 19321–19337 | 17   | 0    | Frozen facts of one timed-out step                                                      |
| `AgenticTimeoutPolicy`          | 19340–19604 | 265  | 8    | 3-state escalation + nudge + ceiling                                                    |
| `StepFeedback`                  | 19630–19830 | 201  | 8    | Per-step streaming feedback                                                             |
| `ToolExecution`                 | 19845–20144 | 300  | 9    | Tool run, observation cap, same-tool guard, message append                              |
| `AgenticFinalizer`              | 20150–20533 | 384  | 8    | Post-loop phase + bounded streaming re-entry                                            |
| `ReActState`                    | 20550–20592 | 43   | 1    | One query's mutable state (replaces nine threaded locals)                               |
| `AgenticRunner`                 | 20595–21491 | 897  | 25   | The step loop                                                                           |
| `WireView`                      | 21497–21981 | 485  | 19   | Outbound request accounting + `/promptshow`                                             |
| `ConversationStore`             | 21988–22458 | 471  | 21   | Messages, session binding, compaction                                                   |
| `ChatLoop`                      | 22465–25449 | 2985 | 126  | Command dispatch, query paths, doors onto every collaborator                            |

## 4. Architecture

```
main()                                    20786+
 ├── resolve_connection()                  CLI > saved MRU > auto-detect > default
 ├── list_models_*()                       no context needed
 └── ChatLoop.run()                       17176
      ├── CommandContext (singleton)      1409   all shared state
      ├── ChatCompleter                   12629  readline completion
      ├── ModelQuery                      11884
      │    ├── RequestBuilder             10986  payload/URL shaping
      │    ├── StreamParser               11580  chunk shapes, SSE vs NDJSON
      │    └── StatsCalculator            11332  timings + stats line
      ├── WireView                        16213  what the next request costs
      ├── ConversationStore               16704  messages + sessions + compact
      └── agentic chain
           ├── ToolCallParser             14007
           ├── AgenticRunner              15467  step loop
           │    ├── StepFeedback          14563  streaming feedback
           │    ├── ToolExecution         14772  run + observe
           │    └── AgenticTimeoutPolicy  14280  escalation
           ├── AgenticFinalizer           15039  final answer / re-entry
           └── ToolRegistry               10422  dispatch
                ├── Executor              4531   host / container
                ├── PathAcl               5710   read/write policy
                └── MCPManager            10187  external tools
```

Every extracted collaborator keeps **thin doors** on its old owner (e.g.
`ChatLoop._execute_tool_calls` → `ToolExecution.run_calls`,
`ModelQuery.build_request_payload` → `RequestBuilder.build_request_payload`).
That is what lets the 1 249 tests stub the old surface and still intercept.

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
