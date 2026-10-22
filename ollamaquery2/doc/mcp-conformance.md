# MCP conformance review — ollamaquery2 vs. mcp-cpp (2025-11-25)

Review of the MCP client slice in `ollamaquery2.py` against the reference C++20 SDK
[Neumann-Labs/mcp-cpp](https://github.com/Neumann-Labs/mcp-cpp), which pins the wire
format to MCP revision **2025-11-25**.

| Field     | Value                                                                   |
| --------- | ----------------------------------------------------------------------- |
| Reviewed  | `ollamaquery2.py` v0.3.0, MCP block lines ~6292–7711, 2026-10-22        |
| Reference | `Neumann-Labs/mcp-cpp` `main`, latest commit 2026-07-06, rev 2025-11-25 |
| Baseline  | `python3 -m unittest tests.test_mcp` — 46 tests OK                      |
| Verdict   | Core wire behavior conforms; 6 P1 gaps, all local and small to fix      |

Line numbers cited below are from v0.3.0. Prefer the symbol names when they drift.

---

## 1. Scope and method

`mcp-cpp` is a **full-stack** SDK (server **and** client) covering tools, resources,
prompts, completion, logging, progress, cancellation, sampling, roots, elicitation,
tasks, pagination, stdio and Streamable HTTP + OAuth. `ollamaquery2.py` is deliberately
a **stdio, tools-only client**. Feature parity is therefore not the goal; the useful
comparison is:

1. wire-level conformance with revision 2025-11-25 (what we send, what we accept),
2. client robustness at protocol edges (timeouts, cancellation, pagination, errors),
3. whether the unimplemented surface fails closed rather than speaking the protocol wrongly.

Method: read the reference `include/mcp/protocol.hpp`, `include/mcp/error.hpp`,
`include/mcp/client.hpp`, `src/client.cpp`, `src/session.cpp`,
`tests/audit_regression_test.cpp`, plus the official spec text for cancellation and
lifecycle. Then traced the corresponding code paths in `ollamaquery2.py` and confirmed
behavior against the 46-test local suite (all green before the review, so nothing below
is a known-red test).

---

## 2. Reference tooling

### 2.1 Trust assessment

| Field         | Value                                                                                      |
| ------------- | ------------------------------------------------------------------------------------------ |
| Repository    | https://github.com/Neumann-Labs/mcp-cpp                                                    |
| What it is    | C++20 SDK for MCP: `Server`, `Client`, `Session`, `stdio` + Streamable HTTP transports     |
| Revision      | `kLatestProtocolVersion = "2025-11-25"` (`include/mcp/protocol.hpp:35`)                    |
| License       | **GPL-3.0** (`LICENSE` is the GPL-3 text; the README claims Apache-2.0 — the file governs) |
| Maturity      | Beta, 62 commits, created 2025-03-04, active as of 2026-07-06                              |
| Adoption      | 65 stars, 6 forks — reference-grade, not ecosystem-standard                                |
| Trust signals | 19 test files (incl. `audit_regression_test.cpp`), ASAN + TSAN CI, Linux/macOS/Windows     |
| Cautions      | Single-vendor project; C++ only (no Python interop); license metadata is contradictory     |
| Verdict       | **Trustworthy as a wire-format oracle and test fixture. Not a code source to copy.**       |

Why it earns trust:

- **Spec-faithful by construction.** Field names and method names mirror the official
  `schema/2025-11-25/schema.ts` verbatim; the README states this as the primary design goal.
- **Conformance tests, not smoke tests.** `pagination_test.cpp`, `tasks_test.cpp`,
  `elicitation_test.cpp` (form + URL), `audit_regression_test.cpp` (CR stripping,
  non-string method, whole-float request ids, disconnect races).
- **Sanitizer CI.** ASAN + TSAN jobs, plus an install/downstream-consume job and a
  three-OS matrix — the usual hallmarks of a maintained protocol library.
- **Honest changelog.** Post-0.1.0 entries document bugs found by real-client interop
  (port handling, 404 for terminated `Mcp-Session-Id`, per-call thread removal), which is
  a better signal than a clean history.

Why not to copy from it:

- **GPL-3.0.** Copying code into this repo would relicense it. Do not lift implementation
  code; reading method names, field names and JSON shapes for interoperability is a
  different act from copying source.
- **Beta + single vendor.** The C++ API is explicitly expected to change before 1.0.

### 2.2 Using it as a test oracle

Built examples (`-DMCP_BUILD_EXAMPLES=ON`, POSIX only for stdio):

```bash
git clone https://github.com/Neumann-Labs/mcp-cpp.git /tmp/opencode/mcp-cpp
cmake -S /tmp/opencode/mcp-cpp -B /tmp/opencode/mcp-cpp/build -G Ninja -DMCP_BUILD_EXAMPLES=ON
cmake --build /tmp/opencode/mcp-cpp/build --target calculator_server mcp_notes_server
```

Then point a throwaway config at it:

```json
{
  "servers": {
    "calc": {
      "transport": "stdio",
      "command": ["/tmp/opencode/mcp-cpp/build/examples/calculator_server/calculator_server"]
    }
  }
}
```

`calculator_server` is the best fixture for this client's surface:

- `serverInfo = {name: "mcp-cpp-calculator", version: "0.1.0"}` + non-empty `instructions`
  — exercises the captured-but-unused `instructions` field (§4, P2-9).
- Four tools (`add`, `subtract`, `multiply`, `divide`) with `title` **and** `description`
  — exercises the title/description fallback in `_mcp_tool_to_agentic_def`.
- `divide(a, 0)` returns `isError: true` — exercises `_mcp_result_to_output`'s
  `success: false` mapping without a raised exception.
- **No `annotations` at all** — exercises the fail-closed policy: every tool is
  unannotated, so `destructive_names()` gates all four behind confirmation and none is
  plan-eligible. That is the intended behavior, not a fixture bug; use `auto_confirm`
  or `confirm_tools` when writing tests.
- It has no pagination or image/audio content; those stay covered by the scripted fake
  server in `tests/test_mcp.py`. `mcp_notes_server` adds elicitation (needs
  `libsqlite3-dev`).

Running a GPL-3.0 **binary** locally for testing carries no licensing obligation on this
repo (no distribution). Do not vendor the source.

A cheap future win: add an opt-in integration test that skips unless
`MCP_CPP_CALCULATOR_SERVER` is set, mirroring how `tests/test_agentic.py` gates on a
live backend. That gives an external, independent conformance check instead of only our
own fake server.

---

## 3. What already conforms

| Area                   | Reference behavior                                | Local behavior                                      | Status |
| ---------------------- | ------------------------------------------------- | --------------------------------------------------- | ------ |
| Revision pinning       | `"2025-11-25"`                                    | `MCP_PREFERRED_PROTOCOL_VERSION` (6318) + allowlist | OK     |
| Handshake              | initialize → result → `notifications/initialized` | 7422–7439                                           | OK     |
| Client capabilities    | only what is implemented                          | `"capabilities": {}` (7428)                         | OK     |
| `tools/list`           | follow `nextCursor` until absent                  | 7453–7466                                           | OK     |
| `tools/call`           | `{name, arguments}`                               | 7468–7473                                           | OK     |
| Content blocks         | text / image / audio / resource_link / resource   | 6747–6763                                           | OK     |
| `CallToolResult`       | `content`, `structuredContent`, `isError`         | 6799–6823                                           | OK     |
| Server `ping`          | reply `{}`                                        | 7441–7445                                           | OK     |
| Unknown server request | reply `-32601`                                    | 7289–7299 (matches `session.cpp:369`)               | OK     |
| Timeout cancellation   | `notifications/cancelled` + `requestId`/`reason`  | 7341–7345 (except `initialize`, P1-1)               | OK*    |
| stdio framing          | newline-delimited JSON, no embedded newlines      | 6590–6593                                           | OK     |
| Frame ceiling          | configurable max frame, resync after oversize     | 6608–6652, 16 MiB                                   | OK     |
| CR tolerance           | `\r\n` accepted                                   | `_mcp_decode_frame` `.strip()`                      | OK     |
| stderr isolation       | diagnostics never on the JSON-RPC stream          | 7100–7109                                           | OK     |
| Process lifecycle      | no orphaned children                              | own session + `killpg` (7078–7164)                  | OK     |
| Pagination termination | absent `nextCursor` ends the loop                 | 7462                                                | OK     |

\* The cancellation *mechanism* conforms; the `initialize` carve-out does not (§4 P1-1).

The annotation policy is **stricter than the reference**, which is correct for an
LLM-facing surface: `mcp-cpp` only parses the hints (policy is the application's job),
while `ollamaquery2.py` refuses to infer "safe" from an absent `readOnlyHint`
(6665–6680, 6877–6880). That matches the spec defaults (`readOnlyHint=false`,
`destructiveHint=true`) and should not be relaxed.

---

## 4. Findings

| ID    | Severity | Area            | Effect                                                                |
| ----- | -------- | --------------- | --------------------------------------------------------------------- |
| P1-1  | High     | Cancellation    | `initialize` is cancelled on timeout — spec violation                 |
| P1-2  | High     | Connect         | Tools-less server can never connect; session torn down                |
| P1-3  | High     | Errors          | `error.data` discarded (validation detail, URL elicitation)           |
| P1-4  | High     | Pagination      | Repeating `nextCursor` hangs connect indefinitely                     |
| P1-5  | High     | Tasks           | `taskSupport: required` always fails; task ack reads as empty success |
| P1-6  | High     | Cancellation    | Inbound `notifications/cancelled` ignored — 120 s stall               |
| P2-7  | Medium   | Staleness       | `tools/list_changed` notice written to a field nothing reads          |
| P2-8  | Medium   | Progress        | No `progressToken`, no `notifications/progress` — silent calls        |
| P2-9  | Medium   | Prompt          | `serverInfo.instructions` captured, never used                        |
| P2-10 | Medium   | Diagnostics     | `_stderr_tail` collected, never surfaced                              |
| P2-11 | Medium   | Framing         | Non-string `method` answered; batch arrays dropped while advertised   |
| P2-12 | Medium   | Env             | Child env narrower than mainstream clients (breaks real servers)      |
| P2-13 | Medium   | Schema fidelity | Normalization leaks into the native `tools` array                     |
| P3-14 | Low      | Spill           | Deterministic filename overwrites an earlier pointer                  |
| P3-15 | Low      | Content         | image/audio flattened, never reaches a vision model                   |
| P3-16 | Low      | Logging         | `notifications/message` dropped, no debug sink                        |

### P1-1 — `initialize` must never be cancelled

**Where:** `MCPConnection.request()` timeout branch, 7338–7346; used by
`MCPClient._initialize()` at 7424.

**Spec:** cancellation.mdx line 35 — *"The `initialize` request **MUST NOT** be cancelled
by clients"*. lifecycle.mdx 208–209 still requires a cancellation on timeout for other
requests, so the fix is a carve-out, not removing cancellation.

**Impact:** on an init timeout we send `notifications/cancelled` for a request that had
no session yet. A server that logs or enforces the rule can classify the peer as broken;
either way the notification is meaningless.

**Fix:** skip the notify when `method == "initialize"`. Test: an init timeout produces
zero `notifications/cancelled` frames on the transport.

### P1-2 — a tools-less server can never connect

**Where:** `MCPClient.connect()` 7414–7419 closes the transport on *any* exception from
`self.list_tools()`; `list_tools()` 7453–7466 re-raises protocol errors.

**Reference:** `Client::list_tools()` returns a future; a failed call is an error for
that call only. The session stays connected.

**Impact:** a resources-only or prompts-only server answers `-32601` to `tools/list`, so
the whole session is torn down and `/mcp status` shows "not connected" forever. The user
gets an opaque JSON-RPC error with no hint that the server simply has no tools.

**Fix:** check `capabilities.tools` first; treat `-32601`/`-32602` from `tools/list` as
"no tools" (`self.tools = []`) and keep the session; only tear down on transport failure
(`MCPConnectionClosed`, `MCPTimeout`).

### P1-3 — `error.data` is discarded

**Where:** `MCPProtocolError` 6357–6363; raise site 7351–7354.

**Reference:** `ErrorObject { int code; std::string message; nlohmann::json data; }` with
`has_data()`, plus `error_code::url_elicitation_required = -32042` (2025-11-25 URL
elicitation: the action the human must take lives in `data`).

**Impact:** server-side validation errors are the model's main self-correction signal
(field name, expected type, offending value). Today it sees `JSON-RPC error -32602:
Invalid params` and nothing else. URL elicitation cannot be surfaced at all.

**Fix:** store `data` on the exception and append a capped JSON rendering to the message
the tool result carries back to the model.

### P1-4 — unbounded pagination

**Where:** `list_tools()` 7455–7465.

**Reference:** cursor handling is validated and covered by `tests/pagination_test.cpp`.

**Impact:** a buggy or hostile server that echoes the same `nextCursor`, or returns a new
one forever, hangs `connect()` (and therefore the terminal) with no timeout — the loop
itself has no wall-clock bound.

**Fix:** keep a seen-cursor set plus a page cap (e.g. `MCP_MAX_LIST_PAGES = 100`) and
raise `MCPProtocolError` naming the offending cursor.

### P1-5 — 2025-11-25 tasks are invisible

**Where:** `_mcp_tool_to_agentic_def` 6728–6735 ignores `Tool.execution`;
`_mcp_result_to_output` 6799–6823 finds no `content` in a `CreateTaskResult` and returns
`{"success": True, "output": ""}`.

**Reference:** `ToolExecution.task_support` = `forbidden|optional|required` (default
`optional`), `CreateTaskResult{task}`, `tasks/get|result|list|cancel`,
`notifications/tasks/status`.

**Impact:** two distinct failures. A tool declaring `taskSupport: "required"` always
errors (we never send the `task` field), and the model cannot know why. A task-ack result
is flattened to an **empty success**, which reads as "the tool did nothing" — worse than
an error, because the model proceeds as if the work succeeded.

**Fix (minimum, no task engine):** surface `task=` in `/mcp tools --all`; refuse to call
a `required` tool with an explicit message; detect `result["task"]` and return a
non-empty failure ("task created; polling unsupported"). Full task support is a separate
slice and should be gated behind a `tasks` capability declaration.

### P1-6 — inbound `notifications/cancelled` ignored

**Where:** `_handle_notification` 7447–7451 only inspects `tools/list_changed`;
`request()` waits the full timeout at 7338.

**Reference:** `Session` routes a cancellation to the pending request and resolves it
with an error.

**Impact:** when the server abandons our request, we stall for the entire
`call_timeout_seconds` (120 s default) and then send a cancel back to a request that is
already dead. Inside the ReAct loop that is a 2-minute freeze with no feedback.

**Fix:** index pending requests by id, let `_handle_notification` resolve the matching
one with a "cancelled by server" marker, and raise immediately.

### P2 — capability and staleness gaps

- **P2-7 `list_changed` is a dead-end** (7450–7451). The notice is appended to
  `MCPClient.errors`, and nothing reads that list (only `manager.errors` is printed, at
  7709). The catalog goes stale silently. Fix: re-list + `_refresh(server)` on the
  reader thread (the manager already holds an `RLock`) and print a one-liner.
- **P2-8 no progress channel** (7468–7473). `tools/call` sends no
  `_meta.progressToken` and `notifications/progress` is dropped, so a 120 s call is
  silent. The agentic heartbeat cannot help: it is driven by model streaming, and the
  tool call blocks the loop. Reference: `set_progress_handler` + typed params.
- **P2-9 `instructions` unused** (7438). Captured, never read. Either inject it into the
  agentic prompt block, clearly labelled as server-supplied and namespaced per server, or
  drop the field. Leaving it dead is the worst option because it looks implemented.
- **P2-10 stderr discarded** (7061, 7106). `_stderr_tail` is written and never read, so a
  server that dies at startup yields only "connection closed". Print the tail on connect
  failure and in `/mcp status`. Reference has leveled logging (`mcp::log`).
- **P2-11 framing nits.** A non-string `method` is answered instead of dropped (7272)
  — reference asserts `NonStringMethodIsDropped`. `"jsonrpc": "2.0"` is never validated.
  JSON-RPC **batch arrays** are dropped by `_mcp_decode_frame`, which returns only dicts
  (6596–6605) — fine for 2025-06-18+ (batching removed), wrong for `2024-11-05` and
  `2025-03-26`, both of which remain in `MCP_PROTOCOL_VERSIONS`.
- **P2-12 child environment too narrow** (7074–7076). Only
  `PATH, HOME, LANG, LC_ALL, TERM` are passed. The TS SDK's default set also includes
  `SHELL`, `USER`, `LOGNAME`; the Python SDK inherits the parent wholesale. Dropping
  `KUBECONFIG`, `VIRTUAL_ENV`, `TMPDIR`, `SSL_CERT_FILE` and proxy variables will break
  real servers (`uvx`, cloud-credential servers, anything wrapping `oc`). Fix: start from
  `dict(os.environ)` and overlay `cfg.env` (the config already wins).
- **P2-13 schema rewrite leaks into the native array** (6691–6725, 7608–7619). The
  normalized schema (synthetic `type`, empty `description`, forced `required: []`) feeds
  both the inline renderer and `build_openai_specs()`. For a `$ref`/`oneOf` property this
  adds a contradictory sibling `type`; the reference passes `input_schema` through
  untouched. Keep normalization for `get_system_prompt_block()` and emit the raw
  `inputSchema` for the native `tools` array.

### P3 — minor

- **P3-14** `_mcp_spill_path` (6766–6770) is deterministic per server/tool, so a second
  call overwrites the file an earlier `read_file` pointer still references. Add a call
  counter or timestamp to the name (the cluster spill has the same shape).
- **P3-15** `_mcp_content_to_text` (6747–6763) turns image/audio blocks into text
  placeholders, so MCP vision results never reach a vision model. At least the
  placeholder is honest; wiring `ctx.images` would be a real improvement.
- **P3-16** `notifications/message` (RFC-5424 logging) is dropped with no debug sink.

---

## 5. Capability matrix

| Capability                                | mcp-cpp | ollamaquery2 | Notes                                   |
| ----------------------------------------- | ------- | ------------ | --------------------------------------- |
| initialize / capabilities negotiation     | Yes     | Yes          | P1-1 carve-out missing                  |
| tools/list + pagination                   | Yes     | Yes          | P1-4 guard missing                      |
| tools/call + content blocks + `isError`   | Yes     | Yes          | image/audio flattened (P3-15)           |
| ping                                      | Yes     | Yes          |                                         |
| cancellation                              | Yes     | Partial      | P1-1, P1-6                              |
| progress                                  | Yes     | No           | P2-8                                    |
| tools `list_changed`                      | Yes     | No           | notice only, unread (P2-7)              |
| tasks (`tools/call` augmentation)         | Yes     | No           | P1-5                                    |
| resources (list/read/templates/subscribe) | Yes     | No           | documented, not exposed                 |
| prompts (list/get)                        | Yes     | No           | documented, not exposed                 |
| completion                                | Yes     | No           | documented, not exposed                 |
| logging (`setLevel`, notifications)       | Yes     | No           | P3-16                                   |
| sampling                                  | Yes     | No           | not declared — fail closed              |
| roots                                     | Yes     | No           | not declared — fail closed              |
| elicitation (form + URL)                  | Yes     | No           | not declared; `-32042` data lost (P1-3) |
| stdio transport                           | Yes     | Yes          | env whitelist narrow (P2-12)            |
| Streamable HTTP + OAuth 2.1               | Yes     | No           | refused at config validation            |

---

## 6. Fix plan

| Group | Items                        | Size      | Test classes                                       |
| ----- | ---------------------------- | --------- | -------------------------------------------------- |
| 1     | P1-1, P1-2, P1-3, P1-4, P1-6 | ~40 lines | `TestConnection`, `TestStdioServerEndToEnd`        |
| 2     | P2-7, P2-10                  | ~30 lines | `TestStdioServerEndToEnd`, `TestListingAndCommand` |
| 3     | P1-5 (minimal), P2-8         | medium    | `TestToolRegistryIntegration`, `TestConnection`    |
| 4     | P2-11, P2-12, P2-13          | small     | `TestFraming`, `TestConnection`                    |
| 5     | P3 items                     | small     | `TestSpill`                                        |

Group 1 specifics: an init timeout must emit no `notifications/cancelled`; a `-32601`
from `tools/list` must leave the session connected with an empty catalog; a repeated
`nextCursor` must raise instead of looping; an inbound cancellation must abort the wait;
`error.data` must reach the tool-result text.

Group 1 is self-contained, has obvious test hooks in `tests/test_mcp.py`, and is the
recommended first pass. P1-5's full task engine should stay out until a server that
actually emits tasks is in the test matrix.

---

## 7. Deliberately unimplemented (fail-closed)

These are documented in the source header (6292–6308) and enforced honestly; conformance
is preserved precisely because we do **not** advertise them:

- `initialize` sends `"capabilities": {}`, so servers must not send sampling, roots or
  elicitation requests. If one does anyway, `_handle_server_request` returns `None`,
  which `_route` converts to `-32601` — the correct refusal.
- HTTP transports are flagged by `validate_mcp_server` with a warning and refused at
  connect time, instead of being silently spoken wrongly; an existing Claude/VS Code
  config with `url` still parses (and warns).
- Server-initiated features that a config *enables* (`allow_sampling`,
  `allow_elicitation`, `allow_roots`, `oauth`) are accepted by the parser but reported as
  unimplemented, so a config never over-promises.

Keep this discipline: adding a capability declaration is a promise to implement the
matching method table.

---

## 8. Reference shapes for later implementation

Useful when adding tasks, elicitation or error passthrough. All names are verbatim from
`include/mcp/protocol.hpp` (revision 2025-11-25).

| Wire item            | Reference definition                                                                                               |
| -------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `initialize` params  | `protocolVersion`, `capabilities`, `clientInfo`                                                                    |
| `initialize` result  | `protocolVersion`, `capabilities`, `serverInfo`, `instructions?`, `_meta?`                                         |
| `ClientCapabilities` | `experimental?`, `roots?`, `sampling?`, `elicitation?`, `tasks?`                                                   |
| `ServerCapabilities` | `experimental?`, `logging?`, `completions?`, `prompts?`, `resources?`, `tools?`, `tasks?`                          |
| `Tool`               | `name`, `title?`, `description?`, `inputSchema`, `outputSchema?`, `annotations?`, `icons?`, `execution?`, `_meta?` |
| `ToolAnnotations`    | `title?`, `readOnlyHint?`, `destructiveHint?`, `idempotentHint?`, `openWorldHint?`                                 |
| `ToolExecution`      | `taskSupport`: `forbidden` / `optional` (default) / `required`                                                     |
| `CallToolResult`     | `content`, `structuredContent?`, `isError?`, `_meta?`                                                              |
| `CreateTaskResult`   | `task { taskId, status, statusMessage?, createdAt, lastUpdatedAt, ttl?, pollInterval? }`                           |
| `ErrorObject`        | `code`, `message`, `data` (optional)                                                                               |
| Error codes          | JSON-RPC `-32700/-32600/-32601/-32602/-32603`; MCP `-32042` (URL elicitation required)                             |

Two normative points that are easy to miss:

1. `initialize` MUST NOT be cancelled (cancellation.mdx:35); every other request SHOULD
   be cancelled on timeout (lifecycle.mdx:208–209).
2. `taskSupport: "required"` means the client must augment the call with a task. Without
   a tasks capability, the tool must not be offered to the model at all.

---

## 9. Provenance

- Reference read at `Neumann-Labs/mcp-cpp@main` on 2026-10-22 (`protocol.hpp`, `error.hpp`,
  `client.hpp`, `client.cpp`, `session.cpp`, `audit_regression_test.cpp`, `CHANGELOG.md`,
  `examples/calculator_server/main.cpp`, `.github/workflows/ci.yml`, `LICENSE`).
- Spec text read from `modelcontextprotocol/modelcontextprotocol` —
  `docs/specification/2025-06-18/basic/utilities/cancellation.mdx` and
  `.../basic/lifecycle.mdx` (the 2025-11-25 schema is the reference's pinned revision; the
  cancellation and lifecycle rules are unchanged in substance).
- Local baseline verified with `python3 -m unittest tests.test_mcp -v` (46 tests OK).
