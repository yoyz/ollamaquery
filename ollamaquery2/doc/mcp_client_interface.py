# ============================================================================
# doc/mcp_client_interface.py (design recipe — not imported by ollamaquery2.py)
#
# PROTOTYPE / RECIPE for adding a full-fledged MCP (Model Context Protocol)
# client to ollamaquery2.py.
#
# This file is NOT imported by ollamaquery2.py. It is a design artifact: it
# contains the protocol cheat-sheet, the config schema, the data shapes, and a
# skeleton of every class/method you need, with the algorithm spelled out in
# each docstring. Pure helpers (config load, SSE parsing, tool conversion,
# result mapping) are given as reference implementations so the wiring is
# unambiguous; anything that touches a live process/socket is a stub marked
# `STUB -> implement in ollamaquery2.py`.
#
# Run `python3 doc/mcp_client_interface.py` to print the recipe.
# Run `python3 doc/mcp_client_interface.py --config ~/.ollamaquery.d/mcp.json`
# to dry-run config parsing + validation (no server is contacted).
#
# Target server: containers/kubernetes-mcp-server (Go, MCP go-sdk v1.8.0),
# which supports protocol versions 2026-07-28, 2025-11-25, 2025-06-18,
# 2025-03-26, 2024-11-05. A simple initialize-based client should request
# 2025-11-25: the server negotiates down and stays on the classic lifecycle.
# ============================================================================
"""Recipe + prototype for an MCP client inside ollamaquery2.py.

============================================================================
0. WHY THIS IS A SEPARATE, SELF-CONTAINED LAYER
============================================================================
ollamaquery2 is an *agent runtime* (model + ReAct loop + tools + UX).
An MCP server is a *tool provider* (JSON-RPC over stdio or HTTP).
The client's whole job is:

    MCP tool list  ->  AGENTIC_TOOL_DEFS / openai_tools_spec
    model tool call ->  tools/call  ->  observation text

Everything else (compaction, sessions, confirmations, spill files) is already
in ollamaquery2 and must be *reused*, not reimplemented.

============================================================================
1. PROTOCOL CHEAT-SHEET (what actually goes on the wire)
============================================================================
Transport 1 - stdio (default for a local server):
  * Spawn the server process yourself (no shell).
  * Frames are NEWLINE-DELIMITED JSON-RPC 2.0 objects on the child's
    stdin/stdout. stdout is protocol-only; all logs must go to stderr.
  * One reader thread demultiplexes responses by `id`; a write lock serializes
    frames. Max frame the SDK buffers: 16 MiB (DefaultMaxLineLength).

Transport 2 - Streamable HTTP (when the server sets `port = "..."`):
  * Endpoint is POST /mcp (kubernetes-mcp-server registers exactly `/mcp`).
  * Headers on every POST:
        Content-Type: application/json
        Accept: application/json, text/event-stream
    After initialize, also send:
        Mcp-Session-Id: <value returned on the initialize response>
        MCP-Protocol-Version: <negotiated version>
  * The response body is either `application/json` (single object) or
    `text/event-stream` (SSE; each `data:` line is a JSON-RPC object).
  * A notification POST returns `202 Accepted` with an empty body.
  * A separate GET (same URL, Accept: text/event-stream, session id) opens the
    server-initiated stream. DELETE terminates the session.
  * 404 on a session id => session expired; re-run initialize.
  * Optional OAuth: an unauthenticated call returns 401 with
        WWW-Authenticate: Bearer resource_metadata="<url>"
    Discovery endpoints on this server:
        /.well-known/oauth-protected-resource   (RFC 9728)
        /.well-known/oauth-authorization-server (RFC 8414)
        /.well-known/openid-configuration
  * Legacy SSE transport (2024-11-05, GET /sse + POST message endpoint) is
    optional; skip it unless you must talk to an old server.

JSON-RPC 2.0 envelope:
  request       {"jsonrpc":"2.0","id":1,"method":"tools/call","params":{...}}
  notification  {"jsonrpc":"2.0","method":"notifications/initialized"}
  response ok   {"jsonrpc":"2.0","id":1,"result":{...}}
  response err  {"jsonrpc":"2.0","id":1,"error":{"code":-32601,"message":...}}

Lifecycle (classic, <= 2025-11-25):
  1. client -> initialize {protocolVersion, capabilities, clientInfo}
  2. server -> InitializeResult {protocolVersion, capabilities, serverInfo,
                                 instructions}
  3. client -> notifications/initialized           (a notification, no id)
  4. client -> tools/list (paginated with `nextCursor`)
             -> optional resources/list, resources/templates/list,
                prompts/list, logging/setLevel
  5. use tools/call; server may send notifications/progress mid-call.

Methods you will care about:
  initialize, ping, tools/list, tools/call,
  resources/list, resources/templates/list, resources/read,
  resources/subscribe, resources/unsubscribe,
  prompts/list, prompts/get, completion/complete, logging/setLevel
Notifications:
  notifications/initialized, notifications/cancelled, notifications/progress,
  notifications/message (logging), notifications/tools/list_changed,
  notifications/resources/list_changed, notifications/resources/updated,
  notifications/prompts/list_changed
Server -> client requests (implement handlers):
  ping, roots/list, sampling/createMessage, elicitation/create

Error codes:
  -32700 Parse error        -32600 Invalid Request   -32601 Method not found
  -32602 Invalid params     -32603 Internal error
  MCP extras: -32002 Resource not found, -32022 Unsupported protocol version,
  -32042 URL elicitation required.

============================================================================
2. DOUBLE-THINK (design decisions and the trade-offs)
============================================================================
D1. stdio vs HTTP.
    Prefer stdio for a *local* server: no port, no auth, process lifetime is
    the connection. Use HTTP only when the server is remote / shared / already
    running. Support both behind one MCPTransport interface.

D2. threads vs asyncio.
    ollamaquery2 is sync + threads (Executor, reader threads, signals).
    Stay sync: one reader thread per transport, a `pending: dict[id, Event]`,
    one write lock. Do NOT introduce asyncio - it would fight the existing
    Ctrl+C/signal handling.

D3. multiplexing is mandatory.
    The server can send a request (ping/roots/sampling/elicitation) or a
    notification while a tools/call is outstanding. A reader thread must route
    by `id` vs `method`, never assume request/response pairing.

D4. tool-name namespacing.
    Multiple servers will collide (`resources_list`, `pods_list`, ...).
    Namespace every MCP tool: `<prefix>__<tool>` (prefix defaults to server
    name). OpenAI's native tools API constrains function names to
    ^[a-zA-Z0-9_-]{1,64}$ -> sanitize and, if needed, truncate + append a
    short hash, keeping a reverse map back to (server, tool).

D5. schema rendering.
    MCP `inputSchema` is JSON Schema. For the native tools API you can pass it
    almost verbatim. For inline (non-native) tool format,
    `ToolRegistry.get_system_prompt_block` only renders flat `properties` /
    `required`, and it does `pdef['type']` + `pdef['description']` literally:
    a property without either key raises KeyError. MCP schemas are free to omit
    descriptions and use oneOf/$ref. Therefore either (a) force native tool
    format whenever any MCP tool is active, or (b) run every MCP schema through
    `normalize_schema_for_inline()` (below) so the inline renderer is safe.
    Also make `_plan_allowed_tools()` merge MCP definitions - it reads
    AGENTIC_TOOL_DEFS directly, so mutating a local dict is not enough.

D6. safety must be three-state.
    MCP annotations are *hints* and may be absent. Spec defaults: readOnlyHint
    false, destructiveHint true, idempotentHint false, openWorldHint true. Map
    to:
        readOnlyHint == True              -> allow in plan mode
        destructiveHint == True           -> confirm (and never plan)
        unknown / both absent             -> ask (conservative)
    Reuse ollamaquery2's `_confirm`, `DESTRUCTIVE_TOOLS`, path/plan gates.
    Do not silently trust a server's self-description.

D7. results can be huge.
    Reuse the existing spill pattern so a large tools/call result becomes a
    path + preview, pageable by read_file. Two details are load-bearing:
    (a) spill into the SAME dir the cluster/fetch spills use
    (~/.ollamaquery.d/spill) - one dir, one cleanup pass, one Path-ACL rule;
    (b) register `acl.add("read", "allow", <spill dir>, source="session")`
    before handing the path to the model. Strict plan mode confines every path
    to the CWD and turns `ask` into deny, so skipping the rule produces a
    pointer the model cannot follow. Cap by `resolve_max_result_chars()`,
    which treats the configured `max_result_chars` as a ceiling and lowers it
    to a fraction of the live context window.

D8. server-initiated features are opt-in.
    `sampling/createMessage` makes the *server* drive your model (cost, prompt
    leakage). `elicitation/create` makes the *server* prompt your user. Both
    off by default, enabled per-server in config, and gated on a TTY.

D9. lazy startup.
    Connecting to N servers at launch is slow and can hang. Default `startup =
    "lazy"`: connect on first tool use; expose metadata from config meanwhile.
    Eager is available per server. Never block the chat prompt.

D10. single-file placement.
    Put this code as one contiguous section in ollamaquery2.py, after the
    constants/small helpers and BEFORE `class ToolRegistry` (ToolRegistry wants
    the manager), i.e. roughly between the cluster-query helpers (~line 4030)
    and `class ToolRegistry` (~line 6045). Use forward references (strings) to
    avoid import-order issues. Keep stdlib-only (no `mcp`, no `requests`).

============================================================================
3. CONFIG FILE (how the client finds a server)
============================================================================
There is no OS-level "MCP socket registry"; configuration is client-side. For
stdio the "socket" is a pipe you create by launching the command. For HTTP it
is a URL. Put the file at (first hit wins):
    --mcp-config <path>
    $OLLAMAQUERY_MCP_CONFIG
    ./.ollamaquery-mcp.json                (project)
    ~/.ollamaquery.d/mcp.json              (user; also chmod 0600)
The loader also accepts the Claude/VS Code shape `{"mcpServers": {...}}` so
you can reuse existing configs. See `mcp.json.example` next to this prototype.

============================================================================
4. INTEGRATION POINTS IN ollamaquery2.py (exact hook list)
============================================================================
  CommandContext.__init__                 ~:1297  add mcp_manager + config
  AGENTIC_TOOL_DEFS                       ~:3757  merge adapted MCP tools
  DESTRUCTIVE_TOOLS (after defs)          ~:3860  union dynamic MCP set
  PLAN_MODE_TOOLS / plan_active_tools     ~:3940  union read-only MCP tools
  spill helpers (_spill_*, ~.d dirs)      ~:4030  reuse for MCP results
  ToolRegistry.__init__ / _handlers       ~:6060  register MCP dispatch
  ToolRegistry.get_system_prompt_block    ~:6053  unchanged if you merge defs
  ToolRegistry.openai_tools_spec          ~:6112  merge namespaced specs
  ToolRegistry.list_tools_str             ~:6084  show "[mcp:k8s]" tag
  ToolRegistry._confirm                   ~:6092  bridge elicitation
  ToolRegistry.execute                    ~:6134  route MCP names
  CommandContext / ChatLoop /mcp command  ~:8991  status/connect/tools/...
  ModelQuery                             ~:6429  bridge sampling/createMessage
  _build_parser                          ~:13588 add --mcp-config/--no-mcp
  main()                                 ~:14072 construct + start manager

============================================================================
5. COMMAND SURFACE TO ADD
============================================================================
  /mcp                          status of every configured server
  /mcp list                     configured servers + connection state
  /mcp connect <server>         start/reconnect
  /mcp disconnect <server>
  /mcp reload                   re-read config (hot reload)
  /mcp tools [server]           grouped listing: marker, name, [plan] tag, description
  /mcp tools --plan             audit view: only plan-eligible tools + raw hints
  /mcp tools --all              everything, with openWorld/idempotent hints
  /mcp prompts [server]         list prompts
  /mcp prompt <server> <name> [k=v ...]   inject prompt into conversation
  /mcp resources [server]       list resources / templates
  /mcp read <server> <uri>      read a resource into context
  /mcp auth <server>            run OAuth login for an HTTP server
  /mcp call <server> <tool> <json>   debug a raw call
  /mcp log <server> <level>     logging/setLevel

============================================================================
6. TESTING RECIPE
============================================================================
Unit (no server, fake transport):
  * config precedence + `mcpServers` import + env expansion
  * config errors: explicit `--mcp-config`/env path must exist (FileNotFoundError);
    unknown keys and a bad `startup` are reported, not silently ignored
  * loopback parsing: `http://127.0.0.1.evil.com/` is NOT loopback
  * name namespacing, 64-char sanitize + reverse map, collision resolution
  * JSON-Schema -> AGENTIC_TOOL_DEFS and -> OpenAI spec (must be the SAME schema)
  * tool result -> {"success","output","error"} incl. isError + spill
  * listing: markers R/+ /! /? /x, filtered tools shown with their reason, [plan]
    printed in every view, grouped per server, eligible-first ordering,
    byte-identical output for an unchanged catalog (KV-cache stability)
  * plan union: only explicitly read-only AND config-visible names
  * spill: writes under the shared spill dir AND registers the Path-ACL
    read-allow rule; cleanup removes only `mcp_result_*`
  * SSE parser (multi-line data, comments, event types, CRLF)
  * newline JSON framing (partial line, chunk boundaries, >max-frame drop +
    resync, EOF without trailing newline)
  * pagination (nextCursor loop, cursor echo)
  * cancellation (notifications/cancelled, timeout), reader-thread routing
  * three-state annotation policy; plan-mode filtering
Integration (live):
  * stdio: `npx -y kubernetes-mcp-server@latest` with a read-only TOML,
    tools/list, then call `pods_list` and `resources_list`.
  * http: server `port="8080"` -> POST http://127.0.0.1:8080/mcp.
  * resilience: kill the child mid-call -> clean error, reconnect on next use.
  * OAuth: server with require_oauth -> /mcp auth -> token reused.
Smoke (CLI): `/mcp status`, `/mcp tools kubernetes`, one agentic round-trip.

============================================================================
7. SECURITY RECIPE
============================================================================
  * stdio: never use shell=True; pass argv list; `env` is an allowlist merged
    over a minimal base (PATH, HOME) - do not leak the whole environment.
  * Config file with tokens: chmod 0600, redact in logs, use ${VAR} expansion
    and prefer env vars for secrets.
  * HTTP: default to TLS; allow plain http only for loopback; never send the
    session id or Authorization to a different origin after a redirect.
  * Treat tool descriptions as untrusted input (prompt injection): they enter
    the system prompt. Optionally prefix with "[mcp:<server>]".
  * Keep plan mode strictly readOnlyHint==True; never let an unannotated tool
    into plan mode.
  * OAuth: PKCE S256, loopback redirect, store tokens 0600, refresh before
    expiry, never log the raw token.

============================================================================
8. REFERENCES (in this repo / the Go server)
============================================================================
  MCP spec (lifecycle/transports): modelcontextprotocol.io/specification
  Go SDK constants: vendor/github.com/modelcontextprotocol/go-sdk/mcp/shared.go
  stdio framing:    vendor/.../mcp/transport.go      (newline-delimited JSON)
  HTTP transport:   vendor/.../mcp/streamable.go     (/mcp, Mcp-Session-Id)
  tool -> SDK:      pkg/mcp/tools_gosdk.go           (annotations, schemas)
  HTTP endpoints:   pkg/http/http.go                 (mcpEndpoint = "/mcp")
  OAuth discovery:  pkg/http/wellknown.go            (RFC 9728 / 8414)
  Server README:    README.md (toolset table), docs/configuration.md
"""

from __future__ import annotations

# region imports (stdlib only - ollamaquery2.py has no pip dependencies)
import abc
import atexit
import base64
import dataclasses
import hashlib
import json
import os
import queue
import re
import secrets
import shlex
import ssl
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
import webbrowser
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Optional

# endregion

# ============================================================================
# Constants
# ============================================================================

MCP_PROTOCOL_VERSIONS = (
    "2026-07-28",
    "2025-11-25",
    "2025-06-18",
    "2025-03-26",
    "2024-11-05",
)
# Simplest interoperable choice: the classic initialize lifecycle. The Go
# server negotiates down from 2026-07-28 to this when we ask for it.
# NOTE: 2026-07-28 introduces SEP-2575 per-request `_meta` metadata and
# `server/discover`, with initialize deprecated. Do NOT target it until you
# implement that path; 2025-11-25 is fully supported and stable.
MCP_PREFERRED_PROTOCOL_VERSION = "2025-11-25"

MCP_CLIENT_NAME = "ollamaquery2"
MCP_CLIENT_VERSION = "0.0.0-prototype"

# Config discovery (first hit wins).
MCP_CONFIG_ENV = "OLLAMAQUERY_MCP_CONFIG"
MCP_CONFIG_PROJECT = ".ollamaquery-mcp.json"
MCP_CONFIG_USER = os.path.expanduser("~/.ollamaquery.d/mcp.json")
OLLAMAQUERY_D_DIR = os.path.expanduser("~/.ollamaquery.d")
# Reuse the ONE shared spill dir that the cluster/fetch/write spills already use.
# A separate `mcp-spill` dir would need its own Path-ACL read-allow rule on every
# write AND its own cleanup pass; sharing the dir means the existing rule covers
# MCP results too. When integrating, add ("mcp_result_", ".json") to the
# prefixes tuple in ollamaquery2's `_cleanup_spill_files` (the atexit pass here
# is only for the standalone prototype).
MCP_SPILL_DIR = os.path.join(OLLAMAQUERY_D_DIR, "spill")
MCP_SPILL_PREFIX = "mcp_result_"
MCP_AUTH_DIR = os.path.join(OLLAMAQUERY_D_DIR, "mcp-auth")

# Max inbound frame on stdio, matching the Go SDK's DefaultMaxLineLength (16 MiB).
# Not a micro-optimization: an unbounded readline() lets a buggy/hostile server
# exhaust client memory with one gigantic line.
MCP_MAX_FRAME_BYTES = 16 * 1024 * 1024

# Result-size cap: the configured `max_result_chars` is a CEILING, but the real
# budget comes from the live context window (D7) - a fixed 100K chars is roughly
# 25K tokens, which is most of a small model's window. Chars-per-token is a
# deliberate over-estimate of English/code so the cap errs small.
MCP_RESULT_CHARS_PER_TOKEN = 4
MCP_RESULT_CONTEXT_FRACTION = 0.25  # at most a quarter of the window per result

# OpenAI native tools API function-name constraint.
OPENAI_TOOL_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")

# JSON-RPC 2.0 + MCP error codes.
ERR_PARSE = -32700
ERR_INVALID_REQUEST = -32600
ERR_METHOD_NOT_FOUND = -32601
ERR_INVALID_PARAMS = -32602
ERR_INTERNAL = -32603
ERR_RESOURCE_NOT_FOUND = -32002
ERR_UNSUPPORTED_PROTOCOL_VERSION = -32022
ERR_URL_ELICITATION_REQUIRED = -32042

# MCP method + notification names.
M_INITIALIZE = "initialize"
M_INITIALIZED = "notifications/initialized"
M_PING = "ping"
M_LIST_TOOLS = "tools/list"
M_CALL_TOOL = "tools/call"
M_LIST_RESOURCES = "resources/list"
M_LIST_RESOURCE_TEMPLATES = "resources/templates/list"
M_READ_RESOURCE = "resources/read"
M_SUBSCRIBE = "resources/subscribe"
M_UNSUBSCRIBE = "resources/unsubscribe"
M_LIST_PROMPTS = "prompts/list"
M_GET_PROMPT = "prompts/get"
M_COMPLETE = "completion/complete"
M_SET_LEVEL = "logging/setLevel"
M_LIST_ROOTS = "roots/list"
M_CREATE_MESSAGE = "sampling/createMessage"
M_ELICIT = "elicitation/create"
N_CANCELLED = "notifications/cancelled"
N_PROGRESS = "notifications/progress"
N_MESSAGE = "notifications/message"
N_TOOLS_CHANGED = "notifications/tools/list_changed"
N_RESOURCES_CHANGED = "notifications/resources/list_changed"
N_RESOURCE_UPDATED = "notifications/resources/updated"
N_PROMPTS_CHANGED = "notifications/prompts/list_changed"

# ============================================================================
# Protocol data shapes (TypedDict-ish; use plain dicts at runtime)
# ============================================================================
# JSON-RPC request:   {"jsonrpc":"2.0","id":<int|str>,"method":str,"params":obj}
# JSON-RPC response:  {"jsonrpc":"2.0","id":<id>,"result":obj} | {...,"error":{...}}
# JSON-RPC notify:    {"jsonrpc":"2.0","method":str,"params":obj}
#
# InitializeResult:
#   {"protocolVersion":str, "capabilities":{...}, "serverInfo":{...},
#    "instructions":str|None}
# Tool:
#   {"name":str, "title":str|None, "description":str|None,
#    "inputSchema":{...JSON Schema...}, "outputSchema":{...}|None,
#    "annotations":{"title":..., "readOnlyHint":bool, "destructiveHint":bool,
#                   "idempotentHint":bool, "openWorldHint":bool}|None,
#    "icons":[...]|None}
# CallToolResult:
#   {"content":[{"type":"text","text":...} | {"type":"image",...} |
#               {"type":"resource","resource":{...}} |
#               {"type":"resource_link",...}],
#    "structuredContent":obj|None, "isError":bool|None}
#
# Content block types to handle: text, image, audio, resource, resource_link.
# JSON Schema draft used by MCP inputSchema: 2020-12.

# ============================================================================
# Config: dataclasses + reference loader
# ============================================================================


@dataclass
class MCPServerConfig:
    """One configured MCP server. See `mcp.json.example`."""

    name: str
    transport: str = "stdio"                 # stdio | streamable-http | sse
    enabled: bool = True
    startup: str = "lazy"                    # lazy | eager

    # --- stdio ---
    command: list[str] = field(default_factory=list)   # argv, no shell
    env: dict[str, str] = field(default_factory=dict)  # allowlist over base env
    cwd: Optional[str] = None

    # --- http ---
    url: str = ""                            # e.g. http://127.0.0.1:8080/mcp
    headers: dict[str, str] = field(default_factory=dict)
    verify_tls: bool = True
    oauth: dict[str, Any] = field(default_factory=dict)

    # --- naming / filtering ---
    tool_prefix: Optional[str] = None        # default: server name
    expose_as: dict[str, str] = field(default_factory=dict)  # tool -> model name
    allowed_tools: list[str] = field(default_factory=list)   # empty = all
    denied_tools: list[str] = field(default_factory=list)
    read_only_only: bool = False             # drop non-readOnly tools entirely
    confirm_tools: list[str] = field(default_factory=list)   # force confirm

    # --- timeouts / limits ---
    init_timeout_seconds: float = 15.0
    call_timeout_seconds: float = 120.0
    list_timeout_seconds: float = 30.0
    max_result_chars: int = 100_000
    retries: int = 1
    backoff_seconds: float = 0.5

    # --- server-initiated features (opt-in) ---
    allow_sampling: bool = False
    allow_elicitation: bool = False
    allow_roots: bool = False
    roots: list[str] = field(default_factory=list)  # advertised via roots/list

    protocol_version: str = MCP_PREFERRED_PROTOCOL_VERSION

    # Loader bookkeeping: JSON keys that no dataclass field accepts. Kept so
    # validate_server_config() can report the typo instead of the setting simply
    # having no effect ("tool_prefx": "k8s" -> prefix silently defaults).
    _ignored_keys: list[str] = field(default_factory=list)

    def effective_prefix(self) -> str:
        return self.tool_prefix or re.sub(r"[^a-zA-Z0-9_-]", "-", self.name)


@dataclass
class MCPConfig:
    """Whole client config: an ordered map of server name -> config."""

    servers: dict[str, MCPServerConfig] = field(default_factory=dict)
    source_path: Optional[str] = None

    def enabled(self) -> list[MCPServerConfig]:
        return [s for s in self.servers.values() if s.enabled]


_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand_env(value: Any) -> Any:
    """Expand ${VAR} / ${VAR:-default} in strings, lists, and dict values.

    Secrets belong in the environment, not in the JSON file. Unset vars with
    no default expand to "" and are logged by the caller (never the value).
    """
    if isinstance(value, str):
        def repl(m: "re.Match[str]") -> str:
            name, default = m.group(1), m.group(2)
            return os.environ.get(name, default if default is not None else "")
        return _ENV_RE.sub(repl, value)
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    return value


def _coerce_server(name: str, raw: dict) -> MCPServerConfig:
    """Accept both the extended shape and the Claude/VS Code `mcpServers` shape.

    Claude shape: {"command": "npx", "args": ["-y", "..."], "env": {...},
                   "url": "...", "headers": {...}, "type": "stdio"|"http"}
    Extended shape adds: transport, startup, tool_prefix, expose_as, filters,
    timeouts, read_only_only, confirm_tools, oauth, allow_sampling, ...
    """
    data = dict(raw)
    # Compatible aliases.
    if "args" in data and isinstance(data.get("command"), str):
        data["command"] = [data["command"], *data.get("args", [])]
    data.pop("args", None)
    if "type" in data:
        if "transport" not in data:
            data["transport"] = {
                "http": "streamable-http",
                "streamable-http": "streamable-http",
                "sse": "sse",
                "stdio": "stdio",
            }.get(str(data["type"]).lower(), "stdio")
        data.pop("type", None)
    if "url" in data and "transport" not in data:
        data["transport"] = "streamable-http"

    known = {f.name for f in dataclasses.fields(MCPServerConfig)}
    kwargs = {k: v for k, v in data.items() if k in known and k != "name"}
    cfg = MCPServerConfig(name=name, **expand_env(kwargs))
    cfg._ignored_keys = sorted(k for k in data if k not in known)
    return cfg


def load_mcp_config(path: Optional[str] = None) -> MCPConfig:
    """Load config with the precedence documented in the module docstring.

    Explicit sources - the `path` argument or $OLLAMAQUERY_MCP_CONFIG - are a
    contract: if the user named a file, a missing file is an error. Only the
    implicit project/user candidates may be absent silently. (Silently falling
    back to a *different* config is how you end up running with `-` the wrong
    denied_tools list while the flag "worked".)

    REFERENCE IMPLEMENTATION - safe to copy into ollamaquery2.py as-is.
    """
    explicit: list[str] = []
    if path:
        explicit.append(path)
    env_path = os.environ.get(MCP_CONFIG_ENV)
    if env_path:
        explicit.append(env_path)
    for candidate in explicit:
        expanded = os.path.expanduser(candidate)
        if not os.path.isfile(expanded):
            raise FileNotFoundError(
                f"MCP config not found: {expanded} "
                f"(source: {'--mcp-config' if candidate == path else MCP_CONFIG_ENV})")

    candidates = explicit + [os.path.join(os.getcwd(), MCP_CONFIG_PROJECT),
                             MCP_CONFIG_USER]

    for candidate in candidates:
        candidate = os.path.expanduser(candidate)
        if not candidate or not os.path.isfile(candidate):
            continue
        try:
            with open(candidate, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
        except (OSError, ValueError) as exc:
            raise ValueError(f"invalid MCP config {candidate}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"invalid MCP config {candidate}: top level must be an object")

        raw_servers = raw.get("servers") or raw.get("mcpServers") or {}
        if not isinstance(raw_servers, dict):
            raise ValueError(f"invalid MCP config {candidate}: 'servers' must be an object")
        servers: dict[str, MCPServerConfig] = {}
        for name, entry in raw_servers.items():
            if not isinstance(entry, dict):
                raise ValueError(
                    f"invalid MCP config {candidate}: server {name!r} must be an object")
            servers[name] = _coerce_server(name, entry)
        return MCPConfig(servers=servers, source_path=candidate)

    return MCPConfig(servers={}, source_path=None)


def _is_loopback_host(host: Optional[str]) -> bool:
    """Return True for the literal loopback hosts accepted for plain http.

    Parsed-host identity, not substring matching: `http://127.0.0.1.evil.com/`
    and `http://evil.com/?x=127.0.0.1` are NOT loopback, however much they look
    like it to a naive `"127.0.0.1" in url` check.

    Args:
        host: Hostname from urlsplit (already lowercased/decoded).

    Returns:
        True only for localhost, 127.0.0.0/8, and the IPv6 loopback.
    """
    if not host:
        return False
    host = host.strip("[]").lower()
    if host in ("localhost", "::1", "0:0:0:0:0:0:0:1"):
        return True
    return bool(re.fullmatch(r"127(?:\.\d{1,3}){3}", host))


def validate_server_config(cfg: MCPServerConfig) -> list[str]:
    """Return a list of human-readable problems (empty == valid).

    Deliberately strict about the things that fail quietly in practice: a
    dropped/typo'd config key, an unrecognized `startup` (any string other than
    "eager" behaves as lazy), contradictory tool filters, and non-positive
    limits.

    Args:
        cfg: One server config, as produced by `_coerce_server`.

    Returns:
        Problem strings; the caller decides whether to fail or warn.
    """
    problems: list[str] = []
    if cfg.transport == "stdio":
        if not cfg.command:
            problems.append(f"{cfg.name}: stdio requires 'command'")
        elif not all(isinstance(a, str) and a for a in cfg.command):
            problems.append(f"{cfg.name}: 'command' must be a non-empty string list")
        elif cfg.command[0] == "sh" or cfg.command[:2] == ["bash", "-c"]:
            problems.append(f"{cfg.name}: 'command' looks like a shell wrapper; "
                            f"pass argv directly (no shell)")
        if cfg.cwd is not None and not isinstance(cfg.cwd, str):
            problems.append(f"{cfg.name}: 'cwd' must be a string or null")
    elif cfg.transport in ("streamable-http", "sse"):
        if not cfg.url:
            problems.append(f"{cfg.name}: {cfg.transport} requires 'url'")
        elif not (cfg.url.startswith("http://") or cfg.url.startswith("https://")):
            problems.append(f"{cfg.name}: 'url' must be http(s)")
        else:
            try:
                parts = urllib.parse.urlsplit(cfg.url)
                hostname = parts.hostname
            except ValueError as exc:
                parts, hostname = None, None
                problems.append(f"{cfg.name}: unparsable url {cfg.url!r} ({exc})")
            if parts is not None and parts.scheme == "http" \
                    and not _is_loopback_host(hostname):
                problems.append(
                    f"{cfg.name}: plain http to non-loopback host {hostname!r} - "
                    f"use https, or a 127.x/localhost address")
        if not isinstance(cfg.headers, dict):
            problems.append(f"{cfg.name}: 'headers' must be an object")
        if cfg.transport == "sse":
            problems.append(f"{cfg.name}: legacy 'sse' transport is not implemented "
                            f"(use 'streamable-http')")
    else:
        problems.append(f"{cfg.name}: unknown transport {cfg.transport!r}")

    if cfg.startup not in ("lazy", "eager"):
        problems.append(f"{cfg.name}: 'startup' must be 'lazy' or 'eager' "
                        f"(got {cfg.startup!r})")
    if cfg.protocol_version not in MCP_PROTOCOL_VERSIONS:
        problems.append(f"{cfg.name}: unknown protocol_version {cfg.protocol_version!r}")
    if cfg._ignored_keys:
        problems.append(f"{cfg.name}: unknown config key(s) ignored: "
                        f"{', '.join(cfg._ignored_keys)}")
    if not isinstance(cfg.env, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in cfg.env.items()):
        problems.append(f"{cfg.name}: 'env' must map strings to strings")

    overlap = set(cfg.allowed_tools) & set(cfg.denied_tools)
    if overlap:
        problems.append(f"{cfg.name}: tool(s) in both allowed_tools and denied_tools "
                        f"(deny wins, so the allow entries are dead): "
                        f"{', '.join(sorted(overlap))}")
    denied_confirm = set(cfg.denied_tools) & set(cfg.confirm_tools)
    if denied_confirm:
        problems.append(f"{cfg.name}: tool(s) in both denied_tools and confirm_tools "
                        f"(denied tools never reach the confirm gate): "
                        f"{', '.join(sorted(denied_confirm))}")
    unknown_confirm = set(cfg.confirm_tools) - set(cfg.allowed_tools) \
        if cfg.allowed_tools else set()
    if unknown_confirm:
        problems.append(f"{cfg.name}: confirm_tools outside allowed_tools: "
                        f"{', '.join(sorted(unknown_confirm))}")

    for label in ("init_timeout_seconds", "call_timeout_seconds",
                  "list_timeout_seconds", "backoff_seconds"):
        value = getattr(cfg, label)
        if not isinstance(value, (int, float)) or value <= 0:
            problems.append(f"{cfg.name}: '{label}' must be > 0 (got {value!r})")
    if not isinstance(cfg.max_result_chars, int) or cfg.max_result_chars <= 0:
        problems.append(f"{cfg.name}: 'max_result_chars' must be a positive int "
                        f"(got {cfg.max_result_chars!r})")
    if not isinstance(cfg.retries, int) or cfg.retries < 0:
        problems.append(f"{cfg.name}: 'retries' must be >= 0 (got {cfg.retries!r})")
    return problems


# ============================================================================
# Wire helpers (reference implementations)
# ============================================================================


def jsonrpc_request(method: str, params: Optional[dict], req_id: int) -> dict:
    msg: dict = {"jsonrpc": "2.0", "id": req_id, "method": method}
    if params is not None:
        msg["params"] = params
    return msg


def jsonrpc_notification(method: str, params: Optional[dict] = None) -> dict:
    msg: dict = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        msg["params"] = params
    return msg


def jsonrpc_error(req_id: Any, code: int, message: str, data: Any = None) -> dict:
    err: dict = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": req_id, "error": err}


def jsonrpc_result(req_id: Any, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def encode_frame(message: dict) -> bytes:
    """stdio framing: compact JSON + exactly one trailing newline."""
    return (json.dumps(message, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def _decode_frame(line: bytes) -> Optional[dict]:
    """Decode one newline-delimited frame; None when it is not a JSON object.

    Malformed input (a server banner on stdout, a truncated tail) is dropped
    rather than raised: one bad line must never kill the reader thread.
    """
    text = line.strip()
    if not text:
        return None
    try:
        obj = json.loads(text.decode("utf-8", errors="replace"))
    except ValueError:
        return None
    return obj if isinstance(obj, dict) else None


def iter_json_lines(stream, max_frame_bytes: int = MCP_MAX_FRAME_BYTES
                    ) -> Iterator[dict]:
    """Yield JSON objects from a newline-delimited stream (stdio transport).

    Reads in chunks so the frame-size guard is actually enforceable: `for line
    in stream` (and `readline()`) hand you the whole frame first, so a hostile
    or buggy server can exhaust memory with one enormous line before any check
    runs. Oversize frames are dropped and the reader resyncs on the next
    newline.

    Args:
        stream: Text or binary stream (the child's stdout pipe).
        max_frame_bytes: Frame ceiling; defaults to the SDK's 16 MiB.

    Yields:
        Parsed JSON objects; never raises on malformed frames.
    """
    buf = bytearray()
    oversize = False
    while True:
        chunk = stream.read(65536)
        if not chunk:
            break
        if isinstance(chunk, str):
            chunk = chunk.encode("utf-8")
        buf.extend(chunk)
        while True:
            newline = buf.find(b"\n")
            if newline < 0:
                break
            line = bytes(buf[:newline])
            del buf[:newline + 1]
            if oversize:
                # Tail of a frame we already abandoned; drop and resync.
                oversize = False
                continue
            if len(line) > max_frame_bytes:
                # A complete frame can exceed the limit too when it arrived
                # inside one chunk; the ceiling applies to delivered frames, not
                # just to how much we are willing to buffer.
                continue
            obj = _decode_frame(line)
            if obj is not None:
                yield obj
        if len(buf) > max_frame_bytes:
            oversize = True
            buf.clear()
    if buf and not oversize:
        # Tolerate a final frame without a trailing newline.
        obj = _decode_frame(bytes(buf))
        if obj is not None:
            yield obj


def parse_sse_stream(lines: Iterable[str]) -> Iterator[dict]:
    """Parse `text/event-stream` lines into {"event","data","id"} dicts.

    REFERENCE IMPLEMENTATION. SSE fields: `event:`, `data:` (repeatable),
    `id:`, `retry:`, `:` comment. Events are separated by a blank line.
    Each MCP event's joined `data` is one JSON-RPC object.
    """
    event, data_lines, last_id = None, [], None
    for raw in lines:
        line = raw.rstrip("\r\n")
        if line == "":
            if data_lines:
                yield {
                    "event": event or "message",
                    "data": "\n".join(data_lines),
                    "id": last_id,
                }
            event, data_lines = None, []
            continue
        if line.startswith(":"):
            continue
        field_name, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if field_name == "event":
            event = value
        elif field_name == "data":
            data_lines.append(value)
        elif field_name == "id":
            last_id = value
        # `retry` is intentionally ignored.


def parse_sse_json(lines: Iterable[str]) -> Iterator[dict]:
    for evt in parse_sse_stream(lines):
        try:
            obj = json.loads(evt["data"])
        except ValueError:
            continue
        if isinstance(obj, dict):
            yield obj


# ============================================================================
# Tool adaptation (MCP tool -> ollamaquery2 tool shapes)
# ============================================================================


def sanitize_tool_name(name: str) -> str:
    """Force a name into ^[a-zA-Z0-9_-]{1,64}$ (OpenAI native tools API)."""
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    if len(cleaned) <= 64:
        return cleaned or "tool"
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
    return f"{cleaned[:55]}_{digest}"


def namespaced_tool_name(server: MCPServerConfig, tool_name: str) -> str:
    """Resolve the model-visible name and keep a reverse map on the manager.

    Priority: explicit expose_as > <prefix>__<tool>.
    Collisions are resolved by the manager (it owns the reverse map).
    """
    if tool_name in server.expose_as:
        return sanitize_tool_name(server.expose_as[tool_name])
    return sanitize_tool_name(f"{server.effective_prefix()}__{tool_name}")


def _annotation(tool: dict, key: str) -> Optional[bool]:
    """Read one boolean hint from a tool's `annotations`, three-state.

    `None` means the server did not declare it — which is NOT the same as false.
    The spec states defaults in prose (readOnlyHint false, destructiveHint true,
    idempotentHint false, openWorldHint true), so callers that care about the
    effective value must apply those defaults; callers that care about what the
    server actually said must keep the `None`.

    Args:
        tool: Raw MCP Tool dict.
        key: Annotation name, e.g. "readOnlyHint".

    Returns:
        The declared boolean, or None when absent.
    """
    ann = tool.get("annotations") or {}
    if not isinstance(ann, dict) or key not in ann:
        return None
    return bool(ann[key])


def is_read_only(tool: dict) -> Optional[bool]:
    """Three-state read-only: True / False / None(undeclared).

    Only `True` may be treated as read-only; the spec default when undeclared is
    false, so `None` must never be read as "safe".
    """
    return _annotation(tool, "readOnlyHint")


def is_destructive(tool: dict) -> Optional[bool]:
    """Three-state destructive: True / False / None(undeclared).

    The spec default when undeclared is TRUE, so `None` means "assume destructive".
    """
    return _annotation(tool, "destructiveHint")


def mcp_tool_to_agentic_def(server: MCPServerConfig, tool: dict,
                            model_name: str) -> dict:
    """Convert an MCP Tool into the AGENTIC_TOOL_DEFS shape.

    AGENTIC_TOOL_DEFS values are {"description": str, "parameters": {JSON Schema}}.
    MCP's inputSchema IS JSON Schema, so it maps almost verbatim. `parameters` is
    the *normalized* schema because recipe D merges at `_plan_allowed_tools`,
    which feeds BOTH the inline renderer and `openai_tools_spec()` - one shape
    for both surfaces, so they cannot drift apart. Normalisation is additive and
    preserves `oneOf`/`$ref`, so the native request loses nothing; the untouched
    schema stays at `_mcp.inputSchemaRaw` for diagnostics.
    """
    description = tool.get("description") or tool.get("title") or model_name
    prefix = f"[mcp:{server.name}] "
    raw_schema = tool.get("inputSchema") or {"type": "object", "properties": {}}
    return {
        "description": prefix + description,
        # Safe for the inline renderer; nested schema info is preserved.
        "parameters": normalize_schema_for_inline(raw_schema),
        # Extra, non-AGENTIC keys are ignored by the existing renderer but
        # available to the manager / native tools spec:
        "_mcp": {
            "server": server.name,
            "tool": tool.get("name"),
            "readOnlyHint": is_read_only(tool),
            "destructiveHint": is_destructive(tool),
            "inputSchemaRaw": raw_schema,
            "outputSchema": tool.get("outputSchema"),
        },
    }


def mcp_tool_to_openai_spec(server: MCPServerConfig, tool: dict,
                            model_name: str) -> dict:
    """OpenAI native tools entry: {"type":"function","function":{...}}.

    Uses the same `normalize_schema_for_inline` output as the agentic def, so the
    native and inline surfaces describe one schema. (Pre-fix this used the raw
    schema, which disagreed with the path recipe D actually wires up - the raw
    variant was unreachable dead code.)
    """
    return {
        "type": "function",
        "function": {
            "name": model_name,
            "description": (f"[mcp:{server.name}] "
                            + (tool.get("description") or tool.get("title") or model_name)),
            "parameters": normalize_schema_for_inline(
                tool.get("inputSchema") or {"type": "object", "properties": {}}),
        },
    }


# -- /mcp tools listing -------------------------------------------------------
#
# Pure formatting over the cached catalog: no server is contacted, so the same
# function serves `mcp tools`, the plan-mode audit view, and the manager's
# `plan_read_only_names()`. Three presentation decisions, agreed with the
# operator:
#   * filtered-out tools are SHOWN with their reason - silent absence is how
#     people stop trusting a filter;
#   * the [plan] tag is ALWAYS printed - plan eligibility is a property of the
#     config plus the annotations, not of the current session;
#   * output is grouped per server, sorted by model name, so rebuilding the
#     system prompt from an unchanged catalog stays byte-identical (the prompt
#     is the KV-cache prefix; volatile ordering would churn it).

MARKER_READ_ONLY = "R"
MARKER_ADDITIVE = "+"
MARKER_DESTRUCTIVE = "!"
MARKER_UNDECLARED = "?"
MARKER_FILTERED = "x"

LISTING_VIEWS = ("default", "plan", "all")


@dataclass
class MCPToolEntry:
    """One MCP tool plus the policy verdicts derived from its annotations."""

    server: str
    mcp_name: str
    model_name: str
    description: str
    read_only: Optional[bool]
    destructive: Optional[bool]
    idempotent: Optional[bool]
    open_world: Optional[bool]
    filtered_reason: Optional[str] = None  # set => never exposed, in any mode
    confirm: bool = False                  # config forces a confirmation gate

    @property
    def marker(self) -> str:
        """One-character state marker used by the listing."""
        if self.filtered_reason:
            return MARKER_FILTERED
        if self.read_only is True and self.destructive is not True:
            return MARKER_READ_ONLY
        if self.destructive is False:
            return MARKER_ADDITIVE
        if self.destructive is True:
            return MARKER_DESTRUCTIVE
        return MARKER_UNDECLARED

    @property
    def plan_eligible(self) -> bool:
        """True only for a config-visible tool declared read-only.

        Note the asymmetry with `marker`: a filtered tool is `x` even when it is
        annotated read-only, because config filtering removes it from every mode.
        """
        return (self.filtered_reason is None and self.read_only is True
                and self.destructive is not True)


def _config_filter_reason(cfg: MCPServerConfig, tool: dict) -> Optional[str]:
    """Return why the tool is never exposed, or None when config allows it.

    Order matters: a tool in both `allowed_tools` and `denied_tools` is denied
    (deny wins), and `read_only_only` applies last since it is the broadest.

    Args:
        cfg: The owning server config.
        tool: Raw MCP Tool dict.

    Returns:
        Short reason tag for the listing, or None.
    """
    name = tool.get("name") or ""
    if not cfg.enabled:
        return "server disabled"
    if name in cfg.denied_tools:
        return "denied_tools"
    if cfg.allowed_tools and name not in cfg.allowed_tools:
        return "not in allowed_tools"
    if cfg.read_only_only and is_read_only(tool) is not True:
        return "read_only_only"
    return None


def _sort_key(entry: MCPToolEntry) -> tuple:
    """Order rows for reading: plan-eligible, then exposed, then hidden.

    Still a pure function of the catalog (deterministic, so the prompt/listing
    order cannot churn), but the top of each group is the answer to the question
    the listing exists for: what can plan mode actually call?
    """
    if entry.plan_eligible:
        rank = 0
    elif entry.filtered_reason is None:
        rank = 1
    else:
        rank = 2
    return (rank, entry.model_name)


def classify_tools(cfg: MCPServerConfig, tools: Iterable[dict]) -> list[MCPToolEntry]:
    """Classify one server's `tools/list` result into listing entries.

    Config filtering and annotation classification are kept separate on purpose:
    a config-filtered tool is absent in *every* mode, while a `!` tool is usable
    in full agentic mode and merely not plan-eligible. Collapsing the two into one
    "why not" column is what makes such listings misleading.

    Args:
        cfg: The owning server config.
        tools: Raw MCP Tool dicts from `tools/list`.

    Returns:
        Entries sorted for display (plan-eligible first, then by model name).
    """
    entries: list[MCPToolEntry] = []
    for tool in tools:
        mcp_name = tool.get("name") or ""
        if not mcp_name:
            continue
        entries.append(MCPToolEntry(
            server=cfg.name,
            mcp_name=mcp_name,
            model_name=namespaced_tool_name(cfg, mcp_name),
            description=(tool.get("description") or tool.get("title") or "").strip(),
            read_only=is_read_only(tool),
            destructive=is_destructive(tool),
            idempotent=_annotation(tool, "idempotentHint"),
            open_world=_annotation(tool, "openWorldHint"),
            filtered_reason=_config_filter_reason(cfg, tool),
            confirm=mcp_name in cfg.confirm_tools,
        ))
    entries.sort(key=_sort_key)
    return entries


def plan_tool_names(catalogs: dict[str, list[MCPToolEntry]]) -> set[str]:
    """Model-visible names that `/agentic plan` may expose.

    This is the whole plan-mode MCP policy: union of config-visible tools whose
    `readOnlyHint` is explicitly true. Feed the result into `plan_active_tools()`.

    Args:
        catalogs: server name -> classified entries.

    Returns:
        Set of model-visible tool names.
    """
    return {e.model_name for entries in catalogs.values() for e in entries
            if e.plan_eligible}


def _clip(text: str, width: int) -> str:
    """Collapse whitespace and truncate to `width` chars with an ellipsis."""
    text = " ".join(text.split())
    return text if len(text) <= width else text[:max(0, width - 1)] + "\u2026"


def _bool_label(value: Optional[bool]) -> str:
    """Render a tri-state hint as true / false / unset."""
    return {True: "true", False: "false", None: "unset"}[value]


def _entry_tag(entry: MCPToolEntry, view: str) -> str:
    """Build the tag column: [plan], [confirm], filter reason, raw flags."""
    if entry.filtered_reason:
        return f"[{entry.filtered_reason}]"
    tags = []
    if entry.plan_eligible:
        tags.append("[plan]")
    if entry.confirm:
        tags.append("[confirm]")
    if view in ("plan", "all"):
        flags = f"ro={_bool_label(entry.read_only)} ds={_bool_label(entry.destructive)}"
        if view == "all":
            flags += f" ow={_bool_label(entry.open_world)} idem={_bool_label(entry.idempotent)}"
        tags.append(flags)
    return " ".join(tags)


def format_tool_listing(catalogs: dict[str, list[MCPToolEntry]],
                        view: str = "default",
                        connected: Optional[dict[str, bool]] = None,
                        width: int = 78) -> list[str]:
    """Render the `/mcp tools` listing, grouped per server.

    Args:
        catalogs: server name -> classified entries, in config order.
        view: "default" (everything, descriptions), "plan" (plan-eligible only,
            raw hints instead of descriptions), or "all" (everything with hints).
        connected: server name -> connection state, for the group header.
        width: separator width.

    Returns:
        Lines to print; the caller owns the printing and colouring.
    """
    if view not in LISTING_VIEWS:
        raise ValueError(f"unknown view {view!r} (expected one of {LISTING_VIEWS})")
    connected = connected or {}
    lines: list[str] = []
    server_count = len(catalogs)
    lines.append(f"MCP tools ({server_count} server{'s' if server_count != 1 else ''} configured)")
    lines.append("\u2500" * width)

    total_plan = total_hidden = total_listed = 0
    for server, entries in catalogs.items():
        visible = [e for e in entries if e.filtered_reason is None]
        plan = [e for e in visible if e.plan_eligible]
        hidden = len(entries) - len(visible)
        total_plan += len(plan)
        total_hidden += hidden
        total_listed += len(entries)

        state = connected.get(server)
        state_label = "connected" if state else ("not connected" if state is False else "unknown")
        lines.append("")
        lines.append(f"{server} \u2014 {state_label}, {len(entries)} listed \u00b7 "
                     f"{len(plan)} plan-eligible \u00b7 {hidden} hidden")
        if state is False:
            lines.append(f"  (connect it first: /mcp connect {server})")
            continue

        rows = plan if view == "plan" else entries
        if not rows:
            lines.append("  (no tools)")
            continue
        rows = sorted(rows, key=_sort_key)
        name_width = max(20, min(max(len(e.model_name) for e in rows), 40))
        for entry in rows:
            tag = _entry_tag(entry, view)
            detail = "" if view == "plan" else "  " + _clip(entry.description, 50)
            lines.append(f"  {entry.marker}  {entry.model_name:<{name_width}}  "
                         f"{tag:<15}{detail}".rstrip())

    lines.append("")
    lines.append("\u2500" * width)
    lines.append("  R read-only (plan-eligible)   + additive write   ! destructive   "
                 "? undeclared")
    lines.append("  x never exposed (reason from mcp.json)   [plan] visible in /agentic plan   "
                 "[confirm] asks every call")
    lines.append("  ? / destructive are the spec defaults when a server declares nothing: "
                 "undeclared is never plan-eligible")
    lines.append(f"  {total_listed} listed \u00b7 {total_plan} plan-eligible \u00b7 "
                 f"{total_hidden} hidden across {server_count} server"
                 f"{'s' if server_count != 1 else ''}")
    return lines


def format_server_summary(cfg: MCPServerConfig, entries: list[MCPToolEntry],
                          connected: bool) -> str:
    """One-line server state for `/mcp` (never includes per-tool detail).

    Args:
        cfg: Server config.
        entries: Classified entries (empty when never connected).
        connected: Whether a session is live.

    Returns:
        A single summary line.
    """
    plan = sum(1 for e in entries if e.plan_eligible)
    hidden = sum(1 for e in entries if e.filtered_reason)
    state = "connected" if connected else "not connected"
    return (f"{cfg.name:<14} {cfg.transport:<15} {state:<14} "
            f"{len(entries)} tools \u00b7 {plan} plan \u00b7 {hidden} hidden")


def normalize_schema_for_inline(schema: Optional[dict]) -> dict:
    """Make an MCP inputSchema safe for the existing inline prompt renderer.

    `ToolRegistry.get_system_prompt_block` does `pdef['type']` and
    `pdef['description']` (hard subscripts), and assumes a flat `properties`
    object. This pass fills those keys without changing the schema semantics
    used by the native tools API (call it only for the inline copy).

    REFERENCE IMPLEMENTATION.
    """
    schema = dict(schema or {})
    schema.setdefault("type", "object")
    props = schema.get("properties")
    if not isinstance(props, dict):
        props = {}
    fixed: dict[str, dict] = {}
    for name, prop in props.items():
        if not isinstance(prop, dict):
            prop = {}
        prop = dict(prop)
        if "type" not in prop:
            if "enum" in prop:
                prop["type"] = "string"
            elif "$ref" in prop or "oneOf" in prop or "anyOf" in prop \
                    or "allOf" in prop:
                prop["type"] = "object"
            else:
                prop["type"] = "string"
        prop.setdefault("description", "")
        fixed[name] = prop
    schema["properties"] = fixed
    if not isinstance(schema.get("required"), list):
        schema["required"] = []
    return schema


# -- content-block rendering --------------------------------------------------


def _content_block_to_text(block: dict) -> str:
    """Flatten one MCP content block to text. Handle every spec type."""
    btype = block.get("type")
    if btype == "text":
        return block.get("text", "")
    if btype == "image":
        # Do not inline base64 into context; return a placeholder + spill hint.
        return f"[image {block.get('mimeType', '?')}, {len(block.get('data', ''))} b64 chars]"
    if btype == "audio":
        return f"[audio {block.get('mimeType', '?')}]"
    if btype == "resource_link":
        return f"[resource_link {block.get('uri')} ({block.get('name', '')})]"
    if btype == "resource":
        res = block.get("resource") or {}
        if isinstance(res.get("text"), str):
            return res["text"]
        if res.get("blob"):
            return f"[resource blob {res.get('uri')} {res.get('mimeType', '')}]"
        return f"[resource {res.get('uri')}]"
    return json.dumps(block, ensure_ascii=False)


def call_result_to_tool_output(result: dict, server_name: str, tool_name: str,
                               max_chars: int,
                               spill: Optional[Callable[[str, str, str], str]] = None
                               ) -> dict:
    """Map CallToolResult -> ollamaquery2's {"success","output","error"}.

    REFERENCE IMPLEMENTATION. Pass `spill=make_spill_helper(ctx)` to get the
    standard behaviour for oversized results: the text goes to the shared spill
    dir and the observation becomes a path + preview, pageable with `read_file`.
    Without a spill callback the text is returned in full (the caller's
    observation cap then truncates it).
    """
    parts = [_content_block_to_text(b) for b in (result.get("content") or [])]
    text = "\n".join(p for p in parts if p)
    if result.get("structuredContent") is not None:
        text += "\n```json\n" + json.dumps(result["structuredContent"], indent=2) + "\n```"
    is_error = bool(result.get("isError"))
    if len(text) > max_chars and spill is not None:
        path = spill(text, server_name, tool_name)
        preview = text[:400].replace("\n", " ")
        text = (f"[mcp:{server_name}] Result is large ({len(text)} chars) - the output is "
                f"NOT included here to avoid duplicating it in context. Read it with "
                f"read_file(file_path=\"{path}\"); read_file pages large files "
                f"automatically via the `next` offset (up to 2000 lines / 100KB per page). "
                f"Preview: {preview}...")
    return {
        "success": not is_error,
        "output": text,
        "error": None if not is_error else f"[mcp:{server_name}/{tool_name}] tool returned isError",
    }


# -- result spill (shares the ~/.ollamaquery.d/spill dir) ---------------------

_MCP_SPILL_CLEANUP_REGISTERED = False


def mcp_spill_path(server_name: str, tool_name: str) -> str:
    """Deterministic spill filename for one server/tool pair.

    Scoped by server+tool (plus a short hash of the raw pair) so a truncated
    sanitized stem cannot collide two different tools onto one file.

    Args:
        server_name: Configured server name.
        tool_name: Original MCP tool name (not the model-visible one).

    Returns:
        Absolute path under MCP_SPILL_DIR.
    """
    stem = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{server_name}_{tool_name}")[:100]
    digest = hashlib.sha1(f"{server_name}/{tool_name}".encode("utf-8")).hexdigest()[:8]
    return os.path.join(MCP_SPILL_DIR, f"{MCP_SPILL_PREFIX}{stem}_{digest}.json")


def _cleanup_mcp_spill_files() -> None:
    """Remove MCP spill files at exit (standalone-prototype cleanup pass)."""
    try:
        if not os.path.isdir(MCP_SPILL_DIR):
            return
        for fname in os.listdir(MCP_SPILL_DIR):
            if fname.startswith(MCP_SPILL_PREFIX) and fname.endswith(".json"):
                try:
                    os.unlink(os.path.join(MCP_SPILL_DIR, fname))
                except OSError:
                    pass
    except OSError:
        pass


def spill_mcp_result(ctx: object, text: str, server_name: str,
                     tool_name: str) -> str:
    """Write an oversized MCP result to the spill dir and return its path.

    Registers a session Path-ACL `read allow` rule for the spill dir *before*
    handing the path to the model. That rule is load-bearing, not politeness:
    the dir is a home dotfile outside the CWD, and strict `/agentic plan`
    confines every path to the CWD (implicit `$HOME` reads and every `ask` rule
    become hard denies). Without the rule the model is given a pointer it cannot
    follow - `read_file` returns a Path-ACL denial - which is exactly the failure
    the cluster/fetch spill helpers already guard against.

    Args:
        ctx: CommandContext (anything with an optional `path_acl`).
        text: Full result text.
        server_name: Configured server name.
        tool_name: Original MCP tool name.

    Returns:
        Absolute path the model can `read_file`.
    """
    global _MCP_SPILL_CLEANUP_REGISTERED
    os.makedirs(MCP_SPILL_DIR, exist_ok=True)
    acl = getattr(ctx, "path_acl", None)
    if acl is not None:
        acl.add("read", "allow", MCP_SPILL_DIR, source="session")
    if not _MCP_SPILL_CLEANUP_REGISTERED:
        atexit.register(_cleanup_mcp_spill_files)
        _MCP_SPILL_CLEANUP_REGISTERED = True
    path = mcp_spill_path(server_name, tool_name)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


def make_spill_helper(ctx: object) -> Callable[[str, str, str], str]:
    """Bind `spill_mcp_result` to `ctx`, matching the 3-arg callback shape.

    Args:
        ctx: CommandContext holding the session PathAcl.

    Returns:
        Callable(text, server_name, tool_name) -> path.
    """
    def spill(text: str, server_name: str, tool_name: str) -> str:
        return spill_mcp_result(ctx, text, server_name, tool_name)
    return spill


def resolve_max_result_chars(ctx: object, configured: int) -> int:
    """Effective result cap: the configured ceiling, lowered to fit the live window.

    D7 requires the cap to follow the context window rather than a constant, so
    `max_result_chars` is a *ceiling* and the window budget wins when it is
    smaller. `context_window_size` is 0 when unknown (the real CommandContext
    sentinel), in which case the ceiling stands unmodified.

    Args:
        ctx: CommandContext (reads `context_window_size`).
        configured: Per-server `max_result_chars` ceiling.

    Returns:
        Char budget, floored at 2000 so a tiny window cannot make every result
        spill to a file the model must then page through.
    """
    configured = int(configured)
    window = int(getattr(ctx, "context_window_size", 0) or 0)
    if window <= 0:
        return configured
    budget = int(window * MCP_RESULT_CONTEXT_FRACTION) * MCP_RESULT_CHARS_PER_TOKEN
    return max(2000, min(configured, budget))


# ============================================================================
# Transports
# ============================================================================


class MCPTransport(abc.ABC):
    """Byte/JSON transport. One reader thread per instance, one write lock."""

    @abc.abstractmethod
    def start(self, timeout: float) -> None:
        """Establish the pipe/socket. Raise on failure."""  # noqa: D401

    @abc.abstractmethod
    def send(self, message: dict) -> None:
        """Serialize + write one JSON-RPC message (thread-safe)."""

    @abc.abstractmethod
    def receive(self, timeout: float) -> Optional[dict]:
        """Block up to `timeout` for one inbound message; None on timeout."""

    @abc.abstractmethod
    def close(self) -> None:
        """Terminate the child / close the socket. Idempotent."""

    @abc.abstractmethod
    def is_alive(self) -> bool:
        """False once the child exited / the socket dropped."""

    @property
    def endpoint(self) -> str:
        """Human-readable endpoint for logs/status."""
        return "?"


class StdioTransport(MCPTransport):
    """Local server launched as a subprocess (the usual case).

    RECIPE
    ------
    start():
      * subprocess.Popen(argv, stdin=PIPE, stdout=PIPE, stderr=PIPE,
        shell=False, bufsize=0, env=<base+allowlist>, cwd=cfg.cwd,
        start_new_session=True)   # own process group -> clean kill
      * one thread: for line in stdout -> json.loads -> inbound queue
      * one thread: for line in stderr -> debug log (NEVER protocol)
    send(): write encode_frame(msg); guard with a lock.
    receive(): queue.get(timeout=...).
    close(): stdout.close(); child.terminate(); wait(2s); kill(); reap.
             Join reader threads. Kill the process group on POSIX so children
             (omc/oc spawned by a server, if any) don't linger.
    is_alive(): child.poll() is None.

    Notes:
      * No shell=True, ever. argv comes from config.
      * stdout is reserved for protocol; if the server prints a banner to
        stdout the first frame will be malformed -> that is a server bug;
        drop non-JSON lines in the reader and log.
      * 16 MiB max frame (SDK DefaultMaxLineLength); stream-read rather than
        readline() if you want to enforce it strictly.
    """

    def __init__(self, cfg: MCPServerConfig) -> None:
        self._cfg = cfg
        self._proc: Optional[subprocess.Popen] = None
        self._inbound: "queue.Queue[dict]" = queue.Queue()
        self._write_lock = threading.Lock()
        self._threads: list[threading.Thread] = []

    def start(self, timeout: float) -> None:
        raise NotImplementedError("STUB -> implement in ollamaquery2.py")

    def send(self, message: dict) -> None:
        raise NotImplementedError("STUB -> implement in ollamaquery2.py")

    def receive(self, timeout: float) -> Optional[dict]:
        raise NotImplementedError("STUB -> implement in ollamaquery2.py")

    def close(self) -> None:
        raise NotImplementedError("STUB -> implement in ollamaquery2.py")

    def is_alive(self) -> bool:
        raise NotImplementedError("STUB -> implement in ollamaquery2.py")

    @property
    def endpoint(self) -> str:
        return "stdio:" + " ".join(self._cfg.command)


class StreamableHTTPTransport(MCPTransport):
    """Remote server over Streamable HTTP (POST /mcp + SSE responses).

    RECIPE
    ------
    start():
      * Store cfg.url (e.g. http://127.0.0.1:8080/mcp), headers, TLS context.
      * Lazily open a standalone GET stream only if you need server-initiated
        messages while idle; many clients skip it until needed.
      * If the server runs `stateless = true`, it returns no Mcp-Session-Id:
        simply omit the header and never assume notifications arrive.
    send(msg):
      * POST cfg.url
        Content-Type: application/json
        Accept: application/json, text/event-stream
        [Mcp-Session-Id: <sid>]  [MCP-Protocol-Version: <ver>]
        [Authorization: Bearer <token>]
      * body = json.dumps(msg), using http.client/urllib (stdlib).
      * Response handling:
          202 Accepted            -> notification ack, no body
          200 + application/json  -> one JSON-RPC object
          200 + text/event-stream -> iterate SSE, feed each object to inbound
          401                     -> raise MCPAuthRequired(challenge)
          404                     -> session expired -> MCPNeedsReinitialize
      * Capture Mcp-Session-Id from the initialize *response* and reuse it.
    receive(timeout): drain the inbound queue (filled by the POST reader and,
      if used, the standalone GET reader thread).
    close(): DELETE cfg.url with Mcp-Session-Id (best-effort); stop threads.
    is_alive(): last POST didn't fail with a connection error.

    OAuth recipe (only if a 401 challenge appears):
      1. Parse WWW-Authenticate: Bearer resource_metadata="<url>".
      2. GET resource_metadata -> authorization_servers[0].
      3. GET <authz>/.well-known/oauth-authorization-server (RFC 8414);
         fall back to openid-configuration.
      4. Dynamic client registration if registration_endpoint is present
         (unless disabled by the server).
      5. Authorization Code + PKCE(S256), loopback redirect, open browser.
      6. Exchange code -> access_token (+refresh_token, expires_in).
      7. Store 0600 under MCP_AUTH_DIR/<server>.json; refresh on expiry.
    """

    def __init__(self, cfg: MCPServerConfig) -> None:
        self._cfg = cfg
        self._session_id: Optional[str] = None
        self._protocol_version: str = cfg.protocol_version
        self._inbound: "queue.Queue[dict]" = queue.Queue()
        self._threads: list[threading.Thread] = []

    def start(self, timeout: float) -> None:
        raise NotImplementedError("STUB -> implement in ollamaquery2.py")

    def send(self, message: dict) -> None:
        raise NotImplementedError("STUB -> implement in ollamaquery2.py")

    def receive(self, timeout: float) -> Optional[dict]:
        raise NotImplementedError("STUB -> implement in ollamaquery2.py")

    def close(self) -> None:
        raise NotImplementedError("STUB -> implement in ollamaquery2.py")

    def is_alive(self) -> bool:
        raise NotImplementedError("STUB -> implement in ollamaquery2.py")

    @property
    def endpoint(self) -> str:
        return self._cfg.url


class InMemoryTransport(MCPTransport):
    """Test double: scripted replies + injected notifications.

    Use it to unit-test the client, manager, tool adaptation, and the
    integration hooks without a real server or process.

    Script semantics: each `send()` with an id consumes script entries. Entries
    that carry a `method` are *server-initiated* (requests/notifications) and are
    delivered as-is, keeping their own id, before the reply to the client
    request. That is what lets tests exercise the D3 multiplexing path - a server
    `ping` or `roots/list` arriving while a `tools/call` is outstanding - instead
    of the double stamping the client's id onto it.
    """

    def __init__(self, script: Optional[list[dict]] = None) -> None:
        self.script = list(script or [])
        self.sent: list[dict] = []
        self._inbound: "queue.Queue[dict]" = queue.Queue()
        self._alive = False

    def enqueue(self, message: dict) -> None:
        self._inbound.put(message)

    def start(self, timeout: float) -> None:
        self._alive = True

    def send(self, message: dict) -> None:
        self.sent.append(message)
        req_id = message.get("id")
        if req_id is None:
            return
        # Server-initiated traffic is delivered verbatim (own id preserved).
        while self.script and "method" in self.script[0]:
            self._inbound.put(dict(self.script.pop(0)))
        if self.script:
            reply = dict(self.script.pop(0))
            # A scripted reply may already carry an id (e.g. an error for a
            # specific request); only fill it in when absent.
            reply.setdefault("id", req_id)
            self._inbound.put(reply)

    def receive(self, timeout: float) -> Optional[dict]:
        try:
            return self._inbound.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self) -> None:
        self._alive = False

    def is_alive(self) -> bool:
        return self._alive


# ============================================================================
# JSON-RPC connection (multiplexer)
# ============================================================================


class MCPConnection:
    """Correlate JSON-RPC requests/responses over one transport.

    RESPONSIBILITIES
    ----------------
      * one reader thread: transport.receive() -> route
            response with id in `_pending`     -> resolve its Event
            request  (id + method)             -> dispatch handler, send reply
            notification (method, no id)       -> dispatch notification handler
      * request(method, params, timeout): assign monotonic id, register
        pending, send, wait on Event, raise MCPTimeout on expiry.
      * notify(method, params): fire-and-forget.
      * cancel(request_id, reason): send notifications/cancelled; drop pending.
      * close(): stop reader, fail all pending with MCPConnectionClosed.

    IMPORTANT
    ---------
      * Never hold the write lock while waiting for a response.
      * On timeout, send notifications/cancelled and leave a tombstone so a
        late response is ignored rather than misdelivered.
      * A request from the server while we wait on ours must be answered
        concurrently (ping -> {}, roots/list -> {"roots": ...}); do not
        serialize on the same lock you use to send the tool call.
    """

    def __init__(self, transport: MCPTransport) -> None:
        self._transport = transport
        self._pending: dict[Any, threading.Event] = {}
        self._results: dict[Any, dict] = {}
        self._next_id = 0
        self._reader: Optional[threading.Thread] = None
        self._closed = False
        self.on_request: Optional[Callable[[dict], Optional[dict]]] = None
        self.on_notification: Optional[Callable[[dict], None]] = None

    def start(self, timeout: float) -> None:
        raise NotImplementedError("STUB -> implement in ollamaquery2.py")

    def request(self, method: str, params: Optional[dict],
                timeout: float) -> dict:
        raise NotImplementedError("STUB -> implement in ollamaquery2.py")

    def notify(self, method: str, params: Optional[dict] = None) -> None:
        raise NotImplementedError("STUB -> implement in ollamaquery2.py")

    def cancel(self, request_id: Any, reason: str = "user cancelled") -> None:
        raise NotImplementedError("STUB -> implement in ollamaquery2.py")

    def close(self) -> None:
        raise NotImplementedError("STUB -> implement in ollamaquery2.py")


# ============================================================================
# Client
# ============================================================================


@dataclass
class MCPCapabilities:
    """Subset of ServerCapabilities we actually branch on."""

    tools: bool = False
    tools_list_changed: bool = False
    resources: bool = False
    resources_subscribe: bool = False
    resources_list_changed: bool = False
    prompts: bool = False
    prompts_list_changed: bool = False
    logging: bool = False
    completions: bool = False

    @classmethod
    def from_dict(cls, caps: Optional[dict]) -> "MCPCapabilities":
        caps = caps or {}
        tools = caps.get("tools") or {}
        resources = caps.get("resources") or {}
        prompts = caps.get("prompts") or {}
        return cls(
            tools=bool(tools),
            tools_list_changed=bool(tools.get("listChanged")),
            resources=bool(resources),
            resources_subscribe=bool(resources.get("subscribe")),
            resources_list_changed=bool(resources.get("listChanged")),
            prompts=bool(prompts),
            prompts_list_changed=bool(prompts.get("listChanged")),
            logging=bool(caps.get("logging")),
            completions=bool(caps.get("completions")),
        )


class MCPClient:
    """One connected, initialized MCP server session.

    LIFECYCLE
    ---------
      connect():
        transport.start(); connection.start(); _initialize()
      _initialize():
        request("initialize", {
            "protocolVersion": cfg.protocol_version,
            "capabilities": {
                "roots": {"listChanged": False} | omitted,
                "sampling": {} | omitted,
                "elicitation": {} | omitted,
            },
            "clientInfo": {"name": MCP_CLIENT_NAME, "version": MCP_CLIENT_VERSION},
        }, cfg.init_timeout_seconds)
        -> store serverInfo / capabilities / instructions
        -> validate returned protocolVersion is one we support, else raise
        notify("notifications/initialized")
        -> optional logging/setLevel
      close(): connection.close(); transport.close()

    CAPABILITY-AWARE METHODS
    ------------------------
      list_tools()        -> follows nextCursor until exhausted
      call_tool(name, args, progress=None, timeout=None)
      list_resources() / list_resource_templates() / read_resource(uri)
      subscribe(uri) / unsubscribe(uri)
      list_prompts() / get_prompt(name, args)
      complete(ref, argument)      # completion/complete
      ping()
      set_log_level(level)

    SERVER-INITIATED HANDLERS (wire these into MCPConnection.on_request)
    -----------------------------------------------------------------
      ping                -> return {}
      roots/list          -> return {"roots": [{"uri": "file://...", "name": ...}]}
                             (only when cfg.allow_roots)
      sampling/createMessage -> convert MCP messages to ModelQuery.query_sync
                             and return {"role":"assistant","content":{...},
                             "model":..., "stopReason":...}
                             (only when cfg.allow_sampling; else error -32601)
      elicitation/create  -> bridge to ollamaquery2's TTY confirm/input and
                             return {"action":"accept"|"decline"|"cancel",
                             "content": {...}}
                             Handle requestedSchema; -32042 means a URL-mode
                             elicitation the client cannot render.
    """

    def __init__(self, cfg: MCPServerConfig,
                 transport: Optional[MCPTransport] = None) -> None:
        self.cfg = cfg
        self.transport = transport or self._make_transport(cfg)
        self.connection = MCPConnection(self.transport)
        self.server_info: dict = {}
        self.instructions: str = ""
        self.capabilities = MCPCapabilities()
        self.protocol_version: str = cfg.protocol_version
        self.tools: list[dict] = []          # raw MCP tools
        self._progress_cb: Optional[Callable[[dict], None]] = None

    @staticmethod
    def _make_transport(cfg: MCPServerConfig) -> MCPTransport:
        if cfg.transport == "stdio":
            return StdioTransport(cfg)
        if cfg.transport == "streamable-http":
            return StreamableHTTPTransport(cfg)
        if cfg.transport == "sse":
            # Legacy 2024-11-05 SSE is a different handshake (GET /sse for the
            # event stream, then POST to the endpoint it announces). Reusing the
            # streamable-http class here would connect to the wrong protocol and
            # fail confusingly, so refuse instead.
            raise ValueError(
                f"MCP server {cfg.name!r}: legacy 'sse' transport is not implemented; "
                f"use 'streamable-http' (see the transport cheat-sheet in §1)")
        raise ValueError(f"unsupported transport {cfg.transport!r}")

    # -- lifecycle ----------------------------------------------------------
    def connect(self) -> None:
        raise NotImplementedError("STUB -> implement in ollamaquery2.py")

    def _initialize(self) -> None:
        raise NotImplementedError("STUB -> implement in ollamaquery2.py")

    def close(self) -> None:
        raise NotImplementedError("STUB -> implement in ollamaquery2.py")

    # -- tools --------------------------------------------------------------
    def list_tools(self) -> list[dict]:
        """tools/list with cursor pagination; refresh self.tools."""
        raise NotImplementedError("STUB -> implement in ollamaquery2.py")

    def call_tool(self, name: str, arguments: dict,
                  progress: Optional[Callable[[dict], None]] = None,
                  timeout: Optional[float] = None) -> dict:
        """tools/call. Attach _meta.progressToken to receive progress."""
        raise NotImplementedError("STUB -> implement in ollamaquery2.py")

    # -- resources ----------------------------------------------------------
    def list_resources(self) -> list[dict]:
        raise NotImplementedError("STUB -> implement in ollamaquery2.py")

    def list_resource_templates(self) -> list[dict]:
        raise NotImplementedError("STUB -> implement in ollamaquery2.py")

    def read_resource(self, uri: str) -> dict:
        raise NotImplementedError("STUB -> implement in ollamaquery2.py")

    def subscribe(self, uri: str) -> None:
        raise NotImplementedError("STUB -> implement in ollamaquery2.py")

    def unsubscribe(self, uri: str) -> None:
        raise NotImplementedError("STUB -> implement in ollamaquery2.py")

    # -- prompts / completion / misc ----------------------------------------
    def list_prompts(self) -> list[dict]:
        raise NotImplementedError("STUB -> implement in ollamaquery2.py")

    def get_prompt(self, name: str, arguments: Optional[dict] = None) -> dict:
        raise NotImplementedError("STUB -> implement in ollamaquery2.py")

    def complete(self, ref: dict, argument: dict) -> dict:
        raise NotImplementedError("STUB -> implement in ollamaquery2.py")

    def ping(self) -> None:
        raise NotImplementedError("STUB -> implement in ollamaquery2.py")

    def set_log_level(self, level: str) -> None:
        raise NotImplementedError("STUB -> implement in ollamaquery2.py")


# ============================================================================
# Manager (multiple servers)
# ============================================================================


class MCPManager:
    """Own every configured server, aggregate tools, dispatch calls.

    RESPONSIBILITIES
    ----------------
      * Hold MCPConfig + {server_name: MCPClient}.
      * start(server_name) / start_all(eager-only) - lazy by default.
      * aggregate_tools() -> merged AGENTIC_TOOL_DEFS-compatible dict and a
        reverse map {model_name: (server_name, tool_name)}.
      * build_openai_specs() -> list for openai_tools_spec().
      * call(model_name, arguments) -> routes to the owning client, applies
        timeout + progress, maps the result via call_result_to_tool_output.
      * Dynamic policy sets: read_only_names(), destructive_names(),
        confirm_names() - feed DESTRUCTIVE_TOOLS / PLAN_MODE_TOOLS.
      * handle list_changed notifications: refresh tools for that server and
        flag the registry for a rebuild.
      * Circuit breaker: after N consecutive failures mark a server degraded
        and stop calling it for X seconds (never block the agent loop).
      * reload(): re-read config, start new servers, stop removed ones.

    COLLISION POLICY
    ----------------
      model_name = sanitize(prefix__tool); if it still collides, append
      "_{n}". Keep the authoritative reverse map here. Never let two MCP tools
      share a model-visible name, even within one server.
    """

    def __init__(self, config: MCPConfig) -> None:
        self.config = config
        self.clients: dict[str, MCPClient] = {}
        self._reverse: dict[str, tuple[str, str]] = {}
        self._lock = threading.RLock()
        self._failures: dict[str, int] = {}

    # -- lifecycle ----------------------------------------------------------
    def start(self, server_name: str) -> MCPClient:
        raise NotImplementedError("STUB -> implement in ollamaquery2.py")

    def start_all(self) -> None:
        """Start only `startup == "eager"` servers; others connect on demand."""
        raise NotImplementedError("STUB -> implement in ollamaquery2.py")

    def stop(self, server_name: str) -> None:
        raise NotImplementedError("STUB -> implement in ollamaquery2.py")

    def shutdown(self) -> None:
        raise NotImplementedError("STUB -> implement in ollamaquery2.py")

    def reload(self) -> None:
        raise NotImplementedError("STUB -> implement in ollamaquery2.py")

    # -- tool aggregation ---------------------------------------------------
    def ensure_started(self, server_name: str) -> MCPClient:
        """Lazy-connect helper used by aggregation and dispatch."""
        raise NotImplementedError("STUB -> implement in ollamaquery2.py")

    def aggregate_tools(self) -> dict[str, dict]:
        """Merge all enabled servers into AGENTIC_TOOL_DEFS shape + reverse map.

        Filtering order per server:
          1. enabled
          2. allowed_tools (if non-empty) then denied_tools
          3. read_only_only -> keep only readOnlyHint == True
        Then namespacing/collision resolution.
        """
        raise NotImplementedError("STUB -> implement in ollamaquery2.py")

    def build_openai_specs(self) -> list[dict]:
        """Native `tools` array for every aggregated MCP tool.

        Built from `aggregate_tools()['parameters']` - the same normalized schema
        the inline renderer sees - so the two surfaces cannot describe different
        parameters. Convenience for callers that want the list without going
        through ToolRegistry; recipe D needs neither, because merging into
        `_plan_allowed_tools` already feeds `openai_tools_spec()`.
        """
        raise NotImplementedError("STUB -> implement in ollamaquery2.py")

    # -- policy helpers (called by the integration hooks) -------------------
    def read_only_names(self) -> set[str]:
        """Model names declared readOnlyHint true (any config filter applied)."""
        raise NotImplementedError("STUB -> implement in ollamaquery2.py")

    def destructive_names(self) -> set[str]:
        """destructiveHint == True, OR readOnlyHint == False."""
        raise NotImplementedError("STUB -> implement in ollamaquery2.py")

    def confirm_names(self) -> set[str]:
        raise NotImplementedError("STUB -> implement in ollamaquery2.py")

    def plan_read_only_names(self) -> set[str]:
        """Model names `/agentic plan` may expose.

        Implement as `plan_tool_names(self.classified_tools())`: config-visible
        tools whose readOnlyHint is explicitly true. Deliberately NOT the same as
        `read_only_names()` - that one ignores the config filters, and a tool the
        operator denied must not reappear just because the server calls it
        read-only.
        """
        raise NotImplementedError("STUB -> implement in ollamaquery2.py")

    def classified_tools(self) -> dict[str, list[MCPToolEntry]]:
        """Per-server `classify_tools()` output for connected servers.

        Single source for `/mcp tools`, the `[plan]` tags and `plan_read_only_names()`,
        so the listing can never disagree with what plan mode actually exposes.
        Servers that are not connected yet map to [] (or connect them here if
        aggregation-time connection is chosen over first-call connection).
        """
        raise NotImplementedError("STUB -> implement in ollamaquery2.py")

    # -- dispatch -----------------------------------------------------------
    def call(self, model_name: str, arguments: dict,
             progress: Optional[Callable[[dict], None]] = None) -> dict:
        """Route one model tool call. Returns {"success","output","error"}.

        Must never raise into the ReAct loop: convert every failure (unknown
        tool, server down, timeout, protocol error) into success=False.
        """
        raise NotImplementedError("STUB -> implement in ollamaquery2.py")

    # -- integration surface ------------------------------------------------
    def tool_registry_handlers(self) -> dict[str, Callable[..., dict]]:
        """Return {model_name: handler} to splice into ToolRegistry._handlers.

        ollamaquery2 handlers are module functions called as `handler(self,
        args)` (registry, arguments) - see ToolRegistry.execute. So each value
        must be a callable with that signature, e.g. a closure:

            def make_handler(model_name, manager):
                def handler(_registry, args):
                    return manager.call(model_name, args)
                return handler

        Do not return a bound `manager.call` directly (wrong arity).
        """
        raise NotImplementedError("STUB -> implement in ollamaquery2.py")

    def status_lines(self) -> list[str]:
        """For `/mcp`: one line per server (state, endpoint, tool count)."""
        raise NotImplementedError("STUB -> implement in ollamaquery2.py")


# ============================================================================
# Integration recipe (copy-paste checklist for ollamaquery2.py)
# ============================================================================

PROTOCOL_TRANSCRIPT = r"""
MINIMAL LIVE TRANSCRIPT (stdio, newline-delimited JSON)
--------------------------------------------------------
-> {"jsonrpc":"2.0","id":1,"method":"initialize","params":{
      "protocolVersion":"2025-11-25",
      "capabilities":{"elicitation":{},"roots":{"listChanged":false}},
      "clientInfo":{"name":"ollamaquery2","version":"0.0.1"}}}
<- {"jsonrpc":"2.0","id":1,"result":{
      "protocolVersion":"2025-11-25",
      "capabilities":{"tools":{"listChanged":true},"prompts":{},"resources":{}},
      "serverInfo":{"name":"kubernetes-mcp-server","version":"..."},
      "instructions":"..."}}
-> {"jsonrpc":"2.0","method":"notifications/initialized"}
-> {"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}
<- {"jsonrpc":"2.0","id":2,"result":{
      "tools":[{"name":"pods_list","description":"...",
                "inputSchema":{"type":"object","properties":{...}},
                "annotations":{"readOnlyHint":true,"destructiveHint":false}}],
      "nextCursor":"<opaque>"}}          # repeat with {"cursor":"<opaque>"}
-> {"jsonrpc":"2.0","id":3,"method":"tools/call","params":{
      "name":"pods_list",
      "arguments":{"namespace":"default"},
      "_meta":{"progressToken":"p1"}}}
<- {"jsonrpc":"2.0","method":"notifications/progress","params":{
      "progressToken":"p1","progress":1,"total":1,"message":"listing"}}
<- {"jsonrpc":"2.0","id":3,"result":{
      "content":[{"type":"text","text":"NAME ..."}],
      "structuredContent":{"items":[...]},
      "isError":false}}
-> {"jsonrpc":"2.0","method":"notifications/cancelled",
      "params":{"requestId":3,"reason":"timeout"}}

STREAMABLE HTTP (same JSON bodies)
----------------------------------
POST http://127.0.0.1:8080/mcp
  Content-Type: application/json
  Accept: application/json, text/event-stream
  MCP-Protocol-Version: 2025-11-25          # after initialize
  Mcp-Session-Id: <from initialize response> # after initialize
  Authorization: Bearer <token>              # if OAuth
body: <one JSON-RPC object>
response: 200 application/json | 200 text/event-stream | 202 (notification)
         401 (auth) | 404 (session expired -> re-initialize)

SERVER -> CLIENT REQUESTS (answer these while tools/call is outstanding)
-----------------------------------------------------------------------
<- {"jsonrpc":"2.0","id":"srv-1","method":"ping"}
-> {"jsonrpc":"2.0","id":"srv-1","result":{}}
<- {"jsonrpc":"2.0","id":"srv-2","method":"roots/list"}
-> {"jsonrpc":"2.0","id":"srv-2","result":{"roots":[
      {"uri":"file:///home/ollama/build/ollamaquery/ollamaquery2",
       "name":"cwd"}]}}
<- {"jsonrpc":"2.0","id":"srv-3","method":"elicitation/create","params":{
      "message":"Run pods_exec?","requestedSchema":{...}}}
-> {"jsonrpc":"2.0","id":"srv-3","result":{"action":"accept","content":{}}}
   # action: accept | decline | cancel
"""

BUILD_ORDER = """
M0  Config + discovery + validation (`load_mcp_config`)            [given here]
M1  InMemoryTransport + MCPConnection + unit tests                 (no server)
M2  StdioTransport + initialize + tools/list                       (first live)
M3  Tool adaptation + ToolRegistry merge + manager.call            (agentic use)
M4  HTTP transport + session id + SSE + OAuth                      (remote)
M5  Prompts/resources/completion + server-initiated sampling/elicit
M6  list_changed refresh, reload, circuit breaker, spill caps      (polish)
Do M0->M3 before touching HTTP: it delivers the whole local-server value.
"""


INTEGRATION_RECIPE = """
A. CommandContext.__init__ (~:1297)
     self.mcp_config = load_mcp_config()          # or from args
     self.mcp_manager = MCPManager(self.mcp_config) if self.mcp_config.servers else None

B. Startup in main() (~:14072), after args are parsed and before the chat loop
     if not args.no_mcp and ctx.mcp_manager:
         ctx.mcp_manager.start_all()              # eager servers only
     atexit.register(ctx.mcp_manager.shutdown)    # reuse the existing atexit hook

C. ToolRegistry.__init__ (~:6060)
     if ctx and ctx.mcp_manager:
         self._handlers.update(ctx.mcp_manager.tool_registry_handlers())

D. Tool surface merge - one choke point, not two
     `_plan_allowed_tools()` (~:6073) is what BOTH get_system_prompt_block()
     and openai_tools_spec() read. Merge there so inline and native stay in
     sync (mutating a local dict is not enough; mutating the AGENTIC_TOOL_DEFS
     global works but is harder to test):

         def _plan_allowed_tools(self):
             defs = dict(AGENTIC_TOOL_DEFS)
             mgr = getattr(self._ctx, "mcp_manager", None)
             if mgr:
                 defs.update(mgr.aggregate_tools())   # already namespaced
             if self._ctx and getattr(self._ctx, "plan_mode", False):
                 allowed = plan_active_tools(self._ctx)
                 return {n: d for n, d in defs.items() if n in allowed}
             return defs

     list_tools_str() (~:6084): add an `[mcp:<server>]` tag for discoverability.
     TOOL_ARG_ALIASES (~:6020) will NOT cover MCP tools: the model must use the
     exact property names from each inputSchema. Do not alias MCP args silently;
     a wrong name is a protocol error the server will report.

E. Safety sets - keep them dynamic
     DESTRUCTIVE_TOOLS is a module-level set consulted by _confirm() and
     list_tools_str(). MCP tools can appear/disappear via list_changed, so
     prefer consulting the manager instead of freezing the union at startup:

         def _is_destructive_name(ctx, name):
             return name in DESTRUCTIVE_TOOLS or (
                 ctx.mcp_manager and name in ctx.mcp_manager.destructive_names())

     Same for plan mode - extend plan_active_tools() (~:3936) rather than
     editing PLAN_MODE_TOOLS once:

         def plan_active_tools(ctx):
             base = (PLAN_MODE_RUN_TOOLS if getattr(ctx, "plan_run", False)
                     else PLAN_MODE_TOOLS)
             mgr = getattr(ctx, "mcp_manager", None)
             return base | (mgr.read_only_names() if mgr else set())

     Three-state rule: only readOnlyHint == True enters plan mode; unknown or
     absent annotation stays out.

F. Confirmation bridge
     ToolRegistry._confirm (~:6092): if the name is in confirm_names(), route
     through the existing prompt instead of the `tool_name in DESTRUCTIVE_TOOLS`
     shortcut. Also implement elicitation/create here for server-driven prompts.

G. Result handling
     ToolRegistry.execute (~:6134) routes MCP names to manager.call(); pass
     `spill=make_spill_helper(ctx)` and
     `max_chars=resolve_max_result_chars(ctx, cfg.max_result_chars)` into
     call_result_to_tool_output(), then let the existing
     summarize_tool_results() pipeline handle what remains.
     ADD ("mcp_result_", ".json") to the `prefixes` tuple in
     `_cleanup_spill_files` (~:4272) - MCP results share CLUSTER_SPILL_DIR, so
     without that entry they are never removed at exit (the prototype's own
     atexit pass only exists for the standalone dry-run).

H. Commands + CLI
     /mcp ... in ChatLoop dispatch (~:8991); --mcp-config/--no-mcp in
     _build_parser (~:13588).

I. Sampling bridge (optional)
     If a server sends sampling/createMessage, translate its messages to the
     ModelQuery message format and call query_sync(); return the assistant
     message. Off unless cfg.allow_sampling.
"""


# ============================================================================
# Self-test / recipe printer
# ============================================================================


def _print_recipe() -> None:
    print(__doc__)
    print(PROTOCOL_TRANSCRIPT)
    print(BUILD_ORDER)
    print(INTEGRATION_RECIPE)


def _main(argv: list[str]) -> int:
    config_path = None
    if "--config" in argv:
        idx = argv.index("--config")
        if idx + 1 >= len(argv):
            print("--config needs a path", file=sys.stderr)
            return 2
        config_path = argv[idx + 1]

    if config_path:
        cfg = load_mcp_config(config_path)
        print(f"config: {cfg.source_path}")
        problems: list[str] = []
        for server in cfg.servers.values():
            print(f"  - {server.name}: transport={server.transport} "
                  f"enabled={server.enabled} startup={server.startup}")
            problems.extend(validate_server_config(server))
        if problems:
            print("\nproblems:", file=sys.stderr)
            for p in problems:
                print(f"  ! {p}", file=sys.stderr)
            return 1
        print("\nOK")
        return 0

    _print_recipe()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
