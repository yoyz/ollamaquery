# MCP client — configuration and a local test server

`ollamaquery2` can consume local [Model Context Protocol](https://modelcontextprotocol.io)
servers as agentic tools. This build implements the **stdio** transport and the
**tools** capability only, targeting protocol revision `2025-11-25`.

Everything else fails closed with a validation warning: Streamable HTTP/SSE
transports, OAuth, prompts, resources, completion, sampling, elicitation, roots,
`list_changed` refresh and `reload`. See `doc/mcp-conformance.md` for the
wire-level review against the reference C++ SDK.

## Config discovery

First hit wins:

| Priority   | Source                                     | Notes                                     |
| ---------: | ------------------------------------------ | ----------------------------------------- |
| 1          | `--mcp-config PATH`                        | Must exist (`FileNotFoundError` if not)   |
| 2          | `$OLLAMAQUERY_MCP_CONFIG`                  | Must exist                                |
| 3          | `./.ollamaquery-mcp.json` (project)        | Optional                                  |
| 4          | `~/.ollamaquery.d/mcp.json` (user)         | Optional                                  |

`--no-mcp` skips loading entirely. Invalid JSON or a wrong top-level shape is a
`ValueError`; individual server problems are printed as `[MCP] ...` warnings and
do not stop the client.

```bash
python3 ollamaquery2.py --mcp-config ./mcp.test.json -c
OLLAMAQUERY_MCP_CONFIG=./mcp.test.json python3 ollamaquery2.py -c
```

Inside the chat: `/mcp status`, `/mcp tools [server] [--plan|--all]`,
`/mcp connect <server>`, `/mcp disconnect <server>`.

## Example configuration

Save this as `mcp.test.json` (or `~/.ollamaquery.d/mcp.json`). It is the same
shape the test suite validates; the HTTP entry is deliberately disabled because
this build refuses it.

```json
{
  "version": 1,
  "servers": {
    "kubernetes": {
      "transport": "stdio",
      "command": ["kubernetes-mcp-server", "--config", "/path/to/config.toml"],
      "env": {
        "KUBECONFIG": "${KUBECONFIG:-/path/to/kubeconfig}"
      },
      "cwd": null,
      "enabled": true,
      "startup": "lazy",
      "tool_prefix": "k8s",
      "allowed_tools": [],
      "denied_tools": ["pods_exec", "resources_delete"],
      "read_only_only": false,
      "confirm_tools": [],
      "init_timeout_seconds": 15.0,
      "call_timeout_seconds": 120.0,
      "list_timeout_seconds": 30.0,
      "max_result_chars": 100000,
      "retries": 1,
      "backoff_seconds": 0.5,
      "protocol_version": "2025-11-25"
    },
    "kubernetes-http": {
      "transport": "streamable-http",
      "url": "http://127.0.0.1:10080/mcp",
      "enabled": false,
      "startup": "lazy",
      "tool_prefix": "k8shttp",
      "read_only_only": true
    }
  }
}
```

The top-level key may be `servers` or `mcpServers`. Strings, lists and dict
values expand `${VAR}` and `${VAR:-default}` from the environment.

Claude/VS Code-shaped entries are accepted too: `command` may be a string with
an `args` list, and `type` maps to `transport`:

```json
{
  "mcpServers": {
    "filesystem": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/workspace"]
    }
  }
}
```

## Field reference

| Field                      | Default              | Meaning                                                                     |
| -------------------------- | -------------------- | --------------------------------------------------------------------------- |
| `transport`                | `stdio`              | `stdio` is the only implemented transport                                   |
| `command`                  | `[]`                 | argv list (no shell). Absolute path or `${HOME}/...`                        |
| `env`                      | `{}`                 | Extra environment for the child                                             |
| `cwd`                      | `null`               | Working directory for the child                                             |
| `enabled`                  | `true`               | Disabled servers are never started or listed as tools                       |
| `startup`                  | `lazy`               | `lazy` = start on first call; `eager` = start at client init                |
| `tool_prefix`              | sanitized name       | Model-visible namespace; tools become `<prefix>__<tool>`                    |
| `expose_as`                | `{}`                 | Rename individual tools: `{"original": "visible_name"}`                     |
| `allowed_tools`            | `[]`                 | Empty = all; otherwise only these are exposed                               |
| `denied_tools`             | `[]`                 | Always wins over `allowed_tools`                                            |
| `read_only_only`           | `false`              | Hide tools that are not explicitly annotated read-only                      |
| `confirm_tools`            | `[]`                 | Force a confirmation prompt before each call                                |
| `init_timeout_seconds`     | `15.0`               | `initialize` / `tools/list` deadline                                        |
| `call_timeout_seconds`     | `120.0`              | Per `tools/call` deadline                                                   |
| `max_result_chars`         | `100000`             | Ceiling; the live context window may lower it                               |
| `retries`                  | `1`                  | Retries on transport failure                                                |
| `backoff_seconds`          | `0.5`                | Delay between retries                                                       |
| `protocol_version`         | `2025-11-25`         | One of `2026-07-28`, `2025-11-25`, `2025-06-18`, `2025-03-26`, `2024-11-05` |

Unknown keys are reported as `unknown config key(s) ignored` rather than
silently accepted.

## Security model

- **No shell.** The child is spawned with an argv (`shell=False`), its own
  process group, and an environment stripped to `PATH`, `HOME`, `LANG`,
  `LC_ALL`, `TERM` plus the configured `env`.
- **Plan mode is annotation-driven and fail-closed.** A tool enters
  `/agentic plan` only when the server declares `readOnlyHint: true`,
  `destructiveHint` is not true, and no config filter removes it. An absent
  annotation is **not** "safe": the spec defaults `readOnlyHint=false` and
  `destructiveHint=true`.
- **Results spill like other tools.** Output over the effective cap is written
  to `~/.ollamaquery.d/spill/mcp_result_*.json` (mode `0600`) and the model gets
  a pure pointer; a session Path-ACL read-allow rule is registered first so the
  pointer is followable even in strict plan mode.
- **HTTP/OAuth are not implemented.** An `http`/`streamable-http`/`sse`
  transport produces a validation warning and is never used, even if enabled.

## Building a local test server with mcp-cpp

[`Neumann-Labs/mcp-cpp`](https://github.com/Neumann-Labs/mcp-cpp) is a C++20
MCP SDK that speaks the same revision (`2025-11-25`). It is used here as a
reference/oracle, not vendored — check its license before copying source.

### 1. Minimal stdio server

`mcp_test_server.cpp`:

```cpp
#include <mcp/mcp.hpp>

int main() {
    mcp::Server server{
        mcp::Implementation{.name = "oq-test-server", .version = "1.0.0"},
    };

    server.tool("echo",
        nlohmann::json{
            {"type", "object"},
            {"properties", {{"text", {{"type", "string"}}}}},
            {"required", nlohmann::json::array({"text"})},
        },
        [](const nlohmann::json& args) -> mcp::CallToolResult {
            return {.content = {mcp::TextContent{
                .text = args.at("text").get<std::string>()}}};
        });

    server.tool("add",
        nlohmann::json{
            {"type", "object"},
            {"properties", {
                {"a", {{"type", "number"}}},
                {"b", {{"type", "number"}}},
            }},
            {"required", nlohmann::json::array({"a", "b"})},
        },
        [](const nlohmann::json& args) -> mcp::CallToolResult {
            const double a = args.at("a").get<double>();
            const double b = args.at("b").get<double>();
            return {.content = {mcp::TextContent{
                .text = std::to_string(a + b)}}};
        });

    // stdout is the JSON-RPC stream; diagnostics belong on stderr.
    server.run(std::make_unique<mcp::StdioTransport>());
}
```

To make `add` eligible for `/agentic plan`, annotate it read-only. With the SDK
that is the standard tool annotation object; the client requires an explicit
`readOnlyHint: true` on the tool definition (see `doc/mcp-conformance.md` for
the exact wire shape the parser expects).

### 2. CMake build

`CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.24)
project(oq_mcp_test_server LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(mcp
    GIT_REPOSITORY https://github.com/Neumann-Labs/mcp-cpp.git
    GIT_TAG        main)
FetchContent_MakeAvailable(mcp)

add_executable(mcp_test_server mcp_test_server.cpp)
target_link_libraries(mcp_test_server PRIVATE mcp::mcp)
```

```bash
cmake -S . -B build -G Ninja
cmake --build build
# -> build/mcp_test_server
```

The core `mcp::mcp` target is stdio-only (no OpenSSL). Link `mcp::http` only if
you are building the SDK's HTTP example for another client.

### 3. Point ollamaquery2 at it

```json
{
  "version": 1,
  "servers": {
    "calc": {
      "transport": "stdio",
      "command": ["/absolute/path/to/build/mcp_test_server"],
      "startup": "eager",
      "tool_prefix": "calc"
    }
  }
}
```

```bash
python3 ollamaquery2.py --mcp-config ./mcp.test.json -c
```

```text
/mcp status
/mcp tools
/mcp connect calc
```

Once connected, the agentic tool surface includes `calc__echo` and `calc__add`.
A quick manual check that the binary speaks the framing (stdout must carry only
JSON-RPC lines):

```bash
printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"probe","version":"0"}}}' \
  | ./build/mcp_test_server
```

### 4. HTTP variant (reference only)

The SDK ships `examples/http_calculator_server`, which binds `127.0.0.1:8080`
by default. That port is commonly taken (llama.cpp, other tooling); if you run
it, change the port in `mcp::HttpServerHost::Options` to `10080`:

```cpp
mcp::HttpServerHost host{
    mcp::Implementation{.name = "calc", .version = "1.0"},
    mcp::HttpServerHost::Options{
        .host = "127.0.0.1",
        .port = 10080,
        .path = "/mcp",
    },
    [](mcp::Server& s) { /* register tools */ },
};
```

`ollamaquery2` will not connect to it in this build (HTTP is not implemented).
It is useful for testing the server with the official MCP inspector or another
client, and for exercising the disabled HTTP entry in the example configuration.

## Troubleshooting

| Symptom                                            | Likely cause                                                          |
| -------------------------------------------------- | --------------------------------------------------------------------- |
| `[MCP] ... transport is not implemented`           | HTTP/SSE entry enabled; use stdio or disable the server               |
| `[MCP] ... requires 'command'`                     | stdio server without a `command` argv                                 |
| `[MCP] ... unknown config key(s) ignored`          | Typo in a field name; the server still loads                          |
| Connect hangs, then times out                      | Server never answered `initialize`; check its stderr                  |
| `[MCP] no servers configured`                      | No config file found, or `--no-mcp` was passed                        |
| Tools missing from the prompt                      | `allowed_tools`/`denied_tools`/`read_only_only` filtered them         |
| Tools missing in `/agentic plan` only              | No explicit `readOnlyHint: true` on the server's tool annotation      |

Server stderr is preserved for diagnostics; the client never parses it as
protocol data.
