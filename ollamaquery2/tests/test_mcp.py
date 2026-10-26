#!/usr/bin/env python3
"""Tests for the MCP client, the /mcp command, and the plan-mode MCP policy.

The stdio path is exercised against a real child process (a scripted fake MCP
server), not a mock, so JSON-RPC framing, pagination, and the multiplexer are
covered end to end. No external MCP server is required.
"""

import io
import json
import os
import sys
import types
import tempfile
import textwrap
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ollamaquery2 as m

FAKE_SERVER = textwrap.dedent('''
    #!/usr/bin/env python3
    """Scripted MCP server: pagination, annotations, isError, server->client ping."""
    import json
    import sys

    PAGE1 = [{
        "name": "pods_list",
        "description": "List pods",
        "inputSchema": {"type": "object", "properties": {"ns": {"type": "string"}}},
        "annotations": {"readOnlyHint": True, "destructiveHint": False},
    }]
    PAGE2 = [
        {"name": "pods_delete", "description": "Delete a pod",
         "inputSchema": {"type": "object", "properties": {}},
         "annotations": {"readOnlyHint": False, "destructiveHint": True}},
        {"name": "config_view", "description": "View config",
         "inputSchema": {"type": "object", "properties": {}}},
    ]

    def send(obj):
        sys.stdout.write(json.dumps(obj) + "\\n")
        sys.stdout.flush()

    sys.stderr.write("fake mcp server starting\\n")
    sys.stderr.flush()

    for line in sys.stdin:
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        method = msg.get("method")
        mid = msg.get("id")
        if method == "initialize":
            send({"jsonrpc": "2.0", "id": mid, "result": {
                "protocolVersion": "2025-11-25",
                "capabilities": {"tools": {"listChanged": True}},
                "serverInfo": {"name": "fake", "version": "1.0"}}})
        elif method == "notifications/initialized":
            continue
        elif method == "tools/list":
            cursor = (msg.get("params") or {}).get("cursor")
            if not cursor:
                send({"jsonrpc": "2.0", "id": mid,
                      "result": {"tools": PAGE1, "nextCursor": "page2"}})
            else:
                send({"jsonrpc": "2.0", "id": mid, "result": {"tools": PAGE2}})
        elif method == "tools/call":
            params = msg.get("params") or {}
            name = params.get("name")
            args = params.get("arguments") or {}
            if name == "pods_list":
                # Server-initiated request while tools/call is outstanding: the
                # client must answer it (multiplexing), then we reply.
                send({"jsonrpc": "2.0", "id": "srv-ping", "method": "ping"})
                json.loads(sys.stdin.readline())
                send({"jsonrpc": "2.0", "id": mid, "result": {
                    "content": [{"type": "text", "text": "PODS OK " + json.dumps(args)}],
                    "isError": False}})
            elif name == "boom":
                send({"jsonrpc": "2.0", "id": mid, "result": {
                    "content": [{"type": "text", "text": "nope"}], "isError": True}})
            else:
                send({"jsonrpc": "2.0", "id": mid, "result": {
                    "content": [{"type": "text", "text": "ok"}]}})
        elif method == "notifications/cancelled":
            continue
        elif mid is not None:
            send({"jsonrpc": "2.0", "id": mid,
                  "error": {"code": -32601, "message": "unsupported"}})
''')


REPEATING_CURSOR_SERVER = textwrap.dedent('''
    #!/usr/bin/env python3
    """Malformed MCP server: tools/list always returns the same nextCursor."""
    import json
    import sys

    def send(obj):
        sys.stdout.write(json.dumps(obj) + "\\n")
        sys.stdout.flush()

    for line in sys.stdin:
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        method = msg.get("method")
        mid = msg.get("id")
        if method == "initialize":
            send({"jsonrpc": "2.0", "id": mid, "result": {
                "protocolVersion": "2025-11-25",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "looper", "version": "1.0"}}})
        elif method == "notifications/initialized":
            continue
        elif method == "tools/list":
            send({"jsonrpc": "2.0", "id": mid, "result": {
                "tools": [{"name": "t", "inputSchema": {"type": "object"}}],
                "nextCursor": "again"}})
        elif mid is not None:
            send({"jsonrpc": "2.0", "id": mid, "result": {}})
''')


class _StubConnection:
    """Minimal MCPConnection stand-in that serves synthetic tools/list pages."""

    def __init__(self, page_factory):
        self.page_factory = page_factory
        self.requests = 0

    def request(self, method, params, timeout=None):
        self.requests += 1
        return self.page_factory(self.requests)


class _FakeClock:
    """Monotonic clock that advances on every read."""

    def __init__(self, step=100.0):
        self.now = 0.0
        self.step = step

    def __call__(self):
        self.now += self.step
        return self.now


class FakeManager:
    """Minimal duck-typed manager for the registry/plan integration tests."""

    def __init__(self, plan_names=("fake__pods_list",),
                 destructive=("fake__pods_delete",), confirm=(), defs=None):
        self._plan = set(plan_names)
        self._destructive = set(destructive)
        self._confirm = set(confirm)
        self._defs = defs or {
            "fake__pods_list": {"description": "List pods",
                                "parameters": {"type": "object", "properties": {}},
                                "_mcp": {"server": "fake", "tool": "pods_list"}},
            "fake__pods_delete": {"description": "Delete a pod",
                                  "parameters": {"type": "object", "properties": {}},
                                  "_mcp": {"server": "fake", "tool": "pods_delete"}},
        }
        self.calls = []

    def plan_read_only_names(self):
        return set(self._plan)

    def destructive_names(self):
        return set(self._destructive)

    def confirm_names(self):
        return set(self._confirm)

    def aggregate_tools(self):
        return dict(self._defs)

    def ensure_started_all(self):
        return None

    def resolve(self, name):
        return ("fake", name.split("__", 1)[1]) if name in self._defs else None

    def call(self, name, args):
        self.calls.append((name, args))
        return {"success": True, "output": f"called {name}", "error": None}


def _ctx(**attrs):
    """Build a lightweight context for ToolRegistry/plan tests."""
    base = {"plan_mode": False, "plan_run": False, "agentic_mode": False,
            "auto_confirm": False, "mcp_manager": None, "path_acl": None,
            "context_window_size": 0}
    base.update(attrs)
    return types.SimpleNamespace(**base)


class TestPlanPolicy(unittest.TestCase):
    def test_union_adds_read_only_names(self):
        ctx = _ctx(mcp_manager=FakeManager())
        self.assertIn("fake__pods_list", m.plan_active_tools(ctx))
        self.assertNotIn("fake__pods_delete", m.plan_active_tools(ctx))

    def test_no_manager_is_unchanged(self):
        self.assertEqual(m.plan_active_tools(_ctx()), m.PLAN_MODE_TOOLS)

    def test_run_tier_keeps_run_command_and_adds_mcp(self):
        ctx = _ctx(mcp_manager=FakeManager(), plan_run=True)
        active = m.plan_active_tools(ctx)
        self.assertIn("run_command", active)
        self.assertIn("fake__pods_list", active)

    def test_broken_manager_cannot_widen_plan_mode(self):
        class Broken:
            def plan_read_only_names(self):
                raise RuntimeError("boom")
        self.assertEqual(m.plan_active_tools(_ctx(mcp_manager=Broken())),
                         m.PLAN_MODE_TOOLS)


class TestToolRegistryIntegration(unittest.TestCase):
    def test_mcp_tools_merged_into_surface(self):
        ctx = _ctx(mcp_manager=FakeManager())
        registry = m.ToolRegistry(ctx)
        names = registry._plan_allowed_tools()
        self.assertIn("read_file", names)
        self.assertIn("fake__pods_list", names)

    def test_plan_mode_hides_non_plan_mcp_tools(self):
        ctx = _ctx(mcp_manager=FakeManager(), plan_mode=True)
        names = m.ToolRegistry(ctx)._plan_allowed_tools()
        self.assertIn("fake__pods_list", names)
        self.assertNotIn("fake__pods_delete", names)

    def test_execute_routes_mcp_call(self):
        manager = FakeManager()
        ctx = _ctx(mcp_manager=manager, plan_mode=True)
        result = m.ToolRegistry(ctx).execute("fake__pods_list", {"ns": "default"})
        self.assertTrue(result["success"], result)
        self.assertEqual(manager.calls, [("fake__pods_list", {"ns": "default"})])

    def test_execute_denies_write_mcp_tool_in_plan_mode(self):
        manager = FakeManager()
        ctx = _ctx(mcp_manager=manager, plan_mode=True)
        result = m.ToolRegistry(ctx).execute("fake__pods_delete", {})
        self.assertFalse(result["success"])
        self.assertIn("plan mode", result["error"])
        self.assertEqual(manager.calls, [])

    def test_execute_allows_destructive_mcp_tool_outside_plan_with_confirm(self):
        manager = FakeManager()
        ctx = _ctx(mcp_manager=manager, plan_mode=False)
        with patch("builtins.input", return_value="y"):
            result = m.ToolRegistry(ctx).execute("fake__pods_delete", {})
        self.assertTrue(result["success"], result)

    def test_declined_confirmation_blocks_that_call(self):
        manager = FakeManager()
        ctx = _ctx(mcp_manager=manager, plan_mode=False)
        with patch("builtins.input", return_value="n"):
            result = m.ToolRegistry(ctx).execute("fake__pods_delete", {})
        self.assertFalse(result["success"])
        self.assertIn("Cancelled", result["error"])
        self.assertEqual(manager.calls, [])

    def test_unknown_tool_still_rejected(self):
        ctx = _ctx(mcp_manager=FakeManager())
        result = m.ToolRegistry(ctx).execute("not_a_tool", {})
        self.assertFalse(result["success"])
        self.assertIn("Unknown tool", result["error"])

    def test_list_tools_str_tags_mcp_origin(self):
        ctx = _ctx(mcp_manager=FakeManager())
        text = m.ToolRegistry(ctx).list_tools_str()
        self.assertIn("[mcp:fake]", text)

    def test_chat_mode_does_not_connect_lazy_servers(self):
        class Recording(FakeManager):
            def __init__(self):
                super().__init__()
                self.connected = 0
            def ensure_started_all(self):
                self.connected += 1
        manager = Recording()
        registry = m.ToolRegistry(_ctx(mcp_manager=manager, agentic_mode=False))
        registry._plan_allowed_tools()
        self.assertEqual(manager.connected, 0)
        registry = m.ToolRegistry(_ctx(mcp_manager=manager, agentic_mode=True))
        registry._plan_allowed_tools()
        self.assertEqual(manager.connected, 1)


class TestStdioServerEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.server = os.path.join(cls.tmp.name, "fake_mcp_server.py")
        with open(cls.server, "w", encoding="utf-8") as fh:
            fh.write(FAKE_SERVER)
        os.chmod(cls.server, 0o755)
        cls.cfg = m.MCPServerConfig(name="fake", transport="stdio",
                                    command=[sys.executable, cls.server],
                                    startup="lazy")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_connect_initialize_and_paginate(self):
        client = m.MCPClient(self.cfg)
        try:
            client.connect()
            self.assertTrue(client.connected)
            self.assertEqual(client.protocol_version, "2025-11-25")
            self.assertEqual(client.server_info.get("name"), "fake")
            self.assertEqual([tool["name"] for tool in client.tools],
                             ["pods_list", "pods_delete", "config_view"])
        finally:
            client.close()

    def test_call_tool_answers_server_ping_mid_call(self):
        client = m.MCPClient(self.cfg)
        try:
            client.connect()
            result = client.call_tool("pods_list", {"ns": "default"}, timeout=10)
            text = result["content"][0]["text"]
            self.assertIn("PODS OK", text)
            self.assertIn("default", text)
        finally:
            client.close()

    def test_is_error_propagates(self):
        client = m.MCPClient(self.cfg)
        try:
            client.connect()
            output = m._mcp_result_to_output(
                _ctx(), client.call_tool("boom", {}, timeout=10), self.cfg, "boom")
            self.assertFalse(output["success"])
            self.assertIn("isError", output["error"])
        finally:
            client.close()

    def test_manager_lifecycle_and_policy(self):
        config = m.MCPConfig({"fake": self.cfg})
        manager = m.MCPManager(config, _ctx())
        try:
            manager.start("fake")
            self.assertEqual(manager.plan_read_only_names(), {"fake__pods_list"})
            self.assertIn("fake__pods_delete", manager.destructive_names())
            # Unannotated tools must fail closed, not be treated as harmless.
            self.assertIn("fake__config_view", manager.destructive_names())
            defs = manager.aggregate_tools()
            self.assertEqual(set(defs), {"fake__pods_list", "fake__pods_delete",
                                         "fake__config_view"})
            self.assertEqual(defs["fake__pods_list"]["parameters"]["required"], [])
            result = manager.call("fake__pods_list", {"ns": "x"})
            self.assertTrue(result["success"], result)
            self.assertIn("PODS OK", result["output"])
        finally:
            manager.shutdown()

    def test_manager_call_after_shutdown_is_an_error_not_an_exception(self):
        config = m.MCPConfig({"fake": self.cfg})
        manager = m.MCPManager(config, _ctx())
        manager.start("fake")
        manager.shutdown()
        result = manager.call("fake__pods_list", {})
        self.assertFalse(result["success"])
        self.assertIn("Unknown MCP tool", result["error"])

    def test_unknown_server_name_is_reported(self):
        manager = m.MCPManager(m.MCPConfig({}), _ctx())
        result = manager.call("ghost__tool", {})
        self.assertFalse(result["success"])
        self.assertIn("Unknown MCP tool", result["error"])

    def test_status_lines(self):
        config = m.MCPConfig({"fake": self.cfg})
        manager = m.MCPManager(config, _ctx())
        try:
            manager.start("fake")
            lines = manager.status_lines()
            self.assertEqual(len(lines), 1)
            self.assertIn("connected", lines[0])
            self.assertIn("1 plan", lines[0])
        finally:
            manager.shutdown()

    def test_attach_manager_from_cli_args(self):
        config_path = os.path.join(self.tmp.name, "mcp.json")
        with open(config_path, "w", encoding="utf-8") as fh:
            json.dump({"servers": {"fake": {
                "transport": "stdio",
                "command": [sys.executable, self.server],
                "startup": "eager"}}}, fh)
        ctx = _ctx()
        with patch("sys.stderr", new=io.StringIO()):
            m._attach_mcp_manager(ctx, types.SimpleNamespace(
                mcp_config=config_path, no_mcp=False))
        try:
            self.assertIsNotNone(ctx.mcp_manager)
            self.assertTrue(ctx.mcp_manager.clients["fake"].connected)
            self.assertEqual(ctx.mcp_manager.plan_read_only_names(),
                             {"fake__pods_list"})
        finally:
            ctx.mcp_manager.shutdown()

    def test_no_mcp_flag_disables_client(self):
        ctx = _ctx()
        m._attach_mcp_manager(ctx, types.SimpleNamespace(
            mcp_config="/nonexistent/mcp.json", no_mcp=True))
        self.assertIsNone(ctx.mcp_manager)

    def test_missing_config_warns_and_continues(self):
        ctx = _ctx()
        with patch("sys.stderr", new=io.StringIO()) as captured:
            m._attach_mcp_manager(ctx, types.SimpleNamespace(
                mcp_config="/nonexistent/mcp.json", no_mcp=False))
        self.assertIsNone(ctx.mcp_manager)
        self.assertIn("not found", captured.getvalue())


class TestPaginationHardening(unittest.TestCase):
    """A broken server must not hang connect() or grow the catalog forever."""

    def _client_with_pages(self, page_factory):
        cfg = m.MCPServerConfig(name="flood", transport="stdio", command=["x"])
        client = m.MCPClient(cfg)
        client.connection = _StubConnection(page_factory)
        return client

    def test_repeated_cursor_aborts_connect(self):
        """End-to-end: a server that repeats its cursor must fail fast."""
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "looper.py")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(REPEATING_CURSOR_SERVER)
            cfg = m.MCPServerConfig(name="looper", transport="stdio",
                                    command=[sys.executable, path],
                                    startup="lazy")
            client = m.MCPClient(cfg)
            try:
                with self.assertRaises(m.MCPError) as caught:
                    client.connect()
                self.assertIn("repeated cursor", str(caught.exception))
                self.assertFalse(client.connected)
            finally:
                client.close()

    def test_repeated_cursor_stops_on_second_page(self):
        client = self._client_with_pages(
            lambda n: {"tools": [{"name": f"t{n}"}], "nextCursor": "again"})
        with self.assertRaises(m.MCPError) as caught:
            client.list_tools()
        self.assertIn("repeated cursor", str(caught.exception))
        self.assertEqual(client.connection.requests, 2)

    def test_unique_cursor_flood_hits_page_cap(self):
        client = self._client_with_pages(
            lambda n: {"tools": [], "nextCursor": f"c{n}"})
        with patch.object(m, "MCP_LIST_MAX_PAGES", 3):
            with self.assertRaises(m.MCPError) as caught:
                client.list_tools()
        self.assertIn("pages", str(caught.exception))
        self.assertEqual(client.connection.requests, 3)

    def test_catalog_size_cap(self):
        client = self._client_with_pages(
            lambda n: {"tools": [{"name": f"a{n}"}, {"name": f"b{n}"}],
                       "nextCursor": f"c{n}"})
        with patch.object(m, "MCP_LIST_MAX_TOOLS", 3):
            with self.assertRaises(m.MCPError) as caught:
                client.list_tools()
        self.assertIn("more than 3 tools", str(caught.exception))

    def test_total_deadline_bounds_a_fresh_cursor_flood(self):
        client = self._client_with_pages(
            lambda n: {"tools": [], "nextCursor": f"c{n}"})
        with patch.object(m.time, "monotonic", side_effect=_FakeClock()):
            with self.assertRaises(m.MCPError) as caught:
                client.list_tools()
        self.assertIn("time budget", str(caught.exception))

    def test_large_but_valid_catalog_still_completes(self):
        """A 1,000-tool catalog served one tool per page must still load."""
        total_pages = 1000
        client = self._client_with_pages(
            lambda n: {"tools": [{"name": f"t{n}"}],
                       "nextCursor": None if n >= total_pages else f"c{n}"})
        tools = client.list_tools()
        self.assertEqual(len(tools), total_pages)
        self.assertEqual(client.connection.requests, total_pages)


class TestConnection(unittest.TestCase):
    def _client(self, script):
        transport = m.InMemoryMCPTransport(script)
        connection = m.MCPConnection(transport)
        return transport, connection

    def test_request_response(self):
        transport, connection = self._client([{"jsonrpc": "2.0", "result": {"ok": 1}}])
        connection.transport.start(1)
        connection.start()
        try:
            self.assertEqual(connection.request("ping", {}, timeout=2), {"ok": 1})
        finally:
            connection.close()

    def test_timeout_sends_cancellation(self):
        transport, connection = self._client([])
        connection.transport.start(1)
        connection.start()
        try:
            with self.assertRaises(m.MCPTimeout):
                connection.request("slow", {}, timeout=0.2)
            methods = [msg.get("method") for msg in transport.sent]
            self.assertIn("notifications/cancelled", methods)
        finally:
            connection.close()

    def test_protocol_error_raises(self):
        _, connection = self._client(
            [{"jsonrpc": "2.0", "error": {"code": -32601, "message": "nope"}}])
        connection.transport.start(1)
        connection.start()
        try:
            with self.assertRaises(m.MCPProtocolError) as caught:
                connection.request("missing", {}, timeout=2)
            self.assertEqual(caught.exception.code, -32601)
        finally:
            connection.close()

    def test_server_request_is_answered(self):
        transport, connection = self._client(
            [{"jsonrpc": "2.0", "id": "srv-1", "method": "ping"},
             {"jsonrpc": "2.0", "result": {"done": True}}])
        connection.on_request = lambda msg: {} if msg.get("method") == "ping" else None
        connection.transport.start(1)
        connection.start()
        try:
            connection.request("tools/list", {}, timeout=2)
        finally:
            connection.close()
        replies = [msg for msg in transport.sent if msg.get("id") == "srv-1"]
        self.assertTrue(replies, transport.sent)
        self.assertEqual(replies[0]["result"], {})

    def test_unsupported_server_request_gets_method_not_found(self):
        transport, connection = self._client(
            [{"jsonrpc": "2.0", "id": "srv-2", "method": "sampling/createMessage"},
             {"jsonrpc": "2.0", "result": {}}])
        connection.transport.start(1)
        connection.start()
        try:
            connection.request("tools/list", {}, timeout=2)
        finally:
            connection.close()
        replies = [msg for msg in transport.sent if msg.get("id") == "srv-2"]
        self.assertEqual(replies[0]["error"]["code"], m.MCP_ERR_METHOD_NOT_FOUND)


class TestFraming(unittest.TestCase):
    def test_garbage_tolerated_and_partial_tail(self):
        stream = io.BytesIO(b'banner\n{"a":1}\n[1,2]\n{"b":2}')
        self.assertEqual(list(m._mcp_iter_json_lines(stream)), [{"a": 1}, {"b": 2}])

    def test_oversize_frame_dropped_and_resync(self):
        payload = b'{"x":"' + b"a" * 5000 + b'"}\n{"after":1}\n'
        self.assertEqual(list(m._mcp_iter_json_lines(io.BytesIO(payload),
                                                     max_frame_bytes=1024)),
                         [{"after": 1}])

    def test_encode_frame_is_single_line(self):
        frame = m._mcp_encode_frame({"jsonrpc": "2.0", "id": 1})
        self.assertTrue(frame.endswith(b"\n"))
        self.assertEqual(frame.count(b"\n"), 1)


class TestConfig(unittest.TestCase):
    def test_explicit_missing_config_raises(self):
        with self.assertRaises(FileNotFoundError):
            m.load_mcp_config("/nonexistent/mcp.json")

    def test_example_config_is_valid(self):
        """The documented example shape parses; HTTP is the only refusal."""
        example = {
            "version": 1,
            "servers": {
                "kubernetes": {
                    "transport": "stdio",
                    "command": ["kubernetes-mcp-server", "--config", "/path/to/config.toml"],
                    "env": {"KUBECONFIG": "${KUBECONFIG:-/path/to/kubeconfig}"},
                    "startup": "lazy",
                    "tool_prefix": "k8s",
                    "denied_tools": ["pods_exec", "resources_delete"],
                },
                "kubernetes-http": {
                    "transport": "streamable-http",
                    "url": "http://127.0.0.1:10080/mcp",
                    "enabled": False,
                    "startup": "lazy",
                    "tool_prefix": "k8shttp",
                    "read_only_only": True,
                },
            },
        }
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "mcp.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(example, fh)
            config = m.load_mcp_config(path)
        self.assertIn("kubernetes", config.servers)
        problems = [problem for cfg in config.servers.values()
                    for problem in m.validate_mcp_server(cfg)]
        # The example ships an HTTP entry, which this build refuses explicitly.
        self.assertTrue(problems, "the HTTP example entry must be reported")
        self.assertTrue(all("not implemented" in problem for problem in problems),
                        problems)

    def test_unknown_keys_and_bad_startup_are_reported(self):
        cfg = m.MCPServerConfig(name="s", command=["x"], startup="lazyy",
                                tool_prefx="typo")
        problems = m.validate_mcp_server(cfg)
        self.assertTrue(any("startup" in problem for problem in problems))
        self.assertTrue(any("tool_prefx" in problem for problem in problems))

    def test_deny_and_confirm_overlap_reported(self):
        cfg = m.MCPServerConfig(name="s", command=["x"],
                                denied_tools=["pods_exec"], confirm_tools=["pods_exec"])
        self.assertTrue(any("deny wins" in problem
                            for problem in m.validate_mcp_server(cfg)))

    def test_claude_shape_is_accepted(self):
        cfg = m._mcp_coerce_server("s", {"command": "npx", "args": ["-y", "srv"],
                                         "type": "stdio"})
        self.assertEqual(cfg.command, ["npx", "-y", "srv"])
        self.assertEqual(cfg.transport, "stdio")
        self.assertEqual(m.validate_mcp_server(cfg), [])

    def _load_one(self, entry, config_name="mcp.json"):
        """Write a one-server config and return its parsed entry."""
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, config_name)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"mcpServers": {"srv": entry}}, fh)
            config = m.load_mcp_config(path)
            # Paths are absolute already; the temp dir may vanish afterwards.
            return config.servers["srv"], td

    def test_tilde_command_is_expanded(self):
        with tempfile.TemporaryDirectory() as home:
            os.makedirs(os.path.join(home, "Downloads"))
            server = os.path.join(home, "Downloads", "server")
            open(server, "w", encoding="utf-8").close()
            with patch.dict(os.environ, {"HOME": home}):
                cfg, _ = self._load_one({"command": "~/Downloads/server",
                                         "args": ["--stdio"]})
            self.assertEqual(cfg.command[0], server)
            self.assertEqual(cfg.command[1:], ["--stdio"])
            self.assertEqual(m.validate_mcp_server(cfg), [])

    def test_relative_command_anchored_to_config_dir(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "mcp.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"mcpServers": {"srv": {
                    "command": "./bin/server", "args": ["mcp"]}}}, fh)
            cfg = m.load_mcp_config(path).servers["srv"]
            self.assertEqual(cfg.command[0],
                             os.path.abspath(os.path.join(td, "bin", "server")))

    def test_bare_command_next_to_config_is_preferred(self):
        with tempfile.TemporaryDirectory() as td:
            local = os.path.join(td, "srv-binary")
            open(local, "w", encoding="utf-8").close()
            path = os.path.join(td, "mcp.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"mcpServers": {"srv": {
                    "command": "srv-binary", "args": []}}}, fh)
            cfg = m.load_mcp_config(path).servers["srv"]
            self.assertEqual(cfg.command[0], local)

    def test_vars_expand_in_args_and_env(self):
        cfg, _ = self._load_one({
            "command": "srv", "args": ["--connect", "{host}"],
            "vars": {"host": "127.0.0.1:10090"},
            "env": {"TARGET": "{host}"},
        })
        self.assertEqual(cfg.command[1:], ["--connect", "127.0.0.1:10090"])
        self.assertEqual(cfg.env, {"TARGET": "127.0.0.1:10090"})
        self.assertEqual(m.validate_mcp_server(cfg), [])

    def test_unknown_var_placeholder_is_left_verbatim(self):
        cfg, _ = self._load_one({"command": "srv", "args": ["{missing}"],
                                 "vars": {"host": "x"}})
        self.assertEqual(cfg.command[1:], ["{missing}"])

    def test_timeout_alias_sets_every_request_deadline(self):
        cfg, _ = self._load_one({"command": "srv", "timeout": 90})
        self.assertEqual((cfg.init_timeout_seconds, cfg.list_timeout_seconds,
                          cfg.call_timeout_seconds), (90.0, 90.0, 90.0))
        self.assertEqual(m.validate_mcp_server(cfg), [])

    def test_explicit_deadline_wins_over_timeout_alias(self):
        cfg, _ = self._load_one({"command": "srv", "timeout": 90,
                                 "init_timeout_seconds": 5})
        self.assertEqual(cfg.init_timeout_seconds, 5.0)
        self.assertEqual(cfg.call_timeout_seconds, 90.0)

    def test_bad_timeout_alias_is_reported(self):
        cfg, _ = self._load_one({"command": "srv", "timeout": "soon"})
        self.assertTrue(any("'timeout' must be a number" in problem
                            for problem in m.validate_mcp_server(cfg)))

    def test_env_reference_to_unset_variable_is_dropped(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MCP_TEST_UNSET_VAR", None)
            cfg, _ = self._load_one({"command": "srv",
                                     "env": {"KUBECONFIG": "${MCP_TEST_UNSET_VAR}"}})
            self.assertEqual(cfg.env, {})
        with patch.dict(os.environ, {"MCP_TEST_UNSET_VAR": "/tmp/kubeconfig"}):
            cfg, _ = self._load_one({"command": "srv",
                                     "env": {"KUBECONFIG": "${MCP_TEST_UNSET_VAR}"}})
            self.assertEqual(cfg.env, {"KUBECONFIG": "/tmp/kubeconfig"})

    def test_env_relative_path_anchored_to_config_dir(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "mcp.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"mcpServers": {"srv": {
                    "command": "srv",
                    "env": {"KUBECONFIG": "./tests/kubeconfig-fake.yaml"}}}}, fh)
            cfg = m.load_mcp_config(path).servers["srv"]
            self.assertEqual(cfg.env["KUBECONFIG"],
                             os.path.join(td, "tests", "kubeconfig-fake.yaml"))

    def test_known_extra_keys_are_not_reported(self):
        cfg, _ = self._load_one({
            "command": "srv", "dialect": "mcp", "description": "hello",
            "timeout": 30, "vars": {"host": "h"},
        })
        self.assertEqual(cfg.ignored_keys, [])
        self.assertEqual(m.validate_mcp_server(cfg), [])
        self.assertEqual(cfg.description, "hello")

    def test_unsupported_dialect_is_reported(self):
        cfg, _ = self._load_one({"command": "srv", "dialect": "raw"})
        self.assertTrue(any("dialect 'raw' is not implemented" in problem
                            for problem in m.validate_mcp_server(cfg)))

    def test_cli_flags(self):
        parser = m._build_parser()
        args = parser.parse_args(["--mcp-config", "/tmp/x.json", "--no-mcp"])
        self.assertEqual(args.mcp_config, "/tmp/x.json")
        self.assertTrue(args.no_mcp)


class TestListingAndCommand(unittest.TestCase):
    def test_status_columns_align_across_servers(self):
        config = m.MCPConfig({
            "a": m.MCPServerConfig(name="a", command=["x"]),
            "a-very-long-server-name": m.MCPServerConfig(
                name="a-very-long-server-name", command=["x"]),
        })
        lines = m.MCPManager(config).status_lines()
        self.assertEqual(len(lines), 2)
        # The short name is padded to the 30-column minimum.
        self.assertEqual(lines[0][:m.MCP_STATUS_NAME_WIDTH], "a" + " " * 29)
        for marker in ("stdio", "not connected", "0 tools", "0 plan", "0 hidden"):
            self.assertEqual(lines[0].index(marker), lines[1].index(marker), marker)

    def test_status_name_column_widens_for_very_long_names(self):
        long_name = "s" * 40
        config = m.MCPConfig({
            "short": m.MCPServerConfig(name="short", command=["x"]),
            long_name: m.MCPServerConfig(name=long_name, command=["x"]),
        })
        lines = m.MCPManager(config).status_lines()
        self.assertEqual(lines[0].index("stdio"), lines[1].index("stdio"))
        self.assertEqual(lines[0].index("stdio"), 41)

    def test_status_count_columns_align(self):
        config = m.MCPConfig({
            "a": m.MCPServerConfig(name="a", command=["x"]),
            "b": m.MCPServerConfig(name="b", command=["x"]),
        })
        manager = m.MCPManager(config)
        entry = types.SimpleNamespace(plan_eligible=True, filtered_reason=None)
        manager.classified_tools = lambda: {"a": [], "b": [entry] * 12}
        lines = manager.status_lines()
        for marker in ("tools", "plan", "hidden"):
            self.assertEqual(lines[0].index(marker), lines[1].index(marker), marker)

    def setUp(self):
        tools = [
            {"name": "pods_list", "description": "List pods",
             "inputSchema": {"type": "object", "properties": {}},
             "annotations": {"readOnlyHint": True, "destructiveHint": False}},
            {"name": "pods_delete", "description": "Delete a pod",
             "inputSchema": {"type": "object", "properties": {}},
             "annotations": {"readOnlyHint": False, "destructiveHint": True}},
            {"name": "config_view", "description": "View config",
             "inputSchema": {"type": "object", "properties": {}}},
        ]
        self.cfg = m.MCPServerConfig(name="k8s", command=["x"],
                                     denied_tools=["pods_delete"])
        self.catalogs = {"k8s": m.classify_mcp_tools(self.cfg, tools)}

    def test_denied_shown_with_reason(self):
        text = "\n".join(m.format_mcp_tool_listing(self.catalogs))
        self.assertIn("k8s__pods_delete", text)
        self.assertIn("[denied_tools]", text)

    def test_plan_tag_always_present(self):
        for view in m.MCP_LISTING_VIEWS:
            text = "\n".join(m.format_mcp_tool_listing(self.catalogs, view))
            self.assertIn("[plan]", text, view)

    def test_undeclared_is_not_plan_eligible(self):
        self.assertEqual(m.plan_mcp_tool_names(self.catalogs), {"k8s__pods_list"})

    def test_deterministic(self):
        self.assertEqual(m.format_mcp_tool_listing(self.catalogs),
                         m.format_mcp_tool_listing(self.catalogs))

    def test_handler_prints_listing(self):
        fake_self = types.SimpleNamespace(ctx=_ctx(mcp_manager=ManagerStub(
            self.catalogs)))
        with patch("sys.stderr", new=io.StringIO()) as captured:
            result = m.ChatLoop.run_handle_mcp(fake_self, "/mcp tools --plan")
        self.assertIs(result, False)
        self.assertIn("k8s__pods_list", captured.getvalue())

    def test_handler_without_manager_hints(self):
        fake_self = types.SimpleNamespace(ctx=_ctx())
        with patch("sys.stderr", new=io.StringIO()) as captured:
            m.ChatLoop.run_handle_mcp(fake_self, "/mcp")
        self.assertIn("no servers configured", captured.getvalue())

    def test_handler_ignores_other_commands(self):
        fake_self = types.SimpleNamespace(ctx=_ctx())
        self.assertIsNone(m.ChatLoop.run_handle_mcp(fake_self, "/listtool"))


class TestSpill(unittest.TestCase):
    def test_oversized_result_spills_and_registers_acl(self):
        class Acl:
            def __init__(self):
                self.calls = []
            def add(self, op, action, path, source="session"):
                self.calls.append((op, action, path))

        ctx = _ctx()
        ctx.path_acl = Acl()
        cfg = m.MCPServerConfig(name="k8s", command=["x"], max_result_chars=10)
        result = m._mcp_result_to_output(
            ctx, {"content": [{"type": "text", "text": "x" * 200}]}, cfg, "pods_list")
        self.assertIn("read_file(file_path=", result["output"])
        self.assertIn(("read", "allow", os.path.realpath(m.CLUSTER_SPILL_DIR)),
                      [(op, action, os.path.realpath(path))
                       for op, action, path in ctx.path_acl.calls])
        m._cleanup_spill_files()
        leftovers = [name for name in os.listdir(m.CLUSTER_SPILL_DIR)
                     if name.startswith(m.MCP_SPILL_PREFIX)] if os.path.isdir(
                         m.CLUSTER_SPILL_DIR) else []
        self.assertEqual(leftovers, [])

    def test_cap_follows_context_window(self):
        self.assertEqual(m._mcp_resolve_max_result_chars(_ctx(), 100000), 100000)
        self.assertEqual(m._mcp_resolve_max_result_chars(
            _ctx(context_window_size=8000), 100000), 8000)
        self.assertEqual(m._mcp_resolve_max_result_chars(
            _ctx(context_window_size=100), 100000), 2000)

    def test_configured_cap_is_a_ceiling_on_a_live_window(self):
        """A tiny configured cap must not be raised to the 2k budget floor.

        Regression: `max(2000, min(configured, budget))` turned a configured
        10 into 2000 as soon as the window was known, so results that should
        have spilled were returned inline. `_ctx()` defaults to
        context_window_size=0, which short-circuits to `configured`, so the
        spill test alone could not catch this.
        """
        for window in (100, 8182, 8192, 32768, 131072):
            self.assertEqual(
                m._mcp_resolve_max_result_chars(_ctx(context_window_size=window), 10),
                10, msg=f"window={window}")

    def test_small_result_not_spilled_when_cap_is_small(self):
        """The effective cap decides: a 200-char result spills under a 10 cap."""
        ctx = _ctx(context_window_size=32768)
        cfg = m.MCPServerConfig(name="k8s", command=["x"], max_result_chars=10)
        with patch.object(m, "CLUSTER_SPILL_DIR", tempfile.mkdtemp()), \
             patch.object(m, "_cleanup_spill_files", lambda: None):
            result = m._mcp_result_to_output(
                ctx, {"content": [{"type": "text", "text": "y" * 200}]}, cfg, "pods_list")
        self.assertIn("read_file(file_path=", result["output"])


class TestToolListingLayout(unittest.TestCase):
    """`/listtool` renders aligned columns with hanging-indented descriptions."""

    def _registry(self):
        long_desc = "Get detailed resource usage statistics from a node. " * 6
        defs = {
            "short_tool": {
                "description": "Small description.",
                "parameters": {"type": "object", "properties": {}},
            },
            "kubernetes__nodes_stats_summary": {
                "description": f"[mcp:kubernetes] {long_desc}",
                "parameters": {"type": "object", "properties": {}},
                "_mcp": {"server": "kubernetes", "tool": "nodes_stats_summary"},
            },
        }
        manager = FakeManager(plan_names=(), destructive=(), defs=defs)
        return m.ToolRegistry(_ctx(mcp_manager=manager))

    def test_origin_tag_is_rendered_once(self):
        self.assertEqual(self._registry().list_tools_str().count("[mcp:kubernetes]"), 1)

    def test_columns_align_and_continuations_indent(self):
        lines = self._registry().list_tools_str().splitlines()
        column = len("kubernetes__nodes_stats_summary") + 4
        short = next(line for line in lines if line.startswith("  short_tool"))
        k8s = next(line for line in lines
                   if line.startswith("  kubernetes__nodes_stats_summary"))
        self.assertEqual(short.index("Small description."), column)
        self.assertEqual(k8s.index("[mcp:kubernetes]"), column)
        continuations = [line for line in lines if line[:column].strip() == ""]
        self.assertTrue(continuations, "long descriptions must wrap")
        for line in continuations:
            self.assertEqual(len(line) - len(line.lstrip()), column, repr(line))

    def test_wraps_instead_of_running_to_terminal_edge(self):
        lines = self._registry().list_tools_str().splitlines()
        self.assertTrue(all(len(line) <= m.TOOL_LIST_MAX_WIDTH for line in lines))
        wrapped = [line for line in lines
                   if "resource usage statistics" in line or "from a node." in line]
        self.assertGreater(len(wrapped), 2)


class TestMCPUsageAndCompletion(unittest.TestCase):
    """Bare `/mcp` shows help, and tab completion knows subcommands + servers."""

    def _run_handler(self, line, manager=None):
        fake_self = types.SimpleNamespace(ctx=_ctx(mcp_manager=manager))
        with patch("sys.stderr", new=io.StringIO()) as captured:
            m.ChatLoop.run_handle_mcp(fake_self, line)
        return captured.getvalue()

    def _completer(self, manager=None):
        return m.ChatCompleter("http://localhost:11434", "ollama",
                               ctx=_ctx(mcp_manager=manager))

    def _candidates(self, completer, buffer, text):
        with patch.object(m.readline, "get_line_buffer", return_value=buffer):
            out = []
            for state in range(64):
                candidate = completer.complete(text, state)
                if candidate is None:
                    break
                out.append(candidate)
        return out

    def test_bare_mcp_shows_status_and_usage(self):
        output = self._run_handler("/mcp", ManagerStub({"k8s": []}))
        self.assertIn("MCP servers", output)
        self.assertIn("Usage: /mcp", output)
        self.assertIn("connect <server>", output)

    def test_mcp_help_lists_every_subcommand(self):
        output = self._run_handler("/mcp help")
        for sub in m.MCP_SUBCOMMANDS:
            self.assertIn(sub, output)
        self.assertNotIn("no servers configured", output)

    def test_unknown_subcommand_shows_usage(self):
        output = self._run_handler("/mcp aaa", ManagerStub({"k8s": []}))
        self.assertIn("unknown command: aaa", output)
        self.assertIn("Usage: /mcp", output)

    def test_completion_offers_subcommands(self):
        candidates = self._candidates(self._completer(), "/mcp ", "")
        for sub in m.MCP_SUBCOMMANDS:
            self.assertIn(sub, candidates)

    def test_completion_offers_server_names(self):
        manager = types.SimpleNamespace(config=types.SimpleNamespace(
            enabled=lambda: [types.SimpleNamespace(name="alpha"),
                             types.SimpleNamespace(name="beta")]))
        completer = self._completer(manager)
        self.assertEqual(self._candidates(completer, "/mcp connect ", ""),
                         ["alpha", "beta"])
        self.assertEqual(self._candidates(completer, "/mcp connect a", "a"),
                         ["alpha"])
        self.assertEqual(self._candidates(completer, "/mcp disconnect ", ""),
                         ["alpha", "beta"])

    def test_completion_offers_tool_flags(self):
        self.assertEqual(self._candidates(self._completer(), "/mcp tools -", "-"),
                         ["--plan", "--all"])


class TestPlanModeSurface(unittest.TestCase):
    """Plan mode advertises the union of base and read-only MCP tools."""

    def _loop(self, manager):
        loop = object.__new__(m.ChatLoop)
        loop.ctx = _ctx(mcp_manager=manager, plan_mode=True, plan_run=False,
                        agentic_mode=True, model="qwen3.8", system_prompt="BASE",
                        _plan_prev_prompt=None, _saved_plan_prompt=None)
        return loop

    def test_status_lists_read_only_mcp_tools(self):
        loop = self._loop(FakeManager())
        with patch("sys.stderr", new=io.StringIO()) as captured:
            loop._print_plan_mode_status()
        output = captured.getvalue()
        self.assertIn("Tools restricted to:", output)
        self.assertIn("read-only MCP tool", output)
        self.assertIn("fake", output)
        self.assertIn("pods_list", output)
        self.assertNotIn("pods_delete", output)
        self.assertNotIn("fake__", output)

    def test_no_mcp_tools_adds_no_line(self):
        loop = self._loop(None)
        with patch("sys.stderr", new=io.StringIO()) as captured:
            loop._print_plan_mode_status()
        self.assertNotIn("MCP tool", captured.getvalue())

    def test_banner_lists_the_real_tier_surface(self):
        """The banner must not drift from PLAN_MODE_TOOLS.

        It was hardcoded and omitted `git_inspect` (added in v0.3.4), so plan mode
        looked like it had no git access while it did. Every tier tool has to be
        named, and the run tier has to show its extra verb.
        """
        strict = self._loop(None)
        with patch("sys.stderr", new=io.StringIO()) as captured:
            strict._print_plan_mode_status()
        strict_out = captured.getvalue()
        for name in m.PLAN_MODE_TOOLS:
            self.assertIn(name, strict_out)
        self.assertIn("git_inspect", strict_out)  # the name the banner used to omit
        self.assertNotIn("run_command", strict_out)

        run = self._loop(None)
        run.ctx.plan_run = True
        with patch("sys.stderr", new=io.StringIO()) as captured:
            run._print_plan_mode_status()
        run_out = captured.getvalue()
        for name in m.PLAN_MODE_RUN_TOOLS:
            self.assertIn(name, run_out)
        self.assertIn("confirmation", run_out)

    def test_asking_twice_prints_the_same_surface(self):
        loop = self._loop(FakeManager())
        with patch("sys.stderr", new=io.StringIO()):
            loop._agentic_toggle_plan(['/agentic', 'plan'])
        with patch("sys.stderr", new=io.StringIO()) as captured:
            loop._agentic_toggle_plan(['/agentic', 'plan'])
        output = captured.getvalue()
        self.assertIn("Agentic plan mode: ON", output)
        self.assertIn("Tools restricted to:", output)
        self.assertIn("pods_list", output)


class ManagerStub:
    """Manager-shaped helper for the /mcp handler test."""

    def __init__(self, catalogs):
        self._catalogs = catalogs
        self.clients = {name: types.SimpleNamespace(connected=True)
                        for name in catalogs}
        self.config = types.SimpleNamespace(source_path="/tmp/mcp.json")

    def classified_tools(self):
        return self._catalogs

    def status_lines(self):
        return ["k8s  stdio  connected  3 tools"]


if __name__ == "__main__":
    unittest.main(verbosity=2)
