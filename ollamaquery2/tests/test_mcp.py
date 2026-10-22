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

    def test_cli_flags(self):
        parser = m._build_parser()
        args = parser.parse_args(["--mcp-config", "/tmp/x.json", "--no-mcp"])
        self.assertEqual(args.mcp_config, "/tmp/x.json")
        self.assertTrue(args.no_mcp)


class TestListingAndCommand(unittest.TestCase):
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
