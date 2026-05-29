#!/usr/bin/env python3
"""Tests for agentic mode: ReAct loop, tool execution, and parse_tool_call."""

import io
import os
import re
import sys
import json
import tempfile
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ollamaquery2 as q

OLLAMA_HOST = os.environ.get('OLLAMA_HOST', 'http://127.0.0.1:11434')
LLAMACPP_HOST = os.environ.get('LLAMACPP_HOST', 'http://127.0.0.1:8080')

# Auto-detect available backend
BACKEND = None
BASE_URL = None
MODEL = None
BACKEND_AVAILABLE = False


def _check_backend(url, label):
    """Try to reach a backend, return (available, model_name)."""
    # Normalize URL — add http:// if missing
    if not url.startswith(('http://', 'https://')):
        url = f"http://{url}"
    try:
        req = q.Request(url, method='HEAD')
        q.urlopen(req, timeout=2)
        # Try to get a loaded model
        try:
            tags = json.loads(q.urlopen(q.Request(f"{url}/api/tags"), timeout=3).read())
            models = [m['name'] for m in tags.get('models', [])]
            # Prefer models with "dns" capability, or pick first non-embedding
            for m in models:
                if 'embed' not in m.lower() and '3.5:9b' in m:
                    return True, m
            for m in models:
                if 'embed' not in m.lower():
                    return True, m
        except Exception:
            pass
        return True, None
    except Exception:
        return False, None


def setUpModule():
    """Detect available backend and pick a model."""
    global BACKEND, BASE_URL, MODEL, BACKEND_AVAILABLE

    avail, model = _check_backend(OLLAMA_HOST, "Ollama")
    if avail:
        BACKEND = "ollama"
        BASE_URL = OLLAMA_HOST
        MODEL = os.environ.get("TEST_MODEL") or model or "qwen3.5:9b"
        BACKEND_AVAILABLE = True
        return

    avail, model = _check_backend(LLAMACPP_HOST, "llama.cpp")
    if avail:
        BACKEND = "llamacpp"
        BASE_URL = LLAMACPP_HOST
        MODEL = os.environ.get("TEST_MODEL") or model or "google_gemma-4-E4B-it-Q4_K_M.gguf"
        BACKEND_AVAILABLE = True
        return


class TestParseToolCall(unittest.TestCase):
    """Test the JSON tool call parser."""

    def setUp(self):
        self.ctx = q.CommandContext()
        self.loop = q.ChatLoop(self.ctx)

    def test_parse_clean_json(self):
        text = '{"tool": "fetch_url", "arguments": {"url": "example.com"}}'
        result = self.loop.parse_tool_call(text)
        self.assertEqual(result, {"tool": "fetch_url", "arguments": {"url": "example.com"}})

    def test_parse_fenced_code_block(self):
        text = '```json\n{"tool": "read_file", "arguments": {"file": "test.py"}}\n```'
        result = self.loop.parse_tool_call(text)
        self.assertEqual(result, {"tool": "read_file", "arguments": {"file": "test.py"}})

    def test_parse_fenced_no_lang(self):
        text = '```\n{"tool": "write_file", "arguments": {"file": "x.py", "content": "print(1)"}}\n```'
        result = self.loop.parse_tool_call(text)
        self.assertEqual(result["tool"], "write_file")
        self.assertEqual(result["arguments"]["file"], "x.py")

    def test_parse_inline_in_text(self):
        text = 'I think I need to fetch the URL. {"tool": "fetch_url", "arguments": {"url": "http://example.com"}} Let me do that.'
        # Strict mode (lazy_tool=False): embedded calls should not be extracted
        self.ctx.lazy_tool = False
        result = self.loop.parse_tool_call(text)
        self.assertIsNone(result, "Embedded tool calls in text should not be extracted in strict mode")

    def test_parse_plain_text_returns_none(self):
        text = "The date today is May 21, 2026."
        result = self.loop.parse_tool_call(text)
        self.assertIsNone(result)

    def test_parse_empty_string_returns_none(self):
        self.assertIsNone(self.loop.parse_tool_call(""))
        self.assertIsNone(self.loop.parse_tool_call("   "))

    def test_parse_invalid_json_returns_none(self):
        text = '{"tool": "fetch_url" "missing": "comma"}'
        result = self.loop.parse_tool_call(text)
        self.assertIsNone(result)


class TestToolRegistry(unittest.TestCase):
    """Test ToolRegistry execution and confirmation."""

    def setUp(self):
        self.ctx = q.CommandContext()
        self.ctx.auto_confirm = True  # skip prompts in tests
        self.reg = q.ToolRegistry(ctx=self.ctx)

    def test_unknown_tool(self):
        result = self.reg.execute("nonexistent", {})
        self.assertFalse(result["success"])
        self.assertIn("Unknown tool", result["error"])

    def test_list_tools_str(self):
        output = self.reg.list_tools_str()
        self.assertIn("fetch_url", output)
        self.assertIn("write_file", output)
        self.assertIn("run_python", output)
        self.assertIn("! ", output)  # destructive marker

    def test_system_prompt_block(self):
        block = self.reg.get_system_prompt_block()
        self.assertIn("fetch_url", block)
        self.assertIn("write_file", block)
        self.assertIn("Available tools", block)
        self.assertIn("read_file", block)
        self.assertIn("run_python", block)
        self.assertNotIn("JSON tool call", block)  # format reminders are in format blocks now

    def test_read_file(self):
        """Test reading a file within CWD."""
        cwd = os.getcwd()
        testfile = os.path.join(cwd, ".agentic_test_read.tmp")
        try:
            with open(testfile, "w") as f:
                f.write("hello world")
            args = {"file": ".agentic_test_read.tmp"}
            result = self.reg.execute("read_file", args)
            self.assertTrue(result["success"], msg=result.get("error"))
            self.assertIn("hello world", result["output"])
        finally:
            if os.path.exists(testfile):
                os.unlink(testfile)

    def test_write_and_read_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                result = self.reg.execute("write_file", {
                    "file": "test.txt", "content": "agentic test"
                })
                self.assertTrue(result["success"])
                self.assertIn("Written", result["output"])
                self.assertTrue(os.path.exists("test.txt"))
                with open("test.txt") as f:
                    self.assertEqual(f.read(), "agentic test")
            finally:
                os.chdir(old_cwd)

    def test_list_directory(self):
        result = self.reg.execute("list_directory", {"path": "."})
        self.assertTrue(result["success"])
        self.assertIn("ollamaquery2.py", result["output"])  # main file

    def test_glob(self):
        result = self.reg.execute("glob", {"pattern": "*.py"})
        self.assertTrue(result["success"])

    def test_diff(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                with open("file1.txt", "w") as f1:
                    f1.write("line1\nline2\n")
                with open("file2.txt", "w") as f2:
                    f2.write("line1\nline3\n")
                result = self.reg.execute("diff", {"file1": "file1.txt", "file2": "file2.txt"})
                self.assertTrue(result["success"])
                self.assertIn("-line2", result["output"])
                self.assertIn("+line3", result["output"])
            finally:
                os.chdir(old_cwd)

    def test_patch_with_target(self):
        """patch tool with explicit target file."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                with open("test.txt", "w") as f:
                    f.write("hello\nworld\n")
                diff = "--- a/test.txt\n+++ b/test.txt\n@@ -1,2 +1,2 @@\n-hello\n+hi\n world\n"
                result = self.reg.execute("patch", {"diff": diff, "target": "test.txt"})
                self.assertTrue(result["success"], f"patch failed: {result.get('error')}")
                with open("test.txt") as f:
                    content = f.read()
                self.assertIn("hi", content)
            finally:
                os.chdir(old_cwd)

    def test_patch_without_target_uses_inline_flag(self):
        """patch tool without target must use -i (regression: cwd-as-target bug).

        When target is omitted, args.get('target', '') → '' → os.path.join(cwd, '')
        returns cwd itself, making target always truthy. The -i fallback was dead code.
        """
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                with open("test.txt", "w") as f:
                    f.write("hello\nworld\n")
                diff = "--- test.txt\n+++ test.txt\n@@ -1,2 +1,2 @@\n-hello\n+hi\n world\n"
                result = self.reg.execute("patch", {"diff": diff})
                self.assertTrue(result["success"], f"patch without target failed: {result.get('error')}")
                with open("test.txt") as f:
                    content = f.read()
                self.assertIn("hi", content)
            finally:
                os.chdir(old_cwd)

    def test_write_compile_run(self):
        """Full write_file → run_command(gcc) → run_command(test) pipeline."""
        c_code = (
            '#include <stdio.h>\n'
            'int main(int argc, char *argv[]) {\n'
            '    for (int i = 1; i < argc; i++) printf("%s\\n", argv[i]);\n'
            '    return 0;\n'
            '}\n'
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                wr = self.reg.execute("write_file", {
                    "file": "echo_args.c", "content": c_code
                })
                self.assertTrue(wr["success"], msg=wr.get("error"))

                cc = self.reg.execute("run_command", {
                    "command": "gcc -o echo_args echo_args.c"
                })
                self.assertTrue(cc["success"], msg=f"compile failed: {cc.get('error')}")
                self.assertTrue(os.path.exists("echo_args"))

                run = self.reg.execute("run_command", {
                    "command": "./echo_args hello world 42"
                })
                self.assertTrue(run["success"], msg=run.get("error"))
                self.assertIn("hello", run["output"])
                self.assertIn("world", run["output"])
                self.assertIn("42", run["output"])
            finally:
                os.chdir(old_cwd)


class TestRunPythonPath(unittest.TestCase):
    """run_python tool must handle file_path without host absolute paths."""

    def setUp(self):
        self.ctx = q.CommandContext()
        self.ctx.auto_confirm = True
        self.reg = q.ToolRegistry(ctx=self.ctx)

    def test_run_python_with_file_path_uses_direct_path(self):
        """run_python with file_path should use the path directly (regression: sandbox absolute path)."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                with open("test_script.py", "w") as f:
                    f.write("print('hello from script')")
                result = self.reg.execute("run_python", {"file_path": "test_script.py"})
                self.assertTrue(result["success"], msg=result.get("error", ""))
                self.assertIn("hello from script", result["output"])
            finally:
                os.chdir(old_cwd)

    def test_run_python_with_file_uses_direct_path(self):
        """run_python with 'file' alias should use the path directly."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                with open("script.py", "w") as f:
                    f.write("print('direct')")
                result = self.reg.execute("run_python", {"file": "script.py"})
                self.assertTrue(result["success"], msg=result.get("error", ""))
                self.assertIn("direct", result["output"])
            finally:
                os.chdir(old_cwd)


class TestExecutorContainerModeSafety(unittest.TestCase):
    """Container mode must bypass host-side safety validator."""

    def test_container_mode_passes_semicolon_commands(self):
        """Container mode commands with ; should not be rejected by safety validator."""
        exec = q.Executor(mode="container")
        with patch.object(exec, '_run_shell') as mock_shell:
            mock_shell.return_value = {"stdout": "ok", "stderr": "", "returncode": 0}
            result = exec.run("echo hello; echo world", timeout=5)
            self.assertEqual(result["returncode"], 0,
                             "Container mode should not reject multi-statement commands")
            # Verify _run_shell was called with is_container=True
            _, kwargs = mock_shell.call_args
            self.assertTrue(kwargs.get('is_container', False),
                            "_run_shell should be called with is_container=True in container mode")


class TestAgenticStreamObservations(unittest.TestCase):
    """Streaming tool observations must be persisted in self.messages."""

    def setUp(self):
        self.ctx = q.CommandContext()
        self.ctx.auto_confirm = True
        self.ctx.base_url = "http://localhost:9999"
        self.ctx.backend = "ollama"
        self.ctx.model = "test-model"
        self.loop = q.ChatLoop(self.ctx)

    def test_finalize_appends_observations_for_stream_tool_calls(self):
        """_finalize_agentic_query must append tool observations after stream tool calls."""
        self.loop.messages = []
        messages = [{'role': 'system', 'content': 'test'}]
        # Mock query_stream to populate tool_calls_out
        with patch.object(self.loop.query_handler, 'query_stream',
                          return_value="") as mock_stream:
            mock_stream.return_value = ""
            def side_effect(*args, **kwargs):
                tool_calls_out = kwargs.get('tool_calls_out')
                if tool_calls_out is not None:
                    tool_calls_out.append({
                        "id": "call_1", "type": "function",
                        "function": {"name": "fetch_url", "arguments": '{"url": "http://example.com"}'}
                    })
                return ""
            mock_stream.side_effect = side_effect

            with patch.object(self.loop.tool_registry, 'execute',
                              return_value={"success": True, "output": "Fetched example.com", "error": None}):
                self.loop._finalize_agentic_query(
                    messages, final_answer="Hello world", final_content="test content",
                    send_tools_api=False, openai_tools=[], logger=None,
                    iteration=1, response_text=""
                )

        user_msgs = [m for m in self.loop.messages if m['role'] == 'user']
        tool_result_msgs = [m for m in user_msgs if 'Tool result' in m.get('content', '')]
        self.assertGreaterEqual(len(tool_result_msgs), 1,
                                "Tool observations not appended to self.messages")
        self.assertIn("fetch_url", tool_result_msgs[0]['content'],
                      "Tool result content missing tool name")


class TestExecutor(unittest.TestCase):
    """Test Executor (host mode only — no container runtime required)."""

    def setUp(self):
        self.exec = q.Executor(mode="host")

    def test_echo(self):
        result = self.exec.run("echo hello", timeout=5)
        self.assertEqual(result["stdout"].strip(), "hello")
        self.assertEqual(result["returncode"], 0)

    def test_failure(self):
        result = self.exec.run("false", timeout=5)
        self.assertNotEqual(result["returncode"], 0)

    def test_timeout(self):
        result = self.exec.run("sleep 10", timeout=1)
        self.assertNotEqual(result["returncode"], 0)

    def test_safety_blocklist(self):
        result = self.exec.run("rm -rf /", timeout=5)
        self.assertIn("rejected", result["stderr"])

    def test_run_python_with_semicolon(self):
        """run_python with semicolons in code must not be rejected by safety validator (regression)."""
        command = "python3 -c 'import sys; print(sys.version)'"
        result = self.exec.run(command, timeout=5)
        self.assertEqual(result["returncode"], 0,
                         "Semicolons inside quoted python -c should not trigger safety rejection")
        self.assertIn('.', result["stdout"],
                      "Should print Python version")

    def test_run_python_with_pipe_in_string(self):
        """run_python with pipe character inside a string literal must not be rejected."""
        command = "python3 -c \"import re; print(re.match('a|b', 'a').group())\""
        result = self.exec.run(command, timeout=5)
        self.assertEqual(result["returncode"], 0,
                         "Pipe chars inside quoted python -c should pass safety validator")
        self.assertIn('a', result["stdout"])

    def test_run_python_with_semicolon_multiple_statements(self):
        """Multi-statement python code with semicolons must execute."""
        command = "python3 -c 'x=1; y=2; print(x + y)'"
        result = self.exec.run(command, timeout=5)
        self.assertEqual(result["stdout"].strip(), "3")


class TestAgenticReActEndToEnd(unittest.TestCase):
    """End-to-end ReAct loop tests with a real LLM backend.

    Auto-detects Ollama or llama.cpp. Requires a loaded model capable of
    code generation and tool use (e.g. qwen3.5:9b, dolphin3:8b).
    All tests skip if no backend is reachable.
    """

    def setUp(self):
        if not BACKEND_AVAILABLE:
            self.skipTest(
                f"No backend available "
                f"(Ollama at {OLLAMA_HOST} / llama.cpp at {LLAMACPP_HOST})"
            )
        self.ctx = q.CommandContext()
        self.ctx.base_url = BASE_URL
        self.ctx.backend = BACKEND
        self.ctx.model = MODEL
        self.ctx.agentic_mode = True
        self.ctx.agentic_logging = False
        self.ctx.auto_confirm = True  # skip prompts
        self.ctx.agentic_verbose = True
        self.ctx.agentic_show_thinking = True
        self.ctx.agentic_trace = True
        self.ctx.lazy_tool = True
        self.ctx.agentic_max_iterations = 30
        self.loop = q.ChatLoop(self.ctx)

    def test_direct_answer_no_tool(self):
        """Query that should be answered directly without tools."""
        result = self.loop.run_agentic_query("Say hello in one word")
        # Should have an assistant response in messages
        self.assertTrue(hasattr(self.loop, 'messages'))
        self.assertGreater(len(self.loop.messages), 1)
        last = self.loop.messages[-1]
        self.assertEqual(last["role"], "assistant")
        self.assertIsInstance(last["content"], str)
        self.assertGreater(len(last["content"]), 0)

    def test_user_prompt_persisted_in_history(self):
        """Agentic mode must save user prompts to self.messages (regression: history amnesia).

        Without the fix, self.messages only gets assistant responses but never
        the user's prompt — on the next turn the model loses all user context.
        """
        self.loop.run_agentic_query("My name is Alex")
        # self.messages must contain the user prompt
        self.assertTrue(hasattr(self.loop, 'messages'))
        user_msgs = [m for m in self.loop.messages if m['role'] == 'user']
        self.assertGreaterEqual(len(user_msgs), 1,
                                "User prompt was not saved to self.messages")
        self.assertTrue(
            any('Alex' in m['content'] for m in user_msgs),
            "User message content 'Alex' not found in self.messages"
        )

    def test_multi_turn_user_history_preserved(self):
        """Multiple agentic turns preserve all user prompts in self.messages."""
        self.loop.run_agentic_query("My favorite color is blue")
        self.loop.run_agentic_query("My name is Bob")
        user_msgs = [m for m in self.loop.messages if m['role'] == 'user']
        self.assertGreaterEqual(len(user_msgs), 2,
                                "Not all user prompts saved across turns")
        contents = [m['content'] for m in user_msgs]
        self.assertTrue(
            any('blue' in c for c in contents),
            "First turn content lost from history"
        )
        self.assertTrue(
            any('Bob' in c for c in contents),
            "Second turn content lost from history"
        )

    def test_fetch_url_tool(self):
        """Query that requires fetch_url — fetch a known URL."""
        result = self.loop.run_agentic_query(
            "Fetch http://example.com and tell me what the page title is"
        )
        self.assertTrue(hasattr(self.loop, 'messages'))
        # Observations are now stored in history, so the last message may be a
        # tool result (user role). Verify an assistant response exists somewhere.
        assistant_msgs = [m for m in self.loop.messages if m['role'] == 'assistant']
        self.assertGreaterEqual(len(assistant_msgs), 1,
                                "No assistant response found in history")
        self.assertGreater(len(assistant_msgs[-1]['content']), 0)

    def test_multi_step_write_file(self):
        """Multi-step: write a file with write_file tool, then verify via read_file.
        
        Uses explicit instructions to force actual tool use rather than simulation.
        """
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                self.loop.run_agentic_query(
                    "Write a file called 'test.txt' with content 'Hello Agentic' using the write_file tool. "
                    "Then read it back using the read_file tool. "
                    "IMPORTANT: You MUST use the write_file and read_file tools, do NOT simulate."
                )
                self.assertTrue(hasattr(self.loop, 'messages'))
                # Observations are now stored, so the last message may be a
                # tool result. Check that an assistant response exists.
                assistant_msgs = [m for m in self.loop.messages if m['role'] == 'assistant']
                self.assertGreaterEqual(len(assistant_msgs), 1,
                                        "No assistant response found in history")
                self.assertGreater(len(assistant_msgs[-1]['content']), 0)
            finally:
                os.chdir(old_cwd)

    def test_dns_resolver_write_compile_run(self):
        """End-to-end: LLM writes dns_resolver.c, compiles it, and tests it.

        Uses a temporary directory for isolation. Verifies the binary exists
        and can be invoked after the agentic session completes.
        """
        import tempfile, os
        query = (
            "write a dns_resolver.c, compile it, then test it. "
            "This dns_resolve will take two arguments: <dnsserverip> and <FQDN> "
            "and the tool will ask to the <dnsserverip>:53 using udp the dns query "
            "and give back the IPv4 address of the resolution."
        )
        self.ctx.lazy_tool = True
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                self.loop.run_agentic_query(query)
                self.assertTrue(
                    os.path.exists("dns_resolver"),
                    msg="dns_resolver binary was not created by the agentic workflow"
                )
            finally:
                os.chdir(old_cwd)

    def test_port_scanner_write_compile_run(self):
        """End-to-end: LLM writes a simple TCP port scanner, compiles, and tests it.

        Writes a C program that takes an IP and port, does a TCP connect,
        prints 'OPEN' or 'CLOSED'. Compiles with gcc, tests against
        localhost:22 (SSH, usually open) and localhost:19999 (expected closed).
        """
        import tempfile
        query = (
            "Write a C program called 'portscanner' that takes two arguments: "
            "<IP address> and <port number>. "
            "It attempts a TCP connect() to that IP:port and prints "
            "'PORT <port> is OPEN' or 'PORT <port> is CLOSED'. "
            "Save it as portscanner.c, compile it with gcc -o portscanner portscanner.c, "
            "then test it twice: "
            "first against 127.0.0.1:22 (SSH, should be OPEN), "
            "then against 127.0.0.1:19999 (should be CLOSED)."
        )
        self.ctx.lazy_tool = True
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                self.loop.run_agentic_query(query)
                self.assertTrue(
                    os.path.exists("portscanner"),
                    msg="portscanner binary was not created by the agentic workflow"
                )
            finally:
                os.chdir(old_cwd)

    def test_web_server_write_compile_run(self):
        """End-to-end: LLM writes a simple HTTP web server, compiles, starts, and tests with curl.

        Writes a C program that listens on a given port, responds with
        'Hello World' to any HTTP GET. Compiles, starts in background,
        curls it, then reports the response.
        """
        import tempfile
        query = (
            "Write a C program called 'webserver' that takes one argument <port>. "
            "It binds to 0.0.0.0, listens, accepts ONE connection, "
            "reads the HTTP request, responds with "
            "'HTTP/1.1 200 OK\\r\\nContent-Length: 12\\r\\n\\r\\nHello World\\n', "
            "then closes and exits. "
            "Save it as webserver.c, compile it with gcc -o webserver webserver.c. "
            "Then use run_python to test it: start webserver in background "
            "with subprocess.Popen on port 18999, "
            "curl http://127.0.0.1:18999 with urllib.request, "
            "print the response body, then kill the server."
        )
        self.ctx.lazy_tool = True
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                self.loop.run_agentic_query(query)
                self.assertTrue(
                    os.path.exists("webserver"),
                    msg="webserver binary was not created by the agentic workflow"
                )
            finally:
                os.chdir(old_cwd)

    def test_directory_lister_write_compile_run(self):
        """End-to-end: LLM writes a program that lists directory contents, compiles, and tests.

        Writes a C program that takes a directory path as argument and
        prints all file names (one per line) in that directory.
        Compiles with gcc, tests against the current directory.
        """
        import tempfile
        query = (
            "Write a C program called 'dirlister' that takes one argument: <directory path>. "
            "It opens the directory using opendir(), reads entries with readdir(), "
            "and prints each entry name on its own line. "
            "If no argument given, default to current directory '.'. "
            "Save it as dirlister.c, compile with gcc -o dirlister dirlister.c, "
            "then test it by listing the contents of the current directory "
            "and verifying that files like 'dirlister.c' and 'portscanner.c' appear."
        )
        self.ctx.lazy_tool = True
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                self.loop.run_agentic_query(query)
                self.assertTrue(
                    os.path.exists("dirlister"),
                    msg="dirlister binary was not created by the agentic workflow"
                )
            finally:
                os.chdir(old_cwd)

    def test_path_traversal_is_blocked(self):
        """LLM must be blocked from reading files outside CWD via read_file."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                self.loop.run_agentic_query(
                    "Read the file ../etc/passwd using the read_file tool "
                    "and tell me what's in it. "
                    "IMPORTANT: Actually use the read_file tool, do not simulate."
                )
                self.assertTrue(hasattr(self.loop, 'messages'))
                last = self.loop.messages[-1]
                self.assertEqual(last["role"], "assistant")
                # The assistant should report that the path was denied,
                # not return actual sensitive content
                content = last.get("content", "").lower()
                self.assertNotIn("root:", content,
                                 msg="Assistant leaked /etc/passwd content despite path traversal guard")
            finally:
                os.chdir(old_cwd)


class TestReActLoopUnit(unittest.TestCase):
    """Deterministic unit tests for the ReAct loop logic using mocked LLM."""

    def setUp(self):
        self.ctx = q.CommandContext()
        self.ctx.base_url = "http://test:8080"
        self.ctx.backend = "llamacpp"
        self.ctx.model = "test-model"
        self.ctx.agentic_mode = True
        self.ctx.auto_confirm = True
        self.loop = q.ChatLoop(self.ctx)

    def _make_sync_response(self, content):
        """Create a mock llama.cpp sync response."""
        return {
            "choices": [{"message": {"role": "assistant", "content": content}}]
        }

    def test_tool_call_triggers_execution(self):
        """When LLM returns a tool call JSON, the tool is executed and observation appended."""
        # Mock query_sync to return: tool call → final answer
        calls = [
            self._make_sync_response(
                '{"tool": "write_file", "arguments": {"file": "hello.txt", "content": "world"}}'
            ),
            self._make_sync_response("The file was written successfully."),
        ]
        call_idx = [0]

        def mock_sync(*args, **kwargs):
            idx = call_idx[0]
            call_idx[0] += 1
            return calls[idx] if idx < len(calls) else self._make_sync_response("done")

        self.loop.query_handler.query_sync = mock_sync
        self.loop.query_handler.query_stream = MagicMock(return_value="The file was written successfully.")

        self.loop.run_agentic_query("Create a file with hello world")

        # Should have assistant message at the end
        last = self.loop.messages[-1]
        self.assertEqual(last["role"], "assistant")
        self.assertIn("successfully", last["content"])

    def test_tool_call_cancelled_aborts(self):
        """When user cancels a destructive tool, the loop aborts cleanly."""
        self.ctx.auto_confirm = False  # Re-enable confirmation

        calls = [
            self._make_sync_response(
                '{"tool": "write_file", "arguments": {"file": "x.txt", "content": "data"}}'
            ),
        ]
        call_idx = [0]

        def mock_sync(*args, **kwargs):
            idx = call_idx[0]
            call_idx[0] += 1
            return calls[idx] if idx < len(calls) else self._make_sync_response("done")

        self.loop.query_handler.query_sync = mock_sync
        self.loop.query_handler.query_stream = MagicMock(return_value="")

        # Mock input() to say 'n' (cancel)
        with patch("builtins.input", return_value="n"):
            self.loop.run_agentic_query("Create a file")

        # On cancellation, no new assistant message is appended
        # (only the original system prompt message exists)
        if hasattr(self.loop, 'messages'):
            assistant_msgs = [m for m in self.loop.messages if m["role"] == "assistant"]
            self.assertEqual(len(assistant_msgs), 0,
                             "No assistant message should be added on cancellation")

    def test_max_iterations_reached(self):
        """When loop hits max_iterations, it stops and returns last response."""
        tool_response = self._make_sync_response(
            '{"tool": "write_file", "arguments": {"file": "x.txt", "content": "y"}}'
        )

        def mock_sync(*args, **kwargs):
            return tool_response

        self.loop.query_handler.query_sync = mock_sync
        self.loop.query_handler.query_stream = MagicMock(return_value="")

        self.loop.run_agentic_query("Keep using tools")

        self.assertTrue(hasattr(self.loop, 'messages'))
        # Should have terminated, no crash

    def test_parse_tool_call_with_real_usage(self):
        """Real parse_tool_call usage within the loop."""
        self.loop.query_handler.query_sync = MagicMock(
            side_effect=[
                self._make_sync_response(
                    '{"tool": "read_file", "arguments": {"file": "nonexistent.txt"}}'
                ),
                self._make_sync_response("File does not exist.")
            ]
        )
        self.loop.query_handler.query_stream = MagicMock(return_value="File does not exist.")

        self.loop.run_agentic_query("Read a file")

        last = self.loop.messages[-1] if hasattr(self.loop, 'messages') else {"role": "system", "content": ""}
        self.assertEqual(last["role"], "assistant")


class TestStuckDetection(unittest.TestCase):
    """Test the _is_stuck and _call_with_timeout helpers."""

    def setUp(self):
        self.ctx = q.CommandContext()
        self.loop = q.ChatLoop(self.ctx)

    def test_is_stuck_normal_text(self):
        """Diverse text should not be detected as stuck."""
        text = "The quick brown fox jumps over the lazy dog. " * 20
        self.assertFalse(self.loop._is_stuck(text))

    def test_is_stuck_repetitive(self):
        """Highly repetitive text should be detected as stuck."""
        # Repeated identical 50-char chunk pattern
        chunk = "A" * 50
        text = chunk * 20
        self.assertTrue(self.loop._is_stuck(text))

    def test_is_stuck_identical_chars(self):
        """Identical character sequences should be detected as stuck."""
        text = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        text = text * 20
        self.assertTrue(self.loop._is_stuck(text))

    def test_is_stuck_short_text(self):
        """Short text (< 200 chars) should never be stuck."""
        self.assertFalse(self.loop._is_stuck("Hello world"))
        self.assertFalse(self.loop._is_stuck("A" * 50))

    def test_call_with_timeout_success(self):
        """Function completing before timeout returns normally."""
        result = self.loop._call_with_timeout(lambda x: x + 1, 5, 41)
        self.assertEqual(result, 42)

    def test_call_with_timeout_timeout(self):
        """Function exceeding timeout returns None."""
        def slow():
            import time
            time.sleep(10)
            return 42
        result = self.loop._call_with_timeout(slow, 1)
        self.assertIsNone(result)


class TestNormalizeToolJson(unittest.TestCase):
    """Test _normalize_tool_json with various formats."""

    def setUp(self):
        self.ctx = q.CommandContext()
        self.loop = q.ChatLoop(self.ctx)

    def test_internal_format(self):
        """Internal {'tool': ..., 'arguments': ...} passes through."""
        json_text = '{"tool": "write_file", "arguments": {"file": "test.c"}}'
        result = self.loop._normalize_tool_json(json_text)
        self.assertEqual(result, {"tool": "write_file", "arguments": {"file": "test.c"}})

    def test_openai_function_wrapper(self):
        """OpenAI format with 'function' wrapper is normalized."""
        json_text = '{"type": "function", "function": {"name": "run_command", "arguments": {"command": "ls"}}}'
        result = self.loop._normalize_tool_json(json_text)
        self.assertEqual(result["tool"], "run_command")
        self.assertEqual(result["arguments"]["command"], "ls")

    def test_openai_no_function_wrapper(self):
        """OpenAI format without 'function' wrapper is normalized."""
        json_text = '{"type": "function", "name": "write_file", "arguments": {"file": "test.c", "content": "int main(){}"}}'
        result = self.loop._normalize_tool_json(json_text)
        self.assertEqual(result["tool"], "write_file")
        self.assertEqual(result["arguments"]["file"], "test.c")

    def test_compact_format(self):
        """Compact format {'function': {'name': ...}} is normalized."""
        json_text = '{"function": {"name": "list_directory", "arguments": {"path": "."}}}'
        result = self.loop._normalize_tool_json(json_text)
        self.assertEqual(result["tool"], "list_directory")
        self.assertEqual(result["arguments"]["path"], ".")

    def test_arguments_as_json_string(self):
        """OpenAI sometimes encodes arguments as a JSON string."""
        json_text = '{"type": "function", "function": {"name": "write_file", "arguments": "{\\"file\\": \\"x.c\\", \\"content\\": \\"int main(){}\\"}"}}'
        result = self.loop._normalize_tool_json(json_text)
        self.assertEqual(result["tool"], "write_file")
        self.assertEqual(result["arguments"]["file"], "x.c")

    def test_not_a_tool_call(self):
        """Plain text or non-tool JSON returns None."""
        self.assertIsNone(self.loop._normalize_tool_json("plain text"))
        self.assertIsNone(self.loop._normalize_tool_json('{"not": "a tool"}'))


class TestParseToolCalls(unittest.TestCase):
    """Test multi-tool extraction via parse_tool_calls."""

    def setUp(self):
        self.ctx = q.CommandContext()
        self.ctx.lazy_tool = False  # strict mode by default
        self.loop = q.ChatLoop(self.ctx)

    def test_single_tool_strict(self):
        """Single tool call at the start is found in strict mode."""
        text = '{"tool": "write_file", "arguments": {"file": "a.c", "content": "x"}}'
        result = self.loop.parse_tool_calls(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["tool"], "write_file")

    def test_multiple_tools_strict(self):
        """Multiple consecutive tool calls are extracted in strict mode."""
        text = ('{"tool": "write_file", "arguments": {"file": "a.c", "content": "x"}}'
                '{"tool": "run_command", "arguments": {"command": "gcc a.c"}}')
        result = self.loop.parse_tool_calls(text)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["tool"], "write_file")
        self.assertEqual(result[1]["tool"], "run_command")

    def test_multiple_tools_with_whitespace(self):
        """Whitespace between consecutive tool calls is allowed."""
        text = ('{"tool": "write_file", "arguments": {"file": "a.c", "content": "x"}}\n\n'
                '{"tool": "run_command", "arguments": {"command": "gcc a.c"}}')
        result = self.loop.parse_tool_calls(text)
        self.assertEqual(len(result), 2)

    def test_strict_rejects_embedded(self):
        """Embedded tool calls (text before JSON) are rejected in strict mode."""
        text = 'First let me check. {"tool": "list_directory", "arguments": {"path": "."}}'
        result = self.loop.parse_tool_calls(text)
        self.assertEqual(len(result), 0)

    def test_lazy_finds_embedded(self):
        """Lazy mode finds tool calls anywhere in the text."""
        self.ctx.lazy_tool = True
        text = ('I need to write a file first.\n'
                '{"tool": "write_file", "arguments": {"file": "a.c", "content": "x"}}\n'
                'Then compile it:\n'
                '{"tool": "run_command", "arguments": {"command": "gcc a.c"}}')
        result = self.loop.parse_tool_calls(text)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["tool"], "write_file")
        self.assertEqual(result[1]["tool"], "run_command")

    def test_lazy_deduplicates(self):
        """Lazy mode deduplicates identical tool+args."""
        self.ctx.lazy_tool = True
        text = ('{"tool": "list_directory", "arguments": {"path": "."}}\n'
                '{"tool": "list_directory", "arguments": {"path": "."}}')
        result = self.loop.parse_tool_calls(text)
        self.assertEqual(len(result), 1)


class TestArgumentAliases(unittest.TestCase):
    """Test TOOL_ARG_ALIASES in ToolRegistry.execute."""

    def setUp(self):
        self.ctx = q.CommandContext()
        self.ctx.auto_confirm = True
        self.reg = q.ToolRegistry(ctx=self.ctx)

    def test_path_aliased_to_file(self):
        """write_file with 'path' maps to 'file'."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                result = self.reg.execute("write_file", {
                    "path": "test.txt", "content": "hello"
                })
                self.assertTrue(result["success"])
                self.assertTrue(os.path.exists("test.txt"))
            finally:
                os.chdir(old_cwd)

    def test_file_path_aliased_to_file(self):
        """write_file with 'file_path' maps to 'file'."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                result = self.reg.execute("write_file", {
                    "file_path": "test.txt", "content": "hello"
                })
                self.assertTrue(result["success"])
            finally:
                os.chdir(old_cwd)

    def test_file_content_aliased_to_content(self):
        """write_file with 'file_content' maps to 'content'."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                result = self.reg.execute("write_file", {
                    "file": "test.txt", "file_content": "hello"
                })
                self.assertTrue(result["success"])
            finally:
                os.chdir(old_cwd)

    def test_cmd_aliased_to_command(self):
        """run_command with 'cmd' maps to 'command'."""
        result = self.reg.execute("run_command", {"cmd": "echo hello"})
        self.assertTrue(result["success"])

    def test_directory_aliased_to_path(self):
        """list_directory with 'directory' maps to 'path'."""
        result = self.reg.execute("list_directory", {"directory": "."})
        self.assertTrue(result["success"])


class TestLazyToolMode(unittest.TestCase):
    """Test lazy_tool mode in parse_tool_call."""

    def setUp(self):
        self.ctx = q.CommandContext()
        self.ctx.lazy_tool = False  # default
        self.loop = q.ChatLoop(self.ctx)

    def test_strict_rejects_embedded_bare_json(self):
        """Strict mode rejects bare JSON tool calls in the middle of text."""
        text = 'Some explanation then {"tool": "run_command", "arguments": {"command": "ls"}}'
        result = self.loop.parse_tool_call(text)
        self.assertIsNone(result)

    def test_lazy_accepts_embedded_bare_json(self):
        """Lazy mode accepts bare JSON tool calls anywhere in text."""
        self.ctx.lazy_tool = True
        text = 'Some explanation then {"tool": "run_command", "arguments": {"command": "ls"}}'
        result = self.loop.parse_tool_call(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["tool"], "run_command")

    def test_strict_rejects_embedded_code_block(self):
        """Strict mode rejects code-block tool calls deep in text."""
        text = ('Here are the steps:\n\n```json\n'
                '{"tool": "run_command", "arguments": {"command": "gcc test.c"}}\n'
                '```\n\nThen run it.')
        result = self.loop.parse_tool_call(text)
        self.assertIsNone(result)

    def test_lazy_accepts_embedded_code_block(self):
        """Lazy mode accepts code-block tool calls anywhere in text."""
        self.ctx.lazy_tool = True
        text = ('Here are the steps:\n\n```json\n'
                '{"tool": "run_command", "arguments": {"command": "gcc test.c"}}\n'
                '```\n\nThen run it.')
        result = self.loop.parse_tool_call(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["tool"], "run_command")


class TestComposableSystemPrompt(unittest.TestCase):
    """Test the composable agentic system prompt blocks."""

    def test_get_prompt_style_default(self):
        style = q.get_prompt_style("unknown-model")
        self.assertEqual(style, "strict")

    def test_get_prompt_style_nemotron(self):
        style = q.get_prompt_style("nemotron-cascade-2-30b")
        self.assertEqual(style, "soft")

    def test_get_prompt_style_substring(self):
        style = q.get_prompt_style("some-nemotron-cascade-v2")
        self.assertEqual(style, "soft")

    def test_get_agentic_prompt_includes_role(self):
        prompt = q.get_agentic_prompt("test-model")
        self.assertIn("capable AI agent", prompt)
        self.assertIn("terminal environment", prompt)

    def test_get_agentic_prompt_includes_tool_defs(self):
        tool_defs = "## Available tools\n- test_tool: does something"
        prompt = q.get_agentic_prompt("test-model", tool_defs)
        self.assertIn("test_tool", prompt)

    def test_get_agentic_prompt_strict_format(self):
        prompt = q.get_agentic_prompt("test-model")
        self.assertIn("Output ONLY the JSON tool call", prompt)
        self.assertIn("ReAct protocol", prompt)

    def test_get_agentic_prompt_soft_format(self):
        prompt = q.get_agentic_prompt("nemotron-cascade")
        self.assertIn("code block", prompt)
        self.assertNotIn("ONLY the JSON tool call", prompt)

    def test_get_agentic_prompt_includes_rules(self):
        prompt = q.get_agentic_prompt("test-model")
        self.assertIn("Mirror the user's language", prompt)
        self.assertIn("Be precise with file paths", prompt)


if __name__ == "__main__":
    unittest.main()
