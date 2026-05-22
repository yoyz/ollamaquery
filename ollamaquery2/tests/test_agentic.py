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

BACKEND = "llamacpp"
BASE_URL = os.environ.get("LLAMACPP_HOST", "http://127.0.0.1:8080")
MODEL = os.environ.get("TEST_MODEL", "google_gemma-4-E4B-it-Q4_K_M.gguf")

BACKEND_AVAILABLE = False

def setUpModule():
    """Check backend availability (does NOT skip all tests — only integration tests skip)."""
    global BACKEND_AVAILABLE
    try:
        req = q.Request(BASE_URL, method="HEAD")
        q.urlopen(req, timeout=2)
        BACKEND_AVAILABLE = True
    except Exception:
        BACKEND_AVAILABLE = False


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
        self.assertIn("JSON tool call", block)
        self.assertIn("final answer", block)
        self.assertIn("Available tools", block)

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
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f1:
            f1.write("line1\nline2\n")
            p1 = f1.name
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f2:
            f2.write("line1\nline3\n")
            p2 = f2.name
        r1 = os.path.relpath(p1)
        r2 = os.path.relpath(p2)
        result = self.reg.execute("diff", {"file1": r1, "file2": r2})
        self.assertTrue(result["success"])
        self.assertIn("-line2", result["output"])
        self.assertIn("+line3", result["output"])


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


@unittest.skipIf(not BACKEND_AVAILABLE, f"llamacpp backend not available at {BASE_URL}")
class TestAgenticReActEndToEnd(unittest.TestCase):
    """End-to-end ReAct loop tests with a real LLM backend.

    These tests require a running llama.cpp server with a model loaded.
    """

    def setUp(self):
        self.ctx = q.CommandContext()
        self.ctx.base_url = BASE_URL
        self.ctx.backend = BACKEND
        self.ctx.model = MODEL
        self.ctx.agentic_mode = True
        self.ctx.auto_confirm = True  # skip prompts
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

    def test_fetch_url_tool(self):
        """Query that requires fetch_url — fetch a known URL."""
        result = self.loop.run_agentic_query(
            "Fetch http://example.com and tell me what the page title is"
        )
        self.assertTrue(hasattr(self.loop, 'messages'))
        last = self.loop.messages[-1]
        self.assertEqual(last["role"], "assistant")
        # Should have some content
        self.assertGreater(len(last["content"]), 0)

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
                last = self.loop.messages[-1]
                self.assertEqual(last["role"], "assistant")
                self.assertGreater(len(last["content"]), 0)
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


if __name__ == "__main__":
    unittest.main()
