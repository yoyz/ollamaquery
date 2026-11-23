#!/usr/bin/env python3
"""Tests for agentic mode: ReAct loop, tool execution, and parse_tool_call."""

import io
import os
import re
import sys
import json
import time
import shlex
import shutil
import tempfile
import threading
import contextlib
import types
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ollamaquery2 as q

OLLAMA_HOST = os.environ.get('OLLAMA_HOST', 'http://127.0.0.1:11434')
LLAMACPP_HOST = os.environ.get('LLAMACPP_HOST', 'http://127.0.0.1:8080')
STRATA_HOST = os.environ.get('STRATA_HOST', 'http://127.0.0.1:8080')

# The UDP stub-resolver end-to-end test is by far the slowest in the suite: the
# model generates a C DNS resolver, compiles it, and walks a name, which costs
# many minutes and produces a large file. It is off by default; export
# RUN_SLOW_DNS_E2E=1 to re-enable it (e.g. before a release run).
SLOW_DNS_E2E = os.environ.get('RUN_SLOW_DNS_E2E', '') == '1'

# bwrap is local and fast, but it is a Linux/userns feature: the offline argv
# tests always run, the real sandbox writes are opt-in (RUN_BWRAP_E2E=1).
BWRAP_AVAILABLE = shutil.which("bwrap") is not None
RUN_BWRAP_E2E = os.environ.get('RUN_BWRAP_E2E', '') == '1'

# Auto-detect available backend
BACKEND = None
BASE_URL = None
MODEL = None
BACKEND_AVAILABLE = False


def _normalize_url(url):
    if not url.startswith(('http://', 'https://')):
        url = f"http://{url}"
    return url

def _try_ollama(url):
    """Check ollama at url, return (ok, model). Prefer gpt-oss:20b for E2E speed."""
    url = _normalize_url(url)
    if not q.check_backend_with_get(url, 'ollama'):
        return False, None
    # Try to pick a model via /api/tags — prefer gpt-oss:20b (fast, passes web-server E2E in 37s)
    try:
        tags = json.loads(q.urlopen(q.Request(f"{url}/api/tags"), timeout=3).read().decode('utf-8'))
        models = [m['name'] for m in tags.get('models', [])]
        for m in models:
            if 'gpt-oss:20b' in m.lower():
                return True, m
        for m in models:
            if 'embed' not in m.lower() and '3.5:9b' in m:
                return True, m
        for m in models:
            if 'embed' not in m.lower():
                return True, m
    except Exception:
        pass
    return True, None

def _try_llamacpp(url):
    """Check llamacpp at url, return (ok, model). Handles 415 via check_backend_with_head."""
    url = _normalize_url(url)
    if not q.check_backend_with_head(url, 'llama.cpp'):
        return False, None
    # Try /v1/models for a model name (llamacpp /v1/models)
    try:
        data = json.loads(q.urlopen(q.Request(f"{url}/v1/models"), timeout=3).read().decode('utf-8'))
        models = data.get('data', [])
        if models:
            m = models[0].get('id', models[0].get('name', ''))
            if m.startswith('models/'):
                m = m[7:]
            return True, m
    except Exception:
        pass
    return True, None

def _try_strata(url):
    """Check strata at url, return (ok, model). Keyed on the /health service marker."""
    url = _normalize_url(url)
    if not q.check_strata(url):
        return False, None
    # Try /v1/models for a model name (strata is OpenAI-compatible)
    try:
        data = json.loads(q.urlopen(q.Request(f"{url}/v1/models"), timeout=3).read().decode('utf-8'))
        models = data.get('data', [])
        if models:
            m = models[0].get('id', models[0].get('name', ''))
            if m.startswith('models/'):
                m = m[7:]
            return True, m
    except Exception:
        pass
    return True, None

def _discover_with_fallback(env_url, default_url, probe_fn):
    """Try env_url, then default, then local IPs via probe_fn."""
    candidates = []
    if env_url:
        candidates.append(_normalize_url(env_url))
    candidates.append(_normalize_url(default_url))
    # Add local network IPs (like test_features)
    try:
        import socket
        ips = socket.gethostbyname_ex(socket.gethostname())[-1]
        for ip in ips:
            port = default_url.split(':')[-1]
            candidates.append(f"http://{ip}:{port}")
    except Exception:
        pass
    seen = set()
    for url in candidates:
        if url in seen:
            continue
        seen.add(url)
        ok, model = probe_fn(url)
        if ok:
            return True, url, model
    return False, None, None


def setUpModule():
    """Detect available backend and pick a model (no hardcoded URL required).

    Probes env → localhost → local IPs using the same helpers as
    ollamaquery2.py (handles 415, Server header, missing http://).
    """
    global BACKEND, BASE_URL, MODEL, BACKEND_AVAILABLE

    ok, url, model = _discover_with_fallback(OLLAMA_HOST, 'http://127.0.0.1:11434', _try_ollama)
    if ok:
        BACKEND = "ollama"
        BASE_URL = url
        MODEL = os.environ.get("TEST_MODEL") or model or "gpt-oss:20b"
        BACKEND_AVAILABLE = True
        return

    # Strata shares llama.cpp's 8080 port but sends a generic Server header, so
    # probe it before llama.cpp (mirrors auto_detect_backend's ordering).
    ok, url, model = _discover_with_fallback(STRATA_HOST, 'http://127.0.0.1:8080', _try_strata)
    if ok:
        BACKEND = "strata"
        BASE_URL = url
        MODEL = os.environ.get("TEST_MODEL") or model or "strata"
        BACKEND_AVAILABLE = True
        return

    ok, url, model = _discover_with_fallback(LLAMACPP_HOST, 'http://127.0.0.1:8080', _try_llamacpp)
    if ok:
        BACKEND = "llamacpp"
        BASE_URL = url
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

    def test_plan_mode_restricts_system_prompt_block(self):
        self.ctx.plan_mode = True
        try:
            block = self.reg.get_system_prompt_block()
            self.assertIn("read_file", block)
            self.assertIn("list_directory", block)
            self.assertIn("glob", block)
            self.assertIn("fetch_url", block)
            self.assertIn("diff", block)
            self.assertNotIn("### run_command", block)  # no shell in strict plan
            self.assertNotIn("write_file", block)
            self.assertNotIn("run_python", block)
            self.assertNotIn("apply_patch", block)
        finally:
            self.ctx.plan_mode = False

    def test_plan_run_mode_surface_adds_run_command(self):
        self.ctx.plan_mode = True
        self.ctx.plan_run = True
        try:
            block = self.reg.get_system_prompt_block()
            self.assertIn("read_file", block)
            self.assertIn("### run_command", block)
            self.assertNotIn("write_file", block)
            self.assertNotIn("run_python", block)
            spec = self.reg.openai_tools_spec()
            names = {t["function"]["name"] for t in spec}
            self.assertEqual(names, q.PLAN_MODE_RUN_TOOLS)
        finally:
            self.ctx.plan_mode = False
            self.ctx.plan_run = False

    def test_plan_mode_openai_tools_spec_restricted(self):
        self.ctx.plan_mode = True
        try:
            spec = self.reg.openai_tools_spec()
            names = {t["function"]["name"] for t in spec}
            self.assertEqual(names, q.PLAN_MODE_TOOLS)
        finally:
            self.ctx.plan_mode = False

    def test_plan_mode_denies_write_tools(self):
        """Execution backstop: hidden write tools are denied even if called."""
        self.ctx.plan_mode = True
        try:
            result = self.reg.execute("write_file", {"file_path": "x.txt", "content": "x"})
            self.assertFalse(result["success"])
            self.assertIn("read-only plan mode", result["error"])
            result = self.reg.execute("run_python", {"code": "print(1)"})
            self.assertFalse(result["success"])
            self.assertIn("read-only plan mode", result["error"])
        finally:
            self.ctx.plan_mode = False

    def test_plan_mode_allows_read_tools(self):
        cwd = os.getcwd()
        testfile = os.path.join(cwd, ".agentic_test_read.tmp")
        self.ctx.plan_mode = True
        try:
            with open(testfile, "w") as f:
                f.write("plan hello")
            result = self.reg.execute("read_file", {"file": ".agentic_test_read.tmp"})
            self.assertTrue(result["success"], msg=result.get("error"))
            self.assertIn("plan hello", result["output"])
            result = self.reg.execute("list_directory", {"path": "."})
            self.assertTrue(result["success"], msg=result.get("error"))
            result = self.reg.execute("glob", {"pattern": "*.py"})
            self.assertTrue(result["success"], msg=result.get("error"))
        finally:
            self.ctx.plan_mode = False
            if os.path.exists(testfile):
                os.unlink(testfile)

    def test_plan_mode_run_command_denied_strict(self):
        """Strict plan mode has no shell: run_command is denied, never prompted."""
        self.ctx.plan_mode = True
        self.ctx.auto_confirm = True
        try:
            with patch("builtins.input", return_value="y") as mocked:
                result = self.reg.execute("run_command", {"command": "echo plan-test"})
            self.assertFalse(result["success"])
            self.assertIn("read-only plan mode", result["error"])
            mocked.assert_not_called()
        finally:
            self.ctx.plan_mode = False

    def test_plan_run_command_confirms_even_auto(self):
        """run_command in the `plan run` tier prompts even with auto_confirm."""
        self.ctx.plan_mode = True
        self.ctx.plan_run = True
        self.ctx.auto_confirm = True
        try:
            with patch("builtins.input", return_value="n"):
                result = self.reg.execute("run_command", {"command": "echo plan-test"})
            self.assertFalse(result["success"])
            self.assertIn("Cancelled by user", result["error"])
        finally:
            self.ctx.plan_mode = False
            self.ctx.plan_run = False

    def test_plan_strict_confines_reads_to_cwd(self):
        """Strict plan refuses an absolute path outside CWD instead of prompting."""
        with tempfile.TemporaryDirectory() as outside:
            victim = os.path.join(outside, "secret.txt")
            with open(victim, "w") as f:
                f.write("top secret")
            self.ctx.plan_mode = True
            try:
                with patch("builtins.input", return_value="y") as mocked:
                    result = self.reg.execute("read_file", {"file": victim})
                self.assertFalse(result["success"])
                self.assertIn("confined", result["error"].lower())
                mocked.assert_not_called()
            finally:
                self.ctx.plan_mode = False

    def test_plan_strict_confines_list_directory_outside_cwd(self):
        """Strict plan refuses listing a directory outside CWD (no prompt)."""
        with tempfile.TemporaryDirectory() as outside:
            self.ctx.plan_mode = True
            try:
                with patch("builtins.input", return_value="y") as mocked:
                    result = self.reg.execute("list_directory", {"path": outside})
                self.assertFalse(result["success"])
                self.assertIn("confined", result["error"].lower())
                mocked.assert_not_called()
            finally:
                self.ctx.plan_mode = False

    def test_plan_strict_honors_session_allow(self):
        """An explicit session allow rule still works in strict plan mode."""
        with tempfile.TemporaryDirectory() as outside:
            victim = os.path.join(outside, "notes.txt")
            with open(victim, "w") as f:
                f.write("allowed notes")
            self.ctx.plan_mode = True
            self.ctx.path_acl.add("read", "allow", outside, source="session")
            try:
                result = self.reg.execute("read_file", {"file": victim})
                self.assertTrue(result["success"], msg=result.get("error"))
                self.assertIn("allowed notes", result["output"])
            finally:
                self.ctx.plan_mode = False
                self.ctx.path_acl.remove(outside)

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

    def _grep_fixture(self):
        """Create a temp CWD with fixture files, returning their dir path."""
        import tempfile
        tmpdir = tempfile.mkdtemp()
        self._grep_old_cwd = os.getcwd()
        os.chdir(tmpdir)
        _write("a.md", "Hello Storage nodes here\nno match\n")
        os.mkdir("sub")
        _write(os.path.join("sub", "b.md"), "storageDeviceSets present\n")
        _write("notes.txt", "STORAGE NODES in txt\n")
        return tmpdir

    def _grep_cleanup(self):
        if getattr(self, "_grep_old_cwd", None):
            os.chdir(self._grep_old_cwd)

    def test_grep_basic_case_insensitive_default(self):
        """grep matches case-insensitively by default with include filter."""
        self._grep_fixture()
        try:
            result = self.reg.execute("grep", {"pattern": "storage[ _-]?nodes|storageDeviceSets",
                                               "include": "*.md"})
            self.assertTrue(result["success"], msg=result.get("error"))
            self.assertIn("Found 2 matches", result["output"])
            self.assertIn("a.md:", result["output"])
            self.assertIn("sub" + os.sep + "b.md:", result["output"])
            self.assertIn("Line 1:", result["output"])
            self.assertNotIn("notes.txt", result["output"])  # include=*.md excluded it
        finally:
            self._grep_cleanup()

    def test_grep_case_sensitive(self):
        """case_insensitive=false must not match a differently-cased substring."""
        self._grep_fixture()
        try:
            result = self.reg.execute("grep", {"pattern": "Storage Nodes", "case_insensitive": False,
                                               "include": "*.md"})
            self.assertTrue(result["success"], msg=result.get("error"))
            self.assertEqual("No files found", result["output"])
            result = self.reg.execute("grep", {"pattern": "Storage nodes", "case_insensitive": False,
                                               "include": "*.md"})
            self.assertTrue(result["success"], msg=result.get("error"))
            self.assertIn("Found 1 matches", result["output"])
        finally:
            self._grep_cleanup()

    def test_grep_path_subdir_and_single_file(self):
        """path can be a subdirectory or a single file."""
        self._grep_fixture()
        try:
            result = self.reg.execute("grep", {"pattern": "storage", "path": "sub"})
            self.assertTrue(result["success"], msg=result.get("error"))
            self.assertIn("Found 1 matches", result["output"])
            result = self.reg.execute("grep", {"pattern": "storage", "path": os.path.join("sub", "b.md")})
            self.assertTrue(result["success"], msg=result.get("error"))
            self.assertIn("Found 1 matches", result["output"])
            result = self.reg.execute("grep", {"pattern": "storage", "path": "notes.txt"})
            self.assertTrue(result["success"], msg=result.get("error"))
            self.assertIn("STORAGE", result["output"])
        finally:
            self._grep_cleanup()

    def test_grep_no_match(self):
        self._grep_fixture()
        try:
            result = self.reg.execute("grep", {"pattern": "zzz_nope"})
            self.assertTrue(result["success"], msg=result.get("error"))
            self.assertEqual("No files found", result["output"])
        finally:
            self._grep_cleanup()

    def test_grep_alias_and_defaults(self):
        """Argument aliases (regex/count) map to pattern/max_results."""
        self._grep_fixture()
        try:
            result = self.reg.execute("grep", {"regex": "nodes", "count": 5})
            self.assertTrue(result["success"], msg=result.get("error"))
            self.assertIn("Found 2 matches", result["output"])
        finally:
            self._grep_cleanup()

    def test_grep_max_results_capped(self):
        """Result cap: at most max_results matches plus a truncation note."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                for i in range(150):
                    _write(f"f{i}.txt", "SAME\n")
                result = self.reg.execute("grep", {"pattern": "SAME"})
                self.assertTrue(result["success"], msg=result.get("error"))
                self.assertEqual(result["output"].count("Line 1:"), q.MAX_GREP_RESULTS)
                self.assertIn("capped at", result["output"])
            finally:
                os.chdir(old_cwd)

    def test_grep_invalid_regex(self):
        self._grep_fixture()
        try:
            result = self.reg.execute("grep", {"pattern": "(unclosed"})
            self.assertFalse(result["success"])
            self.assertTrue(result["error"])
        finally:
            self._grep_cleanup()

    def test_grep_acl_deny(self):
        """A search rooted at a denied/ask path must fail without executing."""
        result = self.reg.execute("grep", {"pattern": "x", "path": "/proc/1"})
        self.assertFalse(result["success"])
        self.assertIn("denied", result["error"])

    def test_grep_missing_pattern_and_path(self):
        result = self.reg.execute("grep", {})
        self.assertFalse(result["success"])
        self.assertIn("pattern is required", result["error"])
        result = self.reg.execute("grep", {"pattern": "x", "path": "no_such_dir_xyz"})
        self.assertFalse(result["success"])
        self.assertIn("Path not found", result["error"])

    def test_grep_plan_mode_allowed(self):
        """grep is part of the read-only plan surface and runs without prompts."""
        self._grep_fixture()
        self.ctx.plan_mode = True
        try:
            result = self.reg.execute("grep", {"pattern": "storage", "include": "*.md"})
            self.assertTrue(result["success"], msg=result.get("error"))
            self.assertIn("Found", result["output"])
        finally:
            self.ctx.plan_mode = False
            self._grep_cleanup()

    def test_grep_pure_python_fallback(self):
        """When rg and grep are both missing, the pure-Python backend still works."""
        self._grep_fixture()
        try:
            with patch("ollamaquery2.shutil.which", return_value=None):
                result = self.reg.execute("grep", {"pattern": "storage", "include": "*.md"})
            self.assertTrue(result["success"], msg=result.get("error"))
            self.assertIn("Found 2 matches", result["output"])
        finally:
            self._grep_cleanup()

    def test_grep_skips_dot_git_recursively(self):
        """Recursive search skips .git in every backend (rg/grep/Python)."""
        self._grep_fixture()
        try:
            os.makedirs(os.path.join(".git", "objects"))
            _write(os.path.join(".git", "config"), "NESTED_GIT_TOKEN\n")
            _write(os.path.join(".git", "HEAD"), "NESTED_GIT_TOKEN\n")
            result = self.reg.execute("grep", {"pattern": "NESTED_GIT_TOKEN"})
            self.assertTrue(result["success"], msg=result.get("error"))
            self.assertEqual("No files found", result["output"])
            # Same guarantee through the pure-Python walk.
            with patch("ollamaquery2.shutil.which", return_value=None):
                result = self.reg.execute("grep", {"pattern": "NESTED_GIT_TOKEN"})
            self.assertTrue(result["success"], msg=result.get("error"))
            self.assertEqual("No files found", result["output"])
        finally:
            self._grep_cleanup()

    def test_grep_explicit_git_file_is_searched(self):
        """An explicit path inside .git is honoured, not silently excluded."""
        self._grep_fixture()
        try:
            os.makedirs(".git")
            _write(os.path.join(".git", "config"), "EXPLICIT_GIT_TOKEN\n")
            result = self.reg.execute("grep", {"pattern": "EXPLICIT_GIT_TOKEN",
                                               "path": os.path.join(".git", "config")})
            self.assertTrue(result["success"], msg=result.get("error"))
            self.assertIn("EXPLICIT_GIT_TOKEN", result["output"])
        finally:
            self._grep_cleanup()

    def test_grep_per_file_acl_hides_restricted_matches(self):
        """A match the per-file ACL would ask for is hidden, not leaked.

        The root-level gate stays permissive (CWD), so without the per-file
        check a recursive search would return this file's contents silently.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                _write("normal.txt", "SHARED_TOKEN\n")
                _write("secret.txt", "SHARED_TOKEN\n")
                with patch("ollamaquery2._is_home_dotfile",
                           side_effect=lambda p: os.path.basename(p) == "secret.txt"):
                    result = self.reg.execute("grep", {"pattern": "SHARED_TOKEN"})
                self.assertTrue(result["success"], msg=result.get("error"))
                self.assertIn("normal.txt", result["output"])
                self.assertNotIn("secret.txt", result["output"])
                self.assertIn("hidden by the path ACL", result["output"])
            finally:
                os.chdir(old_cwd)

    def test_grep_explicit_home_keeps_non_dotfiles(self):
        """An explicit ~ search still returns normal files but not home dotfiles."""
        with tempfile.TemporaryDirectory() as home, tempfile.TemporaryDirectory() as cwd:
            old_cwd = os.getcwd()
            old_acl = self.ctx.path_acl
            os.chdir(cwd)
            try:
                # Force the Python backend: rg skips hidden files on its own, so
                # only grep/os.walk would ever surface the dotfile to filter.
                with patch.dict(os.environ, {"HOME": home}), \
                     patch("ollamaquery2.shutil.which", return_value=None):
                    self.ctx.path_acl = q.PathAcl()  # defaults resolve against patched HOME
                    _write(os.path.join(home, "visible.txt"), "HOME_TOKEN\n")
                    _write(os.path.join(home, ".hidden.txt"), "HOME_TOKEN\n")
                    result = self.reg.execute("grep", {"pattern": "HOME_TOKEN", "path": home})
                self.assertTrue(result["success"], msg=result.get("error"))
                self.assertIn("visible.txt", result["output"])
                self.assertNotIn(".hidden.txt", result["output"])
                self.assertIn("hidden by the path ACL", result["output"])
            finally:
                self.ctx.path_acl = old_acl
                os.chdir(old_cwd)

    def test_grep_output_char_cap(self):
        """Rendered output is hard-capped so one call cannot flood context."""
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                with open("big.txt", "w") as f:
                    for _ in range(120):
                        f.write("CAP_TOKEN " + "x" * 600 + "\n")
                result = self.reg.execute("grep", {"pattern": "CAP_TOKEN"})
                self.assertTrue(result["success"], msg=result.get("error"))
                self.assertLessEqual(len(result["output"]), q.MAX_GREP_OUTPUT_CHARS)
                self.assertIn("Output capped", result["output"])
            finally:
                os.chdir(old_cwd)

    def test_grep_cap_observation_exempt(self):
        """grep results bypass the generic observation cap (self-bounding)."""
        big = "x" * 9000
        self.assertEqual(q.ChatLoop._cap_tool_observation("grep", big), big)

    def test_harmless_get_fetch_detector(self):
        """Only pure GET curl/wget with a single URL and no side effects match."""
        ok = {
            'curl -sL "https://example.com/a"': "https://example.com/a",
            "curl -sSL -L https://x.y": "https://x.y",
            "curl -A UA --max-time 5 https://x.y": "https://x.y",
            "wget -q -O - https://x.y/z": "https://x.y/z",
            "wget --quiet --output-document=- https://x.y": "https://x.y",
        }
        for cmd, url in ok.items():
            self.assertEqual(q._is_harmless_get_fetch(cmd), url, cmd)
        no = [
            "curl -d a=1 https://x.y",             # POST body
            "curl -o /tmp/a.html https://x.y",     # writes to disk
            "curl -s https://x | grep foo",        # pipe
            "curl -s https://x.y https://z",       # two URLs
            "curl -X POST https://x.y",            # method override
            "wget -q https://x.y",                 # wget writes files w/o -O -
            "wget -O out.html https://x.y",        # output to file
            "curl -s https://x.y ; echo hi",       # shell chaining
            "grep foo bar.md", "", "rm -rf /",
        ]
        for cmd in no:
            self.assertEqual(q._is_harmless_get_fetch(cmd), "", cmd)

    def test_run_command_curl_get_redirects_to_fetch_url(self):
        """A harmless GET curl in `plan run` mode is served by fetch_url, no Y/N."""
        self.ctx.plan_mode = True
        self.ctx.plan_run = True
        self.ctx.auto_confirm = False  # would prompt if not redirected
        try:
            with patch("ollamaquery2.fetch_and_convert_url",
                       return_value=("SERVED PAGES", "htmlstrip")):
                result = self.reg.execute(
                    "run_command", {"command": 'curl -sL "https://example.com"'})
            self.assertTrue(result["success"], msg=result.get("error"))
            self.assertIn("fetch_url", result["output"])       # steering note
            self.assertIn("SERVED PAGES", result["output"])    # actual content
        finally:
            self.ctx.plan_mode = False
            self.ctx.plan_run = False

    def test_run_command_curl_failure_redirect(self):
        """A redirected fetch still surfaces the failure instead of OK."""
        self.ctx.plan_mode = True
        self.ctx.plan_run = True
        try:
            with patch("ollamaquery2.fetch_and_convert_url",
                       return_value=("[Failed to fetch URL: HTTP Error 403: Forbidden]", "None")):
                result = self.reg.execute(
                    "run_command", {"command": "curl -s https://blocked.example"})
            self.assertFalse(result["success"])
            self.assertIn("fetch_url", result["error"])
            self.assertIn("403", result["error"])
        finally:
            self.ctx.plan_mode = False
            self.ctx.plan_run = False

    def test_non_fetch_curl_still_gated(self):
        """curl that writes to disk is not intercepted; `plan run` still asks."""
        self.ctx.plan_mode = True
        self.ctx.plan_run = True
        self.ctx.auto_confirm = True
        try:
            with patch("ollamaquery2.fetch_and_convert_url",
                       return_value=("SHOULD NOT BE USED", "htmlstrip")):
                with patch("builtins.input", return_value="n"):
                    result = self.reg.execute(
                        "run_command", {"command": "curl -o /tmp/fetch_test.html https://x.example"})
            self.assertFalse(result["success"])
            self.assertIn("Cancelled by user", result["error"])
            self.assertNotIn("SHOULD NOT BE USED", result.get("output", ""))
        finally:
            self.ctx.plan_mode = False
            self.ctx.plan_run = False

    def test_fetch_url_surfaces_failure(self):
        """fetch_url must not report success while embedding a failure string."""
        with patch("ollamaquery2.fetch_and_convert_url",
                   return_value=("[Failed to fetch URL: HTTP Error 403: Forbidden]", "None")):
            result = self.reg.execute("fetch_url", {"url": "https://blocked.example"})
        self.assertFalse(result["success"])
        self.assertIn("403", result["error"])

    def test_fetch_url_empty_output_is_failure(self):
        """fetch_url must not report success with an empty output (regression:
        success=True with output="" made models conclude the tool was broken
        and fall back to shell curl)."""
        with patch("ollamaquery2.fetch_and_convert_url",
                   return_value=("", "htmlstrip")):
            result = self.reg.execute("fetch_url", {"url": "https://empty.example"})
        self.assertFalse(result["success"])
        self.assertTrue(result["error"])

    def test_fetch_url_404_steers_to_follow_links(self):
        """A 404 error steers the model to follow links from fetched pages
        instead of guessing more non-existent URLs."""
        with patch("ollamaquery2.fetch_and_convert_url",
                   return_value=("[Failed to fetch URL: HTTP Error 404: Not Found]", "None")):
            result = self.reg.execute("fetch_url", {"url": "https://www.insee.fr/fr/statistiques/2028001"})
        self.assertFalse(result["success"])
        self.assertIn("404", result["error"])
        self.assertIn("follow links", result["error"])

    def test_fetch_url_5xx_steers_away_from_retrying(self):
        """A 5xx error steers the model to links/search instead of retrying
        (insee.fr serves 500 on search/RSS/sitemap to any programmatic client)."""
        with patch("ollamaquery2.fetch_and_convert_url",
                   return_value=("[Failed to fetch URL: HTTP Error 500: Internal Server Error]", "None")):
            result = self.reg.execute("fetch_url", {"url": "https://www.insee.fr/fr/recherche/recherche"})
        self.assertFalse(result["success"])
        self.assertIn("500", result["error"])
        self.assertIn("search engine", result["error"])

    def test_fetch_url_large_result_spilled_to_file(self):
        """fetch_url results over the spill threshold go to a file; the observation
        is a pointer (path + preview, no content) and read_file can page it."""
        with tempfile.TemporaryDirectory() as tmp:
            spill_dir = os.path.join(tmp, "spill")
            with patch.object(q, "CLUSTER_SPILL_DIR", spill_dir):
                spill_file = q._fetch_spill_path("https://xyz.example/page")
                big = ("PREVIEW-ONLY-HERE-PAGE-TITLE-LONG-ENOUGH\n"
                       + ("PARAGRAPH-7f3a9c\n" * 800))  # ~13KB, over threshold
                with patch("ollamaquery2.fetch_and_convert_url",
                           return_value=(big, "htmlstrip")):
                    result = self.reg.execute("fetch_url", {"url": "https://xyz.example/page"})
            self.assertTrue(result["success"], msg=result.get("error"))
            self.assertIn(spill_file, result["output"])
            self.assertIn("chars", result["output"])
            self.assertIn("Preview:", result["output"])
            self.assertIn("PREVIEW-ONLY-HERE-PAGE-TITLE-LONG-ENOUGH", result["output"])
            # No payload inline — pointer + preview only, so the content is not
            # fed into context twice (only via read_file paging).
            self.assertNotIn("PARAGRAPH-7f3a9c", result["output"])
            self.assertLess(len(result["output"]), 600)
            with open(spill_file, encoding="utf-8") as f:
                self.assertEqual(f.read(), big)
            # read_file can page the spill file (session ACL allow was added)
            rd = self.reg.execute("read_file", {"file_path": spill_file})
            self.assertTrue(rd["success"], msg=rd.get("error"))
            self.assertIn("PARAGRAPH-7f3a9c", rd["output"])

    def test_fetch_url_small_result_inline_no_spill(self):
        """Results under the spill threshold are returned inline, nothing spilled."""
        with tempfile.TemporaryDirectory() as tmp:
            spill_dir = os.path.join(tmp, "spill")
            with patch.object(q, "CLUSTER_SPILL_DIR", spill_dir):
                small = "Hello page content"
                with patch("ollamaquery2.fetch_and_convert_url",
                           return_value=(small, "htmlstrip")):
                    result = self.reg.execute("fetch_url", {"url": "https://xyz.example/small"})
            self.assertTrue(result["success"])
            self.assertEqual(result["output"], small)
            self.assertFalse(os.path.exists(spill_dir))  # nothing spilled

    def test_cleanup_spill_files_removes_fetch_results(self):
        """Atexit cleanup removes fetch_result_*.txt and cluster_result_*.json."""
        with tempfile.TemporaryDirectory() as tmp:
            spill_dir = os.path.join(tmp, "spill")
            os.makedirs(spill_dir)
            names = ["cluster_result_get_pods.json", "fetch_result_https_x.txt",
                     "fetch_result_other.log", "unrelated.txt"]
            for name in names:
                with open(os.path.join(spill_dir, name), "w") as f:
                    f.write("x")
            with patch.object(q, "CLUSTER_SPILL_DIR", spill_dir):
                q._cleanup_spill_files()
            remaining = sorted(os.listdir(spill_dir))
            self.assertEqual(remaining, ["fetch_result_other.log", "unrelated.txt"])

    def test_fetch_url_curl_fallback(self):
        """fetch_and_convert_url falls back to curl when urllib is blocked."""
        with patch("ollamaquery2._request_with_retry",
                   side_effect=Exception("HTTP Error 403: Forbidden")):
            with patch("ollamaquery2._curl_fetch_text",
                       return_value=("<html><body>curl rescued content</body></html>", False)):
                text, tool = q.fetch_and_convert_url("https://blocked.example")
        self.assertIn("curl rescued content", text)
        self.assertEqual(tool, "htmlstrip")

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

    def test_read_file_truncation_marked(self):
        """read_file on a file larger than MAX_READ_FILE_SIZE must mark truncation."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                with open("big.txt", "w") as f:
                    f.write("x" * (q.MAX_READ_FILE_SIZE + 1000))
                result = self.reg.execute("read_file", {"file": "big.txt"})
                self.assertTrue(result["success"])
                self.assertIn("(line truncated to", result["output"])
                self.assertLessEqual(len(result["output"]), q.MAX_READ_LINE_LENGTH + 200)
            finally:
                os.chdir(old_cwd)

    def test_read_file_pages_large_file(self):
        """A file over MAX_READ_FILE_SIZE auto-pages with a `next` offset."""
        import tempfile
        import re
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                with open("big.txt", "w") as f:
                    for i in range(5000):
                        f.write(f"line {i:04d} " + "padding data " * 6 + "\n")
                self.assertGreater(os.path.getsize("big.txt"), q.MAX_READ_FILE_SIZE)
                result = self.reg.execute("read_file", {"file": "big.txt"})
                self.assertTrue(result["success"], msg=result.get("error"))
                head = result["output"].split("\n")[0]
                self.assertIn("more available", head)
                mh = re.search(r'offset=(\d+)', head)
                self.assertIsNotNone(mh, f"no next offset in header: {head}")
                self.assertGreater(int(mh.group(1)), 1)
                self.assertIn("read_file", head)
            finally:
                os.chdir(old_cwd)

    def test_read_file_chained_to_end(self):
        """Following the returned `next` offsets must reach end-of-file."""
        import tempfile
        import re
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                with open("big.txt", "w") as f:
                    for i in range(3000):
                        f.write(f"line {i:04d} " + "padding data " * 6 + "\n")
                result = self.reg.execute("read_file", {"file": "big.txt"})
                head = result["output"].split("\n")[0]
                mh = re.search(r'offset=(\d+)', head)
                off = int(mh.group(1)) if mh else None
                pages = 1
                while off:
                    r = self.reg.execute("read_file", {"file": "big.txt", "offset": off})
                    self.assertTrue(r["success"], msg=r.get("error"))
                    h = r["output"].split("\n")[0]
                    m2 = re.search(r'offset=(\d+)', h)
                    off = int(m2.group(1)) if m2 else None
                    pages += 1
                    self.assertLess(pages, 50)
                self.assertIn("line 2999", r["output"])
            finally:
                os.chdir(old_cwd)

    def test_read_file_offset_window(self):
        """An explicit offset/limit returns exactly that line range."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                with open("w.txt", "w") as f:
                    for i in range(100):
                        f.write(f"row {i}\n")
                result = self.reg.execute("read_file", {"file": "w.txt", "offset": 10, "limit": 5})
                self.assertTrue(result["success"], msg=result.get("error"))
                head = result["output"].split("\n")[0]
                self.assertIn("lines 10-14", head)
                self.assertIn("row 9", result["output"])
                self.assertIn("row 13", result["output"])
                self.assertNotIn("row 14", result["output"])
            finally:
                os.chdir(old_cwd)

    def test_read_file_offset_out_of_range(self):
        """An offset past the last line reports end-of-file, not an error."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                with open("w.txt", "w") as f:
                    f.write("a\nb\n")
                result = self.reg.execute("read_file", {"file": "w.txt", "offset": 50})
                self.assertTrue(result["success"])
                self.assertIn("no lines at offset 50", result["output"])
            finally:
                os.chdir(old_cwd)

    def test_read_file_small_file_unchanged(self):
        """Small files read whole with no paging header."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                with open("small.txt", "w") as f:
                    f.write("line1\nline2\n")
                result = self.reg.execute("read_file", {"file": "small.txt"})
                self.assertTrue(result["success"], msg=result.get("error"))
                self.assertEqual(result["output"], "line1\nline2\n")
            finally:
                os.chdir(old_cwd)

    def test_read_file_long_line_truncated(self):
        """Single lines longer than MAX_READ_LINE_LENGTH are truncated per-line."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                with open("long.txt", "w") as f:
                    f.write("x" * 5000 + "\n")
                result = self.reg.execute("read_file", {"file": "long.txt", "limit": 5})
                self.assertTrue(result["success"], msg=result.get("error"))
                self.assertIn(f"(line truncated to {q.MAX_READ_LINE_LENGTH} chars)", result["output"])
                self.assertLessEqual(len(result["output"]), q.MAX_READ_LINE_LENGTH + 200)
            finally:
                os.chdir(old_cwd)

    def test_cap_tool_observation_exempts_read_file(self):
        """read_file observations skip the generic 4000-char context cap."""
        big = "z" * 9000
        self.assertEqual(q.ChatLoop._cap_tool_observation("read_file", big), big)
        capped = q.ChatLoop._cap_tool_observation("run_command", big)
        self.assertLess(len(capped), 4000 + 100)
        self.assertIn("[truncated", capped)

    def test_list_directory(self):
        result = self.reg.execute("list_directory", {"path": "."})
        self.assertTrue(result["success"])
        self.assertIn("ollamaquery2.py", result["output"])  # main file

    def test_glob(self):
        result = self.reg.execute("glob", {"pattern": "*.py"})
        self.assertTrue(result["success"])

    def test_plan_run_block_names_the_fetch_exception(self):
        """Plan run confirms commands; the harmless GET redirect is the one
        documented exception, so the prompt must say it instead of promising
        a confirmation that never happens."""
        self.assertIn("read-only fetch_url tool without a prompt",
                      q.AGENTIC_PLAN_RUN_BLOCK)

    def test_write_file_keeps_the_original_when_replace_fails(self):
        with tempfile.TemporaryDirectory() as td:
            old_cwd = os.getcwd()
            os.chdir(td)
            try:
                with open("keep.txt", "w", encoding="utf-8") as f:
                    f.write("original")
                with patch.object(q.os, "replace",
                                  side_effect=OSError("disk full")):
                    result = self.reg.execute(
                        "write_file", {"file_path": "keep.txt",
                                       "content": "new"})
                self.assertFalse(result["success"])
                with open("keep.txt", encoding="utf-8") as f:
                    self.assertEqual(f.read(), "original")
            finally:
                os.chdir(old_cwd)

    def test_write_file_preserves_an_existing_mode(self):
        with tempfile.TemporaryDirectory() as td:
            old_cwd = os.getcwd()
            os.chdir(td)
            try:
                with open("mode.txt", "w", encoding="utf-8") as f:
                    f.write("old")
                os.chmod("mode.txt", 0o640)
                result = self.reg.execute(
                    "write_file", {"file_path": "mode.txt", "content": "new"})
                self.assertTrue(result["success"], msg=result.get("error"))
                self.assertEqual(os.stat("mode.txt").st_mode & 0o777, 0o640)
            finally:
                os.chdir(old_cwd)

    def test_edit_file_refuses_a_file_too_large_to_load(self):
        with tempfile.TemporaryDirectory() as td:
            old_cwd = os.getcwd()
            os.chdir(td)
            try:
                with open("big.txt", "w", encoding="utf-8") as f:
                    f.truncate(q.MAX_WRITE_FILE_SIZE + 1)
                result = self.reg.execute(
                    "edit_file", {"file_path": "big.txt", "old_string": "x",
                                  "new_string": "y"})
                self.assertFalse(result["success"])
                self.assertIn("too large", result["error"])
            finally:
                os.chdir(old_cwd)

    def test_apply_patch_roundtrips_utf8(self):
        with tempfile.TemporaryDirectory() as td:
            old_cwd = os.getcwd()
            os.chdir(td)
            try:
                with open("accent.txt", "w", encoding="utf-8") as f:
                    f.write("café\n")
                patch_text = ("--- a/accent.txt\n+++ b/accent.txt\n"
                              "@@ -1 +1 @@\n-café\n+thé\n")
                result = self.reg.execute("apply_patch",
                                          {"patch_text": patch_text})
                self.assertTrue(result["success"], msg=result.get("error"))
                with open("accent.txt", encoding="utf-8") as f:
                    self.assertEqual(f.read(), "thé\n")
            finally:
                os.chdir(old_cwd)

    def test_glob_does_not_follow_a_symlink_outside_the_project(self):
        """A symlink inside the project must not turn glob into an /etc listing."""
        outside = tempfile.mkdtemp(prefix="glob-outside-")
        project = tempfile.mkdtemp(prefix="glob-project-")
        old_cwd = os.getcwd()
        try:
            with open(os.path.join(outside, "secret.txt"), "w",
                      encoding="utf-8") as f:
                f.write("outside")
            try:
                os.symlink(outside, os.path.join(project, "escape"))
            except (OSError, NotImplementedError):
                self.skipTest("symlinks unavailable")
            os.chdir(project)
            result = self.reg.execute("glob", {"pattern": "escape/*"})
            self.assertTrue(result["success"], msg=result.get("error"))
            self.assertNotIn("secret.txt", result["output"])
            self.assertIn("not listed", result["output"])
        finally:
            os.chdir(old_cwd)
            shutil.rmtree(outside, ignore_errors=True)
            shutil.rmtree(project, ignore_errors=True)

    def test_glob_honours_a_session_deny(self):
        project = tempfile.mkdtemp(prefix="glob-deny-")
        old_cwd = os.getcwd()
        saved_rules = list(self.ctx.path_acl.rules)
        try:
            os.chdir(project)
            with open("visible.txt", "w", encoding="utf-8") as f:
                f.write("ok")
            with open("hidden.txt", "w", encoding="utf-8") as f:
                f.write("no")
            self.ctx.path_acl.add(
                "read", "deny", os.path.join(project, "hidden.txt"),
                source="session")
            result = self.reg.execute("glob", {"pattern": "*.txt"})
            self.assertTrue(result["success"], msg=result.get("error"))
            self.assertIn("visible.txt", result["output"])
            self.assertNotIn("hidden.txt", result["output"])
        finally:
            os.chdir(old_cwd)
            self.ctx.path_acl.rules = saved_rules
            shutil.rmtree(project, ignore_errors=True)

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


class TestGrepReliability(unittest.TestCase):
    """grep must be trustworthy for an audit, not merely non-crashing.

    A false negative is worse than an error here: the model writes a conclusion
    on "pattern absent", so a silent skip turns into a confident wrong answer.
    TestToolRegistry's 17 grep cases cover the happy paths, argument handling,
    the ACL and the caps; this class owns the fixtures that made the three
    backends disagree (binary, unreadable, CRLF, a 4.5 MB file, a 20k-char line,
    symlinks, match text with its own colons) plus the two blocking properties:
    nothing is skipped silently, and a pathological scan is bounded.

    The external branch asserts the pure-Python fallback did NOT run, so a
    "backend parity" claim cannot be satisfied by both sides accidentally
    executing the same code.
    """

    BASE_PATTERN = r"HIT_(TEXT|LATIN|CRLF|NEWLINE)"

    def setUp(self):
        self.ctx = q.CommandContext()
        self.ctx.auto_confirm = True
        self.reg = q.ToolRegistry(ctx=self.ctx)
        self.tree = tempfile.mkdtemp()
        self._old_cwd = os.getcwd()
        os.chdir(self.tree)
        self._write("text.py", "import os\nHIT_TEXT alpha\nno match here\n")
        self._write("latin1.txt", b"caf\xe9 HIT_LATIN latin1\n")
        self._write("crlf.txt", b"HIT_CRLF windows line\r\nplain\r\n")
        self._write("nonewline.txt", "before\nHIT_NEWLINE tail")
        self._write("empty.txt", "")

    def tearDown(self):
        os.chdir(self._old_cwd)
        shutil.rmtree(self.tree, ignore_errors=True)

    # -- fixtures ---------------------------------------------------------
    def _write(self, name, data):
        """Write a fixture file, bytes or text."""
        with open(name, "wb" if isinstance(data, bytes) else "w") as fh:
            fh.write(data)

    def _make_big_log(self):
        """4.5 MB with a match on the first, middle and last line."""
        with open("big.log", "w") as fh:
            fh.write("HIT_FIRST first line\n")
            for i in range(70000):
                fh.write(f"filler {i} " + "y" * 50 + "\n")
                if i == 35000:
                    fh.write("HIT_MIDDLE middle line\n")
            fh.write("HIT_LAST last line\n")

    def _make_unreadable(self, name, content):
        """Create a file, then make it unreadable (no-op for root)."""
        self._write(name, content)
        os.chmod(name, 0o000)

    # -- backend harness --------------------------------------------------
    @contextlib.contextmanager
    def _backend(self, name):
        """Force one backend; report whether the Python fallback ran.

        Args:
            name: "python" (both external tools missing), "external" (only rg,
                or else only grep, exists) or "default" (nothing patched).
        """
        used = {"fallback": False}

        def _fallback(*_args, **_kwargs):
            used["fallback"] = True
            return [], False, "python fallback ran unexpectedly", {"unreadable": 0, "binary": 0}

        if name == "python":
            with patch("ollamaquery2.shutil.which", return_value=None):
                yield used
            return
        if name == "default":
            yield used
            return
        real_which = shutil.which
        tool = "rg" if real_which("rg") else ("grep" if real_which("grep") else None)
        if tool is None:
            self.skipTest("neither rg nor grep is installed")

        def _which(name_to_find):
            return real_which(name_to_find) if name_to_find == tool else None

        with patch("ollamaquery2.shutil.which", side_effect=_which), \
             patch("ollamaquery2._grep_search_python_bounded", side_effect=_fallback):
            yield used

    def _run(self, backend, pattern=None, **args):
        """Execute the grep tool through one backend."""
        with self._backend(backend) as used:
            result = self.reg.execute("grep", {"pattern": pattern or self.BASE_PATTERN, **args})
        return result, used

    # -- row 2: parity ----------------------------------------------------
    def test_backend_parity_on_the_same_tree(self):
        """Same tree + same pattern: identical hits and identical rendering."""
        outputs = {}
        for backend in ("external", "python"):
            result, used = self._run(backend)
            self.assertTrue(result["success"], msg=f"{backend}: {result.get('error')}")
            self.assertFalse(used["fallback"], msg=f"{backend} ran the wrong backend")
            outputs[backend] = result["output"]
        self.assertEqual(outputs["python"], outputs["external"])
        self.assertIn("Found 4 matches", outputs["external"])
        for token in ("HIT_TEXT", "HIT_LATIN", "HIT_CRLF", "HIT_NEWLINE"):
            self.assertIn(token, outputs["external"])

    # -- row 1: the audit's scenario --------------------------------------
    def test_large_file_matches_first_middle_and_last(self):
        """A 4.5 MB file is searched whole: first, middle and last match."""
        self._make_big_log()
        for backend in ("external", "python"):
            result, used = self._run(backend, r"HIT_(FIRST|MIDDLE|LAST)")
            self.assertTrue(result["success"], msg=f"{backend}: {result.get('error')}")
            self.assertFalse(used["fallback"])
            for token in ("HIT_FIRST", "HIT_MIDDLE", "HIT_LAST"):
                self.assertIn(token, result["output"])
            self.assertEqual(result["output"].count("  Line "), 3)

    # -- rows 3 and 4: binary + encoding ----------------------------------
    def test_binary_file_is_skipped_by_every_backend(self):
        """A NUL-containing file must not be matched by only one backend."""
        self._write("with_nul.bin", b"prefix\x00HIT_BINARY text\n")
        for backend in ("external", "python"):
            result, _ = self._run(backend, r"HIT_BINARY")
            self.assertTrue(result["success"], msg=f"{backend}: {result.get('error')}")
            self.assertNotIn("with_nul.bin", result["output"])
            self.assertTrue(result["output"].startswith("No files found"), msg=result["output"])
            if backend == "python":
                self.assertIn("+1 binary file(s) skipped", result["output"])

    def test_latin1_file_is_matched_not_lost(self):
        """Invalid UTF-8 must not cost the line (errors='replace' on both sides)."""
        for backend in ("external", "python"):
            result, _ = self._run(backend, r"HIT_LATIN")
            self.assertIn("latin1.txt:", result["output"])
            self.assertIn("HIT_LATIN latin1", result["output"])

    # -- row 5: CRLF ------------------------------------------------------
    def test_crlf_lines_render_without_a_stray_carriage_return(self):
        for backend in ("external", "python"):
            result, _ = self._run(backend, r"HIT_CRLF")
            self.assertIn("HIT_CRLF windows line", result["output"])
            self.assertNotIn("\r", result["output"])

    # -- row 6: unreadable files ------------------------------------------
    def test_unreadable_file_is_counted_never_silent(self):
        """A match is reported AND the unreadable file is counted."""
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            self.skipTest("chmod 000 does not block root")
        self._make_unreadable("locked.txt", "HIT_LOCKED inside\n")
        result, _ = self._run("python")
        self.assertTrue(result["success"], msg=result.get("error"))
        self.assertIn("HIT_TEXT", result["output"])
        self.assertIn("+1 file(s) could not be read", result["output"])
        # Only the unreadable file matches: count, never a bare "No files found".
        result, _ = self._run("python", r"HIT_LOCKED")
        self.assertTrue(result["success"], msg=result.get("error"))
        self.assertTrue(result["output"].startswith("No files found"), msg=result["output"])
        self.assertIn("+1 file(s) could not be read", result["output"])

    def test_unreadable_file_degrades_to_the_counting_backend(self):
        """Default path: rg/grep exit 2 on EACCES, so the scan runs and counts."""
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            self.skipTest("chmod 000 does not block root")
        self._make_unreadable("locked.txt", "HIT_LOCKED inside\n")
        result, _ = self._run("default", r"HIT_TEXT")
        self.assertTrue(result["success"], msg=result.get("error"))
        self.assertIn("HIT_TEXT", result["output"])
        self.assertIn("could not be read", result["output"])

    # -- row 7: long lines ------------------------------------------------
    def test_long_line_match_is_reported(self):
        """The match is beyond MAX_GREP_LINE_LENGTH: report it, say it is cut."""
        self._write("long_line.txt", "a" * 15000 + "HIT_LONG" + "b" * 5000 + "\n")
        for backend in ("external", "python"):
            result, _ = self._run(backend, r"HIT_LONG")
            self.assertTrue(result["success"], msg=f"{backend}: {result.get('error')}")
            self.assertIn("long_line.txt:", result["output"])
            self.assertIn("Line 1:", result["output"])
            self.assertIn("chars]", result["output"])

    # -- rows 8 and 9: caps -----------------------------------------------
    def test_match_on_every_line_terminates_and_caps(self):
        self._make_big_log()
        for backend in ("external", "python"):
            started = time.monotonic()
            result, _ = self._run(backend, r"filler")
            self.assertLess(time.monotonic() - started, q.GREP_SEARCH_TIMEOUT)
            self.assertIn("capped at", result["output"])
            self.assertEqual(result["output"].count("  Line "), q.MAX_GREP_RESULTS)

    def test_output_char_cap_keeps_the_match_count(self):
        with open("wide.txt", "w") as fh:
            for _ in range(100):
                fh.write("HIT_WIDE " + "x" * 500 + "\n")
        for backend in ("external", "python"):
            result, _ = self._run(backend, r"HIT_WIDE")
            self.assertLessEqual(len(result["output"]), q.MAX_GREP_OUTPUT_CHARS)
            self.assertIn("Found 100 matches", result["output"])
            self.assertIn("Output capped", result["output"])

    # -- row 10: empty file, no trailing newline --------------------------
    def test_last_line_without_trailing_newline_is_searched(self):
        for backend in ("external", "python"):
            result, _ = self._run(backend, r"HIT_NEWLINE")
            self.assertIn("nonewline.txt:", result["output"])
            self.assertIn("HIT_NEWLINE tail", result["output"])

    # -- row 11: symlinks -------------------------------------------------
    def test_symlinks_do_not_hang_or_leak_outside_the_tree(self):
        """A loop, a dangling link, and a link out of the CWD."""
        os.symlink(".", "loop")
        os.symlink("missing-target", "dangling.txt")
        os.symlink("/etc/passwd", "outside_link.txt")
        result, _ = self._run("python", r"root:")
        self.assertTrue(result["success"], msg=result.get("error"))
        self.assertNotIn("root:x:", result["output"])
        self.assertIn("hidden by the path ACL", result["output"])
        result, _ = self._run("external", r"root:")
        self.assertTrue(result["success"], msg=result.get("error"))
        self.assertNotIn("root:x:", result["output"])

    # -- row 12: filenames ------------------------------------------------
    def test_spaced_and_non_ascii_filenames_are_verbatim(self):
        self._write("unicode name ü.txt", "HIT_UNICODE\n")
        self._write("with space.txt", "HIT_SPACE\n")
        for backend in ("external", "python"):
            result, _ = self._run(backend, r"HIT_(UNICODE|SPACE)")
            self.assertIn("unicode name ü.txt:", result["output"])
            self.assertIn("with space.txt:", result["output"])

    # -- row 13: determinism ----------------------------------------------
    def test_repeated_query_is_byte_identical(self):
        self._make_big_log()
        for backend in ("external", "python"):
            first, _ = self._run(backend, r"filler")
            second, _ = self._run(backend, r"filler")
            self.assertEqual(first["output"], second["output"])

    # -- row 14: bounded work ---------------------------------------------
    def test_python_scan_honours_the_deadline(self):
        """Finding B: the fallback gets the deadline rg/grep already had."""
        self._make_big_log()
        with patch.object(q, "GREP_SEARCH_TIMEOUT", 0.0):
            _hits, _trunc, err, _stats = q._grep_search_python(
                ".", r"filler", None, False, 100)
        self.assertTrue(err and err.startswith(q.GREP_TIMEOUT_PREFIX), msg=err)

    def test_python_scan_thread_abandons_a_call_that_outlasts_the_budget(self):
        """A single long call (I/O) is bounded by the wrapper, not the loop."""

        def _slow(*_args, **_kwargs):
            time.sleep(0.4)
            return [], False, None, {"unreadable": 0, "binary": 0}

        with patch.object(q, "GREP_SEARCH_TIMEOUT", 0.05), \
             patch.object(q, "_grep_search_python", side_effect=_slow):
            _hits, _trunc, err, _stats = q._grep_search_python_bounded(
                ".", r"filler", None, False, 100)
        self.assertTrue(err and err.startswith(q.GREP_TIMEOUT_PREFIX), msg=err)

    def test_timeout_stops_the_backend_chain(self):
        """A timeout must not be re-run with a slower engine."""
        ran = {"python": False}

        def _fallback(*_args, **_kwargs):
            ran["python"] = True
            return [], False, None, {"unreadable": 0, "binary": 0}

        timeout_err = q._grep_timeout_error(q.GREP_SEARCH_TIMEOUT)
        with patch.object(q, "_grep_run_external", return_value=([], timeout_err)), \
             patch.object(q, "_grep_search_python_bounded", side_effect=_fallback):
            _hits, _trunc, err, _stats = q._grep_search(".", "x", None, False, 10)
        self.assertEqual(err, timeout_err)
        self.assertFalse(ran["python"])

    # -- row 15: include glob, recursion, explicit file -------------------
    def test_include_glob_and_explicit_path_agree(self):
        os.mkdir("nested")
        self._write(os.path.join("nested", "inner.md"), "HIT_TEXT nested\n")
        for backend in ("external", "python"):
            globbed, _ = self._run(backend, r"HIT_TEXT", include="*.md")
            self.assertIn("nested" + os.sep + "inner.md:", globbed["output"])
            self.assertNotIn("text.py", globbed["output"])
            single, _ = self._run(backend, r"HIT_TEXT",
                                  path=os.path.join("nested", "inner.md"))
            self.assertIn("HIT_TEXT nested", single["output"])
            recursive, _ = self._run(backend, r"HIT_TEXT", path=".")
            self.assertIn("nested" + os.sep + "inner.md:", recursive["output"])
            self.assertIn("text.py:", recursive["output"])

    # -- row 16: match text that itself contains a colon ------------------
    def test_matched_text_containing_a_colon_is_reported(self):
        """A colon in the matched line must not discard the hit.

        The parse used to run from the right, so for every YAML/JSON/C line
        carrying a colon (`- name: ...`, `url: ...`, `// note:`) the line number
        was read as the text, `int()` failed, and the hit was dropped. The
        search then answered "No files found" for content `read_file` shows
        plainly — a false negative the model reports as "pattern absent".
        """
        self._write("role.yml",
                    "---\n"
                    "- name: Create robot for service provisioner\n"
                    "  url: https://quay/{{ token }}\n"
                    "  image: csi-external-provisioner\n")
        for backend in ("external", "python"):
            result, used = self._run(backend, r"provisioner")
            self.assertTrue(result["success"], msg=f"{backend}: {result.get('error')}")
            self.assertFalse(used["fallback"], msg=f"{backend} ran the wrong backend")
            self.assertIn("Found 2 matches", result["output"])
            self.assertIn("role.yml:", result["output"])
            self.assertNotIn("No files found", result["output"])
            self.assertIn("Line 2: - name: Create robot for service provisioner",
                          result["output"])
            self.assertIn("Line 4:   image: csi-external-provisioner", result["output"])

    def test_parse_match_keeps_colons_in_path_and_text(self):
        """`path:LINE:text` is anchored from the left; both sides keep colons."""
        self.assertEqual(q._grep_parse_match("a/b.yml:12:- name: x"),
                         ("a/b.yml", 12, "- name: x"))
        # A colon inside the path is not mistaken for the line separator.
        self.assertEqual(q._grep_parse_match("dir:1/x.txt:3:key: value"),
                         ("dir:1/x.txt", 3, "key: value"))
        self.assertEqual(q._grep_parse_match("file.c:9:  // note: why"),
                         ("file.c", 9, "  // note: why"))
        self.assertEqual(q._grep_parse_match("a.py:3:print('x'): done"),
                         ("a.py", 3, "print('x'): done"))
        self.assertIsNone(q._grep_parse_match("no line number here: nope"))
        self.assertIsNone(q._grep_parse_match("garbage"))


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

    def test_inline_code_is_not_exposed_on_the_command_line(self):
        """The source goes to a scratch file, not to `python3 -c`."""
        seen = {}

        def fake_run(command, timeout=10):
            seen["command"] = command
            script = shlex.split(command)[1]
            seen["exists_during_run"] = os.path.exists(script)
            with open(script, encoding="utf-8") as f:
                seen["code"] = f.read()
            return {"stdout": "ok", "stderr": "", "returncode": 0}

        with patch.object(self.reg.executor, "run", side_effect=fake_run):
            result = self.reg.execute(
                "run_python", {"code": "print('SECRET-CODE')"})
        self.assertTrue(result["success"], msg=result.get("error"))
        self.assertNotIn("SECRET-CODE", seen["command"])
        self.assertTrue(seen["exists_during_run"])
        self.assertIn("SECRET-CODE", seen["code"])


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


class TestContainerSandboxConfig(unittest.TestCase):
    """Image/container selection, pull memoization and exec mode (offline).

    Real podman runs are not required: what is pinned here is the argv we build
    and how many times we pay for a pull, which is where the ~3 s/command cost
    and the "wrong container" bugs live.
    """

    def _executor(self, **kwargs):
        """Build an Executor with an explicit runtime, ignoring the environment."""
        kwargs.setdefault("container_runtime", "podman")
        return q.Executor(**kwargs)

    def _fake_run(self, seen, image_present=False, pull_rc=0, pull_err="",
                  inspect_stdout="[]", cmd_rc=0, cmd_out="ok"):
        """subprocess.run stand-in that records argv per subcommand."""
        def _run(argv, **_kwargs):
            argv = list(argv)
            seen.append(argv)
            if argv[1:3] == ["image", "inspect"]:
                return types.SimpleNamespace(
                    returncode=0 if image_present else 1, stdout="", stderr="")
            if argv[1:2] == ["pull"]:
                return types.SimpleNamespace(returncode=pull_rc, stdout="", stderr=pull_err)
            if argv[1:2] == ["inspect"]:
                return types.SimpleNamespace(returncode=0, stdout=inspect_stdout, stderr="")
            return types.SimpleNamespace(returncode=cmd_rc, stdout=cmd_out, stderr="")
        return _run

    def test_local_image_is_never_pulled(self):
        """A locally built `localhost/...` image must not be pulled at all.

        Regression: pulling it makes podman treat `localhost/` as a registry and
        fail after a TCP timeout, which looked like a broken image even though
        `run` used the local copy happily.
        """
        seen = []
        with patch("ollamaquery2.subprocess.run",
                   side_effect=self._fake_run(seen, image_present=True)):
            ex = self._executor(mode="container", container_image="localhost/img:1")
            for _ in range(3):
                ex.run("true")
        self.assertEqual([a for a in seen if a[1:2] == ["pull"]], [])
        self.assertEqual(len([a for a in seen if a[1:3] == ["image", "inspect"]]), 1)
        self.assertEqual(ex._pull_state, "done")

    def test_absent_image_is_pulled_once_per_session(self):
        """The probe/pull pair is memoized: one per session, not one per command."""
        seen = []
        with patch("ollamaquery2.subprocess.run", side_effect=self._fake_run(seen)):
            ex = self._executor(mode="container", container_image="img:1")
            for _ in range(3):
                ex.run("true")
        pulls = [argv for argv in seen if argv[1:2] == ["pull"]]
        self.assertEqual(len(pulls), 1)
        self.assertEqual(pulls[0][2], "img:1")
        self.assertEqual(len([a for a in seen if a[1:3] == ["image", "inspect"]]), 1)

    def test_failed_pull_warns_once_and_is_not_retried(self):
        seen = []
        err = io.StringIO()
        fake = self._fake_run(seen, pull_rc=125, pull_err="Error: image not found",
                             cmd_rc=125, cmd_out="")
        with patch("ollamaquery2.subprocess.run", side_effect=fake), \
             contextlib.redirect_stderr(err):
            ex = self._executor(mode="container", container_image="nope:1")
            ex.run("true")
            ex.run("true")
        self.assertEqual(len([a for a in seen if a[1:2] == ["pull"]]), 1)
        self.assertEqual(len([a for a in seen if a[1:3] == ["image", "inspect"]]), 1)
        self.assertEqual(err.getvalue().count("[Sandbox] cannot pull"), 1)
        self.assertIn("nope:1", err.getvalue())
        self.assertIn("image not found", err.getvalue())

    def test_pull_exception_disables_further_pulls(self):
        seen = []

        def _run(argv, **_kwargs):
            argv = list(argv)
            seen.append(argv)
            if argv[1:3] == ["image", "inspect"]:
                raise OSError("podman missing")  # probe fails -> treated as absent
            if argv[1:2] == ["pull"]:
                raise OSError("podman missing")
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        err = io.StringIO()
        with patch("ollamaquery2.subprocess.run", side_effect=_run), \
             contextlib.redirect_stderr(err):
            ex = self._executor(mode="container", container_image="img:2")
            ex.run("true")
            ex.run("true")
        self.assertEqual(len([a for a in seen if a[1:2] == ["pull"]]), 1)
        self.assertEqual(len([a for a in seen if a[1:3] == ["image", "inspect"]]), 1)
        self.assertEqual(err.getvalue().count("[Sandbox] cannot pull"), 1)

    def test_running_container_uses_exec_and_never_pulls(self):
        seen = []
        with patch("ollamaquery2.subprocess.run", side_effect=self._fake_run(seen)):
            ex = self._executor(mode="container", container_name="dev-box")
            result = ex.run("echo hi")
        self.assertEqual(result["stdout"], "ok")
        self.assertFalse([a for a in seen if a[1:2] == ["pull"]])
        exec_argv = [a for a in seen if a[1:2] == ["exec"]][0]
        self.assertEqual(exec_argv[:3], ["podman", "exec", "dev-box"])
        self.assertNotIn("--rm", exec_argv)
        self.assertNotIn("-v", exec_argv)
        self.assertEqual(exec_argv[-3:], ["sh", "-c", "echo hi"])

    def test_running_container_uses_the_mount_of_the_cwd_as_workdir(self):
        cwd = os.path.realpath(os.getcwd())
        mounts = json.dumps([{"Source": "/tmp", "Destination": "/tmp"},
                             {"Source": cwd, "Destination": "/src/project"}])
        seen = []
        with patch("ollamaquery2.subprocess.run",
                   side_effect=self._fake_run(seen, inspect_stdout=mounts)):
            ex = self._executor(mode="container", container_name="dev-box")
            ex.run("pwd")
        exec_argv = [a for a in seen if a[1:2] == ["exec"]][0]
        self.assertEqual(exec_argv[2:5], ["-w", "/src/project", "dev-box"])

    def test_cwd_in_a_subdirectory_of_a_mounted_volume_maps_to_the_dest(self):
        """A CWD under a mounted volume maps to dest + the relative path.

        Exact-match-only resolution returned None here, so `exec` kept the
        container's workdir and the model's relative writes silently landed
        inside the container instead of the host project (GPT-5.6 review, P2).
        """
        cwd = os.path.realpath(os.getcwd())
        mounts = json.dumps([{"Source": os.path.dirname(cwd), "Destination": "/w"}])
        seen = []
        with patch("ollamaquery2.subprocess.run",
                   side_effect=self._fake_run(seen, inspect_stdout=mounts)):
            ex = self._executor(mode="container", container_name="dev-box")
            ex.run("pwd")
        exec_argv = [a for a in seen if a[1:2] == ["exec"]][0]
        expected = "/w/" + os.path.basename(cwd)
        self.assertEqual(exec_argv[2:5], ["-w", expected, "dev-box"])

    def test_deepest_matching_mount_wins(self):
        """Nested mounts resolve to the most specific source, not the first one."""
        cwd = os.path.realpath(os.getcwd())
        mounts = json.dumps([
            {"Source": os.path.dirname(cwd), "Destination": "/outer"},
            {"Source": cwd, "Destination": "/exact"},
        ])
        seen = []
        with patch("ollamaquery2.subprocess.run",
                   side_effect=self._fake_run(seen, inspect_stdout=mounts)):
            ex = self._executor(mode="container", container_name="dev-box")
            ex.run("pwd")
        exec_argv = [a for a in seen if a[1:2] == ["exec"]][0]
        self.assertEqual(exec_argv[2:5], ["-w", "/exact", "dev-box"])

    def test_a_root_source_mount_matches_any_cwd(self):
        """A mount of `/` covers every CWD (its prefix must not become "//").

        Review P3: `cwd.startswith(source + os.sep)` produced "//" for a source of
        "/", so mounting the whole filesystem (e.g. -v /:/host) never matched and
        exec silently kept the container workdir.
        """
        cwd = os.path.realpath(os.getcwd())
        mounts = json.dumps([{"Source": "/", "Destination": "/host"}])
        seen = []
        with patch("ollamaquery2.subprocess.run",
                   side_effect=self._fake_run(seen, inspect_stdout=mounts)):
            ex = self._executor(mode="container", container_name="dev-box")
            ex.run("pwd")
        exec_argv = [a for a in seen if a[1:2] == ["exec"]][0]
        expected = "/host" + cwd.rstrip("/")
        self.assertEqual(exec_argv[2:5], ["-w", expected, "dev-box"])

    def test_a_specific_mount_beats_the_root_mount(self):
        """With `/` and a nested volume both mounted, the specific one wins."""
        cwd = os.path.realpath(os.getcwd())
        mounts = json.dumps([{"Source": "/", "Destination": "/host"},
                             {"Source": cwd, "Destination": "/exact"}])
        seen = []
        with patch("ollamaquery2.subprocess.run",
                   side_effect=self._fake_run(seen, inspect_stdout=mounts)):
            ex = self._executor(mode="container", container_name="dev-box")
            ex.run("pwd")
        exec_argv = [a for a in seen if a[1:2] == ["exec"]][0]
        self.assertEqual(exec_argv[2:5], ["-w", "/exact", "dev-box"])

    def test_running_container_without_a_matching_mount_keeps_its_workdir(self):
        seen = []
        with patch("ollamaquery2.subprocess.run",
                   side_effect=self._fake_run(seen, inspect_stdout='[{"Source": "/elsewhere"}]')):
            ex = self._executor(mode="container", container_name="dev-box")
            ex.run("pwd")
        exec_argv = [a for a in seen if a[1:2] == ["exec"]][0]
        self.assertNotIn("-w", exec_argv)

    def test_run_with_an_image_mounts_the_cwd_and_the_scratch_dir(self):
        seen = []
        with patch("ollamaquery2.subprocess.run", side_effect=self._fake_run(seen)):
            ex = self._executor(mode="container", container_image="img:3")
            ex.run("id")
        argv = seen[-1]
        self.assertEqual(argv[1:3], ["run", "--rm"])
        mounts = [argv[i + 1] for i, part in enumerate(argv) if part == "-v"]
        self.assertTrue(mounts[0].endswith(":/workspace:Z"))
        scratch = q._scratch_root()
        # Same absolute path on both sides: a path write_file handed to the model
        # must also work inside the throwaway container.
        self.assertIn(f"{scratch}:{scratch}:Z", mounts)
        self.assertEqual(argv[argv.index("-w") + 1], "/workspace")
        self.assertEqual(argv[-4:], ["img:3", "sh", "-c", "id"])

    def test_exec_mode_warns_once_when_the_scratch_dir_is_not_mounted(self):
        """`exec` cannot add mounts, so an invisible scratch dir must be reported."""
        mounts = json.dumps([{"Source": os.getcwd(), "Destination": "/workspace"}])
        seen, err = [], io.StringIO()
        with patch("ollamaquery2.subprocess.run",
                   side_effect=self._fake_run(seen, inspect_stdout=mounts)), \
             contextlib.redirect_stderr(err):
            ex = self._executor(mode="container", container_name="dev-box")
            ex.run("pwd")
            ex.run("pwd")
        self.assertEqual(err.getvalue().count("[Sandbox]"), 1)
        self.assertIn(q._scratch_root(), err.getvalue())
        self.assertIn("-v", err.getvalue())

    def test_exec_mode_is_quiet_when_the_temp_area_is_mounted(self):
        mounts = json.dumps([{"Source": q.TEMP_ROOT, "Destination": q.TEMP_ROOT}])
        seen, err = [], io.StringIO()
        with patch("ollamaquery2.subprocess.run",
                   side_effect=self._fake_run(seen, inspect_stdout=mounts)), \
             contextlib.redirect_stderr(err):
            ex = self._executor(mode="container", container_name="dev-box")
            ex.run("pwd")
        self.assertNotIn("[Sandbox]", err.getvalue())

    def test_throwaway_container_has_a_default_resource_ceiling(self):
        """1 cpu / 1g by default: a stray model process must not cost the host."""
        seen = []
        with patch("ollamaquery2.subprocess.run", side_effect=self._fake_run(seen)):
            ex = self._executor(mode="container", container_image="img:7")
            ex.run("true")
        argv = seen[-1]
        self.assertEqual(argv[3:7], ["--memory", "1g", "--cpus", "1"])
        self.assertEqual(argv[argv.index("--memory") + 1], q.DEFAULT_CONTAINER_MEMORY)
        self.assertEqual(ex.describe_limits(), "1 cpu, 1g memory")
        self.assertEqual(ex.describe(), "container (image img:7; 1 cpu, 1g memory)")

    def test_limits_can_be_overridden(self):
        seen = []
        with patch("ollamaquery2.subprocess.run", side_effect=self._fake_run(seen)):
            ex = self._executor(mode="container", container_image="img:8",
                                container_memory="4g", container_cpus="2")
            ex.run("true")
        argv = seen[-1]
        self.assertEqual(argv[3:7], ["--memory", "4g", "--cpus", "2"])
        self.assertEqual(ex.describe_limits(), "2 cpu, 4g memory")

    def test_exec_mode_never_sets_limits(self):
        """The running container's limits belong to whoever started it."""
        seen = []
        with patch("ollamaquery2.subprocess.run", side_effect=self._fake_run(seen)):
            ex = self._executor(mode="container", container_name="dev-box",
                                container_memory="4g", container_cpus="2")
            ex.run("true")
        exec_argv = [a for a in seen if a[1:2] == ["exec"]][0]
        self.assertNotIn("--memory", exec_argv)
        self.assertNotIn("--cpus", exec_argv)
        self.assertEqual(ex.describe_limits(), "the container's own limits")
        self.assertEqual(ex.describe(),
                         "container (exec into dev-box; the container's own limits)")

    def test_cli_flags_reach_the_executor_via_the_context(self):
        args = q._build_parser().parse_args(
            ["--sandbox-image", "img:4", "--sandbox-runtime", "docker",
             "--sandbox-memory", "3g", "--sandbox-cpus", "1.5"])
        ctx = types.SimpleNamespace()
        q._apply_sandbox_args(ctx, args)
        self.assertEqual(ctx.executor_mode, "container")
        ex = q.CommandContext.create_executor(ctx)
        self.assertEqual((ex.mode, ex.image, ex.runtime, ex.container_name),
                         ("container", "img:4", "docker", None))
        self.assertEqual((ex.memory, ex.cpus), ("3g", "1.5"))

    def test_limit_env_vars_still_work(self):
        with patch.dict(os.environ, {"OLLAMAQUERY_CONTAINER_MEMORY": "512m",
                                     "OLLAMAQUERY_CONTAINER_CPUS": "0.5"}):
            ex = q.Executor(mode="container")
        self.assertEqual((ex.memory, ex.cpus), ("512m", "0.5"))

    def test_sandbox_container_flag_beats_the_image(self):
        args = q._build_parser().parse_args(
            ["--sandbox-container", "dev-box", "--sandbox-image", "img:5"])
        ctx = types.SimpleNamespace()
        q._apply_sandbox_args(ctx, args)
        ex = q.CommandContext.create_executor(ctx)
        self.assertTrue(ex.describe().startswith("container (exec into dev-box;"))
        self.assertEqual(ex.mode, "container")

    def test_no_sandbox_flag_keeps_host_mode(self):
        args = q._build_parser().parse_args([])
        ctx = types.SimpleNamespace()
        q._apply_sandbox_args(ctx, args)
        self.assertFalse(hasattr(ctx, "executor_mode"))
        ex = q.CommandContext.create_executor(
            types.SimpleNamespace(executor_mode="host", container_runtime=None,
                                  container_image=None, container_name=None))
        self.assertEqual(ex.mode, "host")

    def test_env_vars_still_configure_the_executor(self):
        with patch.dict(os.environ, {"OLLAMAQUERY_CONTAINER_IMAGE": "env:9",
                                     "OLLAMAQUERY_CONTAINER_RT": "docker"}):
            ex = q.Executor(mode="container")
            explicit = q.Executor(mode="container", container_image="arg:1")
        self.assertEqual((ex.image, ex.runtime), ("env:9", "docker"))
        self.assertEqual(explicit.image, "arg:1")

    def test_default_image_and_runtime(self):
        env = {k: v for k, v in os.environ.items()
               if not k.startswith("OLLAMAQUERY_CONTAINER")}
        with patch.dict(os.environ, env, clear=True):
            ex = q.Executor(mode="container")
        self.assertEqual(ex.image, q.DEFAULT_CONTAINER_IMAGE)
        self.assertEqual(ex.runtime, q.DEFAULT_CONTAINER_RUNTIME)

    def test_sandbox_toggle_reports_the_target(self):
        loop = object.__new__(q.ChatLoop)
        loop.executor = self._executor(mode="host", container_image="img:6")
        loop.tool_registry = types.SimpleNamespace(executor=loop.executor)
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            loop._agentic_toggle_sandbox()
        self.assertEqual(loop.executor.mode, "container")
        self.assertEqual(loop.tool_registry.executor.mode, "container")
        self.assertIn("img:6", err.getvalue())


class TestBwrapSandboxConfig(unittest.TestCase):
    """bwrap engine: mount policy, probe fail-closed, CLI/env wiring (offline).

    The mount policy is the contract in this mode (the shell gate is bypassed),
    so what is pinned here is the exact argv and the refusal behaviour, not a
    real sandbox. The opt-in live class below proves the daemon-free writes.
    """

    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="bwrap-home-")
        self.cwd = os.path.realpath(os.getcwd())
        self.scratch = os.path.realpath(q._scratch_root())

    def tearDown(self):
        shutil.rmtree(self.home, ignore_errors=True)

    def _executor(self, **kwargs):
        kwargs.setdefault("bwrap_home", self.home)
        kwargs.setdefault("bwrap_project", "rw")
        kwargs.setdefault("bwrap_net", "shared")
        ex = q.Executor(mode="bwrap", **kwargs)
        ex.bwrap_path = "/usr/bin/bwrap"
        ex._bwrap_state = "ok"          # skip the probe unless a test wants it
        return ex

    @staticmethod
    def _is_probe(argv) -> bool:
        """The probe is the production argv run with `true` as the command."""
        return "--unshare-all" in argv and argv[-1] == "true"

    def _fake_run(self, seen, probe_rc=0, cmd_out="ok", cmd_err=""):
        """subprocess.run stand-in recording argv and answering the probe."""
        def _run(argv, **_kwargs):
            argv = list(argv)
            seen.append(argv)
            if self._is_probe(argv):        # the bwrap probe
                return types.SimpleNamespace(
                    returncode=probe_rc, stdout="",
                    stderr="probe denied" if probe_rc else "")
            return types.SimpleNamespace(returncode=0, stdout=cmd_out, stderr=cmd_err)
        return _run

    def _argv_for(self, command, **kwargs):
        seen = []
        with patch("ollamaquery2.subprocess.run", side_effect=self._fake_run(seen)):
            ex = self._executor(**kwargs)
            result = ex.run(command, timeout=5)
        return seen[-1], result

    def test_bwrap_argv_implements_the_mount_policy(self):
        argv, result = self._argv_for("echo hi")
        self.assertEqual(result["stdout"], "ok")
        self.assertEqual(argv[0], "/usr/bin/bwrap")
        self.assertIn("--unshare-all", argv)
        self.assertIn("--share-net", argv)
        self.assertIn("--die-with-parent", argv)
        self.assertIn("--new-session", argv)
        self.assertEqual(argv[argv.index("--cap-drop") + 1], "ALL")
        self.assertIn("--clearenv", argv)
        env = {}
        for i, part in enumerate(argv):
            if part == "--setenv":
                env[argv[i + 1]] = argv[i + 2]
        self.assertEqual(env["HOME"], self.home)
        self.assertTrue(env["PATH"].startswith("/usr/local/bin"))
        self.assertEqual(env["GIT_OPTIONAL_LOCKS"], "0")
        self.assertEqual(env["GIT_TERMINAL_PROMPT"], "0")
        self.assertEqual(env["TMPDIR"], self.scratch)
        for path in q.BWRAP_READ_ONLY_BINDS:
            self.assertIn(["--ro-bind-try", path, path],
                          [argv[i:i + 3] for i in range(len(argv) - 2)])
        self.assertEqual(argv[argv.index("--tmpfs") + 1], "/tmp")
        self.assertIn("/var/tmp", argv)
        self.assertIn("--dev", argv)
        self.assertIn("/dev", argv)
        self.assertIn("--proc", argv)
        self.assertIn("/proc", argv)
        # The project keeps its host path, so model paths and ACL stay coherent.
        self.assertIn(["--bind", self.cwd, self.cwd],
                      [argv[i:i + 3] for i in range(len(argv) - 2)])
        self.assertIn(["--bind", self.scratch, self.scratch],
                      [argv[i:i + 3] for i in range(len(argv) - 2)])
        self.assertIn(["--bind", self.home, self.home],
                      [argv[i:i + 3] for i in range(len(argv) - 2)])
        self.assertEqual(argv[argv.index("--chdir") + 1], self.cwd)
        self.assertEqual(argv[-3:], [q._bwrap_shell(), "-c", "echo hi"])

    def test_probe_runs_the_production_argv(self):
        """P1-5: the probe must exercise the argv real commands use.

        The old probe was `bwrap --dev /dev --ro-bind / / true`: a host could
        pass it and still refuse `--proc`, `--unshare-all`, `--clearenv` or the
        shell, while the banner promised a working sandbox.
        """
        seen = []
        with patch("ollamaquery2.shutil.which", return_value="/usr/bin/bwrap"), \
             patch("ollamaquery2.subprocess.run", side_effect=self._fake_run(seen)):
            ex = q.Executor(mode="bwrap", bwrap_project="ro", bwrap_net="none",
                            bwrap_home=self.home)
            ex.run("echo hi", timeout=5)
        probe = seen[0]
        self.assertTrue(self._is_probe(probe))
        for flag in ("--unshare-all", "--unshare-net", "--clearenv",
                     "--cap-drop", "--tmpfs", "--proc"):
            self.assertIn(flag, probe)
        triples = [probe[i:i + 3] for i in range(len(probe) - 2)]
        self.assertIn(["--ro-bind", self.cwd, self.cwd], triples)
        self.assertIn(["--bind", self.scratch, self.scratch], triples)
        self.assertEqual(probe[-3:], [q._bwrap_shell(), "-c", "true"])

    def test_filesystem_root_as_the_project_is_refused(self):
        """P1-9: `--bind / /` would mount the whole host (rw in work)."""
        seen = []
        with patch("ollamaquery2.subprocess.run", side_effect=self._fake_run(seen)), \
             patch("ollamaquery2.os.getcwd", return_value="/"):
            result = self._executor().run("echo hi", timeout=5)
        self.assertEqual(result["returncode"], -1)
        self.assertIn("filesystem root", result["stderr"])
        self.assertEqual(seen, [])

    def test_scratch_inside_a_read_only_project_is_refused(self):
        """P1-6: TMPDIR=$PWD/tmp would be a writable hole in `exec`."""
        fake_scratch = os.path.join(self.cwd, "tmp")
        with patch.object(q, "_scratch_root", return_value=fake_scratch), \
             patch("ollamaquery2.subprocess.run", side_effect=self._fake_run([])):
            refused = self._executor(bwrap_project="ro").run("echo hi", timeout=5)
            allowed = self._executor(bwrap_project="rw").run("echo hi", timeout=5)
        self.assertEqual(refused["returncode"], -1)
        self.assertIn("session scratch", refused["stderr"])
        self.assertEqual(allowed["returncode"], 0)   # a rw project is already writable

    def test_sandbox_home_inside_a_read_only_project_is_refused(self):
        """P1-6: CWD=$HOME puts the sandbox HOME inside the ro project."""
        inside_home = os.path.join(self.cwd, "sandbox-home")
        os.makedirs(inside_home, exist_ok=True)
        try:
            with patch("ollamaquery2.subprocess.run", side_effect=self._fake_run([])):
                refused = self._executor(bwrap_project="ro",
                                         bwrap_home=inside_home).run("echo hi", timeout=5)
        finally:
            shutil.rmtree(inside_home, ignore_errors=True)
        self.assertEqual(refused["returncode"], -1)
        self.assertIn("sandbox HOME", refused["stderr"])

    def test_run_is_not_mounted_whole_because_sockets_survive_ro_binds(self):
        """A ro-bind of /run still permits connect(2) on the host's Unix sockets.

        Reproduced 2026-11-22: with the old `--ro-bind-try /run /run`, a command
        in the exec preset (`--unshare-net`, `--cap-drop ALL`) reached
        `/run/user/<uid>/podman/podman.sock` and `/_ping` answered "OK", i.e.
        the sandbox could drive the host container runtime. Only the resolver
        state may be mounted.
        """
        argv, _ = self._argv_for("echo hi")
        triples = [argv[i:i + 3] for i in range(len(argv) - 2)]
        self.assertNotIn(["--ro-bind-try", "/run", "/run"], triples)
        self.assertNotIn(["--bind", "/run", "/run"], triples)
        self.assertNotIn("/run", q.BWRAP_READ_ONLY_BINDS)
        for path in q.BWRAP_RESOLVER_BINDS:
            self.assertIn(["--ro-bind-try", path, path], triples)
        self.assertIn("/run/systemd/resolve", q.BWRAP_RESOLVER_BINDS)

    def test_project_ro_uses_ro_bind_and_extra_rw_comes_after(self):
        build = os.path.join(self.cwd, "build")
        argv, _ = self._argv_for("echo hi", bwrap_project="ro", bwrap_rw=[build])
        pairs = [argv[i:i + 3] for i in range(len(argv) - 2)]
        self.assertIn(["--ro-bind", self.cwd, self.cwd], pairs)
        self.assertNotIn(["--bind", self.cwd, self.cwd], pairs)
        ro_index = next(i for i, part in enumerate(argv)
                        if part == "--ro-bind" and argv[i + 1] == self.cwd)
        rw_index = next(i for i, part in enumerate(argv)
                        if part == "--bind-try" and argv[i + 1] == build)
        self.assertLess(ro_index, rw_index)

    def test_net_none_unshares_the_network(self):
        argv, _ = self._argv_for("echo hi", bwrap_net="none")
        self.assertIn("--unshare-net", argv)
        self.assertNotIn("--share-net", argv)

    def test_extra_binds_are_parsed_and_try_mounted(self):
        ro = os.path.join(self.cwd, "ro-dir")
        rw = os.path.join(self.cwd, "rw-dir")
        argv, _ = self._argv_for("echo hi", bwrap_binds=[(ro, "ro"), (rw, "rw")])
        pairs = [argv[i:i + 3] for i in range(len(argv) - 2)]
        self.assertIn(["--ro-bind-try", ro, ro], pairs)
        self.assertIn(["--bind-try", rw, rw], pairs)

    def test_gate_is_not_called_in_bwrap_mode(self):
        with patch("ollamaquery2.check_shell_approval") as gate, \
             patch("ollamaquery2.subprocess.run", side_effect=self._fake_run([])):
            ex = self._executor()
            # A command the host gate would deny outright: the mounts are the
            # barrier, so the gate must not even be consulted.
            result = ex.run("rm -rf /", timeout=5)
        gate.assert_not_called()
        self.assertEqual(result["returncode"], 0)

    def test_missing_bwrap_refuses_and_never_runs_on_the_host(self):
        seen = []
        with patch("ollamaquery2.shutil.which", return_value=None), \
             patch("ollamaquery2.subprocess.run", side_effect=self._fake_run(seen)):
            ex = q.Executor(mode="bwrap", bwrap_home=self.home)
            result = ex.run("echo hi", timeout=5)
        self.assertEqual(result["returncode"], -1)
        self.assertIn("not installed", result["stderr"])
        self.assertEqual(seen, [])

    def test_missing_private_home_is_refused_before_the_probe(self):
        """P2: an uncreatable sandbox HOME is named, not reported by bwrap.

        Caching the path after a silent `_ensure_private_dir` failure made bwrap
        exit with "source path missing", which reads like a broken bwrap install
        and sends the user debugging the wrong thing.
        """
        seen, err = [], io.StringIO()
        saved_cache = list(q._BWRAP_HOME_CACHE)
        saved_error = list(q._BWRAP_HOME_ERROR)
        q._BWRAP_HOME_CACHE[:] = []
        q._BWRAP_HOME_ERROR[:] = []
        try:
            with patch.object(q, "_private_dir_error", return_value="Permission denied"), \
                 patch("ollamaquery2.subprocess.run", side_effect=self._fake_run(seen)), \
                 contextlib.redirect_stderr(err):
                ex = q.Executor(mode="bwrap", bwrap_project="rw", bwrap_net="shared")
                first = ex.run("echo hi", timeout=5)
                second = ex.run("echo hi", timeout=5)
        finally:
            q._BWRAP_HOME_CACHE[:] = saved_cache
            q._BWRAP_HOME_ERROR[:] = saved_error
        self.assertEqual(first["returncode"], -1)
        self.assertEqual(second["returncode"], -1)
        self.assertIn("private sandbox HOME", first["stderr"])
        self.assertIn("Permission denied", first["stderr"])
        self.assertNotIn("not installed", first["stderr"])
        self.assertEqual(seen, [])          # refused BEFORE the probe
        self.assertEqual(err.getvalue().count("[Sandbox] REFUSED"), 1)

    def test_home_creation_failure_is_not_cached(self):
        """P2: the path is cached only once the directory actually exists."""
        saved_cache = list(q._BWRAP_HOME_CACHE)
        saved_error = list(q._BWRAP_HOME_ERROR)
        saved_dir = q.BWRAP_SANDBOX_DIR
        sandbox_dir = tempfile.mkdtemp(prefix="bwrap-root-")
        q._BWRAP_HOME_CACHE[:] = []
        q._BWRAP_HOME_ERROR[:] = []
        try:
            q.BWRAP_SANDBOX_DIR = sandbox_dir
            with patch.object(q, "_private_dir_error", return_value="No space left"):
                self.assertIn("No space left", q._bwrap_home_error())
            self.assertEqual(q._BWRAP_HOME_CACHE, [])   # nothing cached on failure
            self.assertEqual(q._bwrap_home_error(), "")  # retried, now succeeds
            self.assertEqual(len(q._BWRAP_HOME_CACHE), 1)
            self.assertTrue(os.path.isdir(q._BWRAP_HOME_CACHE[0]))
        finally:
            q.BWRAP_SANDBOX_DIR = saved_dir
            q._BWRAP_HOME_CACHE[:] = saved_cache
            q._BWRAP_HOME_ERROR[:] = saved_error
            shutil.rmtree(sandbox_dir, ignore_errors=True)

    def test_probe_failure_refuses_and_warns_once(self):
        seen, err = [], io.StringIO()
        with patch("ollamaquery2.shutil.which", return_value="/usr/bin/bwrap"), \
             patch("ollamaquery2.subprocess.run",
                   side_effect=self._fake_run(seen, probe_rc=1)), \
             contextlib.redirect_stderr(err):
            ex = q.Executor(mode="bwrap", bwrap_home=self.home)
            first = ex.run("echo hi", timeout=5)
            second = ex.run("echo hi", timeout=5)
        self.assertEqual(first["returncode"], -1)
        self.assertEqual(second["returncode"], -1)
        self.assertIn("bwrap unavailable", first["stderr"])
        # Exactly one warning, one probe, and no command ever ran.
        self.assertEqual(err.getvalue().count("[Sandbox] bwrap unavailable"), 1)
        self.assertEqual(len([a for a in seen if self._is_probe(a)]), 1)
        self.assertEqual(len(seen), 1)

    def test_probe_is_memoized(self):
        seen = []
        with patch("ollamaquery2.shutil.which", return_value="/usr/bin/bwrap"), \
             patch("ollamaquery2.subprocess.run", side_effect=self._fake_run(seen)):
            ex = q.Executor(mode="bwrap", bwrap_home=self.home)
            ex.run("echo one", timeout=5)
            ex.run("echo two", timeout=5)
        self.assertEqual(len([a for a in seen if self._is_probe(a)]), 1)
        self.assertEqual(len(seen), 3)   # probe + two commands

    def test_describe_reports_the_policy_and_the_missing_ceiling(self):
        ex = self._executor(bwrap_project="ro", bwrap_net="none")
        self.assertEqual(ex.describe_target(), "project ro, net none, host ro")
        self.assertEqual(ex.describe_limits(), "no cgroup cap (v1)")
        self.assertEqual(ex.describe(),
                         "bwrap (project ro, net none, host ro; no cgroup cap (v1))")

    def test_describe_lists_extra_bind_mounts(self):
        """P1-7: the extra mounts are the contract, so they cannot be hidden."""
        ex = self._executor(bwrap_project="ro", bwrap_net="none",
                            bwrap_binds=[("/data", "ro"), ("/srv/x", "rw")],
                            bwrap_rw=["/build"])
        described = ex.describe_target()
        self.assertIn("/data (ro)", described)
        self.assertIn("/srv/x (rw)", described)
        self.assertIn("/build (rw)", described)
        self.assertIn("extra:", described)

    def test_cli_flags_reach_the_executor(self):
        args = q._build_parser().parse_args([
            "--sandbox-engine", "bwrap", "--sandbox-bwrap-project", "ro",
            "--sandbox-bwrap-net", "none",
            "--sandbox-bwrap-bind", f"{self.cwd}/a:rw",
            "--sandbox-bwrap-bind", f"{self.cwd}/b",
            "--sandbox-bwrap-rw", f"{self.cwd}/build"])
        ctx = types.SimpleNamespace()
        q._apply_sandbox_args(ctx, args)
        self.assertEqual(ctx.executor_mode, "bwrap")
        self.assertEqual((ctx.bwrap_project, ctx.bwrap_net), ("ro", "none"))
        self.assertEqual(ctx.bwrap_binds, [(os.path.realpath(f"{self.cwd}/a"), "rw"),
                                           (os.path.realpath(f"{self.cwd}/b"), "ro")])
        self.assertEqual(ctx.bwrap_rw, [os.path.realpath(f"{self.cwd}/build")])
        ex = q.CommandContext.create_executor(ctx)
        self.assertEqual((ex.mode, ex.bwrap_project, ex.bwrap_net),
                         ("bwrap", "ro", "none"))
        self.assertEqual(ex.bwrap_binds, ctx.bwrap_binds)

    def test_engine_env_var_and_defaults(self):
        with patch.dict(os.environ, {"OLLAMAQUERY_SANDBOX_ENGINE": "bwrap",
                                     "OLLAMAQUERY_BWRAP_PROJECT": "ro",
                                     "OLLAMAQUERY_BWRAP_NET": "none"}):
            args = q._build_parser().parse_args([])
            ctx = types.SimpleNamespace()
            q._apply_sandbox_args(ctx, args)
        self.assertEqual((ctx.executor_mode, ctx.sandbox_engine_override),
                         ("bwrap", "bwrap"))
        self.assertEqual((ctx.bwrap_project, ctx.bwrap_net), ("ro", "none"))

    def test_engine_conflicts_with_container_flags(self):
        args = q._build_parser().parse_args(
            ["--sandbox-engine", "bwrap", "--sandbox-image", "img:1"])
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                q._apply_sandbox_args(types.SimpleNamespace(), args)

    def test_invalid_bind_mode_is_an_error(self):
        args = q._build_parser().parse_args(
            ["--sandbox-engine", "bwrap", "--sandbox-bwrap-bind", "/tmp/x:bogus"])
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                q._apply_sandbox_args(types.SimpleNamespace(), args)

    def test_bwrap_toggle_sets_both_executors(self):
        ctx = q.CommandContext()
        ctx.executor_mode = "host"
        loop = object.__new__(q.ChatLoop)
        loop.ctx = ctx
        loop.executor = q.Executor(mode="host")
        loop.tool_registry = types.SimpleNamespace(executor=loop.executor)
        with patch.object(q.Executor, "_bwrap_probe", return_value=True), \
             contextlib.redirect_stderr(io.StringIO()):
            loop._agentic_toggle_sandbox(['/agentic', 'sandbox', 'bwrap'])
        self.assertEqual(loop.executor.mode, "bwrap")
        self.assertEqual(loop.tool_registry.executor.mode, "bwrap")
        self.assertEqual(ctx.executor_mode, "bwrap")

    def test_bwrap_toggle_refuses_when_the_probe_fails(self):
        ctx = q.CommandContext()
        ctx.executor_mode = "host"
        loop = object.__new__(q.ChatLoop)
        loop.ctx = ctx
        loop.executor = q.Executor(mode="host")
        loop.tool_registry = types.SimpleNamespace(executor=loop.executor)
        with patch.object(q.Executor, "_bwrap_probe", return_value=False), \
             contextlib.redirect_stderr(io.StringIO()):
            loop._agentic_toggle_sandbox(['/agentic', 'sandbox', 'bwrap'])
        self.assertEqual(loop.executor.mode, "host")
        self.assertEqual(ctx.executor_mode, "host")

    def test_file_tools_stay_in_the_harness(self):
        """The documented limit: write_file is governed by the Path ACL, not bwrap."""
        ctx = q.CommandContext()
        ctx.preset = "work"
        ctx.auto_confirm = True
        td = tempfile.mkdtemp(prefix="bwrap-acl-")
        try:
            target = os.path.join(td, "written.txt")
            ctx.path_acl.add("write", "allow", td, source="session")
            ctx.path_acl.add("read", "allow", td, source="session")
            registry = ctx.create_tool_registry()
            registry.executor = self._executor(bwrap_project="ro")
            with patch.object(registry.executor, "run",
                              side_effect=AssertionError("file tools must not use bwrap")):
                result = registry.execute("write_file",
                                          {"file_path": target, "content": "hello"})
            self.assertTrue(result["success"], msg=result.get("error", ""))
            with open(target) as fh:
                self.assertEqual(fh.read(), "hello")
        finally:
            shutil.rmtree(td, ignore_errors=True)


def _write(path: str, text: str) -> None:
    """Write a fixture file and close it (keeps ResourceWarnings meaningful)."""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


class TestAgenticPresets(unittest.TestCase):
    """The four MVP presets: one resolver, four projections, no drift."""

    def setUp(self):
        self.ctx = q.CommandContext()
        self.ctx.system_prompt = "BASE PROMPT"
        self.ctx._saved_system_prompt = None
        self.ctx._plan_prev_prompt = None
        self.ctx._plan_prev_preset = None
        self.ctx.agentic_mode = False
        self.ctx.plan_mode = False
        self.ctx.plan_run = False
        self.ctx.preset = "work"
        self.ctx.executor_mode = "host"
        self.ctx.sandbox_engine_override = None
        self.ctx.auto_confirm = False
        self.ctx.bwrap_project = q.DEFAULT_BWRAP_PROJECT
        self.ctx.bwrap_net = q.DEFAULT_BWRAP_NET
        self.ctx.bwrap_project_override = None
        self.ctx.bwrap_net_override = None
        self.ctx.bwrap_binds = []
        self.ctx.bwrap_rw = []
        self.ctx._plan_prev_state = None

    def tearDown(self):
        self.setUp()

    def _loop(self):
        loop = object.__new__(q.ChatLoop)
        loop.ctx = self.ctx
        loop.executor = self.ctx.create_executor()
        loop.tool_registry = self.ctx.create_tool_registry()
        return loop

    def test_resolver_projections_do_not_drift(self):
        read = q.resolve_preset("read")
        exec_ = q.resolve_preset("exec")
        work = q.resolve_preset("work")
        trusted = q.resolve_preset("trusted")
        self.assertEqual(read.engine, "none")
        self.assertEqual(read.tools, frozenset(q.PLAN_MODE_TOOLS))
        self.assertIn("run_command", exec_.tools)
        self.assertIn("run_python", exec_.tools)
        self.assertNotIn("write_file", exec_.tools)
        self.assertNotIn("fetch_url", exec_.tools)
        self.assertEqual((exec_.engine, exec_.project, exec_.net),
                         ("bwrap", "ro", "none"))
        self.assertIsNone(work.tools)
        self.assertEqual((work.engine, work.project, work.net),
                         ("bwrap", "rw", "shared"))
        self.assertIsNone(trusted.tools)
        self.assertEqual(trusted.engine, "host")
        self.assertEqual(q.resolve_preset("plan"), read)
        self.assertEqual(q.resolve_preset("full"), trusted)
        self.assertEqual(q.resolve_preset("auto"), trusted)

    def test_unknown_preset_raises(self):
        with self.assertRaises(ValueError):
            q.resolve_preset("yolo-max")

    def test_read_preset_is_the_existing_plan(self):
        loop = self._loop()
        with contextlib.redirect_stderr(io.StringIO()):
            loop.run_handle_agentic("/agentic preset read")
        self.assertTrue(self.ctx.plan_mode)
        self.assertFalse(self.ctx.plan_run)
        self.assertEqual(self.ctx.preset, "read")
        self.assertIn("Read-only planning mode", self.ctx.system_prompt)
        self.assertNotIn("run_command", q.agentic_active_tools(self.ctx))

    def test_exec_preset_is_bwrap_ro_without_write_tools(self):
        loop = self._loop()
        with patch.object(q.Executor, "_bwrap_probe", return_value=True), \
             contextlib.redirect_stderr(io.StringIO()):
            loop.run_handle_agentic("/agentic preset exec")
        self.assertEqual(self.ctx.preset, "exec")
        self.assertEqual(self.ctx.executor_mode, "bwrap")
        self.assertEqual((self.ctx.bwrap_project, self.ctx.bwrap_net), ("ro", "none"))
        self.assertTrue(self.ctx.auto_confirm)
        self.assertIn("Execution preset", self.ctx.system_prompt)
        surface = q.agentic_active_tools(self.ctx)
        self.assertIn("run_command", surface)
        self.assertIn("run_python", surface)
        self.assertNotIn("write_file", surface)
        self.assertNotIn("fetch_url", surface)
        self.assertEqual(loop.tool_registry.executor.mode, "bwrap")
        self.assertEqual(loop.tool_registry.executor.bwrap_project, "ro")

    def test_exec_preset_refuses_an_explicit_non_bwrap_engine(self):
        self.ctx.sandbox_engine_override = "host"
        loop = self._loop()
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            loop.run_handle_agentic("/agentic preset exec")
        self.assertNotEqual(self.ctx.preset, "exec")
        self.assertIn("requires the bwrap engine", err.getvalue())

    def test_work_preset_is_the_default_stance(self):
        loop = self._loop()
        with patch.object(q.Executor, "_bwrap_probe", return_value=True), \
             contextlib.redirect_stderr(io.StringIO()):
            loop.run_handle_agentic("/agentic on")
        self.assertTrue(self.ctx.agentic_mode)
        self.assertEqual(self.ctx.preset, "work")
        self.assertEqual(self.ctx.executor_mode, "bwrap")
        self.assertEqual((self.ctx.bwrap_project, self.ctx.bwrap_net), ("rw", "shared"))
        self.assertIsNone(q.agentic_active_tools(self.ctx))
        self.assertNotIn("Execution preset", self.ctx.system_prompt)

    def test_trusted_preset_is_host_and_auto(self):
        loop = self._loop()
        with contextlib.redirect_stderr(io.StringIO()):
            loop.run_handle_agentic("/agentic preset trusted")
        self.assertEqual(self.ctx.preset, "trusted")
        self.assertEqual(self.ctx.executor_mode, "host")
        self.assertTrue(self.ctx.auto_confirm)
        self.assertTrue(self.ctx.agentic_mode)
        self.assertIsNone(q.agentic_active_tools(self.ctx))

    def test_full_alias_selects_trusted(self):
        loop = self._loop()
        with contextlib.redirect_stderr(io.StringIO()):
            loop.run_handle_agentic("/agentic full")
        self.assertEqual(self.ctx.preset, "trusted")
        self.assertEqual(self.ctx.executor_mode, "host")

    def test_plan_alias_selects_read_and_off_restores_the_previous_preset(self):
        loop = self._loop()
        with patch.object(q.Executor, "_bwrap_probe", return_value=True), \
             contextlib.redirect_stderr(io.StringIO()):
            loop.run_handle_agentic("/agentic preset exec")
            loop.run_handle_agentic("/agentic preset plan")
            self.assertEqual(self.ctx.preset, "read")
            loop.run_handle_agentic("/agentic plan off")
        self.assertEqual(self.ctx.preset, "exec")
        self.assertEqual(self.ctx.executor_mode, "bwrap")
        self.assertEqual(self.ctx.bwrap_project, "ro")

    def test_exec_survives_a_plan_round_trip_without_being_rehosted(self):
        """P0-1: `exec → plan → sandbox host → plan off` must stay sandboxed.

        Pre-fix the engine guard keyed on `ctx.preset`, which plan mode sets to
        "read", so the switch to host was accepted; `plan off` then restored
        only the *name* `exec`, leaving host execution, `auto_confirm=True` and
        a prompt claiming the project was mounted READ-ONLY. Reproduced
        2026-11-22 before the fix.
        """
        loop = self._loop()
        err = io.StringIO()
        with patch.object(q.Executor, "_bwrap_probe", return_value=True), \
             contextlib.redirect_stderr(err):
            loop.run_handle_agentic("/agentic preset exec")
            loop.run_handle_agentic("/agentic plan")
            loop.run_handle_agentic("/agentic sandbox host")
            loop.run_handle_agentic("/agentic plan off")
        self.assertEqual(self.ctx.preset, "exec")
        self.assertEqual(self.ctx.executor_mode, "bwrap")
        self.assertEqual(loop.executor.mode, "bwrap")
        self.assertEqual(loop.tool_registry.executor.mode, "bwrap")
        self.assertEqual(loop.tool_registry.executor.bwrap_project, "ro")
        self.assertEqual((self.ctx.bwrap_project, self.ctx.bwrap_net), ("ro", "none"))
        self.assertTrue(self.ctx.auto_confirm)
        self.assertIn("Execution preset", self.ctx.system_prompt)
        self.assertIn("plan mode owns the engine", err.getvalue())

    def test_plan_off_undoes_state_changes_made_during_plan(self):
        """The restore is authoritative even if something mutated the axes."""
        loop = self._loop()
        with patch.object(q.Executor, "_bwrap_probe", return_value=True), \
             contextlib.redirect_stderr(io.StringIO()):
            loop.run_handle_agentic("/agentic preset exec")
            loop.run_handle_agentic("/agentic plan")
        # Simulate a future transition that forgets to re-apply the preset.
        self.ctx.executor_mode = "host"
        self.ctx.auto_confirm = False
        self.ctx.bwrap_project = "rw"
        self.ctx.bwrap_net = "shared"
        self.ctx.system_prompt = "WRONG PROMPT"
        with contextlib.redirect_stderr(io.StringIO()):
            loop.run_handle_agentic("/agentic plan off")
        self.assertEqual(self.ctx.executor_mode, "bwrap")
        self.assertEqual((self.ctx.bwrap_project, self.ctx.bwrap_net), ("ro", "none"))
        self.assertTrue(self.ctx.auto_confirm)
        self.assertIn("Execution preset", self.ctx.system_prompt)
        self.assertEqual(loop.executor.mode, "bwrap")

    def test_preset_list_and_show(self):
        loop = self._loop()
        err = io.StringIO()
        with patch.object(q.Executor, "_bwrap_probe", return_value=True), \
             contextlib.redirect_stderr(err):
            loop.run_handle_agentic("/agentic preset exec")
            err.truncate(0)
            err.seek(0)
            loop.run_handle_agentic("/agentic preset")
            listing = err.getvalue()
            err.truncate(0)
            err.seek(0)
            loop.run_handle_agentic("/agentic preset show")
            shown = err.getvalue()
        for name in q.PRESET_NAMES:
            self.assertIn(name, listing)
        self.assertIn("aliases", listing)
        self.assertIn("Tools restricted to", shown)
        self.assertIn("run_command", shown)

    def test_exec_backstop_denies_a_hallucinated_write_tool(self):
        self.ctx.preset = "exec"
        self.ctx.agentic_mode = True
        registry = self.ctx.create_tool_registry()
        result = registry.execute("write_file",
                                  {"file_path": "should-not-exist.txt", "content": "x"})
        self.assertFalse(result["success"])
        self.assertIn("disabled", result["error"])
        self.assertFalse(os.path.exists("should-not-exist.txt"))

    def test_exec_surface_filters_the_prompt_and_schema(self):
        self.ctx.preset = "exec"
        self.ctx.agentic_mode = True
        registry = self.ctx.create_tool_registry()
        names = set(registry._plan_allowed_tools())
        self.assertIn("run_command", names)
        self.assertNotIn("write_file", names)
        self.assertNotIn("apply_patch", names)
        self.assertNotIn("fetch_url", names)
        tool_names = {t["function"]["name"] for t in registry.openai_tools_spec()}
        self.assertEqual(tool_names, names)

    def test_exec_preset_does_not_offer_the_cluster_tool(self):
        """P0-2: the cluster tool reaches a cluster from the harness.

        `exec` promises `net none` for the harness too, so the tool is removed
        from that surface instead of being advertised (the prompt used to claim
        it ran in the harness while it actually ran in the sandbox: dead in
        `exec`, no kubeconfig in `work`).
        """
        self.ctx.preset = "exec"
        surface = q.agentic_active_tools(self.ctx)
        self.assertNotIn("kubernetes_cluster_query", surface)
        self.assertIn("run_command", surface)
        prompt = q.agentic_system_prompt(self.ctx)
        self.assertIn("The cluster tool is NOT part of this preset", prompt)
        self.assertNotIn("runs in the harness, not in the sandbox", prompt)

    def test_exec_confines_file_tool_paths_to_the_cwd(self):
        self.ctx.preset = "exec"
        decision, message = q._path_acl_decision(
            self.ctx, "read", os.path.realpath("/etc/passwd"), "read_file")
        self.assertEqual(decision, "deny")
        self.assertIn("strict", message.lower())

    def test_work_honours_an_explicit_project_override(self):
        self.ctx.bwrap_project_override = "ro"
        loop = self._loop()
        with patch.object(q.Executor, "_bwrap_probe", return_value=True), \
             contextlib.redirect_stderr(io.StringIO()):
            loop.run_handle_agentic("/agentic preset work")
        self.assertEqual(self.ctx.bwrap_project, "ro")

    def test_exec_refuses_an_override_that_breaks_its_promise(self):
        self.ctx.bwrap_project_override = "rw"
        loop = self._loop()
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            loop.run_handle_agentic("/agentic preset exec")
        self.assertNotEqual(self.ctx.preset, "exec")
        self.assertIn("requires --sandbox-bwrap-project ro", err.getvalue())

    def test_exec_refuses_a_runtime_engine_switch_to_host(self):
        self.ctx.preset = "exec"
        self.ctx.executor_mode = "bwrap"
        loop = self._loop()
        loop.executor.mode = "bwrap"
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            loop.run_handle_agentic("/agentic sandbox host")
        self.assertEqual(loop.executor.mode, "bwrap")
        self.assertEqual(self.ctx.executor_mode, "bwrap")
        self.assertIn("requires bwrap", err.getvalue())

    def test_trusted_on_bwrap_keeps_the_rw_shared_policy(self):
        self.ctx.sandbox_engine_override = "bwrap"
        self.ctx.bwrap_project = "ro"
        self.ctx.bwrap_net = "none"
        loop = self._loop()
        with patch.object(q.Executor, "_bwrap_probe", return_value=True), \
             contextlib.redirect_stderr(io.StringIO()):
            loop.run_handle_agentic("/agentic preset trusted")
        self.assertEqual(self.ctx.preset, "trusted")
        self.assertEqual(self.ctx.executor_mode, "bwrap")
        self.assertEqual((self.ctx.bwrap_project, self.ctx.bwrap_net),
                         ("rw", "shared"))

    def test_exec_does_not_start_or_expose_mcp_tools(self):
        self.ctx.preset = "exec"
        self.ctx.agentic_mode = True
        manager = types.SimpleNamespace()
        manager.ensure_started_all = MagicMock()
        manager.aggregate_tools = MagicMock(return_value={
            "srv__lookup": {"description": "net tool", "parameters": {
                "type": "object", "properties": {}, "required": []}}})
        self.ctx.mcp_manager = manager
        try:
            registry = self.ctx.create_tool_registry()
            names = set(registry._plan_allowed_tools())
            manager.ensure_started_all.assert_not_called()
            manager.aggregate_tools.assert_not_called()
            self.assertNotIn("srv__lookup", names)
        finally:
            self.ctx.mcp_manager = None

    def test_exec_does_not_redirect_curl_to_the_host_fetch_tool(self):
        """A harmless GET must still run in bwrap (and fail with no network)."""
        self.ctx.preset = "exec"
        self.ctx.agentic_mode = True
        self.ctx.auto_confirm = True
        registry = self.ctx.create_tool_registry()
        registry.executor = q.Executor(mode="bwrap")
        with patch.object(registry.executor, "run",
                          return_value={"stdout": "", "stderr": "Could not resolve host",
                                        "returncode": 6}) as run, \
             patch.object(registry, "_serve_fetch_redirect") as redirect:
            result = registry.execute("run_command",
                                      {"command": "curl -s http://example.com"})
        redirect.assert_not_called()
        run.assert_called_once()
        self.assertFalse(result["success"])
        self.assertIn("Could not resolve host", result["error"])

    def test_work_net_none_does_not_promote_curl_to_a_host_fetch(self):
        """P1-1: `net none` must govern `curl`, not just non-GET commands.

        The redirect used to depend only on the tool surface, and `work`'s
        surface is "everything", so a `curl` GET was served by the harness with
        the host's network and environment even when the sandbox had
        `--unshare-net` (measured: the sandbox executor was never called).
        """
        self.ctx.preset = "work"
        self.ctx.executor_mode = "bwrap"
        self.ctx.bwrap_net = "none"
        self.ctx.auto_confirm = True
        registry = self.ctx.create_tool_registry()
        registry.executor = q.Executor(mode="bwrap", bwrap_project="rw", bwrap_net="none")
        registry.executor.run = MagicMock(return_value={
            "stdout": "", "stderr": "curl: (6) Could not resolve host", "returncode": 6})
        with patch.object(registry, "_serve_fetch_redirect") as redirect:
            result = registry.execute("run_command",
                                      {"command": "curl -s https://example.com"})
        redirect.assert_not_called()
        registry.executor.run.assert_called_once()
        self.assertFalse(result["success"])
        self.assertIn("Could not resolve host", result["error"])

    def test_net_none_hides_the_harness_network_tools(self):
        self.ctx.preset = "work"
        self.ctx.executor_mode = "bwrap"
        self.ctx.bwrap_net = "none"
        surface = q.agentic_active_tools(self.ctx)
        self.assertIsNotNone(surface)
        for name in q.HARNESS_NETWORK_TOOLS:
            self.assertNotIn(name, surface)
        self.assertIn("read_file", surface)
        self.assertIn("run_command", surface)
        # Same preset, network back on -> no restriction.
        self.ctx.bwrap_net = "shared"
        self.assertIsNone(q.agentic_active_tools(self.ctx))
        # A host engine is not governed by bwrap_net at all.
        self.ctx.bwrap_net = "none"
        self.ctx.executor_mode = "host"
        self.assertIsNone(q.agentic_active_tools(self.ctx))

    def test_net_none_backstop_names_the_network_stance(self):
        self.ctx.preset = "work"
        self.ctx.executor_mode = "bwrap"
        self.ctx.bwrap_net = "none"
        self.ctx.auto_confirm = True
        registry = self.ctx.create_tool_registry()
        result = registry.execute("fetch_url", {"url": "https://example.com"})
        self.assertFalse(result["success"])
        self.assertIn("network-isolated sandbox", result["error"])

    def test_net_none_does_not_start_mcp_servers_either(self):
        self.ctx.preset = "work"
        self.ctx.executor_mode = "bwrap"
        self.ctx.bwrap_net = "none"
        self.ctx.agentic_mode = True
        manager = types.SimpleNamespace()
        manager.ensure_started_all = MagicMock()
        manager.aggregate_tools = MagicMock(return_value={
            "srv__lookup": {"description": "net tool", "parameters": {
                "type": "object", "properties": {}, "required": []}}})
        self.ctx.mcp_manager = manager
        try:
            registry = self.ctx.create_tool_registry()
            names = set(registry._plan_allowed_tools())
            manager.ensure_started_all.assert_not_called()
            manager.aggregate_tools.assert_not_called()
            self.assertNotIn("srv__lookup", names)
        finally:
            self.ctx.mcp_manager = None

    def test_sandbox_net_shared_restores_the_network_at_runtime(self):
        """The easy way back: `--sandbox-bwrap-net` was a one-way startup override."""
        self.ctx.preset = "work"
        self.ctx.executor_mode = "bwrap"
        self.ctx.bwrap_net = "none"
        self.ctx.bwrap_net_override = "none"
        loop = self._loop()
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            loop.run_handle_agentic("/agentic sandbox net shared")
        self.assertEqual(self.ctx.bwrap_net, "shared")
        self.assertEqual(self.ctx.bwrap_net_override, "shared")
        self.assertEqual(loop.executor.bwrap_net, "shared")
        self.assertEqual(loop.tool_registry.executor.bwrap_net, "shared")
        self.assertIsNone(q.agentic_active_tools(self.ctx))
        self.assertIn("net: shared", err.getvalue())

    def test_sandbox_net_reset_hands_the_decision_back_to_the_preset(self):
        self.ctx.preset = "work"
        self.ctx.executor_mode = "bwrap"
        self.ctx.bwrap_net = "none"
        self.ctx.bwrap_net_override = "none"
        loop = self._loop()
        with contextlib.redirect_stderr(io.StringIO()):
            loop.run_handle_agentic("/agentic sandbox net reset")
        self.assertEqual(self.ctx.bwrap_net, "shared")
        self.assertIsNone(self.ctx.bwrap_net_override)

    def test_sandbox_net_is_refused_for_the_exec_preset(self):
        self.ctx.preset = "exec"
        self.ctx.executor_mode = "bwrap"
        self.ctx.bwrap_net = "none"
        self.ctx.bwrap_net_override = "none"
        loop = self._loop()
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            loop.run_handle_agentic("/agentic sandbox net shared")
        self.assertEqual(self.ctx.bwrap_net, "none")
        self.assertIn("requires net none", err.getvalue())
        # `reset` is the escape hatch: it clears the override, exec stays at none.
        with contextlib.redirect_stderr(io.StringIO()):
            loop.run_handle_agentic("/agentic sandbox net reset")
        self.assertEqual(self.ctx.bwrap_net, "none")
        self.assertIsNone(self.ctx.bwrap_net_override)

    def test_sandbox_net_is_refused_while_plan_mode_is_active(self):
        self.ctx.agentic_mode = True
        self.ctx.bwrap_net = "none"
        loop = self._loop()
        with contextlib.redirect_stderr(io.StringIO()):
            loop.run_handle_agentic("/agentic plan")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            loop.run_handle_agentic("/agentic sandbox net shared")
        self.assertEqual(self.ctx.bwrap_net, "none")
        self.assertIn("plan mode owns the sandbox policy", err.getvalue())

    def test_sandbox_net_completion_is_wired(self):
        self.assertIn("net", q.AGENTIC_SECOND_LEVEL["sandbox"])
        self.assertEqual(q.AGENTIC_THIRD_LEVEL[("sandbox", "net")],
                         ["shared", "none", "reset", "status"])
        completer = object.__new__(q.ChatCompleter)
        with patch.object(completer, "_completion_word_index", return_value=3):
            self.assertEqual(
                completer._complete_subcommands(
                    "/agentic sandbox net sh", "sh", q.AGENTIC_SUBCOMMANDS,
                    q.AGENTIC_SECOND_LEVEL, q.AGENTIC_THIRD_LEVEL),
                ["shared"])

    def test_exec_refuses_a_rw_bind_inside_the_project(self):
        """P1-8: `--sandbox-bwrap-rw $PWD` re-opened the read-only project."""
        self.ctx.bwrap_rw = [os.path.realpath(os.getcwd())]
        loop = self._loop()
        err = io.StringIO()
        with patch.object(q.Executor, "_bwrap_probe", return_value=True), \
             contextlib.redirect_stderr(err):
            loop.run_handle_agentic("/agentic preset exec")
        self.assertNotEqual(self.ctx.preset, "exec")
        self.assertIn("re-opens the project", err.getvalue())

    def test_exec_refuses_a_rw_bind_that_contains_the_project(self):
        parent = os.path.dirname(os.path.realpath(os.getcwd()))
        self.ctx.bwrap_binds = [(parent, "rw")]
        loop = self._loop()
        err = io.StringIO()
        with patch.object(q.Executor, "_bwrap_probe", return_value=True), \
             contextlib.redirect_stderr(err):
            loop.run_handle_agentic("/agentic preset exec")
        self.assertNotEqual(self.ctx.preset, "exec")
        self.assertIn("re-opens the project", err.getvalue())

    def test_exec_refuses_a_rw_bind_at_the_project_root(self):
        """The exact `--sandbox-bwrap-bind $PWD:rw` reproducer."""
        self.ctx.bwrap_binds = [(os.path.realpath(os.getcwd()), "rw")]
        loop = self._loop()
        err = io.StringIO()
        with patch.object(q.Executor, "_bwrap_probe", return_value=True), \
             contextlib.redirect_stderr(err):
            loop.run_handle_agentic("/agentic preset exec")
        self.assertNotEqual(self.ctx.preset, "exec")
        self.assertIn(":rw re-opens the project", err.getvalue())

    def test_exec_allows_a_rw_bind_outside_the_project(self):
        outside = tempfile.mkdtemp(prefix="bwrap-outside-")
        try:
            self.ctx.bwrap_rw = [outside]
            loop = self._loop()
            with patch.object(q.Executor, "_bwrap_probe", return_value=True), \
                 contextlib.redirect_stderr(io.StringIO()):
                loop.run_handle_agentic("/agentic preset exec")
            self.assertEqual(self.ctx.preset, "exec")
        finally:
            shutil.rmtree(outside, ignore_errors=True)

    def test_work_keeps_the_rw_subtree_lift(self):
        build = os.path.join(os.getcwd(), "build")
        self.ctx.bwrap_rw = [build]
        loop = self._loop()
        with patch.object(q.Executor, "_bwrap_probe", return_value=True), \
             contextlib.redirect_stderr(io.StringIO()):
            loop.run_handle_agentic("/agentic preset work")
        self.assertEqual(self.ctx.preset, "work")
        self.assertEqual(loop.executor.bwrap_rw, [build])

    def test_preset_status_warns_when_the_mount_policy_is_unsafe(self):
        loop = self._loop()
        err = io.StringIO()
        with patch.object(q.Executor, "_bwrap_probe", return_value=True), \
             patch.object(q.Executor, "_bwrap_policy_error",
                          return_value="scratch inside project"), \
             contextlib.redirect_stderr(err):
            loop.run_handle_agentic("/agentic preset work")
        self.assertIn("REFUSED: scratch inside project", err.getvalue())

    def test_plan_run_projection_matches_the_surface(self):
        """P1-4: the resolver must describe the surface plan-run actually has."""
        self.ctx.preset = "work"
        loop = self._loop()
        with patch.object(q.Executor, "_bwrap_probe", return_value=True), \
             contextlib.redirect_stderr(io.StringIO()):
            loop.run_handle_agentic("/agentic plan run")
        resolved = q.resolve_preset(self.ctx)
        self.assertEqual(resolved.name, "read-run")
        self.assertEqual(set(resolved.tools), set(q.PLAN_MODE_RUN_TOOLS))
        self.assertIn("run_command", q.agentic_active_tools(self.ctx))
        # The documented contract is a host shell under the gate, not an
        # inherited bwrap sandbox (where the gate is bypassed).
        self.assertEqual(self.ctx.executor_mode, "host")
        self.assertEqual(loop.executor.mode, "host")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            loop.run_handle_agentic("/agentic preset show")
        self.assertIn("read-run", err.getvalue())
        self.assertIn("run_command", err.getvalue())

    def test_plan_off_restores_the_engine_after_a_plan_run(self):
        self.ctx.preset = "work"
        loop = self._loop()
        with patch.object(q.Executor, "_bwrap_probe", return_value=True), \
             contextlib.redirect_stderr(io.StringIO()):
            loop.run_handle_agentic("/agentic preset work")
            loop.run_handle_agentic("/agentic plan run")
            self.assertEqual(self.ctx.executor_mode, "host")
            loop.run_handle_agentic("/agentic plan off")
        self.assertEqual(self.ctx.preset, "work")
        self.assertEqual(self.ctx.executor_mode, "bwrap")
        self.assertEqual(loop.executor.mode, "bwrap")

    def test_read_run_is_not_directly_selectable(self):
        loop = self._loop()
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            loop.run_handle_agentic("/agentic preset read-run")
        self.assertIn("internal", err.getvalue())
        self.assertFalse(self.ctx.plan_mode)

    def test_off_on_round_trip_keeps_the_exec_stance(self):
        """`/agentic off` used to rewrite preset=work: an exec session escalated."""
        self.ctx.preset = "work"
        loop = self._loop()
        with patch.object(q.Executor, "_bwrap_probe", return_value=True), \
             contextlib.redirect_stderr(io.StringIO()):
            loop.run_handle_agentic("/agentic preset exec")
            loop.run_handle_agentic("/agentic off")
            loop.run_handle_agentic("/agentic on")
        self.assertEqual(self.ctx.preset, "exec")
        self.assertEqual(self.ctx.executor_mode, "bwrap")
        self.assertEqual((self.ctx.bwrap_project, self.ctx.bwrap_net), ("ro", "none"))
        self.assertEqual(loop.executor.mode, "bwrap")

    def test_off_on_round_trip_keeps_the_trusted_stance(self):
        self.ctx.preset = "work"
        loop = self._loop()
        with contextlib.redirect_stderr(io.StringIO()):
            loop.run_handle_agentic("/agentic preset trusted")
            loop.run_handle_agentic("/agentic off")
            loop.run_handle_agentic("/agentic on")
        self.assertEqual(self.ctx.preset, "trusted")
        self.assertEqual(self.ctx.executor_mode, "host")

    def test_plan_from_chat_then_off_on_lands_on_work(self):
        """The idle default is not a stance the user chose."""
        self.ctx.agentic_mode = False
        self.ctx.preset = "trusted"          # the pre-agentic idle default
        loop = self._loop()
        with patch.object(q.Executor, "_bwrap_probe", return_value=True), \
             contextlib.redirect_stderr(io.StringIO()):
            loop.run_handle_agentic("/agentic plan")
            loop.run_handle_agentic("/agentic off")
            loop.run_handle_agentic("/agentic on")
        self.assertEqual(self.ctx.preset, "work")

    def test_plan_from_chat_then_plan_off_then_on_lands_on_work(self):
        self.ctx.agentic_mode = False
        self.ctx.preset = "trusted"
        loop = self._loop()
        with patch.object(q.Executor, "_bwrap_probe", return_value=True), \
             contextlib.redirect_stderr(io.StringIO()):
            loop.run_handle_agentic("/agentic plan")
            loop.run_handle_agentic("/agentic plan off")
            loop.run_handle_agentic("/agentic on")
        self.assertEqual(self.ctx.preset, "work")
        self.assertEqual(self.ctx.executor_mode, "bwrap")

    def test_bare_sandbox_toggle_leaves_a_sandbox_for_the_host(self):
        self.ctx.preset = "work"
        loop = self._loop()
        with patch.object(q.Executor, "_bwrap_probe", return_value=True), \
             contextlib.redirect_stderr(io.StringIO()):
            loop.run_handle_agentic("/agentic preset work")
            self.assertEqual(loop.executor.mode, "bwrap")
            loop.run_handle_agentic("/agentic sandbox")
            self.assertEqual(loop.executor.mode, "host")
            self.assertEqual(self.ctx.executor_mode, "host")
            loop.run_handle_agentic("/agentic sandbox")
            self.assertEqual(loop.executor.mode, "container")

    def test_sandbox_reset_hands_the_engine_back_to_the_preset(self):
        self.ctx.preset = "work"
        self.ctx.sandbox_engine_override = "host"
        self.ctx.executor_mode = "host"
        loop = self._loop()
        with patch.object(q.Executor, "_bwrap_probe", return_value=True), \
             contextlib.redirect_stderr(io.StringIO()):
            loop.run_handle_agentic("/agentic sandbox reset")
        self.assertIsNone(self.ctx.sandbox_engine_override)
        self.assertEqual(self.ctx.executor_mode, "bwrap")
        self.assertEqual(loop.executor.mode, "bwrap")
        self.assertEqual(loop.tool_registry.executor.mode, "bwrap")

    def test_plan_tiers_confinement_matches_the_tier(self):
        self.ctx.preset = "work"
        loop = self._loop()
        with patch.object(q.Executor, "_bwrap_probe", return_value=True), \
             contextlib.redirect_stderr(io.StringIO()):
            loop.run_handle_agentic("/agentic plan")
            self.assertTrue(q.strict_path_confinement(self.ctx))
            loop.run_handle_agentic("/agentic plan run")
            self.assertFalse(q.strict_path_confinement(self.ctx))

    def test_apply_preset_announces_the_auto_confirm_change(self):
        """P1-2: `/agentic on` flips auto-confirm — say it, don't imply it."""
        self.ctx.auto_confirm = False
        loop = self._loop()
        err = io.StringIO()
        with patch.object(q.Executor, "_bwrap_probe", return_value=True), \
             contextlib.redirect_stderr(err):
            loop.run_handle_agentic("/agentic preset exec")
        self.assertIn("Auto-confirm: ON", err.getvalue())
        self.assertIn("preset exec approval=auto", err.getvalue())

    def test_full_alias_honours_a_bwrap_override_with_rw_shared(self):
        self.ctx.sandbox_engine_override = "bwrap"
        self.ctx.bwrap_project = "ro"
        self.ctx.bwrap_net = "none"
        loop = self._loop()
        with patch.object(q.Executor, "_bwrap_probe", return_value=True), \
             contextlib.redirect_stderr(io.StringIO()):
            loop.run_handle_agentic("/agentic full")
        self.assertEqual(self.ctx.preset, "trusted")
        self.assertEqual(self.ctx.executor_mode, "bwrap")
        self.assertEqual((self.ctx.bwrap_project, self.ctx.bwrap_net),
                         ("rw", "shared"))

    def test_read_preset_does_not_confine_beyond_plan(self):
        self.ctx.preset = "work"
        self.assertFalse(q.strict_path_confinement(self.ctx))


class TestBwrapSandboxLive(unittest.TestCase):
    """Opt-in real bwrap runs (RUN_BWRAP_E2E=1; skipped when bwrap is absent)."""

    def setUp(self):
        if not (RUN_BWRAP_E2E and BWRAP_AVAILABLE):
            self.skipTest("set RUN_BWRAP_E2E=1 with bwrap installed")
        self.tmp = tempfile.mkdtemp(prefix="bwrap-e2e-")
        self.home = os.path.join(self.tmp, "home")
        self.project = os.path.join(self.tmp, "project")
        os.makedirs(self.home)
        os.makedirs(self.project)
        self._old_cwd = os.getcwd()
        os.chdir(self.project)

    def tearDown(self):
        if getattr(self, "tmp", None):
            os.chdir(self._old_cwd)
            shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, command, project="rw", net="none", timeout=20):
        ex = q.Executor(mode="bwrap", bwrap_project=project, bwrap_net=net,
                        bwrap_home=self.home)
        return ex.run(command, timeout=timeout)

    def test_project_rw_writes_scratch_and_etc_is_read_only(self):
        result = self._run(
            "echo ok > in-project.txt; echo s > $HOME/cache.txt; "
            "echo x > /etc/should-not-exist; echo etc=$?")
        self.assertEqual(result["returncode"], 0)
        self.assertTrue(os.path.exists(os.path.join(self.project, "in-project.txt")))
        self.assertTrue(os.path.exists(os.path.join(self.home, "cache.txt")))
        self.assertFalse(os.path.exists("/etc/should-not-exist"))
        self.assertIn("etc=1", result["stdout"])

    def test_project_ro_refuses_writes_but_the_scratch_stays_rw(self):
        result = self._run("echo x > refused.txt; echo rc=$?; "
                           "echo y > $HOME/ok.txt; echo home=$?",
                           project="ro")
        self.assertIn("rc=1", result["stdout"])
        self.assertIn("home=0", result["stdout"])
        self.assertFalse(os.path.exists(os.path.join(self.project, "refused.txt")))

    def test_net_none_blocks_dns_and_tcp(self):
        result = self._run("getent hosts example.com >/dev/null 2>&1; echo dns=$?; "
                           "timeout 3 bash -c 'echo > /dev/tcp/192.168.1.20/8090' "
                           "2>/dev/null; echo tcp=$?")
        self.assertIn("dns=2", result["stdout"])
        self.assertNotIn("tcp=0", result["stdout"])

    def test_host_home_is_not_mounted(self):
        host_home = os.path.expanduser("~")
        result = self._run(f"test -e {shlex.quote(host_home)}; echo home=$?")
        self.assertIn("home=1", result["stdout"])

    def test_host_runtime_sockets_are_not_reachable(self):
        """The sandbox may read host config, but must not reach live Unix sockets.

        `/run` used to be ro-bound whole, which still allowed connect(2): this
        is the live half of
        `TestBwrapSandboxConfig.test_run_is_not_mounted_whole_because_sockets_survive_ro_binds`.
        """
        uid = os.getuid()
        sockets = [f"/run/user/{uid}/podman/podman.sock",
                   f"/run/user/{uid}/docker.sock",
                   "/run/docker.sock", "/var/run/docker.sock",
                   "/run/dbus/system_bus_socket"]
        present = [p for p in sockets if os.path.exists(p)]
        checks = " ".join(
            f"test -e {shlex.quote(p)} && echo REACHABLE:{p};" for p in sockets)
        result = self._run(checks + " echo done")
        self.assertIn("done", result["stdout"])
        for path in present:
            self.assertNotIn(f"REACHABLE:{path}", result["stdout"])
        # The resolver state is still mounted, so DNS names resolve when the
        # network is shared (net none keeps getent failing, which is expected).
        if os.path.exists("/run/systemd/resolve"):
            listed = self._run("test -e /run/systemd/resolve && echo RESOLVER_OK")
            self.assertIn("RESOLVER_OK", listed["stdout"])

    @unittest.skipUnless(shutil.which("gcc"), "gcc not installed")
    def test_gcc_builds_and_runs_inside_the_sandbox(self):
        result = self._run(
            "printf 'int main(){return 0;}\\n' > m.c && gcc m.c -o m && ./m && echo BUILD_OK")
        self.assertIn("BUILD_OK", result["stdout"])


class TestAgenticStreamObservations(unittest.TestCase):
    """Streaming tool observations must be persisted in self.messages."""

    def setUp(self):
        self.ctx = q.CommandContext()
        self.ctx.auto_confirm = True
        self.ctx.base_url = "http://localhost:9999"
        # No tokenize endpoint / whole-request counter: keep the test offline.
        self.ctx.backend = "opencodezen"
        self.ctx.model = "test-model"
        self.loop = q.ChatLoop(self.ctx)

    def test_finalize_appends_observations_for_stream_tool_calls(self):
        """_finalize_agentic_query must append tool observations after stream tool calls."""
        self.loop.messages = []
        messages = [{'role': 'system', 'content': 'test'}]
        # The finalize path can re-enter through query_sync_stream; keep it offline.
        self.loop.query_handler.query_sync_stream = MagicMock(
            return_value={"choices": [{"message": {"role": "assistant", "content": ""}}]})
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

    def test_finalize_skips_restream_when_content_streamed(self):
        """content_streamed=True (thinking mode) must skip the final query_stream
        re-generation — the answer was already streamed live — and just record it."""
        self.loop.messages = []
        messages = [{'role': 'system', 'content': 'test'}]
        with patch.object(self.loop.query_handler, 'query_stream',
                          return_value="Response") as mock_stream:
            self.loop._finalize_agentic_query(
                messages, final_answer="Hello world", final_content="test content",
                send_tools_api=False, openai_tools=[], logger=None,
                iteration=1, response_text="", content_streamed=True
            )
        mock_stream.assert_not_called()
        assistant_msgs = [m for m in self.loop.messages if m['role'] == 'assistant']
        self.assertEqual(len(assistant_msgs), 1)
        self.assertEqual(assistant_msgs[0]['content'], "Hello world")

    def test_finalize_restreams_when_content_not_streamed(self):
        """Without content_streamed (thinking off) the final query_stream
        re-generation still runs — the step content was never displayed."""
        self.loop.messages = []
        messages = [{'role': 'system', 'content': 'test'}]
        with patch.object(self.loop.query_handler, 'query_stream',
                          return_value="Streamed answer") as mock_stream:
            self.loop._finalize_agentic_query(
                messages, final_answer="", final_content="test content",
                send_tools_api=False, openai_tools=[], logger=None,
                iteration=1, response_text="step content"
            )
        mock_stream.assert_called_once()
        assistant_msgs = [m for m in self.loop.messages if m['role'] == 'assistant']
        self.assertEqual(len(assistant_msgs), 1)
        self.assertEqual(assistant_msgs[0]['content'], "Streamed answer")


class TestAgenticFinalStreamSystemPrompt(unittest.TestCase):
    """The final stream must reuse the wire prompt the ReAct step saw.

    Regression: `_finalize_agentic_query()` replaced the ReAct list's system
    message with `ctx.system_prompt`, which `/agentic plan` and `/agentic full`
    set without `tool_defs_block`. Inline-tool models therefore streamed their
    final answer with the mode prose only — no tool JSON schemas — so a late
    tool call in that stream had no current schema to form a reliable call.
    Several mode transitions in one session made the behaviour look
    model-dependent.
    """

    def setUp(self):
        self.ctx = q.CommandContext()
        self.ctx.auto_confirm = True
        self.ctx.base_url = "http://localhost:9999"
        # No tokenize endpoint / whole-request counter: keep the test offline.
        self.ctx.backend = "opencodezen"
        self.ctx.model = "test-model"  # not in the registry -> inline delivery
        self.ctx.system_prompt = "CHAT PROMPT"
        self.ctx._saved_system_prompt = None
        self.ctx._plan_prev_prompt = None
        self.ctx.agentic_mode = False
        self.ctx.plan_mode = False
        self.ctx.plan_run = False
        self.loop = q.ChatLoop(self.ctx)

    def tearDown(self):
        self.ctx.agentic_mode = False
        self.ctx.plan_mode = False
        self.ctx.plan_run = False
        self.ctx.system_prompt = "CHAT PROMPT"
        self.ctx._saved_system_prompt = None
        self.ctx._plan_prev_prompt = None

    def _capture_final_stream(self):
        """Init a ReAct query, then finalize; return (step system, final payload, kwargs)."""
        self.loop.messages = [{'role': 'system', 'content': self.ctx.system_prompt}]
        messages, logger, openai_tools, send_tools_api = self.loop._init_agentic_query("ask")
        try:
            step_system = messages[0]['content']
            with patch.object(self.loop.query_handler, 'query_stream',
                              return_value="Final answer") as mock_stream:
                self.loop._finalize_agentic_query(
                    messages, final_answer="Final answer", final_content="ask",
                    send_tools_api=send_tools_api, openai_tools=openai_tools,
                    logger=None, iteration=1, response_text="")
            mock_stream.assert_called_once()
            return step_system, mock_stream.call_args[0][0], mock_stream.call_args[1]
        finally:
            # The product closes the session logger at query end; a test that
            # discards it leaves the JSONL handle open until GC.
            if logger is not None:
                logger.close()

    def test_plan_mode_final_stream_keeps_inline_tool_defs(self):
        """`/agentic plan`: the final stream keeps the step's restricted schemas."""
        self.loop._agentic_toggle_plan(['/agentic', 'plan'])
        step_system, payload, kwargs = self._capture_final_stream()
        # Sanity: the ReAct prompt carries the inline tool-definition block,
        # while ctx.system_prompt (the mode prose) does not.
        self.assertIn("## Available tools", step_system)
        self.assertNotIn("## Available tools", self.ctx.system_prompt)
        self.assertEqual(payload[0]['role'], 'system')
        self.assertEqual(
            payload[0]['content'], step_system,
            "final stream must send the same system prompt as the ReAct step")
        self.assertNotEqual(payload[0]['content'], self.ctx.system_prompt)
        self.assertNotIn('tools', kwargs, "inline models must not get a native tools array")

    def test_full_mode_final_stream_keeps_inline_tool_defs(self):
        """`/agentic full`: the final stream keeps the step's unrestricted schemas."""
        self.loop._agentic_set_full()
        step_system, payload, kwargs = self._capture_final_stream()
        self.assertIn("## Available tools", step_system)
        self.assertNotIn("## Available tools", self.ctx.system_prompt)
        self.assertEqual(
            payload[0]['content'], step_system,
            "final stream must send the same system prompt as the ReAct step")
        self.assertNotEqual(payload[0]['content'], self.ctx.system_prompt)
        self.assertNotIn('tools', kwargs)

    def test_final_stream_rebuilds_prompt_when_step_list_has_no_system(self):
        """An unexpected system-less step list still gets the agentic wire prompt."""
        self.loop._agentic_toggle_plan(['/agentic', 'plan'])
        with patch.object(self.loop.query_handler, 'query_stream',
                          return_value="Final answer") as mock_stream:
            self.loop._finalize_agentic_query(
                [], final_answer="Final answer", final_content="ask",
                send_tools_api=False, openai_tools=[], logger=None,
                iteration=1, response_text="")
        payload = mock_stream.call_args[0][0]
        self.assertEqual(payload[0]['role'], 'system')
        self.assertIn("## Available tools", payload[0]['content'])


class TestExecutor(unittest.TestCase):
    """Test Executor (host mode only — no container runtime required)."""

    def setUp(self):
        self.exec = q.Executor(mode="host")

    def test_echo(self):
        result = self.exec.run("echo hello", timeout=5)
        self.assertEqual(result["stdout"].strip(), "hello")
        self.assertEqual(result["returncode"], 0)

    def test_host_command_gets_tmpdir_scratch(self):
        result = self.exec.run("echo $TMPDIR", timeout=5)
        self.assertEqual(result["stdout"].strip(), q._scratch_root())

    def test_failure(self):
        result = self.exec.run("false", timeout=5)
        self.assertNotEqual(result["returncode"], 0)

    def test_timeout(self):
        result = self.exec.run("sleep 10", timeout=1)
        self.assertNotEqual(result["returncode"], 0)

    def test_timeout_kills_background_children(self):
        """A timed-out shell must not leave `sleep 60 &` orphaned."""
        with tempfile.TemporaryDirectory() as td:
            pidfile = os.path.join(td, "child.pid")
            command = (f"sh -c 'sleep 60 & echo $! > {shlex.quote(pidfile)}; "
                       f"wait'")
            with patch("ollamaquery2.check_shell_approval",
                       return_value={"approved": True, "effect": "allow",
                                     "message": None, "verdict": None}):
                result = self.exec.run(command, timeout=1)
            self.assertEqual(result["returncode"], -1)
            self.assertIn("Timed out", result["stderr"])
            deadline = time.time() + 2
            while time.time() < deadline and not os.path.exists(pidfile):
                time.sleep(0.05)
            if not os.path.exists(pidfile):
                self.skipTest("the child pid was not written before the kill")
            with open(pidfile, encoding="utf-8") as f:
                pid = int(f.read().strip())
            try:
                for _ in range(30):
                    try:
                        os.kill(pid, 0)
                    except ProcessLookupError:
                        return
                    time.sleep(0.05)
                self.fail(f"background child {pid} survived the timeout")
            finally:
                try:
                    os.kill(pid, 9)
                except (ProcessLookupError, PermissionError, OSError):
                    pass

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
        # Defensive: E2E does write/compile/run, which can never succeed in
        # read-only plan mode. Never inherit a plan flag from another test.
        self.ctx.plan_mode = False
        self.ctx.plan_run = False
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
                # Observations are now stored, so the last message may be a
                # tool result. Check that an assistant response exists.
                assistant_msgs = [m for m in self.loop.messages if m['role'] == 'assistant']
                self.assertGreaterEqual(len(assistant_msgs), 1,
                                        "No assistant response found in history")
                self.assertGreater(len(assistant_msgs[-1]['content']), 0)
            finally:
                os.chdir(old_cwd)

    @unittest.skipUnless(SLOW_DNS_E2E,
                         "slow UDP DNS resolver E2E disabled; set RUN_SLOW_DNS_E2E=1 to run it")
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
        # These are unit tests: the LLM is mocked, so token stamping must not
        # reach the fake backend either. Without this every run_agentic_query
        # stamps system+user via _request_with_retry, which retries 3x with a
        # 1s delay against an unresolvable host and dominates the test log.
        self._stamp_patch = patch.object(
            q.CommandContext, "stamp_tokens", lambda _ctx, msg: msg)
        self._stamp_patch.start()
        self.addCleanup(self._stamp_patch.stop)
        # The context bar also tokenizes: keep it on the local estimator and
        # skip the backend's whole-request counter (both would hit the fake URL).
        self._tokenize_patch = patch.object(
            q.CommandContext, "tokenize",
            lambda _ctx, text: _ctx.estimate_tokens(text))
        self._tokenize_patch.start()
        self.addCleanup(self._tokenize_patch.stop)
        self._wire_patch = patch.object(
            q.CommandContext, "count_wire_tokens", lambda *_a, **_k: None)
        self._wire_patch.start()
        self.addCleanup(self._wire_patch.stop)
        self.loop = q.ChatLoop(self.ctx)
        # These are *unit* tests, but they drive the real tool handlers with
        # relative paths (`write_file("x.txt")`). Point the CWD at a throwaway
        # directory: pre-fix they dropped hello.txt / x.txt / z.txt at the
        # repository root on every suite run.
        self._repo_cwd = os.getcwd()
        self._workdir = tempfile.TemporaryDirectory(prefix="aq_react_")
        os.chdir(self._workdir.name)
        self.addCleanup(self._leave_workdir)

    def _leave_workdir(self):
        """Restore the caller's CWD and delete the scratch directory."""
        os.chdir(self._repo_cwd)
        self._workdir.cleanup()

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

        self.loop.query_handler.query_sync_stream = mock_sync
        self.loop.query_handler.query_stream = MagicMock(return_value="The file was written successfully.")

        self.loop.run_agentic_query("Create a file with hello world")

        # Should have assistant message at the end
        last = self.loop.messages[-1]
        self.assertEqual(last["role"], "assistant")
        self.assertIn("successfully", last["content"])

    def test_step_stats_line_printed(self):
        """Each ReAct step prints a chat-mode stats line (time, rates, context)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
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

                self.loop.query_handler.query_sync_stream = mock_sync
                self.loop.query_handler.query_stream = MagicMock(return_value="The file was written successfully.")
                err = io.StringIO()
                with patch("sys.stderr", err):
                    self.loop.run_agentic_query("Create a file with hello world")
                self.assertIn("Stats:", err.getvalue())
                self.assertIn("Context:", err.getvalue())
            finally:
                os.chdir(old_cwd)

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

        self.loop.query_handler.query_sync_stream = mock_sync
        self.loop.query_handler.query_stream = MagicMock(return_value="")

        # Mock input() to say 'n' (cancel)
        with patch("builtins.input", return_value="n"):
            self.loop.run_agentic_query("Create a file")

        # On cancellation, no new assistant message is appended
        # (only the original system prompt message exists)
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

        self.loop.query_handler.query_sync_stream = mock_sync
        self.loop.query_handler.query_stream = MagicMock(return_value="")

        self.loop.run_agentic_query("Keep using tools")

        # Terminated at the cap without crashing, and the turn was persisted.
        self.assertGreater(len(self.loop.messages), 2)

    def test_parse_tool_call_with_real_usage(self):
        """Real parse_tool_call usage within the loop."""
        self.loop.query_handler.query_sync_stream = MagicMock(
            side_effect=[
                self._make_sync_response(
                    '{"tool": "read_file", "arguments": {"file": "nonexistent.txt"}}'
                ),
                self._make_sync_response("File does not exist.")
            ]
        )
        self.loop.query_handler.query_stream = MagicMock(return_value="File does not exist.")

        self.loop.run_agentic_query("Read a file")

        last = self.loop.messages[-1]
        self.assertEqual(last["role"], "assistant")

    def test_api_error_skips_finalize_and_preserves_history(self):
        """When the backend is unreachable, the loop must NOT re-query a dead
        endpoint for a final answer, must NOT print/merge the '[Agentic: no
        answer produced]' placeholder, and must leave conversation history
        un-polluted so repeated attempts don't pile failures into context."""
        self.loop.query_handler.query_sync_stream = MagicMock(
            return_value={"error": {"message": "Connection refused"}})
        self.loop.query_handler.query_stream = MagicMock(return_value="")

        self.loop.run_agentic_query("ok update AGENTS.md")

        self.assertFalse(self.loop.query_handler.query_stream.called,
                         "Finalize must not re-query an unreachable backend")
        assistant_msgs = [m for m in self.loop.messages if m["role"] == "assistant"]
        self.assertEqual(len(assistant_msgs), 0,
                         "No assistant placeholder should be recorded on API error")
        for m in self.loop.messages:
            content = m.get("content", "")
            self.assertNotIn("[Agentic: no answer produced]", content)
        # The user's own message stays (real input), but nothing artificial was added.
        self.assertLessEqual(len(self.loop.messages), 2)

    def test_api_error_after_tool_turns_keeps_real_work(self):
        """If the backend dies partway through a query, the successful tool turns
        must still be merged into history — only the failed finalize is skipped."""
        calls = [
            self._make_sync_response(
                '{"tool": "write_file", "arguments": {"file": "z.txt", "content": "data"}}'
            ),
            {"error": {"message": "HTTP Error 503: Service Unavailable"}},
        ]
        call_idx = [0]

        def mock_sync(*args, **kwargs):
            idx = call_idx[0]
            call_idx[0] += 1
            return calls[idx] if idx < len(calls) else self._make_sync_response("done")

        self.loop.query_handler.query_sync_stream = mock_sync
        self.loop.query_handler.query_stream = MagicMock(return_value="")

        self.loop.run_agentic_query("fix the server")

        contents = [m.get("content", "") for m in self.loop.messages]
        self.assertTrue(any("Tool result" in c for c in contents),
                        "Successful tool turns must persist after an API error")
        self.assertFalse(self.loop.query_handler.query_stream.called)

    def test_empty_response_nudge_not_persisted(self):
        """The empty-step 'continue where you stopped' nudge is loop-internal
        steering and must NOT be merged into the persistent conversation history."""
        calls = [
            self._make_sync_response(""),                      # empty → nudge injected
            self._make_sync_response("Done."),                 # real answer
        ]
        call_idx = [0]

        def mock_sync(*args, **kwargs):
            idx = call_idx[0]
            call_idx[0] += 1
            return calls[idx] if idx < len(calls) else self._make_sync_response("done")

        self.loop.query_handler.query_sync_stream = mock_sync
        self.loop.query_handler.query_stream = MagicMock(return_value="Done.")

        self.loop.run_agentic_query("Say done")

        self.assertFalse(
            any(m.get("content") == q.AgenticRunner.continuation_nudge("empty")
                for m in self.loop.messages),
            "System nudge must not persist in conversation history")
        self.assertEqual(self.loop.messages[-1]["role"], "assistant")
        self.assertIn("Done.", self.loop.messages[-1]["content"])

    def test_interrupt_persists_completed_turns(self):
        """F2/T1(a): a ^C mid-loop must persist completed tool turns into
        self.messages before control returns, so the next query continues from
        the full history instead of re-exploring."""
        # Seed a prior turn so agentic_seed_len > 1 and ordering is meaningful.
        self.loop.messages = [
            {'role': 'system', 'content': 'sys'},
            {'role': 'user', 'content': 'read this project'},
        ]
        tool_resp = self._make_sync_response(
            '{"tool": "write_file", "arguments": {"file": "x.txt", "content": "data"}}')
        call_idx = [0]

        def fake_timeout(*args, **kwargs):
            idx = call_idx[0]
            call_idx[0] += 1
            if idx == 0:
                return tool_resp          # step 1 produces a tool call → executed
            raise KeyboardInterrupt()     # step 2 interrupted by ^C

        # The step call lives on AgenticRunner; patch it there so the fake
        # timeout drives the loop's timeout/^C paths.
        runner = self.loop._agentic_runner()
        with patch.object(runner, 'call_with_timeout', side_effect=fake_timeout):
            with self.assertRaises(KeyboardInterrupt):
                self.loop.run_agentic_query("read this project")

        roles = [m['role'] for m in self.loop.messages]
        self.assertIn('assistant', roles, "completed assistant turn must be persisted")
        self.assertTrue(
            any('Tool result' in (m.get('content') or '') for m in self.loop.messages),
            "completed tool result must be persisted")

    def test_query_sync_stream_cancel_quiet(self):
        """T1(e): query_sync_stream returns the 'cancelled' marker quietly (no
        [ERROR] spam) when the cancel event is set, so an aborted read on ^C or
        timeout doesn't error the terminal."""
        mq = self.loop.query_handler
        ctx = self.ctx

        class FakeResp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def close(self):
                pass

            def __iter__(self):
                raise OSError("connection closed by cancel")

        cancel = {"event": threading.Event(), "close": None}
        cancel["event"].set()  # simulate an already-aborted step

        with patch.object(mq, '_build_stream_request', return_value=("http://x", {}, {})), \
             patch('ollamaquery2._request_with_retry', return_value=FakeResp()):
            stderr = io.StringIO()
            with patch('sys.stderr', stderr):
                resp = mq.query_sync_stream(
                    [{'role': 'user', 'content': 'hi'}], "m", cancel=cancel, timeout=5)

        self.assertEqual(resp.get("error", {}).get("message"), "cancelled")
        self.assertNotIn("[ERROR]", stderr.getvalue())

    # -- iteration budget + truncated steps (2026-10-29 regression) ----------
    def test_iteration_limit_runs_exactly_max_steps(self):
        """`agentic_max_iterations` counts STEPS, not two per step.

        `run()` incremented the counter and `_run_step()` incremented it again
        (the increment stayed in the extracted body), so every pass cost 2: the
        real budget was `ceil(max/2)`, a configured 6 ran 3 steps, and the first
        step header read "Step 2/6".
        """
        self.ctx.agentic_max_iterations = 3
        calls = []

        def mock_sync(*args, **kwargs):
            calls.append(1)
            return self._make_sync_response(
                '{"tool": "write_file", "arguments": {"file": "x.txt", "content": "y"}}')

        self.loop.query_handler.query_sync_stream = mock_sync
        self.loop.query_handler.query_stream = MagicMock(return_value="done")
        self.loop.run_agentic_query("Keep using tools")

        self.assertEqual(len(calls), 3)

    def test_first_step_header_is_step_one(self):
        """The header must print the step that is running, not the next one."""
        self.ctx.agentic_max_iterations = 2

        def mock_sync(*args, **kwargs):
            return self._make_sync_response("Direct answer, no tool.")

        self.loop.query_handler.query_sync_stream = mock_sync
        self.loop.query_handler.query_stream = MagicMock(return_value="Direct answer, no tool.")
        stderr = io.StringIO()
        with patch('sys.stderr', stderr):
            self.loop.run_agentic_query("Say hello")

        self.assertIn("Step 1/2", stderr.getvalue())

    def test_iteration_limit_does_not_reexecute_the_tool(self):
        """At the iteration limit the last tool call must not run a second time.

        `finish_query` promoted `response_text` (the tool-call JSON) to the final
        answer, and the finalizer parsed and executed it again — a duplicate
        `run_command`/`apply_patch` is not idempotent.
        """
        self.ctx.agentic_max_iterations = 1
        executed = []
        real_execute = self.loop.tool_registry.execute

        def spy(name, args):
            executed.append(name)
            return real_execute(name, args)

        self.loop.tool_registry.execute = spy

        def mock_sync(*args, **kwargs):
            return self._make_sync_response(
                '{"tool": "write_file", "arguments": {"file": "x.txt", "content": "y"}}')

        self.loop.query_handler.query_sync_stream = mock_sync
        self.loop.query_handler.query_stream = MagicMock(return_value="done")
        self.loop.run_agentic_query("Write it")

        self.assertEqual(executed.count("write_file"), 1)

    def test_step_ending_mid_reasoning_is_nudged_not_accepted(self):
        """A step that ends on a reasoning chunk is a preamble, not the answer.

        Observed live: content ("Let me write the file now…") then the model
        returned to its reasoning channel and the generation ended. The loop
        recorded the 41-token preamble as the final answer and the query stopped.
        """
        replies = [("Let me write the document.", "still planning"),  # ends in thought
                   ("The document is written.", "")]
        idx = [0]
        calls = []

        def mock_sync(*args, **kwargs):
            on_chunk = kwargs.get("on_chunk")
            content, thought = replies[min(idx[0], len(replies) - 1)]
            idx[0] += 1
            calls.append(1)
            if on_chunk:
                if content:
                    on_chunk("", content, False)
                if thought:
                    on_chunk(thought, "", False)
            return self._make_sync_response(content)

        self.loop.query_handler.query_sync_stream = mock_sync
        self.loop.query_handler.query_stream = MagicMock(return_value="The document is written.")
        self.loop.run_agentic_query("Write the document")

        self.assertEqual(len(calls), 2, "the truncated step must be retried, not accepted")
        self.assertNotIn("_system_nudge",
                         [k for m in self.loop.messages for k in m],
                         "the nudge is loop-internal and must not be persisted")

    def test_truncated_step_nudge_is_bounded(self):
        """A model that always dies mid-thought is nudged twice, then accepted."""
        calls = []

        def mock_sync(*args, **kwargs):
            on_chunk = kwargs.get("on_chunk")
            calls.append(1)
            if on_chunk:
                on_chunk("", "Preamble.", False)
                on_chunk("still reasoning", "", False)
            return self._make_sync_response("Preamble.")

        self.loop.query_handler.query_sync_stream = mock_sync
        self.loop.query_handler.query_stream = MagicMock(return_value="Preamble.")
        self.loop.run_agentic_query("Do the thing")

        self.assertEqual(len(calls), 1 + q.AGENTIC_TRUNCATED_STEP_MAX)

    def test_step_feedback_tracks_ending_in_thinking(self):
        """`ended_in_thinking` follows the LAST text chunk, whatever is displayed."""
        feedback = q.StepFeedback(self.ctx, show_thinking=False)
        feedback.feed("reasoning", "", False)
        self.assertTrue(feedback.ended_in_thinking)
        feedback.feed("", "visible text", False)
        self.assertFalse(feedback.ended_in_thinking)
        feedback.feed("more reasoning", "", False)
        self.assertTrue(feedback.ended_in_thinking)
        # A usage-only chunk (no text) must not change the verdict.
        feedback.feed("", "", True)
        self.assertTrue(feedback.ended_in_thinking)

    # -- stream completeness (2026-11-13) ------------------------------------
    #
    # Live symptom: the loop stopped mid-work after a step whose prose announced
    # the next action ("Je lance la transformation…") and sometimes nudged the
    # model to "resume" mid-cycle. Both come from a step whose turn never
    # finished being treated as an answer; these pin the classification and the
    # bounded continuation nudges.

    def test_truncation_reason_classification(self):
        """Only a stream that ended without a terminal marker is 'cut'."""
        cls = q.AgenticRunner.truncation_reason
        self.assertIsNone(cls(None))
        self.assertIsNone(cls({}))
        self.assertIsNone(cls({"parsed_lines": 4, "bad_lines": 0, "saw_final": True,
                               "finish_reason": "stop"}))
        self.assertEqual(cls({"parsed_lines": 4, "saw_final": False,
                              "finish_reason": ""}), "cut")
        self.assertEqual(cls({"parsed_lines": 4, "saw_final": True,
                              "finish_reason": "length"}), "length")
        self.assertEqual(cls({"parsed_lines": 4, "saw_final": True, "bad_lines": 2,
                              "finish_reason": "stop"}), "badstream")

    def test_cut_stream_prose_is_nudged_not_finalized(self):
        """A step cut before its terminal marker is a preamble, not the answer."""
        cut = dict(self._make_sync_response("I'll write the file now."),
                   _stream={"parsed_lines": 5, "bad_lines": 0, "saw_final": False,
                            "finish_reason": ""})
        replies = [
            cut,
            dict(self._make_sync_response("The file is written."),
                 _stream={"parsed_lines": 5, "bad_lines": 0, "saw_final": True,
                          "finish_reason": "stop"}),
        ]
        idx = [0]
        seen = []

        def mock_sync(messages, model, **kwargs):
            seen.append([dict(m) for m in messages])
            i = idx[0]
            idx[0] += 1
            return replies[min(i, len(replies) - 1)]

        self.loop.query_handler.query_sync_stream = mock_sync
        self.loop.query_handler.query_stream = MagicMock(return_value="The file is written.")
        self.loop.run_agentic_query("Write the file")

        self.assertEqual(len(seen), 2, "a cut step must be retried, not accepted")
        self.assertIn(q.AgenticRunner.continuation_nudge("cut"),
                      [m.get("content") for m in seen[1]])
        self.assertNotIn(q.AgenticRunner.continuation_nudge("cut"),
                         [m.get("content") for m in self.loop.messages])

    def test_length_capped_step_is_nudged(self):
        """`finish_reason: length` means the output cap cut the turn: continue it."""
        capped = dict(self._make_sync_response("Step one…"),
                      _stream={"parsed_lines": 3, "bad_lines": 0, "saw_final": True,
                               "finish_reason": "length"})
        seen = []
        idx = [0]
        replies = [capped, self._make_sync_response("All done.")]

        def mock_sync(messages, model, **kwargs):
            seen.append([dict(m) for m in messages])
            i = idx[0]
            idx[0] += 1
            return replies[min(i, len(replies) - 1)]

        self.loop.query_handler.query_sync_stream = mock_sync
        self.loop.query_handler.query_stream = MagicMock(return_value="All done.")
        self.loop.run_agentic_query("Do it")

        self.assertEqual(len(seen), 2)
        self.assertIn(q.AgenticRunner.continuation_nudge("length"),
                      [m.get("content") for m in seen[1]])

    def test_empty_step_asks_to_continue_not_to_start(self):
        """An empty step nudges 'continue where you stopped' (bounded)."""
        seen = []
        idx = [0]

        def mock_sync(messages, model, **kwargs):
            seen.append([dict(m) for m in messages])
            idx[0] += 1
            return self._make_sync_response("") if idx[0] == 1 else \
                self._make_sync_response("Done.")

        self.loop.query_handler.query_sync_stream = mock_sync
        self.loop.query_handler.query_stream = MagicMock(return_value="Done.")
        self.loop.run_agentic_query("Say done")

        self.assertEqual(len(seen), 2)
        texts = [m.get("content") for m in seen[1]]
        self.assertIn(q.AgenticRunner.continuation_nudge("empty"), texts)
        self.assertNotIn("Please provide a tool call or your final answer.", texts)

    def test_reasoning_only_step_names_the_cause(self):
        """A clean turn with reasoning but no content/call gets its own nudge.

        A thinking backend (MaaS/GLM-5.x) can end a turn with everything in the
        private reasoning channel. The old "empty" nudge did not say why, so the
        model re-derived the same analysis instead of emitting the call.
        """
        seen = []
        idx = [0]

        def mock_sync(messages, model, **kwargs):
            seen.append([dict(m) for m in messages])
            idx[0] += 1
            if idx[0] == 1:
                on_chunk = kwargs.get("on_chunk")
                if on_chunk:
                    on_chunk("private analysis that never became content", "", False)
                return self._make_sync_response("")
            return self._make_sync_response("Done.")

        self.loop.query_handler.query_sync_stream = mock_sync
        self.loop.query_handler.query_stream = MagicMock(return_value="Done.")
        self.loop.run_agentic_query("Say done")

        self.assertEqual(len(seen), 2)
        texts = [m.get("content") for m in seen[1]]
        self.assertIn(q.AgenticRunner.continuation_nudge("reasoning"), texts)
        self.assertNotIn(q.AgenticRunner.continuation_nudge("empty"), texts)

    def test_empty_steps_are_bounded(self):
        """A model that always returns nothing is nudged twice, then finalized."""
        calls = []

        def mock_sync(*args, **kwargs):
            calls.append(1)
            return self._make_sync_response("")

        self.loop.query_handler.query_sync_stream = mock_sync
        self.loop.query_handler.query_stream = MagicMock(return_value="")
        self.loop.run_agentic_query("Say something")

        self.assertEqual(len(calls), 1 + q.AGENTIC_TRUNCATED_STEP_MAX)

    def test_corrupt_native_tool_call_is_not_executed_with_empty_args(self):
        """Truncated native arguments must not run the tool with `{}`."""
        bad = {"choices": [{"message": {
            "role": "assistant", "content": "",
            "tool_calls": [{"id": "c1", "type": "function",
                            "function": {"name": "write_file",
                                         "arguments": '{"file": "x.txt", "content": "y'}}],
        }}]}
        valid = {"choices": [{"message": {
            "role": "assistant", "content": "",
            "tool_calls": [{"id": "c2", "type": "function",
                            "function": {"name": "write_file",
                                         "arguments": '{"file": "x.txt", "content": "ok"}'}}],
        }}]}
        seen, idx = [], [0]
        replies = [bad, valid, self._make_sync_response("Written.")]

        def mock_sync(messages, model, **kwargs):
            seen.append([dict(m) for m in messages])
            i = idx[0]
            idx[0] += 1
            return replies[min(i, len(replies) - 1)]

        self.loop.query_handler.query_sync_stream = mock_sync
        self.loop.query_handler.query_stream = MagicMock(return_value="Written.")
        with patch.object(self.loop.tool_registry, "execute",
                          wraps=self.loop.tool_registry.execute) as spy:
            self.loop.run_agentic_query("Write x.txt")

        # Step 1 (corrupt) must not execute; step 2 (valid) must.
        self.assertEqual(spy.call_count, 1, "a corrupt native call must not run")
        self.assertEqual(spy.call_args[0][0], "write_file")
        self.assertTrue(spy.call_args[0][1].get("file_path") or spy.call_args[0][1].get("file"))
        # And the model was asked to re-emit it rather than to finalize.
        self.assertIn(q.AgenticRunner.continuation_nudge("unparsed"),
                      [m.get("content") for m in seen[1]])

    def test_parse_calls_drops_unparseable_native_arguments(self):
        """`parse_calls` keeps "" (zero-arg tool) but drops corrupt JSON."""
        execution = q.ToolExecution(self.ctx, self.loop.tool_registry, self.loop.parser)
        blank = [{"function": {"name": "path_policy", "arguments": ""}}]
        corrupt = [{"function": {"name": "write_file", "arguments": '{oops'}}]
        nameless = [{"function": {"name": "", "arguments": "{}"}}]
        self.assertEqual(execution.parse_calls("", blank),
                         [{"tool": "path_policy", "arguments": {}}])
        self.assertEqual(execution.parse_calls("", corrupt), [])
        self.assertEqual(execution.parse_calls("", nameless), [])

    def test_truncated_inline_tool_call_is_nudged(self):
        """An unbalanced inline `{"tool": …` (inline-format model) is not an answer."""
        truncated = ('{"tool": "write_file", "arguments": {"file": "x.txt", '
                     '"content": "a long body that never closes')
        seen, idx = [], [0]
        replies = [self._make_sync_response(truncated),
                   self._make_sync_response(
                       '{"tool": "write_file", "arguments": {"file": "x.txt", "content": "ok"}}'),
                   self._make_sync_response("Written.")]

        def mock_sync(messages, model, **kwargs):
            seen.append([dict(m) for m in messages])
            i = idx[0]
            idx[0] += 1
            return replies[min(i, len(replies) - 1)]

        self.loop.query_handler.query_sync_stream = mock_sync
        self.loop.query_handler.query_stream = MagicMock(return_value="Written.")
        self.loop.run_agentic_query("Write x.txt")

        self.assertEqual(len(seen), 3, "the truncated call must be re-asked, then executed")
        self.assertIn(q.AgenticRunner.continuation_nudge("unparsed"),
                      [m.get("content") for m in seen[1]])
        with open("x.txt", encoding="utf-8") as f:
            self.assertEqual(f.read(), "ok")


class TestReActLoopDoesNotLitterTheCheckout(unittest.TestCase):
    """The mocked ReAct tests must leave the repository checkout untouched.

    They drive the *real* tool handlers with relative paths, so an earlier
    version wrote `hello.txt`, `x.txt` and `z.txt` at the repo root after every
    suite run (visible in `git status` as untracked files). The class now runs
    inside a TemporaryDirectory; this pins the contract by running the
    write-heaviest case and comparing the checkout listing before/after.
    """

    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def test_write_case_leaves_no_file_in_the_checkout(self):
        before = set(os.listdir(self.REPO_ROOT))
        cwd_before = os.getcwd()
        try:
            result = unittest.TestResult()
            TestReActLoopUnit("test_tool_call_triggers_execution").run(result)
            self.assertTrue(
                result.wasSuccessful(),
                msg=f"representative case failed: {result.failures or result.errors}")
            self.assertEqual(
                set(os.listdir(self.REPO_ROOT)), before,
                "mocked ReAct tests must not create files in the repository checkout")
        finally:
            os.chdir(cwd_before)

    def test_cwd_is_restored_after_the_case(self):
        cwd_before = os.getcwd()
        try:
            result = unittest.TestResult()
            TestReActLoopUnit("test_max_iterations_reached").run(result)
            self.assertTrue(result.wasSuccessful())
            self.assertEqual(os.path.realpath(os.getcwd()), os.path.realpath(cwd_before),
                             "setUp/tearDown must restore the caller's working directory")
        finally:
            os.chdir(cwd_before)


class TestStuckDetection(unittest.TestCase):
    """Test the _is_stuck and _call_with_timeout helpers."""

    def setUp(self):
        self.ctx = q.CommandContext()
        self.loop = q.ChatLoop(self.ctx)

    def test_is_stuck_normal_text(self):
        """Diverse text should not be detected as stuck."""
        text = ("The quick brown fox jumps over the lazy dog. "
                "Pack my box with five dozen liquor jugs. "
                "How vexingly quick daft zebras jump! "
                "The five boxing wizards jump quickly. "
                "Sphinx of black quartz, judge my vow. " * 5)
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

    def test_is_stuck_indentation_not_stuck(self):
        """Deep indentation / trailing whitespace must not be flagged as stuck."""
        body = "\n".join(f"        value_{i} = {i}" for i in range(30))
        text = "def process():\n" + body + "\n" + " " * 80
        self.assertGreater(len(text), 200)
        self.assertFalse(self.loop._is_stuck(text))

    def test_is_stuck_whitespace_tail_not_stuck(self):
        """A response ending in trailing whitespace/newlines must not be flagged."""
        text = ("The quick brown fox jumps over the lazy dog. "
                "Pack my box with five dozen liquor jugs. " * 4
                + "\n" * 60 + " " * 60)
        self.assertFalse(self.loop._is_stuck(text))

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


class TestToolCallParserExtraction(unittest.TestCase):
    """Pin the ToolCallParser extraction (parsing moved out of ChatLoop).

    `ChatLoop` keeps thin doors for the existing call sites; these tests pin
    both the delegation and the compatibility surfaces other tests rely on.
    """

    def setUp(self):
        self.ctx = q.CommandContext()
        self.ctx.lazy_tool = False
        self.loop = q.ChatLoop(self.ctx)

    def test_parser_is_the_implementation(self):
        self.assertIsInstance(self.loop.parser, q.ToolCallParser)

    def test_unbound_static_aliases_still_work(self):
        """Tests call these on the class, not on an instance."""
        self.assertIsInstance(q.ChatLoop.__dict__['_extract_json_balanced'], staticmethod)
        self.assertEqual(q.ChatLoop._extract_json_balanced('{"a": 1}', 0), '{"a": 1}')
        text = '{"tool": 1} {"tool": 2}'
        self.assertEqual(q.ChatLoop._find_tool_call_brace('x {"tool": 1}'), 2)
        self.assertEqual(text[q.ChatLoop._rfind_tool_call_brace(text):], '{"tool": 2}')
        self.assertEqual(q.ChatLoop._normalize_tool_json('{"tool": "a", "arguments": {}}'),
                         {"tool": "a", "arguments": {}})

    def test_doors_work_on_an_instance_built_without_init(self):
        """`object.__new__(ChatLoop)` fixtures have no `.parser` attribute."""
        bare = object.__new__(q.ChatLoop)
        bare.ctx = self.ctx
        self.assertEqual(bare.parse_tool_call('{"tool": "a", "arguments": {}}'),
                         {"tool": "a", "arguments": {}})
        self.assertEqual(len(bare.parse_tool_calls('{"tool": "a", "arguments": {}}')), 1)


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

    def test_internal_shape_with_string_arguments(self):
        """`{"tool": …}` with a JSON-string `arguments` is parsed (M7).

        OpenAI-trained models send string arguments in this shape too; leaving
        them as a string made the executor call `.items()` on a str and abort
        the whole ReAct session.
        """
        result = self.loop._normalize_tool_json(
            '{"tool": "run_command", "arguments": "{\\"command\\": \\"ls\\"}"}')
        self.assertEqual(result["tool"], "run_command")
        self.assertEqual(result["arguments"], {"command": "ls"})

    def test_internal_shape_unparsable_arguments_are_kept_raw(self):
        result = self.loop._normalize_tool_json(
            '{"tool": "run_command", "arguments": "{not json"}')
        self.assertEqual(result["arguments"], {"raw": "{not json"})

    def test_internal_shape_preserves_extra_keys(self):
        result = self.loop._normalize_tool_json(
            '{"tool": "x", "arguments": {"a": 1}, "id": "call_1"}')
        self.assertEqual(result["id"], "call_1")
        self.assertEqual(result["arguments"], {"a": 1})


class TestToolCallJsonAcceptability(unittest.TestCase):
    """The XML-hybrid filter must not reject legitimate `<`/`>` arguments (M8)."""

    def test_accepts_shell_redirection(self):
        self.assertTrue(q.tool_call_json_acceptable(
            {"tool": "run_command",
             "arguments": {"command": "sort < in.txt > out.txt"}}))

    def test_accepts_comparison_in_patch(self):
        self.assertTrue(q.tool_call_json_acceptable(
            {"tool": "patch",
             "arguments": {"content": "if (a < b && c > d) return;"}}))

    def test_accepts_cxx_template_angle_brackets(self):
        self.assertTrue(q.tool_call_json_acceptable(
            {"tool": "write_file",
             "arguments": {"content": "std::vector<int> v;"}}))

    def test_rejects_real_xml_tags(self):
        for content in ('<tool_call>{"tool": "x"}</tool_call>',
                        '</function>',
                        '<div class="x">y</div>'):
            self.assertFalse(q.tool_call_json_acceptable(
                {"tool": "x", "arguments": {"a": content}}), content)

    def test_empty_or_none_is_rejected(self):
        self.assertFalse(q.tool_call_json_acceptable(None))
        self.assertFalse(q.tool_call_json_acceptable({}))


class TestParseToolCallsPassZero(unittest.TestCase):
    """Pass 0 (text starting with `{`) must normalize like every other path (M7)."""

    def setUp(self):
        self.ctx = q.CommandContext()
        self.ctx.lazy_tool = False
        self.loop = q.ChatLoop(self.ctx)

    def test_string_arguments_are_a_dict_after_pass_zero(self):
        calls = self.loop.parse_tool_calls(
            '{"tool": "run_command", "arguments": "{\\"command\\": \\"ls\\"}"}')
        self.assertEqual(len(calls), 1)
        self.assertIsInstance(calls[0]["arguments"], dict)
        self.assertEqual(calls[0]["arguments"], {"command": "ls"})

    def test_openai_function_shape_at_the_start_is_normalized(self):
        calls = self.loop.parse_tool_calls(
            '{"type": "function", "function": {"name": "run_command", '
            '"arguments": "{\\"command\\": \\"ls\\"}"}}')
        self.assertEqual(len(calls), 1, calls)
        self.assertEqual(calls[0]["tool"], "run_command")
        self.assertEqual(calls[0]["arguments"], {"command": "ls"})


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

    def test_execute_does_not_mutate_the_caller_args(self):
        """Alias normalization must not rewrite the parser's dict in place."""
        args = {"directory": "."}
        result = self.reg.execute("list_directory", args)
        self.assertTrue(result["success"], msg=result.get("error"))
        self.assertEqual(args, {"directory": "."})


class TestLazyToolMode(unittest.TestCase):
    """Test lazy_tool mode in parse_tool_call."""

    def setUp(self):
        self.ctx = q.CommandContext()
        self.ctx.lazy_tool = False  # default
        self.loop = q.ChatLoop(self.ctx)

    def test_strict_rejects_embedded_bare_json(self):
        """Strict mode rejects bare JSON tool calls in the middle of text (trailing text after JSON)."""
        text = 'Some explanation then {"tool": "run_command", "arguments": {"command": "ls"}} and more'
        result = self.loop.parse_tool_call(text)
        self.assertIsNone(result)

    def test_strict_accepts_trailing_bare_json_after_prose(self):
        """Strict mode accepts a bare JSON tool call at the very END of text, even after a prose preamble."""
        text = 'Some explanation then {"tool": "run_command", "arguments": {"command": "ls"}}'
        result = self.loop.parse_tool_call(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["tool"], "run_command")

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


class TestSameToolLoopGuard(unittest.TestCase):
    """Same-tool-loop guard: repeats are allowed, abort only after N in a row."""

    def setUp(self):
        self.ctx = q.CommandContext()
        self.loop = object.__new__(q.ChatLoop)
        self.loop.ctx = self.ctx
        # The guard counts calls; execution itself is stubbed out.
        self.loop._tool_exec().run_single = MagicMock(return_value={
            "observation": "[read_file] ok", "raw": "ok",
            "success": True, "cancelled": False})

    def _call(self, n=1, last=None, count=0):
        calls = [{"tool": "read_file", "arguments": {"file_path": "x.txt"}} for _ in range(n)]
        return self.loop._execute_tool_calls(calls, last, count, 1, None, [], "", [])

    def test_single_repeat_is_allowed(self):
        """Two identical calls across two batches run fine (streak carries to 2)."""
        obs, raw, abort, last, count, final = self._call(n=1)
        self.assertFalse(abort)
        self.assertEqual(count, 1)
        obs, raw, abort, last, count, final = self._call(n=1, last=last, count=count)
        self.assertFalse(abort)
        self.assertEqual(count, 2)
        self.assertEqual(final, "")

    def test_loop_aborts_at_max_in_single_batch(self):
        """AGENTIC_SAME_TOOL_MAX identical calls: 9 run, the 10th is blocked."""
        obs, raw, abort, last, count, final = self._call(n=q.AGENTIC_SAME_TOOL_MAX)
        self.assertTrue(abort)
        self.assertEqual(final, "[Agentic: model stuck in tool loop]")
        self.assertEqual(count, q.AGENTIC_SAME_TOOL_MAX)
        self.assertEqual(len(obs), q.AGENTIC_SAME_TOOL_MAX - 1)

    def test_loop_aborts_across_batches(self):
        """The streak carries across _execute_tool_calls invocations."""
        obs, raw, abort, last, count, final = self._call(n=1)
        self.assertFalse(abort)
        self.assertEqual(count, 1)
        obs, raw, abort, last, count, final = self._call(n=q.AGENTIC_SAME_TOOL_MAX - 1, last=last, count=count)
        self.assertTrue(abort)
        self.assertEqual(count, q.AGENTIC_SAME_TOOL_MAX)

    def test_streak_resets_when_args_change(self):
        """A, A, B (different args) resets the streak on B — never aborts."""
        a1 = {"tool": "read_file", "arguments": {"file_path": "a.txt"}}
        b = {"tool": "read_file", "arguments": {"file_path": "b.txt"}}
        calls = [a1, a1, b]
        obs, raw, abort, last, count, final = self.loop._execute_tool_calls(
            calls, None, 0, 1, None, [], "", [])
        self.assertFalse(abort)
        self.assertEqual(count, 1)
        self.assertEqual(len(obs), 3)

    def test_streak_resets_on_different_tool(self):
        """A, A, B(tool), A keeps count per consecutive run — no false abort."""
        a = {"tool": "read_file", "arguments": {"file_path": "a.txt"}}
        b = {"tool": "list_directory", "arguments": {"path": "."}}
        calls = [a, a, b, a]
        obs, raw, abort, last, count, final = self.loop._execute_tool_calls(
            calls, None, 0, 1, None, [], "", [])
        self.assertFalse(abort)
        self.assertEqual(count, 1)
        self.assertEqual(len(obs), 4)


class TestToolExecutionExtraction(unittest.TestCase):
    """Pin the ToolExecution extraction: usable without ChatLoop, doors intact."""

    def setUp(self):
        self.ctx = q.CommandContext()
        self.ctx.lazy_tool = False

    def test_usable_without_chatloop(self):
        executor = q.ToolExecution(self.ctx, None, q.ToolCallParser(self.ctx))
        calls = executor.parse_calls('{"tool": "read_file", "arguments": {"file_path": "x"}}', [])
        self.assertEqual(calls[0]["tool"], "read_file")
        messages = []
        # Appending now stamps token counts; keep the test offline.
        with patch.object(q.CommandContext, "tokenize", return_value=5):
            executor.append_messages(messages, "text", [], calls, ["[read_file] ok"], ["ok"], False)
        self.assertEqual(messages[0]["role"], "assistant")
        self.assertTrue(messages[1]["content"].startswith("Tool result:"))

    def test_unbound_static_aliases(self):
        self.assertIsInstance(q.ChatLoop.__dict__['_cap_tool_observation'], staticmethod)
        self.assertIsInstance(q.ChatLoop.__dict__['_complete_stream_tool_calls'], staticmethod)
        self.assertEqual(q.ChatLoop._cap_tool_observation("read_file", "x" * 9000), "x" * 9000)
        self.assertEqual(q.ChatLoop._complete_stream_tool_calls(
            [{"function": {"name": "a", "arguments": '{"x": 1}'}}])[0]["function"]["name"], "a")

    def test_doors_resolve_one_lazy_executor(self):
        loop = object.__new__(q.ChatLoop)
        loop.ctx = self.ctx
        first = loop._tool_exec()
        self.assertIsInstance(first, q.ToolExecution)
        self.assertIs(first, loop._tool_exec())


class TestSessionAccessInventory(unittest.TestCase):
    """A1: `/agentic files` reports what the tools actually observed.

    The inventory is fed from `ToolRegistry.execute` (not from the shell gate),
    so it works with `/agentic log off` and in `work`/`exec` (bwrap) where the
    gate is skipped. A refusal must read `denied`, never `write`.
    """

    def setUp(self):
        self.ctx = q.CommandContext()
        # CommandContext is a singleton: restore the flags this class flips,
        # or the next test class inherits auto-confirm/logging.
        self._saved_auto = self.ctx.auto_confirm
        self._saved_logging = self.ctx.agentic_logging
        self.addCleanup(self._restore_ctx)
        self.ctx.auto_confirm = True
        self.ctx.agentic_logging = False
        self.ctx.session_access.reset()
        self.scratch = tempfile.mkdtemp(prefix="a1_scratch_")
        self.addCleanup(shutil.rmtree, self.scratch, True)

    def _restore_ctx(self):
        self.ctx.auto_confirm = self._saved_auto
        self.ctx.agentic_logging = self._saved_logging
        self.ctx.session_access.reset()

    def _record_command(self, command, success=True, error=None, mode="host"):
        """Feed one synthetic run_command result through the inventory."""
        return self.ctx.session_access.record_tool(
            "run_command", {"command": command},
            {"success": success, "output": "", "error": error},
            executor_mode=mode, cwd=os.getcwd(), tmpdir=self.scratch)

    def _rows(self, op):
        return [row for (row_op, _path, _status, _sandbox), row in
                self.ctx.session_access.paths.items() if row_op == op]

    def test_three_writes_to_one_file_are_one_row(self):
        """The same file written three times is one path with a counter."""
        registry = q.ToolRegistry(self.ctx)
        with tempfile.TemporaryDirectory() as td:
            old = os.getcwd()
            os.chdir(td)
            try:
                for _ in range(3):
                    result = registry.execute(
                        "write_file", {"file_path": "out.txt", "content": "x"})
                    self.assertTrue(result["success"])
                    # The per-call summary is what run_single puts in the JSONL.
                    self.assertEqual(result["_access"]["status"], "ok")
            finally:
                os.chdir(old)
        rows = self._rows("write")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["count"], 3)
        self.assertTrue(rows[0]["path"].endswith("/out.txt"))

    def test_common_writers_map_to_read_and_write(self):
        """`cp`, `sed -i` and `tar -cf` keep the H0 direction."""
        cases = (
            ("cp a.txt b.txt", {("write", "b.txt"), ("read", "a.txt")}),
            ("sed -i 's/a/b/' f.txt", {("write", "f.txt"), ("read", "f.txt")}),
            ("tar -cf out.tar f.txt", {("write", "out.tar"), ("read", "f.txt")}),
        )
        for command, expected in cases:
            with self.subTest(command=command):
                self.ctx.session_access.reset()
                self._record_command(command)
                got = {(row["op"], os.path.basename(row["path"]))
                       for row in self.ctx.session_access.paths.values()}
                self.assertEqual(got, expected)

    def test_effective_cwd_is_used(self):
        """`cd / && echo x > rel` is recorded as the /rel the shell wrote (H1)."""
        self._record_command("cd / && echo x > rel")
        self.assertEqual([row["path"] for row in self._rows("write")], ["/rel"])

    def test_denied_command_is_denied_not_a_write(self):
        """A refused command contributes no path — the report says denied."""
        access = self._record_command(
            "cp f /var/x", success=False,
            error="BLOCKED (rejected): denied by permission rules")
        self.assertEqual(access["status"], "denied")
        self.assertEqual(self.ctx.session_access.paths, {})
        rows = list(self.ctx.session_access.commands.values())
        self.assertEqual([(row["command"], row["status"]) for row in rows],
                         [("cp f /var/x", "denied")])

    def test_real_gate_denial_is_denied_not_a_write(self):
        """The real non-TTY gate refuses `git diff --ext-diff` (H6); no path is claimed."""
        registry = q.ToolRegistry(self.ctx)
        self.ctx.auto_confirm = False
        with patch.object(q.ToolRegistry, "_confirm", return_value=True), \
                patch("sys.stdin.isatty", return_value=False):
            result = registry.execute(
                "run_command", {"command": "git diff --ext-diff"})
        self.assertFalse(result["success"])
        access = result.pop("_access", None)
        self.assertEqual(access["status"], "denied")
        self.assertNotIn("paths", access)
        self.assertEqual(self.ctx.session_access.paths, {})

    def test_container_paths_are_not_host_paths(self):
        """Container `/workspace/...` names must not be presented as host paths."""
        access = self._record_command(
            "cp /workspace/a /workspace/b", mode="container")
        self.assertEqual(access["sandbox"], "container")
        self.assertNotIn("paths", access)
        self.assertEqual(self.ctx.session_access.paths, {})

    def test_bwrap_commands_are_still_recorded(self):
        """`work`/`exec` skip the gate; the inventory must not skip them."""
        access = self._record_command("echo x > out.txt", mode="bwrap")
        self.assertEqual(access["sandbox"], "bwrap")
        self.assertTrue(any(row["op"] == "write" for row in access["paths"]))
        self.assertTrue(self._rows("write"))
        text = "\n".join(self.ctx.session_access.status_lines())
        self.assertIn("[bwrap]", text)

    def test_bwrap_paths_are_not_host_paths(self):
        """`/tmp/a` inside bwrap is the sandbox tmpfs, not the host's /tmp/a."""
        access = self._record_command("echo x > /tmp/a", mode="bwrap")
        write = self._rows("write")[0]
        self.assertEqual(write["sandbox"], "bwrap")
        self.assertEqual(access["paths"][0]["sandbox"], "bwrap")
        text = "\n".join(self.ctx.session_access.status_lines())
        self.assertIn("[bwrap] /tmp/a", text)

    def test_bwrap_home_is_shown_raw(self):
        """bwrap HOME is private; the host `~` expansion must not be printed."""
        self._record_command("echo x > ~/a", mode="bwrap")
        text = "\n".join(self.ctx.session_access.status_lines())
        self.assertIn("[bwrap] ~/a", text)
        self.assertNotIn(os.path.realpath(os.path.expanduser("~/a")), text)

    def test_run_python_is_opaque_not_unknown_path(self):
        """Inline code decides its own writes; the call says writes unknown."""
        access = self.ctx.session_access.record_tool(
            "run_python", {"code": "open('x', 'w').write('y')"},
            {"success": True, "output": "", "error": None},
            cwd=os.getcwd(), tmpdir=self.scratch)
        self.assertEqual(access["command"], "run_python (inline code)")
        self.assertTrue(access["opaque"])
        self.assertNotIn("paths", access)
        self.assertEqual(self.ctx.session_access.paths, {})
        text = "\n".join(self.ctx.session_access.status_lines())
        self.assertIn("run_python (inline code) [writes unknown]", text)

    def test_failed_write_file_leaves_a_trace(self):
        """A refused file tool must not disappear from the audit."""
        registry = q.ToolRegistry(self.ctx)
        result = registry.execute(
            "write_file", {"file_path": "/etc/a1-never-written", "content": "x"})
        self.assertFalse(result["success"])
        access = result.pop("_access", None)
        self.assertEqual(access["status"], "denied")
        self.assertEqual(access["command"], "write_file /etc/a1-never-written")
        self.assertEqual(self.ctx.session_access.paths, {})
        self.assertIn("write_file /etc/a1-never-written",
                      "\n".join(self.ctx.session_access.status_lines()))

    def test_failed_read_file_leaves_a_trace(self):
        """A missing file is a call with status error, not a silent no-op."""
        registry = q.ToolRegistry(self.ctx)
        result = registry.execute(
            "read_file", {"file_path": "definitely-missing-a1.txt"})
        self.assertFalse(result["success"])
        access = result.pop("_access", None)
        self.assertEqual(access["status"], "error")
        self.assertEqual(access["command"], "read_file definitely-missing-a1.txt")
        self.assertEqual(self.ctx.session_access.paths, {})

    def test_glob_records_its_search(self):
        """`glob` reads a tree; the report names the search root and the pattern."""
        registry = q.ToolRegistry(self.ctx)
        with tempfile.TemporaryDirectory() as td:
            old = os.getcwd()
            os.chdir(td)
            try:
                open("a.txt", "w").close()
                result = registry.execute("glob", {"pattern": "**/*.txt"})
                self.assertTrue(result["success"])
                access = result.pop("_access", None)
                self.assertEqual(access["command"], "glob **/*.txt")
                reads = [row["path"] for row in self._rows("read")]
                self.assertIn(os.path.realpath(td), reads)
            finally:
                os.chdir(old)

    def test_report_lists_paths_and_commands(self):
        """The rendered report separates writes, reads and commands."""
        self._record_command("cp a.txt b.txt")
        text = "\n".join(self.ctx.session_access.status_lines())
        self.assertIn("Writes (1)", text)
        self.assertIn("Reads (1)", text)
        self.assertIn("Commands / tool calls (1)", text)
        self.assertIn("×1", text)

    def test_empty_and_logging_off_message_and_memory_source(self):
        """An empty inventory is explicit and does not depend on JSONL logging."""
        loop = object.__new__(q.ChatLoop)
        loop.ctx = self.ctx
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            loop.run_handle_agentic("/agentic files")
        self.assertIn("nothing recorded yet this session", err.getvalue())

        # Logging stays off: the inventory is fed by tool execution, not the log.
        self.assertFalse(self.ctx.agentic_logging)
        self._record_command("echo hi")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            loop.run_handle_agentic("/agentic files")
        self.assertNotIn("nothing recorded yet this session", err.getvalue())
        self.assertIn("echo hi", err.getvalue())

    def test_reset_and_completion_surface(self):
        """`reset` clears the memory; `files` completes its own actions."""
        self._record_command("echo hi")
        self.assertTrue(self.ctx.session_access.commands)
        loop = object.__new__(q.ChatLoop)
        loop.ctx = self.ctx
        with contextlib.redirect_stderr(io.StringIO()):
            loop.run_handle_agentic("/agentic files reset")
        self.assertEqual(self.ctx.session_access.paths, {})
        self.assertEqual(self.ctx.session_access.commands, {})
        self.assertIn("files", q.AGENTIC_SUBCOMMANDS)
        self.assertEqual(q.AGENTIC_SECOND_LEVEL["files"], ["list", "reset"])

    def test_result_entry_carries_access(self):
        """The JSONL `type=result` entry carries the same per-call summary."""
        with tempfile.TemporaryDirectory() as home, \
                tempfile.TemporaryDirectory() as td, \
                patch.dict(os.environ, {"HOME": home}, clear=False):
            old = os.getcwd()
            os.chdir(td)
            try:
                registry = q.ToolRegistry(self.ctx)
                execution = q.ToolExecution(self.ctx, registry,
                                            q.ToolCallParser(self.ctx))
                log = q.AgenticLogger()
                try:
                    self.ctx.agentic_logging = True
                    with contextlib.redirect_stderr(io.StringIO()):
                        result = execution.run_single(
                            "write_file",
                            {"file_path": "a1.txt", "content": "x"},
                            1, log, [])
                    self.assertTrue(result["success"])
                    # The private key is consumed, not leaked into observations.
                    self.assertNotIn("_access", result)
                finally:
                    log.close()
                with open(log.path, encoding="utf-8") as handle:
                    entries = [json.loads(line) for line in handle if line.strip()]
            finally:
                os.chdir(old)
        entry = [item for item in entries if item.get("type") == "result"][-1]
        self.assertEqual(entry["access"]["status"], "ok")
        self.assertTrue(entry["access"]["paths"][0]["path"].endswith("/a1.txt"))


class TestAppendMessagesStampsTokens(unittest.TestCase):
    """Turns appended by the loop must carry `_tokens`.

    Without it, `calculate_context_tokens()` falls back to
    `estimate_token_count()`, whose regex count × 2.0 reports a measured
    1.7-2.1× the real size: the stored sum then disagreed with the outbound
    number the status bar shows (measured live: 382 150 stored vs 148 784 sent).
    """

    API_CALLS = [{"id": "c1", "type": "function",
                  "function": {"name": "write_file",
                               "arguments": '{"path": "a.py", "content": "x = 1"}'}}]

    def setUp(self):
        self.ctx = q.CommandContext()
        self.ctx.lazy_tool = False
        self.executor = q.ToolExecution(self.ctx, None, q.ToolCallParser(self.ctx))

    def _append(self, messages, send_tools_api):
        with patch.object(q.CommandContext, "tokenize",
                          side_effect=lambda text: max(1, len(text) // 4)):
            self.executor.append_messages(
                messages, "calling it", self.API_CALLS if send_tools_api else [],
                [{"tool": "write_file", "arguments": {"path": "a.py"}}],
                ["[write_file] ok"], ["ok"], send_tools_api)

    def test_native_path_stamps_both_messages(self):
        messages = []
        self._append(messages, True)
        self.assertEqual([m["role"] for m in messages], ["assistant", "tool"])
        for msg in messages:
            self.assertIn("_tokens", msg)
        # The call's arguments are part of the assistant turn's cost, not just
        # its text ("calling it" alone would be 2 tokens under this tokenizer).
        self.assertGreater(messages[0]["_tokens"], len("calling it") // 4)

    def test_inline_path_stamps_the_user_note(self):
        messages = []
        self._append(messages, False)
        self.assertEqual([m["role"] for m in messages], ["assistant", "user"])
        self.assertIn("_tokens", messages[1])

    def test_stored_sum_no_longer_falls_back_to_the_heuristic(self):
        messages = []
        self._append(messages, True)
        with patch.object(q, "estimate_token_count") as estimate:
            stored = self.ctx.calculate_context_tokens(messages)
        estimate.assert_not_called()
        self.assertEqual(stored, sum(m["_tokens"] for m in messages))

    def test_no_context_is_tolerated(self):
        """Bare fixtures build ToolExecution(None, ...); stamping must no-op."""
        executor = q.ToolExecution(None, None, q.ToolCallParser(None))
        messages = []
        executor.append_messages(messages, "text", [], [], ["obs"], ["obs"], False)
        self.assertEqual([m["role"] for m in messages], ["assistant", "user"])
        self.assertNotIn("_tokens", messages[0])


class TestSpillIsolation(unittest.TestCase):
    """Spill files are per-process.

    Several sessions legitimately run at once and share the spill root; before
    this, closing one session purged *every* session's files by prefix, so a
    model paging a spilled page got "not found" on a path handed out a turn
    earlier. Per-process directories also stop fixed names
    (unfinished_write_file, git_result_<action>) from overwriting each other.
    """

    def test_spill_dir_is_per_process(self):
        self.assertEqual(os.path.basename(q.CLUSTER_SPILL_DIR), f"pid-{os.getpid()}")
        self.assertEqual(os.path.dirname(q.CLUSTER_SPILL_DIR), q.SPILL_ROOT)

    def test_cleanup_is_scoped_to_this_process_and_sweeps_dead_sessions(self):
        with tempfile.TemporaryDirectory() as td:
            ours = os.path.join(td, f"pid-{os.getpid()}")
            dead = os.path.join(td, "pid-999999")
            live = os.path.join(td, "pid-1")
            for directory in (ours, dead, live):
                os.makedirs(directory)
                with open(os.path.join(directory, "fetch_result_x.txt"), "w") as f:
                    f.write("x")
            # pid 1 stays alive; 999999 is declared dead so the test does not
            # depend on the host's PID space.
            with patch.object(q, "CLUSTER_SPILL_DIR", ours), \
                 patch.object(q, "_pid_alive", side_effect=lambda pid: pid == 1):
                q._cleanup_spill_files()
            self.assertFalse(os.path.exists(ours), "own spill dir must be removed")
            self.assertFalse(os.path.exists(dead), "dead session's dir must be swept")
            self.assertTrue(os.path.exists(live), "a live session's dir must survive")

    def test_long_urls_do_not_collide_on_one_spill_file(self):
        """Two URLs sharing their first 64 sanitized chars must not overwrite."""
        prefix = "https://example.com/very/deep/path/" + "x" * 120
        alpha = q._fetch_spill_path(prefix + "/alpha")
        beta = q._fetch_spill_path(prefix + "/beta")
        self.assertNotEqual(alpha, beta)
        # ...and the name stays stable for a re-fetch of the same URL.
        self.assertEqual(alpha, q._fetch_spill_path(prefix + "/alpha"))


class TestAgenticRunnerExtraction(unittest.TestCase):
    """Pin the AgenticRunner extraction: step helpers usable without ChatLoop."""

    def setUp(self):
        self.ctx = q.CommandContext()
        self.ctx.lazy_tool = False
        # CommandContext is a singleton shared with the rest of the suite: pin the
        # backend so `_extract_sync_content` reads the Ollama response shape.
        self.ctx.backend = "ollama"
        self.runner = q.AgenticRunner(self.ctx, None, None, None,
                                      q.ToolCallParser(self.ctx), None)

    def test_static_helpers(self):
        self.assertTrue(q.AgenticRunner.is_stuck("ab" * 400))
        self.assertFalse(q.AgenticRunner.is_stuck("hello"))
        self.assertEqual(q.AgenticRunner.call_with_timeout(lambda x: x + 1, 5, 41), 42)

    def test_step_content_without_chatloop(self):
        text, calls = self.runner.step_content({"message": {"content": "hello"}})
        self.assertEqual(text, "hello")
        self.assertEqual(calls, [])

    def test_loop_door_reuses_one_runner(self):
        loop = object.__new__(q.ChatLoop)
        loop.ctx = self.ctx
        first = loop._agentic_runner()
        self.assertIsInstance(first, q.AgenticRunner)
        self.assertIs(first, loop._agentic_runner())


class TestClusterQueryTool(unittest.TestCase):
    """kubernetes_cluster_query tool: verb whitelist, backend auto-detect, command build."""

    # Realistic `oc api-resources` output (columns separated by 2+ spaces, the
    # SHORTNAMES cell empty for `bindings`). Resource validation is driven by
    # this table, so a change here changes which names are accepted.
    API_RESOURCES = (
        "NAME                     SHORTNAMES   APIVERSION          NAMESPACED   KIND\n"
        "bindings                              v1                  true         Binding\n"
        "configmaps               cm           v1                  true         ConfigMap\n"
        "deployments              deploy       apps/v1             true         Deployment\n"
        "events                   ev           v1                  true         Event\n"
        "namespaces               ns           v1                  false        Namespace\n"
        "nodes                    no           v1                  false        Node\n"
        "persistentvolumeclaims   pvc          v1                  true         PersistentVolumeClaim\n"
        "persistentvolumes        pv           v1                  false        PersistentVolume\n"
        "pods                     po           v1                  true         Pod\n"
        "services                 svc          v1                  true         Service\n"
        "storageclasses           sc           storage.k8s.io/v1   false        StorageClass\n"
    )

    def setUp(self):
        self.ctx = q.CommandContext()
        self.ctx.auto_confirm = True
        self.reg = q.ToolRegistry(ctx=self.ctx)
        self.commands = []
        # Cluster commands run in the harness (kubeconfig/HOME/must-gather live
        # there), never through the session sandbox executor.
        self.harness = q.Executor(mode="host")
        self.harness.run = MagicMock(side_effect=self._fake_run)
        self.reg._harness_executor = lambda: self.harness
        self.reg.executor.run = MagicMock(
            side_effect=AssertionError("cluster queries must not use the sandbox executor"))

    def _fake_run(self, command, timeout=120):
        self.commands.append(command)
        if "api-resources" in command:
            return {"stdout": self.API_RESOURCES, "stderr": "", "returncode": 0}
        if "current-context" in command:
            return {"stdout": "api.cluster.example\n", "stderr": "", "returncode": 0}
        return {"stdout": "NAME READY\npod/a 1/1 Running\n", "stderr": "", "returncode": 0}

    def _cmd(self, prefix):
        """Return the last command starting with `prefix` (order-independent)."""
        matches = [c for c in self.commands if c.startswith(prefix)]
        self.assertTrue(matches, msg=f"no command starting with {prefix!r}: {self.commands!r}")
        return matches[-1]

    def _run(self, **args):
        return self.reg.execute("kubernetes_cluster_query", args)

    def test_commands_run_in_the_harness_not_the_sandbox(self):
        """P0-2: oc/kubectl/omc need the user's config, so the sandbox is bypassed."""
        self.ctx.preset = "work"
        self.reg.executor = q.Executor(mode="bwrap", bwrap_project="ro", bwrap_net="none")
        self.reg.executor.run = MagicMock(
            side_effect=AssertionError("cluster queries must not use the sandbox executor"))
        with patch.dict(os.environ, {"KUBECONFIG": "/tmp/kube"}, clear=False), \
             patch.object(q.shutil, "which", return_value="/usr/bin/oc"):
            result = self._run(verb="get", resource="pods")
        self.assertTrue(result["success"], msg=result.get("error", ""))
        self.assertEqual(self.harness.mode, "host")
        self._cmd("oc get pods")
        self._cmd("oc api-resources")

    def test_unknown_verb_rejected(self):
        result = self._run(verb="delete")
        self.assertFalse(result["success"])
        self.assertIn("Unsupported verb", result["error"])

    def test_bad_output_rejected(self):
        result = self._run(verb="get", resource="pods", output="xml")
        self.assertFalse(result["success"])
        self.assertIn("output format", result["error"])

    def test_resource_required(self):
        result = self._run(verb="get")
        self.assertFalse(result["success"])
        self.assertIn("resource is required", result["error"])

    def test_auto_kubeconfig_uses_oc(self):
        with patch.dict(os.environ, {"KUBECONFIG": "/tmp/kube"}, clear=False), \
             patch.object(q.shutil, "which", return_value="/usr/bin/oc"):
            result = self._run(verb="get", resource="pods", namespace="default")
        self.assertTrue(result["success"], msg=result.get("error"))
        self.assertIn("oc config current-context", self.commands)
        self.assertIn("oc get pods -n default -o json", self.commands)
        self.assertIn("platform: oc", result["output"])
        self.assertIn("context: api.cluster.example", result["output"])

    def test_auto_no_kubeconfig_offline_when_omc_configured(self):
        with patch.dict(os.environ, {"KUBECONFIG": ""}, clear=False), \
             patch.object(q.shutil, "which",
                          side_effect=lambda c: f"/usr/bin/{c}" if c == "omc" else None), \
             patch.object(q.os.path, "exists", return_value=True):
            result = self._run(verb="get", resource="pods")
        self.assertTrue(result["success"], msg=result.get("error"))
        self.assertTrue(self.commands[0].startswith("omc get pods -o json"))
        self.assertIn("platform: omc", result["output"])

    def test_omc_explicit_dir_runs_use(self):
        with patch.object(q.shutil, "which", return_value="/usr/bin/omc"):
            result = self._run(cluster_type="omc", verb="get", resource="pods",
                               offline_dir="/tmp/mg")
        self.assertTrue(result["success"], msg=result.get("error"))
        self.assertEqual(self.commands[0], "omc use /tmp/mg")
        self.assertTrue(self.commands[1].startswith("omc get pods"))
        self.assertIn("must-gather: /tmp/mg", result["output"])

    def test_omc_config_fallback_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = os.path.join(tmp, "omc.json")
            with open(cfg, "w") as f:
                json.dump({"must_gather_dir": "/tmp/mg"}, f)
            with patch.object(q.shutil, "which", return_value="/usr/bin/omc"), \
                 patch.object(q, "OMC_CONFIG_PATH", cfg):
                result = self._run(cluster_type="omc", verb="logs",
                                   resource="pod/mypod", tail=20)
            self.assertTrue(result["success"], msg=result.get("error"))
            self.assertEqual(self.commands[0], "omc use /tmp/mg")
            self.assertTrue(self.commands[1].startswith("omc logs pod/mypod --tail=20"))
            self.assertIn("must-gather: /tmp/mg", result["output"])

    def test_omc_without_any_dir_errors(self):
        with patch.object(q.shutil, "which", return_value="/usr/bin/omc"), \
             patch.object(q, "OMC_CONFIG_PATH", "/nonexistent/omc.json"):
            result = self._run(cluster_type="omc", verb="get", resource="pods")
        self.assertFalse(result["success"])
        self.assertIn("must-gather dir", result["error"])

    def test_current_context_verb(self):
        with patch.dict(os.environ, {"KUBECONFIG": "/tmp/kube"}, clear=False), \
             patch.object(q.shutil, "which", return_value="/usr/bin/oc"):
            result = self._run(verb="current-context")
        self.assertTrue(result["success"], msg=result.get("error"))
        self.assertEqual(self.commands[0], "oc config current-context")
        self.assertIn("context: api.cluster.example", result["output"])

    def test_events_verb_and_plan_mode_allowed(self):
        self.assertIn("kubernetes_cluster_query", q.PLAN_MODE_TOOLS)
        with patch.dict(os.environ, {"KUBECONFIG": "/tmp/kube"}, clear=False), \
             patch.object(q.shutil, "which", return_value="/usr/bin/oc"):
            result = self._run(verb="events", namespace="default")
        self.assertTrue(result["success"], msg=result.get("error"))
        self.assertTrue(self.commands[1].startswith("oc get events -n default -o json"))

    def test_large_result_spilled_to_file(self):
        """Results over the threshold spill to a per-query file; read_file can page it."""
        with tempfile.TemporaryDirectory() as tmp:
            spill_dir = os.path.join(tmp, "spill")
            spill_file = os.path.join(spill_dir, "cluster_result_get_pods.json")
            with patch.object(q, "CLUSTER_SPILL_DIR", spill_dir), \
                 patch.object(q.shutil, "which", return_value="/usr/bin/omc"):
                big = "PV-ITEM-7f3a9c\n" * 5000  # ~65KB, well over the spill threshold
                def fake_run(command, timeout=120):
                    if "omc get" in command:
                        return {"stdout": big, "stderr": "", "returncode": 0}
                    return {"stdout": "", "stderr": "", "returncode": 0}
                self.harness.run = MagicMock(side_effect=fake_run)
                result = self._run(cluster_type="omc", verb="get", resource="pods",
                                   offline_dir="/tmp/mg")
            self.assertTrue(result["success"], msg=result.get("error"))
            self.assertIn(spill_file, result["output"])
            self.assertIn("chars", result["output"])
            self.assertIn("lines", result["output"])
            # No payload inline — the observation is a pure pointer + counts, so the
            # content is not fed into context twice (only via read_file paging).
            self.assertNotIn("PV-ITEM", result["output"])
            self.assertLess(len(result["output"]), 600)
            with open(spill_file) as f:
                self.assertEqual(f.read(), big)
            # read_file can now page the spill file (session ACL allow was added)
            rd = self.reg.execute("read_file", {"file_path": spill_file})
            self.assertTrue(rd["success"], msg=rd.get("error"))
            self.assertIn("PV-ITEM", rd["output"])

    def test_scoped_spill_filenames_do_not_clobber(self):
        """Different resources spill to different files, so paging isn't clobbered."""
        with tempfile.TemporaryDirectory() as tmp:
            spill_dir = os.path.join(tmp, "spill")
            with patch.object(q, "CLUSTER_SPILL_DIR", spill_dir), \
                 patch.object(q.shutil, "which", return_value="/usr/bin/omc"):
                big_pv = "PV-ITEM-A\n" * 5000
                big_pvc = "PVC-ITEM-B\n" * 5000
                def fake_run(command, timeout=120):
                    if "omc get" in command and "persistentvolumes" in command:
                        return {"stdout": big_pv, "stderr": "", "returncode": 0}
                    if "omc get" in command and "persistentvolumeclaims" in command:
                        return {"stdout": big_pvc, "stderr": "", "returncode": 0}
                    return {"stdout": "", "stderr": "", "returncode": 0}
                self.harness.run = MagicMock(side_effect=fake_run)
                r_pv = self._run(cluster_type="omc", verb="get", resource="persistentvolumes",
                                 offline_dir="/tmp/mg")
                r_pvc = self._run(cluster_type="omc", verb="get", resource="persistentvolumeclaims",
                                  offline_dir="/tmp/mg")
            self.assertTrue(r_pv["success"] and r_pvc["success"])
            pv_file = os.path.join(spill_dir, "cluster_result_get_persistentvolumes.json")
            pvc_file = os.path.join(spill_dir, "cluster_result_get_persistentvolumeclaims.json")
            self.assertIn(pv_file, r_pv["output"])
            self.assertIn(pvc_file, r_pvc["output"])
            with open(pv_file) as f:
                self.assertEqual(f.read(), big_pv)
            with open(pvc_file) as f:
                self.assertEqual(f.read(), big_pvc)

    def test_custom_columns_output(self):
        """output=custom-columns builds -o custom-columns=<fields>."""
        with patch.dict(os.environ, {"KUBECONFIG": "/tmp/kube"}, clear=False), \
             patch.object(q.shutil, "which", return_value="/usr/bin/oc"):
            result = self._run(verb="get", resource="pv",
                               output="custom-columns",
                               fields="NAME:.metadata.name,SIZE:.spec.capacity.storage")
        self.assertTrue(result["success"], msg=result.get("error"))
        self.assertTrue(self._cmd("oc get pv").startswith(
            "oc get pv -o custom-columns=NAME:.metadata.name,SIZE:.spec.capacity.storage"))

    def test_custom_columns_requires_fields(self):
        result = self._run(verb="get", resource="pv", output="custom-columns")
        self.assertFalse(result["success"])
        self.assertIn("fields is required", result["error"])

    def test_jsonpath_output(self):
        with patch.dict(os.environ, {"KUBECONFIG": "/tmp/kube"}, clear=False), \
             patch.object(q.shutil, "which", return_value="/usr/bin/oc"):
            result = self._run(verb="get", resource="pv", output="jsonpath",
                               fields="{.items[*].metadata.name}")
        self.assertTrue(result["success"], msg=result.get("error"))
        # The jsonpath token carries shell metacharacters -> shlex.quote'd.
        self.assertTrue(self._cmd("oc get pv").startswith(
            "oc get pv -o 'jsonpath={.items[*].metadata.name}'"))

    def test_jsonpath_requires_fields(self):
        result = self._run(verb="get", resource="pv", output="jsonpath")
        self.assertFalse(result["success"])
        self.assertIn("fields is required", result["error"])

    def test_verb_args_are_shell_quoted(self):
        """_cluster_query_verb_args returns shlex.quote'd tokens (injection defense)."""
        parts = q._cluster_query_verb_args("get", {"resource": "pods; touch x"})
        self.assertEqual(parts, ["get", "'pods; touch x'", "-o", "json"])
        safe = q._cluster_query_verb_args("get", {"resource": "pods"})
        self.assertEqual(safe, ["get", "pods", "-o", "json"])
        ns = q._cluster_query_verb_args("get", {"resource": "pods", "namespace": "ns1 -o wide; id"})
        self.assertEqual(ns, ["get", "pods", "-n", "'ns1 -o wide; id'", "-o", "json"])

    def test_selector_filter(self):
        """selector (-l) narrows get and lands unquoted for safe values."""
        with patch.dict(os.environ, {"KUBECONFIG": "/tmp/kube"}, clear=False), \
             patch.object(q.shutil, "which", return_value="/usr/bin/oc"):
            result = self._run(verb="get", resource="pods", selector="app=nginx")
        self.assertTrue(result["success"], msg=result.get("error"))
        self.assertTrue(self._cmd("oc get pods").startswith("oc get pods -l app=nginx -o json"))

    def test_selector_rejected_for_describe(self):
        result = self._run(verb="describe", resource="pods", selector="app=nginx")
        self.assertFalse(result["success"])
        self.assertIn("selector", result["error"])

    def test_field_selector_get(self):
        """field_selector (--field-selector) narrows get."""
        with patch.dict(os.environ, {"KUBECONFIG": "/tmp/kube"}, clear=False), \
             patch.object(q.shutil, "which", return_value="/usr/bin/oc"):
            result = self._run(verb="get", resource="pods",
                               field_selector="status.phase=Running")
        self.assertTrue(result["success"], msg=result.get("error"))
        self.assertTrue(self._cmd("oc get pods").startswith(
            "oc get pods --field-selector=status.phase=Running -o json"))

    def test_field_selector_rejected_for_logs(self):
        result = self._run(verb="logs", resource="pod/mypod",
                           field_selector="status.phase=Running")
        self.assertFalse(result["success"])
        self.assertIn("field_selector", result["error"])

    def test_container_logs(self):
        """container (-c) selects one container for logs."""
        with patch.object(q.shutil, "which", return_value="/usr/bin/omc"):
            result = self._run(cluster_type="omc", verb="logs", resource="pod/mypod",
                               container="sidecar", tail=10, offline_dir="/tmp/mg")
        self.assertTrue(result["success"], msg=result.get("error"))
        self.assertTrue(self.commands[1].startswith("omc logs pod/mypod -c sidecar --tail=10"))

    def test_logs_resource_and_name_folded_into_one_target(self):
        """logs takes ONE positional target: resource + name -> TYPE/NAME."""
        with patch.object(q.shutil, "which", return_value="/usr/bin/omc"):
            result = self._run(cluster_type="omc", verb="logs", resource="pods",
                               name="mypod", offline_dir="/tmp/mg")
        self.assertTrue(result["success"], msg=result.get("error"))
        self.assertTrue(self.commands[1].startswith("omc logs pods/mypod --tail=50"))
        self.assertNotIn("pods mypod", self.commands[1])

    def test_logs_requires_a_pod_target(self):
        """logs with only a resource kind is rejected before execution."""
        result = self._run(cluster_type="omc", verb="logs", resource="pods",
                           offline_dir="/tmp/mg")
        self.assertFalse(result["success"])
        self.assertIn("logs requires a pod name", result["error"])
        self.assertEqual(self.commands, [])

    def test_logs_default_tail_is_50(self):
        """Bare logs is tailed by default (schema/prompt promise)."""
        with patch.object(q.shutil, "which", return_value="/usr/bin/omc"):
            result = self._run(cluster_type="omc", verb="logs", resource="pod/mypod",
                               offline_dir="/tmp/mg")
        self.assertTrue(result["success"], msg=result.get("error"))
        self.assertTrue(self.commands[1].startswith("omc logs pod/mypod --tail=50"))

    def test_logs_rejects_ambiguous_resource_and_name(self):
        """resource already carrying TYPE/NAME cannot also take a name."""
        result = self._run(cluster_type="omc", verb="logs", resource="pod/mypod",
                           name="other", offline_dir="/tmp/mg")
        self.assertFalse(result["success"])
        self.assertIn("not both", result["error"])

    def test_logs_selector_omits_positional_target(self):
        """logs -l selects pods itself; a resource positional would be invalid."""
        with patch.object(q.shutil, "which", return_value="/usr/bin/omc"):
            result = self._run(cluster_type="omc", verb="logs", resource="pods",
                               selector="app=nginx", offline_dir="/tmp/mg")
        self.assertTrue(result["success"], msg=result.get("error"))
        self.assertTrue(self.commands[1].startswith("omc logs -l app=nginx --tail=50"))

    def test_logs_selector_rejects_pod_name(self):
        result = self._run(cluster_type="omc", verb="logs", resource="pods",
                           name="mypod", selector="app=nginx", offline_dir="/tmp/mg")
        self.assertFalse(result["success"])
        self.assertIn("selector cannot be combined", result["error"])

    def test_container_rejected_for_get(self):
        result = self._run(verb="get", resource="pods", container="sidecar")
        self.assertFalse(result["success"])
        self.assertIn("container", result["error"])

    def test_injection_semicolon_is_quoted(self):
        """A `;`-injected resource becomes a quoted literal operand, not a second command."""
        with patch.dict(os.environ, {"KUBECONFIG": "/tmp/kube"}, clear=False), \
             patch.object(q.shutil, "which", return_value="/usr/bin/oc"):
            result = self._run(verb="get", resource="pods; touch /tmp/opencode/pwned")
        self.assertTrue(result["success"], msg=result.get("error"))
        self.assertIn("'pods; touch /tmp/opencode/pwned'", self._cmd("oc get"))
        self.assertNotIn("pods; touch /tmp/opencode/pwned -o", self._cmd("oc get"))

    def test_injection_command_substitution_is_quoted(self):
        """A `$()`-injected resource is quoted into a literal operand (was approved + executed pre-fix)."""
        with patch.dict(os.environ, {"KUBECONFIG": "/tmp/kube"}, clear=False), \
             patch.object(q.shutil, "which", return_value="/usr/bin/oc"):
            result = self._run(verb="get", resource="pods $(id > /tmp/opencode/pwned2)")
        self.assertTrue(result["success"], msg=result.get("error"))
        self.assertIn("'pods $(id > /tmp/opencode/pwned2)'", self._cmd("oc get"))

    def test_small_result_stays_inline(self):
        with patch.dict(os.environ, {"KUBECONFIG": "/tmp/kube"}, clear=False), \
             patch.object(q.shutil, "which", return_value="/usr/bin/oc"):
            result = self._run(verb="get", resource="pods")
        self.assertTrue(result["success"], msg=result.get("error"))
        self.assertNotIn("written to", result["output"])
        self.assertIn("pod/a 1/1 Running", result["output"])

    # --- Resource-name validation (a refusal, never a rewrite) --------------

    def test_unknown_resource_is_refused_with_closest_real_name(self):
        """`pvs` exists nowhere: refuse, name the real type, run no query."""
        with patch.dict(os.environ, {"KUBECONFIG": "/tmp/kube"}, clear=False), \
             patch.object(q.shutil, "which", return_value="/usr/bin/oc"):
            result = self._run(verb="get", resource="pvs")
        self.assertFalse(result["success"])
        self.assertIn("unknown resource 'pvs'", result["error"])
        self.assertIn("persistentvolumes", result["error"])
        self.assertFalse([c for c in self.commands if " oc get " in c],
                         msg=f"the query must not run: {self.commands!r}")

    def test_resource_is_sent_verbatim(self):
        """A valid short name is not rewritten to its full name."""
        with patch.dict(os.environ, {"KUBECONFIG": "/tmp/kube"}, clear=False), \
             patch.object(q.shutil, "which", return_value="/usr/bin/oc"):
            result = self._run(verb="get", resource="pv")
        self.assertTrue(result["success"], msg=result.get("error"))
        self.assertEqual(self._cmd("oc get"), "oc get pv -o json")

    def test_type_name_target_is_checked_but_not_rewritten(self):
        with patch.dict(os.environ, {"KUBECONFIG": "/tmp/kube"}, clear=False), \
             patch.object(q.shutil, "which", return_value="/usr/bin/oc"):
            ok = self._run(verb="get", resource="pod/mypod")
            bad = self._run(verb="get", resource="poddss/mypod")
        self.assertTrue(ok["success"], msg=ok.get("error"))
        self.assertEqual(self._cmd("oc get"), "oc get pod/mypod -o json")
        self.assertFalse(bad["success"])
        self.assertIn("poddss", bad["error"])

    def test_offline_passes_the_name_through(self):
        """omc has no api-resources: the backend decides, nothing is rewritten."""
        with patch.object(q.shutil, "which", return_value="/usr/bin/omc"):
            result = self._run(cluster_type="omc", verb="get", resource="pvs",
                               offline_dir="/tmp/mg")
        self.assertTrue(result["success"], msg=result.get("error"))
        self.assertEqual(self.commands, ["omc use /tmp/mg", "omc get pvs -o json"])

    def test_failed_discovery_passes_the_name_through(self):
        """A discovery failure disables validation; the backend's error is the answer."""
        def fake_run(command, timeout=120):
            self.commands.append(command)
            if "api-resources" in command:
                return {"stdout": "", "stderr": "forbidden", "returncode": 1}
            if "current-context" in command:
                return {"stdout": "api.cluster.example\n", "stderr": "", "returncode": 0}
            return {"stdout": "ok", "stderr": "", "returncode": 0}
        self.harness.run = MagicMock(side_effect=fake_run)
        with patch.dict(os.environ, {"KUBECONFIG": "/tmp/kube"}, clear=False), \
             patch.object(q.shutil, "which", return_value="/usr/bin/oc"):
            result = self._run(verb="get", resource="poddsss")
        self.assertTrue(result["success"], msg=result.get("error"))
        self.assertEqual(self._cmd("oc get"), "oc get poddsss -o json")

    def test_discovery_runs_once_per_session(self):
        with patch.dict(os.environ, {"KUBECONFIG": "/tmp/kube"}, clear=False), \
             patch.object(q.shutil, "which", return_value="/usr/bin/oc"):
            self._run(verb="get", resource="pods")
            self._run(verb="get", resource="nodes")
        self.assertEqual(sum("api-resources" in c for c in self.commands), 1)

    def test_pseudo_and_complex_resources_skip_discovery(self):
        with patch.dict(os.environ, {"KUBECONFIG": "/tmp/kube"}, clear=False), \
             patch.object(q.shutil, "which", return_value="/usr/bin/oc"):
            for resource in ("all", "pods,services", "pods; touch /tmp/x"):
                result = self._run(verb="get", resource=resource)
                self.assertTrue(result["success"], msg=result.get("error"))
        self.assertFalse([c for c in self.commands if "api-resources" in c])

    def test_events_skip_discovery(self):
        with patch.dict(os.environ, {"KUBECONFIG": "/tmp/kube"}, clear=False), \
             patch.object(q.shutil, "which", return_value="/usr/bin/oc"):
            result = self._run(verb="events", namespace="default")
        self.assertTrue(result["success"], msg=result.get("error"))
        self.assertFalse([c for c in self.commands if "api-resources" in c])

    def test_parse_api_resources_table(self):
        known = q._cluster_parse_api_resources(self.API_RESOURCES)
        self.assertEqual(known["pods"], "pods")
        self.assertEqual(known["po"], "pods")
        self.assertEqual(known["pod"], "pods")
        self.assertEqual(known["binding"], "bindings")  # empty SHORTNAMES row
        self.assertEqual(known["pvc"], "persistentvolumeclaims")
        self.assertEqual(known["sc"], "storageclasses")
        self.assertEqual(known["storageclass"], "storageclasses")

    def test_parse_api_resources_ignores_header_and_junk(self):
        self.assertEqual(q._cluster_parse_api_resources(""), {})
        self.assertEqual(q._cluster_parse_api_resources("NAME SHORTNAMES\n"), {})

    def test_api_resources_probe_is_allow_listed(self):
        """The probe must not prompt in strict plan mode (no shell, no TTY)."""
        for command in ("oc api-resources", "kubectl api-resources"):
            verdict = q.shell_check_command(command, q.DEFAULT_SHELL_PERMISSION)
            self.assertEqual(verdict["effect"], "allow", msg=command)

    def test_description_has_no_invented_name_prose(self):
        text = q.AGENTIC_TOOL_DEFS["kubernetes_cluster_query"]["description"]
        self.assertNotIn("do NOT invent", text)
        self.assertIn("full resource name", text)


HAS_GIT = q.shutil.which("git") is not None


def _git(cwd, *args):
    """Run a real git command in `cwd` for test fixture setup."""
    import subprocess
    return subprocess.run(["git", *args], cwd=cwd, check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True)


class _FakeGitProc:
    """Stand-in for `subprocess.Popen` in git_inspect argv/env assertions."""

    def __init__(self, output="", returncode=0):
        self.stdout = io.StringIO(output)
        self.returncode = returncode
        self.killed = False

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        return self.returncode


class TestGitInspect(unittest.TestCase):
    """git_inspect: read-only, fixed-argv repo inspection (the plan-mode git gap)."""

    def setUp(self):
        self.ctx = q.CommandContext()
        self.ctx.auto_confirm = True
        self.reg = q.ToolRegistry(ctx=self.ctx)

    def _run(self, **args):
        return self.reg.execute("git_inspect", args)

    def test_available_in_strict_plan_mode(self):
        self.assertIn("git_inspect", q.PLAN_MODE_TOOLS)
        self.assertNotIn("run_command", q.PLAN_MODE_TOOLS)
        self.ctx.plan_mode = True
        try:
            names = {t["function"]["name"] for t in self.reg.openai_tools_spec()}
            self.assertIn("git_inspect", names)
            self.assertIn("### git_inspect", self.reg.get_system_prompt_block())
        finally:
            self.ctx.plan_mode = False

    def test_unknown_action_rejected(self):
        result = self._run(action="commit")
        self.assertFalse(result["success"])
        self.assertIn("Unsupported action", result["error"])

    def test_ref_cannot_be_an_option(self):
        for ref in ("--output=/tmp/x", "-c", "--exec=id", "HEAD; rm -rf /"):
            result = self._run(action="show", ref=ref)
            self.assertFalse(result["success"], msg=ref)
            self.assertIn("Unsupported ref", result["error"])

    def test_path_outside_project_rejected(self):
        for path in ("../../etc/passwd", "/etc/passwd", "-x", "sub/../../out"):
            result = self._run(action="diff", path=path)
            self.assertFalse(result["success"], msg=path)
            self.assertIn("inside the project", result["error"])

    def test_argv_is_fixed_and_hardened(self):
        """No shell, no model text in argv, no repo-supplied hooks, no GIT_DIR.

        `diff.<driver>.textconv` and `core.fsmonitor` both execute arbitrary
        programs from the repo's own config, so a "read-only" tool that skipped
        these flags would still run code. A `GIT_DIR` inherited from the parent
        environment would redirect the whole command to another repository.
        """
        fake = _FakeGitProc("ok\n")
        with patch.dict(os.environ, {"GIT_DIR": "/tmp/elsewhere",
                                     "GIT_WORK_TREE": "/tmp/elsewhere",
                                     "GIT_INDEX_FILE": "/tmp/elsewhere/index"}, clear=False), \
             patch.object(q.subprocess, "Popen", return_value=fake) as popen:
            result = self._run(action="diff", limit="5")
        self.assertTrue(result["success"], msg=result.get("error"))
        argv = popen.call_args[0][0]
        kwargs = popen.call_args[1]
        self.assertEqual(argv[0], "git")
        self.assertIn("--no-optional-locks", argv)
        self.assertIn("core.fsmonitor=false", argv)
        self.assertIn("--no-textconv", argv)
        self.assertIn("--no-ext-diff", argv)
        self.assertFalse(kwargs["shell"])
        self.assertEqual(kwargs["cwd"], os.getcwd())
        for key in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
            self.assertNotIn(key, kwargs["env"])
        self.assertEqual(kwargs["env"]["GIT_TERMINAL_PROMPT"], "0")
        self.assertEqual(kwargs["env"]["GIT_OPTIONAL_LOCKS"], "0")
        # No ref was requested, so no revision may appear in argv.
        self.assertNotIn("HEAD", argv)
        self.assertTrue(all(isinstance(part, str) for part in argv))

    def test_ref_is_rejected_for_actions_that_would_ignore_it(self):
        """Silently dropping `ref` yields a confidently wrong answer.

        A model asking for `diff` of another branch used to receive the
        working-tree diff. Ref-taking actions are `log` and `show`; everything
        else refuses instead of guessing.
        """
        for action in ("status", "diff", "diff_staged", "diff_head"):
            with patch.object(q.subprocess, "Popen") as popen:
                result = self._run(action=action, ref="main")
            self.assertFalse(result["success"], msg=action)
            self.assertIn("does not take a ref", result["error"])
            popen.assert_not_called()

    def test_log_uses_ref_and_pathspec(self):
        capture = {"returncode": 0, "stdout": "abc123 one\n", "stderr": "",
                   "truncated": False, "timed_out": False}
        with patch.object(q, "_git_capture", return_value=capture) as fake:
            self._run(action="log", ref="main", limit=5, path="tests")
            argv = fake.call_args[0][0]
            self._run(action="log")
            default_argv = fake.call_args[0][0]
        self.assertIn("main", argv)
        self.assertIn("-n5", argv)
        self.assertIn("tests", argv)
        self.assertEqual(argv[argv.index("--") + 1], "tests")
        # `log` without a ref walks HEAD implicitly: no revision in argv.
        self.assertNotIn("HEAD", default_argv)

    def test_log_ref_cannot_be_an_option(self):
        for ref in ("--all", "-c", "--output=/tmp/x", "main; id"):
            result = self._run(action="log", ref=ref)
            self.assertFalse(result["success"], msg=ref)
            self.assertIn("Unsupported ref", result["error"])

    # ---- blame: per-line archaeology (bounded by -L, no textconv) ----------

    def test_blame_needs_path_and_lines(self):
        missing_path = self._run(action="blame", lines="1,10")
        self.assertFalse(missing_path["success"])
        self.assertIn("path", missing_path["error"])
        missing_lines = self._run(action="blame", path="a.txt")
        self.assertFalse(missing_lines["success"])
        self.assertIn("lines", missing_lines["error"])

    def test_blame_argv_is_hardened_and_bounded(self):
        capture = {"returncode": 0, "stdout": "abc123 (T 2026-01-01 1) x\n",
                   "stderr": "", "truncated": False, "timed_out": False}
        with patch.object(q, "_git_capture", return_value=capture) as fake:
            result = self._run(action="blame", path="a.txt",
                               lines="120,150", ref="feature")
        self.assertTrue(result["success"], msg=result.get("error"))
        argv = fake.call_args[0][0]
        self.assertIn("blame", argv)
        # textconv is opt-out, not opt-in: blame runs it unless told otherwise.
        self.assertIn("--no-textconv", argv)
        self.assertIn("-L 120,150", argv)
        self.assertIn("--abbrev=8", argv)
        self.assertIn("--date=short", argv)
        self.assertIn("feature", argv)
        self.assertLess(argv.index("feature"), argv.index("--"))  # rev before pathspec
        self.assertEqual(argv[argv.index("--") + 1], "a.txt")
        self.assertFalse([p for p in argv if p.startswith("-C")])

    def test_blame_line_range_validation(self):
        cases = {
            "10,5": "reversed",
            "0,5": "start at 1",
            "abc": "numeric",
            "1,2,3": "numeric",
            "1,5000": "too wide",
        }
        for lines, needle in cases.items():
            result = self._run(action="blame", path="a.txt", lines=lines)
            self.assertFalse(result["success"], msg=lines)
            self.assertIn(needle, result["error"], msg=lines)
        # A single line is a valid shortcut for start==end.
        capture = {"returncode": 0, "stdout": "x\n", "stderr": "",
                   "truncated": False, "timed_out": False}
        with patch.object(q, "_git_capture", return_value=capture) as fake:
            self._run(action="blame", path="a.txt", lines="42")
        self.assertIn("-L 42,42", fake.call_args[0][0])

    def test_blame_ref_cannot_be_an_option(self):
        for ref in ("--exec=id", "-c", "HEAD; id"):
            result = self._run(action="blame", path="a.txt", lines="1,5", ref=ref)
            self.assertFalse(result["success"], msg=ref)
            self.assertIn("Unsupported ref", result["error"])

    @unittest.skipUnless(HAS_GIT, "git not installed")
    def test_blame_live_reports_commit_and_uncommitted_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            _git(tmp, "init", "-q", "-b", "main", ".")
            _git(tmp, "config", "user.email", "t@example.com")
            _git(tmp, "config", "user.name", "T")
            with open(os.path.join(tmp, "a.txt"), "w") as f:
                f.write("one\ntwo\n")
            _git(tmp, "add", "a.txt")
            _git(tmp, "commit", "-qm", "base commit")
            with open(os.path.join(tmp, "a.txt"), "w") as f:
                f.write("one\ntwo\nthree\n")
            with open(os.path.join(tmp, "new.txt"), "w") as f:
                f.write("untracked\n")
            with patch.object(q.os, "getcwd", return_value=tmp):
                blamed = self._run(action="blame", path="a.txt", lines="1,3")
                untracked = self._run(action="blame", path="new.txt", lines="1,1")
        self.assertTrue(blamed["success"], msg=blamed.get("error"))
        self.assertIn("Not Committed Yet", blamed["output"])
        self.assertEqual(len(blamed["output"].splitlines()), 4)  # header + one row per line
        self.assertFalse(untracked["success"])
        self.assertIn("tracked", untracked["error"])

    @unittest.skipUnless(HAS_GIT, "git not installed")
    def test_blame_another_ref_and_line_numbers_match_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            _git(tmp, "init", "-q", "-b", "main", ".")
            _git(tmp, "config", "user.email", "t@example.com")
            _git(tmp, "config", "user.name", "T")
            with open(os.path.join(tmp, "a.txt"), "w") as f:
                f.write("one\ntwo\n")
            _git(tmp, "add", "a.txt")
            _git(tmp, "commit", "-qm", "base commit")
            _git(tmp, "checkout", "-q", "-b", "feature")
            with open(os.path.join(tmp, "a.txt"), "w") as f:
                f.write("one\ntwo from feature\n")
            _git(tmp, "commit", "-qam", "feature edit")
            _git(tmp, "checkout", "-q", "main")
            with patch.object(q.os, "getcwd", return_value=tmp):
                at_feature = self._run(action="blame", path="a.txt",
                                      lines="2,2", ref="feature")
                at_main = self._run(action="blame", path="a.txt", lines="2,2")
        self.assertTrue(at_feature["success"], msg=at_feature.get("error"))
        self.assertIn("two from feature", at_feature["output"])
        self.assertIn(" 2) two from feature", at_feature["output"])  # 1-based, file order
        # The same line on main belongs to the base commit, not the feature one.
        self.assertTrue(at_main["success"], msg=at_main.get("error"))
        self.assertNotIn("two from feature", at_main["output"])
        self.assertNotEqual(at_main["output"], at_feature["output"])

    # ---- log follow / name_status: file history through renames -----------

    def test_follow_requires_a_path(self):
        result = self._run(action="log", follow=True)
        self.assertFalse(result["success"])
        self.assertIn("path", result["error"])

    def test_log_history_flags_argv(self):
        capture = {"returncode": 0, "stdout": "", "stderr": "",
                   "truncated": False, "timed_out": False}
        with patch.object(q, "_git_capture", return_value=capture) as fake:
            self._run(action="log", path="a.txt", follow=True, name_status=True)
            history_argv = fake.call_args[0][0]
            self._run(action="log")
            plain_argv = fake.call_args[0][0]
        self.assertIn("--follow", history_argv)
        self.assertIn("--name-status", history_argv)
        self.assertIn("--date=short", history_argv)
        self.assertEqual(history_argv[history_argv.index("--") + 1], "a.txt")
        # A plain log keeps its compact format and gains no archaeology flags.
        self.assertNotIn("--follow", plain_argv)
        self.assertNotIn("--name-status", plain_argv)
        self.assertNotIn("--date=short", plain_argv)

    @unittest.skipUnless(HAS_GIT, "git not installed")
    def test_follow_traverses_a_rename(self):
        with tempfile.TemporaryDirectory() as tmp:
            _git(tmp, "init", "-q", "-b", "main", ".")
            _git(tmp, "config", "user.email", "t@example.com")
            _git(tmp, "config", "user.name", "T")
            with open(os.path.join(tmp, "a.txt"), "w") as f:
                f.write("one\n")
            _git(tmp, "add", "a.txt")
            _git(tmp, "commit", "-qm", "pre-rename commit")
            _git(tmp, "mv", "a.txt", "b.txt")
            _git(tmp, "commit", "-qm", "rename commit")
            with open(os.path.join(tmp, "b.txt"), "w") as f:
                f.write("one\ntwo\n")
            _git(tmp, "commit", "-qam", "post-rename edit")
            with patch.object(q.os, "getcwd", return_value=tmp):
                plain = self._run(action="log", path="b.txt")
                followed = self._run(action="log", path="b.txt", follow=True,
                                     name_status=True)
        self.assertTrue(plain["success"], msg=plain.get("error"))
        self.assertNotIn("pre-rename commit", plain["output"])
        self.assertTrue(followed["success"], msg=followed.get("error"))
        self.assertIn("pre-rename commit", followed["output"])
        self.assertIn("M\tb.txt", followed["output"])



    def test_limit_is_clamped(self):
        capture = {"returncode": 0, "stdout": "", "stderr": "",
                   "truncated": False, "timed_out": False}
        with patch.object(q, "_git_capture", return_value=capture) as fake:
            self._run(action="log", limit=100000)
            self.assertIn("-n100", fake.call_args[0][0])
            self._run(action="log", limit=-4)
            self.assertIn("-n1", fake.call_args[0][0])
        bad = self._run(action="log", limit="abc")
        self.assertFalse(bad["success"])
        self.assertIn("integer", bad["error"])

    def test_not_a_repo_reports_error(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(q.os, "getcwd", return_value=tmp):
            result = self._run(action="status")
        self.assertFalse(result["success"])
        self.assertTrue(result["error"])

    def test_missing_git_reports_error(self):
        with patch.object(q.subprocess, "Popen", side_effect=FileNotFoundError()):
            result = self._run(action="status")
        self.assertFalse(result["success"])
        self.assertIn("git is not installed", result["error"])

    def test_timeout_reports_error(self):
        capture = {"returncode": -1, "stdout": "partial", "stderr": "",
                   "truncated": False, "timed_out": True}
        with patch.object(q, "_git_capture", return_value=capture):
            result = self._run(action="diff_head")
        self.assertFalse(result["success"])
        self.assertIn("timed out", result["error"])

    def test_capture_is_bounded(self):
        """stdout is read under a ceiling, not buffered whole and truncated later."""
        big = "+" + ("x" * 50000) + "\n"
        fake = _FakeGitProc(big)
        with patch.object(q, "GIT_INSPECT_MAX_OUTPUT_CHARS", 100), \
             patch.object(q.subprocess, "Popen", return_value=fake), \
             patch.object(q, "CLUSTER_SPILL_THRESHOLD", 10 ** 9):
            result = self._run(action="diff")
        self.assertTrue(result["success"], msg=result.get("error"))
        self.assertIn("truncated at 100 chars", result["output"])
        self.assertEqual(result["output"].count("x"), 99)  # 99 kept + the "+"

    @unittest.skipUnless(HAS_GIT, "git not installed")
    def test_inherited_git_dir_is_ignored(self):
        """A GIT_DIR/GIT_WORK_TREE pointing elsewhere must not win over the CWD."""
        with tempfile.TemporaryDirectory() as proj, tempfile.TemporaryDirectory() as other:
            for path, name in ((proj, "IN-PROJECT"), (other, "ELSEWHERE")):
                _git(path, "init", "-q", ".")
                _git(path, "config", "user.email", "t@example.com")
                _git(path, "config", "user.name", "T")
                with open(os.path.join(path, "a.txt"), "w") as f:
                    f.write("one\n")
                _git(path, "add", "a.txt")
                _git(path, "commit", "-qm", "init")
                with open(os.path.join(path, "a.txt"), "w") as f:
                    f.write(name + "\n")
            with patch.dict(os.environ, {"GIT_DIR": os.path.join(other, ".git"),
                                         "GIT_WORK_TREE": other}, clear=False), \
                 patch.object(q.os, "getcwd", return_value=proj):
                result = self._run(action="diff_head")
        self.assertTrue(result["success"], msg=result.get("error"))
        self.assertIn("IN-PROJECT", result["output"])
        self.assertNotIn("ELSEWHERE", result["output"])

    def test_large_output_spills_to_file(self):
        big = "@@ -1 +1 @@\n" + ("+CHANGED-LINE\n" * 5000)
        capture = {"returncode": 0, "stdout": big, "stderr": "",
                   "truncated": False, "timed_out": False}
        with tempfile.TemporaryDirectory() as tmp:
            spill_dir = os.path.join(tmp, "spill")
            spill_file = os.path.join(spill_dir, "git_result_diff_head.txt")
            with patch.object(q, "CLUSTER_SPILL_DIR", spill_dir), \
                 patch.object(q, "_git_capture", return_value=capture):
                result = self._run(action="diff_head")
            self.assertTrue(result["success"], msg=result.get("error"))
            self.assertIn(spill_file, result["output"])
            self.assertIn("chars", result["output"])
            self.assertNotIn("CHANGED-LINE", result["output"])
            with open(spill_file) as f:
                self.assertEqual(f.read(), big)
            # The spill dir is ACL-allowed so read_file can page it in plan mode.
            rd = self.reg.execute("read_file", {"file_path": spill_file})
            self.assertTrue(rd["success"], msg=rd.get("error"))

    def test_spill_failure_returns_the_output_unchanged(self):
        """A failed spill must lose nothing (regression: `return text` NameError).

        `_spill_git_result` promises "On any failure the output is returned
        unchanged", but the fallback named an undefined `text`: a full disk (or
        a quota/permission error) turned a usable diff into `name 'text' is not
        defined`, which `ToolRegistry.execute` reported to the model as a
        failed tool. Found by `ruff --select F821`.
        """
        big = "@@ -1 +1 @@\n" + ("+CHANGED-LINE\n" * 5000)
        capture = {"returncode": 0, "stdout": big, "stderr": "",
                   "truncated": False, "timed_out": False}
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(q, "CLUSTER_SPILL_DIR", os.path.join(tmp, "spill")), \
                 patch.object(q, "_git_capture", return_value=capture), \
                 patch.object(q, "_open_private",
                              side_effect=OSError(28, "No space left on device")):
                result = self._run(action="diff_head")
        self.assertTrue(result["success"], msg=result.get("error"))
        self.assertIn("CHANGED-LINE", result["output"])

    @unittest.skipUnless(HAS_GIT, "git not installed")
    def test_status_and_log_on_real_repo(self):
        with tempfile.TemporaryDirectory() as tmp:
            _git(tmp, "init", "-q", ".")
            _git(tmp, "config", "user.email", "t@example.com")
            _git(tmp, "config", "user.name", "T")
            with open(os.path.join(tmp, "a.txt"), "w") as f:
                f.write("one\n")
            _git(tmp, "add", "a.txt")
            _git(tmp, "commit", "-qm", "first commit")
            with open(os.path.join(tmp, "a.txt"), "w") as f:
                f.write("two\n")
            with patch.object(q.os, "getcwd", return_value=tmp):
                status = self._run(action="status")
                log = self._run(action="log")
                head = self._run(action="diff_head")
                scoped = self._run(action="diff_head", path="a.txt")
                missing = self._run(action="diff_head", path="nope.txt")
        self.assertTrue(status["success"], msg=status.get("error"))
        self.assertIn("a.txt", status["output"])
        self.assertTrue(log["success"], msg=log.get("error"))
        self.assertIn("first commit", log["output"])
        self.assertTrue(head["success"], msg=head.get("error"))
        self.assertIn("+two", head["output"])
        self.assertTrue(scoped["success"], msg=scoped.get("error"))
        self.assertIn("+two", scoped["output"])
        self.assertTrue(missing["success"])  # empty pathspec match is not an error

    @unittest.skipUnless(HAS_GIT, "git not installed")
    def test_log_ref_lists_another_branch(self):
        """`log` with a ref walks that revision, and `path` narrows it."""
        with tempfile.TemporaryDirectory() as tmp:
            _git(tmp, "init", "-q", "-b", "main", ".")
            _git(tmp, "config", "user.email", "t@example.com")
            _git(tmp, "config", "user.name", "T")
            with open(os.path.join(tmp, "a.txt"), "w") as f:
                f.write("one\n")
            _git(tmp, "add", "a.txt")
            _git(tmp, "commit", "-qm", "base commit")
            _git(tmp, "checkout", "-q", "-b", "feature")
            with open(os.path.join(tmp, "b.txt"), "w") as f:
                f.write("two\n")
            _git(tmp, "add", "b.txt")
            _git(tmp, "commit", "-qm", "feature commit")
            _git(tmp, "checkout", "-q", "main")
            with patch.object(q.os, "getcwd", return_value=tmp):
                head_log = self._run(action="log")
                feature_log = self._run(action="log", ref="feature")
                scoped = self._run(action="log", ref="feature", path="b.txt")
                unscoped = self._run(action="log", ref="feature", path="a.txt")
        self.assertTrue(head_log["success"], msg=head_log.get("error"))
        self.assertNotIn("feature commit", head_log["output"])
        self.assertTrue(feature_log["success"], msg=feature_log.get("error"))
        self.assertIn("feature commit", feature_log["output"])
        self.assertIn("base commit", feature_log["output"])
        self.assertTrue(scoped["success"], msg=scoped.get("error"))
        self.assertIn("feature commit", scoped["output"])
        # a.txt was not touched by the feature commit, so the pathspec excludes it
        self.assertTrue(unscoped["success"], msg=unscoped.get("error"))
        self.assertNotIn("feature commit", unscoped["output"])

    @unittest.skipUnless(HAS_GIT, "git not installed")
    def test_clean_tree_is_not_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            _git(tmp, "init", "-q", ".")
            _git(tmp, "config", "user.email", "t@example.com")
            _git(tmp, "config", "user.name", "T")
            with open(os.path.join(tmp, "a.txt"), "w") as f:
                f.write("one\n")
            _git(tmp, "add", "a.txt")
            _git(tmp, "commit", "-qm", "initial")
            with patch.object(q.os, "getcwd", return_value=tmp):
                diff = self._run(action="diff")
        self.assertTrue(diff["success"], msg=diff.get("error"))
        self.assertIn("(no output)", diff["output"])


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

    def test_persona_leads_the_blocks(self):
        """The chat persona is the voice; the agentic blocks are the mechanics."""
        prompt = q.get_agentic_prompt("test-model", persona="PERSONA TEXT")
        self.assertTrue(prompt.startswith("PERSONA TEXT"))
        self.assertLess(prompt.index("PERSONA TEXT"), prompt.index("capable AI agent"))
        self.assertIn("## General rules", prompt)

    def test_persona_is_optional_and_blank_is_ignored(self):
        plain = q.get_agentic_prompt("test-model")
        self.assertTrue(plain.startswith("You are a capable AI agent"))
        self.assertEqual(q.get_agentic_prompt("test-model", persona="   "), plain)
        self.assertEqual(q.get_agentic_prompt("test-model", persona=""), plain)

    def test_get_agentic_prompt_strict_format(self):
        prompt = q.get_agentic_prompt("test-model")
        self.assertIn("Output ONLY the JSON tool call", prompt)
        self.assertIn("ReAct protocol", prompt)

    def test_get_agentic_prompt_native_format_for_openai_models(self):
        """Models using the native tools API must NOT get the bare-JSON format
        block (which caused Qwen3.6 to emit hybrid/XML tool calls)."""
        for model in ("qwen3.6-35b-a3b", "qwen3:8b", "gpt-oss-20b", "gemini-2.5"):
            prompt = q.get_agentic_prompt(model)
            self.assertNotIn("Output ONLY the JSON tool call", prompt,
                             f"{model} must not receive the strict JSON format")
            self.assertIn("function-calling interface", prompt,
                          f"{model} should receive the native-tools format")
            self.assertNotIn("JSON tool call", prompt)

    def test_get_agentic_prompt_soft_format(self):
        prompt = q.get_agentic_prompt("nemotron-cascade")
        self.assertIn("code block", prompt)
        self.assertNotIn("ONLY the JSON tool call", prompt)

    def test_get_agentic_prompt_includes_rules(self):
        prompt = q.get_agentic_prompt("test-model")
        self.assertIn("Mirror the user's language", prompt)
        self.assertIn("Be precise with file paths", prompt)

    def test_get_agentic_prompt_plan_mode_includes_plan_block(self):
        prompt = q.get_agentic_prompt("test-model", plan_mode=True)
        self.assertIn("Read-only planning mode", prompt)
        self.assertIn("READ-ONLY", prompt)
        self.assertIn("Do NOT execute the plan yourself", prompt)
        self.assertIn("Mirror the user's language", prompt)  # rules block still present

    def test_get_agentic_prompt_plan_mode_absent_by_default(self):
        prompt = q.get_agentic_prompt("test-model")
        self.assertNotIn("Read-only planning mode", prompt)

    def test_get_agentic_prompt_plan_mode_for_native_models(self):
        """Native-tools models must keep the function-calling format AND get the plan block."""
        prompt = q.get_agentic_prompt("qwen3:8b", plan_mode=True)
        self.assertIn("function-calling interface", prompt)
        self.assertIn("Read-only planning mode", prompt)


class TestAgenticStatusDisplay(unittest.TestCase):
    """`/agentic` status table: every setting listed, value brackets aligned."""

    SETTINGS = (
        "agentic", "preset", "plan", "auto", "sandbox", "verbose", "thinking",
        "trace", "log", "lazytool", "iterations", "timeout",
        "heartbeattokens", "compactthreshold", "logretention",
    )

    def _render(self) -> str:
        loop = object.__new__(q.ChatLoop)
        loop.ctx = q.CommandContext()
        loop.executor = q.Executor(mode="host")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            loop._print_agentic_status()
        return err.getvalue()

    def test_every_setting_is_listed(self):
        out = self._render()
        for name in self.SETTINGS:
            self.assertIn(name, out)

    def test_value_brackets_align(self):
        """A name longer than the column used to push its `[value]` out of line."""
        out = self._render()
        rows = [ln for ln in out.splitlines()
                if re.match(r"^ {2}[> ] \S+ +\[[^\]]*\] ", ln)]
        self.assertEqual(len(rows), len(self.SETTINGS), out)
        columns = {ln.index("[") for ln in rows}
        self.assertEqual(len(columns), 1, "value brackets not aligned:\n" + "\n".join(rows))


class TestAgenticPlanCommand(unittest.TestCase):
    """Test the /agentic plan command handler (read-only planning mode)."""

    def setUp(self):
        ctx = q.CommandContext()
        ctx.plan_mode = False
        ctx.plan_run = False
        ctx.agentic_mode = False
        ctx.preset = "work"   # the stance `/agentic off` must restore

    def tearDown(self):
        ctx = q.CommandContext()
        ctx.plan_mode = False
        ctx.plan_run = False
        ctx.agentic_mode = False
        ctx.preset = "work"
        ctx.executor_mode = "host"
        ctx.auto_confirm = False
        ctx.sandbox_engine_override = None

    def _loop(self):
        loop = object.__new__(q.ChatLoop)
        loop.ctx = q.CommandContext()
        return loop

    def _isolate(self, loop):
        """Reset the singleton state earlier tests in this class leave behind.

        CommandContext is a process-wide singleton, so `_saved_system_prompt`
        and the toggle flags leak between tests; a prompt-restore assertion
        needs a deterministic starting point.
        """
        loop.ctx.system_prompt = "BASE PROMPT"
        loop.ctx._saved_system_prompt = None
        loop.ctx._plan_prev_prompt = None
        loop.ctx.agentic_mode = False
        loop.ctx.plan_mode = False
        loop.ctx.plan_run = False

    def test_plan_on_sets_plan_mode_and_agentic(self):
        loop = self._loop()
        self.assertFalse(loop._agentic_toggle_plan(['/agentic', 'plan']))
        self.assertTrue(loop.ctx.plan_mode)
        self.assertTrue(loop.ctx.agentic_mode)
        self.assertFalse(loop.ctx.plan_run)
        self.assertIn("Read-only planning mode", loop.ctx.system_prompt)

    def test_plan_strict_prompt_is_confined(self):
        loop = self._loop()
        loop._agentic_toggle_plan(['/agentic', 'plan'])
        self.assertFalse(loop.ctx.plan_run)
        self.assertIn("confined", loop.ctx.system_prompt.lower())
        self.assertIn("no shell access", loop.ctx.system_prompt.lower())

    def test_plan_run_sets_plan_run_flag(self):
        loop = self._loop()
        self.assertFalse(loop._agentic_toggle_plan(['/agentic', 'plan', 'run']))
        self.assertTrue(loop.ctx.plan_mode)
        self.assertTrue(loop.ctx.plan_run)
        self.assertTrue(loop.ctx.agentic_mode)
        self.assertIn("run_command", loop.ctx.system_prompt)
        self.assertIn("confirm", loop.ctx.system_prompt.lower())
        loop._agentic_toggle_plan(['/agentic', 'plan', 'off'])
        self.assertFalse(loop.ctx.plan_mode)
        self.assertFalse(loop.ctx.plan_run)

    def test_plan_switch_tier_while_active(self):
        loop = self._loop()
        loop._agentic_toggle_plan(['/agentic', 'plan'])
        self.assertFalse(loop.ctx.plan_run)
        loop._agentic_toggle_plan(['/agentic', 'plan', 'run'])
        self.assertTrue(loop.ctx.plan_mode)
        self.assertTrue(loop.ctx.plan_run)
        loop._agentic_toggle_plan(['/agentic', 'plan', 'on'])
        self.assertTrue(loop.ctx.plan_mode)
        self.assertFalse(loop.ctx.plan_run)

    def test_plan_off_restores_previous_agentic_mode(self):
        loop = self._loop()
        loop.ctx.agentic_mode = True
        loop._agentic_toggle_plan(['/agentic', 'plan'])
        loop._agentic_toggle_plan(['/agentic', 'plan', 'off'])
        self.assertFalse(loop.ctx.plan_mode)
        self.assertTrue(loop.ctx.agentic_mode)

    def test_plan_off_from_normal_restores_off(self):
        loop = self._loop()
        loop.ctx.agentic_mode = False
        loop._agentic_toggle_plan(['/agentic', 'plan'])
        loop._agentic_toggle_plan(['/agentic', 'plan', 'off'])
        self.assertFalse(loop.ctx.plan_mode)
        self.assertFalse(loop.ctx.agentic_mode)

    def test_plan_off_when_not_active(self):
        loop = self._loop()
        self.assertFalse(loop._agentic_toggle_plan(['/agentic', 'plan', 'off']))
        self.assertFalse(loop.ctx.plan_mode)

    def test_run_handle_agentic_dispatches_plan(self):
        loop = self._loop()
        self.assertFalse(loop.run_handle_agentic('/agentic plan'))
        self.assertTrue(loop.ctx.plan_mode)
        loop.run_handle_agentic('/agentic plan off')
        self.assertFalse(loop.ctx.plan_mode)

    def test_agentic_off_clears_plan_flags(self):
        """Master `/agentic off` must not leave a stale plan flag behind."""
        loop = self._loop()
        loop.ctx.agentic_mode = True
        loop._agentic_toggle_plan(['/agentic', 'plan', 'run'])
        self.assertTrue(loop.ctx.plan_mode)
        self.assertTrue(loop.ctx.plan_run)
        loop._agentic_toggle_on_off(False)
        self.assertFalse(loop.ctx.agentic_mode)
        self.assertFalse(loop.ctx.plan_mode)
        self.assertFalse(loop.ctx.plan_run)
        self.assertEqual(loop.ctx.preset, "work")
        # Re-enabling agentic must NOT silently restore plan restrictions.
        loop._agentic_toggle_on_off(True)
        self.assertFalse(loop.ctx.plan_mode)
        self.assertFalse(loop.ctx.plan_run)

    def test_full_after_plan_clears_plan_restrictions(self):
        """`/agentic full` after plan must drop the read-only surface."""
        loop = self._loop()
        loop._agentic_toggle_plan(['/agentic', 'plan'])
        self.assertTrue(loop.ctx.plan_mode)
        loop._agentic_set_full()
        self.assertFalse(loop.ctx.plan_mode)
        self.assertFalse(loop.ctx.plan_run)
        self.assertTrue(loop.ctx.agentic_mode)
        self.assertTrue(loop.ctx.auto_confirm)
        self.assertNotIn("Read-only planning mode", loop.ctx.system_prompt)

    def test_full_after_plan_run_clears_plan_restrictions(self):
        loop = self._loop()
        loop._agentic_toggle_plan(['/agentic', 'plan', 'run'])
        self.assertTrue(loop.ctx.plan_run)
        loop.run_handle_agentic('/agentic full')
        self.assertFalse(loop.ctx.plan_mode)
        self.assertFalse(loop.ctx.plan_run)
        self.assertTrue(loop.ctx.agentic_mode)
        self.assertNotIn("Read-only planning mode", loop.ctx.system_prompt)

    def test_on_after_plan_clears_plan_mode(self):
        """`/agentic on` while in plan mode also leaves the read-only surface."""
        loop = self._loop()
        loop._agentic_toggle_plan(['/agentic', 'plan'])
        loop._agentic_toggle_on_off(True)
        self.assertFalse(loop.ctx.plan_mode)
        self.assertFalse(loop.ctx.plan_run)
        self.assertTrue(loop.ctx.agentic_mode)
        self.assertNotIn("Read-only planning mode", loop.ctx.system_prompt)

    def test_full_then_off_after_plan_restores_pre_agentic_prompt(self):
        """on -> plan -> full -> off must restore the pre-agentic (normal) prompt,
        not the agentic or planning prompt. A known baseline prompt is set first:
        CommandContext is a singleton, so leftover state from earlier tests in
        this class must not leak into the assertion."""
        loop = self._loop()
        self._isolate(loop)
        loop._agentic_toggle_on_off(True)
        loop._agentic_toggle_plan(['/agentic', 'plan'])
        loop._agentic_set_full()
        self.assertNotIn("Read-only planning mode", loop.ctx.system_prompt)
        loop._agentic_toggle_on_off(False)
        self.assertFalse(loop.ctx.agentic_mode)
        self.assertEqual(loop.ctx.system_prompt, "BASE PROMPT")

    def test_full_then_off_after_plan_from_scratch_restores_pre_plan_prompt(self):
        """plan -> full -> off (agentic never on before) must restore the
        pre-plan prompt, not the planning prompt."""
        loop = self._loop()
        self._isolate(loop)
        loop._agentic_toggle_plan(['/agentic', 'plan'])
        loop._agentic_set_full()
        loop._agentic_toggle_on_off(False)
        self.assertFalse(loop.ctx.agentic_mode)
        self.assertEqual(loop.ctx.system_prompt, "BASE PROMPT")

    def test_plan_off_from_agentic_restores_agentic_prompt(self):
        """on -> plan -> plan off returns to the agentic prompt with agentic ON."""
        loop = self._loop()
        self._isolate(loop)
        loop._agentic_toggle_on_off(True)
        agentic_prompt = loop.ctx.system_prompt
        self.assertNotEqual(agentic_prompt, "BASE PROMPT")
        loop._agentic_toggle_plan(['/agentic', 'plan'])
        loop._agentic_toggle_plan(['/agentic', 'plan', 'off'])
        self.assertTrue(loop.ctx.agentic_mode)
        self.assertEqual(loop.ctx.system_prompt, agentic_prompt)


class TestPersonaInAgenticPrompt(unittest.TestCase):
    """The chat persona survives `/agentic ...` (composition, not replacement).

    `/agentic on` used to swap `ctx.system_prompt` for the protocol blocks, and
    the three prompt builders called `get_agentic_prompt()` directly, so a
    `-P teacher` session silently lost its persona the moment agentic mode was
    enabled — including in the ReAct step and the final stream.
    """

    PERSONA = "You are a patient teacher. Explain with analogies and examples."

    def setUp(self):
        self.ctx = q.CommandContext()
        self.ctx.auto_confirm = True
        self.ctx.base_url = "http://localhost:9999"
        self.ctx.backend = "opencodezen"   # offline: no tokenize / whole-request counter
        self.ctx.model = "test-model"      # inline delivery, no native tools array
        self.ctx.system_prompt = self.PERSONA
        self.ctx._saved_system_prompt = None
        self.ctx._plan_prev_prompt = None
        self.ctx.agentic_mode = False
        self.ctx.plan_mode = False
        self.ctx.plan_run = False
        self.loop = q.ChatLoop(self.ctx)

    def tearDown(self):
        self.ctx.agentic_mode = False
        self.ctx.plan_mode = False
        self.ctx.plan_run = False
        self.ctx._saved_system_prompt = None
        self.ctx._plan_prev_prompt = None
        self.ctx.system_prompt = q.DEFAULT_SYSTEM_PROMPT
        self.ctx.preset = "work"
        self.ctx.executor_mode = "host"
        self.ctx.auto_confirm = False
        self.ctx.sandbox_engine_override = None

    def test_persona_leads_and_the_protocol_follows(self):
        self.loop._agentic_toggle_on_off(True)
        prompt = self.ctx.system_prompt
        self.assertTrue(prompt.startswith(self.PERSONA))
        self.assertIn("capable AI agent", prompt)
        self.assertIn("## General rules", prompt)

    def test_wire_prompt_carries_the_persona(self):
        """The step/final-stream builders must agree with the toggled prompt."""
        self.loop._agentic_toggle_on_off(True)
        self.loop.messages = [{'role': 'system', 'content': self.ctx.system_prompt}]
        messages, logger, _tools, _send = self.loop._init_agentic_query("ask")
        try:
            wire = messages[0]['content']
            self.assertTrue(wire.startswith(self.PERSONA))
            self.assertEqual(wire, self.loop._agentic_prompt_and_tools()[0])
            self.assertIn("## Available tools", wire)
        finally:
            if logger is not None:
                logger.close()

    def test_plan_and_full_keep_the_persona_without_duplicating_it(self):
        self.loop._agentic_toggle_plan(['/agentic', 'plan'])
        self.assertTrue(self.ctx.system_prompt.startswith(self.PERSONA))
        self.assertEqual(self.ctx.system_prompt.count(self.PERSONA), 1)
        self.loop._agentic_toggle_plan(['/agentic', 'plan', 'run'])
        self.assertEqual(self.ctx.system_prompt.count(self.PERSONA), 1)
        self.loop._agentic_toggle_plan(['/agentic', 'plan', 'off'])
        self.assertTrue(self.ctx.system_prompt.startswith(self.PERSONA))
        self.assertEqual(self.ctx.system_prompt.count(self.PERSONA), 1)
        self.loop._agentic_set_full()
        self.assertTrue(self.ctx.system_prompt.startswith(self.PERSONA))
        self.assertEqual(self.ctx.system_prompt.count(self.PERSONA), 1)

    def test_off_restores_the_persona_alone(self):
        self.loop._agentic_toggle_on_off(True)
        self.loop._agentic_toggle_on_off(False)
        self.assertEqual(self.ctx.system_prompt, self.PERSONA)

    def test_without_a_saved_persona_the_prompt_is_unchanged(self):
        self.ctx._saved_system_prompt = None
        self.ctx.system_prompt = ""
        self.loop._agentic_toggle_on_off(True)
        self.assertTrue(self.ctx.system_prompt.startswith("You are a capable AI agent"))


class TestAppendToolMessages(unittest.TestCase):
    """Assistant-turn shape after a tool call.

    Native-tools backends must get an empty assistant `content` plus the real
    `tool_calls`; synthesizing a `{"tool": ...}` JSON as content makes Qwen-style
    templates render hybrid JSON+XML (the bug the native format block prevents).
    Inline backends still need the call echoed as text.
    """

    def setUp(self):
        # Appending now stamps token counts via the backend tokenizer; these
        # tests are about message shape, so keep them offline.
        self._tokenizer = patch.object(q.CommandContext, 'tokenize', return_value=5)
        self._tokenizer.start()

    def tearDown(self):
        self._tokenizer.stop()

    def _loop(self):
        q.CommandContext._instance = None
        q.CommandContext._initialized = False
        ctx = q.CommandContext()
        ctx.backend = 'strata'
        ctx.model = 'qwen3.8-test'
        return q.ChatLoop(ctx)

    def _api_calls(self):
        return [{"id": "call_1", "type": "function",
                 "function": {"name": "list_directory", "arguments": '{"path": "."}'}}]

    def test_native_mode_keeps_content_empty(self):
        loop = self._loop()
        messages = []
        tool_calls = [{"tool": "list_directory", "arguments": {"path": "."}}]
        loop._append_tool_messages(messages, "", self._api_calls(), tool_calls,
                                   ["obs"], ["raw"], send_tools_api=True)
        self.assertEqual(messages[0]["role"], "assistant")
        self.assertEqual(messages[0]["content"], "")
        self.assertIn("tool_calls", messages[0])
        self.assertEqual(messages[1]["role"], "tool")
        self.assertEqual(messages[1]["content"], "raw")

    def test_inline_mode_synthesizes_json_content(self):
        loop = self._loop()
        messages = []
        tool_calls = [{"tool": "list_directory", "arguments": {"path": "."}}]
        loop._append_tool_messages(messages, "", self._api_calls(), tool_calls,
                                   ["obs"], ["raw"], send_tools_api=False)
        self.assertEqual(messages[0]["content"], json.dumps(tool_calls[0]))
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("Tool result:", messages[1]["content"])

    def test_preamble_content_is_preserved(self):
        loop = self._loop()
        messages = []
        tool_calls = [{"tool": "list_directory", "arguments": {"path": "."}}]
        loop._append_tool_messages(messages, "Let me look.", self._api_calls(), tool_calls,
                                   ["obs"], ["raw"], send_tools_api=True)
        self.assertEqual(messages[0]["content"], "Let me look.")


class TestCallStepFinalizesFeedback(unittest.TestCase):
    """The per-step watchdog must stop on EVERY exit path.

    A transport error used to propagate from `call_with_timeout` straight past
    `call_step`'s finalize, leaving the StepFeedback daemon printing one dot per
    second for the rest of the session: the prompt returned but the terminal was
    spammed, which reads as "Ctrl-C does nothing".
    """

    def _runner(self):
        runner = object.__new__(q.AgenticRunner)
        runner.ctx = q.CommandContext()
        runner.ctx.model = "m"
        runner.ctx.context_size = None
        runner.query_handler = MagicMock()
        return runner

    def test_worker_exception_still_finalizes(self):
        runner = self._runner()
        finalized = []

        def boom(*args, **kwargs):
            raise RuntimeError("gateway 500")

        runner.call_with_timeout = boom
        with self.assertRaises(RuntimeError):
            runner.call_step([], 5, [], {}, lambda *a: None,
                             lambda: finalized.append(True),
                             {"event": None, "close": None}, [])
        self.assertEqual(finalized, [True])

    def test_keyboard_interrupt_finalizes_and_aborts(self):
        runner = self._runner()
        finalized = []
        aborted = []

        def boom(*args, **kwargs):
            raise KeyboardInterrupt

        runner.call_with_timeout = boom
        with patch.object(q, "_signal_abort",
                          side_effect=lambda cancel: aborted.append(True)):
            with self.assertRaises(KeyboardInterrupt):
                runner.call_step([], 5, [], {}, lambda *a: None,
                                 lambda: finalized.append(True),
                                 {"event": None, "close": None}, [])
        self.assertEqual(finalized, [True])
        self.assertEqual(aborted, [True])

    def test_success_finalizes_once(self):
        runner = self._runner()
        finalized = []

        runner.call_with_timeout = lambda *a, **k: {"ok": True}
        response, _start = runner.call_step(
            [], 5, [], {}, lambda *a: None, lambda: finalized.append(True),
            {"event": None, "close": None}, [])
        self.assertEqual(response, {"ok": True})
        self.assertEqual(finalized, [True])


if __name__ == "__main__":
    unittest.main()
