#!/usr/bin/env python3
"""Tests for agentic mode: ReAct loop, tool execution, and parse_tool_call."""

import io
import os
import re
import sys
import json
import tempfile
import threading
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ollamaquery2 as q

OLLAMA_HOST = os.environ.get('OLLAMA_HOST', 'http://127.0.0.1:11434')
LLAMACPP_HOST = os.environ.get('LLAMACPP_HOST', 'http://127.0.0.1:8080')
STRATA_HOST = os.environ.get('STRATA_HOST', 'http://127.0.0.1:8080')

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
            self.assertIn("run_command", block)
            self.assertIn("list_directory", block)
            self.assertIn("glob", block)
            self.assertIn("fetch_url", block)
            self.assertIn("diff", block)
            self.assertNotIn("write_file", block)
            self.assertNotIn("run_python", block)
            self.assertNotIn("apply_patch", block)
        finally:
            self.ctx.plan_mode = False

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

    def test_plan_mode_run_command_always_confirms(self):
        """run_command must prompt even with auto_confirm (bypass-immune)."""
        self.ctx.plan_mode = True
        self.ctx.auto_confirm = True
        try:
            with patch("builtins.input", return_value="n"):
                result = self.reg.execute("run_command", {"command": "echo plan-test"})
            self.assertFalse(result["success"])
            self.assertIn("Cancelled by user", result["error"])
        finally:
            self.ctx.plan_mode = False

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
        open("a.md", "w").write("Hello Storage nodes here\nno match\n")
        os.mkdir("sub")
        open(os.path.join("sub", "b.md"), "w").write("storageDeviceSets present\n")
        open("notes.txt", "w").write("STORAGE NODES in txt\n")
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
                    open(f"f{i}.txt", "w").write("SAME\n")
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
        """A harmless GET curl in plan mode is served by fetch_url, no Y/N."""
        self.ctx.plan_mode = True
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

    def test_run_command_curl_failure_redirect(self):
        """A redirected fetch still surfaces the failure instead of OK."""
        self.ctx.plan_mode = True
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

    def test_non_fetch_curl_still_gated(self):
        """curl that writes to disk is not intercepted; plan mode still asks."""
        self.ctx.plan_mode = True
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

    def test_fetch_url_surfaces_failure(self):
        """fetch_url must not report success while embedding a failure string."""
        with patch("ollamaquery2.fetch_and_convert_url",
                   return_value=("[Failed to fetch URL: HTTP Error 403: Forbidden]", "None")):
            result = self.reg.execute("fetch_url", {"url": "https://blocked.example"})
        self.assertFalse(result["success"])
        self.assertIn("403", result["error"])

    def test_fetch_url_curl_fallback(self):
        """fetch_and_convert_url falls back to curl when urllib is blocked."""
        with patch("ollamaquery2._request_with_retry",
                   side_effect=Exception("HTTP Error 403: Forbidden")):
            with patch("ollamaquery2._curl_fetch_text",
                       return_value="<html><body>curl rescued content</body></html>"):
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

        self.loop.query_handler.query_sync_stream = mock_sync
        self.loop.query_handler.query_stream = MagicMock(return_value="")

        self.loop.run_agentic_query("Keep using tools")

        self.assertTrue(hasattr(self.loop, 'messages'))
        # Should have terminated, no crash

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

        last = self.loop.messages[-1] if hasattr(self.loop, 'messages') else {"role": "system", "content": ""}
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
        """The injected 'Please provide a tool call or your final answer.' nudge
        is loop-internal steering and must NOT be merged into the persistent
        conversation history."""
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
            any(m.get("content") == "Please provide a tool call or your final answer."
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

        with patch.object(self.loop, '_call_with_timeout', side_effect=fake_timeout):
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
        self.loop._execute_single_tool = MagicMock(return_value={
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


class TestClusterQueryTool(unittest.TestCase):
    """kubernetes_cluster_query tool: verb whitelist, backend auto-detect, command build."""

    def setUp(self):
        self.ctx = q.CommandContext()
        self.ctx.auto_confirm = True
        self.reg = q.ToolRegistry(ctx=self.ctx)
        self.commands = []
        self.reg.executor.run = MagicMock(side_effect=self._fake_run)

    def _fake_run(self, command, timeout=120):
        self.commands.append(command)
        if "current-context" in command:
            return {"stdout": "api.cluster.example\n", "stderr": "", "returncode": 0}
        return {"stdout": "NAME READY\npod/a 1/1 Running\n", "stderr": "", "returncode": 0}

    def _run(self, **args):
        return self.reg.execute("kubernetes_cluster_query", args)

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
        self.assertTrue(self.commands[0].startswith("oc config current-context"))
        self.assertTrue(self.commands[1].startswith("oc get pods -n default -o json"))
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
                self.reg.executor.run = MagicMock(side_effect=fake_run)
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
                self.reg.executor.run = MagicMock(side_effect=fake_run)
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
        self.assertTrue(self.commands[1].startswith(
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
        self.assertTrue(self.commands[1].startswith(
            "oc get pv -o jsonpath={.items[*].metadata.name}"))

    def test_jsonpath_requires_fields(self):
        result = self._run(verb="get", resource="pv", output="jsonpath")
        self.assertFalse(result["success"])
        self.assertIn("fields is required", result["error"])

    def test_small_result_stays_inline(self):
        with patch.dict(os.environ, {"KUBECONFIG": "/tmp/kube"}, clear=False), \
             patch.object(q.shutil, "which", return_value="/usr/bin/oc"):
            result = self._run(verb="get", resource="pods")
        self.assertTrue(result["success"], msg=result.get("error"))
        self.assertNotIn("written to", result["output"])
        self.assertIn("pod/a 1/1 Running", result["output"])


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


class TestAgenticPlanCommand(unittest.TestCase):
    """Test the /agentic plan command handler (read-only planning mode)."""

    def setUp(self):
        ctx = q.CommandContext()
        ctx.plan_mode = False
        ctx.agentic_mode = False

    def tearDown(self):
        ctx = q.CommandContext()
        ctx.plan_mode = False
        ctx.agentic_mode = False

    def _loop(self):
        loop = object.__new__(q.ChatLoop)
        loop.ctx = q.CommandContext()
        return loop

    def test_plan_on_sets_plan_mode_and_agentic(self):
        loop = self._loop()
        self.assertFalse(loop._agentic_toggle_plan(['/agentic', 'plan']))
        self.assertTrue(loop.ctx.plan_mode)
        self.assertTrue(loop.ctx.agentic_mode)
        self.assertIn("Read-only planning mode", loop.ctx.system_prompt)

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


class TestAppendToolMessages(unittest.TestCase):
    """Assistant-turn shape after a tool call.

    Native-tools backends must get an empty assistant `content` plus the real
    `tool_calls`; synthesizing a `{"tool": ...}` JSON as content makes Qwen-style
    templates render hybrid JSON+XML (the bug the native format block prevents).
    Inline backends still need the call echoed as text.
    """

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


if __name__ == "__main__":
    unittest.main()
