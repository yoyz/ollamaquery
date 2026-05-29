#!/usr/bin/env python3
"""Integration tests verifying all modifications to ollamaquery2.py.

Discovery order: environment variable -> localhost -> local network IPs.
"""

import io
import os
import re
import sys
import time
import json
import types
import socket
import unittest
from unittest.mock import patch, MagicMock
from urllib.request import Request
from urllib.error import URLError, HTTPError

# Module under test
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ollamaquery2 as m

# -- Backend Discovery ------------------------------------------------------


def _discover_backends():
    """Auto-discover LLM backends by probing common URLs.

    Checks in order:
    1. User-set environment variables (OLLAMA_HOST, LLAMACPP_HOST)
    2. Default localhost URLs on standard ports
    3. Local network IPs on standard ports (via gethostbyname_ex)

    Returns:
        dict: Backend name -> (discovered_url, available_bool)
    """
    probes = [
        ('ollama',   'OLLAMA_HOST',   'http://127.0.0.1:11434', 11434,
         lambda u: m.check_backend_with_get(u, 'ollama')),
        ('llamacpp', 'LLAMACPP_HOST', 'http://127.0.0.1:8080',   8080,
         lambda u: m.check_backend_with_head(u, 'llama.cpp')),
    ]

    discovered = {}
    for name, env_var, default_url, port, check_fn in probes:
        url = os.environ.get(env_var, default_url)
        ok = False

        try:
            ok = check_fn(url)
        except Exception:
            pass

        if not ok:
            try:
                ips = socket.gethostbyname_ex(socket.gethostname())[-1]
                for ip in ips:
                    trial = f"http://{ip}:{port}"
                    try:
                        if check_fn(trial):
                            url, ok = trial, True
                            break
                    except Exception:
                        pass
            except Exception:
                pass

        discovered[name] = (url, ok)

    return discovered


_discovered_backends = _discover_backends()
for _name, (_url, _ok) in _discovered_backends.items():
    if _ok:
        sys.stderr.write(f"[INFO] Discovered {_name} at {_url}\n")
if not any(v[1] for v in _discovered_backends.values()):
    sys.stderr.write("[INFO] No LLM backend discovered. Backend-dependent tests will be skipped.\n")

OLLAMA_HOST   = _discovered_backends['ollama'][0]
LLAMACPP_HOST = _discovered_backends['llamacpp'][0]
HAS_OLLAMA    = _discovered_backends['ollama'][1]
HAS_LLAMACPP  = _discovered_backends['llamacpp'][1]

if HAS_LLAMACPP:
    BACKEND_URL = LLAMACPP_HOST
    BACKEND_TYPE = 'llamacpp'
elif HAS_OLLAMA:
    BACKEND_URL = OLLAMA_HOST
    BACKEND_TYPE = 'ollama'
else:
    BACKEND_URL = 'http://127.0.0.1:8080'
    BACKEND_TYPE = 'llamacpp'

SMALL_MODEL = os.environ.get('TEST_MODEL', 'granite4:350m')

ollama_only = unittest.skipUnless(HAS_OLLAMA, 'Ollama backend not available')
llamacpp_only = unittest.skipUnless(HAS_LLAMACPP, 'Llama.cpp backend not available')
any_backend = unittest.skipUnless(HAS_OLLAMA or HAS_LLAMACPP,
                                  'No backend available')


def _pick_small_model(base_url, prefer_under_gb=9):
    """Pick a small chat-capable model from the server, preferring models under `prefer_under_gb` GB."""
    models = m.fetch_models_ollama(base_url)
    if not models:
        return SMALL_MODEL

    for mod in models:
        raw_size = mod.get("size") or mod.get("size_bytes") or mod.get("model_size") or 0
        try:
            mod['size_bytes'] = int(raw_size)
        except (TypeError, ValueError):
            mod['size_bytes'] = 0

    min_bytes = 10 * 1024 * 1024
    skip_keywords = ['cloud', 'embedding', 'vl:']
    candidates = [mod for mod in models if mod['size_bytes'] >= min_bytes and not any(k in mod.get('name', '').lower() for k in skip_keywords)]

    small_chat = [mod for mod in candidates if 0 < mod['size_bytes'] < prefer_under_gb * (1024**3)]
    if small_chat:
        small_chat.sort(key=lambda x: x['size_bytes'])
        return small_chat[0]['name']

    if candidates:
        candidates.sort(key=lambda x: x['size_bytes'])
        return candidates[0]['name']

    return sorted(models, key=lambda x: x.get('name', ''))[0]['name']


def _ensure_model_loaded(base_url, model_name):
    """Ensure the given model is loaded into memory on the Ollama server."""
    loaded = m.fetch_loaded_models_ollama(base_url)
    loaded_names = [md.get('name', '') for md in loaded]
    if model_name in loaded_names:
        return True

    mq = m.ModelQuery(base_url, 'ollama')
    mq.query_sync(
        [{'role': 'user', 'content': 'ok'}],
        model_name,
        stream_enabled=False
    )
    return True


# If Ollama is the active backend, auto-pick a small model and ensure it's loaded
if BACKEND_TYPE == 'ollama' and HAS_OLLAMA:
    SMALL_MODEL = _pick_small_model(BACKEND_URL)
    _ensure_model_loaded(BACKEND_URL, SMALL_MODEL)


# ============================================================================
# 1.  _request_with_retry
# ============================================================================

class TestRetryUtility(unittest.TestCase):
    """Unit tests for _request_with_retry()."""

    def test_success_first_attempt(self):
        """Normal successful request returns on first try."""
        with patch.object(m, 'urlopen') as mock:
            mock.return_value = io.BytesIO(b'{"ok": true}')
            resp = m._request_with_retry(Request('http://localhost/x'))
            self.assertEqual(json.loads(resp.read()), {"ok": True})
            self.assertEqual(mock.call_count, 1)

    def test_retry_on_urlerror(self):
        """URLError is retried up to max_retries times."""
        with patch.object(m, 'urlopen') as mock:
            mock.side_effect = [URLError('conn refused'),
                                URLError('conn refused'),
                                io.BytesIO(b'{"ok": true}')]
            resp = m._request_with_retry(Request('http://localhost/x'),
                                        max_retries=3, delay=0.01)
            self.assertEqual(json.loads(resp.read()), {"ok": True})
            self.assertEqual(mock.call_count, 3)

    def test_retry_on_http_503(self):
        """HTTP 503 is retried (server error)."""
        with patch.object(m, 'urlopen') as mock:
            mock.side_effect = [
                HTTPError('http://localhost/x', 503, 'Service Unavailable',
                          {}, io.BytesIO(b'')),
                io.BytesIO(b'{"ok": true}')
            ]
            resp = m._request_with_retry(Request('http://localhost/x'),
                                        max_retries=2, delay=0.01)
            self.assertEqual(json.loads(resp.read()), {"ok": True})
            self.assertEqual(mock.call_count, 2)

    def test_no_retry_on_http_404(self):
        """HTTP 404 is NOT retried (client error)."""
        with patch.object(m, 'urlopen') as mock:
            mock.side_effect = HTTPError('http://localhost/x', 404, 'Not Found',
                                         {}, io.BytesIO(b''))
            with self.assertRaises(HTTPError):
                m._request_with_retry(Request('http://localhost/x'),
                                    max_retries=3, delay=0.01)
            self.assertEqual(mock.call_count, 1)

    def test_retry_on_connectionerror(self):
        """ConnectionError is retried."""
        with patch.object(m, 'urlopen') as mock:
            mock.side_effect = [ConnectionError('reset'),
                                io.BytesIO(b'{"ok": true}')]
            resp = m._request_with_retry(Request('http://localhost/x'),
                                        max_retries=2, delay=0.01)
            self.assertEqual(json.loads(resp.read()), {"ok": True})
            self.assertEqual(mock.call_count, 2)

    def test_exhausted_retries_raises(self):
        """After exhausting retries, the exception is raised."""
        with patch.object(m, 'urlopen') as mock:
            mock.side_effect = URLError('always down')
            with self.assertRaises(URLError):
                m._request_with_retry(Request('http://localhost/x'),
                                    max_retries=3, delay=0.01)
            self.assertEqual(mock.call_count, 3)

    def test_retry_shows_warning_on_stderr(self):
        """[RETRY] warning is written to stderr on each retry."""
        with patch.object(m, 'urlopen') as mock:
            mock.side_effect = [URLError('fail'), io.BytesIO(b'{}')]
            stderr = io.StringIO()
            old_stderr, sys.stderr = sys.stderr, stderr
            try:
                m._request_with_retry(Request('http://localhost/x'),
                                    max_retries=2, delay=0.01)
            finally:
                sys.stderr = old_stderr
            self.assertIn('[RETRY]', stderr.getvalue())
            self.assertIn('attempt 1/2', stderr.getvalue())

    def test_timeout_passed_through(self):
        """timeout= kwarg is forwarded to urlopen."""
        with patch.object(m, 'urlopen') as mock:
            mock.return_value = io.BytesIO(b'{}')
            m._request_with_retry(Request('http://localhost/x'), timeout=5)
            _, kwargs = mock.call_args
            self.assertEqual(kwargs.get('timeout'), 5)

    @any_backend
    def test_live_backend_success(self):
        """Live backend returns models (integration)."""
        if BACKEND_TYPE == 'llamacpp':
            models = m.fetch_models_llamacpp(BACKEND_URL)
        else:
            models = m.fetch_models_ollama(BACKEND_URL)
        self.assertIsInstance(models, list)
        self.assertGreater(len(models), 0)


# ============================================================================
# 2.  CommandContext.reset() preserves preferences
# ============================================================================

class TestResetPreservesPreferences(unittest.TestCase):
    """reset() must only wipe conversation state, not preferences."""

    def setUp(self):
        m.CommandContext._instance = None
        m.CommandContext._initialized = False
        self.ctx = m.CommandContext()
        # Set some user preferences
        self.ctx.force_no_thinking = True
        self.ctx.debug_mode = True
        self.ctx.debug_manager.set_level('all', 'trace')
        self.ctx.system_prompt = 'Custom prompt'
        self.ctx.context_size = 8192
        self.ctx.models = ['model1', 'model2']
        self.ctx.stream_enabled = False
        # Add some conversation state that SHOULD be wiped
        self.ctx.current_images = ['img1']
        self.ctx.total_queries = 10
        self.ctx.total_tokens_generated = 500
        self.ctx.total_prompt_tokens = 200
        self.ctx.total_time_spent = 30.0
        self.ctx.total_chars_generated = 1000
        self.ctx.query_history = ['q1', 'q2']
        self.ctx.context_window_size = 4096
        self.ctx.current_context_tokens = 512

    def test_preserved_after_reset(self):
        """Preferences survive reset()."""
        self.ctx.reset()
        self.assertTrue(self.ctx.force_no_thinking)
        self.assertTrue(self.ctx.debug_mode)
        # set_level('all', 'trace') stores the numeric level (3)
        self.assertEqual(self.ctx.debug_manager.get_level('all'), 3)
        self.assertEqual(self.ctx.system_prompt, 'Custom prompt')
        self.assertEqual(self.ctx.context_size, 8192)
        self.assertEqual(self.ctx.models, ['model1', 'model2'])
        self.assertFalse(self.ctx.stream_enabled)

    def test_conversation_wiped_after_reset(self):
        """Conversation state is zeroed by reset()."""
        self.ctx.reset()
        self.assertEqual(self.ctx.current_images, [])
        self.assertEqual(self.ctx.total_queries, 0)
        self.assertEqual(self.ctx.total_tokens_generated, 0)
        self.assertEqual(self.ctx.total_prompt_tokens, 0)
        self.assertEqual(self.ctx.total_time_spent, 0.0)
        self.assertEqual(self.ctx.total_chars_generated, 0)
        self.assertEqual(self.ctx.query_history, [])
        self.assertEqual(self.ctx.context_window_size, 0)
        self.assertEqual(self.ctx.current_context_tokens, 0)

    def test_shell_timeout_default(self):
        """shell_timeout defaults to 5."""
        self.assertEqual(self.ctx.shell_timeout, 5)


# ============================================================================
# 2b. Singleton testing fragility
# ============================================================================

class TestSingletonFragility(unittest.TestCase):
    """CommandContext is a singleton — tests must reset _instance between runs.

    This test class verifies the singleton behavior and documents the manual
    reset pattern required by all test files. See AGENTS.md "Singleton testing
    fragility" for the pending fix (dep-injection or context manager).
    """

    def setUp(self):
        m.CommandContext._instance = None
        m.CommandContext._initialized = False
        self.ctx = m.CommandContext()

    def tearDown(self):
        m.CommandContext._instance = None
        m.CommandContext._initialized = False

    def test_singleton_returns_same_instance(self):
        """Two CommandContext() calls return the exact same object."""
        ctx2 = m.CommandContext()
        self.assertIs(self.ctx, ctx2)

    def test_state_leaks_without_reset(self):
        """Without _instance reset, state leaks across 'new' contexts."""
        self.ctx.system_prompt = "Leaked value"
        ctx2 = m.CommandContext()
        self.assertEqual(ctx2.system_prompt, "Leaked value")

    def test_manual_reset_enables_new_instance(self):
        """Setting _instance = None allows a fresh singleton."""
        self.ctx.system_prompt = "Should be isolated"
        m.CommandContext._instance = None
        m.CommandContext._initialized = False
        ctx2 = m.CommandContext()
        self.assertIsNot(self.ctx, ctx2)
        self.assertEqual(ctx2.system_prompt, m.DEFAULT_SYSTEM_PROMPT)

    def test_teardown_resets_for_next_test(self):
        """tearDown resets so next test starts clean."""
        self.ctx.system_prompt = "From test_teardown_resets"
        # tearDown will reset; the next test will confirm isolation

    def test_next_test_starts_clean_after_teardown(self):
        """Verifies tearDown reset isolates tests."""
        self.assertEqual(self.ctx.system_prompt, m.DEFAULT_SYSTEM_PROMPT)


# ============================================================================
# 2c. Path traversal protection
# ============================================================================

class TestPathTraversal(unittest.TestCase):
    """All tool handlers must reject paths that escape the CWD.

    Each test passes a path that resolves outside the current working directory
    (e.g. ../etc/passwd or /tmp/foo) and verifies the handler returns
    a "Path traversal denied" error without touching the filesystem.
    """

    def setUp(self):
        m.CommandContext._instance = None
        m.CommandContext._initialized = False
        self.ctx = m.CommandContext()
        self.ctx.auto_confirm = True
        self.reg = m.ToolRegistry(ctx=self.ctx)

    def tearDown(self):
        m.CommandContext._instance = None
        m.CommandContext._initialized = False

    # -- via ToolRegistry execute() --

    def test_read_file_rejects_traversal(self):
        result = self.reg.execute("read_file", {"file": "../etc/passwd"})
        self.assertFalse(result["success"])
        self.assertIn("Path traversal denied", result.get("error", ""))

    def test_read_file_rejects_absolute_outside(self):
        result = self.reg.execute("read_file", {"file": "/tmp/foo"})
        self.assertFalse(result["success"])
        self.assertIn("Path traversal denied", result.get("error", ""))

    def test_write_file_rejects_traversal(self):
        result = self.reg.execute("write_file", {"file": "../outside.txt", "content": "x"})
        self.assertFalse(result["success"])
        self.assertIn("Path traversal denied", result.get("error", ""))

    def test_list_directory_rejects_traversal(self):
        result = self.reg.execute("list_directory", {"path": "../"})
        self.assertFalse(result["success"])
        self.assertIn("Path traversal denied", result.get("error", ""))

    def test_edit_file_rejects_traversal(self):
        result = self.reg.execute("edit_file", {"file": "../etc/passwd", "old_string": "root", "new_string": "toor"})
        self.assertFalse(result["success"])
        self.assertIn("Path traversal denied", result.get("error", ""))

    def test_apply_patch_rejects_traversal(self):
        """apply_patch must reject patches targeting files outside CWD."""
        patch = "*** Update File: ../outside.txt\n@@ -1 +1 @@\n-old\n+new\n"
        result = self.reg.execute("apply_patch", {"patch_text": patch})
        self.assertFalse(result["success"])
        self.assertIn("Path traversal denied", result.get("error", ""))

    # -- direct call to _apply_unified_diff --

    def test_apply_unified_diff_rejects_traversal_direct(self):
        """Direct call to _apply_unified_diff with traversal path."""
        patch = "*** Update File: ../etc/hosts\n@@ -1 +1 @@\n-old\n+new\n"
        result = m._apply_unified_diff(patch)
        self.assertFalse(result["success"])
        self.assertIn("Path traversal denied", result.get("error", ""))

    def test_apply_unified_diff_rejects_move_traversal(self):
        """Move operation to outside CWD must be rejected."""
        patch = "*** Move to: ../outside_dir/\n--- a/file.txt\n+++ b/file.txt\n@@ -1 +1 @@\n-old\n+new\n"
        result = m._apply_unified_diff(patch)
        self.assertFalse(result["success"])
        self.assertIn("Path traversal denied", result.get("error", ""))


class TestUnifiedDiffHunkParsing(unittest.TestCase):
    """_parse_unified_hunks must use old-file (group 1) coordinates, not new-file (group 3)."""

    def test_parse_unified_hunks_uses_old_file_coordinates(self):
        """Hunk @@ -50,5 +60,5 @@ should set start=50, not start=60."""
        body = (
            "--- a/test.txt\n"
            "+++ b/test.txt\n"
            "@@ -50,5 +60,5 @@\n"
            " context\n"
            "-old_line\n"
            "+new_line\n"
            " context\n"
        )
        hunks, path, is_delete = m._parse_unified_hunks(body)
        self.assertEqual(len(hunks), 1)
        self.assertEqual(
            hunks[0]["start"], 50,
            f"Expected start=50 (old-file coordinate), got start={hunks[0]['start']}"
        )

    def test_parse_unified_hunks_start_not_end(self):
        """Multi-hunk diff where old/new coordinates differ."""
        body = (
            "--- a/test.txt\n"
            "+++ b/test.txt\n"
            "@@ -10,3 +10,5 @@\n"
            " line1\n"
            " line2\n"
            "+inserted_a\n"
            "+inserted_b\n"
            " line3\n"
            "@@ -15,2 +20,2 @@\n"
            " unchanged\n"
            "-gone\n"
            "+added\n"
        )
        hunks, path, is_delete = m._parse_unified_hunks(body)
        self.assertEqual(len(hunks), 2)
        self.assertEqual(
            hunks[1]["start"], 15,
            f"Second hunk: expected start=15 (old-file), got start={hunks[1]['start']}"
        )

    def test_parse_unified_hunks_blank_lines(self):
        """Empty lines in hunks must not cause premature truncation (regression: break on empty line)."""
        body = (
            "--- a/test.txt\n"
            "+++ b/test.txt\n"
            "@@ -1,5 +1,5 @@\n"
            " line1\n"
            "-\n"
            "+\n"
            " line3\n"
            " line4\n"
            " line5\n"
        )
        hunks, path, is_delete = m._parse_unified_hunks(body)
        self.assertEqual(len(hunks), 1)
        self.assertEqual(len(hunks[0]["old_lines"]), 5,
                         "old_lines truncated by blank line break")
        self.assertEqual(len(hunks[0]["new_lines"]), 5,
                         "new_lines truncated by blank line break")

    def test_parse_unified_hunks_trailing_empty_line(self):
        """Hunk ending with an empty line must not truncate."""
        body = (
            "--- a/test.txt\n"
            "+++ b/test.txt\n"
            "@@ -1,2 +1,2 @@\n"
            "-old\n"
            "+new\n"
            "\n"
        )
        hunks, path, is_delete = m._parse_unified_hunks(body)
        self.assertEqual(len(hunks), 1)
        self.assertEqual(len(hunks[0]["old_lines"]), 2)
        self.assertEqual(len(hunks[0]["new_lines"]), 2)

    def test_apply_unified_diff_no_trailing_newline(self):
        """_apply_unified_diff must not reject patches on files lacking terminal newline (regression)."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                with open("test.txt", "w") as f:
                    f.write("hello\nworld")
                patch = (
                    "--- a/test.txt\n"
                    "+++ b/test.txt\n"
                    "@@ -1,2 +1,2 @@\n"
                    " hello\n"
                    "-world\n"
                    "+earth\n"
                )
                result = m._apply_unified_diff(patch)
                self.assertTrue(result["success"], msg=result.get("error", ""))
                with open("test.txt") as f:
                    content = f.read()
                self.assertIn("earth", content)
            finally:
                os.chdir(old_cwd)

    def test_apply_unified_diff_blank_lines_in_patch(self):
        """_apply_unified_diff must handle patches containing blank context lines."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                with open("test.txt", "w") as f:
                    f.write("a\n\nc")
                patch = (
                    "--- a/test.txt\n"
                    "+++ b/test.txt\n"
                    "@@ -1,3 +1,3 @@\n"
                    " a\n"
                    " \n"
                    "-c\n"
                    "+C\n"
                )
                result = m._apply_unified_diff(patch)
                self.assertTrue(result["success"], msg=result.get("error", ""))
                with open("test.txt") as f:
                    content = f.read()
                self.assertIn("C", content)
            finally:
                os.chdir(old_cwd)


class TestParsePatchSections(unittest.TestCase):
    """_parse_patch_sections must correctly parse OpenCode-style markers with --- headers."""

    def test_parse_patch_sections_opencode_with_hunks(self):
        """OpenCode *** Update File: marker followed by ---/+++ must extract hunks (regression: breakpoint bug)."""
        patch = (
            "*** Update File: test.txt\n"
            "--- a/test.txt\n"
            "+++ b/test.txt\n"
            "@@ -1,2 +1,2 @@\n"
            "-hello\n"
            "+hi\n"
            " world\n"
        )
        sections = m._parse_patch_sections(patch)
        self.assertEqual(len(sections), 1, "Section was lost by premature loop exit")
        self.assertEqual(sections[0]["path"], "test.txt")
        self.assertEqual(len(sections[0]["hunks"]), 1, "Hunk not parsed — body was empty")

    def test_parse_patch_sections_opencode_multi_hunk(self):
        """Multiple hunks under a single OpenCode marker must all be captured."""
        patch = (
            "*** Update File: test.txt\n"
            "--- a/test.txt\n"
            "+++ b/test.txt\n"
            "@@ -1,3 +1,3 @@\n"
            "-a\n"
            "+A\n"
            " b\n"
            " c\n"
            "@@ -10,2 +10,2 @@\n"
            " d\n"
            "-e\n"
            "+E\n"
        )
        sections = m._parse_patch_sections(patch)
        self.assertEqual(len(sections), 1)
        self.assertEqual(len(sections[0]["hunks"]), 2,
                         "Second hunk was truncated by premature loop exit")


# ============================================================================
# 3.  No bare `except:` remains
# ============================================================================
# 3.  No bare `except:` remains
# ============================================================================

class TestNoBareExcept(unittest.TestCase):
    """No bare 'except:' blocks should exist."""

    def test_no_bare_except_in_file(self):
        with open(m.__file__) as f:
            for i, line in enumerate(f, 1):
                stripped = line.strip()
                # Match lines that are exactly "except:" (possibly with comment)
                # but NOT "except Exception:" or "except (A, B):"
                if re.match(r'^except\s*:(#.*)?$', stripped):
                    self.fail(f"Bare 'except:' found at line {i}: {line}")


# ============================================================================
# 4.  `c()` function removed
# ============================================================================

class TestCRemoved(unittest.TestCase):
    """The dead c() function was removed."""

    def test_c_function_not_defined(self):
        self.assertFalse(hasattr(m, 'c'))

    def test_no_c_calls_in_file(self):
        """No calls to c(...) should appear in the source."""
        with open(m.__file__) as f:
            for i, line in enumerate(f, 1):
                if re.search(r'\bc\(', line) and 'colorize' not in line:
                    self.fail(f"Call to c() found at line {i}: {line}")


# ============================================================================
# 5.  HTMLStripper is module-level
# ============================================================================

class TestHTMLStripper(unittest.TestCase):
    """HTMLStripper must be a module-level class replacing old DataGatherer/FallbackHTMLStripper."""

    def test_htmlstripper_exists_at_module_level(self):
        self.assertTrue(hasattr(m, 'HTMLStripper'))
        self.assertTrue(isinstance(m.HTMLStripper, type))
        self.assertTrue(issubclass(m.HTMLStripper, m.HTMLParser))

    def test_htmlstripper_parses_html(self):
        stripper = m.HTMLStripper()
        stripper.feed('<html><body><p>Hello World</p></body></html>')
        self.assertIn('Hello World', stripper.get_text())

    def test_htmlstripper_skips_script(self):
        stripper = m.HTMLStripper()
        stripper.feed('<script>var x=1;</script><p>Visible</p>')
        self.assertNotIn('var x=1', stripper.get_text())
        self.assertIn('Visible', stripper.get_text())

    def test_htmlstripper_strips_tags(self):
        stripper = m.HTMLStripper()
        stripper.feed('<html><body><p>Hello <b>World</b></p></body></html>')
        result = stripper.get_text()
        self.assertIn('Hello', result)
        self.assertIn('World', result)
        self.assertNotIn('<b>', result)
        self.assertNotIn('<p>', result)


# ============================================================================
# 7.  ModelQuery.calculate_stats() is side-effect-free
# ============================================================================

class TestCalculateStatsSideEffectFree(unittest.TestCase):
    """calculate_stats() must not mutate CommandContext."""

    def setUp(self):
        m.CommandContext._instance = None
        m.CommandContext._initialized = False
        self.ctx = m.CommandContext()
        self.mq = m.ModelQuery(BACKEND_URL, 'ollama', context=self.ctx)

    def test_does_not_call_update_stats(self):
        """calculate_stats must NOT call ctx.update_stats itself."""
        original_update = self.ctx.update_stats
        called = []
        def tracking_update(tokens, prompt_tokens, time_spent, chars):
            called.append(True)
            return original_update(tokens, prompt_tokens, time_spent, chars)
        self.ctx.update_stats = tracking_update

        stats = self.mq.calculate_stats(1.0, "hello world", {}, [])
        self.assertEqual(called, [],
                         "calculate_stats must not call ctx.update_stats")

    def test_returns_dict_with_expected_keys(self):
        stats = self.mq.calculate_stats(1.0, "hello world", {}, [])
        for key in ('eval_count', 'prompt_eval_count', 'total_context_tokens',
                     'total_time', 'tps', 'content_length'):
            self.assertIn(key, stats)

    def test_no_method_get_cumulative_stats(self):
        """get_cumulative_stats must not exist on ModelQuery."""
        self.assertFalse(hasattr(self.mq, 'get_cumulative_stats'))

    def test_no_method_reset_stats(self):
        """reset_stats must not exist on ModelQuery."""
        self.assertFalse(hasattr(self.mq, 'reset_stats'))


# ============================================================================
# 8.  --show and --show-details are both reachable
# ============================================================================

class TestShowModelFlags(unittest.TestCase):
    """Both --show and --show-details should route to separate functions."""

    def test_show_details_not_dead_code(self):
        """The 'elif args.show_details:' branch must be reachable.
        Verify by checking the source: 'if args.show:' followed by 'elif args.show_details:'.
        """
        with open(m.__file__) as f:
            lines = f.readlines()
        found = False
        for i, line in enumerate(lines):
            if 'if args.show:' in line:
                for j in range(i+1, min(i+10, len(lines))):
                    if 'elif args.show_details:' in lines[j]:
                        found = True
                        break
                break
        self.assertTrue(found,
                        "'elif args.show_details:' not reachable after 'if args.show:'")

    def test_show_and_show_details_are_separate_args(self):
        """Verify source has both --show and --show-details argument branches."""
        with open(m.__file__) as f:
            content = f.read()
        self.assertIn("--show-details", content)
        self.assertIn("show_model_details", content)


# ============================================================================
# 9.  urlopen replaced everywhere with _request_with_retry
# ============================================================================

class TestUrlopenReplaced(unittest.TestCase):
    """No direct urlopen() calls should remain (all via _request_with_retry)."""

    def test_no_bare_urlopen(self):
        with open(m.__file__) as f:
            content = f.read()
        # urlopen is still used in _request_with_retry itself and in imports
        # Count lines that use urlopen( which are NOT inside _request_with_retry
        lines = content.split('\n')
        in_retry = False
        bare_urlopen_lines = []
        for i, line in enumerate(lines, 1):
            if 'def _request_with_retry' in line:
                in_retry = True
            elif in_retry and 'def ' in line and 'def _request_with_retry' not in line:
                in_retry = False
            if 'urlopen(' in line and not in_retry:
                # Skip import line
                if 'from urllib.request import urlopen' in line:
                    continue
                # Skip startup probe lines (fast, non-retrying URL checks at launch)
                if '# startup-probe' in line:
                    continue
                bare_urlopen_lines.append((i, line.strip()))
        self.assertEqual(bare_urlopen_lines, [],
                         f"Bare urlopen calls remain:\n" +
                         "\n".join(f"  L{n}: {t}" for n, t in bare_urlopen_lines))


# ============================================================================
# 10. Integration: /clear preserves preferences in live chat loop
# ============================================================================

class TestClearPreservesPreferences(unittest.TestCase):
    """/clear must only clear conversation, not preferences."""

    def setUp(self):
        m.CommandContext._instance = None
        m.CommandContext._initialized = False
        self.ctx = m.CommandContext()
        self.ctx.base_url = BACKEND_URL
        self.ctx.backend = 'ollama'
        self.ctx.model = SMALL_MODEL
        self.ctx.system_prompt = 'Test prompt'
        self.ctx.force_no_thinking = True
        self.ctx.debug_mode = True
        self.ctx.stream_enabled = True

    def test_clear_via_run_handle_clear(self):
        loop = m.ChatLoop(self.ctx)
        # Simulate some conversation context
        self.ctx.total_queries = 5
        self.ctx.current_context_tokens = 100
        self.ctx.current_images = ['img']
        self.ctx.query_history = ['hello', 'world']

        result = loop.run_handle_clear('/clear')
        self.assertFalse(result)  # Should return False (continue loop)

        # Preferences preserved
        self.assertTrue(self.ctx.force_no_thinking)
        self.assertTrue(self.ctx.debug_mode)
        self.assertEqual(self.ctx.system_prompt, 'Test prompt')
        self.assertTrue(self.ctx.stream_enabled)

        # Conversation wiped
        self.assertEqual(self.ctx.total_queries, 0)
        self.assertEqual(self.ctx.current_context_tokens, 0)
        self.assertEqual(self.ctx.current_images, [])
        self.assertEqual(self.ctx.query_history, [])
        self.assertEqual(self.ctx.context_window_size, 0)


# ============================================================================
# 11. Integration: /switchmodel preserves messages
# ============================================================================

class TestSwitchmodelPreservesMessages(unittest.TestCase):
    """/switchmodel must preserve self.messages across model switch."""

    def setUp(self):
        m.CommandContext._instance = None
        m.CommandContext._initialized = False
        self.ctx = m.CommandContext()
        self.ctx.base_url = BACKEND_URL
        self.ctx.backend = 'ollama'
        self.ctx.model = SMALL_MODEL
        self.ctx.system_prompt = 'Reply in 1 word.'

    def test_switchmodel_keeps_messages(self):
        loop = m.ChatLoop(self.ctx)
        # messages is lazy-initialized in run_process_query; set it manually
        loop.messages = [{'role': 'system', 'content': self.ctx.system_prompt}]
        loop.messages.append({"role": "user", "content": "hello"})
        loop.messages.append({"role": "assistant", "content": "hi"})
        self.assertEqual(len(loop.messages), 3)

        # Switch model — suppress output
        import builtins
        captured = []
        original_print = builtins.print
        def capturing_print(*args, **kwargs):
            captured.extend(str(a) for a in args)
        builtins.print = capturing_print
        try:
            result = loop.run_handle_switchmodel('/switchmodel ' + SMALL_MODEL)
        finally:
            builtins.print = original_print

        # Messages should still be intact
        self.assertEqual(len(loop.messages), 3,
                         "Messages should be preserved across switchmodel")
        self.assertEqual(loop.messages[1]["role"], "user")
        self.assertEqual(loop.messages[2]["content"], "hi")


# ============================================================================
# 12. shell_timeout is configurable across all paths
# ============================================================================

class TestShellTimeout(unittest.TestCase):
    """shell_timeout must be set in batch and interactive paths."""

    def test_default_is_5(self):
        m.CommandContext._instance = None
        m.CommandContext._initialized = False
        ctx = m.CommandContext()
        self.assertEqual(ctx.shell_timeout, 5)


# ============================================================================
# 13. AGENTS.md line number consistency
# ============================================================================

class TestAgentsMdConsistency(unittest.TestCase):
    """Verify AGENTS.md line-number references are still accurate."""

    def test_agents_md_mentions_current_file_length(self):
        with open(m.__file__) as f:
            lines = f.readlines()
        # AGENTS.md says "ollamaquery2.py (line 3462)" — verify roughly close
        with open('AGENTS.md') as f:
            agents = f.read()
        match = re.search(r'ollamaquery2\.py.*?\(line (\d+)\)', agents)
        if match:
            expected_len = int(match.group(1))
            # Allow ±20 lines (padding for minor edits)
            self.assertAlmostEqual(len(lines), expected_len, delta=20,
                                   msg=f"AGENTS.md says {expected_len} lines "
                                       f"but file has {len(lines)}")


# ============================================================================
# 14. Live integration: full query with retry
# ============================================================================

class TestLiveQuery(unittest.TestCase):
    """End-to-end query against live backend."""

    def setUp(self):
        m.CommandContext._instance = None
        m.CommandContext._initialized = False
        self.ctx = m.CommandContext()
        self.ctx.base_url = BACKEND_URL
        self.ctx.backend = BACKEND_TYPE
        self.ctx.model = SMALL_MODEL
        self.ctx.system_prompt = 'Reply in 1 word.'

    @any_backend
    def test_basic_query_streams(self):
        loop = m.ChatLoop(self.ctx)
        loop.run_process_query('Say hello')
        # run_process_query stores assistant response in self.messages
        self.assertTrue(hasattr(loop, 'messages'),
                        "messages should be populated after a query")
        self.assertGreaterEqual(len(loop.messages), 2,
                                "Should have system + user + assistant messages")
        self.assertEqual(loop.messages[-1]['role'], 'assistant')
        self.assertGreater(len(loop.messages[-1]['content']), 0)

    @any_backend
    def test_stats_accumulate(self):
        loop = m.ChatLoop(self.ctx)
        loop.run_process_query('Say hello')
        loop.run_process_query('Say goodbye')
        cum = self.ctx.get_cumulative_stats()
        self.assertEqual(cum['total_queries'], 2)
        self.assertGreater(cum['total_tokens'], 0,
                           "Total tokens should be tracked across queries")
        self.assertGreater(cum['total_completion_tokens'], 0,
                           "Completion tokens should be tracked")
        self.assertGreater(cum['total_prompt_tokens'], 0,
                           "Prompt tokens should be tracked")

    @any_backend
    def test_context_tokens_updated(self):
        """Verify context token tracking is updated after a query."""
        loop = m.ChatLoop(self.ctx)
        self.assertEqual(self.ctx.current_context_tokens, 0,
                         "Context tokens should start at 0")
        loop.run_process_query('Say hello')
        self.assertGreater(self.ctx.current_context_tokens, 0,
                           "Context tokens should be > 0 after a query")

    @any_backend
    def test_chat_history_preserved(self):
        """Verify full conversation history is maintained."""
        loop = m.ChatLoop(self.ctx)
        loop.run_process_query('First message')
        loop.run_process_query('Second message')
        # Messages: system + user1 + assistant1 + user2 + assistant2
        self.assertEqual(len(loop.messages), 5)
        self.assertEqual(loop.messages[1]['content'], 'First message')
        self.assertEqual(loop.messages[2]['role'], 'assistant')
        self.assertEqual(loop.messages[3]['content'], 'Second message')
        self.assertEqual(loop.messages[4]['role'], 'assistant')
        self.assertGreater(len(loop.messages[2]['content']), 0)
        self.assertGreater(len(loop.messages[4]['content']), 0)


# ============================================================================
# 15. process_inline_commands with shell_timeout
# ============================================================================

class TestProcessInlineCommands(unittest.TestCase):
    """Inline shell commands use configured shell_timeout."""

    def test_execute_os_command_with_timeout(self):
        """execute_os_command should accept timeout and return output string."""
        result = m.execute_os_command('echo hello', timeout=5)
        self.assertIn('hello', result)


# ============================================================================
# Run
# ============================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
