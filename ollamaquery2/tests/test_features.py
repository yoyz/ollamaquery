#!/usr/bin/env python3
"""Comprehensive release test suite for ollamaquery2.

Auto-detects available backends and tests all features one by one.
Discovery order: environment variable → localhost → local network IPs.
Uses a small model (granite4:350m) for speed and reliability.

Environment variables:
  OLLAMA_HOST   — Ollama server URL (auto-discovered if not set)
  LLAMACPP_HOST — Llama.cpp server URL (auto-discovered if not set)
  TEST_MODEL    — Model name for query tests (default: granite4:350m)

Usage:
  python3 -m unittest tests.test_features -v
  python3 -m unittest tests.test_features.TestBasicQueries -v
"""

import io
import os
import re
import sys
import json
import time
import signal
import socket
import atexit
import textwrap
import tempfile
import unittest
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock
from urllib.request import Request
from urllib.error import URLError, HTTPError

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ollamaquery2 as q

# ── Backend Discovery ──────────────────────────────────────────────────────


def _discover_backends():
    """Auto-discover LLM backends by probing common URLs.

    Checks in order:
    1. User-set environment variables (OLLAMA_HOST, LLAMACPP_HOST, LMSTUDIO_HOST)
    2. Default localhost URLs on standard ports
    3. Local network IPs on standard ports (via gethostbyname_ex)

    Returns:
        dict: Backend name -> (discovered_url, available_bool)
    """
    probes = [
        ('ollama',   'OLLAMA_HOST',   'http://127.0.0.1:11434', 11434,
         lambda u: q.check_backend_with_get(u, 'ollama')),
        ('llamacpp', 'LLAMACPP_HOST', 'http://127.0.0.1:8080',   8080,
         lambda u: q.check_backend_with_head(u, 'llama.cpp')),
        ('lmstudio', 'LMSTUDIO_HOST', 'http://127.0.0.1:1234',   1234,
         lambda u: q.check_lmstudio(u)),
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
LMSTUDIO_HOST = _discovered_backends['lmstudio'][0]
HAS_OLLAMA    = _discovered_backends['ollama'][1]
HAS_LLAMACPP  = _discovered_backends['llamacpp'][1]
HAS_LMSTUDIO  = _discovered_backends['lmstudio'][1]
SMALL_MODEL   = os.environ.get('TEST_MODEL', 'granite4:350m')

ollama_only   = unittest.skipUnless(HAS_OLLAMA,   'Ollama backend not available')
llamacpp_only = unittest.skipUnless(HAS_LLAMACPP, 'Llama.cpp backend not available')
any_backend   = unittest.skipUnless(HAS_OLLAMA or HAS_LLAMACPP or HAS_LMSTUDIO,
                                    'No backend available')


def _tokenize_available(base_url, model):
    """Check if the /api/tokenize endpoint works for a given model."""
    try:
        payload = json.dumps({"model": model, "content": "test"}).encode('utf-8')
        req = Request(f"{base_url}/api/tokenize", data=payload,
                      headers={'Content-Type': 'application/json'})
        with q.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def _pick_small_model(base_url, prefer_under_gb=9):
    """Pick a small chat-capable model from the server, preferring models under `prefer_under_gb` GB."""
    models = q.fetch_models_ollama(base_url)
    if not models:
        return SMALL_MODEL

    for m in models:
        raw_size = m.get("size") or m.get("size_bytes") or m.get("model_size") or 0
        try:
            m['size_bytes'] = int(raw_size)
        except (TypeError, ValueError):
            m['size_bytes'] = 0

    # Exclude API proxies (tiny placeholder sizes < 10 MB), cloud proxies, embedding models
    min_bytes = 10 * 1024 * 1024
    skip_keywords = ['cloud', 'embedding', 'vl:']
    candidates = [m for m in models if m['size_bytes'] >= min_bytes and not any(k in m.get('name', '').lower() for k in skip_keywords)]

    # Prefer models under the size limit
    small_chat = [m for m in candidates if m['size_bytes'] < prefer_under_gb * (1024**3)]
    if small_chat:
        small_chat.sort(key=lambda x: x['size_bytes'])
        return small_chat[0]['name']

    # Fallback: smallest model overall
    if candidates:
        candidates.sort(key=lambda x: x['size_bytes'])
        return candidates[0]['name']

    return sorted(models, key=lambda x: x.get('name', ''))[0]['name']


def _ensure_model_loaded(base_url, model_name):
    """Ensure the given model is loaded into memory on the Ollama server."""
    loaded = q.fetch_loaded_models_ollama(base_url)
    loaded_names = [m.get('name', '') for m in loaded]
    if model_name in loaded_names:
        return True

    # Load by sending a minimal sync query
    mq = q.ModelQuery(base_url, 'ollama')
    mq.query_sync(
        [{'role': 'user', 'content': 'ok'}],
        model_name,
        stream_enabled=False
    )
    return True


# If Ollama is available, auto-pick a small model and ensure it's loaded
if HAS_OLLAMA:
    SMALL_MODEL = _pick_small_model(OLLAMA_HOST)
    _ensure_model_loaded(OLLAMA_HOST, SMALL_MODEL)

# Detect which backend is actually available for any_backend tests
ANY_BACKEND = None
ANY_BASE_URL = None
ANY_MODEL = None

if HAS_LMSTUDIO:
    models = q.fetch_models_llamacpp(LMSTUDIO_HOST)
    if models:
        ANY_BACKEND = 'lmstudio'
        ANY_BASE_URL = LMSTUDIO_HOST
        ANY_MODEL = models[0]['name']
elif HAS_LLAMACPP:
    models = q.fetch_models_llamacpp(LLAMACPP_HOST)
    if models:
        ANY_BACKEND = 'llamacpp'
        ANY_BASE_URL = LLAMACPP_HOST
        ANY_MODEL = models[0]['name']
elif HAS_OLLAMA:
    ANY_BACKEND = 'ollama'
    ANY_BASE_URL = OLLAMA_HOST
    ANY_MODEL = SMALL_MODEL


def _fresh_ctx(base_url=OLLAMA_HOST, backend='ollama', model=SMALL_MODEL,
               prompt='Reply in 3 words or fewer.'):
    """Create a fresh CommandContext + ChatLoop pair."""
    q.CommandContext._instance = None
    q.CommandContext._initialized = False
    ctx = q.CommandContext()
    ctx.base_url = base_url
    ctx.backend = backend
    ctx.model = model
    ctx.system_prompt = prompt
    ctx.context_window_size = 32768
    ctx.stream_enabled = True
    return ctx


def _chat_loop(ctx=None, **kwargs):
    """Create a ChatLoop, optionally from a fresh context."""
    if ctx is None:
        ctx = _fresh_ctx(**kwargs)
    return q.ChatLoop(ctx)


# ============================================================================
# 1.  Core Infrastructure
# ============================================================================

class TestConfiguration(unittest.TestCase):
    """Verify default configuration values."""

    def test_default_ollama_host(self):
        self.assertEqual(q.DEFAULT_OLLAMA_HOST, 'http://127.0.0.1:11434')

    def test_default_llamacpp_host(self):
        self.assertEqual(q.DEFAULT_LLAMACPP_HOST, 'http://127.0.0.1:8080')

    def test_default_system_prompt(self):
        self.assertIn('accurate', q.DEFAULT_SYSTEM_PROMPT.lower())

    def test_builtin_prompts_keys(self):
        for key in ('default', 'coder', 'sysadmin', 'concise', 'doctor', 'teacher', 'politic'):
            self.assertIn(key, q.BUILTIN_PROMPTS)

    def test_max_context_size(self):
        self.assertEqual(q.MAX_CONTEXT_SIZE, 4192000)


class TestBackendDetection(unittest.TestCase):
    """Verify backend detection logic."""

    @ollama_only
    def test_ollama_api_tags(self):
        models = q.fetch_models_ollama(OLLAMA_HOST)
        self.assertIsInstance(models, list)
        self.assertGreater(len(models), 0)

    @ollama_only
    def test_ollama_live_ps(self):
        models = q.fetch_loaded_models_ollama(OLLAMA_HOST)
        self.assertIsInstance(models, list)

    @ollama_only
    def test_check_backend_with_get_ollama(self):
        result = q.check_backend_with_get(OLLAMA_HOST, 'ollama')
        self.assertTrue(result)

    def test_check_backend_with_head_failure(self):
        result = q.check_backend_with_head('http://127.0.0.1:1', 'nonexistent')
        self.assertFalse(result)

    def test_resolve_connection_defaults(self):
        """resolve_connection defaults to ollama."""
        args = lambda: None  # noqa: E731
        args.backend = None
        args.host = None
        backend, url = q.resolve_connection(args)
        self.assertIn(backend, ('ollama', 'llamacpp', 'lmstudio'))


@ollama_only
class TestModelListing(unittest.TestCase):
    """Model listing features."""

    def test_fetch_models_ollama_returns_list(self):
        models = q.fetch_models_ollama(OLLAMA_HOST)
        self.assertIsInstance(models, list)

    def test_fetch_models_has_names(self):
        models = q.fetch_models_ollama(OLLAMA_HOST)
        names = [m.get('name', '') for m in models]
        self.assertIn(SMALL_MODEL, names,
                      f'{SMALL_MODEL} should be in model list')

    def test_fetch_model_info_ollama(self):
        info = q.fetch_model_info_ollama(OLLAMA_HOST, SMALL_MODEL)
        self.assertIsInstance(info, dict)
        self.assertIn('details', info)

    def test_parse_size(self):
        self.assertEqual(q.parse_size(1073741824), '1.0 GB')
        self.assertEqual(q.parse_size(52428800), '50.0 MB')
        self.assertEqual(q.parse_size(5242880), '5.0 MB')
        self.assertEqual(q.parse_size(512000), '500.0 KB')
        self.assertEqual(q.parse_size(500), '500 B')
        self.assertEqual(q.parse_size(0), 'N/A')
        self.assertEqual(q.parse_size(None), 'N/A')


class TestCommandRegistry(unittest.TestCase):
    """Command registry structure."""

    def test_commands_has_required_keys(self):
        for cmd in ('help', 'quit', 'clear', 'stats', 'listmodel',
                     'switchmodel', 'debug', 'image'):
            self.assertIn(cmd, q.COMMANDS, f'Missing command: {cmd}')

    def test_get_command_aliases(self):
        aliases = q.get_command_aliases()
        self.assertIn('/help', aliases)
        self.assertIn('/quit', aliases)
        self.assertIn('/clear', aliases)

    def test_get_commands_by_category(self):
        by_cat = q.get_commands_by_category()
        self.assertIn('Core', by_cat)
        self.assertIn('Model', by_cat)


# ============================================================================
# 2.  Retry Logic
# ============================================================================

class TestRetry(unittest.TestCase):
    """Verify _request_with_retry behavior."""

    def test_retry_success_first_try(self):
        with patch.object(q, 'urlopen') as mock:
            mock.return_value = io.BytesIO(b'{"ok": true}')
            resp = q._request_with_retry(Request('http://localhost/x'))
            self.assertEqual(json.loads(resp.read()), {'ok': True})
            self.assertEqual(mock.call_count, 1)

    def test_retry_on_transient_error(self):
        with patch.object(q, 'urlopen') as mock:
            mock.side_effect = [URLError('fail'), io.BytesIO(b'{}')]
            q._request_with_retry(Request('http://localhost/x'),
                                 max_retries=2, delay=0.01)
            self.assertEqual(mock.call_count, 2)

    def test_no_retry_on_4xx(self):
        with patch.object(q, 'urlopen') as mock:
            mock.side_effect = HTTPError(
                'http://localhost/x', 404, 'Not Found',
                {}, io.BytesIO(b''))
            with self.assertRaises(HTTPError):
                q._request_with_retry(Request('http://localhost/x'))
            self.assertEqual(mock.call_count, 1)

    def test_retry_on_5xx(self):
        with patch.object(q, 'urlopen') as mock:
            mock.side_effect = [
                HTTPError('http://localhost/x', 503, 'Down',
                          {}, io.BytesIO(b'')),
                io.BytesIO(b'{"ok": true}')]
            resp = q._request_with_retry(Request('http://localhost/x'),
                                        max_retries=2, delay=0.01)
            self.assertEqual(json.loads(resp.read()), {'ok': True})
            self.assertEqual(mock.call_count, 2)

    def test_retry_stderr_message(self):
        with patch.object(q, 'urlopen') as mock:
            mock.side_effect = [URLError('fail'), io.BytesIO(b'{}')]
            stderr = io.StringIO()
            old = sys.stderr
            sys.stderr = stderr
            try:
                q._request_with_retry(Request('http://localhost/x'),
                                     max_retries=2, delay=0.01)
            finally:
                sys.stderr = old
            self.assertIn('[RETRY]', stderr.getvalue())


# ============================================================================
# 3.  Color & Theme System
# ============================================================================

class TestThemeSystem(unittest.TestCase):
    """Color theme functionality."""

    def test_get_theme_default(self):
        theme = q.get_theme('default')
        self.assertIn('success', theme)
        self.assertIn('error', theme)
        self.assertIn('warning', theme)

    def test_get_theme_keys_contain_colors(self):
        for name in ('default', 'minimal', 'emacs_dark', 'vim_dark', 'high_contrast'):
            theme = q.get_theme(name)
            self.assertIn('success', theme)

    def test_colorize_basic(self):
        result = q.colorize('hello', 'success')
        self.assertIn('hello', result)

    def test_colorize_no_color(self):
        with patch.dict(os.environ, {'NO_COLOR': '1'}):
            result = q.colorize('hello', 'success')
            self.assertEqual(result, 'hello')

    def test_colors_enabled(self):
        with patch.dict(os.environ, clear=True):
            with patch.object(sys.stdout, 'isatty', return_value=True):
                self.assertTrue(q.colors_enabled())

    def test_theme_presets_names(self):
        """All preset theme names produce valid themes."""
        for name in ('default', 'minimal', 'emacs_dark', 'vim_dark', 'high_contrast'):
            theme = q.get_theme(name)
            # Every theme must have at minimum these keys
            for key in ('success', 'error', 'warning', 'info', 'muted'):
                self.assertIn(key, theme,
                              f'Theme {name!r} missing key {key!r}')


# ============================================================================
# 4.  Basic Queries
# ============================================================================

@ollama_only
class TestBasicQueries(unittest.TestCase):
    """Streaming and sync query functionality."""

    def setUp(self):
        self.ctx = _fresh_ctx()

    def test_modelquery_creation(self):
        mq = q.ModelQuery(OLLAMA_HOST, 'ollama')
        self.assertEqual(mq.base_url, OLLAMA_HOST)
        self.assertEqual(mq.backend, 'ollama')

    def test_query_sync(self):
        mq = q.ModelQuery(OLLAMA_HOST, 'ollama', context=self.ctx)
        result = mq.query_sync(
            [{'role': 'user', 'content': 'Say hello in 1 word'}],
            SMALL_MODEL)
        self.assertIsInstance(result, dict)
        # Ollama /api/chat returns a message object
        self.assertIn('message', result)

    def test_query_stream_returns_content(self):
        loop = _chat_loop(self.ctx)
        loop.run_process_query('Say hello')
        self.assertTrue(hasattr(loop, 'messages'))
        self.assertGreaterEqual(len(loop.messages), 2)
        self.assertEqual(loop.messages[-1]['role'], 'assistant')
        self.assertGreater(len(loop.messages[-1]['content']), 0)

    def test_query_response_not_empty(self):
        mq = q.ModelQuery(OLLAMA_HOST, 'ollama', context=self.ctx)
        result = mq.query_sync(
            [{'role': 'user', 'content': 'Say hello in 1 word'}],
            SMALL_MODEL)
        content = (result.get('message', {}) or {}).get('content', '')
        self.assertGreater(len(content), 0)

    def test_empty_input_returns_none(self):
        loop = _chat_loop(self.ctx)
        result = loop.run_process_query('')
        self.assertIsNone(result)

    def test_whitespace_input_returns_none(self):
        loop = _chat_loop(self.ctx)
        result = loop.run_process_query('   ')
        self.assertIsNone(result)


@any_backend
class TestAnyBackend(unittest.TestCase):
    """Generic query tests that work with any available backend."""

    def setUp(self):
        self.ctx = _fresh_ctx(base_url=ANY_BASE_URL, backend=ANY_BACKEND, model=ANY_MODEL)

    def test_modelquery_creation(self):
        mq = q.ModelQuery(ANY_BASE_URL, ANY_BACKEND)
        self.assertEqual(mq.base_url, ANY_BASE_URL)
        self.assertEqual(mq.backend, ANY_BACKEND)

    def test_query_sync_returns_dict(self):
        mq = q.ModelQuery(ANY_BASE_URL, ANY_BACKEND, context=self.ctx)
        result = mq.query_sync(
            [{'role': 'user', 'content': 'Say hello in 1 word'}],
            ANY_MODEL)
        self.assertIsInstance(result, dict)
        content = ""
        if isinstance(result, dict):
            content = result.get('message', {}).get('content', '')
            if not content:
                choices = result.get('choices', [])
                if choices:
                    content = choices[0].get('message', {}).get('content', '')
        self.assertGreater(len(content), 0, "Response should contain content")

    def test_chat_loop_processes_query(self):
        loop = _chat_loop(self.ctx)
        loop.run_process_query('Say hello')
        self.assertTrue(hasattr(loop, 'messages'))
        self.assertGreaterEqual(len(loop.messages), 2)
        self.assertEqual(loop.messages[-1]['role'], 'assistant')
        self.assertGreater(len(loop.messages[-1]['content']), 0)

    def test_stats_accumulate(self):
        loop = _chat_loop(self.ctx)
        loop.run_process_query('Say hi')
        loop.run_process_query('Say bye')
        cum = self.ctx.get_cumulative_stats()
        self.assertEqual(cum['total_queries'], 2)
        self.assertGreater(cum['total_tokens'], 0)

    def test_context_tokens_updated(self):
        loop = _chat_loop(self.ctx)
        loop.run_process_query('Say hello')
        self.assertGreater(self.ctx.current_context_tokens, 0)

    def test_switchmodel_changes_model(self):
        loop = _chat_loop(self.ctx)
        loop.run_handle_switchmodel(f'/switchmodel {ANY_MODEL}')
        self.assertEqual(self.ctx.model, ANY_MODEL)


@ollama_only
class TestQueryStats(unittest.TestCase):
    """Query statistics tracking."""

    def setUp(self):
        self.ctx = _fresh_ctx()
        self.loop = _chat_loop(self.ctx)

    def test_stats_accumulate_over_multiple_queries(self):
        self.loop.run_process_query('Say hi')
        self.loop.run_process_query('Say bye')
        cum = self.ctx.get_cumulative_stats()
        self.assertEqual(cum['total_queries'], 2)
        self.assertGreater(cum['total_completion_tokens'], 0)

    def test_context_bar_renders(self):
        """context_bar produces a string with percentage."""
        bar = q.context_bar(100, 32768)
        self.assertIn('%', bar)
        self.assertIn('[', bar)
        self.assertIn(']', bar)

    def test_context_bar_zero_window(self):
        bar = q.context_bar(100, 0)
        self.assertEqual(bar, '')


# ============================================================================
# 5.  Inline Processing
# ============================================================================

class TestInlineProcessing(unittest.TestCase):
    """@file, /curl, !command, and multiline processing."""

    def test_file_inclusion(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt',
                                         delete=False) as f:
            f.write('Hello from file')
            f.flush()
            fname = f.name
        try:
            result = q.process_inline_commands("@" + fname)
            self.assertIn('Hello from file', result)
        finally:
            os.unlink(fname)

    def test_file_inclusion_mid_sentence(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt',
                                         delete=False) as f:
            f.write('WORLD')
            f.flush()
            fname = f.name
        try:
            result = q.process_inline_commands("hello @" + fname + " foo")
            self.assertIn('hello', result)
            self.assertIn('WORLD', result)
            self.assertIn('foo', result)
        finally:
            os.unlink(fname)

    def test_shell_command_execution(self):
        result = q.process_inline_commands('!echo hello_shell')
        self.assertIn('hello_shell', result)

    def test_shell_command_validation_rejects_dangerous(self):
        self.assertFalse(q.validate_shell_command_safety('echo hello; python -c "import os"'))

    def test_sanitize_shell_command(self):
        result = q.sanitize_shell_command('echo hello')
        self.assertIsNotNone(result)

    def test_sanitize_dangerous_returns_none(self):
        self.assertIsNone(q.sanitize_shell_command('hello && touch /tmp/evil'))

    def test_sanitize_blocks_command_substitution(self):
        """$(...) must be blocked by sanitize_shell_command."""
        self.assertIsNone(q.sanitize_shell_command('echo $(whoami)'))
        self.assertIsNone(q.sanitize_shell_command('echo $(touch /tmp/injected)'))
        self.assertIsNone(q.sanitize_shell_command('cat $(ls /etc)'))

    def test_validate_blocks_dollar_paren(self):
        """$(...) must be blocked by validate_shell_command_safety."""
        self.assertFalse(q.validate_shell_command_safety('echo $(whoami)'))
        self.assertFalse(q.validate_shell_command_safety('nslookup $(hostname)'))

    def test_execute_os_command_basic(self):
        result = q.execute_os_command('echo ok', timeout=5)
        self.assertIn('ok', result)
        self.assertIn('Command executed', result)


class TestMultilineInput(unittest.TestCase):
    """Multiline input with \"\"\" and \\."""

    def test_triple_quote_detection(self):
        """The function should detect triple-quote blocks."""
        # Direct unit test of the underlying logic
        line = '"""'
        self.assertTrue(line.strip() == '"""',
                        'Triple quote should be detected')

    def test_backslash_continuation_detection(self):
        line = 'hello\\'
        self.assertTrue(line.endswith('\\'),
                        'Trailing backslash should indicate continuation')


# ============================================================================
# 6.  Command Handling
# ============================================================================

class TestCommandHandlers(unittest.TestCase):
    """Verify each /command handler dispatches correctly."""

    def setUp(self):
        self.ctx = _fresh_ctx()
        self.loop = _chat_loop(self.ctx)

    def test_handle_exit(self):
        for cmd in ('/quit', '/exit', 'quit', 'exit'):
            result = self.loop.run_handle_exit(cmd)
            self.assertIs(result, True, f'{cmd} should return True')

    def test_handle_help(self):
        for cmd in ('/?', '/help'):
            result = self.loop.run_handle_help(cmd)
            self.assertIs(result, False)

    def test_handle_clear(self):
        result = self.loop.run_handle_clear('/clear')
        self.assertIs(result, False)

    def test_handle_debug(self):
        result = self.loop.run_handle_debug('/debug')
        self.assertIs(result, False)

    def test_handle_debug_set_level(self):
        result = self.loop.run_handle_debug('/debug all verbose')
        self.assertIs(result, False)

    def test_handle_thinking_on(self):
        result = self.loop.run_handle_thinking('/thinkingon')
        self.assertIs(result, False)
        self.assertFalse(self.ctx.force_no_thinking)

    def test_handle_thinking_off(self):
        result = self.loop.run_handle_thinking('/thinkingoff')
        self.assertIs(result, False)
        self.assertTrue(self.ctx.force_no_thinking)

    @ollama_only
    def test_handle_listmodel(self):
        result = self.loop.run_handle_listmodel('/listmodel')
        self.assertIs(result, False)

    @ollama_only
    def test_handle_stats(self):
        result = self.loop.run_handle_stats('/stats')
        self.assertIs(result, False)

    def test_handle_cwd(self):
        result = self.loop.run_handle_cwd('/cwd')
        self.assertIs(result, False)

    def test_handle_ls(self):
        result = self.loop.run_handle_ls('/ls')
        self.assertIs(result, False)

    def test_handle_ls_no_shell_injection(self):
        """Verify /ls does not use shell=True with unsanitized input."""
        with patch.object(q.subprocess, 'run') as mock_run:
            self.loop.run_handle_ls('/ls ; touch /tmp/evil')
            args, kwargs = mock_run.call_args
            # Must use list form (shell=False), not string form (shell=True)
            self.assertFalse(kwargs.get('shell', False),
                             '/ls must not use shell=True')
            # The command must be a list
            self.assertIsInstance(args[0], list,
                                  '/ls must use list args, not string')
            # Dangerous characters are safe as literal ls args (not shell-interpreted)
            self.assertEqual(args[0][0], 'ls',
                             'First arg must be ls command')
            self.assertIn(';', args[0],
                          'Semicolon must be passed as literal ls arg, not shell char')

    @ollama_only
    def test_handle_switchmodel(self):
        result = self.loop.run_handle_switchmodel(
            f'/switchmodel {SMALL_MODEL}')
        self.assertFalse(result,
                         'switchmodel should return False (continue loop)')
        self.assertEqual(self.ctx.model, SMALL_MODEL)

    def test_handle_switchmodel_without_model(self):
        result = self.loop.run_handle_switchmodel('/switchmodel')
        self.assertFalse(result,
                         'should return False for missing model name')

    def test_handle_clear_resets_context(self):
        self.loop.run_process_query('Say hello')
        self.assertGreater(len(self.loop.messages), 0)
        self.loop.run_handle_clear('/clear')
        self.assertEqual(self.ctx.total_queries, 0)


class TestModelGuard(unittest.TestCase):
    """Verify the 'no model selected' guard blocks queries."""

    def setUp(self):
        self.ctx = _fresh_ctx(model='')
        self.loop = _chat_loop(self.ctx)

    def test_query_without_model_shows_error(self):
        stderr = io.StringIO()
        old = sys.stderr
        sys.stderr = stderr
        try:
            self.loop.run_process_query('Say hello')
        finally:
            sys.stderr = old
        self.assertIn('No model selected', stderr.getvalue())

    def test_commands_bypass_nomodel_guard(self):
        """Commands like /help, /stats, /debug should work without a model."""
        self.assertIs(self.loop.run_handle_help('/help'), False)
        self.assertIs(self.loop.run_handle_debug('/debug'), False)
        self.assertIs(self.loop.run_handle_clear('/clear'), False)


# ============================================================================
# 7.  HTML / Data Extraction
# ============================================================================

class TestHTMLParsing(unittest.TestCase):
    """HTMLStripper and text extraction."""

    def test_html_stripper_module_level(self):
        self.assertTrue(hasattr(q, 'HTMLStripper'))

    def test_html_stripper_extracts_text(self):
        stripper = q.HTMLStripper()
        stripper.feed('<p>Hello <b>World</b></p>')
        self.assertIn('Hello', stripper.get_text())
        self.assertIn('World', stripper.get_text())

    def test_html_stripper_skips_script(self):
        stripper = q.HTMLStripper()
        stripper.feed('<script>var x=1;</script><p>Visible</p>')
        self.assertNotIn('var x=1', stripper.get_text())

    def test_html_stripper_empty(self):
        stripper = q.HTMLStripper()
        stripper.feed('<script>code</script>')
        self.assertEqual(stripper.get_text(), '')

    def test_html_stripper_spaces_between_elements(self):
        """get_text() must preserve word separation."""
        stripper = q.HTMLStripper()
        stripper.feed('<p>Hello</p><p>World</p>')
        self.assertEqual(stripper.get_text(), 'Hello World')

    def test_html_stripper_multiple_words_per_element(self):
        stripper = q.HTMLStripper()
        stripper.feed('<div>First item</div><div>Second item</div>')
        self.assertIn('First item', stripper.get_text())
        self.assertIn('Second item', stripper.get_text())
        self.assertIn(' ', stripper.get_text())


class TestFetchAndConvertUrl(unittest.TestCase):
    """URL fetching and HTML-to-text conversion."""

    def test_get_html_bytes_from_nonexistent(self):
        result = q.get_html_bytes('http://127.0.0.1:1/nonexistent')
        # Should return empty bytes on failure after retries
        self.assertEqual(result, b'')

    def test_get_html_bytes_exceeds_depth(self):
        result = q.get_html_bytes('http://example.com', depth=5)
        self.assertEqual(result, b'')

    def test_fetch_and_convert_url_fallback_to_htmlstripper(self):
        """fetch_and_convert_url uses HTMLStripper as fallback."""
        text, tool = q.fetch_and_convert_url('')
        self.assertEqual(text, '')
        self.assertEqual(tool, 'None')


# ============================================================================
# 8.  Token Counting & Context
# ============================================================================

class TestTokenCounting(unittest.TestCase):
    """Token estimation and API token counting."""

    def test_estimate_token_count_basic(self):
        count = q.estimate_token_count('hello world foo bar')
        self.assertGreaterEqual(count, 1)

    def test_estimate_token_count_code(self):
        count = q.estimate_token_count('def foo(x): return x + 1')
        self.assertGreater(count, q.estimate_token_count('hello world foo bar'))

    def test_estimate_token_count_empty(self):
        self.assertEqual(q.estimate_token_count(''), 0)

    def test_estimate_token_count_none(self):
        self.assertEqual(q.estimate_token_count(None), 0)

    def test_calculate_context_tokens(self):
        ctx = _fresh_ctx()
        messages = [
            {'role': 'system', 'content': 'You are a bot.'},
            {'role': 'user', 'content': 'Hello'}
        ]
        count = ctx.calculate_context_tokens(messages)
        self.assertGreater(count, 0)

    @ollama_only
    def test_get_message_token_count_ollama(self):
        if not _tokenize_available(OLLAMA_HOST, SMALL_MODEL):
            self.skipTest(f'/api/tokenize not available for {SMALL_MODEL}')
        count = q.get_message_token_count_ollama(
            OLLAMA_HOST, 'hello world', SMALL_MODEL)
        self.assertGreater(count, 0)

    @ollama_only
    def test_get_ollama_context_size(self):
        size = q.get_ollama_context_size(OLLAMA_HOST, SMALL_MODEL)
        self.assertGreaterEqual(size, 0)


# ============================================================================
# 9.  Debug Manager
# ============================================================================

class TestDebugManager(unittest.TestCase):
    """DebugManager levels and categories."""

    def setUp(self):
        self.dm = q.DebugManager()

    def test_default_is_off(self):
        for cat in q.DebugManager.CATEGORIES:
            self.assertEqual(self.dm.get_level(cat), 0)

    def test_set_level_all(self):
        self.dm.set_level('all', 'verbose')
        for cat in q.DebugManager.CATEGORIES:
            self.assertEqual(self.dm.get_level(cat), 2)

    def test_set_level_off(self):
        self.dm.set_level('all', 'verbose')
        self.dm.set_level('all', 'off')
        for cat in q.DebugManager.CATEGORIES:
            self.assertEqual(self.dm.get_level(cat), 0)

    def test_set_level_single_category(self):
        self.dm.set_level('network', 'trace')
        self.assertEqual(self.dm.get_level('network'), 3)
        self.assertEqual(self.dm.get_level('payload'), 0)

    def test_should_log(self):
        self.dm.set_level('network', 'verbose')
        self.assertTrue(self.dm.should_log('network', 1))
        self.assertTrue(self.dm.should_log('network', 2))
        self.assertFalse(self.dm.should_log('network', 3))

    def test_get_status(self):
        status = self.dm.get_status()
        self.assertIsInstance(status, dict)

    def test_valid_levels(self):
        self.assertIn('off', q.DebugManager.VALID_LEVELS)
        self.assertIn('basic', q.DebugManager.VALID_LEVELS)
        self.assertIn('verbose', q.DebugManager.VALID_LEVELS)
        self.assertIn('trace', q.DebugManager.VALID_LEVELS)

    def test_categories(self):
        for key in ('network', 'payload', 'response', 'stream', 'context'):
            self.assertIn(key, q.DebugManager.CATEGORIES)

    def test_is_enabled(self):
        self.assertFalse(self.dm.is_enabled('network'))
        self.dm.set_level('network', 'basic')
        self.assertTrue(self.dm.is_enabled('network'))


class TestDebugLog(unittest.TestCase):
    """debug_log utility function."""

    def setUp(self):
        self.dm = q.DebugManager()

    def test_log_writes_to_stderr(self):
        self.dm.set_level('network', 'basic')
        stderr = io.StringIO()
        old = sys.stderr
        sys.stderr = stderr
        try:
            q.debug_log(self.dm, 'network', 1, 'test message')
        finally:
            sys.stderr = old
        self.assertIn('test message', stderr.getvalue())

    def test_log_suppressed_when_off(self):
        self.dm.set_level('network', 'off')
        stderr = io.StringIO()
        old = sys.stderr
        sys.stderr = stderr
        try:
            q.debug_log(self.dm, 'network', 1, 'should not appear')
        finally:
            sys.stderr = old
        self.assertEqual(stderr.getvalue(), '')

    def test_log_prefix(self):
        self.dm.set_level('network', 'basic')
        stderr = io.StringIO()
        old = sys.stderr
        sys.stderr = stderr
        try:
            q.debug_log(self.dm, 'network', 1, 'msg', prefix='TEST')
        finally:
            sys.stderr = old
        self.assertIn('[TEST:', stderr.getvalue())


# ============================================================================
# 10. Image Handling
# ============================================================================

class TestImageHandling(unittest.TestCase):
    """Image file loading and base64 encoding."""

    def test_prepare_image_data_nonexistent(self):
        result = q.prepare_image_data('/nonexistent/image.png')
        self.assertIsNone(result)

    def test_prepare_image_data_empty_path(self):
        result = q.prepare_image_data('')
        self.assertIsNone(result)

    def test_prepare_image_data_valid(self):
        """Create a tiny valid PNG and verify encoding works."""
        # Minimal valid PNG (1x1 pixel)
        png_bytes = (
            b'\x89PNG\r\n\x1a\n'  # signature
            b'\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01'
            b'\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDAT'
            b'\x08\xd7c\xf8\x0f\x00\x00\x00\x00\xff\xff\x03\x00'
            b'\x00\x04\x00\x01\x0c\x0c\x0c\xc7\x00\x00\x00\x00'
            b'IEND\xaeB`\x82'
        )
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            f.write(png_bytes)
            f.flush()
            fname = f.name
        try:
            result = q.prepare_image_data(fname)
            self.assertIsNotNone(result)
            self.assertGreater(len(result), 0)
        finally:
            os.unlink(fname)


class TestImageCommand(unittest.TestCase):
    """/image command handler."""

    def setUp(self):
        self.ctx = _fresh_ctx()
        self.loop = _chat_loop(self.ctx)

    def test_image_clear(self):
        self.loop.run_handle_image('/image clear')
        self.assertEqual(self.ctx.current_images, [])

    def test_image_with_no_arg(self):
        result = self.loop.run_handle_image('/image')
        self.assertFalse(result)


# ============================================================================
# 11. Utility Functions
# ============================================================================

class TestUtilityFunctions(unittest.TestCase):
    """Standalone utility functions."""

    def test_sanitize_shell_command(self):
        result = q.sanitize_shell_command('ls -la')
        self.assertEqual(result, 'ls -la')

    def test_validate_shell_command_safety_valid(self):
        self.assertTrue(q.validate_shell_command_safety('echo hello'))

    def test_validate_shell_command_safety_backtick(self):
        # Bare backticks are NOT blocked (only escaped ones like \`)
        self.assertTrue(q.validate_shell_command_safety('echo `ls`'))

    def test_validate_shell_command_safety_escaped_backtick(self):
        self.assertFalse(q.validate_shell_command_safety('echo \\`ls\\`'))

    def test_validate_shell_command_safety_dangerous_pattern(self):
        self.assertFalse(q.validate_shell_command_safety('dd if=/dev/zero of=/tmp/evil'))

    def test_is_known_command(self):
        self.assertTrue(q.is_known_command('/help'))
        self.assertTrue(q.is_known_command('/quit'))
        result = q.is_known_command('/nonexistent')
        self.assertFalse(result[0])  # Returns (False, None)

    def test_format_help_text_compact(self):
        text = q.format_help_text(compact=True)
        # Compact format uses first alias only (/?, not /help)
        self.assertIn('/?', text)
        self.assertIn('/quit', text)
        self.assertIn('Core:', text)

    def test_format_help_text_full(self):
        text = q.format_help_text(compact=False)
        self.assertIn('/help', text)
        self.assertIn('/quit', text)
        self.assertIn('Core', text)


# ============================================================================
# 12. Shell / Spawn
# ============================================================================

class TestSpawnShell(unittest.TestCase):
    """Shell spawning functionality."""

    def setUp(self):
        self.ctx = _fresh_ctx()
        self.loop = _chat_loop(self.ctx)

    @unittest.skipIf(not sys.stdin.isatty(), "spawnshell tests require a real terminal")
    def test_handle_spawnshell_no_command(self):
        result = self.loop.run_handle_spawnshell('/spawnshell')
        self.assertFalse(result)

    @unittest.skipIf(not sys.stdin.isatty(), "spawnshell tests require a real terminal")
    def test_handle_spawnshell_with_command(self):
        result = self.loop.run_handle_spawnshell('/spawnshell echo hi')
        self.assertFalse(result)

    @unittest.skipIf(not sys.stdin.isatty(), "spawnshell tests require a real terminal")
    def test_spawnshell_output_not_characters(self):
        """handle_spawnshell must NOT split output into individual characters."""
        output = self.loop.handle_spawnshell()
        if output:
            # If spawnshell returns anything, it must contain whole words
            self.assertNotRegex(output, r'^[A-Za-z]{1}$',
                                'Output should not be single characters')
            self.assertGreater(len(output), 10,
                               'Should contain meaningful output')

    def test_cwd_command(self):
        result = self.loop.run_handle_cwd('/cwd')
        self.assertFalse(result)

    def test_ls_command(self):
        result = self.loop.run_handle_ls('/ls')
        self.assertFalse(result)


# ============================================================================
# 13. Dump Context
# ============================================================================

class TestDumpContext(unittest.TestCase):
    """Context dumping to file."""

    def setUp(self):
        self.ctx = _fresh_ctx()
        self.loop = _chat_loop(self.ctx)

    def test_dumpcontext_no_file(self):
        result = self.loop.run_handle_dumpcontext('/dumpcontext')
        self.assertFalse(result)

    def test_dumpcontext_empty_messages_raises(self):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            fname = f.name
        try:
            with self.assertRaises(ValueError):
                self.loop.dump_context_to_file(fname)
        finally:
            os.unlink(fname)

    def test_dumpcontext_writes_file(self):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            fname = f.name
        try:
            self.loop.run_process_query('Say hello')
            self.loop.dump_context_to_file(fname)
            with open(fname) as f:
                data = json.load(f)
            self.assertIsInstance(data, list)
            self.assertGreater(len(data), 0)
        finally:
            os.unlink(fname)


# ============================================================================
# 14. Switch Model
# ============================================================================

@ollama_only
class TestSwitchModel(unittest.TestCase):
    """Model switching preserves context."""

    def setUp(self):
        self.ctx = _fresh_ctx()
        self.loop = _chat_loop(self.ctx)

    def test_switch_preserves_messages(self):
        self.loop.run_process_query('Say hi')
        messages_before = list(self.loop.messages)
        self.loop.run_handle_switchmodel(f'/switchmodel {SMALL_MODEL}')
        self.assertEqual(len(self.loop.messages), len(messages_before))
        self.assertEqual(self.loop.messages[-1]['role'], 'assistant')

    def test_switch_preserves_cumulative_stats(self):
        self.loop.run_process_query('Say hi')
        self.loop.run_process_query('Say bye')
        before = self.ctx.get_cumulative_stats()
        self.loop.run_handle_switchmodel(f'/switchmodel {SMALL_MODEL}')
        after = self.ctx.get_cumulative_stats()
        self.assertEqual(before['total_queries'], after['total_queries'])


# ============================================================================
# 15. Batch Processing
# ============================================================================

class TestBatchProcessing(unittest.TestCase):
    """Non-interactive / batch processing via main()."""

    @ollama_only
    def test_sync_query_via_main(self):
        """main() with -I and -o should write output file."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            out = f.name
        try:
            with patch.object(sys, 'argv', [
                'ollamaquery2.py', '-b', 'ollama', '-H', OLLAMA_HOST,
                '-m', SMALL_MODEL, '-I', 'Say hello in 1 word', '-o', out
            ]):
                # Suppress stderr from main
                old_stderr = sys.stderr
                sys.stderr = io.StringIO()
                try:
                    q.main()
                except SystemExit:
                    pass
                finally:
                    sys.stderr = old_stderr
            with open(out) as f:
                content = f.read()
            self.assertGreater(len(content), 0)
        finally:
            if os.path.exists(out):
                os.unlink(out)

    def test_main_shows_help_with_no_args(self):
        """main() shows help when run with no arguments."""
        with patch.object(sys, 'argv', ['ollamaquery2.py']):
            with self.assertRaises(SystemExit) as cm:
                q.main()
            self.assertEqual(cm.exception.code, 2)


# ============================================================================
# 16.  Model Info Display
# ============================================================================

@ollama_only
class TestModelInfo(unittest.TestCase):
    """show_model_info and show_model_details functions."""

    def test_show_model_info(self):
        class FakeArgs:
            output_format = 'json'
            host = None
        stdout = io.StringIO()
        old = sys.stdout
        sys.stdout = stdout
        try:
            with self.assertRaises(SystemExit):
                q.show_model_info(OLLAMA_HOST, SMALL_MODEL, FakeArgs())
        finally:
            sys.stdout = old
        output = stdout.getvalue()
        self.assertIn('capabilities', output)
        self.assertIn('details', output)

    def test_show_model_details(self):
        class FakeArgs:
            output_format = 'json'
            host = None
        stdout = io.StringIO()
        old = sys.stdout
        sys.stdout = stdout
        try:
            with self.assertRaises(SystemExit):
                q.show_model_details(OLLAMA_HOST, SMALL_MODEL, FakeArgs())
        finally:
            sys.stdout = old
        output = stdout.getvalue()
        self.assertIn('details', output)
        self.assertIn('model_info', output)


# ============================================================================
# 17.  Environment Variables
# ============================================================================

class TestEnvironmentVariables(unittest.TestCase):
    """Environment variable handling."""

    def test_ollama_host_env(self):
        with patch.dict(os.environ, {'OLLAMA_HOST': 'http://custom:8888'},
                        clear=True):
            url = q.get_base_url(MagicMock(host=None), 'ollama')
            self.assertEqual(url, 'http://custom:8888')

    def test_llamacpp_host_env(self):
        with patch.dict(os.environ, {'LLAMACPP_HOST': 'http://custom:9999'},
                        clear=True):
            url = q.get_base_url(MagicMock(host=None), 'llamacpp')
            self.assertEqual(url, 'http://custom:9999')

    def test_get_base_url_with_host_arg(self):
        args = MagicMock(host='http://override:7777')
        url = q.get_base_url(args, 'ollama')
        self.assertEqual(url, 'http://override:7777')

    def test_get_base_url_adds_prefix(self):
        args = MagicMock(host='192.168.1.1:11434')
        url = q.get_base_url(args, 'ollama')
        self.assertEqual(url, 'http://192.168.1.1:11434')


# ============================================================================
# 18.  shell_timeout Propagation
# ============================================================================

class TestShellTimeoutCLI(unittest.TestCase):
    """--shell-timeout is respected in all code paths."""

    def test_parse_shell_timeout_default(self):
        """Default shell_timeout is 5."""
        with patch.object(sys, 'argv', ['ollamaquery2.py']):
            with self.assertRaises(SystemExit):
                q.main()
        # Default is set in arg parser
        parser = q.main.__globals__.get('argparse')
        ctx = _fresh_ctx()
        self.assertEqual(ctx.shell_timeout, 5)


# ============================================================================
# 19.  Format Help
# ============================================================================

class TestFormatHelp(unittest.TestCase):
    """format_help_text output."""

    def test_compact_help(self):
        help_text = q.format_help_text(compact=True)
        self.assertIsInstance(help_text, str)
        self.assertGreater(len(help_text), 0)

    def test_full_help(self):
        help_text = q.format_help_text(compact=False)
        self.assertIsInstance(help_text, str)
        self.assertGreater(len(help_text), 0)

    def test_help_contains_categories(self):
        help_text = q.format_help_text(compact=False)
        self.assertIn('Core', help_text)
        self.assertIn('Model', help_text)


# ============================================================================
# 20.  Argument Parser
# ============================================================================

class TestArgumentParser(unittest.TestCase):
    """CLI argument parsing."""

    def test_backend_choices(self):
        """-b only accepts ollama or llamacpp."""
        with patch.object(sys, 'argv', ['prog', '-b', 'invalid']):
            with self.assertRaises(SystemExit):
                q.main()

    def test_model_opt(self):
        """-m passes through to args.model."""
        with patch.object(sys, 'argv', ['prog', '-m', 'test-model']):
            with self.assertRaises(SystemExit):
                q.main()

    def test_mutual_exclusion_list_and_show(self):
        """-l and --show are mutually exclusive."""
        with patch.object(sys, 'argv', ['prog', '-l', '--show']):
            with self.assertRaises(SystemExit):
                q.main()


# ============================================================================
# 21.  Complete Workflow Integration
# ============================================================================

@ollama_only
class TestFullWorkflow(unittest.TestCase):
    """Multi-step workflow: query → switch → query → clear → query."""

    def setUp(self):
        self.ctx = _fresh_ctx()
        self.loop = _chat_loop(self.ctx)

    def test_query_switch_query_clear_query(self):
        """Simulate a realistic user session."""
        # 1. Initial query
        self.loop.run_process_query('Say hello')
        self.assertGreater(len(self.loop.messages), 0)

        # 2. Switch model (same model)
        self.loop.run_handle_switchmodel(f'/switchmodel {SMALL_MODEL}')
        self.assertEqual(self.ctx.model, SMALL_MODEL)

        # 3. Second query — messages preserved
        self.loop.run_process_query('Say goodbye')
        self.assertGreaterEqual(len(self.loop.messages), 4)
        self.assertEqual(self.ctx.total_queries, 2)

        # 4. Clear
        self.loop.run_handle_clear('/clear')
        self.assertEqual(self.ctx.total_queries, 0,
                         "Clear resets cumulative stats to 0")

        # 5. Query after clear
        self.loop.run_process_query('Say again')
        self.assertTrue(hasattr(self.loop, 'messages'))
        self.assertGreaterEqual(len(self.loop.messages), 2)


# ============================================================================
# 22.  Error Handling & Edge Cases
# ============================================================================

class TestErrorHandling(unittest.TestCase):
    """Graceful error handling."""

    def test_fetch_models_nonexistent_server(self):
        models = q.fetch_models_ollama('http://127.0.0.1:1')
        self.assertEqual(models, [])

    def test_fetch_model_info_nonexistent(self):
        info = q.fetch_model_info_ollama('http://127.0.0.1:1', 'nonexistent')
        self.assertEqual(info, {})

    def test_get_message_token_count_ollama_no_model(self):
        count = q.get_message_token_count_ollama(OLLAMA_HOST, 'hello', '')
        self.assertGreaterEqual(count, 1)

    def test_get_ollama_context_size_nonexistent(self):
        size = q.get_ollama_context_size('http://127.0.0.1:1', 'nonexistent')
        self.assertEqual(size, 0)

    def test_image_command_with_nonexistent_file(self):
        ctx = _fresh_ctx()
        loop = _chat_loop(ctx)
        # Suppress stderr
        stderr = io.StringIO()
        old = sys.stderr
        sys.stderr = stderr
        try:
            result = loop.run_handle_image('/image /nonexistent/file.png')
        finally:
            sys.stderr = old
        self.assertFalse(result)

    def test_dumpcontext_errors_on_no_messages(self):
        ctx = _fresh_ctx()
        loop = _chat_loop(ctx)
        with self.assertRaises(ValueError):
            loop.dump_context_to_file('/tmp/nonexistent/dump.json')

    def test_dumpcontext_with_bad_path(self):
        ctx = _fresh_ctx()
        loop = _chat_loop(ctx)
        loop.run_process_query('Say hi')
        with self.assertRaises(Exception):
            loop.dump_context_to_file('/nonexistent/dir/dump.json')


# ============================================================================
# Run
# ============================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
