#!/usr/bin/env python3
"""Release-only tests for ollamaquery2 context bar and context window tracking.

These tests verify that changing the context window size produces correct
context bar output. They are gated behind RELEASE_TEST=1 to avoid adding
overhead to daily test runs.

Environment:
  RELEASE_TEST=1 — enable these tests
  OLLAMA_HOST   — optional Ollama URL for integration tests

Usage:
  RELEASE_TEST=1 python3 -m unittest tests.test_release -v
"""

import io
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ollamaquery2 as q

RUN_RELEASE = os.environ.get('RELEASE_TEST') == '1'
release_only = unittest.skipUnless(RUN_RELEASE, 'Set RELEASE_TEST=1 to run')


# ============================================================================
# Pure unit tests — context_bar() rendering
# ============================================================================


@release_only
class TestContextBarRendering(unittest.TestCase):
    """Verify context_bar() produces correct output for various values."""

    def test_zero_window_returns_empty(self):
        self.assertEqual(q.context_bar(100, 0), '')

    def test_zero_current(self):
        bar = q.context_bar(0, 32768)
        self.assertIn('0/32768', bar)
        self.assertIn('(0%)', bar)
        self.assertIn('[', bar)
        self.assertIn(']', bar)

    def test_half_full(self):
        bar = q.context_bar(16384, 32768)
        self.assertIn('16384/32768', bar)
        self.assertIn('(50%)', bar)

    def test_full(self):
        bar = q.context_bar(32768, 32768)
        self.assertIn('32768/32768', bar)
        self.assertIn('(100%)', bar)

    def test_overflow_clamps(self):
        bar = q.context_bar(99999, 32768)
        self.assertIn('99999/32768', bar)
        self.assertIn('(100%)', bar)
        self.assertNotIn('305%', bar)

    def test_custom_width(self):
        bar = q.context_bar(8192, 32768, width=40)
        # Filled bar should be ~10 chars
        self.assertIn('[', bar)
        self.assertIn(']', bar)

    def test_small_window(self):
        bar = q.context_bar(1, 4)
        self.assertIn('(25%)', bar)

    def test_usage_tracking_across_range(self):
        for pct in (0, 10, 25, 50, 75, 90, 100):
            current = int(pct * 32768 / 100)
            bar = q.context_bar(max(1, current), 32768)
            self.assertIn(f'({pct}%)' if pct > 0 else '(0%)', bar)

    def test_unicode_bar_chars_present(self):
        bar = q.context_bar(16384, 32768)
        self.assertTrue('█' in bar or '░' in bar)


# ============================================================================
# Unit tests — _extract_context_size_from_show
# ============================================================================


@release_only
class TestExtractContextSizeFromShow(unittest.TestCase):
    """Verify /api/show response parsing for context size."""

    def test_architecture_specific_key(self):
        data = {
            "model_info": {
                "general.architecture": "mistral3",
                "mistral3.context_length": 65535,
                "llama.context_length": 262144,
            }
        }
        self.assertEqual(q._extract_context_size_from_show(data), 65535)

    def test_fallback_to_any_context_length(self):
        data = {
            "model_info": {
                "general.architecture": "unknown",
                "llama.context_length": 4096,
            }
        }
        self.assertEqual(q._extract_context_size_from_show(data), 4096)

    def test_no_architecture_key(self):
        data = {
            "model_info": {
                "qwen2.context_length": 32768,
            }
        }
        self.assertEqual(q._extract_context_size_from_show(data), 32768)

    def test_modelfile_num_ctx(self):
        data = {
            "modelfile": "FROM llama2\nPARAMETER num_ctx 8192\nPARAMETER temperature 0.7\n"
        }
        self.assertEqual(q._extract_context_size_from_show(data), 8192)

    def test_parameters_string(self):
        data = {
            "parameters": "num_ctx 16384\nnum_keep 24\n"
        }
        self.assertEqual(q._extract_context_size_from_show(data), 16384)

    def test_empty_model_info(self):
        data = {"model_info": {}}
        self.assertEqual(q._extract_context_size_from_show(data), 0)

    def test_missing_model_info(self):
        data = {}
        self.assertEqual(q._extract_context_size_from_show(data), 0)

    def test_legacy_context_size_field(self):
        data = {"context_size": 2048}
        self.assertEqual(q._extract_context_size_from_show(data), 2048)

    def test_zero_values_skipped(self):
        data = {
            "model_info": {
                "general.architecture": "llama",
                "llama.context_length": 0,
            },
            "modelfile": "FROM llama\nPARAMETER num_ctx 4096\n"
        }
        self.assertEqual(q._extract_context_size_from_show(data), 4096)

    def test_prefers_arch_over_modelfile(self):
        data = {
            "model_info": {
                "general.architecture": "llama",
                "llama.context_length": 8192,
            },
            "modelfile": "FROM llama\nPARAMETER num_ctx 4096\n"
        }
        self.assertEqual(q._extract_context_size_from_show(data), 8192)

    def test_prefers_arch_over_legacy(self):
        """Architecture key beats top-level context_size field."""
        data = {
            "model_info": {
                "general.architecture": "llama",
                "llama.context_length": 16384,
            },
            "context_size": 4096,
        }
        self.assertEqual(q._extract_context_size_from_show(data), 16384)

    def test_float_values_accepted(self):
        data = {
            "model_info": {
                "general.architecture": "llama",
                "llama.context_length": 32768.0,
            }
        }
        self.assertEqual(q._extract_context_size_from_show(data), 32768)


# ============================================================================
# Unit tests — _get_ollama_ps_context_size
# ============================================================================


@release_only
class TestGetOllamaPsContextSize(unittest.TestCase):
    """Verify /api/ps response parsing for context_size / context_length."""

    def test_context_size_field(self):
        models = [{"name": "llama3:8b", "context_size": 8192}]
        self.assertEqual(q._get_ollama_ps_context_size(models, "llama3:8b"), 8192)

    def test_context_length_field(self):
        models = [{"name": "ministral-3:8b", "context_length": 65535}]
        self.assertEqual(q._get_ollama_ps_context_size(models, "ministral-3:8b"), 65535)

    def test_context_length_preferred_when_both(self):
        """context_size takes priority (first checked)."""
        models = [{"name": "test:latest", "context_size": 4096, "context_length": 8192}]
        self.assertEqual(q._get_ollama_ps_context_size(models, "test:latest"), 4096)

    def test_model_not_found(self):
        models = [{"name": "other:latest", "context_size": 4096}]
        self.assertEqual(q._get_ollama_ps_context_size(models, "missing:latest"), 0)

    def test_empty_model_list(self):
        self.assertEqual(q._get_ollama_ps_context_size([], "any"), 0)

    def test_zero_values_ignored(self):
        models = [{"name": "test:latest", "context_size": 0, "context_length": 0}]
        self.assertEqual(q._get_ollama_ps_context_size(models, "test:latest"), 0)

    def test_zero_with_positive_in_other(self):
        models = [{"name": "test:latest", "context_size": 0, "context_length": 16384}]
        self.assertEqual(q._get_ollama_ps_context_size(models, "test:latest"), 16384)

    def test_prefix_match(self):
        models = [{"name": "llama3:8b", "context_size": 8192}]
        self.assertEqual(q._get_ollama_ps_context_size(models, "llama3"), 8192)

    def test_non_dict_entries_skipped(self):
        models = [{"name": "valid", "context_size": 4096}, "not_a_dict"]
        self.assertEqual(q._get_ollama_ps_context_size(models, "valid"), 4096)


# ============================================================================
# Integration tests — run_display_context_bar output
# ============================================================================


@release_only
class TestDisplayContextBarIntegration(unittest.TestCase):
    """Verify run_display_context_bar produces correct output.

    These tests create a ChatLoop, set context_window_size and
    current_context_tokens directly, then capture stderr to
    verify the bar content.
    """

    def setUp(self):
        q.CommandContext._instance = None
        q.CommandContext._initialized = False
        self.ctx = q.CommandContext()
        self.ctx.backend = 'ollama'
        self.ctx.model = 'test:latest'
        self.ctx.base_url = 'http://localhost:11434'
        self.ctx.system_prompt = 'Reply in 1 word.'
        self.loop = q.ChatLoop(self.ctx)

    def test_bar_missing_when_window_size_zero(self):
        self.ctx.context_window_size = 0
        self.ctx.current_context_tokens = 500
        stderr = io.StringIO()
        with patch('sys.stderr', stderr):
            self.loop.run_display_context_bar()
        output = stderr.getvalue()
        self.assertEqual(output, '')

    def test_bar_shows_when_window_size_positive(self):
        self.ctx.context_window_size = 32768
        self.ctx.current_context_tokens = 8192
        stderr = io.StringIO()
        with patch('sys.stderr', stderr):
            self.loop.run_display_context_bar()
        output = stderr.getvalue()
        self.assertIn('8192/32768', output)
        self.assertIn('(25%)', output)

    def test_bar_with_different_windows(self):
        cases = [
            (4096, 1024, '1024/4096'),
            (8192, 4096, '4096/8192'),
            (65535, 1000, '1000/65535'),
            (262144, 500, '500/262144'),
        ]
        for window, used, expected in cases:
            self.ctx.context_window_size = window
            self.ctx.current_context_tokens = used
            stderr = io.StringIO()
            with patch('sys.stderr', stderr):
                self.loop.run_display_context_bar()
            self.assertIn(expected, stderr.getvalue(),
                          f'Expected {expected} in bar for {used}/{window}')

    def test_context_full_warning_at_80(self):
        self.ctx.context_window_size = 1000
        self.ctx.current_context_tokens = 800
        with patch('sys.stderr', new_callable=io.StringIO) as mock_stderr:
            self.loop.run_display_context_bar()
        output = mock_stderr.getvalue()
        self.assertIn('Context almost full', output)

    def test_context_warning_at_60(self):
        self.ctx.context_window_size = 1000
        self.ctx.current_context_tokens = 650
        with patch('sys.stderr', new_callable=io.StringIO) as mock_stderr:
            self.loop.run_display_context_bar()
        output = mock_stderr.getvalue()
        self.assertIn('Context getting full', output)
        self.assertNotIn('almost full', output)

    def test_no_warning_below_60(self):
        self.ctx.context_window_size = 1000
        self.ctx.current_context_tokens = 500
        with patch('sys.stderr', new_callable=io.StringIO) as mock_stderr:
            self.loop.run_display_context_bar()
        output = mock_stderr.getvalue()
        self.assertNotIn('Context almost full', output)
        self.assertNotIn('Context getting full', output)

    def test_bar_updates_after_set_context_size(self):
        """Simulate /contextsizeset then verify bar shows the new window."""
        self.ctx.context_window_size = 65535
        self.ctx.current_context_tokens = 1000
        stderr = io.StringIO()
        with patch('sys.stderr', stderr):
            self.loop.run_display_context_bar()
        output = stderr.getvalue()
        self.assertIn('1000/65535', output)

    def test_context_tokens_tracked_after_query(self):
        """After a query updates current_context_tokens, bar reflects it."""
        self.ctx.context_window_size = 32768
        self.ctx.current_context_tokens = 5000
        stderr = io.StringIO()
        with patch('sys.stderr', stderr):
            self.loop.run_display_context_bar()
        output = stderr.getvalue()
        self.assertIn('5000/32768', output)

    def test_window_change_updates_bar(self):
        """Changing context_window_size produces a different bar."""
        self.ctx.current_context_tokens = 1000
        self.ctx.context_window_size = 4096
        with patch('sys.stderr', new_callable=io.StringIO) as stderr:
            self.loop.run_display_context_bar()
        bar_4k = stderr.getvalue()

        self.ctx.context_window_size = 32768
        with patch('sys.stderr', new_callable=io.StringIO) as stderr:
            self.loop.run_display_context_bar()
        bar_32k = stderr.getvalue()

        self.assertIn('1000/4096', bar_4k)
        self.assertIn('1000/32768', bar_32k)
        self.assertNotEqual(bar_4k, bar_32k)


# ============================================================================
# Integration tests — set_context_size interaction with context bar
# ============================================================================


@release_only
class TestSetContextSizeIntegration(unittest.TestCase):
    """Verify /contextsizeset updates context_size and bar reflects changes."""

    def setUp(self):
        q.CommandContext._instance = None
        q.CommandContext._initialized = False
        self.ctx = q.CommandContext()
        self.ctx.backend = 'ollama'
        self.ctx.model = 'test:latest'
        self.ctx.base_url = 'http://localhost:11434'
        self.ctx.system_prompt = 'Reply in 1 word.'
        self.loop = q.ChatLoop(self.ctx)

    def test_set_context_size_changes_context_size(self):
        self.loop.set_context_size('/contextsizeset 65535')
        self.assertEqual(self.ctx.context_size, 65535)

    def test_set_context_size_zero_resets_to_default(self):
        self.ctx.context_size = 9999
        self.loop.set_context_size('/contextsizeset 0')
        self.assertIsNone(self.ctx.context_size)

    def test_set_context_size_displayed_in_bar(self):
        """Setting context_size via /contextsizeset should not break bar display
        when context_window_size is also set."""
        self.ctx.context_window_size = 65535
        self.ctx.current_context_tokens = 100
        self.loop.set_context_size('/contextsizeset 65535')
        stderr = io.StringIO()
        with patch('sys.stderr', stderr):
            self.loop.run_display_context_bar()
        self.assertIn('100/65535', stderr.getvalue())

    def test_set_context_size_preserves_bar_after_clear(self):
        """After /clear, refresh_context_window_size runs. If it can't fetch,
        context_window_size stays 0 and bar hides. If it can, bar returns."""
        self.ctx.context_window_size = 32768
        self.ctx.current_context_tokens = 500
        self.ctx.reset()
        self.assertEqual(self.ctx.context_window_size, 0)
        stderr = io.StringIO()
        with patch('sys.stderr', stderr):
            self.loop.run_display_context_bar()
        self.assertEqual(stderr.getvalue(), '',
                         'Bar should be empty after reset when window_size=0')


# ============================================================================
# Integration tests — full main loop context bar cycle (live backend needed)
# ============================================================================


_HAS_OLLAMA = False
try:
    _HAS_OLLAMA = q.check_backend_with_get(
        os.environ.get('OLLAMA_HOST', 'http://127.0.0.1:11434'), 'ollama')
except Exception:
    pass

needs_ollama = unittest.skipUnless(RUN_RELEASE and _HAS_OLLAMA,
                                   'RELEASE_TEST=1 and Ollama backend needed')


@needs_ollama
class TestContextBarEndToEnd(unittest.TestCase):
    """End-to-end tests requiring a live Ollama backend.

    Verifies the full chain: query → _update_context_tokens →
    run_display_context_bar with correct values.
    """

    def setUp(self):
        q.CommandContext._instance = None
        q.CommandContext._initialized = False
        self.ctx = q.CommandContext()
        self.ctx.base_url = os.environ.get('OLLAMA_HOST', 'http://127.0.0.1:11434')
        self.ctx.backend = 'ollama'
        self.ctx.model = os.environ.get('TEST_MODEL', 'granite4:350m')
        self.ctx.system_prompt = 'Reply in 1 word.'
        self.ctx.context_window_size = 32768
        self.ctx.stream_enabled = False
        self.loop = q.ChatLoop(self.ctx)
        self._stdout_patch = patch('sys.stdout', io.StringIO())
        self._stdout_patch.__enter__()
        self._stderr_patch = patch('sys.stderr', io.StringIO())
        self._stderr_patch.__enter__()

    def tearDown(self):
        self._stderr_patch.__exit__(None, None, None)
        self._stdout_patch.__exit__(None, None, None)

    def test_context_tokens_increase_after_query(self):
        before = self.ctx.current_context_tokens
        self.loop.run_process_query('Say hello')
        after = self.ctx.current_context_tokens
        self.assertGreater(after, before)

    def test_bar_renders_after_query(self):
        self.loop.run_process_query('Say hello')
        self.assertGreater(self.ctx.current_context_tokens, 0)
        stderr = io.StringIO()
        with patch('sys.stderr', stderr):
            self.loop.run_display_context_bar()
        output = stderr.getvalue()
        self.assertIn(str(self.ctx.current_context_tokens), output,
                      'Bar should contain current_context_tokens')

    def test_bar_updates_after_multiple_queries(self):
        self.loop.run_process_query('First')
        first_tokens = self.ctx.current_context_tokens
        self.assertGreater(first_tokens, 0)
        self.loop.run_process_query('Second')
        second_tokens = self.ctx.current_context_tokens
        self.assertGreater(second_tokens, 0)
        stderr = io.StringIO()
        with patch('sys.stderr', stderr):
            self.loop.run_display_context_bar()
        self.assertIn(f'{second_tokens}/', stderr.getvalue())

    def test_query_updates_context_tokens_and_stats(self):
        self.loop.run_process_query('Count to 3')
        self.assertGreater(self.ctx.current_context_tokens, 0)
        cum = self.ctx.get_cumulative_stats()
        self.assertGreater(cum['total_completion_tokens'], 0)
        self.assertGreater(cum['total_prompt_tokens'], 0)


# ============================================================================
# Entry point
# ============================================================================


if __name__ == '__main__':
    unittest.main()
