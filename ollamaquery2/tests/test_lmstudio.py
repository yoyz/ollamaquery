#!/usr/bin/env python3
"""LM Studio-specific tests for ollamaquery2.

Tests LM Studio API quirks: reasoning field, model listing format,
OpenAI-compatible chat completions, context size fallback.

Requires LM Studio running (default: http://127.0.0.1:1234).

Usage:
  LMSTUDIO_HOST=http://192.168.1.11:1234 python3 -m unittest tests.test_lmstudio -v
"""

import os
import sys
import json
import unittest
from urllib.request import Request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ollamaquery2 as q

HOST = os.environ.get('LMSTUDIO_HOST', 'http://127.0.0.1:1234')


def _is_reachable():
    try:
        req = Request(HOST, method='HEAD')
        q.urlopen(req, timeout=2)
        models = q.fetch_models_llamacpp(HOST)
        return bool(models)
    except Exception:
        return False


HAS_LMSTUDIO = _is_reachable()
MODELS = q.fetch_models_llamacpp(HOST) if HAS_LMSTUDIO else []
FIRST_MODEL = MODELS[0]['name'] if MODELS else None


def setUpModule():
    if HAS_LMSTUDIO:
        print(f"\n[LM Studio @ {HOST}]")
        print(f"[Models: {len(MODELS)}, selected: {FIRST_MODEL}]")


class _LMStudioTestBase(unittest.TestCase):
    """Base class that skips if LM Studio is unreachable."""

    def setUp(self):
        if not HAS_LMSTUDIO:
            self.skipTest("LM Studio not available")


class TestLmStudioDetection(_LMStudioTestBase):
    """Verify LM Studio is reachable and returns models."""

    def test_server_reachable(self):
        self.assertTrue(HAS_LMSTUDIO)

    def test_models_found(self):
        self.assertGreater(len(MODELS), 0)

    def test_model_id_format(self):
        """Most LM Studio model IDs have vendor prefixes like 'openai/gpt-oss-20b'."""
        has_prefix = sum(1 for m in MODELS if '/' in m['name'])
        self.assertGreater(has_prefix, 0, "At least one model should have a vendor prefix")
        # Embedding models may be an exception (no vendor prefix)


class TestLmStudioModelList(_LMStudioTestBase):
    """Model listing via /v1/models."""

    def test_fetch_returns_list(self):
        models = q.fetch_models_llamacpp(HOST)
        self.assertIsInstance(models, list)
        self.assertGreater(len(models), 0)

    def test_each_model_has_name(self):
        models = q.fetch_models_llamacpp(HOST)
        for m in models:
            self.assertIn('name', m)
            self.assertTrue(m['name'])

    def test_list_models_runs(self):
        """list_models_llamacpp should print to stdout without error."""
        import io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            q.list_models_llamacpp(HOST)
            output = sys.stdout.getvalue()
            self.assertIn('NAME', output)
            self.assertIn(FIRST_MODEL, output)
        finally:
            sys.stdout = old_stdout


class TestLmStudioQuery(_LMStudioTestBase):
    """OpenAI-compatible chat completions."""

    def setUp(self):
        super().setUp()
        q.CommandContext._instance = None
        q.CommandContext._initialized = False
        self.ctx = q.CommandContext()
        self.ctx.base_url = HOST
        self.ctx.backend = 'lmstudio'
        self.ctx.model = FIRST_MODEL
        self.ctx.context_window_size = 32768

    def test_sync_query_returns_choices(self):
        """LM Studio returns OpenAI format with choices array."""
        mq = q.ModelQuery(HOST, 'lmstudio', context=self.ctx)
        result = mq.query_sync(
            [{'role': 'user', 'content': 'Say hello in 1 word'}],
            FIRST_MODEL)
        self.assertIsInstance(result, dict)
        # LM Studio uses OpenAI format: choices[0].message.content
        self.assertIn('choices', result,
                      "LM Studio should return 'choices' array (OpenAI format)")
        choices = result.get('choices', [])
        self.assertGreater(len(choices), 0)
        msg = choices[0].get('message', {})
        self.assertIn('content', msg)
        self.assertGreater(len(msg['content']), 0)

    def test_query_has_finish_reason(self):
        mq = q.ModelQuery(HOST, 'lmstudio', context=self.ctx)
        result = mq.query_sync(
            [{'role': 'user', 'content': 'Say hello'}],
            FIRST_MODEL)
        choices = result.get('choices', [])
        self.assertIn('finish_reason', choices[0])

    def test_query_has_usage(self):
        mq = q.ModelQuery(HOST, 'lmstudio', context=self.ctx)
        result = mq.query_sync(
            [{'role': 'user', 'content': 'Say hello'}],
            FIRST_MODEL)
        usage = result.get('usage', {})
        self.assertIn('prompt_tokens', usage)
        self.assertIn('completion_tokens', usage)

    def test_reasoning_field(self):
        """LM Studio returns 'reasoning' (not 'reasoning_content')."""
        mq = q.ModelQuery(HOST, 'lmstudio', context=self.ctx)
        result = mq.query_sync(
            [{'role': 'user', 'content': 'Explain why the sky is blue in 5 words'}],
            FIRST_MODEL)
        choices = result.get('choices', [])
        if choices:
            msg = choices[0].get('message', {})
            reasoning = msg.get('reasoning')
            if reasoning:
                self.assertGreater(len(reasoning), 0)
            # reasoning_content should NOT be present (LM Studio doesn't use it)
            # Note: some models may not output reasoning at all, so this is optional

    def test_inference_params_passed(self):
        """LM Studio accepts temperature, top_p, but not top_k/min_p."""
        mq = q.ModelQuery(HOST, 'lmstudio', context=self.ctx)
        result = mq.query_sync(
            [{'role': 'user', 'content': 'Say hello'}],
            FIRST_MODEL,
            temperature=0.7, top_p=0.9)
        self.assertIsInstance(result, dict)
        # Just verify no crash — LM Studio ignores unknown params

    def test_streaming_returns_content(self):
        """Start a stream and verify content arrives."""
        mq = q.ModelQuery(HOST, 'lmstudio', context=self.ctx)
        import io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            result = mq.query_stream(
                [{'role': 'user', 'content': 'Count to 3'}],
                FIRST_MODEL,
                stream_enabled=True)
            self.assertIsNotNone(result)
            self.assertGreater(len(result), 0)
        finally:
            sys.stdout = old_stdout

    def test_chat_loop_processes_query(self):
        loop = q.ChatLoop(self.ctx)
        loop.run_process_query('Count to 3 slowly')
        self.assertTrue(hasattr(loop, 'messages'))
        self.assertGreaterEqual(len(loop.messages), 2)
        self.assertIn(loop.messages[-1]['role'], ('assistant', 'user'),
                      "Should have assistant response or pending response")
        if loop.messages[-1]['role'] == 'assistant':
            self.assertGreater(len(loop.messages[-1]['content']), 0,
                               "Assistant response should have content")


class TestLmStudioUtils(_LMStudioTestBase):
    """Utility function behavior with LM Studio."""

    def test_no_context_size_endpoint(self):
        """refresh_context_window_size should return 0 for LM Studio."""
        q.CommandContext._instance = None
        q.CommandContext._initialized = False
        ctx = q.CommandContext()
        ctx.base_url = HOST
        ctx.backend = 'lmstudio'
        result = q.refresh_context_window_size(ctx)
        self.assertFalse(result)
        self.assertEqual(ctx.context_window_size, 0)

    def test_fallback_to_estimate_token_count(self):
        """LM Studio has no tokenize endpoint, uses estimation fallback."""
        text = "hello world " * 100
        estimated = q.estimate_token_count(text)
        self.assertGreater(estimated, 0)
