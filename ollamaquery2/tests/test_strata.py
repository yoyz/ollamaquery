#!/usr/bin/env python3
"""Strata-specific tests for ollamaquery2.

Strata is an OpenAI-compatible, llama.cpp-style server: streaming deltas carry
`reasoning_content`, usage/timings use llama.cpp's names, and it exposes
Anthropic's `/v1/messages/count_tokens`. It uniquely identifies itself with
`"service": "strata"` in the `/health` body (its `Server` header is generic),
which is what `check_strata()` keys off.

Offline tests always run; live tests skip unless a Strata server is reachable.

Usage:
  STRATA_HOST=http://127.0.0.1:8080 python3 -m unittest tests.test_strata -v
"""

import os
import sys
import json
import threading
import unittest
from unittest import mock
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ollamaquery2 as q

HOST = os.environ.get('STRATA_HOST', q.DEFAULT_STRATA_HOST)


def _is_reachable():
    return q.check_strata(HOST)


HAS_STRATA = _is_reachable()
MODELS = q.fetch_models_llamacpp(HOST) if HAS_STRATA else []
FIRST_MODEL = MODELS[0]['name'] if MODELS else None


def setUpModule():
    if HAS_STRATA:
        print(f"\n[Strata @ {HOST}]")
        print(f"[Models: {len(MODELS)}, selected: {FIRST_MODEL}]")


class _StrataTestBase(unittest.TestCase):
    """Base class that skips if Strata is unreachable."""

    def setUp(self):
        if not HAS_STRATA:
            self.skipTest("Strata not available")


class _HealthHandler(BaseHTTPRequestHandler):
    """Minimal /health responder whose `service` marker is set per test."""

    service = "strata"

    def do_GET(self):
        body = json.dumps({"status": "ok", "service": self.service}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class TestStrataDetectionOffline(unittest.TestCase):
    """check_strata must key off the /health `service` marker only."""

    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), _HealthHandler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def test_detects_strata(self):
        _HealthHandler.service = "strata"
        self.assertTrue(q.check_strata(f"http://127.0.0.1:{self.port}"))

    def test_rejects_non_strata(self):
        _HealthHandler.service = "llama.cpp"
        self.assertFalse(q.check_strata(f"http://127.0.0.1:{self.port}"))

    def test_rejects_unreachable(self):
        self.assertFalse(q.check_strata("http://127.0.0.1:1"))


class TestStrataSavedConfigShadowing(unittest.TestCase):
    """A stale lmstudio@8080 entry must not shadow the strata entry."""

    def _args(self):
        return type('Args', (), {'backend': None, 'api_key': None})()

    def test_lmstudio_entry_skipped_when_host_is_strata(self):
        saved = [{"backend": "lmstudio", "host": "http://127.0.0.1:8080"},
                 {"backend": "strata", "host": "http://127.0.0.1:8080"}]
        with mock.patch.object(q, '_probe_backend', return_value=True), \
                mock.patch.object(q, 'check_strata', return_value=True), \
                mock.patch.object(q, 'save_backend_config'):
            self.assertEqual(q._probe_saved_backends(self._args(), saved),
                             ("strata", "http://127.0.0.1:8080"))

    def test_lmstudio_entry_kept_when_not_strata(self):
        saved = [{"backend": "lmstudio", "host": "http://127.0.0.1:1234"}]
        with mock.patch.object(q, '_probe_backend', return_value=True), \
                mock.patch.object(q, 'check_strata', return_value=False), \
                mock.patch.object(q, 'save_backend_config'):
            self.assertEqual(q._probe_saved_backends(self._args(), saved),
                             ("lmstudio", "http://127.0.0.1:1234"))


class TestStrataChunkParsing(unittest.TestCase):
    """Strata reuses llama.cpp's wire format, including reasoning_content."""

    def setUp(self):
        q.CommandContext._instance = None
        q.CommandContext._initialized = False
        self.ctx = q.CommandContext()
        self.ctx.base_url = HOST
        self.ctx.backend = 'strata'
        self.ctx.model = 'test-model'
        self.mq = q.ModelQuery(context=self.ctx)

    def test_reasoning_content_is_thinking(self):
        chunk = {"choices": [{"delta": {"reasoning_content": "thinking..."}, "finish_reason": None}]}
        thought, content, is_final, usage, tool_calls = self.mq._parse_chunk(chunk, 'strata')
        self.assertEqual(thought, "thinking...")
        self.assertEqual(content, "")
        self.assertFalse(is_final)

    def test_content_delta(self):
        chunk = {"choices": [{"delta": {"content": "hello"}, "finish_reason": None}]}
        thought, content, is_final, usage, tool_calls = self.mq._parse_chunk(chunk, 'strata')
        self.assertEqual(content, "hello")
        self.assertEqual(thought, "")

    def test_final_usage_from_llamacpp_timings(self):
        chunk = {
            "choices": [{"delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "timings": {"prompt_n": 10, "predicted_n": 5, "prompt_ms": 100.0,
                        "predicted_ms": 200.0, "prompt_per_second": 100.0,
                        "predicted_per_second": 25.0},
        }
        thought, content, is_final, usage, tool_calls = self.mq._parse_chunk(chunk, 'strata')
        self.assertTrue(is_final)
        self.assertEqual(usage["prompt_tokens"], 10)
        self.assertEqual(usage["completion_tokens"], 5)
        self.assertEqual(usage["predicted_per_second"], 25.0)

    def test_iter_stream_lines_strips_sse(self):
        lines = [b'data: {"a": 1}\n', b'data: [DONE]\n']
        out = list(self.mq._iter_stream_lines(lines, 'strata'))
        self.assertEqual(out, ['{"a": 1}'])

    def test_chat_url_is_openai(self):
        self.assertTrue(self.mq._get_chat_url('strata').endswith('/v1/chat/completions'))

    def test_payload_no_thinking(self):
        payload = self.mq.build_request_payload(
            [{"role": "user", "content": "hi"}], "m", no_thinking=True)
        self.assertIs(payload["chat_template_kwargs"].get("enable_thinking"), False)

    def test_payload_reasoning_effort(self):
        payload = self.mq.build_request_payload(
            [{"role": "user", "content": "hi"}], "m", reasoning_effort="low")
        self.assertEqual(payload["chat_template_kwargs"].get("reasoning_effort"), "low")

    def test_payload_inference_params_include_top_k(self):
        payload = self.mq.build_request_payload(
            [{"role": "user", "content": "hi"}], "m", temperature=0.6, top_k=20)
        self.assertEqual(payload.get("temperature"), 0.6)
        self.assertEqual(payload.get("top_k"), 20)


class TestStrataLive(_StrataTestBase):
    """Live behavior against a running Strata server."""

    def setUp(self):
        super().setUp()
        q.CommandContext._instance = None
        q.CommandContext._initialized = False
        self.ctx = q.CommandContext()
        self.ctx.base_url = HOST
        self.ctx.backend = 'strata'
        self.ctx.model = FIRST_MODEL

    def test_detected(self):
        self.assertTrue(HAS_STRATA)

    def test_models_found(self):
        self.assertGreater(len(MODELS), 0)

    def test_context_size_from_slots(self):
        self.assertGreater(q.get_llamacpp_context_size(HOST), 0)

    def test_token_count_endpoint(self):
        self.assertGreater(q.get_message_token_count_strata(HOST, "hello world"), 0)

    def test_sync_query_returns_openai_shape(self):
        mq = q.ModelQuery(context=self.ctx)
        result = mq.query_sync(
            [{"role": "user", "content": "Say hello in one word"}],
            FIRST_MODEL, reasoning_effort="low")
        choices = result.get("choices", [])
        self.assertTrue(choices, "Strata should return an OpenAI choices array")
        message = choices[0].get("message", {})
        self.assertIn("content", message)

    def test_reasoning_content_parsed_from_stream(self):
        mq = q.ModelQuery(context=self.ctx)
        thought, content, is_final, usage, tool_calls = mq._parse_chunk(
            {"choices": [{"delta": {"reasoning_content": "x"}, "finish_reason": None}]}, 'strata')
        self.assertEqual(thought, "x")


if __name__ == "__main__":
    unittest.main()
