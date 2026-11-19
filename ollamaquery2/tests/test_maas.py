#!/usr/bin/env python3
"""MaaS gateway tests for ollamaquery2 (B1: the offline-testable core).

The target is an OpenShift AI / KServe gateway in front of vLLM. Measured
behaviour (doc/maas-handoff/INTEGRATION_BRIEF.md):

  - only `/v1/models` and `/v1/chat/completions` are routed;
  - chat errors are bare codes with no body; only 200 is trusted;
  - `/v1/models` answers 403 / 429 / 500 / 200-with-empty-inventory in bursts,
    with no Retry-After, so the backoff is local;
  - 401 is definitive; a short model name is a silent 404, so it is resolved
    client-side against the listing.

Everything here runs offline: the retry policy is driven by an injected
transport, and one test exercises the real urllib transport against a local
HTTP server to pin the URL, the Bearer header and the response shape.

Usage:
  python3 -m unittest tests.test_maas -v
"""

import json
import os
import sys
import threading
import unittest
import argparse
from unittest import mock
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ollamaquery2 as q

MODELS_BODY = json.dumps({
    "object": "list",
    "data": [
        {"id": "publishers/acme/models/glm-53-flash", "ready": True, "kind": "LLMInferenceService"},
        {"id": "publishers/acme/models/gemma4", "ready": True, "kind": "LLMInferenceService"},
        {"id": "publishers/other/models/gemma4-mini", "ready": False, "kind": "LLMInferenceService"},
    ],
})
EMPTY_BODY = json.dumps({"data": [], "object": "list"})
MINT_FAILURE = json.dumps({
    "error": "Exception thrown while generating token",
    "exceptionCode": "AUTH_FAILURE",
    "refId": "001",
})


class FakeTransport:
    """Replay a scripted list of (code, body) responses; record the calls."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def __call__(self, method, path, body):
        self.calls.append((method, path, body))
        index = min(len(self.calls) - 1, len(self.script) - 1)
        code, text = self.script[index]
        return code, text, {}


class FakeClock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now


def _gateway(transport, **kwargs):
    kwargs.setdefault("sleep", lambda _s: None)
    kwargs.setdefault("cache", {})
    return q.MaaSGateway("https://maas.example.internal", "k", transport=transport, **kwargs)


class TestStatusClassification(unittest.TestCase):
    """`maas_classify_status` is the single retry decision."""

    def test_200_is_ok(self):
        self.assertEqual(q.maas_classify_status(200), "ok")

    def test_401_is_fatal(self):
        self.assertEqual(q.maas_classify_status(401), "fatal")

    def test_rate_limit_and_gateway_faults_are_retryable(self):
        for code in (403, 429, 500, 502, 503, 504):
            self.assertEqual(q.maas_classify_status(code), "retry", code)

    def test_transport_error_is_retryable(self):
        self.assertEqual(q.maas_classify_status(q.MAAS_TRANSPORT_ERROR), "retry")

    def test_404_is_unexpected_not_offline(self):
        """A bare 404 means a wrong origin / incomplete id, not a dead server."""
        self.assertEqual(q.maas_classify_status(404), "unexpected")

    def test_other_codes_are_unexpected(self):
        for code in (400, 405, 422, 501):
            self.assertEqual(q.maas_classify_status(code), "unexpected", code)


class TestBackoff(unittest.TestCase):

    def test_doubles_then_caps(self):
        delays = [q.maas_backoff_seconds(a) for a in range(1, 7)]
        self.assertEqual(delays, [0.5, 1.0, 2.0, 4.0, 8.0, 8.0])

    def test_jitter_is_additive_and_bounded_by_base(self):
        self.assertEqual(q.maas_backoff_seconds(1, jitter=lambda: 1.0), 1.0)
        self.assertEqual(q.maas_backoff_seconds(6, jitter=lambda: 1.0), 8.5)


class TestServerDetail(unittest.TestCase):

    def test_extracts_exception_code_and_ref(self):
        self.assertEqual(q.maas_server_detail(MINT_FAILURE),
                         "exceptionCode=AUTH_FAILURE, refId=001")

    def test_empty_and_non_json_bodies_are_empty(self):
        for body in ("", "   ", "<html>502</html>", "[]"):
            self.assertEqual(q.maas_server_detail(body), "", body)


class TestApiKey(unittest.TestCase):

    def test_explicit_wins(self):
        with mock.patch.dict(os.environ, {"MAAS_API_KEY": "env"}, clear=False):
            self.assertEqual(q.maas_api_key("explicit"), "explicit")

    def test_env_is_used(self):
        with mock.patch.dict(os.environ, {"MAAS_API_KEY": " env-key\n"}, clear=False):
            self.assertEqual(q.maas_api_key(), "env-key")

    def test_key_file_is_read_and_stripped(self):
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".token", delete=False) as handle:
            handle.write("file-key\nsecond-line\n")
            path = handle.name
        try:
            env = {"MAAS_API_KEY": "", "MAAS_API_KEY_FILE": path}
            with mock.patch.dict(os.environ, env, clear=False):
                self.assertEqual(q.maas_api_key(), "file-key")
        finally:
            os.unlink(path)

    def test_missing_key_file_warns_and_returns_empty(self):
        env = {"MAAS_API_KEY": "", "MAAS_API_KEY_FILE": "/nonexistent/token"}
        with mock.patch.dict(os.environ, env, clear=False):
            self.assertEqual(q.maas_api_key(), "")

    def test_nothing_configured(self):
        env = {"MAAS_API_KEY": "", "MAAS_API_KEY_FILE": ""}
        with mock.patch.dict(os.environ, env, clear=False):
            self.assertEqual(q.maas_api_key(), "")


class TestListingRetry(unittest.TestCase):
    """The acceptance scenarios from the brief, driven offline."""

    def test_flaky_sequence_ends_on_the_full_listing(self):
        transport = FakeTransport([
            (403, "rate limited"),
            (500, MINT_FAILURE),
            (200, EMPTY_BODY),
            (200, MODELS_BODY),
        ])
        gateway = _gateway(transport)
        models = gateway.list_models()
        self.assertEqual(len(models), 3)
        self.assertEqual(models[0]["name"], "publishers/acme/models/glm-53-flash")
        self.assertEqual(len(transport.calls), 4)
        self.assertEqual(gateway.sleeps, [0.5, 1.0, 2.0])

    def test_empty_200_is_retried_not_reported_as_success(self):
        transport = FakeTransport([(200, EMPTY_BODY), (200, MODELS_BODY)])
        gateway = _gateway(transport)
        self.assertEqual(len(gateway.list_models()), 3)
        self.assertEqual(len(transport.calls), 2)

    def test_empty_200_on_the_last_attempt_is_accepted(self):
        transport = FakeTransport([(200, EMPTY_BODY)])
        gateway = _gateway(transport, max_attempts=3)
        self.assertEqual(gateway.list_models(), [])
        self.assertEqual(len(transport.calls), 3)

    def test_require_models_rejects_a_last_attempt_empty_inventory(self):
        transport = FakeTransport([(200, EMPTY_BODY)])
        gateway = _gateway(transport, max_attempts=2)
        with self.assertRaises(q.MaaSUnavailable):
            gateway.list_models(require_models=True)

    def test_401_returns_immediately_without_looping(self):
        transport = FakeTransport([(401, '{"error":"bad key"}')])
        gateway = _gateway(transport)
        with self.assertRaises(q.MaaSAuthError) as caught:
            gateway.list_models()
        self.assertIn("401", str(caught.exception))
        self.assertEqual(len(transport.calls), 1)
        self.assertEqual(gateway.sleeps, [])

    def test_unexpected_404_is_reported_without_looping(self):
        transport = FakeTransport([(404, "")])
        gateway = _gateway(transport)
        with self.assertRaises(q.MaaSError) as caught:
            gateway.list_models()
        self.assertIn("404", str(caught.exception))
        self.assertEqual(len(transport.calls), 1)

    def test_exhaustion_reports_the_last_failure_and_the_server_detail(self):
        transport = FakeTransport([(403, ""), (500, MINT_FAILURE)])
        gateway = _gateway(transport, max_attempts=3)
        with self.assertRaises(q.MaaSUnavailable) as caught:
            gateway.list_models()
        message = str(caught.exception)
        self.assertIn("3 attempts", message)
        self.assertIn("exceptionCode=AUTH_FAILURE", message)
        self.assertEqual(len(transport.calls), 3)
        self.assertEqual(len(gateway.sleeps), 2)

    def test_transport_error_is_retried(self):
        transport = FakeTransport([(q.MAAS_TRANSPORT_ERROR, "timed out"),
                                   (200, MODELS_BODY)])
        gateway = _gateway(transport)
        self.assertEqual(len(gateway.list_models()), 3)
        self.assertEqual(len(transport.calls), 2)

    def test_non_json_200_is_an_error_not_an_empty_listing(self):
        transport = FakeTransport([(200, "<html>login</html>")])
        gateway = _gateway(transport)
        with self.assertRaises(q.MaaSError) as caught:
            gateway.list_models()
        self.assertIn("non-JSON", str(caught.exception))


class TestModelCache(unittest.TestCase):

    def test_two_listings_cost_one_request(self):
        transport = FakeTransport([(200, MODELS_BODY)])
        gateway = _gateway(transport)
        gateway.list_models()
        gateway.list_models()
        self.assertEqual(len(transport.calls), 1)

    def test_force_bypasses_a_fresh_cache(self):
        transport = FakeTransport([(200, MODELS_BODY)])
        gateway = _gateway(transport)
        gateway.list_models()
        gateway.list_models(force=True)
        self.assertEqual(len(transport.calls), 2)

    def test_entries_expire_after_the_ttl(self):
        clock = FakeClock()
        transport = FakeTransport([(200, MODELS_BODY)])
        gateway = _gateway(transport, clock=clock, cache_ttl=300)
        gateway.list_models()
        clock.now += 301
        gateway.list_models()
        self.assertEqual(len(transport.calls), 2)

    def test_ttl_zero_disables_the_cache(self):
        transport = FakeTransport([(200, MODELS_BODY)])
        gateway = _gateway(transport, cache_ttl=0)
        gateway.list_models()
        gateway.list_models()
        self.assertEqual(len(transport.calls), 2)

    def test_stale_cache_is_used_when_the_network_fails(self):
        transport = FakeTransport([(200, MODELS_BODY), (403, ""), (403, "")])
        gateway = _gateway(transport, max_attempts=2, cache_ttl=0)
        gateway.list_models()
        models = gateway.list_models()
        self.assertEqual(len(models), 3)
        self.assertEqual(len(transport.calls), 3)

    def test_no_stale_cache_means_a_clear_failure(self):
        transport = FakeTransport([(403, "")])
        gateway = _gateway(transport, max_attempts=2)
        with self.assertRaises(q.MaaSUnavailable):
            gateway.list_models()


class TestModelResolution(unittest.TestCase):

    IDS = [
        "publishers/acme/models/glm-53-flash",
        "publishers/acme/models/gemma4",
        "publishers/other/models/gemma4-mini",
    ]

    def test_full_id_is_returned_without_any_network_use(self):
        transport = FakeTransport([(200, MODELS_BODY)])
        gateway = _gateway(transport)
        full = "publishers/acme/models/glm-53-flash"
        self.assertEqual(gateway.resolve_model(full), full)
        self.assertEqual(transport.calls, [])

    def test_exact_suffix_match_resolves(self):
        self.assertEqual(q.maas_resolve_model_id(self.IDS, "glm-53-flash"),
                         "publishers/acme/models/glm-53-flash")

    def test_suffix_beats_substring_for_a_short_name(self):
        """`gemma4` matches one id by suffix and one only by substring."""
        self.assertEqual(q.maas_resolve_model_id(self.IDS, "gemma4"),
                         "publishers/acme/models/gemma4")

    def test_unique_substring_match_resolves(self):
        self.assertEqual(q.maas_resolve_model_id(self.IDS, "53-flash"),
                         "publishers/acme/models/glm-53-flash")

    def test_ambiguous_name_is_an_explicit_local_error(self):
        with self.assertRaises(q.MaaSError) as caught:
            q.maas_resolve_model_id(self.IDS, "gemma")
        self.assertIn("ambiguous", str(caught.exception))

    def test_missing_name_lists_the_known_ids(self):
        with self.assertRaises(q.MaaSError) as caught:
            q.maas_resolve_model_id(self.IDS, "nope")
        self.assertIn("not exposed", str(caught.exception))
        self.assertIn("glm-53-flash", str(caught.exception))

    def test_empty_name_is_rejected(self):
        with self.assertRaises(q.MaaSError):
            q.maas_resolve_model_id(self.IDS, "  ")

    def test_resolve_model_reuses_the_listing_cache(self):
        transport = FakeTransport([(200, MODELS_BODY)])
        gateway = _gateway(transport)
        gateway.list_models()
        resolved = gateway.resolve_model("glm-53-flash")
        self.assertEqual(resolved, "publishers/acme/models/glm-53-flash")
        self.assertEqual(len(transport.calls), 1)


class TestHttpTransport(unittest.TestCase):
    """One real request against a local server pins URL, header and parsing."""

    @classmethod
    def setUpClass(cls):
        cls.requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                cls.requests.append((self.path, self.headers.get("Authorization")))
                body = MODELS_BODY.encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        cls.server = HTTPServer(("127.0.0.1", 0), Handler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_models_request_uses_the_origin_path_and_bearer_header(self):
        gateway = q.MaaSGateway(f"http://127.0.0.1:{self.port}", "secret",
                                timeout=2, max_attempts=1, cache={})
        models = gateway.list_models()
        self.assertEqual(len(models), 3)
        self.assertEqual(self.requests[-1],
                         ("/v1/models", "Bearer secret"))


class TestBackendRegistration(unittest.TestCase):
    """The four registries B2 touches must all know the backend."""

    def test_cloud_backends_contains_maas(self):
        self.assertIn("maas", q.CLOUD_BACKENDS)

    def test_env_and_default_host_maps(self):
        self.assertEqual(q.CLOUD_API_KEY_ENV["maas"], "MAAS_API_KEY")
        self.assertEqual(q.CLOUD_HOST_ENV["maas"], "MAAS_HOST")
        self.assertEqual(q.CLOUD_DEFAULT_HOST["maas"], "")

    def test_cli_choices_accept_maas(self):
        parser = q._build_parser()
        args = parser.parse_args(["-b", "maas", "-H", "https://maas.example.internal"])
        self.assertEqual(args.backend, "maas")

    def test_context_size_flag_exists(self):
        parser = q._build_parser()
        args = parser.parse_args(["--context-size", "32768"])
        self.assertEqual(args.context_size, 32768)


class TestToolFormat(unittest.TestCase):
    """Both native-tool model families were falling back to `inline`."""

    def test_glm_53_is_native(self):
        self.assertEqual(q.get_tool_format("publishers/acme/models/glm-53-flash"), "openai")

    def test_gemma4_is_native(self):
        self.assertEqual(q.get_tool_format("publishers/acme/models/gemma4"), "openai")

    def test_unrelated_glm_entries_stay_inline(self):
        """`glm-4`/`glm4` target other checkpoints and must not be flipped."""
        self.assertEqual(q.get_tool_format("glm-4-9b"), "inline")
        self.assertEqual(q.get_tool_format("glm4-air"), "inline")


class TestOutputFloor(unittest.TestCase):
    """Reasoning is always on: a small max_tokens returns content null."""

    def setUp(self):
        self.ctx = q.CommandContext()

    def _payload(self, backend, **kwargs):
        builder = q.RequestBuilder(self.ctx)
        payload = {}
        builder._apply_size_options(payload, backend, kwargs)
        return payload

    def test_small_cap_is_raised_to_the_floor(self):
        payload = self._payload("maas", max_output_tokens=100)
        self.assertEqual(payload["max_tokens"], q.MAAS_MIN_OUTPUT_TOKENS)

    def test_large_cap_is_untouched(self):
        payload = self._payload("maas", max_output_tokens=8192)
        self.assertEqual(payload["max_tokens"], 8192)

    def test_other_backends_keep_the_requested_cap(self):
        payload = self._payload("gemini", max_output_tokens=100)
        self.assertEqual(payload["max_tokens"], 100)

    def test_warmup_stays_minimal(self):
        payload = self._payload("maas", is_warmup=True, max_output_tokens=8192)
        self.assertEqual(payload["max_tokens"], 1)

    def test_output_cap_floor_applies_before_the_payload(self):
        self.ctx.backend = "maas"
        self.ctx.context_window_size = 100000
        self.ctx.max_output_tokens = 500
        messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
        _messages, kwargs = q.RequestBuilder(self.ctx)._apply_output_cap(messages, {})
        self.assertEqual(kwargs["max_output_tokens"], q.MAAS_MIN_OUTPUT_TOKENS)


class TestRequestShaping(unittest.TestCase):

    def setUp(self):
        self.ctx = q.CommandContext()
        self.ctx.backend = "maas"
        self.ctx.api_key = "secret"
        self.ctx.base_url = "https://maas.example.internal"

    def test_chat_url_is_the_openai_route(self):
        self.assertEqual(q.RequestBuilder(self.ctx)._get_chat_url("maas"),
                         "https://maas.example.internal/v1/chat/completions")

    def test_bearer_header_is_sent(self):
        query = q.ModelQuery(context=self.ctx)
        self.assertEqual(query._get_headers().get("Authorization"), "Bearer secret")

    def test_inference_params_reach_the_payload(self):
        payload = {}
        q.RequestBuilder._apply_backend_inference_params(
            payload, "maas", {"temperature": 0.2, "top_p": 0.9, "top_k": 20, "min_p": 0.01})
        self.assertEqual(payload, {"temperature": 0.2, "top_p": 0.9,
                                   "top_k": 20, "min_p": 0.01})

    def test_thinking_flag_is_a_noop(self):
        """Reasoning cannot be disabled on this gateway: no template flag."""
        payload = {}
        q.RequestBuilder._apply_no_thinking(payload, "maas")
        self.assertEqual(payload, {})


class TestContextWindow(unittest.TestCase):

    def setUp(self):
        self.ctx = q.CommandContext()
        self.ctx.backend = "maas"
        self.ctx.base_url = "https://maas.example.internal"
        self.ctx.model = "publishers/acme/models/glm-53-flash"
        self.ctx.context_window_size = 0
        self.ctx.context_size_override = 0

    def test_maas_window_is_unknown_and_hinted_once(self):
        import io
        import contextlib
        q._MAAS_WINDOW_HINTED = False
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertFalse(q.refresh_context_window_size(self.ctx))
            self.assertFalse(q.refresh_context_window_size(self.ctx))
        output = err.getvalue()
        self.assertEqual(self.ctx.context_window_size, 0)
        self.assertEqual(output.count("--context-size"), 1, output)

    def test_explicit_override_wins_over_detection(self):
        self.ctx.context_size_override = 32768
        self.assertTrue(q.refresh_context_window_size(self.ctx))
        self.assertEqual(self.ctx.context_window_size, 32768)


class TestOriginNormalization(unittest.TestCase):
    """A pasted OpenAI base URL (`.../v1`) must not become `/v1/v1/models`."""

    def test_variants_normalize_to_the_same_origin(self):
        for raw in ("maas.example.internal",
                    "https://maas.example.internal",
                    "https://maas.example.internal/",
                    "https://maas.example.internal/v1",
                    "https://maas.example.internal/v1/"):
            self.assertEqual(q.maas_origin(raw), "https://maas.example.internal", raw)

    def test_http_origin_is_preserved(self):
        self.assertEqual(q.maas_origin("http://127.0.0.1:8099/v1"),
                         "http://127.0.0.1:8099")

    def test_empty_stays_empty(self):
        self.assertEqual(q.maas_origin(""), "")
        self.assertEqual(q.maas_origin(None), "")

    def test_gateway_strips_v1_before_building_the_path(self):
        transport = FakeTransport([(200, MODELS_BODY)])
        gateway = q.MaaSGateway("https://maas.example.internal/v1", "k",
                                transport=transport, cache={})
        gateway.list_models()
        self.assertEqual(transport.calls[0][1], "/v1/models")
        self.assertEqual(gateway.base_url, "https://maas.example.internal")

    def test_env_host_with_v1_is_normalized(self):
        args = argparse.Namespace(backend="maas", host=None)
        env = {"MAAS_HOST": "https://maas.example.internal/v1"}
        with mock.patch.dict(os.environ, env, clear=False):
            self.assertEqual(q.resolve_connection(args),
                             ("maas", "https://maas.example.internal"))

    def test_host_override_with_v1_is_normalized(self):
        args = argparse.Namespace(host="https://maas.example.internal/v1", backend="maas")
        self.assertEqual(q._resolve_host_override(args),
                         ("maas", "https://maas.example.internal"))


class TestBackendResolution(unittest.TestCase):

    def test_bare_host_becomes_https(self):
        args = argparse.Namespace(host="maas.example.internal", backend="maas")
        self.assertEqual(q._resolve_host_override(args),
                         ("maas", "https://maas.example.internal"))

    def test_explicit_https_is_kept(self):
        args = argparse.Namespace(host="https://maas.example.internal", backend="maas")
        self.assertEqual(q._resolve_host_override(args),
                         ("maas", "https://maas.example.internal"))

    def test_env_host_without_scheme_becomes_https(self):
        args = argparse.Namespace(backend="maas", host=None)
        env = {"MAAS_HOST": "maas.example.internal"}
        with mock.patch.dict(os.environ, env, clear=False):
            self.assertEqual(q.resolve_connection(args),
                             ("maas", "https://maas.example.internal"))

    def test_verify_server_requires_an_origin(self):
        import io
        import contextlib
        args = argparse.Namespace(api_key=None)
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit):
            q._verify_server("maas", "", args)
        self.assertIn("MAAS_HOST", err.getvalue())

    def test_verify_server_requires_a_key(self):
        import io
        import contextlib
        args = argparse.Namespace(api_key=None)
        err = io.StringIO()
        env = {"MAAS_API_KEY": "", "MAAS_API_KEY_FILE": ""}
        with mock.patch.dict(os.environ, env, clear=False):
            with contextlib.redirect_stderr(err), self.assertRaises(SystemExit):
                q._verify_server("maas", "https://maas.example.internal", args)
        self.assertIn("MAAS_API_KEY_FILE", err.getvalue())

    def test_cloud_api_key_routes_maas_through_the_file_resolver(self):
        env = {"MAAS_API_KEY": "env-key"}
        with mock.patch.dict(os.environ, env, clear=False):
            self.assertEqual(q._cloud_api_key("maas"), "env-key")

    def test_probe_backend_uses_check_maas(self):
        with mock.patch.object(q, "check_maas", return_value=True) as probe:
            self.assertTrue(q._probe_backend("maas", "https://x", "key"))
        probe.assert_called_once_with("https://x", "key")


class TestModelSelection(unittest.TestCase):

    def test_short_name_is_resolved_before_use(self):
        args = argparse.Namespace(model="glm-53-flash", api_key=None)
        fake = mock.Mock()
        fake.resolve_model.return_value = "publishers/acme/models/glm-53-flash"
        with mock.patch.object(q, "maas_gateway", return_value=fake) as factory:
            model = q._select_model("maas", "https://maas.example.internal", args)
        self.assertEqual(model, "publishers/acme/models/glm-53-flash")
        factory.assert_called_once()

    def test_full_id_is_not_resolved(self):
        args = argparse.Namespace(model="publishers/acme/models/glm-53-flash", api_key=None)
        with mock.patch.object(q, "maas_gateway") as factory:
            model = q._select_model("maas", "https://maas.example.internal", args)
        self.assertEqual(model, "publishers/acme/models/glm-53-flash")
        factory.assert_not_called()

    def test_unknown_short_name_exits_with_the_known_ids(self):
        import io
        import contextlib
        args = argparse.Namespace(model="nope", api_key=None)
        fake = mock.Mock()
        fake.resolve_model.side_effect = q.MaaSError("not exposed (known: a/b)")
        err = io.StringIO()
        with mock.patch.object(q, "maas_gateway", return_value=fake), \
                contextlib.redirect_stderr(err), self.assertRaises(SystemExit):
            q._select_model("maas", "https://maas.example.internal", args)
        self.assertIn("not exposed", err.getvalue())

    def test_listing_routes_through_the_gateway(self):
        with mock.patch.object(q, "fetch_models_maas", return_value=[{"name": "a/b"}]) as fetch:
            models = q.fetch_models_for_backend("maas", "https://x", "k")
        self.assertEqual(models, [{"name": "a/b"}])
        fetch.assert_called_once_with("https://x", "k")


class TestFlakyListingEndToEnd(unittest.TestCase):
    """The brief's acceptance scenario, through the real urllib transport."""

    @classmethod
    def setUpClass(cls):
        cls.calls = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                cls.calls.append(self.path)
                index = min(len(cls.calls) - 1, 3)
                script = [
                    (403, "rate limited"),
                    (500, MINT_FAILURE),
                    (200, EMPTY_BODY),
                    (200, MODELS_BODY),
                ]
                code, text = script[index]
                body = text.encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        cls.server = HTTPServer(("127.0.0.1", 0), Handler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_flaky_listing_returns_the_models(self):
        url = f"http://127.0.0.1:{self.port}"
        # The retry loop is real; only its sleeps are stubbed out.
        models = q.MaaSGateway(url, "k", timeout=2, cache={},
                               sleep=lambda _s: None).list_models()
        self.assertEqual(len(models), 3)
        self.assertEqual(len(self.calls), 4)


if __name__ == "__main__":
    unittest.main()
