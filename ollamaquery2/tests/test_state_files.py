#!/usr/bin/env python3
"""Private state-file modes (0700 directories, 0600 files).

Covers the helper pair and every transcript writer that previously relied on
the process umask: session directory/index/record, readline history, the
agentic JSONL log, and the spill writers. Offline: no backend, no network.
"""

import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ollamaquery2 as m


def _mode(path: str) -> int:
    """Return the permission bits of `path`."""
    return os.stat(path).st_mode & 0o777


class TestPrivateHelpers(unittest.TestCase):
    """The mode helpers used by every state-file writer."""

    def test_ensure_private_dir_creates_0700(self):
        with tempfile.TemporaryDirectory() as td:
            d = os.path.join(td, "state")
            m._ensure_private_dir(d)
            self.assertEqual(_mode(d), 0o700)

    def test_ensure_private_dir_tightens_existing(self):
        with tempfile.TemporaryDirectory() as td:
            d = os.path.join(td, "state")
            os.makedirs(d)
            os.chmod(d, 0o755)
            m._ensure_private_dir(d)
            self.assertEqual(_mode(d), 0o700)

    def test_open_private_creates_0600(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "f.json")
            with m._open_private(p, "w", encoding="utf-8") as f:
                f.write("{}")
            self.assertEqual(_mode(p), 0o600)

    def test_open_private_append_preserves_content_and_mode(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "f.log")
            with m._open_private(p, "a", encoding="utf-8") as f:
                f.write("a\n")
            with m._open_private(p, "a", encoding="utf-8") as f:
                f.write("b\n")
            self.assertEqual(_mode(p), 0o600)
            with open(p, encoding="utf-8") as f:
                self.assertEqual(f.read(), "a\nb\n")

    def test_open_private_tightens_pre_existing_file(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "f.json")
            with open(p, "w", encoding="utf-8") as f:
                f.write("old")
            os.chmod(p, 0o644)
            with m._open_private(p, "w", encoding="utf-8") as f:
                f.write("new")
            self.assertEqual(_mode(p), 0o600)


class TestSessionFiles(unittest.TestCase):
    """Session transcripts and the index must never be world-readable."""

    def _ctx(self) -> types.SimpleNamespace:
        return types.SimpleNamespace(model="test-model", backend="ollama",
                                     base_url="http://127.0.0.1:11434",
                                     current_context_tokens=0)

    def test_save_session_and_index_are_private(self):
        with tempfile.TemporaryDirectory() as td:
            session_dir = os.path.join(td, "state", "sessions")
            with patch.object(m, "SESSION_DIR", session_dir), \
                 patch.object(m, "SESSION_INDEX", os.path.join(session_dir, "index.json")):
                sm = m.SessionManager()
                sid = sm.save_session([{"role": "user", "content": "hello"}], self._ctx())
                self.assertEqual(_mode(os.path.join(session_dir, f"{sid}.json")), 0o600)
                self.assertEqual(_mode(os.path.join(session_dir, "index.json")), 0o600)
                self.assertEqual(_mode(session_dir), 0o700)
                self.assertEqual(_mode(os.path.dirname(session_dir)), 0o700)

    def test_manager_tightens_existing_transcripts(self):
        with tempfile.TemporaryDirectory() as td:
            session_dir = os.path.join(td, "sessions")
            os.makedirs(session_dir)
            index = os.path.join(session_dir, "index.json")
            record = os.path.join(session_dir, "20260101_000000.json")
            for path in (index, record):
                with open(path, "w", encoding="utf-8") as f:
                    f.write("[]")
                os.chmod(path, 0o644)
            with patch.object(m, "SESSION_DIR", session_dir), \
                 patch.object(m, "SESSION_INDEX", index):
                m.SessionManager()
            self.assertEqual(_mode(index), 0o600)
            self.assertEqual(_mode(record), 0o600)
            self.assertEqual(_mode(session_dir), 0o700)


class TestOtherStateFiles(unittest.TestCase):
    """Agentic logs, readline history and spill files use the same modes."""

    def test_agentic_log_is_private(self):
        with tempfile.TemporaryDirectory() as td:
            with patch.dict(os.environ, {"HOME": td}):
                log = m.AgenticLogger()
                try:
                    log.write(type="test")
                finally:
                    log.close()
                self.assertEqual(_mode(log.path), 0o600)
                self.assertEqual(_mode(os.path.dirname(log.path)), 0o700)

    @unittest.skipUnless(m.READLINE_AVAILABLE, "readline not available")
    def test_readline_history_is_private(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "session")
            with patch.object(m.readline, "get_current_history_length", return_value=0), \
                 patch.object(m.readline, "get_history_length", return_value=0):
                m._save_history_file(path)
            self.assertTrue(os.path.exists(path))
            self.assertEqual(_mode(path), 0o600)

    def test_fetch_spill_is_private(self):
        with tempfile.TemporaryDirectory() as td:
            ctx = types.SimpleNamespace(path_acl=None)
            text = "x" * (m.FETCH_SPILL_THRESHOLD + 10)
            with patch.object(m, "CLUSTER_SPILL_DIR", td):
                out = m._spill_fetch_result(ctx, text, "https://example.com/page")
            spill = [name for name in os.listdir(td) if name.startswith("fetch_result_")]
            self.assertEqual(len(spill), 1)
            path = os.path.join(td, spill[0])
            self.assertEqual(_mode(path), 0o600)
            self.assertIn(path, out)


if __name__ == '__main__':
    unittest.main(verbosity=2)
