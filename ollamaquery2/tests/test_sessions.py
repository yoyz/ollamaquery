#!/usr/bin/env python3
"""Named sessions: `/save <name>`, `/sessions`, `/resume <name|id>`, `/rename`.

A name is a slot: saving twice under one name updates a single transcript, an
unnamed `/save` updates the session this conversation already bound to, and
`/resume` accepts an id (exact or prefix) or a name (accents/case folded,
prefix or substring). Offline: no backend, no network.
"""

import contextlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ollamaquery2 as m


def _ctx(model: str = "test-model") -> types.SimpleNamespace:
    """A minimal CommandContext stand-in (only what save_session reads)."""
    return types.SimpleNamespace(model=model, backend="ollama",
                                 base_url="http://127.0.0.1:11434",
                                 current_context_tokens=0)


def _history(text: str = "hello") -> list:
    """A conversation with a system prompt and one user turn."""
    return [{"role": "system", "content": "sys"},
            {"role": "user", "content": text}]


class _SessionDirCase(unittest.TestCase):
    """Point SessionManager at a throwaway directory for each test."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = os.path.join(self._tmp.name, "state", "sessions")
        for target, value in (("SESSION_DIR", self.dir),
                              ("SESSION_INDEX", os.path.join(self.dir, "index.json"))):
            patcher = patch.object(m, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def manager(self) -> "m.SessionManager":
        return m.SessionManager()


class TestSessionSlug(unittest.TestCase):
    """Name normalization: accents folded, punctuation collapsed, capped."""

    def test_accents_are_folded_not_dropped(self):
        self.assertEqual(m._session_slug("Réunion ACL"), "reunion-acl")

    def test_spaces_and_punctuation_collapse(self):
        self.assertEqual(m._session_slug("  acl :: refactor -- v2  "), "acl-refactor-v2")

    def test_case_is_lowered(self):
        self.assertEqual(m._session_slug("ACL-Refactor"), "acl-refactor")

    def test_length_is_capped_without_trailing_dash(self):
        slug = m._session_slug("x" * 60)
        self.assertEqual(len(slug), m.SESSION_SLUG_MAX_CHARS)
        self.assertFalse(slug.endswith("-"))

    def test_name_without_letters_or_digits_is_empty(self):
        self.assertEqual(m._session_slug("***"), "")
        self.assertEqual(m._session_slug("   "), "")


class TestNamedSaves(_SessionDirCase):
    """Saving, updating and rejecting names."""

    def test_unnamed_save_keeps_the_timestamp_id(self):
        sid = self.manager().save_session(_history(), _ctx())
        self.assertRegex(sid, r"^\d{8}_\d{6}$")
        entry = self.manager().list_sessions()[0]
        self.assertEqual(entry["name"], "")
        self.assertEqual(entry["slug"], "")

    def test_named_save_puts_the_slug_in_the_id_and_index(self):
        sid = self.manager().save_session(_history(), _ctx(), name="Réunion ACL")
        self.assertRegex(sid, r"^\d{8}_\d{6}_reunion-acl$")
        entry = self.manager().list_sessions()[0]
        self.assertEqual(entry["id"], sid)
        self.assertEqual(entry["name"], "Réunion ACL")
        self.assertEqual(entry["slug"], "reunion-acl")

    def test_same_name_updates_one_transcript(self):
        first = self.manager().save_session(_history("one"), _ctx())
        named = self.manager().save_session(_history("two"), _ctx(), name="work")
        again = self.manager().save_session(_history("three two"), _ctx(), name="work")
        self.assertNotEqual(first, named)
        self.assertEqual(named, again)
        entries = self.manager().list_sessions()
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["id"], named)
        self.assertEqual(entries[0]["message_count"], 2)
        self.assertEqual(entries[0]["name"], "work")
        with open(os.path.join(self.dir, f"{named}.json"), encoding="utf-8") as f:
            self.assertIn("three two", f.read())

    def test_explicit_id_updates_in_place_and_can_be_renamed(self):
        sid = self.manager().save_session(_history(), _ctx(), name="first")
        renamed = self.manager().save_session(_history(), _ctx(),
                                              name="second", session_id=sid)
        self.assertEqual(renamed, sid)
        entry = self.manager().list_sessions()[0]
        self.assertEqual((entry["id"], entry["name"], entry["slug"]),
                         (sid, "second", "second"))

    def test_unnamed_resave_of_an_existing_id_keeps_the_name(self):
        sid = self.manager().save_session(_history(), _ctx(), name="keep me")
        again = self.manager().save_session(_history(), _ctx(), session_id=sid)
        self.assertEqual(again, sid)
        self.assertEqual(self.manager().list_sessions()[0]["name"], "keep me")

    def test_invalid_name_is_rejected_without_writing(self):
        with self.assertRaises(ValueError):
            self.manager().save_session(_history(), _ctx(), name="***")
        self.assertEqual(self.manager().list_sessions(), [])

    def test_collision_suffix_keeps_both_transcripts(self):
        sid = self.manager().save_session(_history(), _ctx())
        with open(os.path.join(self.dir, f"{sid}.json"), "w", encoding="utf-8") as f:
            f.write("{}")
        self.assertEqual(self.manager()._unique_id(sid), f"{sid}-2")

    def test_last_message_wins_the_summary_scan(self):
        history = [{"role": "user", "content": "first question"},
                   {"role": "assistant", "content": "answer"},
                   {"role": "user", "content": "second question"}]
        self.manager().save_session(history, _ctx())
        self.assertEqual(self.manager().list_sessions()[0]["summary"], "first question")


class TestResolve(_SessionDirCase):
    """`/resume` handle matching: ids, prefixes, names, accents."""

    def setUp(self) -> None:
        super().setUp()
        sm = self.manager()
        self.acl_id = sm.save_session(_history(), _ctx(), name="Réunion ACL")
        self.other_id = sm.save_session(_history(), _ctx(), name="web server")

    def test_exact_id(self):
        self.assertEqual(self.manager().resolve(self.acl_id)["id"], self.acl_id)

    def test_id_prefix(self):
        # Both transcripts were saved in the same second, so the shared date
        # prefix resolves to the newest one — a prefix never picks an older
        # record over a newer match, which is what a user expects from /resume.
        self.assertEqual(self.manager().resolve(self.acl_id[:8])["id"], self.other_id)

    def test_exact_name_is_accent_and_case_insensitive(self):
        self.assertEqual(self.manager().resolve("réunion acl")["id"], self.acl_id)
        self.assertEqual(self.manager().resolve("REUNION ACL")["id"], self.acl_id)

    def test_slug_prefix(self):
        self.assertEqual(self.manager().resolve("reunion")["id"], self.acl_id)
        self.assertEqual(self.manager().resolve("web")["id"], self.other_id)

    def test_name_substring(self):
        self.assertEqual(self.manager().resolve("server")["id"], self.other_id)

    def test_full_id_beats_a_name_fragment(self):
        self.assertEqual(self.manager().resolve(self.acl_id)["id"], self.acl_id)

    def test_unknown_handle_returns_none(self):
        self.assertIsNone(self.manager().resolve("no-such-session"))
        self.assertIsNone(self.manager().resolve("   "))

    def test_load_by_name_returns_messages_and_labels(self):
        data = self.manager().load_session("reunion")
        self.assertEqual(data["id"], self.acl_id)
        self.assertEqual(data["name"], "Réunion ACL")
        self.assertEqual(data["messages"], _history())


class TestConversationStoreBinding(_SessionDirCase):
    """The store remembers which session `/save` updates."""

    def store(self) -> "m.ConversationStore":
        ctx = _ctx()
        store = m.ConversationStore(lambda: ctx, lambda: None)
        store.replace(_history())
        return store

    def test_named_save_binds_and_repeated_save_updates(self):
        store = self.store()
        sid = store.save_session("Work Notes")
        self.assertEqual(store.active_id, sid)
        self.assertEqual(store.active_name, "Work Notes")
        self.assertEqual(store.save_session(), sid)
        self.assertEqual(len(store.list_sessions()), 1)

    def test_rename_keeps_the_id(self):
        store = self.store()
        sid = store.save_session("draft")
        self.assertEqual(store.rename_session("final"), sid)
        entry = store.list_sessions()[0]
        self.assertEqual((entry["id"], entry["name"], entry["slug"]),
                         (sid, "final", "final"))
        self.assertEqual(store.active_name, "final")

    def test_load_binds_the_resumed_session(self):
        store = self.store()
        sid = store.save_session("resumable")
        other = m.ConversationStore(lambda: _ctx(), lambda: None)
        data = other.load_session("resumable")
        self.assertEqual(data["id"], sid)
        self.assertEqual(other.active_id, sid)
        self.assertEqual(other.active_name, "resumable")
        self.assertEqual(other.save_session(), sid)

    def test_autosave_updates_the_bound_session(self):
        store = self.store()
        store.append({"role": "assistant", "content": "reply"})
        sid = store.save_session("kept")
        self.assertEqual(store.autosave_on_exit(), sid)
        self.assertEqual(len(store.list_sessions()), 1)


class TestLegacyIndex(_SessionDirCase):
    """Index entries written before naming existed still list, resume and rename."""

    def _write_legacy(self, session_id: str = "20260101_000000") -> str:
        os.makedirs(self.dir, exist_ok=True)
        record = os.path.join(self.dir, f"{session_id}.json")
        with open(record, "w", encoding="utf-8") as f:
            json.dump({"id": session_id, "model": "old", "messages": _history()}, f)
        with open(os.path.join(self.dir, "index.json"), "w", encoding="utf-8") as f:
            json.dump([{"id": session_id, "created": "2026-01-01T00:00:00",
                        "model": "old", "message_count": 2, "summary": "old"}], f)
        return session_id

    def test_list_and_resume_an_unnamed_legacy_session(self):
        sid = self._write_legacy()
        entry = self.manager().list_sessions()[0]
        self.assertEqual(entry.get("name", ""), "")
        data = self.manager().load_session("20260101")
        self.assertEqual(data["id"], sid)
        self.assertEqual(data["messages"], _history())

    def test_naming_a_resumed_legacy_session_keeps_its_id(self):
        sid = self._write_legacy()
        store = m.ConversationStore(lambda: _ctx(), lambda: None)
        store.load_session(sid)
        self.assertEqual(store.rename_session("rescued"), sid)
        entry = store.list_sessions()[0]
        self.assertEqual((entry["id"], entry["name"], entry["slug"]),
                         (sid, "rescued", "rescued"))


class TestSessionHandlers(_SessionDirCase):
    """`/save`, `/sessions`, `/resume`, `/rename` through ChatLoop."""

    def loop(self) -> "m.ChatLoop":
        loop = object.__new__(m.ChatLoop)
        loop.ctx = _ctx()
        loop._refresh_context_bar = lambda *a, **k: None
        loop.messages = _history()
        return loop

    def run_capture(self, loop: "m.ChatLoop", line: str) -> str:
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            handled = loop._dispatch_command(line)
        self.assertIs(handled, False, f"{line!r} was not handled")
        return err.getvalue()

    def test_save_name_and_list(self):
        loop = self.loop()
        out = self.run_capture(loop, "/save Réunion ACL")
        self.assertIn("(name: Réunion ACL)", out)
        listing = self.run_capture(loop, "/sessions")
        self.assertIn("Réunion ACL", listing)
        self.assertIn("*", listing)          # active-session marker
        self.assertIn("/resume <name|id>", listing)

    def test_save_without_name_prints_no_name(self):
        loop = self.loop()
        out = self.run_capture(loop, "/save")
        self.assertIn("Session saved as", out)
        self.assertNotIn("(name:", out)

    def test_second_save_reports_an_update(self):
        loop = self.loop()
        self.run_capture(loop, "/save work")
        out = self.run_capture(loop, "/save")
        self.assertIn("Session updated", out)
        self.assertIn("(name: work)", out)

    def test_save_with_invalid_name_is_refused(self):
        loop = self.loop()
        out = self.run_capture(loop, "/save ***")
        self.assertIn("Invalid name", out)
        self.assertEqual(self.manager().list_sessions(), [])

    def test_resume_by_name_and_by_id(self):
        loop = self.loop()
        self.run_capture(loop, "/save Reunion")
        sid = loop._conversation().active_id
        loop.messages = _history("different")
        out = self.run_capture(loop, "/resume reunion")
        self.assertIn("Loaded session", out)
        self.assertIn("'Reunion'", out)
        self.assertEqual(loop.messages, _history())
        out = self.run_capture(loop, f"/resume {sid}")
        self.assertIn(sid, out)

    def test_resume_unknown_and_missing_argument(self):
        loop = self.loop()
        self.assertIn("not found", self.run_capture(loop, "/resume nope"))
        self.assertIn("Usage: /resume", self.run_capture(loop, "/resume"))

    def test_rename_requires_a_saved_session(self):
        loop = self.loop()
        self.assertIn("No saved session", self.run_capture(loop, "/rename foo"))

    def test_rename_refuses_a_taken_name(self):
        loop = self.loop()
        self.run_capture(loop, "/save taken")
        self.run_capture(loop, "/save other")
        self.run_capture(loop, "/resume other")
        out = self.run_capture(loop, "/rename taken")
        self.assertIn("already used by session", out)

    def test_rename_labels_the_active_session(self):
        loop = self.loop()
        self.run_capture(loop, "/save draft")
        sid = loop._conversation().active_id
        out = self.run_capture(loop, "/rename final")
        self.assertIn("is now named 'final'", out)
        entry = self.manager().list_sessions()[0]
        self.assertEqual((entry["id"], entry["name"]), (sid, "final"))

    def test_rename_usage_and_invalid_name(self):
        loop = self.loop()
        self.run_capture(loop, "/save draft")
        self.assertIn("Usage: /rename", self.run_capture(loop, "/rename"))
        self.assertIn("Invalid name", self.run_capture(loop, "/rename ***"))


class TestResumeCompletion(_SessionDirCase):
    """`/resume <TAB>` offers slugs and ids, with a short cache."""

    def completer(self) -> "m.ChatCompleter":
        completer = object.__new__(m.ChatCompleter)
        completer.ctx = None
        return completer

    def test_slug_and_id_are_offered(self):
        sm = self.manager()
        sid = sm.save_session(_history(), _ctx(), name="Web Server")
        handles = self.completer()._complete_session_handles("")
        self.assertIn("web-server", handles)
        self.assertIn(sid, handles)

    def test_prefix_filters(self):
        self.manager().save_session(_history(), _ctx(), name="web server")
        self.manager().save_session(_history(), _ctx(), name="database")
        handles = self.completer()._complete_session_handles("web")
        self.assertEqual(handles, ["web-server"])

    def test_result_is_cached_between_keystrokes(self):
        self.manager().save_session(_history(), _ctx(), name="one")
        completer = self.completer()
        first = completer._complete_session_handles("")
        os.unlink(os.path.join(self.dir, "index.json"))
        self.assertEqual(completer._complete_session_handles(""), first)


class TestAgenticLogSessionLink(unittest.TestCase):
    """A3: the JSONL log points at the session it belongs to."""

    def test_stem_sanitizes_and_defaults(self):
        self.assertEqual(m._agentic_log_stem(None), "no-session")
        self.assertEqual(m._agentic_log_stem(""), "no-session")
        self.assertEqual(m._agentic_log_stem("20261113_152947_reunion-acl"),
                         "20261113_152947_reunion-acl")
        self.assertEqual(m._agentic_log_stem("a/b c"), "a-b-c")
        self.assertLessEqual(len(m._agentic_log_stem("x" * 200)),
                             m.AGENTIC_LOG_SESSION_MAX_CHARS)

    def test_logger_names_and_stamps_the_session(self):
        with tempfile.TemporaryDirectory() as td, \
                patch.dict(os.environ, {"HOME": td}, clear=False):
            log = m.AgenticLogger(session_id="20261113_152947_reunion-acl")
            try:
                log.write(type="start", user_input="hi")
            finally:
                log.close()
            self.assertTrue(os.path.basename(log.path).startswith(
                "20261113_152947_reunion-acl_"))
            with open(log.path, encoding="utf-8") as handle:
                entry = json.loads(handle.readline())
        self.assertEqual(entry["session_id"], "20261113_152947_reunion-acl")

    def test_no_session_is_explicit(self):
        with tempfile.TemporaryDirectory() as td, \
                patch.dict(os.environ, {"HOME": td}, clear=False):
            log = m.AgenticLogger()
            try:
                log.write(type="start")
            finally:
                log.close()
            self.assertTrue(os.path.basename(log.path).startswith("no-session_"))
            with open(log.path, encoding="utf-8") as handle:
                entry = json.loads(handle.readline())
        self.assertEqual(entry["session_id"], "no-session")

    def test_rename_after_open_keeps_the_original_link(self):
        with tempfile.TemporaryDirectory() as td, \
                patch.dict(os.environ, {"HOME": td}, clear=False):
            log = m.AgenticLogger(session_id="old-id")
            try:
                log.write(type="start")
            finally:
                log.close()
            self.assertTrue(os.path.basename(log.path).startswith("old-id_"))
            with open(log.path, encoding="utf-8") as handle:
                entry = json.loads(handle.readline())
        self.assertEqual(entry["session_id"], "old-id")

    def test_logger_captures_the_active_session_id(self):
        loop = object.__new__(m.ChatLoop)
        loop.ctx = types.SimpleNamespace(agentic_logging=True,
                                         agentic_log_retention_days=7)
        loop.conversation = types.SimpleNamespace(active_id="abc")
        with patch.object(m, "AgenticLogger") as logger_cls:
            loop._agentic_logger()
        logger_cls.assert_called_once_with(retention_days=7, session_id="abc")

    def test_logger_off_returns_none(self):
        loop = object.__new__(m.ChatLoop)
        loop.ctx = types.SimpleNamespace(agentic_logging=False)
        self.assertIsNone(loop._agentic_logger())


class TestCommandRegistry(unittest.TestCase):
    """`/rename` is a first-class command; `/save`/`/resume` document names."""

    def test_rename_is_registered_and_completable(self):
        self.assertIn("/rename", m.COMMANDS["rename"]["aliases"])
        self.assertIn("/rename", m.get_command_aliases())

    def test_usage_mentions_names(self):
        self.assertIn("name", m.COMMANDS["save"]["usage"])
        self.assertIn("name", m.COMMANDS["resume"]["usage"])
        self.assertIn("name", m.COMMANDS["sessions"]["description"])


if __name__ == '__main__':
    unittest.main(verbosity=2)
