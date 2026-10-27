#!/usr/bin/env python3
"""Unit tests for the agentic step-timeout escalation policy.

Covers qwen36_agentic Issue 2 (timeout escalation):
- State C: destructive last tool + a tool call pending in the timed-out
  generation -> abort (don't re-execute). Destructive last tool + prose
  narration (no tool call pending) -> continue with backoff.
- State A: no tool executed -> retry once, abort after two
- State B: tool executed -> exponential backoff 120->240->480, abort at max

Tests drive `ChatLoop._handle_agentic_timeout` directly against a fake context,
so no backend is required.
"""

import io
import os
import sys
import time
import threading
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ollamaquery2 as m


class FakeCtx:
    def __init__(self):
        self.agentic_step_timeout = 120
        self.agentic_timeout_max = 600
        self.agentic_progress_grace = 15
        self.agentic_max_thinking_tokens = 2048
        self.agentic_heartbeat_tokens = 10
        self.agentic_show_thinking = False
        self.agentic_consecutive_timeouts = 0
        self.agentic_has_executed_tool = False
        self.agentic_last_tool_name = ""
        self.lazy_tool = False


class TestAgenticTimeoutPolicy(unittest.TestCase):
    def setUp(self):
        self.ctx = FakeCtx()
        self.loop = object.__new__(m.ChatLoop)
        self.loop.ctx = self.ctx
        # Silence the warning/error output the policy prints to stderr.
        self._stderr = io.StringIO()
        self._patched = mock.patch('sys.stderr', self._stderr)
        self._patched.start()

    def tearDown(self):
        self._patched.stop()

    def _call(self, iteration=1, step_timeout=120, partial_text="", partial_thought="",
              elapsed_sec=None, partial_tokens=None, idle_sec=None):
        messages = []
        should_break, new_timeout = self.loop._handle_agentic_timeout(
            iteration, step_timeout, messages, partial_text=partial_text,
            partial_thought=partial_thought, elapsed_sec=elapsed_sec,
            partial_tokens=partial_tokens, idle_sec=idle_sec)
        return should_break, new_timeout, messages

    # --- State C: destructive last tool -------------------------------------

    def test_state_c_destructive_tool_aborts(self):
        self.ctx.agentic_has_executed_tool = True
        self.ctx.agentic_last_tool_name = 'run_command'
        # A tool call was in flight when the generation timed out — abort.
        should_break, new_timeout, messages = self._call(
            partial_text='{"tool": "run_command", "arguments": {"command": "make"}}')
        self.assertTrue(should_break)
        self.assertEqual(new_timeout, 120)
        self.assertEqual(messages, [])

    def test_state_c_patch_aborts(self):
        for tool in ('patch', 'edit_file'):
            self.ctx.agentic_has_executed_tool = True
            self.ctx.agentic_last_tool_name = tool
            should_break, _, _ = self._call(
                partial_text='{"tool": "%s", "arguments": {}}' % tool)
            self.assertTrue(should_break, f"{tool} should abort on timeout")

    def test_state_c_prose_narration_continues(self):
        """Model was mid-narration (no tool call pending) — the destructive tool
        already completed, so retry with backoff instead of aborting."""
        self.ctx.agentic_has_executed_tool = True
        self.ctx.agentic_last_tool_name = 'run_command'
        should_break, new_timeout, messages = self._call(
            partial_text="Let me rewrite the main loop. Step 1: read the request line.\nStep 2: parse headers.")
        self.assertFalse(should_break)
        self.assertEqual(new_timeout, 240)  # State B backoff, not abort
        self.assertEqual(len(messages), 1)  # nudge appended
        self.assertIn('continue your previous response', messages[0]['content'])

    def test_state_c_empty_partial_continues(self):
        """No content generated before the timeout — nothing to re-execute."""
        self.ctx.agentic_has_executed_tool = True
        self.ctx.agentic_last_tool_name = 'run_command'
        should_break, new_timeout, _ = self._call()
        self.assertFalse(should_break)
        self.assertEqual(new_timeout, 240)

    def test_state_c_prose_with_embedded_json_aborts(self):
        """Prose that ends with a bare JSON tool call still counts as pending."""
        self.ctx.agentic_has_executed_tool = True
        self.ctx.agentic_last_tool_name = 'run_command'
        should_break, _, _ = self._call(
            partial_text='Let me compile now. {"tool": "run_command", "arguments": {"command": "make"}}')
        self.assertTrue(should_break)

    # --- State A: no tool executed yet --------------------------------------

    def test_state_a_first_timeout_retries(self):
        self.ctx.agentic_has_executed_tool = False
        should_break, new_timeout, messages = self._call(step_timeout=120)
        self.assertFalse(should_break)
        self.assertEqual(new_timeout, 120)  # no escalation in state A
        self.assertEqual(self.ctx.agentic_consecutive_timeouts, 1)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]['role'], 'user')

    def test_state_a_second_timeout_aborts(self):
        self.ctx.agentic_has_executed_tool = False
        self._call(step_timeout=120)  # first timeout -> retry
        should_break, _, _ = self._call(step_timeout=120)  # second -> abort
        self.assertTrue(should_break)
        self.assertEqual(self.ctx.agentic_consecutive_timeouts, 2)

    # --- State B: a tool already ran ----------------------------------------

    def test_state_b_escalates_doubling(self):
        self.ctx.agentic_has_executed_tool = True
        self.ctx.agentic_last_tool_name = 'read_file'  # non-destructive
        should_break, new_timeout, messages = self._call(step_timeout=120)
        self.assertFalse(should_break)
        self.assertEqual(new_timeout, 240)
        self.assertEqual(len(messages), 1)

    def test_state_b_escalates_to_max(self):
        self.ctx.agentic_has_executed_tool = True
        self.ctx.agentic_last_tool_name = 'read_file'
        _, new_timeout, _ = self._call(step_timeout=240)
        self.assertEqual(new_timeout, 480)

    def test_state_b_raises_max_when_reached(self):
        self.ctx.agentic_has_executed_tool = True
        self.ctx.agentic_last_tool_name = 'read_file'
        # At the ceiling: the max itself doubles instead of aborting.
        should_break, new_timeout, _ = self._call(step_timeout=600)
        self.assertFalse(should_break)
        self.assertEqual(new_timeout, 1200)
        self.assertEqual(self.ctx.agentic_timeout_max, 1200)

    def test_state_b_doubling_capped_at_hard_ceiling(self):
        """Doubling `agentic_timeout_max` never exceeds AGENTIC_TIMEOUT_MAX_CEILING."""
        self.ctx.agentic_has_executed_tool = True
        self.ctx.agentic_last_tool_name = 'read_file'
        self.ctx.agentic_timeout_max = 4800
        should_break, new_timeout, _ = self._call(step_timeout=4800)
        self.assertFalse(should_break)
        self.assertEqual(new_timeout, m.AGENTIC_TIMEOUT_MAX_CEILING)
        self.assertEqual(self.ctx.agentic_timeout_max, m.AGENTIC_TIMEOUT_MAX_CEILING)

    def test_state_b_aborts_at_hard_ceiling(self):
        """Once the hard ceiling is reached, State B stops growing and aborts."""
        self.ctx.agentic_has_executed_tool = True
        self.ctx.agentic_last_tool_name = 'read_file'
        self.ctx.agentic_timeout_max = m.AGENTIC_TIMEOUT_MAX_CEILING
        should_break, new_timeout, messages = self._call(
            step_timeout=m.AGENTIC_TIMEOUT_MAX_CEILING)
        self.assertTrue(should_break)
        self.assertEqual(new_timeout, m.AGENTIC_TIMEOUT_MAX_CEILING)
        self.assertEqual(messages, [])

    def test_state_b_respects_custom_max(self):
        self.ctx.agentic_timeout_max = 240
        self.ctx.agentic_has_executed_tool = True
        self.ctx.agentic_last_tool_name = 'read_file'
        _, new_timeout, _ = self._call(step_timeout=120)
        self.assertEqual(new_timeout, 240)  # capped at custom max

    # --- Nudge message ------------------------------------------------------

    def test_nudge_message_instructs_continue(self):
        self.ctx.agentic_has_executed_tool = True
        self.ctx.agentic_last_tool_name = 'read_file'
        _, _, messages = self._call()
        self.assertIn('continue your previous response', messages[0]['content'])

    def test_timeout_nudge_not_persisted(self):
        """A timeout nudge must stay local to the loop: it is tagged
        `_system_nudge` so `_persist_agentic_history` excludes it from the
        persistent conversation (repeated timeouts must not pile into context)."""
        self.ctx.agentic_has_executed_tool = True
        self.ctx.agentic_last_tool_name = 'read_file'
        _, _, messages = self._call(partial_thought="partial reasoning")
        self.assertTrue(messages[0].get('_system_nudge'))

        self.ctx.context_window_size = 0
        self.ctx.calculate_context_tokens = lambda msgs: 0
        self.loop.messages = [
            {'role': 'system', 'content': 'sys'},
            {'role': 'user', 'content': 'q'},
        ]
        # The fixture is a bare loop (no tool_registry/runner), so the context
        # bar cannot build a real request. The subject here is the nudge, not
        # the bar — stub it rather than relying on the bar's own fallback.
        with mock.patch.object(m.WireView, 'refresh_context_bar', return_value=0):
            self.loop._persist_agentic_history(
                [{'role': 'system', 'content': 'sys'},
                 {'role': 'user', 'content': 'q'}] + messages, 2)
        self.assertFalse(any('interrupted by a timeout' in m.get('content', '')
                             for m in self.loop.messages))

    # --- Partial reasoning fed back to the model ----------------------------

    def test_nudge_includes_partial_thought(self):
        """The model's partial reasoning must be echoed back so it does not
        restart its analysis from scratch after a timeout (time loss)."""
        self.ctx.agentic_has_executed_tool = True
        self.ctx.agentic_last_tool_name = 'read_file'
        _, _, messages = self._call(partial_thought="Let me trace the send_response path.")
        self.assertIn('Your reasoning before the interruption', messages[0]['content'])
        self.assertIn('Let me trace the send_response path.', messages[0]['content'])

    def test_nudge_includes_partial_content_when_no_tool_pending(self):
        self.ctx.agentic_has_executed_tool = True
        self.ctx.agentic_last_tool_name = 'read_file'
        _, _, messages = self._call(partial_text="Let me rewrite the main loop.")
        self.assertIn('Your response before the interruption', messages[0]['content'])
        self.assertIn('Let me rewrite the main loop.', messages[0]['content'])

    def test_nudge_omits_partial_content_when_tool_pending(self):
        """A cut-off tool-call JSON prefix must NOT be echoed back — the model
        would continue the JSON from the middle and emit a malformed call."""
        self.ctx.agentic_has_executed_tool = True
        self.ctx.agentic_last_tool_name = 'read_file'  # non-destructive
        _, _, messages = self._call(
            partial_text='{"tool": "read_file", "arguments": {"path": "main.c"',
            partial_thought="I need to inspect the handler.")
        self.assertIn('Your reasoning before the interruption', messages[0]['content'])
        self.assertNotIn('Your response before the interruption', messages[0]['content'])
        self.assertNotIn('"tool"', messages[0]['content'])

    def test_nudge_truncates_partial_thought(self):
        """Reasoning longer than the cap is truncated to protect context."""
        self.ctx.agentic_has_executed_tool = True
        self.ctx.agentic_last_tool_name = 'read_file'
        _, _, messages = self._call(partial_thought="x" * 10000)
        self.assertIn('Your reasoning before the interruption', messages[0]['content'])
        self.assertIn('x' * 4000, messages[0]['content'])
        self.assertNotIn('x' * 5000, messages[0]['content'])

    def test_timeout_nudge_helper_empty(self):
        nudge = self.loop._timeout_nudge("", "", False)
        self.assertIn('continue your previous response', nudge)

    def test_timeout_nudge_helper_mixes_thought_and_content(self):
        nudge = self.loop._timeout_nudge("content here", "thought here", False)
        self.assertIn('content here', nudge)
        self.assertIn('thought here', nudge)

    def test_timeout_nudge_helper_drops_content_on_pending_tool(self):
        nudge = self.loop._timeout_nudge('{"tool": "x"}', "thought here", True)
        self.assertNotIn('{"tool": "x"}', nudge)
        self.assertIn('thought here', nudge)

    # --- F3: progress-aware State A extension + budget-aware nudge -----------

    def test_state_a_extends_when_generating(self):
        """F3: State A, but the model was actively generating (idle < grace,
        tok/s > 0) -> extend the budget instead of the old same-budget retry."""
        self.ctx.agentic_has_executed_tool = False
        should_break, new_timeout, messages = self._call(
            step_timeout=120, partial_thought="x" * 500,
            elapsed_sec=120, partial_tokens=1000, idle_sec=2)
        self.assertFalse(should_break)
        self.assertEqual(new_timeout, 240)
        self.assertIn('tokens/second', messages[0]['content'])

    def test_state_a_no_progress_still_same_budget(self):
        """F3: State A with no recent activity (stalled) -> old conservative
        behavior (same-budget retry)."""
        self.ctx.agentic_has_executed_tool = False
        should_break, new_timeout, _ = self._call(
            step_timeout=120, partial_thought="x" * 500,
            elapsed_sec=120, partial_tokens=1000, idle_sec=120)
        self.assertFalse(should_break)
        self.assertEqual(new_timeout, 120)

    def test_state_a_progress_extends_until_max(self):
        """F3: progress extension caps at agentic_timeout_max; at the ceiling the
        ceiling itself doubles rather than the step being retried/aborted."""
        self.ctx.agentic_has_executed_tool = False
        self.ctx.agentic_timeout_max = 480
        # 120 -> 240 (progress path extends)
        _, new_timeout, _ = self._call(
            step_timeout=120, partial_tokens=1000, elapsed_sec=120, idle_sec=1)
        self.assertEqual(new_timeout, 240)
        self.ctx.agentic_consecutive_timeouts = 0  # isolate the at-max assertion
        # at max: the max doubles and the step budget follows it
        should_break, new_timeout, _ = self._call(
            step_timeout=480, partial_tokens=1000, elapsed_sec=480, idle_sec=1)
        self.assertFalse(should_break)
        self.assertEqual(new_timeout, 960)
        self.assertEqual(self.ctx.agentic_timeout_max, 960)

    def test_state_a_at_hard_ceiling_aborts_after_two(self):
        """At the hard ceiling State A cannot grow further: the first timeout
        retries at the same budget, the second aborts."""
        self.ctx.agentic_has_executed_tool = False
        self.ctx.agentic_timeout_max = m.AGENTIC_TIMEOUT_MAX_CEILING
        should_break, new_timeout, _ = self._call(
            step_timeout=m.AGENTIC_TIMEOUT_MAX_CEILING,
            partial_tokens=1000, elapsed_sec=m.AGENTIC_TIMEOUT_MAX_CEILING, idle_sec=1)
        self.assertFalse(should_break)
        self.assertEqual(new_timeout, m.AGENTIC_TIMEOUT_MAX_CEILING)
        self.assertEqual(self.ctx.agentic_timeout_max, m.AGENTIC_TIMEOUT_MAX_CEILING)
        should_break, _, _ = self._call(
            step_timeout=m.AGENTIC_TIMEOUT_MAX_CEILING,
            partial_tokens=1000, elapsed_sec=m.AGENTIC_TIMEOUT_MAX_CEILING, idle_sec=1)
        self.assertTrue(should_break)

    def test_nudge_metrics_block(self):
        """F3: the nudge reports generation speed, the expired timeout, and the
        new budget in seconds/tokens so the model can self-throttle."""
        nudge = self.loop._timeout_nudge("", "thought here", False,
            metrics={"tok_per_sec": 8.3, "expired_sec": 120, "budget_sec": 240})
        self.assertIn('~8.3 tokens/second', nudge)
        self.assertIn('cut off after 120s', nudge)
        self.assertIn('new budget: 240s (~1992 tokens', nudge)

    # --- F4: nudge echoes the TAIL of the reasoning --------------------------

    def test_nudge_tail_caps_thought(self):
        """F4: echoed reasoning is capped from the TAIL (where the model was cut
        off), not the head, so a resumed step continues instead of re-deriving."""
        thought = "H" * 4500 + "TAIL_SENTINEL"
        nudge = self.loop._timeout_nudge("", thought, False)
        self.assertIn('TAIL_SENTINEL', nudge)
        self.assertIn('earlier reasoning omitted', nudge)
        self.assertNotIn('H' * 4500 + 'TAIL_SENTINEL', nudge)

    # --- F5: thinking-length detection + steer ------------------------------

    def test_nudge_thinking_capped_steer(self):
        """F5: when thinking exceeded the cap, the nudge tells the model to be
        more concise and move to a tool call / final answer."""
        nudge = self.loop._timeout_nudge("", "thought", False,
            metrics={"tok_per_sec": 8.0, "expired_sec": 120, "budget_sec": 240,
                     "thinking_capped": True, "thinking_cap": 2048})
        self.assertIn('2048-token cap', nudge)
        self.assertIn('be more concise', nudge)

    def test_step_feedback_tracks_thinking_cap(self):
        """F5: feed() accumulates a token estimate and flags thinking_capped
        once it crosses agentic_max_thinking_tokens."""
        feedback = self.loop._make_agentic_step_feedback()
        feedback.feed("x" * 9000, "", False)   # ~2250 tokens > 2048 cap
        self.assertEqual(feedback.thinking_tokens, 2250)
        self.assertTrue(feedback.thinking_capped)
        feedback.finalize()

    def test_step_feedback_under_cap_not_capped(self):
        """F5: short thinking stays under the cap."""
        feedback = self.loop._make_agentic_step_feedback()
        feedback.feed("brief reasoning", "", False)
        self.assertGreater(feedback.thinking_tokens, 0)
        self.assertFalse(feedback.thinking_capped)
        feedback.finalize()

    def _wait_for(self, predicate, timeout=3.0):
        """Poll `predicate()` until it returns truthy or `timeout` elapses."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if predicate():
                return True
            time.sleep(0.05)
        return False

    def test_watchdog_liveness_without_chunks(self):
        """The idle-fallback watchdog emits a liveness dot when no chunks arrive.

        Long tool-call generations (e.g. llama.cpp buffering the response)
        stream no chunks for tens of seconds; the token-driven heartbeat has no
        tokens to count, so after ~2s of silence the watchdog ticks ~1/s.
        """
        feedback = self.loop._make_agentic_step_feedback()
        self.assertEqual(self._stderr.getvalue(), "")
        self.assertTrue(self._wait_for(lambda: "." in self._stderr.getvalue(), timeout=5.0))
        feedback.finalize()
        dots = self._stderr.getvalue().count(".")
        time.sleep(1.1)
        self.assertEqual(self._stderr.getvalue().count("."), dots,
                         "watchdog must stop writing after finalize")

    def test_watchdog_suppressed_while_thinking_streams(self):
        """No dots while reasoning is being streamed live to the terminal."""
        self.ctx.agentic_show_thinking = True
        feedback = self.loop._make_agentic_step_feedback()
        feedback.feed("reasoning text", "", False)
        self.assertFalse(self._wait_for(lambda: "." in self._stderr.getvalue(), timeout=1.5))
        self.assertIn("<thinking>", self._stderr.getvalue())
        feedback.finalize()

    def test_token_dots_suppressed_when_thinking_streams(self):
        """Token-driven dots must not fire when thinking is streamed live — even
        after the <thinking> block closes on content arrival (regression: hidden
        content streamed ~1 dot per 10 tokens while the user was in thinking
        mode, reading as a dot wall after </thinking>)."""
        self.ctx.agentic_show_thinking = True
        feedback = self.loop._make_agentic_step_feedback()
        feedback.feed("reasoning text", "", False)          # opens the block
        feedback.feed("", "answer content " * 50, False)    # closes it, ~125 tokens
        feedback.feed("", "more content", False)            # more hidden content
        self.assertFalse(self._wait_for(lambda: "." in self._stderr.getvalue(), timeout=1.5))
        self.assertIn("</thinking>", self._stderr.getvalue())
        feedback.finalize()
        self.assertEqual(self._stderr.getvalue().count("."), 0,
                         "no dot noise when the user chose live reasoning")

    def test_content_streams_live_when_thinking_streams(self):
        """With thinking on, content streams live to stdout (chat-style: the user
        chose to see thinking AND what the model says), and the state flags
        content_streamed so finalize can skip the re-generation."""
        self.ctx.agentic_show_thinking = True
        out = io.StringIO()
        with mock.patch('sys.stdout', out):
            feedback = self.loop._make_agentic_step_feedback()
            feedback.feed("reasoning text", "", False)
            feedback.feed("", "visible answer text", False)
            feedback.feed("", " and more", False)
            feedback.finalize()
        self.assertIn("<thinking>", self._stderr.getvalue())
        self.assertIn("</thinking>", self._stderr.getvalue())
        self.assertIn("visible answer text and more", out.getvalue())
        self.assertTrue(feedback.content_streamed)

    def test_watchdog_closes_silent_thinking_and_shows_dots(self):
        """A quiet thinking phase (model switching to a native tool call, whose
        deltas carry no content) must not suppress the liveness dots: the
        watchdog closes the `<thinking>` block and resumes ticking."""
        self.ctx.agentic_show_thinking = True
        feedback = self.loop._make_agentic_step_feedback()
        feedback.feed("reasoning text", "", False)
        self.assertIn("<thinking>", self._stderr.getvalue())
        self.assertTrue(self._wait_for(lambda: "." in self._stderr.getvalue(), timeout=5.0))
        self.assertIn("</thinking>", self._stderr.getvalue())
        feedback.finalize()
        dots = self._stderr.getvalue().count(".")
        time.sleep(1.1)
        self.assertEqual(self._stderr.getvalue().count("."), dots,
                         "watchdog must stop writing after finalize")

    def test_token_driven_heartbeat(self):
        """A dot fires per ~agentic_heartbeat_tokens streamed, not per second."""
        feedback = self.loop._make_agentic_step_feedback()
        self.assertEqual(self._stderr.getvalue(), "")
        feedback.feed("", "x" * 40, False)   # ~10 tokens -> one dot
        self.assertEqual(self._stderr.getvalue(), ".")
        feedback.feed("", "x" * 40, False)   # another ~10 tokens -> second dot
        self.assertEqual(self._stderr.getvalue(), "..")
        feedback.feed("", "x" * 16, False)   # ~4 tokens -> under threshold, no dot
        self.assertEqual(self._stderr.getvalue(), "..")
        feedback.finalize()

    def test_token_driven_heartbeat_respects_threshold(self):
        """The per-dot token count follows ctx.agentic_heartbeat_tokens."""
        self.ctx.agentic_heartbeat_tokens = 5
        feedback = self.loop._make_agentic_step_feedback()
        feedback.feed("", "x" * 40, False)   # ~10 tokens / 5 per dot -> two dots
        self.assertEqual(self._stderr.getvalue(), "..")
        feedback.finalize()

    def test_extract_json_balanced_is_staticmethod(self):
        """Regression: `_extract_json_balanced` must be callable unbound — it is
        invoked as `ChatLoop._extract_json_balanced(text, start)` from the
        staticmethod `_looks_like_truncated_tool_call` (previously passed None as
        self, a landmine for any future edit that touched self)."""
        self.assertEqual(m.ChatLoop._extract_json_balanced('{"a": 1}', 0), '{"a": 1}')
        self.assertIsNone(m.ChatLoop._extract_json_balanced('{"a": 1', 0))

    def test_looks_like_truncated_tool_call(self):
        self.assertTrue(m.ChatLoop._looks_like_truncated_tool_call('{"tool": "run_command"'))
        self.assertFalse(m.ChatLoop._looks_like_truncated_tool_call('{"tool": "run_command"}'))


class TestTimeoutPolicyInIsolation(unittest.TestCase):
    """The escalation policy is usable without a ChatLoop (it was extracted out
    of the loop precisely so these rules can be driven directly)."""

    def setUp(self):
        self.ctx = FakeCtx()
        self.parser = m.ToolCallParser(self.ctx)
        self.policy = m.AgenticTimeoutPolicy(self.ctx, self.parser)
        self._stderr = io.StringIO()
        self._patched = mock.patch('sys.stderr', self._stderr)
        self._patched.start()

    def tearDown(self):
        self._patched.stop()

    def test_state_b_backs_off_and_appends_one_nudge(self):
        self.ctx.agentic_has_executed_tool = True
        self.ctx.agentic_last_tool_name = 'read_file'
        messages = []
        should_break, new_timeout = self.policy.on_timeout(2, 120, messages)
        self.assertFalse(should_break)
        self.assertEqual(new_timeout, 240)
        self.assertEqual(len(messages), 1)
        self.assertTrue(messages[0]['_system_nudge'])

    def test_state_c_aborts_when_a_tool_call_is_pending(self):
        self.ctx.agentic_has_executed_tool = True
        self.ctx.agentic_last_tool_name = 'run_command'
        should_break, _ = self.policy.on_timeout(
            1, 120, [], partial_text='{"tool": "run_command", "arguments": {}}')
        self.assertTrue(should_break)

    def test_a_broken_collaborator_surfaces_instead_of_reading_as_no_call(self):
        """`has_tool_call` narrows its catch to TypeError/ValueError on purpose.

        A blind `except Exception` here turned a missing attribute into "no tool
        call pending", silently downgrading State C from ABORT to BACKOFF — the
        failure mode the state-C tests exist to catch.
        """
        broken = mock.Mock()
        broken.has_tool_call.side_effect = AttributeError("no parser")
        policy = m.AgenticTimeoutPolicy(self.ctx, broken)
        self.ctx.agentic_has_executed_tool = True
        self.ctx.agentic_last_tool_name = 'run_command'
        with self.assertRaises(AttributeError):
            policy.on_timeout(1, 120, [], partial_text='{"tool": "run_command"}')


class TestStepFeedbackInIsolation(unittest.TestCase):
    """StepFeedback replaced the free-form `state` dict of one ReAct step.

    The fields are attributes now, so a typo is an AttributeError instead of a
    silently missing key; feeding works without start()/finalize() (no thread),
    which is what makes these tests instant.
    """

    def setUp(self):
        self.ctx = FakeCtx()
        self._stderr = io.StringIO()
        self._patched = mock.patch('sys.stderr', self._stderr)
        self._patched.start()

    def tearDown(self):
        self._patched.stop()

    def test_accumulates_named_fields(self):
        feedback = m.StepFeedback(self.ctx, show_thinking=False)
        feedback.feed("reasoning", "visible", False)
        self.assertEqual(feedback.thought, "reasoning")
        self.assertEqual(feedback.content, "visible")
        self.assertGreater(feedback.thinking_tokens, 0)
        self.assertFalse(hasattr(feedback, "tok_acc"), "private accumulator leaked")

    def test_unknown_field_is_an_attribute_error(self):
        """The point of the extraction: no silent missing-key reads."""
        feedback = m.StepFeedback(self.ctx, show_thinking=False)
        with self.assertRaises(AttributeError):
            _ = feedback.tought          # typo of `thought`

    def test_idle_sec_tracks_the_last_chunk(self):
        feedback = m.StepFeedback(self.ctx, show_thinking=False)
        feedback.last_activity -= 5.0
        self.assertGreaterEqual(feedback.idle_sec(), 5.0)
        feedback.feed("", "", False)     # empty deltas still count as activity
        self.assertLess(feedback.idle_sec(), 1.0)

    def test_factory_returns_a_started_feedback(self):
        loop = object.__new__(m.ChatLoop)
        loop.ctx = self.ctx
        feedback = loop._make_agentic_step_feedback()
        self.assertIsInstance(feedback, m.StepFeedback)
        self.assertFalse(feedback.cancelled)
        feedback.finalize()
        self.assertTrue(feedback.cancelled)


class TestStepFeedbackProgress(unittest.TestCase):
    """Tool-call-only deltas count as generation activity; the watchdog tracks
    visible output rather than raw chunk arrival.

    Regression: native tool-call deltas carry no thought/content, so (a) the
    timeout progress signal saw a busy tool-call write (~120s of silent deltas
    at ~30 tok/s) as a stalled step and aborted, and (b) any chunk-driven
    heartbeat would have starved. `last_activity` now tracks every chunk
    (progress signal) while the watchdog idles on `last_visible` (rendered
    output only).
    """

    def setUp(self):
        self.ctx = FakeCtx()
        self.loop = object.__new__(m.ChatLoop)
        self.loop.ctx = self.ctx
        self._stderr = io.StringIO()
        self._stdout = io.StringIO()
        self._patched_err = mock.patch('sys.stderr', self._stderr)
        self._patched_out = mock.patch('sys.stdout', self._stdout)
        self._patched_err.start()
        self._patched_out.start()

    def tearDown(self):
        self._patched_err.stop()
        self._patched_out.stop()

    def _wait_for(self, predicate, timeout=3.0):
        """Poll `predicate()` until it returns truthy or `timeout` elapses."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if predicate():
                return True
            time.sleep(0.05)
        return False

    def test_empty_tool_call_deltas_count_as_activity(self):
        """feedback.feed('', '', False) refreshes last_activity — the timeout policy
        must see a busy tool-call write as active generation, not a stall."""
        feedback = self.loop._make_agentic_step_feedback()
        feedback.last_activity -= 5.0
        stale = feedback.last_activity
        feedback.feed("", "", False)
        self.assertGreater(feedback.last_activity, stale)
        feedback.finalize()

    def test_watchdog_dots_continue_during_silent_tool_call_deltas(self):
        """Empty deltas refresh activity but render nothing — the watchdog must
        keep dotting while a tool call streams invisibly."""
        feedback = self.loop._make_agentic_step_feedback()
        seen = {"stop": False}

        def feed():
            # Simulate a Strata tool-call write: chunks every 0.2s, all empty.
            while not seen["stop"]:
                feedback.feed("", "", False)
                time.sleep(0.2)

        feeder = threading.Thread(target=feed, daemon=True)
        feeder.start()
        try:
            self.assertTrue(
                self._wait_for(lambda: "." in self._stderr.getvalue(), timeout=5.0),
                "watchdog stopped dotting during tool-call-only deltas")
        finally:
            seen["stop"] = True
            feedback.finalize()
            feeder.join(timeout=2.0)

    def test_visible_output_blocks_watchdog(self):
        """Rendered content keeps last_visible fresh — no watchdog dots while
        the model streams visible text."""
        self.ctx.agentic_show_thinking = True
        feedback = self.loop._make_agentic_step_feedback()
        seen = {"stop": False}

        def feed():
            while not seen["stop"]:
                feedback.feed("", "streaming visible text ", False)
                time.sleep(0.2)

        feeder = threading.Thread(target=feed, daemon=True)
        feeder.start()
        try:
            time.sleep(3.0)   # > idle_grace: only rendered output keeps dots away
            self.assertNotIn(".", self._stderr.getvalue())
        finally:
            seen["stop"] = True
            feedback.finalize()
            feeder.join(timeout=2.0)


class TestTruncationPredicateNarrowing(unittest.TestCase):
    """`looks_like_truncated_tool_call` must be tolerant of bad input, not of bugs.

    It answers "is the model cut off mid tool call?", which decides whether the
    timeout nudge echoes partial content. It used to wrap the two pure helpers in
    `except Exception: return False`, so a defect (renamed helper, changed
    signature) silently answered "not truncated" — the failure mode that once
    downgraded the timeout policy from ABORT to BACKOFF in `has_tool_call`. The
    catch was dead anyway: both helpers are total over `str`, so it is gone.
    """

    def test_empty_input_returns_false(self):
        """Empty/None text is not a truncated call."""
        self.assertFalse(m.looks_like_truncated_tool_call(None))
        self.assertFalse(m.looks_like_truncated_tool_call("   "))

    def test_balanced_call_is_not_truncated(self):
        self.assertFalse(m.looks_like_truncated_tool_call('{"tool": "read_file", "args": {}}'))

    def test_unbalanced_opening_is_truncated(self):
        self.assertTrue(m.looks_like_truncated_tool_call('{"tool": "read_file", "args": {"path"'))

    def test_helper_defect_surfaces(self):
        """An AttributeError from a helper is a bug, not 'not truncated'."""
        with mock.patch.object(m, 'extract_json_balanced',
                               side_effect=AttributeError('renamed helper')):
            with self.assertRaises(AttributeError):
                m.looks_like_truncated_tool_call('{"tool": "read_file"')


if __name__ == '__main__':
    unittest.main(verbosity=2)
