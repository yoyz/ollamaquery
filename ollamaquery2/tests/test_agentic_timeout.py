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
        """F5: on_chunk accumulates a token estimate and flags thinking_capped
        once it crosses agentic_max_thinking_tokens."""
        on_chunk, finalize, state = self.loop._make_agentic_step_feedback()
        on_chunk("x" * 9000, "", False)   # ~2250 tokens > 2048 cap
        self.assertEqual(state["thinking_tokens"], 2250)
        self.assertTrue(state["thinking_capped"])
        finalize()

    def test_step_feedback_under_cap_not_capped(self):
        """F5: short thinking stays under the cap."""
        on_chunk, finalize, state = self.loop._make_agentic_step_feedback()
        on_chunk("brief reasoning", "", False)
        self.assertGreater(state["thinking_tokens"], 0)
        self.assertFalse(state["thinking_capped"])
        finalize()

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
        on_chunk, finalize, state = self.loop._make_agentic_step_feedback()
        self.assertEqual(self._stderr.getvalue(), "")
        self.assertTrue(self._wait_for(lambda: "." in self._stderr.getvalue(), timeout=5.0))
        finalize()
        dots = self._stderr.getvalue().count(".")
        time.sleep(1.1)
        self.assertEqual(self._stderr.getvalue().count("."), dots,
                         "watchdog must stop writing after finalize")

    def test_watchdog_suppressed_while_thinking_streams(self):
        """No dots while reasoning is being streamed live to the terminal."""
        self.ctx.agentic_show_thinking = True
        on_chunk, finalize, state = self.loop._make_agentic_step_feedback()
        on_chunk("reasoning text", "", False)
        self.assertFalse(self._wait_for(lambda: "." in self._stderr.getvalue(), timeout=1.5))
        self.assertIn("<thinking>", self._stderr.getvalue())
        finalize()

    def test_watchdog_closes_silent_thinking_and_shows_dots(self):
        """A quiet thinking phase (model switching to a native tool call, whose
        deltas carry no content) must not suppress the liveness dots: the
        watchdog closes the `<thinking>` block and resumes ticking."""
        self.ctx.agentic_show_thinking = True
        on_chunk, finalize, state = self.loop._make_agentic_step_feedback()
        on_chunk("reasoning text", "", False)
        self.assertIn("<thinking>", self._stderr.getvalue())
        self.assertTrue(self._wait_for(lambda: "." in self._stderr.getvalue(), timeout=5.0))
        self.assertIn("</thinking>", self._stderr.getvalue())
        finalize()
        dots = self._stderr.getvalue().count(".")
        time.sleep(1.1)
        self.assertEqual(self._stderr.getvalue().count("."), dots,
                         "watchdog must stop writing after finalize")

    def test_token_driven_heartbeat(self):
        """A dot fires per ~agentic_heartbeat_tokens streamed, not per second."""
        on_chunk, finalize, state = self.loop._make_agentic_step_feedback()
        self.assertEqual(self._stderr.getvalue(), "")
        on_chunk("", "x" * 40, False)   # ~10 tokens -> one dot
        self.assertEqual(self._stderr.getvalue(), ".")
        on_chunk("", "x" * 40, False)   # another ~10 tokens -> second dot
        self.assertEqual(self._stderr.getvalue(), "..")
        on_chunk("", "x" * 16, False)   # ~4 tokens -> under threshold, no dot
        self.assertEqual(self._stderr.getvalue(), "..")
        finalize()

    def test_token_driven_heartbeat_respects_threshold(self):
        """The per-dot token count follows ctx.agentic_heartbeat_tokens."""
        self.ctx.agentic_heartbeat_tokens = 5
        on_chunk, finalize, state = self.loop._make_agentic_step_feedback()
        on_chunk("", "x" * 40, False)   # ~10 tokens / 5 per dot -> two dots
        self.assertEqual(self._stderr.getvalue(), "..")
        finalize()

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


if __name__ == '__main__':
    unittest.main(verbosity=2)
