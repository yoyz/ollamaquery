#!/usr/bin/env python3
"""Unit tests for the agentic step-timeout escalation policy.

Covers qwen36_agentic Issue 2 (timeout escalation):
- State C: destructive last tool -> abort immediately
- State A: no tool executed -> retry once, abort after two
- State B: tool executed -> exponential backoff 120->240->480, abort at max

Tests drive `ChatLoop._handle_agentic_timeout` directly against a fake context,
so no backend is required.
"""

import io
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ollamaquery2 as m


class FakeCtx:
    def __init__(self):
        self.agentic_step_timeout = 120
        self.agentic_timeout_max = 480
        self.agentic_consecutive_timeouts = 0
        self.agentic_has_executed_tool = False
        self.agentic_last_tool_name = ""


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

    def _call(self, iteration=1, step_timeout=120):
        messages = []
        should_break, new_timeout = self.loop._handle_agentic_timeout(iteration, step_timeout, messages)
        return should_break, new_timeout, messages

    # --- State C: destructive last tool -------------------------------------

    def test_state_c_destructive_tool_aborts(self):
        self.ctx.agentic_has_executed_tool = True
        self.ctx.agentic_last_tool_name = 'run_command'
        should_break, new_timeout, messages = self._call()
        self.assertTrue(should_break)
        self.assertEqual(new_timeout, 120)
        self.assertEqual(messages, [])

    def test_state_c_patch_aborts(self):
        for tool in ('patch', 'edit_file'):
            self.ctx.agentic_has_executed_tool = True
            self.ctx.agentic_last_tool_name = tool
            should_break, _, _ = self._call()
            self.assertTrue(should_break, f"{tool} should abort on timeout")

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

    def test_state_b_aborts_at_max(self):
        self.ctx.agentic_has_executed_tool = True
        self.ctx.agentic_last_tool_name = 'read_file'
        should_break, new_timeout, _ = self._call(step_timeout=480)
        self.assertTrue(should_break)
        self.assertEqual(new_timeout, 480)

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


if __name__ == '__main__':
    unittest.main(verbosity=2)
