#!/usr/bin/env python3
"""Unit tests for the /compact subcommand system.

Covers the dispatcher and helpers added to ChatLoop:
- /compact threshold <v>   (fraction or percent)
- /compact force           (force compaction under budget)
- /compact force <index>   (truncate / LLM-summarize a single message)
- /compact list / status   (no mutation)
"""

import io
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ollamaquery2 as m


class FakeCtx:
    backend = 'llamacpp'
    model = 'test-model'
    context_window_size = 4096
    compaction_threshold = m.COMPACTION_THRESHOLD

    def __init__(self):
        self.current_context_tokens = 0

    def estimate_tokens(self, text):
        return max(1, len(text))

    def stamp_tokens(self, msg):
        content = msg.get('content', '')
        if isinstance(content, list):
            content = "".join(p.get('text', '') for p in content if isinstance(p, dict))
        msg['_tokens'] = (len(content) // 4) + 2
        return msg

    def calculate_context_tokens(self, messages):
        total = 0
        for msg in messages:
            total += msg.get('_tokens', self.estimate_tokens(msg.get('content', '')) + 2)
        return total


class FakeQueryHandler:
    def __init__(self):
        self.called_with_conversation = False

    def query_sync(self, messages, model, **kwargs):
        self.called_with_conversation = 'EARLIER CONVERSATION TO SUMMARIZE' in messages[-1]['content']
        return {'choices': [{'message': {'content': 'SUMMARY'}}]}


def _loop():
    loop = object.__new__(m.ChatLoop)
    loop.ctx = FakeCtx()
    loop.query_handler = FakeQueryHandler()
    loop.messages = [{'role': 'system', 'content': 'sys'}]
    return loop


class TestCompactThreshold(unittest.TestCase):
    def test_fraction(self):
        loop = _loop()
        loop._compact_threshold(['/compact', 'threshold', '0.6'])
        self.assertAlmostEqual(loop.ctx.compaction_threshold, 0.6)

    def test_percent(self):
        loop = _loop()
        loop._compact_threshold(['/compact', 'threshold', '60'])
        self.assertAlmostEqual(loop.ctx.compaction_threshold, 0.6)

    def test_invalid(self):
        loop = _loop()
        loop._compact_threshold(['/compact', 'threshold', 'abc'])
        self.assertEqual(loop.ctx.compaction_threshold, m.COMPACTION_THRESHOLD)

    def test_out_of_range(self):
        loop = _loop()
        loop._compact_threshold(['/compact', 'threshold', '150'])  # 150% -> 1.5 -> invalid
        self.assertEqual(loop.ctx.compaction_threshold, m.COMPACTION_THRESHOLD)


class TestCompactForce(unittest.TestCase):
    def test_force_compacts_under_budget(self):
        ctx = FakeCtx()
        # Small conversation that would normally be under budget.
        msgs = [{'role': 'system', 'content': 'sys', '_tokens': 3}]
        msgs += [{'role': 'user', 'content': f'm{i}', '_tokens': 2} for i in range(10)]
        # Non-forced: under budget -> unchanged.
        not_forced = m.compact_messages(msgs, ctx, target_tokens=100000, keep_recent=2)
        self.assertEqual(not_forced, msgs)
        # Forced: compacts anyway.
        forced = m.compact_messages(msgs, ctx, target_tokens=100000, keep_recent=2, force=True)
        self.assertLess(len(forced), len(msgs))
        self.assertIn('[CONVERSATION HISTORY COMPACTED]', forced[1]['content'])


class TestCompactForceMessage(unittest.TestCase):
    def test_truncates_large_non_tool_message(self):
        loop = _loop()
        loop.messages.append({'role': 'user', 'content': 'x' * 5000})
        loop._compact_force_message('1')
        self.assertLess(len(loop.messages[1]['content']), 5000)
        self.assertIn('truncated', loop.messages[1]['content'])

    def test_llm_summarizes_tool_message(self):
        loop = _loop()
        loop.messages.append({'role': 'tool', 'name': 'read_file', 'content': 'y' * 5000})
        loop._compact_force_message('1')
        self.assertIn('[Summarized tool result (read_file)]', loop.messages[1]['content'])

    def test_out_of_range(self):
        loop = _loop()
        loop.messages.append({'role': 'user', 'content': 'x' * 5000})
        loop._compact_force_message('99')
        self.assertEqual(len(loop.messages), 2)  # unchanged

    def test_already_small(self):
        loop = _loop()
        loop.messages.append({'role': 'user', 'content': 'short'})
        loop._compact_force_message('1')
        self.assertEqual(loop.messages[1]['content'], 'short')


class TestCompactDispatch(unittest.TestCase):
    def test_force_index_routes_to_force_message(self):
        loop = _loop()
        loop.messages.append({'role': 'user', 'content': 'z' * 5000})
        loop.run_handle_compact('/compact force 1')
        self.assertIn('truncated', loop.messages[1]['content'])

    def test_threshold_routes_to_threshold(self):
        loop = _loop()
        loop.run_handle_compact('/compact threshold 60')
        self.assertAlmostEqual(loop.ctx.compaction_threshold, 0.6)

    def test_list_does_not_mutate(self):
        loop = _loop()
        loop.messages.append({'role': 'user', 'content': 'hello'})
        before = list(loop.messages)
        loop.run_handle_compact('/compact list')
        self.assertEqual(loop.messages, before)

    def test_status_does_not_mutate(self):
        loop = _loop()
        before = list(loop.messages)
        loop.run_handle_compact('/compact status')
        self.assertEqual(loop.messages, before)


class TestCompactLlm(unittest.TestCase):
    def _loop(self):
        loop = _loop()
        loop.messages += [{'role': 'user', 'content': f'q{i}'} for i in range(8)]
        loop.messages += [{'role': 'assistant', 'content': 'answer'}]
        return loop

    def test_llm_compacts_conversation(self):
        loop = self._loop()
        loop.run_handle_compact('/compact llm')
        self.assertIn(m.LLM_COMPACT_SUMMARY_MARKER, loop.messages[1]['content'])
        self.assertLess(len(loop.messages), 12)
        self.assertTrue(loop.query_handler.called_with_conversation)

    def test_llm_under_budget_still_summarizes(self):
        # /compact llm is explicit: even a small conversation gets summarized.
        loop = self._loop()
        loop.run_handle_compact('/compact llm')
        self.assertIn(m.LLM_COMPACT_SUMMARY_MARKER, loop.messages[1]['content'])


class _FourToOneCtx(FakeCtx):
    """More realistic token model: ~4 chars per token, consistent with stamping."""

    def estimate_tokens(self, text):
        if not text:
            return 0
        return max(1, len(text) // 4) + 2

    def stamp_tokens(self, msg):
        content = msg.get('content', '')
        if isinstance(content, list):
            content = "".join(p.get('text', '') for p in content if isinstance(p, dict))
        msg['_tokens'] = (len(content) // 4) + 2
        return msg


class TestCompactTruncateLargest(unittest.TestCase):
    def _loop_with_ctx(self, ctx):
        loop = _loop()
        loop.ctx = ctx
        return loop

    def test_truncates_giant_message_to_budget(self):
        ctx = _FourToOneCtx()
        loop = self._loop_with_ctx(ctx)
        loop.messages.append({'role': 'user', 'content': 'x' * 20000})
        loop.messages.append({'role': 'assistant', 'content': 'small'})
        total_before = ctx.calculate_context_tokens(loop.messages)
        target = 3000
        self.assertGreater(total_before, target)
        updated = loop._compact_truncate_largest(target)
        self.assertLess(updated, total_before)
        self.assertIn('truncated', loop.messages[1]['content'])

    def test_noop_when_under_budget(self):
        loop = _loop()
        loop.messages.append({'role': 'user', 'content': 'small'})
        total = loop.ctx.calculate_context_tokens(loop.messages)
        updated = loop._compact_truncate_largest(target=1000000)
        self.assertEqual(updated, total)
        self.assertNotIn('truncated', loop.messages[1]['content'])

    def test_noop_when_excess_overshoots_content(self):
        # 1:1 estimate makes excess*4 > content length → guarded no-op.
        loop = _loop()
        loop.messages.append({'role': 'user', 'content': 'x' * 20000})
        total = loop.ctx.calculate_context_tokens(loop.messages)
        updated = loop._compact_truncate_largest(target=1000)
        self.assertEqual(updated, total)
        self.assertNotIn('truncated', loop.messages[1]['content'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
