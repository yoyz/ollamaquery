#!/usr/bin/env python3
"""Unit tests for context compaction in ollamaquery2.py.

Covers the compaction correctness cluster (qwen36 improvements D/E/G):
- Q6: compacted summary message gets `_tokens` stamped
- Q4: budget calculation prefers cached `_tokens` over estimates
- Q7: role alternation safety (assistant placeholder between two user messages)

These are pure-function tests with no backend dependency — `compact_messages`
is exercised against a deterministic FakeContext stub.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ollamaquery2 as m


class FakeContext:
    """Deterministic stand-in for CommandContext's token API.

    `estimate_tokens` deliberately over-estimates (1 char == 1 token) so tests
    can distinguish exact cached counts from heuristic estimates.
    """

    def __init__(self, context_window_size=0):
        self.context_window_size = context_window_size

    def estimate_tokens(self, text):
        if not text:
            return 0
        return max(1, len(text))

    def stamp_tokens(self, msg):
        if '_tokens' in msg:
            return msg
        content = msg.get('content', '')
        if isinstance(content, list):
            content = "".join(p.get('text', '') for p in content if isinstance(p, dict))
        msg['_tokens'] = (len(content) // 4) + 2
        return msg

    def calculate_context_tokens(self, messages):
        total = 0
        for msg in messages:
            if '_tokens' in msg:
                total += msg['_tokens']
            else:
                total += self.estimate_tokens(msg.get('content', '')) + 2
        return total


def _msg(role, content):
    return {'role': role, 'content': content}


class TestCompactionBasics(unittest.TestCase):
    def setUp(self):
        self.ctx = FakeContext()

    def test_noop_when_too_few_messages(self):
        msgs = [_msg('system', 'sys'), _msg('user', 'hi')]
        result = m.compact_messages(msgs, self.ctx, keep_recent=6)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['role'], 'system')

    def test_noop_when_under_budget(self):
        msgs = [_msg('system', 'sys')] + [_msg('user', f'm{i}') for i in range(10)]
        result = m.compact_messages(msgs, self.ctx, target_tokens=100000, keep_recent=2)
        self.assertEqual(result, msgs)

    def test_basic_structure_system_summary_recent(self):
        msgs = [_msg('system', 'sys')]
        msgs += [_msg('user', f'question {i}') for i in range(5)]
        msgs += [_msg('assistant', f'answer {i}') for i in range(5)]
        result = m.compact_messages(msgs, self.ctx, target_tokens=10, keep_recent=2)
        self.assertEqual(result[0]['role'], 'system')
        self.assertEqual(result[1]['role'], 'user')
        self.assertIn('[CONVERSATION HISTORY COMPACTED]', result[1]['content'])
        self.assertEqual(result[-2:], msgs[-2:])


class TestQ6StampSummaryTokens(unittest.TestCase):
    def test_compacted_summary_has_tokens_stamped(self):
        ctx = FakeContext()
        msgs = [_msg('system', 'sys')]
        msgs += [_msg('user', f'question {i}') for i in range(8)]
        msgs += [_msg('assistant', 'final answer')]
        result = m.compact_messages(msgs, ctx, target_tokens=10, keep_recent=1)
        summary = result[1]
        self.assertIn('_tokens', summary)
        self.assertGreaterEqual(summary['_tokens'], 0)


class TestQ4ExactTokensInBudget(unittest.TestCase):
    def _messages(self, with_tokens):
        system = {'role': 'system', 'content': 'S' * 400}
        recent = {'role': 'assistant', 'content': 'R' * 400}
        if with_tokens:
            system['_tokens'] = 100
            recent['_tokens'] = 100
        middle = [
            {'role': 'user', 'content': 'u' * 600},
            {'role': 'assistant', 'content': 'a' * 200},
            {'role': 'tool', 'name': 'read_file', 'content': 'x' * 50},
        ]
        return [system] + middle + [recent]

    def test_exact_tokens_prevent_over_truncation(self):
        ctx = FakeContext()
        with_tokens = m.compact_messages(self._messages(True), ctx, target_tokens=1000, keep_recent=1)
        without_tokens = m.compact_messages(self._messages(False), ctx, target_tokens=1000, keep_recent=1)

        # With cached exact counts the summary fits the budget and is NOT truncated.
        self.assertNotIn('[... older history truncated ...]', with_tokens[1]['content'])
        # Without exact counts, the inflated estimate shrinks `available` and truncates.
        self.assertIn('[... older history truncated ...]', without_tokens[1]['content'])


class TestQ7RoleAlternation(unittest.TestCase):
    def _msgs_ending_with(self, last_role):
        msgs = [_msg('system', 'sys')]
        msgs += [_msg('user', f'q{i}') for i in range(6)]
        msgs += [_msg('assistant', 'answer')]
        msgs.append(_msg(last_role, 'last message'))
        return msgs

    def test_user_tail_gets_assistant_placeholder(self):
        ctx = FakeContext()
        result = m.compact_messages(self._msgs_ending_with('user'), ctx, target_tokens=10, keep_recent=2)
        self.assertEqual(result[1]['role'], 'user')          # summary
        self.assertEqual(result[2]['role'], 'assistant')     # placeholder
        self.assertIn('[Context continued...]', result[2]['content'])
        self.assertEqual(result[-1]['role'], 'user')         # original tail preserved

    def test_assistant_tail_no_placeholder(self):
        ctx = FakeContext()
        result = m.compact_messages(self._msgs_ending_with('assistant'), ctx, target_tokens=10, keep_recent=2)
        self.assertEqual(result[1]['role'], 'user')          # summary
        self.assertEqual(result[2]['role'], 'assistant')     # the real recent assistant, no placeholder
        self.assertNotIn('[Context continued...]', result[2]['content'])

    def test_no_consecutive_user_roles(self):
        ctx = FakeContext()
        result = m.compact_messages(self._msgs_ending_with('user'), ctx, target_tokens=10, keep_recent=2)
        roles = [msg.get('role') for msg in result]
        for i in range(1, len(roles)):
            if roles[i] == 'user':
                self.assertNotEqual(roles[i], roles[i - 1],
                                    f"consecutive user roles at {i-1},{i}: {roles}")


def _tool_session():
    """system, some users, an assistant tool_call + tool result, then a tail."""
    return [
        {'role': 'system', 'content': 'sys'},
        {'role': 'user', 'content': 'q0'},
        {'role': 'user', 'content': 'q1'},
        {'role': 'assistant', 'content': 'call',
         'tool_calls': [{'id': 'tc', 'function': {'name': 'read_file', 'arguments': '{}'}}]},
        {'role': 'tool', 'content': 'result text', 'tool_call_id': 'tc'},
        {'role': 'assistant', 'content': 'answer'},
        {'role': 'user', 'content': 'next'},
    ]


def _assert_no_orphaned_tools(testcase, msgs):
    """Every tool message must be preceded by an assistant with tool_calls."""
    last_non_tool_ok = False
    for i, msg in enumerate(msgs):
        if msg.get('role') == 'tool':
            testcase.assertTrue(
                last_non_tool_ok,
                f"orphaned tool message at index {i} (prev role: "
                f"{msgs[i - 1].get('role') if i else 'none'})")
        else:
            last_non_tool_ok = (msg.get('role') == 'assistant'
                                and bool(msg.get('tool_calls')))


class TestToolPairingBoundary(unittest.TestCase):
    """Compaction must not split an assistant tool_call from its tool result."""

    def test_boundary_never_starts_on_tool(self):
        ctx = FakeContext()
        for keep in (1, 2, 3, 4, 5, 6):
            result = m.compact_messages(_tool_session(), ctx,
                                        target_tokens=1, keep_recent=keep)
            _assert_no_orphaned_tools(self, result)

    def test_boundary_split_keeps_tool_call_pair(self):
        # keep_recent=3 splits right before the tool result; the fix must pull
        # the preceding assistant tool_call into the kept window.
        ctx = FakeContext()
        result = m.compact_messages(_tool_session(), ctx,
                                    target_tokens=1, keep_recent=3)
        roles = [msg.get('role') for msg in result]
        # summary + placeholder + the preserved tool_call/result pair + tail
        self.assertEqual(roles, ['system', 'user', 'assistant',
                                 'assistant', 'tool', 'assistant', 'user'])
        _assert_no_orphaned_tools(self, result)


class TestSanitizeToolPairing(unittest.TestCase):
    def test_orphaned_tool_demoted_to_user(self):
        msgs = [
            {'role': 'system', 'content': 'sys'},
            {'role': 'user', 'content': 'hi'},
            {'role': 'tool', 'content': 'orphan result', 'tool_call_id': 'x',
             'name': 'read_file'},
        ]
        m.sanitize_tool_pairing(msgs)
        self.assertEqual(msgs[2]['role'], 'user')
        self.assertIn('Tool result (read_file):', msgs[2]['content'])
        self.assertIn('orphan result', msgs[2]['content'])
        self.assertNotIn('tool_call_id', msgs[2])

    def test_valid_pair_untouched(self):
        msgs = [
            {'role': 'system', 'content': 'sys'},
            {'role': 'user', 'content': 'hi'},
            {'role': 'assistant', 'content': 'call',
             'tool_calls': [{'id': 't1', 'function': {'name': 'x', 'arguments': '{}'}}]},
            {'role': 'tool', 'content': 'r1', 'tool_call_id': 't1'},
            {'role': 'assistant', 'content': 'done'},
        ]
        before = [dict(x) for x in msgs]
        m.sanitize_tool_pairing(msgs)
        self.assertEqual(msgs, before)

    def test_parallel_tool_results_kept(self):
        msgs = [
            {'role': 'assistant', 'content': 'call',
             'tool_calls': [{'id': 't1', 'function': {'name': 'x', 'arguments': '{}'}}]},
            {'role': 'tool', 'content': 'r1', 'tool_call_id': 't1'},
            {'role': 'tool', 'content': 'r2', 'tool_call_id': 't1'},
        ]
        m.sanitize_tool_pairing(msgs)
        roles = [x['role'] for x in msgs]
        self.assertEqual(roles, ['assistant', 'tool', 'tool'])

    def test_tool_after_plain_assistant_demoted(self):
        msgs = [
            {'role': 'assistant', 'content': 'plain answer'},
            {'role': 'tool', 'content': 'stray result'},
        ]
        m.sanitize_tool_pairing(msgs)
        self.assertEqual(msgs[1]['role'], 'user')


if __name__ == '__main__':
    unittest.main(verbosity=2)
