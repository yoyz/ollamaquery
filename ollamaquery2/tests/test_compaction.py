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
        self.backend = 'llamacpp'
        self.model = 'test-model'

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


class TestSplitCompactionWindow(unittest.TestCase):
    """The keep-recent window must never start on an orphaned tool message."""

    def test_window_shifts_left_on_tool(self):
        msgs = [_msg('system', 'sys'), _msg('user', 'q')]
        msgs += [{'role': 'assistant', 'content': 'call',
                  'tool_calls': [{'id': 'tc', 'function': {'name': 'x', 'arguments': '{}'}}]},
                 {'role': 'tool', 'content': 'res', 'tool_call_id': 'tc'},
                 {'role': 'user', 'content': 'tail'}]
        system, middle, recent = m._split_compaction_window(msgs, keep_recent=2)
        self.assertEqual(system['role'], 'system')
        self.assertEqual(middle, msgs[1:2])
        # Window starts on the assistant tool_call, not the tool result.
        self.assertEqual(recent[0]['role'], 'assistant')
        self.assertEqual(recent[-1], msgs[-1])

    def test_window_keeps_recent_tail(self):
        msgs = [_msg('system', 'sys')] + [_msg('user', f'q{i}') for i in range(6)]
        system, middle, recent = m._split_compaction_window(msgs, keep_recent=4)
        self.assertEqual(len(recent), 4)
        self.assertEqual(recent, msgs[-4:])
        self.assertEqual(middle, msgs[1:-4])


class TestBuildCompactionSummary(unittest.TestCase):
    def test_tool_and_user_and_assistant(self):
        middle = [
            {'role': 'user', 'content': 'hello world'},
            {'role': 'assistant', 'content': 'some answer'},
            {'role': 'tool', 'name': 'read_file', 'content': 'data'},
            {'role': 'tool', 'name': 'run', 'content': 'ERROR: boom'},
        ]
        text = m._build_compaction_summary(middle)
        self.assertIn('[CONVERSATION HISTORY COMPACTED]', text)
        self.assertIn('User: hello world', text)
        self.assertIn('Assistant: some answer', text)
        self.assertIn('[Tool read_file: OK]', text)
        self.assertIn('[Tool run: FAILED]', text)

    def test_tool_observation_user(self):
        middle = [{'role': 'user', 'content': "Tool result:\nfoo\nbar"}]
        text = m._build_compaction_summary(middle)
        self.assertIn('[Tool observation:', text)


class TestRenderConversationTranscript(unittest.TestCase):
    def test_tool_reduced_to_status_line(self):
        middle = [
            {'role': 'tool', 'name': 'read_file', 'content': 'x' * 5000},
            {'role': 'user', 'content': 'keep this question'},
            {'role': 'assistant', 'content': 'short reply'},
        ]
        text = m._render_conversation_transcript(middle)
        self.assertIn('[tool read_file: OK]', text)
        self.assertIn('user: keep this question', text)
        self.assertNotIn('x' * 200, text, 'huge tool content must be truncated')

    def test_tool_transcript_includes_outcome_excerpt(self):
        middle = [
            {'role': 'tool', 'name': 'cat', 'content': 'cat myfile\nmyfile: No such file or directory'},
            {'role': 'tool', 'name': 'run', 'content': 'ERROR: boom\nstack'},
        ]
        text = m._render_conversation_transcript(middle)
        self.assertIn('[tool cat: OK] cat myfile', text)
        self.assertIn('[tool run: FAILED] ERROR: boom', text)

    def test_max_chars_truncation(self):
        middle = [{'role': 'user', 'content': 'a' * 1000}]
        text = m._render_conversation_transcript(middle, max_chars=200)
        self.assertLessEqual(len(text), 200 + len("[...]") + 1)
        self.assertIn('[...]', text)

    def test_tool_observation_collapsed(self):
        middle = [{'role': 'user', 'content': "Tool result:\nbig output here"}]
        text = m._render_conversation_transcript(middle)
        self.assertIn('[tool observation: Tool result:]', text)
        self.assertNotIn('big output', text)


class TestRenderTokenBreakdown(unittest.TestCase):
    def setUp(self):
        self.ctx = FakeContext()

    def test_roles_tokens_and_tool_ok_ko(self):
        middle = [
            {'role': 'user', 'content': 'I want a simple netcat in go', '_tokens': 9},
            {'role': 'assistant', 'content': 'Let me look at the files.', '_tokens': 7},
            {'role': 'tool', 'name': 'cat', 'content': 'cat myfile\nNo such file or directory', '_tokens': 5},
            {'role': 'tool', 'name': 'glob', 'content': 'ERROR: no matches', '_tokens': 3},
        ]
        text = m._render_token_breakdown(middle, self.ctx)
        self.assertIn('[  0] user', text)
        self.assertIn('[  1] assistant', text)
        self.assertIn('[OK] cat myfile', text)
        self.assertIn('[KO] ERROR: no matches', text)
        self.assertIn('·', text)

    def test_breakdown_truncates_total(self):
        middle = [{'role': 'user', 'content': 'z' * 500}] * 20
        text = m._render_token_breakdown(middle, self.ctx, max_chars=300)
        self.assertLessEqual(len(text), 300 + len("[...]") + 1)
        self.assertIn('[...]', text)

    def test_breakdown_caps_text_at_80_chars(self):
        middle = [{'role': 'user', 'content': 'y' * 1000}]
        text = m._render_token_breakdown(middle, self.ctx)
        self.assertNotIn('y' * 200, text)
        self.assertIn('...', text)

    def test_breakdown_exposes_tool_call_command(self):
        """Tool-call JSON lines keep enough text for the LLM to see the command."""
        middle = [{'role': 'assistant', '_tokens': 146,
                   'content': '{"tool": "run_command", "arguments": {"command": "go build -o nc nc.go && go vet nc.go"}}'}]
        text = m._render_token_breakdown(middle, self.ctx)
        self.assertIn('run_command', text)
        self.assertIn('go build -o nc', text)


class TestExtractGenTokens(unittest.TestCase):
    def setUp(self):
        self.ctx = FakeContext()
        self.ctx.backend = 'llamacpp'

    def test_llamacpp_usage(self):
        resp = {'usage': {'completion_tokens': 42}}
        self.assertEqual(m._extract_gen_tokens(self.ctx, resp), 42)

    def test_ollama_eval_count(self):
        self.ctx.backend = 'ollama'
        self.assertEqual(m._extract_gen_tokens(self.ctx, {'eval_count': 7}), 7)

    def test_missing_usage(self):
        self.assertEqual(m._extract_gen_tokens(self.ctx, {'choices': []}), 0)
        self.assertEqual(m._extract_gen_tokens(self.ctx, None), 0)


class FakeQueryHandler:
    """Deterministic stand-in for ModelQuery used by the LLM summarizers."""

    def __init__(self, content='NARRATIVE SUMMARY'):
        self.content = content
        self.calls = []

    def query_sync(self, messages, model, **kwargs):
        self.calls.append((messages, model, kwargs))
        return {'choices': [{'message': {'content': self.content}}]}


class TestExtractSyncContent(unittest.TestCase):
    def setUp(self):
        self.ctx = FakeContext()
        self.ctx.backend = 'llamacpp'

    def test_llamacpp_shape(self):
        self.assertEqual(
            m._extract_sync_content(self.ctx, {'choices': [{'message': {'content': 'hi'}}]}),
            'hi')

    def test_ollama_shape(self):
        self.ctx.backend = 'ollama'
        self.assertEqual(
            m._extract_sync_content(self.ctx, {'message': {'content': 'hi'}}), 'hi')

    def test_string_response(self):
        self.assertEqual(m._extract_sync_content(self.ctx, 'raw'), 'raw')

    def test_empty_response(self):
        self.assertEqual(m._extract_sync_content(self.ctx, {}), '')
        self.assertEqual(m._extract_sync_content(self.ctx, None), '')


class TestLlmCompactMessages(unittest.TestCase):
    def _session(self):
        msgs = [_msg('system', 'sys')]
        msgs += [_msg('user', f'q{i}') for i in range(8)]
        msgs += [_msg('assistant', 'final answer')]
        return msgs

    def test_llm_summary_structure(self):
        ctx = FakeContext()
        handler = FakeQueryHandler(content='user wanted a summary of X.')
        result = m.llm_compact_messages(self._session(), ctx, handler, keep_recent=2)
        self.assertEqual(result[0]['role'], 'system')
        self.assertEqual(result[1]['role'], 'user')
        content = result[1]['content']
        self.assertIn(m.LLM_COMPACT_SUMMARY_MARKER, content)
        self.assertIn('user wanted a summary of X.', content)
        # The compacted message embeds the token breakdown of the compacted turns
        # plus the before/after context size and the summarization token rate.
        self.assertIn('Token breakdown of the compacted turns:', content)
        self.assertIn('[  0] user', content)
        self.assertIn('Context:', content)
        self.assertIn('tok/s', content)
        self.assertIn('_tokens', result[1])
        self.assertEqual(result[-2:], self._session()[-2:])
        # The summarizer actually got the older turns.
        self.assertTrue(handler.calls)
        sent_user = handler.calls[0][0][1]['content']
        self.assertIn('q0', sent_user)

    def test_llm_summary_breakdown_marks_tool_results(self):
        ctx = FakeContext()
        handler = FakeQueryHandler(content='ok')
        session = [_msg('system', 'sys'),
                   {'role': 'user', 'content': 'find the file'},
                   {'role': 'assistant', 'content': 'calling cat'},
                   {'role': 'tool', 'name': 'cat', 'content': 'cat myfile\nNo such file', 'tool_call_id': 't1'},
                   {'role': 'user', 'content': 'tail'}]
        result = m.llm_compact_messages(session, ctx, handler, keep_recent=1)
        content = result[1]['content']
        self.assertIn('[OK] cat myfile', content)
        self.assertIn('→', content)

    def test_fallback_to_mechanical_when_llm_fails(self):
        ctx = FakeContext()
        handler = FakeQueryHandler(content='')
        result = m.llm_compact_messages(self._session(), ctx, handler, keep_recent=2)
        self.assertIn('[CONVERSATION HISTORY COMPACTED]', result[1]['content'])
        self.assertNotIn(m.LLM_COMPACT_SUMMARY_MARKER, result[1]['content'])

    def test_noop_when_too_few_messages(self):
        ctx = FakeContext()
        handler = FakeQueryHandler()
        msgs = [_msg('system', 'sys'), _msg('user', 'hi')]
        result = m.llm_compact_messages(msgs, ctx, handler, keep_recent=6)
        self.assertEqual(result, msgs)
        self.assertEqual(handler.calls, [])

    def test_role_alternation_and_tool_pairing(self):
        ctx = FakeContext()
        handler = FakeQueryHandler()
        session = [_msg('system', 'sys'), _msg('user', 'q0'), _msg('user', 'q1'),
                   {'role': 'assistant', 'content': 'call',
                    'tool_calls': [{'id': 'tc', 'function': {'name': 'x', 'arguments': '{}'}}]},
                   {'role': 'tool', 'content': 'res', 'tool_call_id': 'tc'},
                   {'role': 'user', 'content': 'tail'}]
        result = m.llm_compact_messages(session, ctx, handler, keep_recent=3)
        _assert_no_orphaned_tools(self, result)
        roles = [msg['role'] for msg in result]
        for i in range(1, len(roles)):
            if roles[i] == 'user':
                self.assertNotEqual(roles[i], roles[i - 1])

    def test_llm_compact_uses_transcript_excerpt(self):
        ctx = FakeContext()
        handler = FakeQueryHandler()
        big = _msg('user', 'z' * 3000)
        session = [_msg('system', 'sys')] + [big] + [_msg('user', f'q{i}') for i in range(6)]
        m.llm_compact_messages(session, ctx, handler, keep_recent=1)
        sent = handler.calls[0][0][1]['content']
        self.assertLessEqual(len(sent), m.LLM_COMPACT_TRANSCRIPT_CHARS + 64)
        self.assertIn('user: q3', sent)

    def test_llm_summary_appends_to_prior_log(self):
        """A second /compact llm keeps the prior dated entry verbatim and appends
        a new one instead of re-summarizing the old away."""
        ctx = FakeContext()
        first = m.llm_compact_messages(self._session(), ctx,
                                       FakeQueryHandler(content='SUMMARY ONE.'), keep_recent=2)
        first_content = first[1]['content']
        self.assertIn('Compaction log', first_content)
        self.assertIn('SUMMARY ONE.', first_content)

        # Continue the session: prior summary stays at index 1, then new turns.
        continued = first[:2] + [_msg('user', 'new question'),
                                 _msg('assistant', 'new answer')]
        second = m.llm_compact_messages(continued, ctx,
                                        FakeQueryHandler(content='SUMMARY TWO.'), keep_recent=1)
        content = second[1]['content']
        self.assertIn('SUMMARY ONE.', content, 'prior entry must be preserved verbatim')
        self.assertIn('SUMMARY TWO.', content)
        self.assertIn(first_content, content, 'prior entry must survive untruncated')
        self.assertEqual(content.count('Token breakdown of the compacted turns:'), 2)
        # Each dated entry carries a [YYYY-MM-DD HH:MM] timestamp.
        self.assertGreaterEqual(content.count('['), 4)

    def test_extract_compaction_log_strips_header(self):
        body = m._extract_compaction_log('x')
        self.assertEqual(body, '')
        prev = m.llm_compact_messages(self._session(), FakeContext(),
                                      FakeQueryHandler(content='S.'), keep_recent=2)
        prev_content = prev[1]['content']
        extracted = m._extract_compaction_log(prev_content)
        self.assertNotIn(m.LLM_COMPACT_SUMMARY_MARKER, extracted)
        self.assertIn('S.', extracted)


if __name__ == '__main__':
    unittest.main(verbosity=2)
