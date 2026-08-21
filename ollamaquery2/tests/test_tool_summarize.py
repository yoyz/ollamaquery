#!/usr/bin/env python3
"""Unit tests for LLM-based tool-result summarization (qwen36 improvement A).

Drives `summarize_tool_results` against a fake query handler and context, so no
backend is required. Verifies eligibility filtering (content size, excluded
tools, oldest-half selection) and the replacement/stamping behavior.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ollamaquery2 as m


class FakeCtx:
    backend = 'llamacpp'
    model = 'test-model'

    def stamp_tokens(self, msg):
        msg['_tokens'] = (len(msg.get('content', '')) // 4) + 2
        return msg

    def estimate_tokens(self, text):
        return max(1, len(text))


class FakeQueryHandler:
    def __init__(self, result='SUMMARY TEXT'):
        self.result = result
        self.calls = []

    def query_sync(self, messages, model, **kwargs):
        self.calls.append((messages, model, kwargs))
        if self.result is None:
            raise RuntimeError('backend down')
        return {'choices': [{'message': {'content': self.result}}]}


def _tool(name, content):
    return {'role': 'tool', 'name': name, 'content': content}


BIG = 'x' * 1500
SMALL = 'short output'


class TestSummarizeToolResults(unittest.TestCase):
    def test_no_tool_messages_returns_unchanged(self):
        ctx = FakeCtx()
        qh = FakeQueryHandler()
        msgs = [{'role': 'user', 'content': 'hello'}]
        result = m.summarize_tool_results(msgs, ctx, qh)
        self.assertEqual(result, msgs)
        self.assertEqual(qh.calls, [])

    def test_small_content_not_summarized(self):
        ctx = FakeCtx()
        qh = FakeQueryHandler()
        msgs = [_tool('read_file', SMALL), _tool('read_file', SMALL),
                {'role': 'user', 'content': 'q'}]
        m.summarize_tool_results(msgs, ctx, qh)
        self.assertEqual(qh.calls, [])

    def test_excluded_tools_not_summarized(self):
        ctx = FakeCtx()
        qh = FakeQueryHandler()
        msgs = [_tool('list_directory', BIG), _tool('diff', BIG),
                _tool('patch', BIG), _tool('edit_file', BIG),
                _tool('apply_patch', BIG), _tool('read_file', BIG),
                {'role': 'user', 'content': 'q'}]
        # read_file is index 5; with 6 tool results, oldest half = first 3,
        # all of which are excluded -> nothing summarized.
        m.summarize_tool_results(msgs, ctx, qh)
        self.assertEqual(qh.calls, [])

    def test_oldest_half_summarized_newest_kept_raw(self):
        ctx = FakeCtx()
        qh = FakeQueryHandler()
        msgs = [
            _tool('read_file', BIG),   # idx 0 -> oldest half -> summarized
            _tool('run_command', BIG), # idx 1 -> oldest half -> summarized
            _tool('glob', BIG),        # idx 2 -> newest half -> kept raw
            _tool('read_file', BIG),   # idx 3 -> newest half -> kept raw
            {'role': 'user', 'content': 'q'},
        ]
        m.summarize_tool_results(msgs, ctx, qh)
        self.assertEqual(len(qh.calls), 2)
        self.assertTrue(msgs[0]['content'].startswith('[Summarized tool result (read_file)]'))
        self.assertTrue(msgs[1]['content'].startswith('[Summarized tool result (run_command)]'))
        self.assertEqual(msgs[2]['content'], BIG)
        self.assertEqual(msgs[3]['content'], BIG)

    def test_summarized_message_restamped(self):
        ctx = FakeCtx()
        qh = FakeQueryHandler()
        msgs = [_tool('read_file', BIG), _tool('read_file', BIG),
                {'role': 'user', 'content': 'q'}]
        m.summarize_tool_results(msgs, ctx, qh)
        self.assertIn('_tokens', msgs[0])

    def test_cap_limits_summarized_count(self):
        ctx = FakeCtx()
        qh = FakeQueryHandler()
        msgs = [_tool('read_file', BIG) for _ in range(8)]
        msgs.append({'role': 'user', 'content': 'q'})
        m.summarize_tool_results(msgs, ctx, qh)
        # 8 tool results -> oldest half = 4, capped at SUMMARIZE_TOOL_MAX (3).
        self.assertEqual(len(qh.calls), m.SUMMARIZE_TOOL_MAX)

    def test_backend_failure_leaves_content_unchanged(self):
        ctx = FakeCtx()
        qh = FakeQueryHandler(result=None)
        msgs = [_tool('read_file', BIG), _tool('read_file', BIG),
                {'role': 'user', 'content': 'q'}]
        m.summarize_tool_results(msgs, ctx, qh)
        self.assertEqual(msgs[0]['content'], BIG)

    def test_ollama_response_extracted(self):
        ctx = FakeCtx()
        ctx.backend = 'ollama'
        qh = FakeQueryHandler(result='OLLAMA SUMMARY')

        def ollama_query_sync(messages, model, **kwargs):
            qh.calls.append((messages, model, kwargs))
            return {'message': {'content': 'OLLAMA SUMMARY'}}

        qh.query_sync = ollama_query_sync
        msgs = [_tool('read_file', BIG), _tool('read_file', BIG),
                {'role': 'user', 'content': 'q'}]
        m.summarize_tool_results(msgs, ctx, qh)
        self.assertIn('OLLAMA SUMMARY', msgs[0]['content'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
