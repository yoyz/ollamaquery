# Agentic Testing Procedure for qwen3:8b

## Overview

This procedure tests agentic mode with `qwen3:8b` across different inference parameter configurations. The goal is to find optimal parameters for reliable tool-calling behavior.

---

## 1. Prerequisites

- Ollama server running with `qwen3:8b` pulled
- `OLLAMA_HOST` environment variable set (e.g., `http://192.168.1.20:11434`)
- Working directory: `/home/ollama/build/ollamaquery/ollamaquery2`

## 2. Parameter Registry

Current qwen3:8b parameters (defined in `ollamaquery2.py:1161`):

| Parameter         | Current Value | Description                                      |
|-------------------|---------------|--------------------------------------------------|
| `temperature`     | 0.5           | Randomness (0.0=deterministic, 2.0=chaotic)      |
| `top_p`           | 0.9           | Nucleus sampling (0.0-1.0, lower = more focused) |
| `top_k`           | 20            | Top-K tokens to consider (0=disabled)            |
| `min_p`           | 0.0           | Minimum probability threshold                    |
| `presence_penalty`| 0.0           | Penalize repeated tokens (0.0=disabled)          |
| `repeat_penalty`  | 1.0           | Repetition penalty (1.0=disabled, >1.0=penalize) |

## 3. Test Suite

### 3.1 Unit Tests (no backend needed)
```bash
python3 -m unittest tests.test_agentic.TestParseToolCall \
  tests.test_agentic.TestToolRegistry \
  tests.test_agentic.TestExecutor \
  tests.test_agentic.TestReActLoopUnit \
  tests.test_agentic.TestStuckDetection \
  tests.test_agentic.TestNormalizeToolJson \
  tests.test_agentic.TestParseToolCalls \
  tests.test_agentic.TestArgumentAliases \
  tests.test_agentic.TestLazyToolMode \
  tests.test_agentic.TestComposableSystemPrompt -v
```

### 3.2 E2E Agentic Tests (need live backend)
```bash
# All E2E tests
TEST_MODEL=qwen3:8b OLLAMA_HOST=http://192.168.1.20:11434 \
  python3 -m unittest tests.test_agentic.TestAgenticReActEndToEnd -v

# Individual tests
TEST_MODEL=qwen3:8b OLLAMA_HOST=http://192.168.1.20:11434 \
  python3 -m unittest \
    tests.test_agentic.TestAgenticReActEndToEnd.test_direct_answer_no_tool \
    tests.test_agentic.TestAgenticReActEndToEnd.test_fetch_url_tool \
    tests.test_agentic.TestAgenticReActEndToEnd.test_multi_step_write_file \
    tests.test_agentic.TestAgenticReActEndToEnd.test_dns_resolver_write_compile_run \
    -v
```

### 3.3 Override Parameters via the Registry

To test different parameter sets, edit `ollamaquery2.py` lines 1161-1172:

```python
# Default qwen3 params
"qwen3": {
    "temperature": 0.5,
    "top_p": 0.9,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repeat_penalty": 1.0,
},
```

Or create a test script that monkey-patches the registry before running.

## 4. Parameter Experimentation Matrix

### Test variation A: Temperature sweep

| Config | temperature | top_p | top_k | Expected Effect          |
|--------|-------------|-------|-------|--------------------------|
| cold   | 0.1         | 0.9   | 20    | Deterministic, repetitive |
| medium | 0.5         | 0.9   | 20    | Balanced (current)        |
| warm   | 0.7         | 0.9   | 20    | Creative, but wanders     |
| hot    | 1.0         | 0.9   | 20    | Chaotic, unreliable       |

### Test variation B: Top-P sweep

| Config | temperature | top_p | top_k | Expected Effect              |
|--------|-------------|-------|-------|------------------------------|
| narrow | 0.5         | 0.5   | 20    | Very focused, may be rigid   |
| medium | 0.5         | 0.9   | 20    | Balanced (current)           |
| broad  | 0.5         | 1.0   | 20    | More diverse token selection |

### Test variation C: Repeat/Presence penalty sweep

| Config | temperature | presence_penalty | repeat_penalty | Expected Effect               |
|--------|-------------|------------------|----------------|-------------------------------|
| off    | 0.5         | 0.0              | 1.0            | Current (no penalty)          |
| light  | 0.5         | 0.1              | 1.1            | Slightly less repetition      |
| medium | 0.5         | 0.3              | 1.2            | Noticeably different output   |

## 5. Running Custom Parameter Configurations

### Method A: Monkey-patch via environment
Create a wrapper test script (e.g., `run_test_with_params.py`):

```python
#!/usr/bin/env python3
"""Run agentic tests with custom inference parameters."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ollamaquery2 as q

# Override qwen3 params
q.MODEL_INFERENCE_PARAMS_REGISTRY["qwen3"] = {
    "temperature": float(os.environ.get("TEST_TEMP", "0.5")),
    "top_p": float(os.environ.get("TEST_TOP_P", "0.9")),
    "top_k": int(os.environ.get("TEST_TOP_K", "20")),
    "min_p": float(os.environ.get("TEST_MIN_P", "0.0")),
    "presence_penalty": float(os.environ.get("TEST_PRESENCE_PENALTY", "0.0")),
    "repeat_penalty": float(os.environ.get("TEST_REPEAT_PENALTY", "1.0")),
}

# Now import and run the tests
import unittest
loader = unittest.TestLoader()
suite = unittest.TestSuite()

# Add specific tests
from tests.test_agentic import TestAgenticReActEndToEnd
suite.addTest(TestAgenticReActEndToEnd('test_direct_answer_no_tool'))
suite.addTest(TestAgenticReActEndToEnd('test_fetch_url_tool'))
suite.addTest(TestAgenticReActEndToEnd('test_multi_step_write_file'))

runner = unittest.TextTestRunner(verbosity=2)
result = runner.run(suite)

# Print config for logging
print(f"\n--- CONFIG ---")
print(f"model=qwen3:8b temp={os.environ.get('TEST_TEMP','0.5')} top_p={os.environ.get('TEST_TOP_P','0.9')} top_k={os.environ.get('TEST_TOP_K','20')} presence={os.environ.get('TEST_PRESENCE_PENALTY','0.0')} repeat={os.environ.get('TEST_REPEAT_PENALTY','1.0')}")

sys.exit(0 if result.wasSuccessful() else 1)
```

Usage:
```bash
TEST_TEMP=0.3 TEST_TOP_P=0.8 python3 run_test_with_params.py
```

### Method B: Edit the registry directly
Edit `ollamaquery2.py` lines 1161-1172, then run tests normally.

## 6. Interactive Multi-Turn Agentic Session

To play with qwen3:8b interactively across multiple agentic turns:

```bash
cd /home/ollama/build/ollamaquery/ollamaquery2
OLLAMA_HOST=http://192.168.1.20:11434 python3 ollamaquery2.py -c -P coder
```

Then in the chat:
```
/agentic on
/listtool
```

Now issue sequential queries to test multi-turn:
```
Write a Python script that calculates fibonacci numbers.
Read the file back and tell me what it does.
Modify the script to add error handling.
Compile and test it.
```

## 7. Metrics to Track

When testing parameter variations, record:

| Metric | How to Measure |
|--------|---------------|
| First tool call accuracy | Did the model pick the right tool on first try? |
| JSON validity | Did `parse_tool_call` succeed? |
| Tool success rate | Did the tool execute without errors? |
| Stuck detection triggered | Was `_is_stuck` true? |
| Same-tool loop triggered | Did the model repeat the same call? |
| Tokens per second | Reported in stats line |
| Total query time | Reported in stats line |
| Number of ReAct steps | Count of iterations needed |

## 8. Quick Commands Reference

```bash
# Run ALL agentic unit tests
python3 -m unittest tests.test_agentic -v

# Run only E2E tests (need backend)
TEST_MODEL=qwen3:8b OLLAMA_HOST=http://192.168.1.20:11434 \
  python3 -m unittest tests.test_agentic.TestAgenticReActEndToEnd -v

# Run a single E2E test
TEST_MODEL=qwen3:8b OLLAMA_HOST=http://192.168.1.20:11434 \
  python3 -m unittest \
    tests.test_agentic.TestAgenticReActEndToEnd.test_direct_answer_no_tool -v

# Interactive agentic mode
OLLAMA_HOST=http://192.168.1.20:11434 python3 ollamaquery2.py -c

# Syntax check
python3 -m py_compile ollamaquery2.py
```

## 9. Known Results (May 2026)

### 6 E2E Tests Added

| Test | Description | Assertion |
|------|-------------|-----------|
| test_port_scanner_write_compile_run | TCP connect scanner in C | binary `portscanner` exists |
| test_web_server_write_compile_run | HTTP server in C, tested with Python | binary `webserver` exists |
| test_directory_lister_write_compile_run | `ls`-like program in C | binary `dirlister` exists |

### Parameter Comparison Table

| Config | temp | top_p | top_k | repeat_pen | Direct Answer | Fetch URL | Write File | Port Scanner | Web Server | Dir Lister | Total Time |
|--------|------|-------|-------|------------|---------------|-----------|------------|--------------|------------|------------|------------|
| **Cold** | 0.1 | 0.9 | 20 | 1.0 | **15.8s** ✅ | — | — | — | — | — | — |
| **Default** | 0.5 | 0.9 | 20 | 1.0 | 111.9s ✅ | 9.9s ✅ | 19.7s ✅ | 192.2s ✅¹ | 119.4s ✅² | 116.4s ✅ | 476s (6 tests) |

¹ First attempt passed; second attempt failed (model output code as text instead of tools)
² First attempt passed; second attempt at temp=0.1 failed (model overthought tool access); third attempt at temp=0.5 passed with correct tool usage

### Temperature Sweep (test_direct_answer_no_tool)

| temp | Time | Speed vs Default | Observation |
|------|------|------------------|-------------|
| **0.1** | **15.8s** | **7x faster** | Minimal thinking, direct answer |
| 0.5 (default) | 111.9s | baseline | Verbose thinking about formatting rules |
| 0.7 | 89.9s | 1.2x faster | Less deliberation, faster answer |

### Penalty Sweep (test_direct_answer_no_tool)

| repeat_penalty | presence_penalty | Time | Observation |
|----------------|------------------|------|-------------|
| 1.0 (default) | 0.0 | 111.9s | Baseline |
| 1.2 | 0.0 | 188.4s | **Penalty backfires** — more overthinking |
| 1.0 | 0.3 | >200s timeout | **Severe overthinking** — never resolved |

### Key Findings

1. **temp=0.1 is optimal for speed** — 7x faster for simple queries, but can get stuck in meta-reasoning loops for complex multi-step tasks (web server). Best for simple tool calls.

2. **temp=0.5 is optimal for reliability** — Default for qwen3. Good balance for complex multi-step tasks involving networking code generation and compilation.

3. **Penalties backfire on qwen3:8b** — Increasing repeat_penalty or presence_penalty causes the model to overthink more, not less. Leave at default (repeat=1.0, presence=0.0).

4. **Task complexity affects tool usage** — Simple tasks (write file, dir list, fetch URL) always use tools. Complex tasks (port scanner, web server) sometimes cause the model to output code as text instead of using `write_file` tool. Success rate is ~50-80% per attempt.

5. **Model uses `file_path` instead of `file`** — Handled by arg aliases in `TOOL_ARG_ALIASES`.

6. **Thinking blocks are verbose** — qwen3:8b outputs detailed `<think>` blocks before every response. This adds significant latency (30-120s per step at temp=0.5).

7. **step_timeout=120s is critical** — Complex thinking can exceed this; the `_call_with_timeout` thread kills slow steps.

8. **DNS resolver test times out** — Complex DNS packet construction requires multiple write-compile-fix iterations exceeding the timeout.

### Multi-Turn Agentic Observations

- qwen3:8b maintains tool-calling JSON format across turns
- Conversation history is preserved in `self.messages` across agentic turns
- The model adapts tool usage based on previous outcomes (e.g., after finding port 11434 CLOSED on 127.0.0.1, it used `ss` to investigate and re-tested on the correct IP)
- Lazy tool mode (`lazy_tool=True`) is needed — the model sometimes embeds tool calls mid-text rather than at the start
