# Model-Specific Parameters

This document consolidates all model-specific tuning parameters used across ollamaquery. Parameters are applied during agentic (ReAct) mode tool-calling and regular chat inference.

## Table of Contents

- [Parameter Registries](#parameter-registries)
  - [Inference Parameters](#inference-parameters)
  - [Prompt Style Registry](#prompt-style-registry)
- [Model Reference](#model-reference)
  - [GLM-4.7](#glm-47)
  - [Qwen3.5](#qwen35)
  - [Qwen3 8B](#qwen3-8b)
  - [Nemotron-Cascade](#nemotron-cascade)
  - [Nemotron (generic)](#nemotron-generic)
  - [GPT-OSS](#gpt-oss)
  - [Default (unknown models)](#default-unknown-models)
- [How Parameters Are Applied](#how-parameters-are-applied)
- [Sources](#sources)

---

## Parameter Registries

### Inference Parameters

Defined in `MODEL_INFERENCE_PARAMS_REGISTRY` (`ollamaquery2.py:1088`). Keys are model name substrings matched case-insensitively. The first matching key wins (order matters — more specific keys like `"nemotron-cascade"` should appear before generic ones like `"nemotron"`).

| Key                  | temperature | top_p | top_k | min_p | presence_penalty | repeat_penalty | Use Case                   | Source                        |
|----------------------|-------------|-------|-------|-------|------------------|----------------|----------------------------|-------------------------------|
| `nemotron-cascade`   | 0.6         | 0.95  | 40    | 0.0   | 0.0              | 1.0            | Tool calling / coding       | Nemotron-Cascade-2-30B        |
| `glm-4.7`            | 0.7         | 1.0   | —     | 0.01  | 0.0              | 1.0            | Tool calling                | Medium guide + Gemini 3.5     |
| `qwen3.5`            | 0.5         | 0.9   | 20    | 0.0   | 0.0              | 1.0            | Precise coding / tool call   | Qwen3.5-9B + Gemini 3.5      |
| `qwen3`              | 0.5         | 0.9   | 20    | 0.0   | 0.0              | 1.0            | Agentic tool calling         | Qwen3-8B + Gemini 3.5        |
| `nemotron`           | 0.6         | 0.95  | 40    | 0.0   | 0.0              | 1.0            | Tool calling                | Nemotron-3-Nano               |
| `gpt-oss`            | 0.7         | 1.0   | —     | 0.0   | 0.0              | 1.0            | Agentic / tool calling       | Gemini 3.5 agentic profile    |
| `(default)`          | 0.7         | 0.9   | 40    | 0.0   | 0.0              | 1.0            | Conservative fallback       | `ollamaquery2.py:1131`        |

### Prompt Style Registry

Defined in `AGENTIC_PROMPT_STYLE_REGISTRY` (`ollamaquery2.py:1050`) and `AGENTIC_PROMPT_STYLE_DEFAULT` (`ollamaquery2.py:1054`).

Controls how the agentic system prompt is assembled — specifically whether the model receives strict instructions ("output ONLY JSON") or a softer format (code blocks and preamble allowed). See `get_agentic_prompt()` at `ollamaquery2.py:1066`.

| Key                  | Style    | Behavior                                                       | Source                        |
|----------------------|----------|----------------------------------------------------------------|-------------------------------|
| `nemotron-cascade`   | `soft`   | Allows code blocks, multi-tool-per-response, preamble OK       | Nemotron-Cascade-2-30B        |
| `(default)`          | `strict` | Bare JSON only, no surrounding text, no markdown fences         | Default for all other models  |

---

## Model Reference

### GLM-4.7

**Source:** [zai-org/GLM-4.7](https://huggingface.co/unsloth/GLM-4.7) / [arXiv 2508.06471](https://arxiv.org/abs/2508.06471)

**Status:** Registered in `MODEL_INFERENCE_PARAMS_REGISTRY`.

**Inference Parameters:**

| Task                              | temperature | top_p | top_k | max_new_tokens | Source                                        |
|-----------------------------------|-------------|-------|-------|----------------|-----------------------------------------------|
| Default (most tasks)              | 1.0         | 0.95  | —     | 131072         | HF model card                                 |
| SWE-bench / Terminal Bench        | 0.7         | 1.0   | —     | 16384          | HF model card                                 |
| τ²-Bench (agentic)                | 0.0         | —     | —     | 16384          | HF model card                                 |
| Tool calling (article)            | 0.7         | 0.6   | 2     | —              | Medium guide (see Sources)                    |
| Coding (article)                  | 0.2         | 0.9   | —     | 4096           | Medium guide (see Sources)                    |
| Agentic / Tool calling (ollamaquery) | 0.7     | 1.0   | —     | —              | Gemini 3.5 agentic profile. min_p=0.01 (crucial — llama.cpp default 0.05 over-prunes vocabulary) |

**Prompt Style:** `strict` (default). GLM-4.7 supports OpenAI-style tool calling format natively via `--tool-call-parser glm47`. No preamble or code blocks needed.

**Special Features:**
- **Interleaved Thinking**: Thinks before every response and tool call.
- **Preserved Thinking**: Retains thinking blocks across multi-turn conversations (enabled via `chat_template_kwargs = {"enable_thinking": true, "clear_thinking": false}`).
- **Turn-level Thinking**: Per-turn control — disable thinking for lightweight requests, enable for complex tasks.
- **Tool calling**: Supports OpenAI-style tool descriptions; use `--tool-call-parser glm47` or `--reasoning-parser glm45` for vLLM/SGLang.

**Context Window:** 131072 tokens (max output).

#### Agentic Test Results (May 2026)

Empirical benchmarks from the 6-test suite in `tests/test_agentic.py::TestAgenticReActEndToEnd`, run on `glm-4.7-flash:q4_K_m` via Ollama.

**Temperature sensitivity** (test_direct_answer_no_tool):

| temp | Time | vs Default | Observation |
|------|------|-----------|-------------|
| 0.1 | >180s | Timeout | Stuck in meta-reasoning loop about formatting |
| 0.5 | >180s | Timeout | Overthinking about markdown vs plain text rules |
| **0.7** (default) | **5.5s** | **baseline** | Fast thinking, correct answer immediately |

**Full test suite** (temp=0.7, top_p=1.0, min_p=0.01):

| Test | Result | Time | Notes |
|------|--------|------|-------|
| test_direct_answer_no_tool | ✅ PASS | 5.5s | Fast, concise thinking block |
| test_fetch_url_tool | ✅ PASS | 5.8s | Correct tool call on first step |
| test_multi_step_write_file | ✅ PASS | 11.8s | Used write_file then read_file correctly |
| test_directory_lister_write_compile_run | ✅ PASS | 18.4s | Full write→gcc→test pipeline |
| test_port_scanner_write_compile_run | ✅ PASS | 132.3s | Intelligent debugging: used netstat to find correct listening IP |
| test_web_server_write_compile_run | ❌ FAIL | timeout | C code had NULL pointer in accept(); testing loop exhausted |

**Key observations:**
1. **temp=0.7 is optimal** — validated empirically. Lower temps cause overthinking the same way qwen3 does but with less severity.
2. **5-20x faster than qwen3:8b** — 5.5s vs 112s for simple queries. Thinking blocks are concise and focused.
3. **Intelligent debugging** — When port scanner reported CLOSED on 127.0.0.1:11434, the model used `netstat` to discover the service was on `192.168.1.20` and retested correctly.
4. **Does NOT try to start listeners** — Unlike qwen3 which tried to start `python3 -m http.server` as a test listener, glm-4.7 used investigation tools (netstat) instead.
5. **Web server test fails** — Same fundamental issue as qwen3: the `accept()` call with `NULL` pointer causes "Bad address" error. The model iterated through 3 C code fixes and multiple Python test scripts but the server kept crashing.
6. **Better shell operator awareness** — After 1-2 failures with `|`, `||`, `&&`, glm-4.7 stopped trying shell operators, while qwen3 kept retrying them.
7. **min_p=0.01 is correct** — Community recommendation validated. The default llama.cpp min_p=0.05 would over-prune during JSON tool call generation.

### Qwen3.5

**Source:** [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B#best-practices)

**Registered Key:** `qwen3.5`

| Parameter         | Value | Rationale                                    |
|-------------------|-------|----------------------------------------------|
| temperature       | 0.5   | Gemini 3.5: 0.0-0.5 for tool calling; lower prevents overthinking loops |
| top_p             | 0.9   | Gemini 3.5: 0.8-0.9 prevents wandering into low-probability tokens |
| top_k             | 20    | HF recommended for focused token selection   |
| min_p             | 0.0   | Disabled                                     |
| presence_penalty  | 0.0   | Disabled; agents repeat keys like "tool"     |
| repeat_penalty    | 1.0   | Critical: do not penalize JSON key repetition|

**Prompt Style:** `strict` — Qwen3.5 reliably outputs clean JSON tool calls with no preamble.

### Qwen3 8B

**Source:** [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B/blob/main/README.md)

**Registered Key:** `qwen3` (matches after the more specific `qwen3.5`)

**Tuning Parameters:**

| Parameter | Agentic tool calling | Rationale |
|-----------|---------------------|-----------|
| temperature | 0.5 | Gemini 3.5: 0.0-0.5 for tool calling; 0.5 allows some reasoning without overthinking loops |
| top_p | 0.9 | Gemini 3.5: 0.8-0.9 prevents wandering during tool selection |
| top_k | 20 | HF recommended for format strictness |
| min_p | 0.0 | Disabled |
| presence_penalty | 0.0 | Disabled (agents repeat keys like "tool") |
| repeat_penalty | 1.0 | Critical: do not penalize JSON syntax repetition |

**Usage notes:**
- **Thinking mode** (`enable_thinking=True`, default): Model outputs `<think>...</think>` before the answer. Use for complex logic, math, coding, and tool calling. Greedy decoding is NOT recommended (causes endless repetition).
- **Non-thinking mode** (`enable_thinking=False`): Faster, direct answers for simple tasks.
- For tool calling/agentic use, thinking mode with temp=0.6 is recommended.

**Context Window:** 32,768 tokens native, up to 131,072 with YaRN.

**Prompt Style:** `strict` — outputs clean JSON tool calls.

#### Agentic Test Results (May 2026)

Empirical benchmarks from the 6-test suite in `tests/test_agentic.py::TestAgenticReActEndToEnd`.

**Temperature sweep** (test_direct_answer_no_tool):

| temp | Time | vs Default | Observation |
|------|------|-----------|-------------|
| **0.1** | **15.8s** | **7x faster** | Minimal thinking, direct answer. Best for simple single-step tool calls |
| 0.5 | 111.9s | baseline | Verbose `<think>` blocks deliberating formatting rules and edge cases |
| 0.7 | 89.9s | 1.2x faster | Less deliberation, but occasional format errors |

**Penalty sweep** (test_direct_answer_no_tool):

| repeat_penalty | presence_penalty | Time | Observation |
|----------------|------------------|------|-------------|
| 1.0 | 0.0 | 111.9s | Baseline |
| 1.2 | 0.0 | 188.4s | Penalty backfires — more overthinking about repetition |
| 1.0 | 0.3 | >200s timeout | Severe overthinking loop, never resolved |

**Full test suite** (temp=0.5, top_p=0.9, top_k=20):

| Test | Pass Rate | Avg Time | Notes |
|------|-----------|----------|-------|
| test_direct_answer_no_tool | 100% | 111.9s | Overthinks "one word" vs markdown formatting |
| test_fetch_url_tool | 100% | 9.9s | Correct tool call on first step |
| test_multi_step_write_file | 100% | 19.7s | Uses write_file then read_file correctly |
| test_directory_lister_write_compile_run | 100% | 116.4s | Full write → gcc → test pipeline works |
| test_port_scanner_write_compile_run | ~60-80% | 192.2s | Sometimes outputs code as text instead of using write_file tool |
| test_web_server_write_compile_run | ~60-80% | 119.4s | Complex task; model sometimes falls back to text output |
| test_dns_resolver_write_compile_run | ~0% | >10min | Times out on complex DNS packet construction |

**Temperature recommendations by task type:**

| Task Type | Recommended temp | Rationale |
|-----------|-----------------|-----------|
| Simple single-step tool calls (fetch, write) | **0.1** | Fast, deterministic, 7x speedup |
| Multi-step pipeline (write → compile → test) | **0.5** | Needs enough randomness to make multiple tool calls |
| Code generation with compilation | **0.5-0.6** | Lower temps get stuck in overthinking about tool access |

**Key empirical observations:**
1. **temp=0.1 is fastest for simple queries** (7x) but causes meta-reasoning loops on complex multi-step tasks where the model debates whether it has tool access
2. **Penalties backfire** — increasing `repeat_penalty` or `presence_penalty` makes thinking blocks longer, never shorter
3. **Step timeout of 120s is critical** — verbose `<think>` blocks regularly consume 30-120s before any output
4. **Multi-turn behavior**: maintains JSON tool-calling format across turns; adapts tool usage based on previous outcomes (e.g., discovers wrong IP, investigates with `ss`, retests correctly)
5. **Lazy tool mode needed** (`lazy_tool=True`): the model sometimes embeds tool calls mid-text rather than at the start of a response
6. **Task complexity affects tool adoption**: simple file ops always use tools; networking code (sockets, HTTP) causes the model to revert to text output ~20-40% of the time — the model treats it as "educational content" rather than a task to execute

### DeepSeek-R1 Distill (Qwen3-8B)

**Source:** [DeepSeek-R1-Distill-Qwen-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-8B) (Ollama: `ollama run deepseek-r1:8b`)

**Registered Key:** `deepseek` (matches `deepseek-r1:8b`, `deepseek-r1:14b`, etc.)

**Architecture:** Based on Qwen3-8B, fine-tuned with 800k reasoning samples from DeepSeek-R1.

| Parameter | Agentic tool calling | Rationale |
|-----------|---------------------|-----------|
| temperature | 0.7 | Ollama modelfile defaults to 0.6; 0.7 is 2-4x faster with same accuracy |
| top_p | 0.95 | Ollama modelfile default |
| top_k | 20 | Qwen3 base architecture recommendation |
| min_p | 0.0 | Disabled |
| presence_penalty | 0.0 | Disabled (agents repeat keys like "tool") |
| repeat_penalty | 1.0 | Critical: do not penalize JSON syntax repetition |

**Prompt Style:** `strict` — outputs clean JSON tool calls when reasoning completes.

**Usage notes:**
- **Reasoning model**: Outputs `<think>...</think>` blocks before every response. Thinking is verbose (30-200s per step).
- **Tool calls bypass reasoning**: When a tool call is needed, the thinking block is much shorter (e.g., 13s for fetch_url vs 104s for text response).
- **Not suitable for speed-critical**: Each agentic step takes 20-200s due to reasoning. Multi-step pipelines (write→compile→test) can take 2-5 minutes.
- **temp=0.7 recommended over 0.6**: 0.6 causes more verbose thinking (238s vs 57s for simple query) without quality improvement.

#### Agentic Test Results (May 2026)

Empirical benchmarks from the 6-test suite in `tests/test_agentic.py::TestAgenticReActEndToEnd`.

**Temperature sensitivity** (test_direct_answer_no_tool):

| temp | Time | vs Default | Observation |
|------|------|-----------|-------------|
| 0.1 | timeout | — | Stuck in endless reasoning loop |
| 0.6 (Ollama default) | 238.1s | 4x slower | Very verbose `<think>` block |
| **0.7** (recommended) | **57.5s** | **baseline** | Fast enough, correct answer |

**Full test suite** (temp=0.7, top_p=0.95, top_k=20):

| Test | Result | Time | Notes |
|------|--------|------|-------|
| test_direct_answer_no_tool | ✅ PASS | 57.5s | Verbose thinking about formatting rules |
| test_fetch_url_tool | ✅ PASS | 21.6s | Tool call bypasses reasoning; fast |
| test_multi_step_write_file | ✅ PASS | 14.2s | Wrote and read file correctly |
| test_directory_lister_write_compile_run | ✅ PASS | 18.2s | Full write→gcc→test pipeline |
| test_port_scanner_write_compile_run | ✅ PASS | 104.5s | Accepted port CLOSED result gracefully |
| test_web_server_write_compile_run | ❌ FAIL | 74.8s | Output code as text instead of tools |

**Key observations:**
1. **Tool calls bypass reasoning verbosity** — When the model needs to output a JSON tool call, the `<think>` block is dramatically shorter (13s vs 104s for text responses). This is because tool calling is a well-trained pattern.
2. **temp=0.7 is recommended over 0.6** — The Ollama modelfile default of 0.6 causes 4x more verbose thinking without better output quality.
3. **Graceful handling of unexpected results** — Unlike qwen3 (which tried to start listeners) or glm-4.7 (which used netstat), deepseek-r1 simply accepted the port CLOSED result and explained it.
4. **5/6 tests passed** — Same as qwen3:8b. Web server test fails due to single-tool-per-step limitation.
5. **Reasoning overhead is significant** — Multi-step pipelines require multiple rounds of reasoning (each 20-60s), adding 2-5x overhead compared to non-reasoning models.

### Nemotron-Cascade

**Source:** [nvidia/Nemotron-Cascade-2-30B-A3B](https://huggingface.co/nvidia/Nemotron-Cascade-2-30B-A3B)

**Registered Key:** `nemotron-cascade`

| Parameter         | Value | Rationale                                              |
|-------------------|-------|--------------------------------------------------------|
| temperature       | 0.6   | HF recommends 1.0; lowered for deterministic tool calling |
| top_p             | 0.95  | HF recommended                                         |
| top_k             | 40    | HF recommended                                         |
| min_p             | 0.0   | Disabled                                               |
| presence_penalty  | 0.0   | Disabled                                               |
| repeat_penalty    | 1.0   | Default                                                |

**Prompt Style:** `soft` — Nemotron-Cascade outputs JSON inside ````json` code blocks with preamble. Refuses to output clean JSON when told "no surrounding text". The soft prompt allows code blocks, preamble, and multi-tool-per-response.

### Nemotron (generic)

**Source:** [nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16)

**Registered Key:** `nemotron`

| Parameter         | Value | Rationale                              |
|-------------------|-------|----------------------------------------|
| temperature       | 0.6   | "Recommended for tool calling" (HF)    |
| top_p             | 0.95  | HF recommended                         |
| top_k             | 40    | HF recommended                         |
| min_p             | 0.0   | Disabled                               |
| presence_penalty  | 0.0   | Disabled                               |
| repeat_penalty    | 1.0   | Default                                |

**Prompt Style:** `strict` (default). Generic Nemotron models follow strict JSON format well.

**Note:** This entry matches any model name containing "nemotron" (e.g., "nvidia-nemotron-3-nano-4b"). The more specific `nemotron-cascade` key takes priority due to ordering in the registry.

### GPT-OSS

**Source:** Gemini 3.5 agentic profile (no specific HF model card).

**Registered Key:** `gpt-oss`

| Parameter         | Value | Rationale                                   |
|-------------------|-------|---------------------------------------------|
| temperature       | 0.7   | Gemini 3.5: 0.0 for strict JSON or 0.7 for CoT reasoning |
| top_p             | 1.0   | Gemini 3.5: model's RL alignment handles token filtering |
| min_p             | 0.0   | Disabled                                    |
| presence_penalty  | 0.0   | Disabled (avoids corrupting chat formatting)|
| repeat_penalty    | 1.0   | Default                                     |

**Prompt Style:** `strict` (default).

### Default (unknown models)

**Source:** `DEFAULT_INFERENCE_PARAMS` (`ollamaquery2.py:1131`).

| Parameter         | Value |
|-------------------|-------|
| temperature       | 0.7   |
| top_p             | 0.9   |
| top_k             | 40    |
| min_p             | 0.0   |
| presence_penalty  | 0.0   |
| repeat_penalty    | 1.0   |

**Prompt Style:** `strict`.

---

## How Parameters Are Applied

Both registries are consulted dynamically at query time in the ReAct loop (`run_agentic_query` at `ollamaquery2.py:3968`):

```python
params = get_inference_params(self.ctx.model)
messages = [{'role': 'system', 'content': get_agentic_prompt(self.ctx.model, tool_defs)}]
response = self.query_handler.query_sync(..., **params)
```

- `get_inference_params(model_name)` — Matches model name against `MODEL_INFERENCE_PARAMS_REGISTRY`, returns matching dict or `DEFAULT_INFERENCE_PARAMS`.
- `get_agentic_prompt(model_name, tool_defs)` — Assembles composable prompt blocks (role, tool defs, format, examples, rules) using the style determined by `get_prompt_style(model_name)`.
- Parameters are **not** stored in `CommandContext`. Changing models mid-session (via `/switchmodel`) automatically triggers a fresh lookup on the next agentic query.
- Params are unpacked as `**kwargs` into both `query_sync` and `query_stream` calls inside the ReAct loop. They become top-level fields in the llama.cpp OpenAI-compatible API payload.

---

## Sources

| Model                     | Source URL                                                                 | Access Method                     |
|---------------------------|----------------------------------------------------------------------------|-----------------------------------|
| GLM-4.7                   | https://huggingface.co/unsloth/GLM-4.7                                     | Direct                            |
| GLM-4.7 Guide             | https://medium.com/@zh.milo/glm-4-7-flash-the-ultimate-2026-guide-to-local-ai-coding-assistant-93a43c3f8db3 | `curl https://r.jina.ai/<URL>` (jina.ai reader proxy bypasses Cloudflare) |
| GLM-4.5 Technical Report  | https://arxiv.org/abs/2508.06471                                           | Direct                            |
| Qwen3.5-9B                | https://huggingface.co/Qwen/Qwen3.5-9B#best-practices                     | Direct                            |
| Nemotron-Cascade          | https://huggingface.co/nvidia/Nemotron-Cascade-2-30B-A3B                   | Direct                            |
| Nemotron-3-Nano           | https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16              | Direct                            |
| Qwen3-8B                  | https://huggingface.co/Qwen/Qwen3-8B/blob/main/README.md                  | Direct (HF README)                 |
| DeepSeek-R1-Distill       | https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-8B             | Access requires HF token (gated repo) |
| DeepSeek-R1 Ollama        | https://ollama.com/library/deepseek-r1                                    | Direct                            |
| Gemini 3.5 agentic advice | Community-sourced tips for local agentic LLM inference — not official model card recommendations, use with caution | Shared by user, untested |
