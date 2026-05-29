# Agentic End-to-End Tests

## Purpose

Validate that a real LLM can use the agentic tool system to solve multi-step tasks end-to-end. Each test exercises the full pipeline: the model must plan, use tools, observe results, and produce a working outcome.

## Test Plan

| # | Test | Tools required | What it proves |
|---|------|---------------|----------------|
| 1 | **DNS resolver** — write `dns_resolver.c`, compile with `gcc`, test it | `write_file`, `run_command` | Code generation, compilation, execution |
| 2 | **Simple answer** — direct answer without tools | (none) | Non-tool queries work |
| 3 | **URL fetch** — fetch a URL, extract page title | `fetch_url` | External data retrieval |
| 4 | **Multi-step write+read** — write file, then read it back | `write_file`, `read_file` | Multi-step tool chaining |

## Test 1: DNS Resolver

**Location:** `tests/test_agentic.py` → `TestAgenticReActEndToEnd.test_dns_resolver_write_compile_run`

**Prompt:**
> write a dns_resolver.c, compile it, then test it. This dns_resolve will take two arguments: <dnsserverip> and <FQDN> and the tool will ask to the <dnsserverip>:53 using udp the dns query and give back the IPv4 address of the resolution.

**Pipeline:**

| Step | Tool | What happens | Verification |
|------|------|-------------|-------------|
| 1 | `write_file` | LLM writes `dns_resolver.c` | File created |
| 2 | `run_command` | LLM compiles with `gcc -o dns_resolver dns_resolver.c` | Binary created |
| 3 | `run_command` | LLM tests with `./dns_resolver <dns> <fqdn>` | Binary runs |
| 4 | — | Session completes | `dns_resolver` binary exists on disk |

**Tool parameters introduced:**
- `run_command` and `run_python` now accept an optional `timeout` parameter (integer, default 10s, max 300s). The LLM can set per-call timeouts for long-running compilations or tests.
- Chaining operators (`&&`, `||`, `;`, `|`, `` ` ``, `$(`) are blocked. The LLM receives a JSON error: `"Only one command at a time is supported."`
- Tool results are now structured JSON: `{"tool": ..., "duration_s": ..., "success": ..., "output": ...}` instead of plain text.

## Requirements

- A running Ollama or llama.cpp server with a loaded model capable of code generation and tool use
- Recommended models: `qwen3.5:9b`, `GLM-4.7-Flash`, `gpt-oss-20b`, `nemotron-cascade-2_30b`
- `gcc` must be installed on the host (for compilation tests)

## Test Configuration

The E2E tests enable these debug features by default (`setUp` in `TestAgenticReActEndToEnd`):

| Setting | Value | Purpose |
|---------|-------|---------|
| `agentic_mode` | `True` | Enables the ReAct loop |
| `lazy_tool` | `True` | Extract tool calls from anywhere in model response |
| `auto_confirm` | `True` | Skip destructive tool confirmation prompts |
| `agentic_verbose` | `True` | Show raw model responses during ReAct |
| `agentic_show_thinking` | `True` | Show model reasoning in `<thinking>` blocks |
| `agentic_trace` | `True` | Show full tool args and results |
| `agentic_logging` | `False` | Disabled (test isolation) |
| `agentic_max_iterations` | `30` | Stop early on stuck models |

## Tool Delivery Format

Tool descriptions are delivered to the model via one of two strategies, selected per-model:

| Format | Mechanism | Models |
|--------|-----------|--------|
| `openai` | Native OpenAI `tools` API parameter. System prompt stays clean (no inline tool defs). | GPT-OSS, Qwen3.x, Llama 3.x |
| `inline` | Tool descriptions embedded in system prompt text. No `tools` API param. | GLM-4.x, Qwen3.5, Nemotron, unknown |

Key rule: **tools are sent only once** — never both inline and via API. If a model rejects the native `tools` API, the system auto-falls back to `inline` mode (via `check_tools_error()`).

## How to Run

```bash
# Run DNS resolver with default Ollama host
python3 -m unittest tests.test_agentic.TestAgenticReActEndToEnd.test_dns_resolver_write_compile_run -v

# With custom host and model
OLLAMA_HOST=http://my-server:11434 TEST_MODEL=qwen3.5:9b \\
  python3 -m unittest tests.test_agentic.TestAgenticReActEndToEnd.test_dns_resolver_write_compile_run -v

# With llama.cpp backend
LLAMACPP_HOST=http://127.0.0.1:8080 TEST_MODEL=GLM-4.7-Flash-Q4_K_M.gguf \\
  python3 -m unittest tests.test_agentic.TestAgenticReActEndToEnd -v

# Run all end-to-end tests (auto-detects backend)
python3 -m unittest tests.test_agentic.TestAgenticReActEndToEnd -v
```

## Backend Auto-Detection

The test suite automatically detects available backends:

1. Tries `OLLAMA_HOST` (default `http://127.0.0.1:11434`)
2. Falls back to `LLAMACPP_HOST` (default `http://127.0.0.1:8080`)
3. Skips with a clear message if neither is reachable

The test picks the first non-embedding model from the server's model list, preferring `qwen3.5:9b` if available.

## Lazy Tool Mode

These tests enable `lazy_tool = True` since many models embed tool call JSONs in explanatory text rather than outputting them cleanly at the start of the response. See `/agentic lazytool` for details.

## Test Results (May 2026)

### Infrastructure Changes

The following changes were made to improve agentic reliability across all models:

| Change | Description |
|--------|-------------|
| `agentic_max_iterations: 50` | Default increased from 10 to give the model room to retry after compilation failures |
| `MODEL_INFERENCE_PARAMS_REGISTRY` | Per-model inference parameters sourced from each model's HuggingFace page. Qwen3.5 uses `temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repeat_penalty=1.0` ("precise coding tasks" setting). Nemotron-Cascade-2 uses `temperature=0.6, top_p=0.95, top_k=40`. Falls back to conservative defaults for unknown models. |
| `edit_file` tool | Precise text replacement in existing files (inspired by OpenCode's `edit`). Accepts `file_path`, `old_string`, `new_string`. Requires exactly one match. |
| `apply_patch` tool | Pure-Python unified diff applicator (inspired by OpenCode's `apply_patch`). Accepts `patch_text` with file paths embedded in diff headers. Supports standard unified diffs (`---`/`+++`/`@@`) and OpenCode-style markers (`*** Add File:`, `*** Update File:`, `*** Delete File:`, `*** Move to:`). |
| `AGENTIC_SYSTEM_PROMPT_REGISTRY` | Per-model agentic system prompts, mirroring the inference params registry. Nemotron-Cascade gets `AGENTIC_SYSTEM_PROMPT_SOFT` (code-block-friendly), Qwen3.5/GPT-OSS get the strict `AGENTIC_SYSTEM_PROMPT` ("ONLY JSON, no surrounding text"). Lookup via `get_agentic_prompt(model_name)`. |
| `parse_tool_call` multi-line JSON fix | Regex `\{"tool"` changed to `\{\s*"(tool\|function\|type)"` to match models (like Nemotron) that put `{` and `"tool"` on separate lines. Applies to lazy-mode bare JSON extraction, strict-mode start-of-text check, and end-of-text check. |
| `parse_tool_calls` multi-line JSON fix | Same `\{\s*"..."` regex applied to both lazy mode (find-all) and strict mode (consecutive-from-start) extraction. |
| `run_agentic_query` model-specific prompt | Previously used a hardcoded prompt string literal. Now calls `get_agentic_prompt(self.ctx.model)` so registered model prompts take effect. |
| `run_agentic_query` model-specific prompt | Previously used a hardcoded prompt string literal. Now calls `get_agentic_prompt(self.ctx.model)` so registered model prompts take effect. |
| Helper methods: `_find_tool_call_json`, `_rfind_tool_call_json`, `_extract_json_balanced` | Extracted from inline loop logic for reuse across `parse_tool_call` and `parse_tool_calls`. `_extract_json_balanced` does brace-depth tracking to extract a complete JSON object from any starting position. |
| `timeout` parameter in tool defs | Added optional `timeout` (integer, default 10, max 300) to `run_command` and `run_python`. Tool returns JSON error if timeout exceeds 300s. |
| Chaining operators blocked in `run_command` | `&&`, `||`, `|`, `;`, `` ` ``, `$(` are rejected with explicit error: `"Only one command at a time is supported."` |
| Tool result is JSON | Observation is now `{"tool": ..., "duration_s": ..., "success": ..., "output": ...}` instead of plain text. On failure, first 4 lines of output are included for context. |
| Default tool timeout lowered to 10s | Reduced from 120s to 10s. LLM can override via the `timeout` parameter. Teaches the model to set appropriate timeouts. |

### Model Comparison

| Dimension | GPT-OSS 20B | Qwen3 8B | GLM-4.7-Flash | Nemotron-Cascade-2 30B | Gemma4 26B |
|-----------|-------------|----------|---------------|------------------------|-------------|
| **Parameter count** | 20B (mxFP4) | 8B | ~9B | 31.6B (MoE, 3B active) | 26B (MoE, 4B active) |
| **Tool format** | `openai` (native API) | `openai` (native API) | `inline` (JSON-in-text) | `inline` (JSON-in-text) | `inline` (JSON-in-text) |
| **Prompt style** | `strict` (bare JSON) | `strict` (bare JSON) | `strict` (bare JSON) | `soft` (code blocks OK) | `strict` (bare JSON) |
| **Native `tools` API** | ✅ | ✅ | ❌ (uses inline) | ❌ (uses inline) | ❌ (uses inline) |
| **`lazy_tool` needed?** | No | No | ✅ | ✅ | ✅ |
| **Temperature** | 0.7 | 0.7 (default) | 0.6 | 0.6 | 0.7 (default) |
| **DNS resolver** | ✅ 4 steps, 89s | ❌ (compile errors) | ✅ 4 steps, 259s | ❌ (C syntax bug) | ✅ 3 steps* |
| **URL fetch** | ✅ Chained http→https | ✅ | ✅ | ✅ | ✅ |
| **Multi-step write+read** | ✅ Clean | ✅ Clean | ✅ Clean | ✅ (recovered from typos) | ✅ Clean |
| **Code quality** | Excellent (single-shot) | Mediocre (compile bugs) | Good | Mediocre (parsing bugs, typos) | Good |
| **`edit_file`/`apply_patch` used?** | No | No | No | No | No |
| **Multi-tool per response** | No | No | No | Yes (verbose preamble + code blocks) | No |

\* Gemma4 needed `agentic_step_timeout > 120s` due to slow thinking generation.

### Detailed Per-Model Results (May 2026)

#### GPT-OSS 20B (openai format — native tools API)

**Run 1 (tool_format="openai", lazy_tool=False):**

| Step | Tool | Observation |
|------|------|-------------|
| 1 | `write_file` | Wrote `dns_resolver.c` (5532 bytes) — correct single-shot code |
| 2 | `run_command` | `gcc dns_resolver.c -o dns_resolver` — compiled cleanly |
| 3 | `run_command` | `./dns_resolver 8.8.8.8 example.com` — **ran successfully**: `172.66.147.243` |
| 4 | (text answer) | Presented full summary with code and test output |
| — | **Result** | Test **OK** (89s). All 4 E2E tests passed. |

**Key observations:**
- Used native `tools` API — system prompt was clean (no inline tool defs)
- Model output native `tool_calls` via API, not inline JSON
- Single-shot correct C code, no debugging iterations needed
- Fastest DNS resolver completion of all tested models

#### GLM-4.7-Flash (inline format — JSON-in-text)

**Run 1 (tool_format="inline", lazy_tool=True):**

| Step | Tool | Observation |
|------|------|-------------|
| 1 | `write_file` | Wrote `dns_resolver.c` (6509 bytes) |
| 2 | `run_command` | `gcc -o dns_resolver dns_resolver.c -Wall -Wextra` — compiled with warnings |
| 3 | `run_command` | `./dns_resolver 8.8.8.8 www.google.com` — timeout (network sandbox) |
| 4–27 | (various) | Debug loop: tried DNS servers, dig tests, fix code |
| — | **Result** | Test **OK** (374s total for all 4 tests). Binary existed. |

**Key observations:**
- Inline format produced reliable multi-step chaining on llama.cpp
- Conversation history includes inline JSON in assistant content (always visible to model)
- Longer debugging loop due to network sandbox restrictions on UDP 53
- `lazy_tool=True` required — model sometimes embeds JSON in preamble text

**Ollama backend run (glm-4.7-flash:q4_K_m @ 192.168.1.20:11434):**

| Test | Result | Notes |
|------|--------|-------|
| `test_direct_answer_no_tool` | ✅ | |
| `test_fetch_url_tool` | ✅ | |
| `test_multi_step_write_file` | ✅ | |
| `test_dns_resolver_write_compile_run` | ❌ | Output code as text, never used `write_file` tool |

**Finding:** GLM-4.7 behavior is **consistent across backends** — on both llama.cpp and Ollama, it handles simple tools reliably but outputs code as text instead of using tools for complex code generation tasks. This is a model-inherent behavior, not backend-dependent.

#### Nemotron-Cascade-2 30B (inline format, soft prompt)

**Run 1 (tool_format="inline", prompt_style="soft"):**

| Step | Tool | Observation |
|------|------|-------------|
| 1 | `write_file` | Wrote `dns_resolver.c` with syntax error — `definition int` vs `int` |
| 2 | `read_file` (typo) | Attempted `dns_olver.c` (wrong name) — file not found |
| 3 | (same loop) | Repeated same typo — loop guard triggered |
| — | **Result** | Test **FAILED**. Binary not created. Other 3 E2E tests passed. |

**Key observations:**
- Soft prompt allows verbose preamble and code block tool calls
- Model makes typos in tool names (`write_ file`) and file paths (spaces in names)
- Multi-tool-per-response supported but error-prone
- DNS resolver failed due to C syntax bug the model couldn't recover from

#### Gemma4 26B (inline format — JSON-in-text)

**Run 1 (tool_format="inline", lazy_tool=True, agentic_step_timeout=300):**

| Step | Tool | Observation |
|------|------|-------------|
| 1 | `write_file` | Wrote `dns_resolver.c` (5017 bytes) with DNS structures |
| 2 | `run_command` | `gcc dns_resolver.c -o dns_resolver` — compiled cleanly |
| 3 | `run_command` | `./dns_resolver 8.8.8.8 google.com` — parser returned `No A records found` |
| 4 | (timeout) | Next step timed out (300s) |
| — | **Result** | Binary existed. C code had simplified parser (no compressed name handling). |

**Key observations:**
- Default 120s step timeout was too short for initial code generation (Gemma4 is slow to start generating)
- With 300s timeout, the write→compile→run pipeline completed in 3 steps
- C code was good but had a simplified DNS parser (couldn't handle compressed names)
- Step timeout requirement is hardware-dependent, not a code issue

#### Qwen3 8B (openai format — native tools API)

**Run 1 (tool_format="openai", native tools API, Ollama backend):**

| Step | Tool | Observation |
|------|------|-------------|
| 1 | `write_file` | Wrote `dns_resolver.c` (6484 bytes) — used `arpa/nameser.h` and `resolv.h` |
| 2 | `write_file` | Rewrote with manual DNS packet building (dropped system headers) |
| 3 | `run_command` | `gcc -o dns_resolver dns_resolver.c -lc -ldns -lnet -lresolv` — **compile error**: redeclared `anc` variable |
| 4 | (streaming) | Output inline JSON tool call in streaming response → rewrote file |
| — | **Result** | Test **FAILED**. Binary not created. |

**Key observations:**
- Native `tools` API worked for the first tool call; subsequent calls fell back to inline JSON in streaming
- 8B model lacks the code quality to produce correct DNS resolver C code
- Compilation bugs (variable redeclaration) and wrong linker flags (`-lc -ldns -lnet`)
- Simple tools (fetch_url, write_file, read_file) worked reliably
- Code generation task exceeded model capability

### Key Findings (All Models)

1. **`edit_file` / `apply_patch` never adopted** — Across all tested models, tool descriptions alone don't steer models away from `write_file`. Even for small changes (adding debug output, fixing a single line), models rewrite the entire file. This is consistent regardless of model size (9B–31.6B) or tool-calling approach (native API vs JSON-in-text).

2. **Inference params improve focus** — With `temperature=0.6` and constrained `top_k` (20–40), models stay on task and make consistent progress toward compilation. Higher temperatures cause wandering (unnecessary rewrites, unrelated changes).

3. **50 iterations gave ample room** — All runs completed within 14 steps. The old 10-iteration limit would have cut off Qwen3.5 Run 2 at step 10, right when the model was mid-fix.

4. **Same-tool-loop guard caught perfectionism** — Qwen3.5 kept rewriting the binary even after it worked. The guard (`identical tool+args twice in a row`) broke it out. Without this guard, the model would fill all 50 iterations with rewrite cycles.

5. **Model size correlates with single-shot accuracy** — GPT-OSS (20B) produced correct code on the first try. Qwen3.5 (9B) needed iterations. Nemotron (31.6B overall but only 3B active MoE params) produced code with bugs.

6. **Native `tools` API vs JSON-in-text** — GPT-OSS uses the OpenAI `tools` parameter and returns `finish_reason: "tool_calls"`. Qwen3.5 and Nemotron don't — they rely on JSON-in-text extraction via `parse_tool_calls` / `parse_tool_call` with `lazy_tool=True`.

7. **Multi-tool-per-response (Nemotron only)** — Nemotron outputs all three steps (write, compile, run) in a single response inside separate ````json` code blocks. Qwen3.5 and GPT-OSS output one tool call per response.

8. **`run_agentic_query` had a hardcoded system prompt bug** — The prompt was embedded as a string literal in `run_agentic_query()` (line 3751), bypassing both `self.ctx.system_prompt` and `get_agentic_prompt()`. This meant model-specific prompts (set by `/agentic on` or `/agentic full` via `run_handle_agentic`) were silently ignored inside the ReAct loop. Fixed to use `get_agentic_prompt(self.ctx.model)`.

9. **Tool descriptions must be short** — GLM-4.7 stopped using tools entirely when the `run_command` description grew to include chaining restrictions and timeout limits (~300 chars). After trimming it back to focus on what the tool **does** (not what it can't do), the model resumed tool use.

10. **Models misinterpret timeout units** — GLM-4.7 used `timeout=30000` in tool calls, treating it as milliseconds. Explicit validation (max 300s + clear JSON error) is essential. Models learn from the error after one try.

11. **JSON tool results help debugging** — Structuring tool observations as `{"tool", "duration_s", "success", "output"}` gives the model machine-readable context. On failure, including the first 4 lines of output helps the model understand why something failed instead of blindly retrying.

12. **`gcc -o <name> <name>.c` conflicts confuse models** — When the binary name matches the source basename (`gcc -o dns_resolver dns_resolver.c`), subsequent compilation fails because the existing binary blocks the output. Models struggle with this, often retrying with `rm -f` (which hits shell chaining blocks) or creating increasingly strange binary names (`_new`, `_v2`, `_v3`).

### Architecture: Model-Specific Agentic Prompts

The prompt system has three layers:

```
AGENTIC_SYSTEM_PROMPT          (strict: "ONLY JSON, no surrounding text")
AGENTIC_SYSTEM_PROMPT_SOFT     (soft: code blocks OK, multi-step example)
AGENTIC_SYSTEM_PROMPT_REGISTRY (maps model substrings → prompt)
```

**Lookup function** (`get_agentic_prompt`, mirroring `get_inference_params`):

```python
def get_agentic_prompt(model_name: str) -> str:
    lower = model_name.lower()
    for key, prompt in AGENTIC_SYSTEM_PROMPT_REGISTRY.items():
        if key in lower:
            return prompt
    return DEFAULT_AGENTIC_SYSTEM_PROMPT
```

**Current registry entries:**

| Key | Prompt | Models matched |
|-----|--------|----------------|
| `nemotron-cascade` | `AGENTIC_SYSTEM_PROMPT_SOFT` | `nemotron-cascade-2_30b.gguf` |
| (default) | `AGENTIC_SYSTEM_PROMPT` | Everything else (Qwen3.5, GPT-OSS, etc.) |

**Where prompts are applied:**

1. **`run_handle_agentic`** (lines ~3340, ~3350) — When `/agentic on` or `/agentic full` is called, stores model-specific prompt in `self.ctx.system_prompt` via `get_agentic_prompt(self.ctx.model)`. Also backs up the previous prompt in `self.ctx._saved_system_prompt` for restore on `/agentic off`.

2. **`run_agentic_query`** (line ~3751) — Uses `get_agentic_prompt(self.ctx.model)` directly as the system message for every iteration of the ReAct loop. Previously used a hardcoded string literal that ignored both the registry and `self.ctx.system_prompt`.

### Technical Details: Regex Fixes

**Problem:** Nemotron-Cascade outputs JSON with `{` and `"tool"` on separate lines:

```json
{
  "tool": "write_file",
  "arguments": {
    "file_path": "hello.c",
    ...
  }
}
```

The original pattern `\{"tool"` only matches `{"tool"` on the same line (no whitespace between `{` and `"tool"`).

**Fix:** Changed all occurrences in `parse_tool_call` and `parse_tool_calls`:

| Location | Before | After |
|----------|--------|-------|
| Lazy mode bare JSON search (line 3526) | `r'\\{"tool"\\s*:\\s*"[^"]+"'` | `r'\\{\\s*"(tool\|function\|type)"\\s*:\\s*"[^"]+"'` |
| Strict mode start check (line 3546) | `stripped.startswith('{"tool"}')` | `re.match(r'\\s*\\{\\s*"(tool\|function\|type)"', text)` |
| Strict mode end rfind (line 3561) | `text.rfind('{"tool"}')` | `_rfind_tool_call_json(text)` using compiled regex `r'\\{\\s*"(tool\|function\|type)"'` |
| `parse_tool_calls` lazy find (line 3601) | `text.find('{"tool"}', pos)` | `_find_tool_call_json(text, pos)` |
| `parse_tool_calls` strict start (line 3626) | `text.startswith('{"tool"}')` | `re.match(r'\\s*\\{\\s*"(tool\|function\|type)"', text)` |

**Compiled regex added as class constant:**

```python
_TOOL_CALL_OPEN_RE = re.compile(r'\{\s*"(tool|function|type)"')
```

**Helper methods added:**

- `_find_tool_call_json(text, pos)` — Find next `{` followed by optional whitespace then `"tool"`, `"function"`, or `"type"`. Returns index or -1.
- `_rfind_tool_call_json(text)` — Find the last such occurrence. Uses `finditer` and returns last match's start position.
- `_extract_json_balanced(text, start)` — Starting at an opening brace `{`, tracks brace depth to extract a complete JSON object. Returns the substring or None if unbalanced.

### Future Work: Refining the System Prompt

The current prompt system is a basic substring-match registry. Several improvements are planned:

1. **Dynamic prompt assembly** — Instead of shipping monolithic prompt strings, the prompt could be assembled from composable blocks:
   - **Role block**: "You are a capable AI agent..."
   - **Tool definitions block**: Automatically generated from `AGENTIC_TOOL_DEFS` (already done via `get_system_prompt_block()`)
   - **Format block**: Model-specific instructions about JSON formatting (strict `"ONLY JSON, no surrounding text"` vs soft `"code blocks are OK"`)
   - **Rules block**: Task-specific rules (multi-tool batching, output style, language mirroring)
   - **Example block**: In-context examples tailored to the model's strengths/weaknesses

2. **Per-model formatting rules** — Models differ in how they prefer to output tool calls:
   - Qwen3.5: Strict JSON at start of response, no preamble
   - GPT-OSS: Native `tools` API (system prompt is secondary)
   - Nemotron-Cascade: Multi-line JSON in ````json` code blocks with preamble
   - Future models may use XML tags, YAML, or other formats

3. **Prompt ordering matters** — Models pay more attention to the beginning and end of long prompts. The current system appends tool definitions at the end via `get_system_prompt_block()`. For models with limited attention to prompt middles, the tool section might need to move earlier.

4. **Example-guided prompting** — Instead of abstract rules ("output a JSON tool call"), provide 1–2 concrete multi-step examples:
   ```
   User: Write hello.c and compile it
   Assistant: {"tool": "write_file", "arguments": {"file_path": "hello.c", "content": "..."}}
   [Tool result: Written]
   Assistant: {"tool": "run_command", "arguments": {"command": "gcc hello.c -o hello"}}
   ```
   Few-shot examples improve compliance for weaker instruction-following models.

5. **Context length awareness** — The full prompt (system + tool definitions) can exceed 2000 tokens. For models with small context windows (4096), this leaves limited room for conversation history and code. The registry could include a `max_prompt_tokens` hint to truncate tool descriptions.

6. **Prompt versioning** — As models receive updates (new GGUF quants, fine-tunes), the optimal prompt may change. Versioned prompts would allow A/B testing. Each registry entry could include a `version` field.

### Recommendations

- **Use model-specific prompts via the registry** — `get_agentic_prompt(model_name)` should be called whenever the agentic system prompt is set. Currently used in `run_handle_agentic` and `run_agentic_query`.
- **Fix `parse_tool_call` regex for any future models** — The `\{\s*"` pattern handles multi-line JSON. New models may need additional patterns (XML tags, YAML).
- **Consider lowering `temperature` further** — 0.4–0.6 is the sweet spot for deterministic tool calling across all tested models.
- **Add a `--tool-bias` or tool ordering** — List `edit_file` before `write_file` in tool definitions to encourage targeted edits over full rewrites.
- **Run each test 5+ times** — LLM output is non-deterministic even at low temperatures. Current observed pass rate is ~80% for Qwen3.5, ~90% for GPT-OSS, ~60% for Nemotron-Cascade.

## Isolation

Each test runs inside a `tempfile.TemporaryDirectory` to avoid polluting the working directory.
