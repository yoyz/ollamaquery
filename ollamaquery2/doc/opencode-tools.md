# OpenCode Tool/Prompt Architecture — Research Notes

Source: https://github.com/anomalyco/opencode (dev branch, ~160K stars)

## Core Difference From ollamaquery

**OpenCode** passes tool schemas via the native OpenAI `tools` API parameter (https://platform.openai.com/docs/guides/function-calling), which is a **separate field in the request body**, not part of the system prompt text.

**ollamaquery** embeds tool definitions directly in the system prompt text via `ToolRegistry.get_system_prompt_block()`, bloating the prompt by several KB.

## Prompt Architecture

### Per-Model Prompt Files

OpenCode maintains separate `.txt` prompt files for each model family, selected based on the provider/model being used:

https://github.com/anomalyco/opencode/tree/dev/packages/opencode/src/session/prompt

| File | Model Family |
|------|-------------|
| `default.txt` | Fallback for unknown models |
| `anthropic.txt` | Claude models |
| `gpt.txt` | OpenAI GPT models |
| `gemini.txt` | Google Gemini models |
| `codex.txt` | GPT Codex models |
| `beast.txt` | "Beast" models |
| `trinity.txt` | Trinity models |
| `kimi.txt` | Kimi models |
| `copilot-gpt-5.txt` | GitHub Copilot GPT-5 |
| `plan-mode.txt` | Plan mode instructions |
| `plan.txt` | Plan agent prompt |
| `plan-reminder-anthropic.txt` | Plan mode reminder for Claude |
| `max-steps.txt` | Instructions when max steps reached |
| `build-switch.txt` | Instructions when switching to build mode |

Prompt selection logic is in:

https://github.com/anomalyco/opencode/blob/dev/packages/opencode/src/session/prompt.ts

(huge file, ~70KB — contains prompt assembly logic)

### Example: `default.txt` (~8.5KB, 130 lines)

https://raw.githubusercontent.com/anomalyco/opencode/dev/packages/opencode/src/session/prompt/default.txt

Contains only:
- Tone and style instructions
- Response format rules
- Code conventions
- Security guidelines
- Task execution workflow

**No tool definitions embedded.** Tools are injected separately via the API's `tools` parameter.

## Agent System

### Agent Definitions

https://github.com/anomalyco/opencode/blob/dev/packages/opencode/src/agent/agent.ts

Defines built-in agents (`build`, `plan`, `general`, `explore`, `compaction`, `title`, `summary`) with:
- `description`: When to use this agent
- `permission`: Tool access rules (allow/ask/deny)
- `mode`: primary or subagent
- `prompt`: Optional custom prompt file
- `temperature`, `topP`: Sampling parameters
- `steps`: Max agentic iterations
- `model`: Optional model override

User-defined agents can be created via:

https://opencode.ai/docs/agents

### Tool Definition System

https://github.com/anomalyco/opencode/blob/dev/packages/llm/src/tool.ts

Type-safe tool definitions with:
- `description`: What the tool does
- `parameters`: Effect Schema for parameter validation
- `success`: Effect Schema for return value validation
- `execute`: Optional handler function

Tools are converted to `ToolDefinition` objects and passed via the `tools` API parameter:

https://github.com/anomalyco/opencode/blob/dev/packages/llm/src/tool-runtime.ts

### Tool Schema Definitions

https://github.com/anomalyco/opencode/blob/dev/packages/llm/src/schema/options.ts

### Provider-Specific Format Handling

Each provider can convert tool definitions to its own format:

https://github.com/anomalyco/opencode/tree/dev/packages/llm/src/providers

| Provider | File |
|----------|------|
| OpenAI | https://github.com/anomalyco/opencode/blob/dev/packages/llm/src/providers/openai.ts |
| OpenAI Compatible | https://github.com/anomalyco/opencode/blob/dev/packages/llm/src/providers/openai-compatible.ts |
| Anthropic | https://github.com/anomalyco/opencode/blob/dev/packages/llm/src/providers/anthropic.ts |
| Google | https://github.com/anomalyco/opencode/blob/dev/packages/llm/src/providers/google.ts |
| xAI | https://github.com/anomalyco/opencode/blob/dev/packages/llm/src/providers/xai.ts |
| GitHub Copilot | https://github.com/anomalyco/opencode/blob/dev/packages/llm/src/providers/github-copilot.ts |

### Model-Specific Tool Call Parsers

GLM-4.7 has a dedicated tool call parser (`--tool-call-parser glm47`):

https://huggingface.co/unsloth/GLM-4.7

OpenCode supports per-model tool call parsers (similar to vLLM/SGLang):

https://github.com/anomalyco/opencode/tree/dev/packages/llm/src

### Provider Model Schema

https://github.com/anomalyco/opencode/blob/dev/packages/llm/src/schema/messages.ts

Contains message types, tool call part structures, and provider options schemas.

## Key Takeaways

1. **Native `tools` API is preferred** — Models that support OpenAI-style `tools` parameter (Claude, GPT, Gemini, GLM-4.7, etc.) get tool schemas via the API, keeping the system prompt short and clean.

2. **JSON-in-text is the fallback** — Model families that don't support the native API (like Qwen3.5, Nemotron) would need inline tool descriptions. However, OpenCode's primary focus is API-callable models.

3. **Per-model prompts** — Each model family gets a tailored system prompt because different models respond differently to tone, structure, and formatting instructions.

4. **Separation of concerns** — System prompt = behavior/rules, Tools API = capabilities, Conversation history = context. These three are kept independent.

5. **GLM-4.7-Flash likely works with native `tools`** — It has `--tool-call-parser glm47` in vLLM, meaning it can parse OpenAI-style tool calls out of the box. The failure was caused by our inline tool descriptions overwhelming it.

6. **Gemma4-E4B may also support native tools** — As a Google model, it likely uses the Gemini-style `tools` API format.

## OpenCode Docs

- https://opencode.ai/docs/agents — Agent configuration
- https://opencode.ai/docs/models — Model configuration
- https://opencode.ai/docs/tools — Tool configuration
- https://opencode.ai/docs/permissions — Permission system
- https://github.com/anomalyco/opencode/blob/dev/AGENTS.md — Code style guide for contributing
