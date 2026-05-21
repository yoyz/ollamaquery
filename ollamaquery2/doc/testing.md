# Testing Guide — ollamaquery2

## Test Suites

Two test files in `tests/`:

| File | Tests | Focus |
|------|-------|-------|
| `test_features.py` | ~95 | Feature coverage: config, retry, themes, inline processing, commands, HTML parsing, tokens, debug, images, utilities, errors |
| `test_modifications.py` | ~35 | Regression: retry utility, context reset, exception handling, streaming, stats, URL-open replacements |

## Running Tests

```bash
# Full suite
python3 -m unittest discover tests -v

# Single file
python3 -m unittest tests.test_features -v

# Single test class
python3 -m unittest tests.test_features.TestInlineProcessing -v

# Single test
python3 -m unittest tests.test_features.TestInlineProcessing.test_sanitize_shell_command -v
```

### Against a specific backend

```bash
# Ollama
OLLAMA_HOST=http://192.168.1.20:11434 python3 -m unittest discover tests -v

# Llama.cpp
LLAMACPP_HOST=http://127.0.0.1:8080 python3 -m unittest discover tests -v

# Custom model (default: auto-picked smallest model < 9 GB)
TEST_MODEL=qwen3:8b OLLAMA_HOST=http://192.168.1.20:11434 python3 -m unittest discover tests -v
```

## Backend Detection

Tests auto-detect available backends at import time:

1. HEAD request to `OLLAMA_HOST` (default `http://127.0.0.1:11434`)
2. HEAD request to `LLAMACPP_HOST` (default `http://127.0.0.1:8080`)
3. Tests decorated with `@ollama_only` / `@llamacpp_only` / `@any_backend` are skipped when the backend is unreachable

If Ollama is detected, the test suite automatically:
- Picks the smallest available model under 9 GB (excluding API proxies and embedding models)
- Loads it into memory via a minimal sync query
- Uses that model for all live-backend tests

## Test Coverage Matrix

### Unit Tests (no backend needed, always run)

| Area | Tests | What it checks |
|------|-------|----------------|
| **Configuration** | 5 | Default host/port values, prompt keys, max context size |
| **Command Registry** | 3 | Required commands exist, aliases resolved, categories correct |
| **Theme System** | 6 | Theme keys, colorize output, NO_COLOR env, preset names |
| **Retry Logic** | 5 | _request_with_retry: success, transient errors, 4xx no-retry, stderr message |
| **Shell Safety** | 6 | sanitize blocks `$()`, `;`, `&&`; validate rejects dangerous patterns |
| **Multiline Input** | 2 | Triple-quote detection, backslash continuation |
| **HTML Parsing** | 6 | HTMLStripper: text extraction, script skipping, word spacing |
| **Token Counting** | 4 | estimate_token_count: basic, empty, None; calculate_context_tokens |
| **Debug Manager** | 10 | Levels, categories, is_enabled, should_log, get_status |
| **Debug Log** | 3 | Log writes to stderr, suppressed when off, custom prefix |
| **Image Handling** | 3 | prepare_image_data: nonexistent, empty, valid PNG |
| **Image Command** | 2 | /image clear, /image no-arg |
| **Utility Functions** | 8 | is_known_command, format_help_text, sanitize_shell_command, validate_shell_command_safety |
| **Argument Parser** | 3 | Backend choices, model opt, mutual exclusion |
| **Error Handling** | 7 | fetch failures, token count with no model, image with bad path |
| **Format Help** | 3 | Compact and full help text contains categories |

### Integration Tests (need a live backend)

| Area | Tests | What it checks |
|------|-------|----------------|
| **Backend Detection** | 3 | check_backend_with_get/head, live model fetch |
| **Model Listing** | 3 | fetch_models returns list, model names exist |
| **Basic Queries** | 5 | model query creation, sync query returns dict, stream returns content, empty input handling |
| **Query Stats** | 2 | context_bar renders with %, handles zero window |
| **Command Handlers** | 12 | exit, help, clear, debug, thinking on/off, listmodel, stats, cwd, ls, switchmodel |
| **Switch Model** | 2 | messages preserved after switch, stats accumulated |
| **Full Workflow** | 1 | query → switch → query → clear → query |
| **Model Info** | 2 | show_model_info, show_model_details return valid JSON |
| **Token Count (API)** | 1 | get_message_token_count_ollama > 0 (skipped if `/api/tokenize` unavailable) |
| **Live Query** | 4 | stream returns content, stats accumulate, context tokens updated, chat history preserved |
| **Sync Query via main** | 1 | main() with -I -o writes output file |
| **Spawn Shell** | 3 | /spawnshell returns without error, output is not individual characters |
| **Dump Context** | 3 | /dumpcontext writes JSON file |

### Regression Tests (modifications file)

| Area | Tests | What it checks |
|------|-------|----------------|
| **Retry Utility** | 11 | Success, retries, 503 retry, 404 no-retry, ConnectionError, timeout passthrough, live fetch |
| **Reset Preferences** | 3 | reset() preserves force_no_thinking, debug_mode, system_prompt; wipes conversation |
| **Bare Exception** | 1 | No bare `except:` blocks |
| **CRemoved** | 2 | Dead `c()` function removed |
| **HTMLStripper** | 4 | Module-level class, parses HTML, skips script, strips tags |
| **Calculate Stats** | 4 | Side-effect-free, expected keys, no duplicate methods |
| **Show Model Flags** | 2 | --show and --show-details reachable |
| **URL open** | 1 | All urlopen calls go through _request_with_retry |
| **Clear Preferences** | 1 | /clear preserves preferences in live loop |
| **Switch Model** | 1 | /switchmodel preserves self.messages |
| **Shell Timeout** | 1 | Default shell_timeout is 5 |
| **AGENTS.md** | 1 | Line-number references roughly match file length |
| **Live Query** | 2 | Stream returns assistant response, stats accumulate |
| **Inline Commands** | 1 | execute_os_command with timeout |

## Release Checklist

Before a release, run the full suite against both backends:

```bash
# 1. Syntax check
python3 -m py_compile ollamaquery2.py

# 2. Unit tests (no backend needed)
python3 -m unittest tests.test_features.TestConfiguration \
  tests.test_features.TestCommandRegistry \
  tests.test_features.TestThemeSystem \
  tests.test_features.TestRetry \
  tests.test_features.TestInlineProcessing \
  tests.test_features.TestMultilineInput \
  tests.test_features.TestHTMLParsing \
  tests.test_features.TestTokenCounting \
  tests.test_features.TestDebugManager \
  tests.test_features.TestDebugLog \
  tests.test_features.TestImageHandling \
  tests.test_features.TestUtilityFunctions \
  tests.test_features.TestArgumentParser \
  tests.test_features.TestFormatHelp \
  tests.test_features.TestErrorHandling \
  tests.test_features.TestCommandHandlers \
  tests.test_features.TestSpawnShell \
  tests.test_features.TestDumpContext \
  tests.test_modifications.TestRetryUtility \
  tests.test_modifications.TestResetPreservesPreferences \
  tests.test_modifications.TestNoBareExcept \
  tests.test_modifications.TestCRemoved \
  tests.test_modifications.TestHTMLStripper \
  tests.test_modifications.TestCalculateStatsSideEffectFree \
  tests.test_modifications.TestShowModelFlags \
  tests.test_modifications.TestUrlopenReplaced \
  tests.test_modifications.TestShellTimeout \
  tests.test_modifications.TestAgentsMdConsistency \
  tests.test_modifications.TestProcessInlineCommands \
  -v

# 3a. Integration tests with Ollama
OLLAMA_HOST=http://192.168.1.20:11434 python3 -m unittest discover tests -v

# 3b. Integration tests with Llama.cpp
LLAMACPP_HOST=http://127.0.0.1:8080 python3 -m unittest discover tests -v

# 4. Check no unexpected failures
# Expected skips (Ollama): tokenizer test if /api/tokenize unavailable
# Expected skips (llama.cpp): Ollama-specific tests (switchmodel, model info, etc.)
```

## Interpreting Results

- **OK (skipped=N)** — All tests passed, N skipped because backend unavailable
- **FAIL** — A test failed. Check the assertion and traceback. Common causes:
  - Backend unreachable or wrong URL
  - Model not found on server
  - API response format changed
- **ERROR** — A test raised an unexpected exception. Check for import errors, missing attributes.

### Typical skip counts

| Scenario | Skipped | Notes |
|----------|---------|-------|
| No backend | 32 | All integration tests skipped |
| Ollama only | 1 | Only tokenizer test if unavailable |
| Llama.cpp only | 27 | All Ollama-specific tests skipped |
