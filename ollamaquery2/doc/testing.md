# Testing guide — ollamaquery2

The canonical command list, per-file matrix and contributor rules live in
[`tests/README.md`](../tests/README.md). This page covers the coverage map, the
release checklist and result interpretation.

## Quick commands

Run from `ollamaquery2/` (not the repo root, not `tests/`):

```bash
# Everything (backend-gated tests skip when no server answers)
python3 -m unittest discover tests -v

# One file
python3 -m unittest tests.test_acl -v

# One class
python3 -m unittest tests.test_acl.TestShellApprovalPrompt -v

# One test
python3 -m unittest tests.test_acl.TestShellApprovalPrompt.test_prompt_label_does_not_claim_always -v
```

Offline suites (no backend, no network — what CI runs):

```bash
python3 -m unittest -v \
  tests.test_acl \
  tests.test_mcp \
  tests.test_compaction \
  tests.test_compact_command \
  tests.test_agentic_timeout \
  tests.test_tool_summarize \
  tests.test_state_files
```

## Backend detection

Test modules that need a live model probe at import time, in this order:

1. `OLLAMA_HOST` (default `http://127.0.0.1:11434`), then
2. `LLAMACPP_HOST` (default `http://127.0.0.1:8080`), then
3. addresses returned by `gethostbyname_ex(gethostname())` on the standard ports.

`test_vision` uses `VISION_MODEL` when set; otherwise it picks the first model
whose name contains `vl`, `vision`, `llava`, `gemma3` or `glm4v`.

Backend-dependent tests are decorated with `@ollama_only` / `@llamacpp_only` /
`@any_backend` and skip when the backend is unreachable. A skipped live test is
not a failure.

When Ollama is detected the suite picks the smallest model under 9 GB
(excluding API proxies and embedding models) and loads it with a minimal sync
query, unless `TEST_MODEL` names one explicitly.

## Coverage map

| Area                                    | Where                                                                      |
| --------------------------------------- | -------------------------------------------------------------------------- |
| Config, themes, commands, help          | `test_features.py`                                                         |
| Retry, URL fetch, token counting        | `test_features.py`, `test_modifications.py`                                |
| Shell gate, path ACL, fetch schemes     | `test_acl.py`                                                              |
| State-file privacy (`0700`/`0600`)      | `test_state_files.py`                                                      |
| Session save/resume, `/drop`            | `test_features.py`, `test_modifications.py` (thinner than the rest)        |
| Compaction (mechanical + LLM)           | `test_compaction.py`, `test_compact_command.py`                            |
| Agentic ReAct, tool parsing, timeouts   | `test_agentic.py`, `test_agentic_timeout.py`                               |
| Tool-result summarization               | `test_tool_summarize.py`                                                   |
| MCP client, plan policy, spill          | `test_mcp.py`                                                              |
| Backend dialects                        | `test_features.py`, `test_strata.py`, `test_lmstudio.py`, `test_vision.py` |
| Release checks                          | `test_release.py`                                                          |

Known thin spots: session save/resume and `/drop` have direct coverage but few
cases; the shell tokenizer is mostly exercised through `check_shell_approval`
rather than directly.

## Release checklist

```bash
# 1. Syntax
python3 -m py_compile ollamaquery2.py tests/*.py

# 2. Offline suites (must be green with no backend)
python3 -m unittest -v \
  tests.test_acl \
  tests.test_mcp \
  tests.test_compaction \
  tests.test_compact_command \
  tests.test_agentic_timeout \
  tests.test_tool_summarize \
  tests.test_state_files

# 3. Full suite against each backend you support
OLLAMA_HOST=http://127.0.0.1:11434   python3 -m unittest discover tests -v
LLAMACPP_HOST=http://127.0.0.1:8080  python3 -m unittest discover tests -v
STRATA_HOST=http://127.0.0.1:8080    python3 -m unittest discover tests -v

# 4. Inspect the log; a clean run is OK with skips, not FAIL/ERROR
```

Redirect long runs to a file (per the test-output rule in `AGENTS.md`), then
grep the file:

```bash
python3 -m unittest discover tests -v > /tmp/opencode/full_suite.log 2>&1
grep -E "^(Ran|OK|FAILED|ERROR)" /tmp/opencode/full_suite.log
```

## Interpreting results

| Output                   | Meaning                                                              |
| ------------------------ | -------------------------------------------------------------------- |
| `OK`                     | Every executed test passed                                           |
| `OK (skipped=N)`         | Passed; N tests skipped because no backend/feature was available     |
| `FAIL`                   | Assertion failed — inspect the first failure, not the last           |
| `ERROR`                  | Unexpected exception (import error, missing attribute, IO error)     |

Common causes of skips and errors:

- Backend not running, wrong URL, or wrong model name.
- `readline`/`pty` unavailable for terminal tests.
- `convert` (ImageMagick) missing for the vision resize path.
- No `gcc` for the agentic write-compile-run end-to-end tests.
- Writes to the real `~/.ollamaquery.d` — tests that care patch the module
  globals; if you add one, do the same.

## CI

`.github/workflows/tests.yml` runs the seven offline suites on Python 3.10 and
3.12 with an isolated `HOME`. It intentionally does not use `discover`, because
importing the backend-gated modules probes localhost/LAN and the free runners
have no LLM. See `tests/README.md` for the contributor rules that keep the
offline half offline.
