# Tests

The suite is `unittest`-based, has no third-party dependencies, and needs no
network for the offline half.

## Run everything

**Run from this directory's parent** (`ollamaquery2/`), not from `tests/` and
not from the repository root:

```bash
cd ollamaquery2
python3 -m unittest discover tests -v
```

The test modules insert the parent directory into `sys.path` so `import
ollamaquery2` works, and a few ACL/permission tests use the process working
directory as the path under test.

## Run one file / class / test

```bash
python3 -m unittest tests.test_acl -v
python3 -m unittest tests.test_acl.TestShellApprovalPrompt -v
python3 -m unittest tests.test_acl.TestShellApprovalPrompt.test_prompt_label_does_not_claim_always -v
```

## Offline vs backend-gated

Offline suites never open a socket, spawn an LLM, or download anything. They are
what CI runs (`.github/workflows/tests.yml`).

| File                        | Tests   | Offline   | Covers                                                                             |
| --------------------------- | ------: | :-------: | ---------------------------------------------------------------------------------- |
| `test_acl.py`               | 47      | yes       | Path ACL defaults/overrides, shell gate, `fetch_url` scheme guard, approval prompt |
| `test_mcp.py`               | 46      | yes       | Scripted stdio MCP server, config loading, plan-mode eligibility, spill            |
| `test_compaction.py`        | 41      | yes       | Compaction window, tool-call pairing, LLM conversation summary                     |
| `test_compact_command.py`   | 18      | yes       | `/compact` dispatcher (status/now/force/llm/threshold/list)                        |
| `test_agentic_timeout.py`   | 43      | yes       | Timeout escalation states, nudges, ceiling doubling                                |
| `test_tool_summarize.py`    | 8       | yes       | LLM tool-result summarization                                                      |
| `test_state_files.py`       | 10      | yes       | `0700` dirs / `0600` files for sessions, history, logs, spill                      |
| `test_features.py`          | 219     | no        | Broad feature sweep; probes localhost/LAN at import                                |
| `test_modifications.py`     | 195     | no        | Regressions: retry, tokens, `/maxtokens`, context bar, thinking                    |
| `test_agentic.py`           | 184     | no        | ReAct loop, tools, JSON parsing, write-compile-run end-to-end                      |
| `test_release.py`           | 47      | no        | Release checks (backend-gated parts skip)                                          |
| `test_strata.py`            | 19      | no        | Strata detection/parsing; live tests skip when absent                              |
| `test_lmstudio.py`          | 15      | no        | LM Studio dialect                                                                  |
| `test_vision.py`            | 8       | no        | Image queries; downloads test images, needs a vision model                         |
| **Total**                   | **900** |           | 14 files                                                                           |

`yes` means safe to run on a machine with no LLM and no internet. `no` means the
module probes for a backend at import (fast, localhost/LAN) and most of its
tests skip when none is reachable.

### The CI command

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

Do **not** use `discover` in CI: importing `test_features`/`test_modifications`
scans `gethostbyname_ex(hostname)` addresses, and `test_vision` downloads images.

## Backend selection

| Variable          | Default                       | Used by                                    |
| ----------------- | ----------------------------- | ------------------------------------------ |
| `OLLAMA_HOST`     | `http://127.0.0.1:11434`      | Ollama suites (and `test_agentic` probe)   |
| `LLAMACPP_HOST`   | `http://127.0.0.1:8080`       | llama.cpp suites                           |
| `STRATA_HOST`     | `http://127.0.0.1:8080`       | `test_strata`                              |
| `TEST_MODEL`      | smallest model under 9 GB     | Ollama live tests                          |
| `VISION_MODEL`    | first vision model found      | `test_vision`                              |

```bash
OLLAMA_HOST=http://192.168.1.20:11434 TEST_MODEL=qwen3.5:9b \
  python3 -m unittest tests.test_features -v
```

When no backend answers, live tests skip with a clear message; they do not fail.
Skip counts depend on which backends are reachable.

## Writing tests

- **Keep offline suites offline.** A new test in the seven offline files must
  not open a socket or launch an LLM. Use `unittest.mock` for transports.
- **Fixtures go in temp dirs.** Write configs, sessions and spill files under
  `tempfile.TemporaryDirectory()`, and patch module globals
  (`SESSION_DIR`, `SESSION_INDEX`, `CLUSTER_SPILL_DIR`) instead of writing to
  the real home directory.
- **Never depend on untracked files.** `AGENTS.md` is local-only; tests must
  skip (not fail) when an optional local file is absent. `README.md` may be used
  as a file-presence fixture only — do not assert on its content or line
  numbers.
- **Reset the `CommandContext` singleton** when a test touches it, or use a
  `SimpleNamespace` when only a few attributes are needed.
- **Redirect long output to a file**, per `AGENTS.md`:

  ```bash
  python3 -m unittest tests.test_acl -v > /tmp/opencode/test_acl.log 2>&1
  ```

See `doc/testing.md` for the release checklist and the coverage map.
