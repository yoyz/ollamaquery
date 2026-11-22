# Self-AI review — have a local LLM audit a diff (read-only)

Status: workflow used for the first time on 2026-11-22 (0.5.0 bwrap + presets).
It is a **review aid**, not an oracle: the model reads the code and produces a
prioritized opinion, but every P0/P1 claim must be verified against the source
before acting on it.

## 1. What it is

Run the repository's own agentic loop (`ChatLoop.run_agentic_query`) against a
local backend, in the **`read` preset**, with a compact bundle of the changes.
The model can list/read files, grep, fetch documentation and inspect git, but it
**cannot write and has no shell**. It returns a structured review (P0/P1/P2,
test gaps, top fixes).

Why the `read` preset and not `work`/`exec`:

| Preset    | What it allows                                          | Fit for a code review                                        |
| --------- | ------------------------------------------------------- | ------------------------------------------------------------ |
| `read`    | inspection tools, no shell, CWD-confined                | **default**: no write, no shell, no accidental mutation      |
| `exec`    | read + `run_command`/`run_python`, project ro, net none | runtime checks (reproduce a bug) without writes              |
| `work`    | every tool, bwrap project rw                            | only if the review must run tests/builds; instruct read-only |
| `trusted` | host, auto-confirm                                      | never for an unreviewed model                                |

`read` still allows `fetch_url`, so the model can consult upstream docs (it did
this for bwrap's man page in the 0.5.0 review). That is intentional: it turns
"the model guesses a flag" into "the model checks the flag".

## 2. Prerequisites

| Item    | Requirement                                                                              |
| ------- | ---------------------------------------------------------------------------------------- |
| Backend | Any backend the harness supports; a long-context, reasoning model is best                |
| Model   | A loaded model on the backend (Strata example: `qwen3.8-flash-next-unsloth-ud-iq4_xs`)   |
| Context | The model's real window; 512K is comfortable for a full diff + sources                   |
| Budget  | 128K output tokens (`max_output_tokens`); enough for long thinking plus a 4K-word report |
| Process | Run it in the background: a full review can take 30–60 min                               |

The workflow does **not** need `bwrap`, a container runtime, or a TTY. It does
need the repository's normal test/venv setup (`ollamaquery2.py` imports only the
stdlib).

## 3. Step 1 — freeze a review bundle

The bundle is the snapshot the model reviews. Keeping it in the project root is
the simplest option: the strict `read` preset confines file tools to the CWD, so
a bundle under `/tmp` would be denied unless an ACL read grant is added.

```python
#!/usr/bin/env python3
"""Build review_bundle.md from the working-tree changes (run from the repo root)."""
import pathlib
import subprocess

ROOT = pathlib.Path.cwd()


def run(*args: str) -> str:
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True).stdout


parts = [
    "# Review bundle\n",
    "Uncommitted changes on top of the current HEAD.\n",
    "## git diff --stat\n```\n" + run("git", "diff", "--stat") + "```\n",
]
for name, cmd in [
    ("ollamaquery2.py", ["git", "diff", "-U8", "--", "ollamaquery2.py"]),
    ("tests/test_agentic.py", ["git", "diff", "-U6", "--", "tests/test_agentic.py"]),
    ("tests/test_features.py", ["git", "diff", "-U6", "--", "tests/test_features.py"]),
    ("README.md", ["git", "diff", "-U4", "--", "README.md"]),
]:
    parts.append(f"## diff: {name}\n```diff\n" + run(*cmd) + "\n```\n")

# New (untracked) files are not in `git diff`: append them explicitly.
for path in ["doc/agentic_bwrap.md"]:
    parts.append(f"## new file: {path}\n```markdown\n"
                 + (ROOT / path).read_text() + "\n```\n")

(ROOT / "review_bundle.md").write_text("\n".join(parts))
print("wrote review_bundle.md")
```

Adapt the file list to the change. Two rules:

- **Bundle the diff, not the whole 24k-line file.** The model will read source
  files for context when it needs them; a 145 KB bundle is already ~40K tokens.
- **Freeze the tree while it reviews**, or tell it to use the bundle only. The
  same reason as the test-suite rule: editing `ollamaquery2.py` while the model
  reads it makes line numbers and behavior move under it.

Remove the bundle before committing; it is a temporary artifact and must not
end up in the release.

## 4. Step 2 — run the reviewer

Copy this template to `/tmp/opencode/self_review.py` (outside the repo) and run
it from the repository root.

```python
#!/usr/bin/env python3
"""Read-only self-review of review_bundle.md via the harness's own agentic loop."""
import os
import sys

sys.path.insert(0, os.getcwd())
import ollamaquery2 as q

BUNDLE = "review_bundle.md"
BASE_URL = os.environ.get("REVIEW_BASE_URL", "http://192.168.1.20:8090")
BACKEND = os.environ.get("REVIEW_BACKEND", "strata")
MODEL = os.environ.get("REVIEW_MODEL", "qwen3.8-flash-next-unsloth-ud-iq4_xs")

ctx = q.CommandContext()
ctx.base_url = BASE_URL
ctx.backend = BACKEND
ctx.model = MODEL

# Budget: 128K output, 512K context. A review step can think for 5-10 minutes,
# so the default 600 s step timeout is too short.
ctx.context_window_size = 512 * 1024
ctx.max_output_tokens = 128 * 1024
ctx.agentic_step_timeout = 1800
ctx.agentic_timeout_max = 3600
ctx.agentic_max_iterations = 40

# A clean state: the singleton keeps whatever the previous session used.
ctx.system_prompt = "You are a rigorous software security and architecture reviewer."
ctx._saved_system_prompt = None
ctx._plan_prev_prompt = None
ctx.agentic_mode = False
ctx.plan_mode = False
ctx.plan_run = False
ctx.preset = "work"
ctx.executor_mode = "host"
ctx.sandbox_engine_override = None
ctx.auto_confirm = False
ctx.bwrap_project = "rw"
ctx.bwrap_net = "shared"

loop = q.ChatLoop(ctx)
loop.run_handle_agentic("/agentic preset read")
print(f"PRESET={ctx.preset} MODE={ctx.executor_mode} "
      f"BUDGET={ctx.max_output_tokens} CONTEXT={ctx.context_window_size}",
      file=sys.stderr)

PROMPT = f"""You are reviewing an uncommitted change. Read `{BUNDLE}` first
(use read_file with offset if it pages), then read source files directly when
you need context.

The change: <one paragraph: what was added, which invariants it claims, which
scope is deliberately deferred>.

Perform a critical security/architecture review. Focus on:
- fail-open paths (sandbox/probe failure, promise bypass, symlinks, bind order);
- drift between prompt, tool surface, approval and mounts;
- state transitions (presets, aliases, CLI overrides, on/off/plan/sandbox);
- documentation honesty and test gaps;
- UX/API problems (defaults, naming, refusal messages).

Rules:
- Read-only: do not modify, create, or delete files. Do not use write tools.
- Cite concrete function/class names or diff hunks. If unsure, say what you
  could not verify.
- Deferred scope is not a bug; a promise the code cannot keep is.
- Structure: verdict (3 lines); P0; P1; P2/nits; test gaps; top-3 changes.
- Tag findings [verified from code] or [needs runtime check].

You have a large budget; use it to verify, not to restate the code.
Aim for a 2,500-4,000 word review.
"""

loop.run_agentic_query(PROMPT)
print("REVIEW_DONE", file=sys.stderr)
```

Run it in the background so a harness timeout cannot kill it:

```bash
cd /path/to/ollamaquery2
nohup python3 /tmp/opencode/self_review.py \
  > /tmp/opencode/self_review.log 2>&1 &
tail -f /tmp/opencode/self_review.log
```

Or, from the OpenCode shell tool, use `background: true`; it returns a shell ID
and notifies on completion.

## 5. Step 3 — read the log correctly

The log shows the agentic loop, not the model's hidden reasoning:

| Line                                               | Meaning                                                                    |
| -------------------------------------------------- | -------------------------------------------------------------------------- |
| `[Agentic] Step N/40…`                             | iteration and progress marks (`Zz` = waiting, `.` = bytes received)        |
| `--- Stats: … think N tok … Context: N tokens ---` | per-step accounting; rising `think` = the model is reasoning               |
| `[Tool] name(args) → OK/ERROR`                     | what it inspected; `ERROR` is usually a model path mistake, not a code bug |
| `REVIEW_DONE`                                      | the final answer was written to stdout (above the marker)                  |

If it reaches `Step 40/40` without a final answer, the finalizer still asks for
a synthesis with the accumulated context; the report will be shorter but usable.
A review that spends 35+ steps reading is not stuck — check that `Context` grows
and that the tool names change. If `Context` stops growing for several steps,
the model is likely thinking, not hung.

## 6. Budget and timeouts (measured)

| Parameter                | Value                      | Why                                                                   |
| ------------------------ | -------------------------- | --------------------------------------------------------------------- |
| `context_window_size`    | `512 * 1024`               | full diff + sources + fetched docs; Strata's window is 524 288        |
| `max_output_tokens`      | `128 * 1024`               | the earlier 16K run capped thinking at 12 288 and kept wrapping up    |
| `agentic_step_timeout`   | `1800`                     | a single audit step was measured at 4–6 min (17.6K think tokens once) |
| `agentic_timeout_max`    | `3600`                     | escalation ceiling for a slow step                                    |
| `agentic_max_iterations` | `40` (60 for a deep audit) | the 0.5.0 review used 35 steps in ~43 min before the final answer     |

A first pass on 0.5.0 (bwrap + presets, ~1 500 changed lines) was still at
step 35/40 after ~43 min, ~79K context, 35+ tool calls, with no timeout and no
context pressure. That is the expected order of magnitude: budget one hour and a
long-context model.

## 7. Safety and pitfalls

- **Read-only is enforced by the preset.** `read` has no shell and no write
  tools, and its paths are CWD-confined (`strict_path_confinement`). Do not use
  `work` just to let it run `git diff`: bundle the diff instead.
- **`fetch_url` is still network access.** In `read`, the model can fetch
  upstream docs. That is useful, but it is not a hermetic review; use `exec`
  (`net none`) if the review must be offline, and give it the docs inline.
- **The model may misread line numbers, flags or APIs.** Verify every P0/P1
  against the current source. A fetched man page helps, but it is not the code.
- **Do not edit the tree during the run** (same rule as the test suite). If you
  must, tell the model to rely on the frozen bundle only.
- **The bundle is untracked and temporary.** Delete `review_bundle.md` before
  `git add`/`git commit`; `git status --short` must show only intended files.
- **A long run can hit the iteration cap.** The finalizer produces a synthesis,
  but the richest analysis is lost; for a deep audit raise the cap to 60 and
  keep the budget at 128K/512K.
- **The report is input, not a verdict.** The usual rule applies: reproduce,
  then fix; a P0 that cannot be reproduced is a question, not a bug.

## 8. After the report

1. Read the verdict and the P0 list.
2. For each P0/P1, locate the cited function and decide: confirmed, false
   positive, or needs a runtime check.
3. Reproduce the confirmed ones with a focused test before changing code; add
   the regression test in the same commit.
4. Keep the model's P2/nits as backlog, not as release blockers.
5. Record the review log path and the confirmed findings in the commit notes
   (the model's report itself stays out of the commit).
