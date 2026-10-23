# `git_inspect` — read-only git inspection for `/agentic plan`

Status: implemented (`ollamaquery2.py`, v0.3.4)
Scope: read-only repository inspection as a first-class tool, no shell involved

---

## 1. Goal

Strict `/agentic plan` (`PLAN_MODE_TOOLS`) has no `run_command`, and the `diff`
tool only compares two files. The first questions a planner asks about a
repository were therefore unanswerable without dropping to `/agentic plan run`
and confirming a shell command per invocation:

- *What is dirty right now?*
- *What changed since the last commit?*
- *What does the recent history look like?*

`git_inspect` answers exactly those, read-only, with no shell and no
model-supplied git arguments.

## 2. Actions

| Action          | Command                                          | Answers                                      |
| --------------- | ------------------------------------------------ | -------------------------------------------- |
| `status`        | `git status --short --branch -- <scope>`         | what is dirty / untracked, current branch    |
| `diff`          | `git diff --no-ext-diff --no-textconv …`         | unstaged changes                             |
| `diff_staged`   | `git diff --cached …`                            | staged changes                               |
| `diff_head`     | `git diff HEAD …`                                | **everything since the last commit**         |
| `log`           | `git log -n<limit> --format='%h %s (%an, %ar)'`  | recent commits                               |
| `show`          | `git show --stat --patch <ref> …`                | one commit: message, diffstat, patch         |

`diff_head` exists because `diff` alone hides staged changes: "what changed
since the last commit" needs working tree **and** index against `HEAD`.

Arguments:

| Arg        | Type      | Notes                                                       |
| ---------- | --------- | ----------------------------------------------------------- |
| `action`   | enum      | default `status`; anything else is rejected                 |
| `limit`    | integer   | `log` only, clamped to 1..100 (default 20)                  |
| `ref`      | string    | `show` only, default `HEAD`, regex-validated                |
| `path`     | string    | optional pathspec, relative and inside the CWD              |

## 3. Why a tool instead of `run_command`

`/agentic plan run` can already execute `git status` — `git *` is
`DEFAULT_SHELL_PERMISSION`-allow. But that tier:

- confirms **every** command (bypass-immune to `auto_confirm`),
- lifts the CWD confinement, and
- hands the model a generic shell.

`git_inspect` is the strict-tier answer: no confirmation, no shell, no argv
constructed by string concatenation, CWD-scoped. `DEFAULT_SHELL_PERMISSION` is
unchanged; the tool is additive.

## 4. Security posture

Two separate problems, both closed.

### 4.1 Argument injection

The command is a fixed argv list executed with `shell=False`:

```
git -c core.pager=cat -c core.fsmonitor=false -c core.quotepath=false \
    --no-optional-locks <verb> [flags] [--] [path]
```

No part of the model's input is ever concatenated into a command string. This
differs from `kubernetes_cluster_query`, which builds a shell string and
therefore needs `shlex.quote` on every operand. `ref` is additionally
regex-validated (`^[A-Za-z0-9_][A-Za-z0-9_./~^@{}-]{0,200}$`) so it can never
be parsed as an option — `--output=/tmp/x`, `-c`, `--exec=id` are all rejected
before git runs. `path` is rejected when absolute, when it starts with `-`, when
it contains a `..` component, or when its realpath escapes the CWD.

### 4.2 Repo config is executable

A repository can carry configuration that makes git run a program. Both vectors
were reproduced in a throwaway repo:

| Config                                        | Runs on              | Closed by                          |
| --------------------------------------------- | -------------------- | ---------------------------------- |
| `diff.<driver>.textconv` (+ `.gitattributes`) | `git diff`           | `--no-textconv`                    |
| `core.fsmonitor`                              | `git status`         | `-c core.fsmonitor=false`          |
| `diff.external` / `GIT_EXTERNAL_DIFF`         | `git diff`           | `--no-ext-diff`                    |

Without these flags a tool advertised as "read-only, no shell" is still an
arbitrary-code-execution primitive driven by repository contents. Verified
before/after: a `.gitattributes`-declared textconv script and a `core.fsmonitor`
hook both ran under plain `git diff` / `git status`, and neither ran through
`git_inspect`.

### 4.3 State

- `--no-optional-locks` + `GIT_OPTIONAL_LOCKS=0`: no `.git/index.lock` is taken
  (plain `git status` opportunistically refreshes the index).
- `GIT_TERMINAL_PROMPT=0`: a credential prompt cannot block the loop.
- `GIT_PAGER=cat` / `core.pager=cat`: no pager can hijack the pipe.
- Inherited location variables are **stripped** from the child environment:
  `GIT_DIR`, `GIT_WORK_TREE`, `GIT_COMMON_DIR`, `GIT_INDEX_FILE`,
  `GIT_OBJECT_DIRECTORY`, `GIT_ALTERNATE_OBJECT_DIRECTORIES`, `GIT_NAMESPACE`,
  `GIT_ASKPASS` (`GIT_INSPECT_STRIP_ENV`). `GIT_DIR` + `GIT_WORK_TREE` is a
  complete repository redirect, so without this a parent environment could make
  strict plan mode inspect a repository outside the CWD — a confinement break,
  not just a wrong result. Covered by
  `test_inherited_git_dir_is_ignored` (two repos, one exported via `GIT_DIR`).
- The tool is **not** in `DESTRUCTIVE_TOOLS`, so no confirmation is requested —
  which is only defensible because of the points above.

## 5. Output bounds

`subprocess.run(stdout=PIPE)` would buffer the entire output before any cap could
apply, so a generated multi-hundred-MB blob in an arbitrary repository would be
held in RAM. `_git_capture()` instead reads stdout in `GIT_INSPECT_READ_CHUNK`
slices up to `GIT_INSPECT_MAX_OUTPUT_CHARS` (2 MB), kills the child as soon as
the ceiling is hit, and reports the truncation to the model. stderr goes to a
temp file rather than a second pipe (a full pipe would block the child and
deadlock the read), and a watchdog kills the child at `GIT_INSPECT_TIMEOUT` so a
hung `read()` cannot wedge the ReAct loop.

Results above `CLUSTER_SPILL_THRESHOLD` (3500 chars) then spill to
`~/.ollamaquery.d/spill/git_result_<action>.txt`, and the model receives a pure
pointer (size + `read_file` path), never the payload inline. `read_file` pages it
via its `next` offset, and the session Path-ACL read-allow rule for the spill dir
is registered **before** the path is handed over (strict plan mode confines every
path to the CWD, so an unregistered path would be unreadable).

On this repository a single `diff_head` is ~26 KB and `show HEAD` ~95 KB, so both
spill — the pointer is the normal path, not the exception.

## 6. Plan-mode integration

`git_inspect` is in `PLAN_MODE_TOOLS` (and therefore `PLAN_MODE_RUN_TOOLS`), and
is named in both prompt blocks:

- `AGENTIC_PLAN_BLOCK` (strict) lists it among the inspection tools.
- `AGENTIC_PLAN_RUN_BLOCK` tells the model to prefer it over a shell `git`
  command, since it needs no confirmation and cannot mutate.

Enabled at all three layers, as with every plan-mode restriction: the prompt
block, the native `tools` schema (`_plan_allowed_tools()`), and the `execute()`
backstop.

## 7. Out of scope

- No mutating verbs (no `add`, `commit`, `checkout`, `stash`, `restore`,
  `reset`). Mutation stays in full agentic mode via `run_command` +
  confirmation.
- No network operations (`fetch`, `pull`, `push`, `clone`).
- No arbitrary git flags, and no `-c` passthrough.
- No blame/archaeology actions; `log` + `show` cover orientation, and deeper
  history work belongs in full agentic mode.
