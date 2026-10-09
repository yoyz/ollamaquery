# `/agentic plan` — read-only run_command policy (design draft)

> **Superseded (2026-10-08).** The read-only-command allowlist below was judged
> too complex to maintain. The implemented direction is instead two tiers, with
> no per-command allowlist:
> - `/agentic plan` (strict): query tools only (`read_file`, `glob`, `grep`,
>   `list_directory`, `fetch_url`, `diff`, `kubernetes_cluster_query`), no
>   shell, and all paths hard-confined to CWD (outside requests are refused,
>   not prompted).
> - `/agentic plan run`: re-adds `run_command` (always confirms) for shell-only
>   inspection; CWD confinement lifted.
>
> See `PLAN_MODE_TOOLS` / `PLAN_MODE_RUN_TOOLS`, `plan_active_tools()`, the
> `_path_acl_decision` strict branch, and `_agentic_toggle_plan`.
> Everything below is kept for historical context.

Status: **discussion draft — no code merged.** This proposes letting verified
read-only `run_command` invocations run inside `/agentic plan` **without a
confirmation prompt**, with `git` as the flagship use case.

---

## 1. Current behavior

`/agentic plan` (read-only planner mode, v0.2.11, `ollamaquery2.py:8203`)
restricts the tool surface to `PLAN_MODE_TOOLS` and makes `run_command`
confirm **every** time via `_confirm()` (`ollamaquery2.py:5160`), bypass-immune
to `auto_confirm`.

```python
# _confirm, plan branch (conceptual)
if plan_mode:
    return self._prompt_confirm(tool_name, args)   # ALWAYS for run_command
```

There is currently **one** exemption, added recently: a harmless pure-GET
`curl`/`wget` (single http(s) URL, no body/method/output/pipes) is intercepted
in `ToolRegistry.execute` *before* `_confirm` and served by the `fetch_url`
builtin (`_is_harmless_get_fetch`, `ollamaquery2.py:5405`), with a steering note
telling the model to use `fetch_url` directly.

Crucially, the **shell approval gate underneath already auto-allows** most
inspection commands. `DEFAULT_SHELL_PERMISSION` (`ollamaquery2.py:2472`) grants
`git *`/`ls *`/`cat *`/`grep *`/`head *`/`tail *`/`find *`/… an `allow` verdict,
and `_shell_acl_scan` re-gates operands that hit a Path-ACL rule (`/etc`,
`/proc`, `/sys`, home dotfiles → `ask`; hard-deny virtual files → `deny`).

**Consequence:** in plan mode a model asking `git status` hits a `[y/N]` wall,
even though the gate itself would approve it and the repo is the very thing it
is meant to inspect. Observed in the live ODF research session: the model burned
prompts on `git`-style inspection and eventually routed around it.

---

## 2. Goal

In `/agentic plan`, let `run_command` **auto-run without confirmation when the
command is provably read-only**, while keeping a hard confirmation on anything
that can change state. Safety stays defense-in-depth:

| Layer | Applies to read-only commands | Stays in force? |
|-------|-------------------------------|-----------------|
| `_confirm` (tool-level) | **skip** (new carve-out)      | —               |
| shell approval gate      | allow verdict via opencode    | yes             |
| Path ACL operand scan    | `/etc`, `/proc`, `/sys`, dotfiles → ask | yes |
| home/root hard-block     | `rm ~`, `/`, `mem`, …       | yes             |

So `git status` inside CWD runs silently; `cat /etc/passwd` still prompts (Path
ACL); `git reset --hard` still prompts (mutating).

---

## 3. Proposed design

A single classifier `_plan_readonly_command(command) -> bool`, consulted by
`_confirm` for `run_command` when `plan_mode` is on:

```python
if plan_mode and tool_name == "run_command" and _plan_readonly_command(cmd):
    return True              # skip prompt; the gate still runs below
```

### 3.1 git surface

`PLAN_READONLY_GIT_SUBCOMMANDS` — subcommands that never mutate:

| Subcommand | Notes |
|------------|-------|
| `status`, `diff`, `log`, `show`, `blame`, `shortlog`, `whatchanged` | core inspection |
| `ls-files`, `ls-tree`, `rev-parse`, `describe`, `name-rev`, `for-each-ref`, `show-ref`, `count-objects`, `check-ignore`, `check-attr`, `diff-tree`, `grep` | read-only plumbing |
| `branch`, `tag`, `remote` | **guard-railed**, see below |

Per-subcommand mutating-flag blocks `_PLAN_READONLY_GIT_MUTATING`:

| Subcommand | Blocked (still prompt)                                                    |
|------------|---------------------------------------------------------------------------|
| `branch`   | `-d -D -m -M -c -C -f -r`, `--delete --move --copy --force --set-upstream --set-upstream-to --track --unset-upstream` |
| `tag`      | `-d -D -a -s -f`, `--delete --annotate --signed --force --local-user`     |
| `remote`   | third token must be `-v`/`--verbose`/`show`/`get-url`; bare `git remote` ok |

Examples that demonstrate why per-subcommand matters:
`git branch -a` (list remote branches → **read**, allow) vs
`git tag -a v1.0` (create annotated tag → **write**, prompt).

### 3.2 binary surface

`PLAN_READONLY_BINARIES` — inspection binaries with a single mutation guard:

| Binary | Guard |
|--------|-------|
| `ls cat head tail wc sort uniq cut file stat du df which pwd whoami uname id groups date echo printf` | none (read-only by nature) |
| `grep rg ugrep` | none |
| `find` | block operands `-delete -exec -execdir -ok` |

Anything else (including `make`, `npm`, `python`, `gcc`, `env`, `sudo`) keeps
the existing confirmation, even though the shell gate may normally allow it.

### 3.3 hard exclusions (always prompt)

- any shell metacharacter: `; & | < > $ \``  — e.g. `git log \| head`, `ls > f`
- anything with more than one positional token path/URL ambiguity — handled by
  program-specific parsing
- commands whose executable is not in the allowlist at all

### 3.4 prompt text

`AGENTIC_PLAN_BLOCK` and the `/agentic plan` ON status lines would change from
"run_command will ask for confirmation before executing" to describe the
carve-out (read-only inspection runs, state-changing commands ask).

---

## 4. What this does NOT do

- Does **not** add a new mode — it refines `/agentic plan` (this file is the
  discussion artifact, not a new `git_plan` mode).
- Does **not** touch `write_file`/`run_python`/`patch`/`edit_file`/`apply_patch`
  — all stay prompt-gated / hidden in plan mode.
- Does **not** disable the shell gate — read-only commands still pass through
  `check_shell_approval`, so `cat /etc/shadow` asks, `head /proc/1/mem` denies.
- Does **not** loosen the same-tool-loop guard or the timeout escalation.

---

## 5. Open questions to resolve before implementing

1. **Trust model.** Binary allowlist + flag blacklist is coarse. Alternative:
   require *all operands inside CWD* (mirroring `_shell_all_operands_inside_cwd`)
   so read-only only auto-runs for the project tree, forcing falls back to
   prompt for anything else. Stricter, but footguns like `ls /home`/`du /` 
   would prompt. Which do we prefer — allow everywhere-if-read-only, or
   CWD-only-if-read-only?

2. **git white-open-ended subcommands.** `git diff --no-index a b`,
   `git log --all`, `git show HEAD:file` are fine; but do we want
   `git remote` (third-token allowlist) at all, or keep it prompting entirely?

3. **flag-space.** Some binaries have mutating long flags that are easy to miss
   (e.g. `gzip` none here, but `dd of=`, `cp` would never be allowed). Keep the
   closed set small, or grow it toward what the observed model actually issues?

4. **Feedback to the model.** On auto-run, should the observation carry a nudge
   ("this ran without confirmation because it is read-only") like the fetch
   redirect does, or stay silent? A nudge teaches the model to prefer the
   builtin `grep`/`read_file` tools rather than shell.

5. **Naming / scope.** Is this the right home (a carve-out inside plan mode), or
   do you want an explicit `/agentic plan [strict|readonly]` sub-level? And
   should the same carve-out apply to `/agentic auto` (which already auto-
   confirms everything except plan-mode/bypass-immune) or only plan mode?

6. **Tests.** Enumerable matrix (allow/block per command) + an integration test
   that `builtins.input` is never called for read-only commands in plan mode.
   Acceptable shape?