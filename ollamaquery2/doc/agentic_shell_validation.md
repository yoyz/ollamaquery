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
| `Executor._run_shell:2471`       | `run_command` / `run_python` tools  | `validate_shell_command_safety:731` + `shell=False` + `shlex.split:2476` |
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
│  L0  container-skip         Executor.mode == "container" → allow     │
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
│  L5  interactive prompt     TTY → y/N/s/a/d  (s=session, a=always)   │
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
    `["echo","hi"]`.
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

Guard `ollamaquery2.py:2366` `_shell_is_home_destructive`:

*   checks both original and normalized verdict nodes for `head == rm` + operand
    containing `~`, `$HOME`, `${HOME}` or stripped `/`, `/*`, `//` etc.
*   anchored via breakdown (not flat regex), so `echo "rm -rf ~"` does not
    trigger — head is `echo`, not `rm`.
*   bypass-immune: evaluated **before** `yolo`/`auto_confirm`/`CWD leniency`
    (`check_shell_approval:2382`). Even `/agentic auto` cannot override it
    (`hermes_implementation_detail_shell_permission_parser.md:48` yolo never
    bypasses hardline).

Only `~/` + `/` are hard-blocked; `rm -rf /tmp/junk` stays `ask` (soft,
promptable) — the tier split `hermes_implementation_detail_shell_permission_parser.md:174`.

### 2.4 CWD Leniency

Requirement: *no over-protection on the directory and subdirectory I'm using*
— relative paths inside the project should not prompt.

`ollamaquery2.py:2330` `_shell_all_operands_inside_cwd`:

*   for each node, tokens `1:` (skip command name, skip flags `-…`) tested
    via `_shell_path_inside_cwd:2308`.
*   Stripped quotes, `~`/`$HOME` → outside (conservative).
*   `a/b`, `./a`, `../a`, `*`, `file.txt` → `os.path.abspath(os.path.join(cwd, p))` +
    `os.path.realpath` + `commonpath([real_cwd, real_p]) == real_cwd`.
*   Globs (`*`, `?`) check base directory only.

Effect in `check_shell_approval:2397`:

```
verdict.effect == ask  +  _all_inside_cwd(nodes) == True  →  allow (no prompt)
```

So `ls | head`, `echo hi > out.txt`, `cat file | grep foo`,
`gcc -o prog prog.c && ./prog` all pass silently when operands are inside
`cwd` — the exact use case for agentic `write_file → run_command → run_python`
pipelines. `cat /etc/passwd` (outside) stays `ask` → `auto_confirm` or prompt.

### 2.5 Gate & Execution

`check_shell_approval:2376` signature
`check_shell_approval(command, ctx, executor_mode) → {approved, effect, message, verdict}`:

1.  `executor_mode == container` → `allow` (host paths bind-mounted but
    container is sandboxed; matches `hermes_style_approval.py:619` skip).
2.  `norm + shell_check_command` on both forms → home guard → `deny` with
    `BLOCKED (rejected): …` (contains `rejected` for `test_agentic.py:401`
    compat).
3.  `verdict.effect == deny` → `BLOCKED (rejected): denied by permission rules`.
4.  `allow` → `allow`.
5.  `ask` + CWD leniency → `allow`.
6.  `auto_confirm` (`ctx.auto_confirm:2400`, wired to `/agentic auto:96`) →
    `allow` (still respects L1/L2 deny).
7.  non-TTY → `deny` with hint `use /agentic auto`.
8.  TTY → prompt `Allow? [y/N/s/a/d]` (`ollamaquery2.py:2411`); `s` appends
    `patterns` to `_SHELL_SESSION_APPROVED:1887`, `a` appends `always`
    (`arity *`) patterns too — mirrors opencode `always` session persistence
    (`opencode_implementation_detail_shell_permission_parser.md:292`).

Call sites `ollamaquery2.py:2471,2723,4331`:

```python
# Executor._run_shell:2471  (host)
ctx = CommandContext() if CommandContext._initialized else None
res = check_shell_approval(command, ctx=ctx, executor_mode=self.mode)
if not res["approved"]: return {"stderr": res["message"], "returncode": -1}
subprocess.run(command, shell=True, executable="/bin/bash", ...)  # pipes/redirs work

# _tool_handle_run_command:2723  (operator ban removed)
# Gate lives in Executor._run_shell; no duplicate check

# execute_os_command:4331  (! inline)
gate = check_shell_approval(command, ctx=ctx, executor_mode="host")
...
subprocess.run(command, shell=True, executable="/bin/bash", stdout=STDOUT)
```

`shell=True` is safe because the gate, not `shell=False`, is the boundary —
the model both Hermes and opencode v2 use (`ollamaquery2_inline_approval_proposal.md:21`).

### 2.6 Default Permission Config

`DEFAULT_SHELL_PERMISSION:1870` (`bash` only, last-match-wins):

| Pattern              | Action | Rationale |
|----------------------|--------|-----------|
| `*`                  | `ask`  | default prompt |
| `git *`, `grep *`, `ls *`, `cat *`, `echo *`, `python3 *`, `gcc *`, `curl *`, … | `allow` | common read/build tools; enables `git status \| grep foo` |
| `rm *`               | `ask`  | soft guard |
| `rm -rf ~` etc.      | `deny` | explicit deny (also covered by hard guard) |

Extensible in-code; could be loaded from `~/.ollamaquery.d/shell_permissions.json` later.

### 2.7 Worked Examples (in `cwd = /tmp/ollama_test_cwd`)

| Command                              | Nodes                              | Verdict   | Gate outcome | Why |
|--------------------------------------|------------------------------------|-----------|--------------|-----|
| `git status \| grep foo`             | `git status`, `grep foo`           | `allow`   | `allow`      | both `allow` rules |
| `echo hi > out.txt`                  | `echo hi > out.txt`                | `allow`   | `allow`      | `echo *` |
| `cat file \| grep hello` (inside)    | `cat file`, `grep hello`           | `ask`→lenient | `allow`  | `rm`? no, CWD operands inside → L3 |
| `rm /tmp/ollama_test_cwd/x` (inside) | `rm /tmp/…/x`                      | `ask`→lenient | `allow`  | CWD leniency, no prompt |
| `cat /etc/passwd` (outside)          | `cat /etc/passwd`                  | `ask`     | `ask` → auto/prompt | outside → L5 |
| `rm -rf ~/junk`                      | `rm -rf ~/junk`                    | `deny`    | `deny` (hard) | L1 home guard, immune to yolo |
| `rm -rf /`                           | `rm -rf /`                         | `deny`    | `deny` (hard) | L1 root guard |
| `rm -rf /tmp/junk`                   | `rm -rf /tmp/junk`                 | `ask`     | `ask` → yolo/prompt | soft, not hard |

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
    `executor_mode == container → allow`.
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

*   `validate_shell_command_safety:731` + `sanitize:705` blocked `| && ; > <`
    outright — pipes impossible.
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
*   Restore heredoc body scanning if agents start using `<<EOF` heavily.

---

*References: `ollamaquery_SHELL_PERMISSION_TODO/comparison_shell_permission.md`,
`opencode_implementation_detail_shell_permission_parser.md`,
`hermes_implementation_detail_shell_permission_parser.md`,
`ollamaquery2.py:1862-2440`.*
