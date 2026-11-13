# Execution target: SSH and Kubernetes — design and location map

Status: **proposal — nothing implemented yet** (`ollamaquery2.py` is unchanged;
this document is the work item)
Date: 2026-10-28
Scope: two new execution targets for `run_command` / `run_python`
(`ssh user@host`, `k8s namespace/pod[/container]`), plus the one thing that must
be written down regardless of the transport: **which tools run where**.

---

## 1. The request

Today tool subprocesses run in one of three places, all selected by `Executor`
(`ollamaquery2.py:4278`):

| Mode                | Selected by             | What happens                                                      |
|---------------------|-------------------------|-------------------------------------------------------------------|
| `host`              | default                 | `subprocess.run(command, shell=True)` behind the shell gate       |
| `container` image   | `--sandbox-image`       | one throwaway `podman run --rm` per command, CWD at `/workspace`  |
| `container` name    | `--sandbox-container`   | `podman exec` into a running container, CWD mapped through it     |

Two more targets are wanted:

1. **SSH** — "run inside an ssh connection, so basically I give `user@host`".
2. **Kubernetes/OpenShift** — "run inside a pod/container … I give
   `namespace/pod/containerpod`, and if I omit the container it goes to the
   default container of that pod".

Neither is a *sandbox* in the isolation sense — both run with the credentials you
already hold. This document therefore says **execution target**, and keeps the
word `sandbox` only where existing CLI/command names demand it (§13 Q3).

---

## 2. What can be reused

Every model-driven subprocess goes through one choke point:
`Executor.run()` → `_run_shell()`. The only other spawners are the cluster tool
(fixed argv `oc`/`kubectl`/`omc`) and the git tool (fixed argv, `shell=False`).

| Concern                  | Where it lives now                                      | Reusable?         |
|--------------------------|---------------------------------------------------------|-------------------|
| Execution                | `Executor.run` (`4486`), `_run_shell` (`4500`)          | add one branch    |
| Container argv           | `_container_command` (`4461`)                           | pattern to copy   |
| Probe + warn once        | `_pre_pull_image` (`4371`)                              | same shape        |
| Banner / status text     | `describe*` (`4333`–`4349`), status (`17941`)           | yes               |
| CLI → ctx → executor     | `_apply_sandbox_args` (`19730`), fields (`1455`)        | yes               |
| Shell gate               | `check_shell_approval` (`4226`)                         | yes, +1 arg       |
| Prompt injection point   | `AgenticRunner.prompt_and_tools` (`14990`)              | yes               |

Two existing properties become load-bearing once the target is remote:

- `Executor.run()` is the only door to a shell, so "where the command runs" is
  one decision, not a per-tool one.
- The prompt is built in exactly one place (`prompt_and_tools`, which
  `/promptshow` and `/tokencount` also read), so a note about the target cannot
  drift from what is sent.

---

## 3. Where each tool runs (the location map)

The part that must be written down even for today's container modes: the model
assumes one filesystem, but the tools do not share one.

| Tool                       | Runs                       | Filesystem          | Gate                 |
|----------------------------|----------------------------|---------------------|----------------------|
| `run_command`              | **target**                 | target              | §6                   |
| `run_python`               | **target**                 | target              | §6                   |
| `read_file`                | local                      | local               | Path ACL             |
| `write_file`               | local                      | local               | Path ACL             |
| `edit_file`                | local                      | local               | Path ACL             |
| `apply_patch`              | local                      | local               | Path ACL + confirm   |
| `grep`                     | local                      | local               | Path ACL, per file   |
| `glob`                     | local                      | local               | Path ACL             |
| `diff`                     | local                      | local               | ACL on operands      |
| `list_directory`           | local                      | local               | Path ACL             |
| `git_inspect`              | local                      | local `git` CWD     | fixed argv           |
| `kubernetes_cluster_query` | local `oc`/`kubectl`/`omc` | cluster             | read-only verbs      |
| `path_policy`              | local                      | none (no access)    | n/a                  |
| `fetch_url`                | **local net**              | target-independent  | scheme + budget      |
| MCP tools                  | local server               | local               | MCP policy           |
| `!cmd`, `/curl` inline     | **local**                  | local               | host gate (`13001`)  |

Spill files are local too: `~/.ollamaquery.d/spill/pid-<n>/`, written by the
cluster tool, `git_inspect` and the timed-out-write recovery, and read back with
`read_file` paging.

Consequences (stated in the model prompt §5.7 and in the README):

1. A file created with `write_file` **does not exist** on the target. To run a
   script there, write it through `run_command` (`cat > f <<'EOF'`), or have the
   target mount the same volume (container mode already relies on this).
2. Inspecting a target file needs `run_command` (`cat`, `sed -n`, `ls`), not
   `read_file`.
3. `read_file` in ssh/k8s mode answers with the **local** file — the sharpest
   footgun of this feature (§10 T6, §13 Q2).
4. A spilled cluster/git result *is* readable with `read_file`, because the spill
   directory is local. That is the tool's own scratch space, not the project
   tree, so there is no inconsistency to fix.

---

## 4. Goals and non-goals

**Goals**

- G1 — `--sandbox-ssh [USER@]HOST` runs `run_command`/`run_python` there.
- G2 — `--sandbox-k8s NS/POD[/CONTAINER]` runs them in that pod; with the
  container omitted, the pod's default container (kubectl's own rule:
  `kubectl.kubernetes.io/default-container`, else the first container).
- G3 — the target shows wherever the container target shows today (banner,
  `/agentic status`, `/agentic sandbox`, `describe()`), and is swappable at
  runtime without a restart.
- G4 — model text can never become an option or a second command locally
  (§5.4, §10 T1/T2).
- G5 — the shell approval gate stays on for remote targets.
- G6 — every turn, the model is told that shell runs remotely and file tools do
  not (§5.7).

**Non-goals (v1)**

- N1 — file synchronisation (`rsync`, `kubectl cp`) or a shared-filesystem map.
  File tools stay local; §3 documents it. Biggest known gap; first v2 candidate.
- N2 — persistent remote sessions. Every command is a new `ssh`/`exec`, so `cd`,
  exported variables and background jobs do not survive; `--sandbox-workdir`
  exists so the model does not need `cd`.
- N3 — TTY allocation (`ssh -t`, `kubectl exec -t`): control sequences in the
  output, and `-t` fails when stdin is not a terminal.
- N4 — `port-forward`, tunnels, `oc debug node`, `kubectl proxy`, `kubectl cp`.
- N5 — picking a pod automatically (selector, deployment, pod restarts).
- N6 — privilege dance inside the pod (`su`, `nsenter`, host namespaces).
- N7 — a jump host of our own: `--sandbox-ssh-option ProxyJump=bastion` is it.

---

## 5. Design

### 5.1 One executor, four modes

`Executor.mode` grows from `{"host", "container"}` to
`{"host", "container", "ssh", "k8s"}`. Everything that branches on the mode
today is reached from `describe()`, so the blast radius is one class and its
callers.

```
run(command)
 ├─ "host"      → _run_shell(command, gate=True, shell=True)
 ├─ "container" → _run_shell(_container_command(cmd), gate=False, argv=…)
 ├─ "ssh"       → _run_remote(_ssh_argv(cmd), …)   gate=True
 └─ "k8s"       → _run_remote(_k8s_argv(cmd), …)   gate=True
```

Constructor additions (`arg > env > default`):

```python
Executor(…, ssh_target=None, ssh_port=None, ssh_key=None, ssh_options=None,
         k8s_target=None, k8s_binary=None, k8s_context=None,
         k8s_kubeconfig=None, remote_workdir=None)
```

| Member                    | Purpose                                     |
|---------------------------|---------------------------------------------|
| `describe_target()`       | `ssh user@host` / `k8s ns/pod [ctr]`        |
| `describe_limits()`       | remote host's / pod's own limits            |
| `_ssh_argv(remote_args)`  | full `ssh` argv (§5.2)                      |
| `_k8s_argv(remote_args)`  | full `oc`/`kubectl exec` argv (§5.3)        |
| `_remote_argv(cmd)`       | workdir prefix + transport wrapping         |
| `_run_remote(cmd, t)`     | gate → probe → argv run (§5.6)              |
| `_probe_remote()`         | once-per-session reachability (§5.5)        |
| `_ssh_control_path()`     | private `%C` socket path (§5.2)             |

`run()` keeps returning `{"stdout", "stderr", "returncode"}`, so
`_tool_handle_run_command` (`6486`), `_tool_handle_run_python` (`6459`) and the
observation capping need **no change at all**.

### 5.2 SSH transport

Command construction (argv, `shell=False` on our side):

```
ssh [-o ProxyJump=…] -o BatchMode=yes -o ControlMaster=auto \
    -o ControlPath=<priv>/%C -o ControlPersist=120 [-p PORT] [-i KEY] \
    -- [user@]host "sh -c '<workdir-prefixed command>'"
```

Three details that are easy to get wrong:

1. **The remote command is one argv element.** `ssh` joins everything after the
   destination with single spaces and hands the result to the remote login
   shell, so `["sh", "-c", "echo 'a b'"]` arrives as `sh -c echo 'a b'` and
   breaks. Pass the single string `sh -c '…'`, inner command already
   `shlex.quote`d (test T2 pins it).
2. **`--` before the destination** so a destination can never be read as an
   option (verified here on OpenSSH 10.2p1). Validation makes a leading `-`
   impossible anyway, so this is belt and braces.
3. **`BatchMode=yes`** so a missing key or unknown host key fails immediately
   with a usable message instead of blocking a tool call on an invisible prompt.

Multiplexing: `ControlMaster=auto` + `ControlPath=~/.ollamaquery.d/ssh/%C`
(`%C` hashes local host, remote host, port, user) + `ControlPersist=120` means a
burst of tool calls pays one handshake. The directory is created `0700`
(`_ensure_private_dir`, `2602`); if it is not writable we fall back to
`tempfile.gettempdir()`. Only the same UID can use the socket — the same trust
level as the key.

Option precedence: user options are emitted **before** our defaults, because ssh
uses the *first* obtained value for a keyword. So
`--sandbox-ssh-option BatchMode=no` (or `StrictHostKeyChecking=accept-new`)
really overrides instead of being silently ignored.

Workdir: prepended as `cd <quoted> && …` — ssh has no working-directory option
(its `-w` is a tunnel device). Empty by default → the login shell's starting
directory.

Deliberately absent: no `-t`, no `-A` (no agent forwarding), no `-n` (we set
`stdin=DEVNULL` at the subprocess level instead, §5.6).

### 5.3 Kubernetes transport

```
oc|kubectl [--context CTX] [--kubeconfig PATH] exec -n NS POD [-c CONTAINER] \
    -- sh -c "<workdir-prefixed command>"
```

- **Binary**: `--sandbox-k8s-binary auto` (default) picks `oc` when on `PATH`,
  else `kubectl`; an explicit choice wins. Both take the same `exec` argv, so
  OpenShift needs no separate mode.
- **Default container**: with no `/container` in the target, `-c` is omitted and
  kubectl applies its own rule (annotation, else first container). The client is
  the authority — reimplementing the rule would rot.
- **`--` before the command** is mandatory: without it, `-c` inside the command
  would be parsed by kubectl.
- **Workdir**: kubectl `exec` has no `-w`, so `cd <quoted> && …` is prepended.
  Empty by default → the container's working directory.
- **No TTY, no stdin**: `-t` would fail on non-interactive stdin, `-i` would let
  a remote `cat` consume the chat's keystrokes.

Each call is a fresh `kubectl` process (~0.3–1 s warm). That is the cost of N2;
`kubectl attach` to a persistent shell is a different feature with its own state
machine.

### 5.4 Validation (no model value becomes an argv option)

| Value                     | Rule                                   | Why                                      |
|---------------------------|----------------------------------------|------------------------------------------|
| `--sandbox-ssh`           | `^[A-Za-z0-9_][A-Za-z0-9._:@\[\]-]*$`  | `-o…=x` as a destination is an injection |
| `--sandbox-ssh-port`      | int 1–65535                            | `-p` takes one value                     |
| `--sandbox-ssh-key`       | path, `~` expanded                     | never string-concatenated                |
| `--sandbox-ssh-option`    | `-x` → raw flag, else `-o <value>`     | keeps `-v` possible                      |
| `--sandbox-k8s`           | 2 or 3 parts, each a DNS name          | namespace/pod/container                  |
| `--sandbox-k8s-binary`    | argparse choices `auto`/`oc`/`kubectl` | never a model path                       |
| `--sandbox-k8s-context`   | one argv element after a fixed flag    | no shell involved                        |
| `--sandbox-workdir`       | `shlex.quote`d into `cd …`             | a shell *does* read this                 |

Names are DNS subdomains: `^[a-z0-9]([a-z0-9.-]*[a-z0-9])?$`; empty parts and a
leading `-` are rejected. Errors come from module helpers
(`_validate_ssh_target`, `_parse_k8s_target`) so they are unit-testable unbound:
surfaced at startup by `_apply_sandbox_args` (exit 2) and defensively at argv
build time (tool error, never a traceback).

### 5.5 Reachability probe (memoized, non-blocking)

`_pre_pull_image` already has the right shape: one subprocess per session, one
warning on failure, no retry, never abort the query. `_probe_remote()` runs
`ssh … -- true` / `exec … -- true` with a 20 s cap and warns once:

```
[Sandbox] cannot reach ssh user@host: Permission denied (publickey).
[Sandbox] commands will still be attempted; check your key/agent or kube context.
```

Order inside `_run_remote`: **gate, then probe, then run** — a denied command
must not open a network connection.

### 5.6 Subprocess details

- `stdin=subprocess.DEVNULL` for remote (and opportunistically container)
  commands. Today `_run_shell` inherits our stdin, so a model-run `cat` can
  swallow the user's keystrokes — over ssh it would also send them across the
  wire. Worth fixing while here (test T7).
- `shell=False` with an argv list, always. No string→shell step on our side.
- `shutil.which(argv[0])` first, so a missing client yields
  `[Sandbox] oc not found in PATH — install it or pass --sandbox-k8s-binary`
  instead of `FileNotFoundError`.
- Exit codes pass through; non-zero is a normal `{"success": False}` observation.

### 5.7 Prompt: `## Execution target`

Injected per query by `AgenticRunner.prompt_and_tools` (`14990`), right after the
`## Path access` block, and also at the two places that rebuild the system
message at runtime (`14755` streaming re-entry, `15533` native-tools fallback).
Because `/promptshow` and `/tokencount` read the same method, the note is visible
in both, and it cannot drift from the payload.

Draft, emitted only when the target is not `host`:

```
## Execution target
run_command and run_python run inside <target>. Every other tool — read_file,
write_file, edit_file, apply_patch, grep, glob, diff, list_directory,
git_inspect — reads and writes the LOCAL filesystem of this machine, which is a
different filesystem unless that target mounts it. To inspect a file inside the
target, use run_command (cat, sed -n, ls). A file you create with write_file does
NOT exist in the target.
```

~95 tokens/turn, static for the session (same KV-cache property as the
`/maxtokens` line).

---

## 6. Gate policy for remote targets

The gate stays on: a host or a pod is real infrastructure, unlike the throwaway
image sandbox. `check_shell_approval(command, ctx, executor_mode, remote=False)`
gets one new parameter and four behaviour changes.

| Gate stage                         | ssh / k8s        | Why                                         |
|------------------------------------|------------------|---------------------------------------------|
| `executor_mode == "container"`     | not applied      | remote is not disposable                    |
| `_shell_hard_block` (`rm ~/`, `/`) | **kept**         | token-based; destroys the remote home too   |
| `_shell_breakdown` allow/ask/deny  | **kept**         | `ls`/`cat`/`git` allow, `rm` asks           |
| `_shell_all_operands_inside_cwd`   | **disabled**     | local CWD is not the remote CWD             |
| `_shell_acl_scan` (`4098`)         | **disabled**     | operands name remote paths → false prompt   |
| `auto_confirm` (`/agentic auto`)   | unchanged        | bypasses ask, never deny                    |
| non-TTY ask                        | deny + log       | unchanged                                   |

On `host`, all of those stages apply exactly as today.

The gate sees the **inner command**, never our client argv, and the model's
string is executed by the remote shell as written — that is what `run_command`
means; it is gated, not rewritten.

One existing interaction to note: `_CLUSTER_MUTATING_VERBS` (`3410`) deny-lists
`oc exec` / `kubectl exec` for the *model's own* `run_command`. That does not
block our transport (built from validated, non-model values), and it does stop a
double hop (`run_command("kubectl exec …")`), which is desirable.

---

## 7. CLI and environment surface

New flags in the existing `sandbox_group` (`19424`). The `--sandbox-*` prefix is
kept for continuity even though only one target is an isolation boundary.

| Setting          | CLI                                | Environment                        | Default            |
|------------------|------------------------------------|------------------------------------|--------------------|
| SSH target       | `--sandbox-ssh [USER@]HOST`        | `OLLAMAQUERY_SANDBOX_SSH`          | —                  |
| SSH port         | `--sandbox-ssh-port PORT`          | `…_SSH_PORT`                       | ssh config (22)    |
| SSH key          | `--sandbox-ssh-key PATH`           | `…_SSH_KEY`                        | agent / config     |
| SSH option       | `--sandbox-ssh-option OPT` (x N)   | `…_SSH_OPTION` (space-separated)   | —                  |
| k8s target       | `--sandbox-k8s NS/POD[/CTR]`       | `OLLAMAQUERY_SANDBOX_K8S`          | —                  |
| k8s binary       | `--sandbox-k8s-binary`             | `…_K8S_BINARY`                     | `auto`             |
| k8s context      | `--sandbox-k8s-context CTX`        | `…_K8S_CONTEXT`                    | current context    |
| k8s kubeconfig   | `--sandbox-k8s-kubeconfig PATH`    | `…_K8S_KUBECONFIG`                 | `~/.kube/config`   |
| Remote workdir   | `--sandbox-workdir DIR`            | `OLLAMAQUERY_SANDBOX_WORKDIR`      | target default     |

`_apply_sandbox_args` (`19730`) then:

1. validates the target strings (exit 2 with a readable message on a typo);
2. **refuses conflicting targets** — `--sandbox-ssh` + `--sandbox-k8s` +
   `--sandbox-image` at once is a mistake, not a priority order, so it exits with
   "choose one target";
3. sets `ctx.executor_mode` to the single surviving target (`ssh` / `k8s` /
   `container`), else `host`.

Existing `OLLAMAQUERY_CONTAINER_*` names are untouched (saved shell history and
tests depend on them); the new family is `OLLAMAQUERY_SANDBOX_*`.

---

## 8. `/agentic sandbox` at runtime

Four targets cannot be a toggle. Shape:

| Command                      | Effect                                       |
|------------------------------|----------------------------------------------|
| `/agentic sandbox`           | status: current target + what is configured  |
| `/agentic sandbox status`    | same                                         |
| `/agentic sandbox host`      | local execution                              |
| `/agentic sandbox container` | image or running container (as today)        |
| `/agentic sandbox ssh`       | error when `--sandbox-ssh` was not given     |
| `/agentic sandbox k8s`       | error when `--sandbox-k8s` was not given     |
| `/agentic sandbox on`        | alias for `container` (old muscle memory)    |
| `/agentic sandbox off`       | alias for `host`                             |

Status block (stderr, like the rest of this code):

```
[Sandbox target]
  > current    [ssh] ssh (ssh deploy@10.0.0.12, workdir /srv/app)
    host       always available
    container  image docker.io/library/python:3.12-alpine
    ssh        ssh deploy@10.0.0.12
    k8s        not configured (--sandbox-k8s namespace/pod[/container])
  /agentic sandbox <host|container|ssh|k8s>   (on = container, off = host)
```

Implementation notes:

- `_agentic_toggle_sandbox()` (`17806`) keeps its container↔host toggle (a test
  pins it) and backs the `on`/`off` aliases. New: `_agentic_sandbox(parts)`
  (dispatch), `_set_sandbox_mode(mode)` and `_print_sandbox_status()`.
- `_set_sandbox_mode` must set **both** `self.executor.mode` and
  `self.tool_registry.executor.mode` — they are two objects, which is exactly
  what the existing toggle already does.
- `/agentic status` (`17954`) already prints `executor.describe()`, so it picks
  up new targets for free; only the subcommand list (`17945`) and the usage
  strings (`17600`, `483`) change.
- Tab completion: add `'sandbox': ['host', 'container', 'ssh', 'k8s', 'on',
  'off', 'status']` to `AGENTIC_SECOND_LEVEL` (`12074`).
- Banner (`19780`): condition changes from `== "container"` to `!= "host"`.

---

## 9. Failure modes and messages

| Situation                        | Result                                                                     |
|----------------------------------|----------------------------------------------------------------------------|
| `--sandbox-ssh -oProxyCommand=…` | exit 2: `invalid ssh target … (expected [user@]host)`                      |
| `--sandbox-k8s mypod`            | exit 2: `invalid k8s target 'mypod' (expected ns/pod[/ctr])`               |
| ssh + k8s both given             | exit 2: `choose one target: ssh, k8s were all given`                       |
| `/agentic sandbox ssh`, unset    | `[Sandbox] no ssh target configured — --sandbox-ssh user@host`             |
| `ssh`/`oc`/`kubectl` missing     | `[Sandbox] oc not found in PATH — install it or pass --sandbox-k8s-binary` |
| first command cannot connect     | one warning with the client's last stderr line; command still runs         |
| remote exit ≠ 0                  | normal tool failure observation, loop continues                            |
| blocked by the gate              | `{"returncode": -1, "stderr": "<gate message>"}` (unchanged)               |
| tool timeout                     | `Timed out after Ns` — the remote process may survive (§13 Q7)             |

---

## 10. Security analysis

| #    | Threat                                 | Mitigation                                                      |
|------|----------------------------------------|-----------------------------------------------------------------|
| T1   | model text becomes an ssh/k8s option   | regex-validated target, `--` separator, one argv element        |
| T2   | model text becomes a second command    | `shell=False`, argv list; only the intentional `sh -c <quoted>` |
| T3   | prompt injection retargets the session | flags are CLI-only; `ssh other-host` is still a gated command   |
| T4   | credential theft via control socket    | `0700` dir, same-UID socket, never `-A`, no key material copied |
| T5   | false "sandbox" confidence             | gate on (§6); named execution target, not isolation             |
| T6   | model assumes files are remote         | prompt block (§5.7), README, this doc §3                        |
| T7   | remote command eats chat stdin         | `stdin=DEVNULL` (§5.6)                                          |
| T8   | gate false positives make it unusable  | leniency + local ACL scan off, breakdown kept                   |
| T9   | cluster mutation via the transport     | only `exec`; inner command gated; `oc exec` stays deny-listed   |

Host keys are the user's ssh policy: `BatchMode=yes` plus `~/.ssh/known_hosts`.
First contact therefore fails with `Host key verification failed` until the user
has connected once, or passes
`--sandbox-ssh-option StrictHostKeyChecking=accept-new` — written down as exactly
the trust decision it is. No `accept-new` default: silently trusting whatever
answers on an address is an MITM invitation.

---

## 11. Test plan

All offline: `subprocess.run` is patched and the **argv is the assertion**, which
is where the quoting and injection bugs live.

New class `tests/test_agentic.py::TestRemoteExecutionTarget`:

| Test                             | Pins                                          |
|----------------------------------|-----------------------------------------------|
| `test_ssh_argv_shape`            | `ssh … -- host "sh -c '…'"`, one element      |
| `test_ssh_quotes_round_trip`     | spaces/quotes in the command survive          |
| `test_ssh_workdir_cd_prefix`     | `cd '/srv/app' && …`, quoted                  |
| `test_ssh_port_key_options`      | `-p`, `-i`, `-o X`, raw `-v`                  |
| `test_ssh_batchmode_default`     | no interactive prompt path                    |
| `test_user_option_wins`          | extras emitted before our defaults            |
| `test_ssh_rejects_injection`     | `-oProxyCommand=…`, `a b`, empty              |
| `test_k8s_argv_shape`            | `exec -n ns pod -c ctr -- sh -c "…"`          |
| `test_k8s_default_container`     | no `-c` when omitted                          |
| `test_k8s_binary_detection`      | `oc` → `kubectl` fallback, explicit wins      |
| `test_k8s_context_kubeconfig`    | global flags before `exec`                    |
| `test_k8s_target_validation`     | 1/4 parts, empty, uppercase, leading `-`      |
| `test_gate_runs_when_remote`     | denial stops the argv, no probe               |
| `test_gate_skips_leniency_acl`   | no false prompt for `cat /etc/passwd`         |
| `test_home_hard_deny_remote`     | `rm -rf ~/` refused                           |
| `test_stdin_devnull`             | T7                                            |
| `test_missing_binary_message`    | `[Sandbox] oc not found in PATH`              |
| `test_probe_once_per_session`    | 3 commands → 1 probe, 1 warning               |
| `test_describe_is_human`         | banner/status strings                         |
| `test_cli_flags_reach_executor`  | `_apply_sandbox_args` → `create_executor`     |
| `test_conflicting_targets`       | exit 2                                        |
| `test_sandbox_subcommand`        | switch + report for all four                  |
| `test_sandbox_switch_errors`     | unconfigured target message                   |
| `test_prompt_block_remote_only`  | via `_agentic_prompt_and_tools()`             |
| `test_prompt_block_names_target` | target text present in the prompt             |

Regression guards that must stay green untouched: `TestContainerSandboxConfig`
(18), `TestExecutorContainerModeSafety`, `tests/test_acl.py`, and
`TestPromptShow`'s "JSON = wire" equality.

Live (opt-in, skipped without the environment):

- **ssh** — `SSH_TEST_HOST=user@host`: `pwd`, `echo $HOME`, workdir `cd`, exit
  code passthrough, gate denial of `rm -rf ~/`, timed-out `sleep`.
- **k8s** — `K8S_TEST_TARGET=ns/pod[/ctr]`: `pwd`, default container, named
  container, missing binary message. No `oc`/`kubectl` exists on the box used to
  write this, so this half needs a cluster-attached machine.

---

## 12. Implementation plan

One commit per step; the offline suite gates each one.

**P1 — `Executor` transports** (`4278`–`4534`). Modes, `_ssh_argv`,
`_k8s_argv`, `_remote_argv`, `_run_remote`, `_probe_remote`, `describe*`,
`stdin=DEVNULL`, `shutil.which` check. Tests: argv, quoting, validation, probe,
describe, stdin.

**P2 — remote gate** (`4226`–`4272`). `remote=` parameter: skip CWD leniency and
the local ACL scan, keep the home/root hard block. Tests: gate on remotely, no
false ACL prompt, home deny.

**P3 — configuration** (`1455`, `1559`, `19424`, `19730`, `19780`).
`CommandContext` fields, `create_executor`, CLI group, `_apply_sandbox_args`
(validation, conflict exit), startup banner. Tests: flags → context → executor,
conflicts, banner condition.

**P4 — runtime switching** (`17587`, `17600`, `17806`, `17945`, `12074`, `483`).
`/agentic sandbox` subcommand, status block, usage strings, completion. Tests:
each action, unconfigured error, status rendering.

**P5 — prompt block** (`14990`, `14755`, `15533`). `## Execution target`, plus
the two runtime prompt rebuilds. Tests: present for ssh/k8s/container, absent for
host, names the target, `/promptshow` sees it.

**P6 — documentation.** README "Execution targets" section + security bullet
(`233`–`309`, `350`–`374`), AGENTS.md component 11, `doc/testing.md` coverage
line, and this file flipped to `Status: implemented` with the line map refreshed.

Verification per commit (`TODO_refactor.md` §0.3): `py_compile`, method-set diff
for the class touched (no silent losses), focused tests, one full offline suite
per batch, and one live-backend run for the batch that closes the feature.

---

## 13. Open questions

Decided during design, kept here so the reasoning survives:

- **Q0 — file tools with a remote target?** Local, documented (§3, §5.7).
- **Q0b — gate on remote?** On (§6).
- **Q0c — ssh auth posture?** `BatchMode=yes`, fail fast, overridable (§5.2).
- **Q0d — `/agentic sandbox` semantics?** Subcommand, bare = status (§8).

Still open, with a recommendation each:

- **Q1 — remote file access (v2).** (a) `--sandbox-map LOCAL:REMOTE` declaring
  "this local dir *is* that remote dir" — cheap, honest, correct when the
  filesystem is shared (NFS, a pod volume); (b) rsync/`kubectl cp` sync before
  and after each command — powerful, but diff-back/conflict semantics are a
  feature of their own; (c) nothing, lean on `run_command`. Recommend (a) then
  (c).
- **Q2 — file-tool labelling in remote mode.** When the target is not `host`,
  prefix `read_file`/`write_file` output with `[local] <path>` so a
  wrong-filesystem read is visible in the transcript instead of silent. Cheap
  (~2 tokens), removes the worst failure mode (§10 T6). Decide before P5 —
  same observation story.
- **Q3 — naming.** `--sandbox-ssh` / `--sandbox-k8s` continue the existing
  family; alternatives were `--exec-target ssh:…` or `--remote-*`. Recommend
  keeping `sandbox` (documented as "target, not isolation") and not adding a
  second spelling: `/agentic sandbox` is what users already type.
- **Q4 — remote timeout policy.** `run_command` defaults to 10 s, caps at 300 s.
  An ssh handshake plus a remote build exceeds that sooner than a local one.
  Recommend keeping the handler defaults (the model can pass `timeout`) and
  revisiting if the prompt block alone does not fix it.
- **Q5 — pod restart.** A pinned `ns/pod` dies with the pod; every later call
  fails with a clear kubectl error and the session must restart with a new flag.
  Recommend keeping it pinned (a selector needs a re-resolve policy and caching);
  consider resolving at first use later if it hurts.
- **Q6 — default container discovery.** We rely on kubectl's rule. Making the
  resolved name visible would cost an extra `oc get pod -o jsonpath` per session.
  Recommend not doing it in v1 and documenting the annotation here.
- **Q7 — timed-out remote commands.** Killing the local client does not
  necessarily stop the remote process (sshd may or may not HUP it; `kubectl
  exec` leaves it to the container). Options: accept and document, or send a
  best-effort cleanup command. Recommend documenting it in the run_command
  timeout message and not building a remote process tracker in v1.

---

## 14. Alternatives considered and rejected

- **Do nothing; the model runs `ssh host '…'` / `kubectl exec …` through
  `run_command`.** Works today for ssh (one prompt each time) but has no argv
  discipline (quote-and-pray), no workdir, no connection reuse, and `kubectl
  exec` is deny-listed by the gate. It also cannot express "this whole session
  is remote", which is what makes file-tool confusion explicit and fixable.
- **A long-lived shell session per target (`ssh` with typed stdin, or `kubectl
  attach`).** Statefulness is a much larger contract: prompt detection,
  interleaved output, dead-session recovery. N2 keeps v1 stateless.
- **A new tool family (`run_remote`, `read_remote_file`, …).** Doubles the
  surface the prompt must teach (`/listtool`, native schema, plan-mode sets) for
  a capability that is a property of the target, not of the tool.
- **Modelling k8s as a container runtime (`--sandbox-runtime kubectl`).**
  `run`/`list`/`inspect`/`pull` do not exist there; `exec` is the only shared
  verb and needs namespace/pod/context, so it would poison every runtime branch.
- **Reusing `kubernetes_cluster_query` for execution.** That tool is
  deliberately read-only (read verbs allow-listed, mutating verbs hard-denied);
  adding `exec` would dissolve the guarantee. Keep the split: cluster *state*
  via `kubernetes_cluster_query`, *execution* in a pod via the k8s target.
- **An MCP server for ssh/k8s.** Viable later (the MCP client exists) and
  probably the right home for remote *file* access (Q1), but MCP tools cannot own
  `run_command`/`run_python`, which is the actual ask.

---

## 15. Reference

### 15.1 Target argv

```
# ssh, defaults, workdir /srv/app, command: cat /etc/os-release | head -1
ssh -o BatchMode=yes -o ControlMaster=auto \
    -o ControlPath=/home/me/.ollamaquery.d/ssh/%C -o ControlPersist=120 \
    -- deploy@10.0.0.12 \
    "sh -c 'cd /srv/app && cat /etc/os-release | head -1'"

# kubernetes, default container
kubectl --context prod exec -n web api-7d9f-abcde -- sh -c "ls -l /tmp"

# kubernetes, named container + explicit kubeconfig
oc --kubeconfig /etc/kube/config exec -n web api-7d9f-abcde -c sidecar \
   -- sh -c "cd /var/log && tail -n 20 app.log"
```

### 15.2 Line map (`ollamaquery2.py`, 20 007 lines at the time of writing)

| Item                                                       | Line                           |
|------------------------------------------------------------|--------------------------------|
| container constants                                        | `257`–`265`                    |
| `DEFAULT_SHELL_PERMISSION` / cluster deny verbs            | `3321` / `3410`                |
| `_shell_all_operands_inside_cwd` / home guard / ACL scan   | `3933` / `3958` / `4098`       |
| `check_shell_approval` (container bypass)                  | `4226` (`4239`)                |
| `class Executor` … `_run_shell`                            | `4278`–`4534`                  |
| `describe*` / `_pre_pull_image` / `_container_command`     | `4333` / `4371` / `4461`       |
| `CommandContext` sandbox fields / `create_executor`        | `1455` / `1559`                |
| file tools (`read_file`, `write_file`, `glob`, `grep`)     | `5657`, `5766`, `5822`, `6371` |
| `diff` / `edit_file` / `apply_patch` / `git_inspect`       | `6571`, `6641`, `7104`, `7933` |
| `run_python` / `run_command` handlers                      | `6459` / `6486`                |
| `AGENTIC_SUBCOMMANDS` / second level / `_complete_agentic` | `12067` / `12074` / `12286`    |
| `prompt_and_tools` / re-entry / tools fallback             | `14990` / `14755` / `15533`    |
| `/agentic sandbox` dispatch / toggle / status              | `17587` / `17806` / `17941`    |
| `_agentic_prompt_and_tools` (ChatLoop door)                | `18126`                        |
| CLI group / `_apply_sandbox_args` / banner                 | `19424` / `19730` / `19780`    |
| inline `!` gate (always host)                              | `12979` (`13001`)              |
| README sandbox / security sections                         | `233`–`309` / `350`–`374`      |
