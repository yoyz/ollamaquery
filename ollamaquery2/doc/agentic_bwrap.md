# Agentic bwrap sandbox — rootless execution engine (0.5.0)

Status: **implemented in 0.5.0** (`Executor`, presets `work`/`exec`).
Scope: `run_command` / `run_python` only. File tools remain in the harness and
are governed by the Path ACL (see `README.md` "Access Control").

---

## 1. Why bwrap

`bwrap` (bubblewrap) is a setuid/user-namespace sandbox binary. Compared with
the container engine:

| Property              | bwrap                                  | podman/docker container                    |
| --------------------- | -------------------------------------- | ------------------------------------------ |
| Daemon / image        | none, a binary                         | daemon + image (pull, storage)             |
| Start cost            | instant                                | ~0.2 s warm, seconds cold                  |
| Rootless              | yes (user namespaces)                  | rootless podman, but more moving parts     |
| Mount policy          | fine-grained ro/rw binds               | `-v`, image filesystem                     |
| Project path          | same absolute path as the host         | `/workspace` (or the container's mount)    |
| cgroup caps (v1)      | **none**                               | `--memory` / `--cpus`                      |
| Package isolation     | host binaries only                     | full image                                 |
| Platforms             | Linux + unprivileged user namespaces   | Linux/macOS/Windows (runtime)              |

The MVP uses bwrap as the everyday engine (`work`), a read-only execution engine
(`exec`), and keeps the container for reproducibility, package installation and
resource caps.

## 2. The mount policy is the contract

In bwrap mode the shell approval gate is **bypassed** (like the container
engine). The barrier is the kernel: anything outside a writable mount returns
`EROFS`. Each command gets a fresh namespace, so the policy can change from one
command to the next.

Default policy (built by `Executor._bwrap_command`):

| Path / object                                                              | Mount                                                                                             | Why                                                                 |
| -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| `/usr`, `/bin`, `/sbin`, `/lib`, `/lib64`, `/etc`, `/var`, `/run`, `/opt`  | `--ro-bind-try`                                                                                   | host binaries, libraries, config, DNS (`/etc/resolv.conf` → `/run`) |
| `/tmp`, `/var/tmp`                                                         | `--tmpfs`                                                                                         | disposable; the session scratch is re-bound inside                  |
| project (CWD)                                                              | `--bind` rw (`work`) or `--ro-bind` (`exec`)                                                      | same absolute path as the host                                      |
| session scratch `/tmp/ollamaquery/pid-N`                                   | `--bind` rw                                                                                       | `/agentic scratch` keeps working                                    |
| `~/.ollamaquery.d/sandbox/pid-N/home`                                      | `--bind` rw                                                                                       | pip/npm caches; the real `$HOME` is never mounted                   |
| `/dev`, `/proc`                                                            | `--dev`, `--proc`                                                                                 | the namespace's own devices and procfs                              |
| network                                                                    | `--share-net` (`work`) or `--unshare-net` (`exec`)                                                | builds/installs vs no DNS/downloads                                 |
| isolation                                                                  | `--unshare-all --die-with-parent --new-session --cap-drop ALL --clearenv`                         | namespaces, parent death, terminal safety, caps, minimal env        |

Environment inside the sandbox: `PATH`, `HOME` (the private sandbox home),
`TERM`, `LANG=C.UTF-8`, `TMPDIR=/tmp`, plus `GIT_OPTIONAL_LOCKS=0` and
`GIT_TERMINAL_PROMPT=0` so a read-only project can still run `git status`
without an `index.lock` fight or a credential prompt. The sandbox home is
removed at exit and dead-PID homes are swept at startup; `OLLAMAQUERY_KEEP_SCRATCH=1`
keeps it for post-mortem, like the scratch dir.

Measured on this machine (bwrap 0.11.1):

| Operation                                     | Result                                  |
| --------------------------------------------- | --------------------------------------- |
| `gcc hello.c && ./hello`                      | works                                   |
| `echo x > /etc/x`                             | `Read-only file system`                 |
| `ls /root`                                    | absent                                  |
| `getent hosts example.com` (`--share-net`)    | resolves                                |
| DNS + TCP 8090 (`--unshare-net`)              | both blocked                            |
| `echo x > project/file` (`project rw`)        | works                                   |
| `echo x > project/file` (`project ro`)        | `EROFS`                                 |
| `sed -i` on a project file (`project ro`)     | `EROFS` (temp file creation)            |
| `echo x > $HOME/cache` (sandbox home)         | works                                   |
| `touch /var/tmp/x`, `touch /tmp/x`            | works (tmpfs)                           |

## 3. Engine selection

```text
--sandbox-engine {host,container,bwrap}
```

| Surface                                                               | Environment                          | Default                                      |
| --------------------------------------------------------------------- | ------------------------------------ | -------------------------------------------- |
| `--sandbox-engine host\|container\|bwrap`                             | `OLLAMAQUERY_SANDBOX_ENGINE`         | preset (`work`/`exec`=bwrap, `trusted`=host) |
| `--sandbox-bwrap-project rw\|ro`                                      | `OLLAMAQUERY_BWRAP_PROJECT`          | `rw`                                         |
| `--sandbox-bwrap-net shared\|none`                                    | `OLLAMAQUERY_BWRAP_NET`              | `shared`                                     |
| `--sandbox-bwrap-bind PATH[:ro\|rw]` (repeatable)                     | `OLLAMAQUERY_BWRAP_BINDS` (comma)    | none                                         |
| `--sandbox-bwrap-rw PATH` (repeatable)                                | `OLLAMAQUERY_BWRAP_RW` (comma)       | none                                         |

`--sandbox-engine` conflicts with `--sandbox-image`/`--sandbox-container` (a
contradictory request is an error, not a silent precedence rule).
`--sandbox-bwrap-project`/`--sandbox-bwrap-net` are overrides: the preset uses
them instead of its own values. `exec` refuses an override that breaks its
promise (`project ro` + `net none`); `work` and `trusted` accept a stricter
policy.

At runtime:

```text
/agentic sandbox                # historical host <-> container toggle
/agentic sandbox status         # current engine, project/net mode, bind counts
/agentic sandbox bwrap          # switch (probes first; refusal keeps the old engine)
/agentic sandbox host|container
```

## 4. Presets and bwrap

| Preset    | Engine | Project | Net    | Surface                                              | Approval |
| --------- | ------ | ------- | ------ | ---------------------------------------------------- | -------- |
| `read`    | none   | n/a     | n/a    | 9 inspection tools, no shell                         | plan     |
| `exec`    | bwrap  | ro      | none   | read + `run_command`/`run_python` (no write tools)   | auto     |
| `work`    | bwrap  | rw      | shared | every tool                                           | auto     |
| `trusted` | host   | n/a     | n/a    | every tool                                           | auto     |

`/agentic on` selects `work`; `plan`/`full` are aliases for `read`/`trusted`.
`auto_confirm` is a **derived** value in the MVP: `read` forces plan mode,
`exec`/`work` run sandboxed without prompts, `trusted` is the host escape hatch.
The bare `/agentic auto` command keeps its historical auto-confirm toggle; the
preset alias is `/agentic preset auto`.

## 5. What bwrap does not cover

1. **File tools stay in the harness.** `read_file`, `write_file`, `edit_file`,
   `apply_patch`, `git_inspect`, `fetch_url` do not go through the Executor;
   their policy is the Path ACL. `exec` removes the write tools from the
   surface rather than claiming the mount protects them. A bare
   `--sandbox-bwrap-rw <project path>` is an explicit escape hatch.
2. **A rw project is destructible.** `work` is the trust stance, exactly like a
   container with `-v cwd:/workspace`. Use `exec` (ro) or git when that matters.
3. **No cgroup caps in v1.** A fork bomb or a RAM-hungry build is not contained.
   v2 may wrap the command in `systemd-run --user --scope -p MemoryMax=…` when
   available; the container keeps `--memory`/`--cpus`.
4. **Host binaries only.** `/usr` is read-only and there is no image: a
   different toolchain needs the container.
5. **Linux + user namespaces.** The probe is **fail-closed**: a missing bwrap
   or a refused userns makes tool commands return an error and a one-time
   warning; it never silently falls back to host. `--sandbox-engine host` (or
   `trusted`) is the explicit unsandboxed mode.
6. **No dedicated seccomp profile in v1** (namespaces + `--cap-drop ALL` only).
7. **No persistent environment.** Each command starts from the same mounts;
   state lives in the project or the sandbox home.
8. **Shared network can exfiltrate.** Use `--sandbox-bwrap-net none` for code
   that must not reach the network; builds that fetch will fail.
9. **No image reproducibility.** The environment is the host's.

## 6. Implementation map

- `Executor.__init__` — `bwrap_project`, `bwrap_net`, `bwrap_binds`,
  `bwrap_rw`, `bwrap_home` (tests).
- `Executor._bwrap_probe` — `shutil.which("bwrap")` + `bwrap --dev /dev
  --ro-bind / / true` (5 s), memoized.
- `Executor._bwrap_command` — the argv above; the project keeps its **host
  absolute path**, so model paths and ACL stay coherent.
- `Executor.run` — a failed probe returns `bwrap unavailable: …` and never
  executes on the host.
- `Executor.describe` — `bwrap (project rw, net shared, host ro; no cgroup cap (v1))`.
- `check_shell_approval(executor_mode in ("container", "bwrap"))` — gate skipped.
- `CommandContext.create_executor` — carries the policy to the loop and the
  ToolRegistry.
- `resolve_preset` / `agentic_active_tools` / `strict_path_confinement` — the
  four projections (intent, surface, approval, mounts) share one resolver.
- `tests/test_agentic.py::TestBwrapSandboxConfig` — offline argv/probe/gate/CLI.
- `tests/test_agentic.py::TestBwrapSandboxLive` — real writes, EROFS, DNS,
  network, gcc (opt-in: `RUN_BWRAP_E2E=1`).

## 7. Test commands

```bash
# offline (always)
python3 -m unittest tests.test_agentic.TestBwrapSandboxConfig -v
python3 -m unittest tests.test_agentic.TestAgenticPresets -v

# real sandbox (Linux with bwrap; skipped otherwise)
RUN_BWRAP_E2E=1 python3 -m unittest tests.test_agentic.TestBwrapSandboxLive -v
```

## 8. Known open decisions (beyond the MVP)

- cgroup caps via `systemd-run` (v2) vs documenting the absence (v1).
- `--disable-userns` as a default hardening (test first; bwrap 0.11 supports it).
- `exec` increment 2: directory ACL write grants become per-command `--bind`
  rw subtrees, and file grants stay with the file tools.
- Per-tool approval map (`allow`/`ask`/`gate`) and the ephemeral `--task`
  grant, which would complete §7.6 of `TODO.md`.
- Persistent user presets (`~/.ollamaquery.d/presets.json`).
