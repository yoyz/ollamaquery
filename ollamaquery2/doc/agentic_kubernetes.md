# Agentic Kubernetes / OpenShift Cluster Consultation — Design

Status: implemented (`ollamaquery2.py`, v0.2.12)
Reference: design discussion (v0.2.11 dev session, 2026-10-02)
Scope: read-only cluster consultation, online (oc/kubectl) + offline (omc must-gather)

---

## 1. Goal

Let the agentic ReAct loop **consult** a cluster — resource state, logs, describe,
events — without ever being able to **mutate** it. Three backends (`oc`, `kubectl`,
`omc`), two platforms, one mental model:

| Mode     | Platform    | Backend          | Data source                 | Query command                  |
|----------|-------------|------------------|-----------------------------|--------------------------------|
| Online   | openshift   | `oc`             | Live cluster via kubeconfig | `oc get`, `oc logs`, ...       |
| Online   | kubernetes  | `kubectl`        | Live cluster via kubeconfig | `kubectl get`, `kubectl logs`, ... |
| Offline  | openshift   | `omc`            | Local must-gather directory | `omc get`, `omc logs`, ...     |

The distinction is two arguments, not a separate mode: `offline_dir` (given →
offline must-gather; omitted → online against the current kubeconfig context) and
`cluster_type` (`omc` | `openshift` | `kubernetes` | `auto`) which selects the
consultation backend and, more importantly, the *resource vocabulary* — OpenShift
has `projects`, `routes`, `buildconfigs`, `deploymentconfigs`; vanilla Kubernetes
does not. In `auto` (the default) the `KUBECONFIG` env var is the online/offline
switch: set → online; unset → offline when omc is configured (`omc` binary +
`~/.omc/omc.json` present), else online via the default `~/.kube/config`.

## 2. Verb scope

Only read-only verbs. The classic troubleshooting trio plus context awareness:

| Verb             | Online (openshift)                 | Online (kubernetes)            | Offline              | Read-only |
|------------------|------------------------------------|--------------------------------|----------------------|-----------|
| get              | `oc get`                           | `kubectl get`                  | `omc get`            | yes       |
| logs             | `oc logs` (tail)                   | `kubectl logs` (tail)          | `omc logs`           | yes       |
| describe         | `oc describe`                      | `kubectl describe`             | `omc describe`       | yes       |
| events           | `oc get events`                    | `kubectl get events`           | `omc get events`     | yes       |
| current-context  | `oc config current-context`        | `kubectl config current-context` | n/a (dir is the ctx) | yes       |

Notes:
- `cluster_type` selects the binary *and* the resource vocabulary the model should
  use. `oc` also works against vanilla clusters and `kubectl` against most
  OpenShift resources, so the parameter is guidance + binary preference, not a
  hard capability gate — except offline: `omc` is OpenShift-only (must-gather is
  an OpenShift concept), so `offline_dir` + `cluster_type=kubernetes` is invalid.
- `events` is subsumed by `get` at the shell-gate level (`oc get *` matches
  `oc get events`) but is a first-class `verb` on the tool so the model doesn't
  have to know the resource spelling.
- `current-context` answers "which cluster am I on" — the model must confirm the
  context before querying, and every online result should echo it (see §5).

Explicitly NOT in scope (never implemented, never allow-listed):

| Verdict | Verdict reason |
|---------|----------------|
| apply / create / delete / edit | mutates cluster state |
| exec / port-forward | interactive / lateral movement |
| adm (must-gather, inspect) | heavy / privileged |
| login / config set-context / use-context | mutates `~/.kube/config` credentials |

## 3. Two complementary layers

The tool and the shell gate are **both** required; they constrain different things:

| Layer              | Constrains                                                              | Failure mode if skipped                     |
|--------------------|-------------------------------------------------------------------------|---------------------------------------------|
| Native tool        | *which verbs exist* for the model (structural boundary)                 | model can `run_command` anything            |
| Shell allow-list   | *those verbs run non-interactively* through `Executor.run` → gate      | every tool call re-prompts                  |

The handler executes via `Executor.run()` → `check_shell_approval`, so the read
verbs must be allow-listed or the tool prompts on each call. Keep the verb set in
one constant so the tool and the allow-list cannot drift.

## 4. Shell gate changes (`DEFAULT_SHELL_PERMISSION`)

```python
# Online — read-only verbs only
"oc get *": "allow",            # also covers `oc get events`
"oc logs *": "allow",
"oc describe *": "allow",
"oc config current-context": "allow",
"kubectl get *": "allow",
"kubectl logs *": "allow",
"kubectl describe *": "allow",
"kubectl config current-context": "allow",
# Offline — local must-gather reads
"omc use *": "allow",           # select a local must-gather dir (local config write)
"omc get *": "allow",
"omc logs *": "allow",
"omc describe *": "allow",
```

Do **not** add bare `oc *` / `kubectl *` / `omc *`. Mutating verbs are
**hard-denied** (generated `_CLUSTER_MUTATING_VERBS` rules: `delete`, `apply`,
`create`, `edit`, `exec`, `port-forward`, `adm`, `login`, `config
set/use-context`, ...). This is stricter than the "default ask" of v0.2.11's
plan: the CWD-leniency auto-allow treats bare operands (`pod`, `x`) as
CWD-relative paths, so `oc delete pod x` would silently pass the gate — hard-deny
keeps the read-only guarantee airtight. A session allow rule (evaluated after the
defaults) can still re-enable a specific mutating command if ever needed. The
per-component breakdown additionally keeps pipelines like `oc get -o json | oc
apply -f -` blocked on the `oc apply` node.

Supporting edits:
- `_SHELL_ARITY`: add `oc`, `kubectl get/logs/describe/config`, `omc` entries so an
  "always" approval stores a tight pattern (`oc get *`) instead of a broad `oc *`.
- `_shell_operand_realpath` currently skips tokens starting with `-`, so a
  flag-value path like `omc --must-gather-dir=/tmp/mg` is **not** ACL-scanned.
  Acceptable for the allow-listed read verbs (local read), but flag-value paths
  landing in `$HOME` bypass the guard — document, or handle in §7.

## 5. Native tool spec

One tool, not five — keeps the schema lean (fits `openai_tools_spec` + the inline
prompt block with no extra surface). `kubernetes_cluster_query` is deliberately
specific about the platform: "cluster" alone is ambiguous (OpenShift, EKS, AKS,
GKE, k3s, ... are all clusters), and the resource vocabulary differs per platform.

```
kubernetes_cluster_query
  cluster_type   enum: omc | openshift | kubernetes | auto   (default: auto)
  verb           enum: get | logs | describe | events | current-context   (default: get)
  resource       string   (required for get/logs/describe)
  name           string   (optional, resource instance)
  namespace      string   (optional, `-n`)
  selector       string   (optional, `-l` label selector; get/logs/events, rejected for describe)
  field_selector string   (optional, `--field-selector`; get only)
  container      string   (optional, `-c` container name; logs only)
  offline_dir    string   (optional; must-gather dir — defaults to the one in ~/.omc/omc.json)
  tail           integer  (logs only, default 50)
  output         enum: json | yaml | wide | custom-columns | jsonpath   (default: json)
  fields         string   (required for custom-columns / jsonpath, e.g.
                           NAME:.metadata.name or {.items[*].metadata.name})
```

Handler behaviour:
1. Re-validate `verb` against a hard whitelist (defense in depth — a hallucinated
   verb is rejected, never forwarded to the shell).
2. Resolve `cluster_type`:
   - `omc` → offline (omc).
   - `auto` → **online** when the `KUBECONFIG` env var is set; **offline** (omc) when it
     is not set AND `omc` binary + `~/.omc/omc.json` are present; otherwise online via
     the default `~/.kube/config` — `oc` binary in `PATH` → openshift, else kubernetes.
     All checks are purely local and need no reachable cluster.
3. Dispatch:
   - omc → point omc at the must-gather with `omc use <dir>` (only when an explicit
     `offline_dir` is given or the config read yields one), then `omc <verb args>`.
     The must-gather dir = explicit `offline_dir`, else read from `~/.omc/omc.json`;
     error when neither is available. omc only reads OpenShift must-gathers. (Note:
     omc has no per-query `--must-gather-dir` flag — selection is `omc use`.)
   - `openshift` → `oc <verb> ...`
   - `kubernetes` → `kubectl <verb> ...`
4. Append `-n <namespace>`, `-l <selector>` (get/logs), `--field-selector=<s>` (get),
   `-c <container>` (logs), `--tail=N` (logs), `-o <output>`. Unsupported
   verb/filter combinations raise a `ValueError` with a corrective message.
5. Shell-quote EVERY token (`shlex.quote`) before joining into the command
   string. The command runs with `shell=True`, so an unquoted LLM-controlled
   value could smuggle a second command (`kubectl get pods; touch x`,
   `pods $(id)` — both executed before the quoting fix). Quoting turns injected
   values into literal operands the backend rejects. Safe tokens pass through
   unchanged, so the allow-listed gate patterns still match.
6. Run via `Executor.run()` (goes through the shell gate; allow-listed → no prompt).
7. Prepend an observation header: `platform: <omc|openshift|kubernetes> |
   context: <current-context> | must-gather: <dir>` so the model (and logs) always
   record which cluster and vocabulary were consulted.
8. Rely on the existing observation cap (`_cap_tool_observation`, ~4K chars);
   `tail=50` keeps raw logs bounded at the source.

## 6. Security posture

Accepted for controlled, local use (documented decision — not a bug to fix silently):

| Risk                                | Mitigation / stance                                        |
|-------------------------------------|------------------------------------------------------------|
| `oc get secret -o yaml` leaks secret values | Accepted: controlled local context + trusted model. No cluster-object ACL exists (only filesystem Path ACL). Deferred (see §7). |
| kubeconfig credentials (`~/.kube/config`) | Already guarded: home-dotfile `ask` via Path ACL, unchanged. |
| Mutating verbs                      | Unreachable structurally (tool whitelist) + not allow-listed (gate). |
| Shell injection via tool args       | Fixed: every token is `shlex.quote`d before joining into the `shell=True` command — `;`, `$()`, backticks become literal operands the backend rejects. |
| Pipeline escapes (`\| oc apply`)     | Per-component gate keeps non-allow-listed nodes at `ask`.  |
| Offline dir outside CWD             | `omc` read verbs are allow-listed; flag-value path not ACL-scanned (accepted, documented). |

## 7. Open questions (decide before implementation)

- `cluster_type` default: `auto`. DECIDED: a `KUBECONFIG` env var always means online;
  without it, offline (omc) when `omc` binary + `~/.omc/omc.json` present, else online
  via the default `~/.kube/config`. The env var is the sole online/offline
  discriminator — a default `~/.kube/config` with no env var does not force online.
- `offline_dir` given but `cluster_type` resolved to openshift/kubernetes: DECIDED —
  ignored (only the omc branch consumes it; online queries never take a dir).
- omc must-gather dir source: DECIDED — explicit `offline_dir` wins, else the path
  stored in `~/.omc/omc.json`; no `must-gather/*` layout auto-detection in v0.2.12.
- `current-context`: separate `verb` vs always baked into the result header (lean:
  bake it in; the verb is a convenience).
- Output shaping: pass `-o json` raw vs parse/pretty-slice in the handler.
- `events` verb vs reuse `get` (lean: keep as a verb for model ergonomics).
- Auth introspection (`oc whoami` / `oc auth can-i --list`) in scope or not (lean: out
  for the first cut).
- Run via `Executor.run` (gate, consistent with everything else) vs a dedicated
  subprocess bypass (not recommended — breaks the "gate everything" invariant).

## 8. Plan-mode integration

`kubernetes_cluster_query` is read-only by construction → add it to
`PLAN_MODE_TOOLS` so `/agentic plan` (read-only planner) also gets safe cluster
consultation for free.

## 9. Out of scope (explicitly deferred)

- Mutating verbs and cluster writes of any kind.
- Cluster-object ACL / secret redaction (needs an object-sensitivity layer, not the
  filesystem Path ACL).
- Persistent must-gather download/extraction management (must-gather tarballs are
  already handled today via the allow-listed `tar`).
- Multi-cluster context management beyond reading the active one.

## 10. Known limitations

- **Large JSON results are truncated, not paged.** `get -o json` / `-o yaml` on a
  big cluster can return megabytes. The handler captures the full output, but the
  ReAct loop then applies the generic observation cap (`_cap_tool_observation`,
  ~4K chars, less as the context fills), so the model only sees the head of the
  document — a hard character cut that is no longer valid JSON. Unlike
  `read_file`, there is no `next` continuation pointer, so the tool cannot page
  through a large result. Fields deep in the document (e.g. PV/PVC
  capacity/status) are unreachable.
Status (v0.2.12):
   1. ~~Pagination~~ — superseded by the temp-file spill below (the tool cannot
      ask the server for a JSON chunk/offset).
   2. **Temp-file spill — IMPLEMENTED.** Results over `CLUSTER_SPILL_THRESHOLD`
      (~3500 chars) are written to per-query files
      `~/.ollamaquery.d/spill/cluster_result_<verb>_<resource>.json` (scoped by
      verb + resource so one query never clobbers another being paged; removed at
      process exit via atexit). The tool returns a PURE POINTER — size in
      chars/lines + a `read_file(file_path="...")` instruction — with NO content
      inline, so the payload is not fed into context twice (it only appears when
      the model pages it via `read_file`, which is cap-exempt and carries a `next`
      offset). The handler adds a session Path-ACL read-allow rule for the spill
      dir so `read_file` doesn't prompt on the home-dotfile path; the observation
      hands the model the absolute path so the home-explicitness gate passes.
   3. **`custom-columns` / `jsonpath` output — IMPLEMENTED.** `output='custom-columns'`
      / `output='jsonpath'` with a `fields` arg let the model request exactly the
      fields it needs (`-o custom-columns=NAME:.metadata.name,...` /
      `-o jsonpath={.items[*].metadata.name}`) so results are small by
      construction and never spill.
  Until then, the mitigation is scoping: the model should pass `namespace`, a
  resource `name`, label selectors, or use `output='wide'` for large surfaces,
  and `logs` are already bounded by `--tail`.