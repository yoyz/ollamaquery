# Similar Projects

To find alternatives to ollamaquery2.py, it helps to look at the specific combination of goals it tries to achieve: local-first inference compatibility (Ollama, Llama.cpp, LM Studio, Strata), terminal-centric execution, and an autonomous ReAct agent loop that can read, write, and patch files.

While ollamaquery2.py is unique in its strict commitment to a zero-dependency, single-file script, several mature open-source tools achieve similar goals with much higher parsing robustness. They generally fall into two categories.

Last reviewed: 2026-11-22 (ollamaquery2 0.5.0). The notes below were checked against the projects' documentation and, for the load-bearing claims (Aider's repo map, gptme's PTC dispatch and `hashline_edit`), against their source on `main`/`master` at that date. These projects move quickly; re-check before copying an implementation detail.

## Category 1: Full Agentic Coding Assistants (the "agentic mode" alternatives)

### 1. Aider

**What it is:** a terminal-based AI pair programmer. Files are explicitly added to the chat, Aider edits them with a model-chosen edit grammar, and it git-commits each change.

**How it matches:** like `apply_patch`/`edit_file`, Aider specializes in turning a feature request or a bug report into edits in the local workspace. It supports Ollama and any OpenAI-compatible endpoint.

**Strengths:**

- **Repository map.** A tree-sitter symbol index (definitions and references) is ranked with a PageRank over the file/symbol graph, then binary-searched to fit a token budget (1k by default, expanded when no files are in the chat). The model gets classes, functions and signatures from the whole repo without reading it. This is the single most transferable idea.
- **Edit formats chosen per model.** `whole`, `diff` (SEARCH/REPLACE blocks), `diff-fenced`, `udiff` (simplified unified diff), and the streamlined `editor-diff`/`editor-whole` used by architect mode. Aider ships defaults per model because models differ sharply in which grammar they can follow.
- **Architect/editor split.** One model proposes the change in prose, a second (or the same) model turns it into syntactic edits. Useful when a reasoning model is strong at planning but weak at file surgery.
- **Git integration.** Commits dirty files before editing, then commits each AI change with a generated message, plus `/undo`, `/diff`, `/commit`, `/git`. Attribution can be toggled (`--no-attribute-author`, co-authored-by trailer, ...).
- **Lint/test loop.** `--lint-cmd` / `--auto-lint` and `--test-cmd` / `--auto-test` run after edits, and a non-zero exit feeds the errors back to the model for repair. `/run` and `/test` do the same on demand.
- Prompt caching, watch mode / IDE integration, and a public code-editing leaderboard.

**Weaknesses:**

- Heavy dependency tree (tree-sitter, networkx, diskcache, pygments, ...) and a Git repository is assumed for the full workflow.
- Auto-commit is opinionated and also commits pre-existing dirty files; it cannot be used in a tree where commits are reserved for another machine.
- No sandbox or shell gate: the model edits files and runs lint/test commands, but the safety model is "review the diff + git undo", not "bounded tool surface".
- Not a general agent: no autonomous multi-step ReAct loop, no browser/vision, no subagents.

**License:** Apache-2.0.

### 2. gptme

**What it is:** a provider-agnostic personal agent that runs in any terminal (laptop, SSH, tmux, CI), with local tools and a first-class story for persistent autonomous agents. One of the earliest agent CLIs (2023), still very active.

**How it matches:** a ReAct-style loop over shell, Python, file reads/edits, web and MCP; local models via Ollama or any OpenAI-compatible server (`-m local/<model>`). It is the closest match to ollamaquery2's "chat + agentic tools + local backend" shape, but built as a general platform rather than a single script.

**Strengths:**

- **Programmatic Tool Calling (PTC).** The default protocol is a fenced code block whose language tag is the tool name (`python`, `shell`, `save`, `patch`, `hashline_edit`); gptme routes the block body to that tool. No JSON schemas are sent by default and no JSON tool-call envelope is parsed. Provider-native tool mode exists as an option. The project argues this is shorter and more robust under long-context "context rot", citing arXiv:2608.06370 (PTC matched or beat JSON on 11/14 models in that study); the benchmark is their argument, not an independent result, but the mechanical benefit for file content is real: a file body is a code block, not a JSON-escaped string.
- **Stateful tools.** The shell tool keeps a persistent bash session (`cd`, env vars, background jobs survive between calls) and the Python tool runs in a persistent IPython kernel. ollamaquery2's `Executor` is one process per command.
- **Snapshot-anchored edits (`hashline_edit`).** `read` returns `[PATH#TAG]`; an edit block must repeat that tag and list `PUT`/`CUT` line operations. If the file changed since the read, the edit is rejected, then recovered with a 3-way merge via `git merge-file`; the final write is atomic (temp file + `os.replace`). This closes the stale-edit / partial-write class directly.
- **Lessons and skills.** Markdown files with YAML frontmatter (`match: {keywords: [...], tools: [...]}`) are indexed from `~/.config/gptme/lessons`, `~/.agents/lessons`, `./lessons`, `./.gptme/lessons`, and `.cursor` rules, auto-injected when the conversation matches, with a session cap (20 by default) and `/lesson list|search|show|refresh`. Skills follow the Anthropic `SKILL.md` folder format. A separate memory store is shared across harnesses.
- **Hooks and guardrails.** Pre/post tool execution, pre/post save and patch, session start/end, generation hooks. Guardrails can block destructive commands and reads of common secret paths, and screen tool output for prompt-injection patterns.
- **Context compression with recovery.** Budget is `min(0.9 × window, window − max_output − headroom, 256k)`, anchored on the provider-reported input count and invalidated by edits/model changes. Two strategies: `trim` (stub stale tool outputs, cap huge results, deterministic) and `summarize` (structured checkpoint: objective, decisions, current state, open items, files to reload; keep the most recent ~20k tokens). Old tool results become `[result #N, M tokens]` references; `recall_result(N)` reads the original from a lossless master log and can page it. A 90% reminder asks the model to persist durable state at a safe boundary.
- **Tool allowlists with capability hints.** `read-only`, `destructive`, `idempotent`, `closed-world`; MCP annotations are mapped onto the same hints, and `--tools read-only` / `hint:read-only` build strict surfaces. Prompt text has conditional blocks so it never references a tool that is not loaded.
- Subagents (optionally in throwaway git worktrees), `todo`, `complete`, and `progress`/`clarify` between subagents, plus a server/web UI/desktop app and a systemd-based autonomous-agent template.

**Weaknesses:**

- Much larger than a single script: roughly 33k lines of core, 36k of tools and 13k of server at the time of writing, plus a plugin/agent ecosystem.
- The plugin and hook system means project configuration (`gptme.toml`, `context_cmd`, hook scripts) can execute code at startup; gptme documents this and tells users to review the file in untrusted repositories. That trust boundary is incompatible with a zero-dependency, single-file tool.
- PTC is a protocol fork. Moving ollamaquery2's inline-JSON path to PTC would change prompts, parsing, tests and model compatibility at once; it needs an eval before it is worth the rewrite.
- Subagents, computer use, browser and computer vision are out of scope for a script whose value is portability.

**License:** MIT.

### 3. Open Interpreter

**What it is:** a local, natural-language interface for the terminal that acts as an autonomous agent.

**How it matches:** it mirrors `run_python`/`run_command` on steroids: write code, execute it locally, read stdout/stderr, self-correct until the task is solved.

**Local LLM support:** a native `--local` flag bridges to Ollama or Llama.cpp.

**Strengths:**

- Can control browsers, process PDFs, manipulate datasets.
- Sandboxed execution environment.
- Self-correcting loop with stdout/stderr analysis.
- Broader scope than file editing.

**Weaknesses:**

- Very heavy dependency footprint.
- Overkill for simple chat or single-query tasks.
- Sandbox mode adds complexity.
- Slower startup due to dependency loading.

## Category 2: Pipeline & Chat CLIs (the "standard mode" alternatives)

If you use ollamaquery2.py primarily to pipe files, fetch URLs, and maintain quick, lightweight terminal chat sessions without autonomous agent loops, these tools excel.

### 4. ShellGPT (sgpt)

**What it is:** a streamlined command-line productivity tool powered by LLMs, designed to sit inside standard shell workflows.

**How it matches:** single queries (`-I`), streaming output, shell history caching.

**Local LLM support:** any local OpenAI-compatible host URL.

**Strengths:**

- Superb shell integration (`--shell` generates platform-specific commands).
- Clean, minimal CLI; fast startup.
- Good for quick one-off queries.

**Weaknesses:**

- No autonomous agent/ReAct loop.
- Limited file inclusion.
- No multi-turn agentic mode.
- Requires pip install.

### 5. Fabric

**What it is:** a framework for hooking LLMs into command-line pipelines using structured markdown "patterns".

**How it matches:** takes terminal output, file contents or text files, runs them through an LLM, returns clean markdown/plain text to stdout.

**Local LLM support:** native Ollama flags.

**Strengths:**

- Crowdsourced, optimized system prompts ("patterns").
- Seamless pipeline integration (`cat log | fabric --pattern summarize`).
- Wide variety of pre-built patterns.
- Clean output formatting.

**Weaknesses:**

- No agentic/tool-use capabilities.
- Requires pattern directory setup.
- Less suitable for interactive multi-turn conversations.
- No file editing or code execution tools.

## Comparison Table

| Dimension      | ollamaquery2.py                    | Aider                                | gptme                                   |
| -------------- | ---------------------------------- | ------------------------------------ | --------------------------------------- |
| Positioning    | local-first chat + ReAct agent     | git-centric pair programmer          | general agent, persistent autonomy      |
| Form           | one 24.6k-line script, stdlib only | Python package, many dependencies    | Python package + server/UI              |
| Tool protocol  | JSON inline or native `tools`      | none; edit grammars                  | PTC code blocks; native tools optional  |
| File editing   | write/edit/patch/apply_patch       | whole, SEARCH/REPLACE, udiff         | save, patch, hashline, patch_many       |
| Context        | compaction, explicit budget        | repo map + PageRank                  | trim/summarize + checkpoint + recall    |
| Safety model   | shell gate, Path ACL, bwrap        | review diff + git undo               | approvals, hints, guardrails            |
| Git            | read-only (`git_inspect`)          | auto-commit, `/undo`, `/diff`        | autocommit/precommit tools, worktrees   |
| Extensibility  | monolith + MCP                     | YAML config                          | plugins, hooks, lessons, skills, memory |
| Autonomy       | ReAct + plan mode                  | chat modes only                      | subagents, services, run loops          |
| Dependencies   | zero                               | heavy                                | heavy                                   |

## What Is Worth Borrowing

Ordered by value/effort for this codebase, not by how prominent the feature is upstream.

### 1. SEARCH/REPLACE edit tool (Aider)

A `replace` tool taking one or more fenced blocks:

> ```text
> <<<<<<< SEARCH
> from flask import Flask
> =======
> import math
> from flask import Flask
> >>>>>>> REPLACE
> ```

Smaller than a unified diff for a one-line change (no hunk headers or line numbers), and — most importantly — a failed match can return the closest candidate (via `difflib`) instead of a bare "did not apply". It belongs next to `edit_file`/`apply_patch` in `AGENTIC_TOOL_DEFS`, with its own prompt block in the format registry. Whitespace-insensitive matching and an explicit failure reason are what make local models recover. Rough cost: 1–2 days including tests.

### 2. Repository map, stdlib-only (Aider)

Aider uses tree-sitter and networkx; this repo deliberately cannot (zero dependencies). A useful subset is available from the standard library: `ast` for Python definitions, a regex fallback for other files, cross-references counted across the tree, a small PageRank/weighted-sum over symbols, a token budget (1–2k) and an mtime cache. For this repository the payoff is concrete: `ollamaquery2.py` is 24.6k lines, and the agent currently has to `grep`/`read_file` its way to a class or method that a map could have shown in 1k tokens. Inject it in chat and `/agentic plan` only — never in `exec`, where the model can look things up itself. `doc/code_structure.md` is a static snapshot; a computed map cannot go stale. Rough cost: 2–3 days.

### 3. Post-edit check loop (Aider)

`--check-cmd` / `--test-cmd` run after every successful write, with a bounded number of retries and the command's output injected as an observation. `run_command` already lets the model do this, but the harness guarantees it, which removes the "looks done, does not compile" class of failures. Reuse `Executor` and the existing shell gate; the check is user-configured, so it is a normal approved command. Rough cost: ~1 day.

### 4. Lessons (gptme)

`AGENTS.md` is for agents working on this repository; it is not runtime context for the model in a session. A `lessons/` directory of markdown files with YAML frontmatter (`match: keywords/tools`), a per-session cap and `/lesson list|show|refresh` would let users teach conventions ("always run this test", "never touch this path", model-specific pitfalls) without editing the monolith or paying for the whole file in every prompt. The existing per-model prompt blocks stay; lessons are additive. Rough cost: 1–2 days.

### 5. Result references and recall (gptme)

The spill-to-file mechanism already covers cluster, MCP and `git_inspect` results. Generalizing it to a per-session result store with stable ids would let compaction replace an old tool result with `[result #N, M tokens]` instead of summarizing or dropping it, plus a read-only `recall_result(N)` tool (with `start_char` paging). This preserves exact evidence across compaction without keeping it in context. It also subsumes the one-off spill paths. Rough cost: 3–5 days.

### 6. A structured 90% checkpoint (gptme)

The near-full nudge currently reports the budget and asks the model to wrap up. gptme's version asks for a durable checkpoint at a safe sub-task boundary: objective, decisions, current state, next move, files to reload. That is a prompt-only change in `_near_full_nudge_message`, and it makes compaction lossy in a controlled way rather than an accident. Rough cost: hours.

### 7. Snapshot-anchored edits (gptme, alternative to #1)

`read_file` returns `[path#TAG]`; edits must carry the tag and are rejected if the file changed, with a 3-way `git merge-file` recovery and an atomic write. This is a stronger contract than SEARCH/REPLACE but a larger change, and ollamaquery2 is single-agent, so concurrent edits are rarer. Worth considering if the edit path is reworked for #1 anyway.

## What Not to Copy

- **Aider's auto-commit.** It commits pre-existing dirty files and then each AI change. In this repository commits belong to the second machine; a local commit is a divergent branch and blocks the fast-forward. If `/undo` is wanted, snapshot the file before each write into `~/.ollamaquery.d/backups` and restore it, without touching git.
- **gptme's plugin/hook system.** Project configuration that executes shell commands at startup is a code-execution primitive and a prompt-injection target. gptme documents the risk; a zero-dependency single file should not add it.
- **A wholesale PTC rewrite.** The argument is plausible, but it changes prompts, parsing, dispatch and model compatibility in one step. Prototype it for inline (non-native) models only, measure tool-call success on two or three local models, then decide.
- **Aider's dependency stack.** tree-sitter, networkx, diskcache and pygments are how Aider gets its map; the stdlib variant above gets most of the value for this codebase.
- **Subagents, computer use, browser automation, desktop/server UIs.** They are gptme's platform features, not agent-loop features, and they contradict the portability goal.

## Bottom Line

The three coding agents answer the same question differently — *where does the agent's knowledge of the code come from?* ollamaquery2 trusts the model to grep and read; Aider precomputes a ranked map; gptme gives the model a persistent shell and Python. On the protocol side, ollamaquery2 is the only one of the three that parses JSON for models without a native tool API, which is exactly where its tool-call failures concentrate. Aider's edit grammars and gptme's PTC are two answers to that; for local models, a SEARCH/REPLACE grammar plus a small repo map is the cheaper bet, and PTC is the experiment to run after that.

## Sources

- Aider: <https://aider.chat/docs/> — usage, chat modes, edit formats, repository map, git integration, lint/test.
- Aider source: <https://github.com/Aider-AI/aider> (`aider/repomap.py` for the tree-sitter + PageRank map; Apache-2.0).
- gptme: <https://gptme.org/docs/> — tools, lessons, context compression, security, "are we tiny".
- gptme PTC design: <https://gptme.org/docs/design/ptc-tool-interface.html>.
- gptme source: <https://github.com/gptme/gptme> (`gptme/tools/hashline_edit.py` for the snapshot + 3-way merge edit path; MIT).
