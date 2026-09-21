# Agentic Step Timeout & Interrupt Handling — Design

Status: implemented (`ollamaquery2.py`, v0.2.6)
Reference: `TODO_agentic_analysis_v0.2.5.md` (root-cause analysis P1–P4)
Discovery: the "read this whole project" / "write a corresponding nc.c" session
(Qwen3.8-27B-UD-IQ2_M @ llama.cpp, 2026-08-27)

---

## 1. High Design

### 1.1 Problem

The agentic ReAct loop (`run_agentic_query`) runs the model on a per-step basis with a
wall-clock budget. Two harness-level defects surfaced in practice:

| Symptom (observed) | Root cause |
|--------------------|------------|
| After `^C` the prompt returned, but leftover `<thinking>`/tokens kept streaming into the terminal and interleaved with the user's next input | `^C` only interrupted the **main** thread; the model call ran in a **daemon worker** thread that kept reading the HTTP stream and writing feedback (`on_chunk`) — nothing had cancelled it |
| The next query re-explored the whole project ("There's no tool result in between.") | the history-merge of completed ReAct turns lived *after* the loop, so `^C` skipped it and `self.messages` kept only `[system, user]` |
| A long-thinking step was killed by the 120s timer right after `</thinking>`, then retried at the *same* 120s and failed again | the step budget was a pure wall-clock `join()` with no notion of "the model is still producing tokens" |

The goal: a harness that (a) aborts the in-flight request on `^C` — not just the display —
and hands control back cleanly, (b) never erases completed work on interrupt, and (c) treats
"healthy but slow" generation differently from "stuck", extending the budget while tokens are
flowing and telling the model how much budget it has left so it self-throttles.

### 1.2 User-facing behaviour / ease of use

All knobs live on `CommandContext` and are tunable at runtime:

| Setting | Default | Effect |
|---------|---------|--------|
| `agentic_step_timeout` | 300 | Initial per-step wall-clock budget. `/agentic timeout <s>` |
| `agentic_timeout_max` | 600 | Ceiling for budget extension (backoff 300→600); doubles when reached (600→1200→…), hard-capped at `AGENTIC_TIMEOUT_MAX_CEILING` (7200). |
| `agentic_progress_grace` | 15 | Seconds of allowed silence before a *running* step is treated as stalled (i.e. not extended). |
| `agentic_max_iterations` | 50 | ReAct loop step cap. `/agentic iterations <n>` |

There is nothing new for the user to learn: `^C` behaves the way they expect (stop + keep the
conversation), and a timed-out step now resumes with a transparent note about speed/budget
instead of silently re-deriving.

---

## 2. Design Decisions

### 2.1 Aborting an in-flight generation on `^C`

**Why a daemon thread?** `query_sync_stream` is a blocking HTTP read, and the ReAct loop needs a
per-step wall-clock budget. `_call_with_timeout` (`:6512`) runs the call in a
`threading.Thread(daemon=True)` and `join(timeout_sec)`s it. The worker is not killable
mid-run, so cancellation must be *cooperative*.

**Cancel token.** Each step builds `step_cancel = {"event": threading.Event(), "close": None}`
and threads it into `query_sync_stream` as a `cancel` kwarg (`:7249-7270`). The worker registers
`cancel["close"] = response.close` once the HTTP response is acquired, and checks
`cancel["event"].is_set()` per chunk. On `^C` (or on the timeout path) the main thread calls
`_abort_step()` (`:7310`):

1. sets the event (so `on_chunk`/loop stops);
2. calls `response.close()` from the main thread — the worker's blocked `.read()` raises, is
   caught, and returns a quiet `{"error": {"message": "cancelled", ...}}` (no
   `[ERROR] ... sync stream failed` spam) instead of an error banner.

This stops the server from generating and frees the single-slot llama.cpp request queue, and
prevents the leftover streaming. `finalize_step()` is also called on `^C` so an open `<thinking>`
block is closed and `on_chunk` stops writing immediately.

### 2.2 Preserving completed work on interrupt (`_persist_agentic_history`, `:7204`)

The ReAct loop builds a *local* `messages` array seeded as
`[agentic system] + self.messages[1:]`; genuinely new turns start at `agentic_seed_len`.
Merging them back into `self.messages` is now a helper, called from **both** the normal
completion path and a new `except KeyboardInterrupt` in `run_agentic_query` — so a `^C`
persists every completed tool turn before control returns. The interrupted step's partial
output is correctly not persisted (it is never in `messages` yet). `_system_nudge` steering
messages are excluded so loop-internal nudges don't pile into context.

### 2.3 Timeout escalation policy (`_handle_agentic_timeout`, `:6698`)

Classified by what the model was doing:

| State | Condition | Policy |
|-------|-----------|--------|
| **C** | Last executed tool was destructive (`run_command`/`patch`/`edit_file`) **and** a tool call was pending in the timed-out generation | Abort — a retry could re-execute it |
| **C′** | Destructive last tool, but only prose (no tool call pending) | Tool already completed; continue via State B backoff |
| **A** | No tool executed yet | **Extend** `min(timeout*2, max)` if the model is still generating; else same-budget retry once, abort after two |
| **B** | A tool already ran | Exponential backoff 300→600; when the ceiling is reached it doubles `agentic_timeout_max` (600→1200→…, hard-capped at 7200) instead of aborting; aborts at the hard ceiling |

On any continue, a nudge message is appended so the model resumes rather than restarts.

### 2.4 Progress-aware budget (F3)

A fixed larger timeout wastes seconds on genuinely stuck steps, so the extension is
**progress-based**, not just "bigger timeout":

- `_make_agentic_step_feedback` (`:6532`) tracks `last_activity` in its buffer, updated on every
  chunk (`on_chunk`), and exposes the accumulated `thought`/`content`.
- On timeout, `run_agentic_query` computes `elapsed = now - step_start`,
  `partial_tokens = estimate_tokens(thought + content)`, and `idle_sec = now - last_activity`.
- `_handle_agentic_timeout` derives `tok_per_sec = partial_tokens / elapsed` and
  `making_progress = (idle_sec <= agentic_progress_grace) and (tok_per_sec > 0)`.
- State A extends the budget **only** while `making_progress`; a silent step falls through to the
  conservative retry/abort. This gives a 200s single-think step room while still catching a
  silent loop quickly.

### 2.5 Budget-aware nudge (F3 companion)

The nudge (`_timeout_nudge`, `:6634`) now tells the model its concrete generation metrics so it
can self-throttle instead of re-deriving and hitting the wall again:

```
Generation metrics:
- ~12.5 tokens/second
- cut off after 300s (step timeout)
- new budget: 600s (~3000 tokens at your current speed)
Keep your total thinking + output within that budget — finish as soon as you have enough to
act, and prefer a tool call or final answer over re-deriving your full analysis.
```

### 2.6 Nudge echoes the TAIL of the reasoning (F4)

A resumed step should continue from *where it was cut off*, not from the start. `_timeout_nudge`
caps the echoed reasoning from the end (`_cap_tail`, `[-4000:]`, prefixed
`"[earlier reasoning omitted]"`) instead of the head, for both `thought` and `content`.

---

## 3. Trade-offs & rationale

- **`estimate_tokens()` heuristic, not the server tokenizer, for tok/s.** The heuristic is cheap,
  has no network round-trip, and is only used to steer the model's *length* — precision is
  unnecessary. `tok_per_sec` is therefore approximate.
- **Wall-clock + progress, not a single big timeout.** A fixed large budget penalizes stuck
  steps; the progress check extends *only* healthy generation. This is why `agentic_progress_grace`
  exists.
- **`response.close()` may not interrupt the socket on all platforms.** Closing the
  `http.client.HTTPResponse` closes the buffered file wrapper; a blocked read on the underlying
  socket can in theory survive. The event flag + `finalize_step()` guarantee the *display* stops
  regardless; the socket close is best-effort prompt cleanup (verified on the local llama.cpp
  stack).
- **`^C` kills the main thread, not the worker.** `except KeyboardInterrupt` in `run_agentic_query`
  persists history and re-raises so `ChatLoop.run` prints `[Interrupted]` and resumes the prompt.
  The worker is cooperative-cancelled, never force-killed.
- **History merge ordering.** Inserting at `agentic_seed_len` (right after the current user query,
  before any finalize-appended messages) keeps chronological order `user → tool work → final answer`
  and avoids the old duplicate-history bug.

## 4. Function & call-site map

| Function | Line | Role |
|----------|------|------|
| `_call_with_timeout` | 6512 | Daemon-thread + `join(timeout)` step executor |
| `_make_agentic_step_feedback` | 6532 | Live `on_chunk`/`finalize` + buffer (`last_activity`, `thought`, `content`) |
| `_timeout_nudge` | 6634 | Timeout-continuation message (tail reasoning + metrics) |
| `_handle_agentic_timeout` | 6698 | State A/B/C escalation + budget extension + nudge |
| `_persist_agentic_history` | 7204 | Merge local ReAct turns into `self.messages` |
| `run_agentic_query` | 7249 | ReAct loop; `_abort_step()` at 7310; `except KeyboardInterrupt` |
| `query_sync_stream` | ~4505 | Streaming aggregator; `cancel` kwarg + quiet-cancel |
| `agentic_step_timeout` / `_max` / `_progress_grace` | 1043–1045 | Knobs |

## 5. Known limitations & future work

- **F5 is detection + steer, not a hard server-side cut.** `agentic_max_thinking_tokens`
  (default 2048, 0 = disabled) flags a step that crosses the thinking cap and tells the model to
  be more concise via the nudge — it cannot truly stop a server-side generation mid-think the way
  `^C` aborts a client stream. Live thinking is still streamed and F4's tail-cap still bounds what
  is echoed.
- **`tok_per_sec` is heuristic.** Derived from `estimate_tokens()` (char/4), not the server
  tokenizer — approximate, but sufficient for length steering.
- **Interrupt-path testing is via mocked `_call_with_timeout`.** The real ^C (SIGINT delivered
  to the main thread during `thread.join()`) is exercised manually; the offline unit test drives
  the same code path by raising `KeyboardInterrupt` from a mocked step executor.

### 5.1 Preserving the interrupted step without polluting context (proposal, to think about)

On a step timeout (or `^C`) the model is cut mid-generation. Today `_timeout_nudge`
echoes up to 4000 chars of the *tail* of the reasoning (and partial content) inline
into a **user** message. Three gaps:

1. **Context pollution / cost.** Every timeout re-injects up to ~4K chars (~1K tokens)
   of the model's own monologue into the conversation; repeated timeouts compound.
2. **Restart, not resume.** The echo is a user message with no assistant prefix, so
   instruction-tuned models often re-derive from scratch rather than continue.
3. **Lost in-flight tool call.** Native `tool_calls` deltas accumulate only in
   `_aggregate_sync_stream` (`tool_call_index`, `ollamaquery2.py:5525`), inside the
   worker thread. On timeout `_call_with_timeout` returns `None` and those partial
   calls are discarded — a cut-off `write_file`/`run_command` emission is gone.

**Idea — spill to a scratch file, point the model at it.** On timeout/interrupt, write
the accumulated `thought` + `content` (+ partial tool-call args) to a per-session
scratch file (e.g. `~/.ollamaquery.d/agentic/<session>/interrupt-<step>.md`, reaped by
the existing `AGENTIC_LOG_RETENTION_DAYS` cleanup). Replace the inline echo in
`_timeout_nudge` with a short pointer: *"Your interrupted reasoning was saved to
`<path>`. Read it if you need it, then continue."* — a few tokens instead of ~1K, and
no monologue in the history. The model can `read_file` it on demand (subject to the
Path ACL) or ignore it.

Three composable mechanisms:

- **A. Spill to file (harness).** Write the partial text; nudge only points to it.
  Pros: cheap, no context pollution, model reads on demand. Cons: an extra `read_file`
  round-trip; ACL/retention/secret-leakage to consider; the model may not read it.
- **B. Self-status tool (model).** A new `remember`/`set_status` tool the model calls to
  record its intent/plan; stored to the same scratch file. Pros: model-authored *intent*
  that is concise and survives compaction and `/switchmodel`. Cons: only helps if the
  model calls it at the right moment (unreliable mid-thought).
- **C. Partial tool-call capture (harness, root cause).** Publish the aggregator's
  `tool_call_index` into the step/cancel state so the nudge can name the in-flight call
  (and, if the accumulated JSON parses, execute it or offer it for confirmation).
  Pros: deterministic, no model cooperation, recovers the lost call. Cons: plumbing the
  partial state out of the worker thread; partial JSON may be invalid.

Suggested order: **C → A → B**. C fixes the lost call, A removes the nudge's context
cost, B adds model intent that survives a full restart. Open questions: scratch location
(session dir vs workspace), Path ACL implications, secret redaction, retention/cleanup,
and whether the same scratch file should back `/compact` summaries and `/switchmodel`
hand-off.

## 6. Tests

```bash
python3 -m unittest tests.test_agentic_timeout -v      # 38 tests (States A/B/C, tail-cap, metrics, F5, nudge exclusion, max ceiling, staticmethod guard)
python3 -m unittest tests.test_agentic.TestReActLoopUnit -v   # 9 tests (incl. interrupt-persist + quiet-cancel)
# plus the offline suites that guard the touched code:
python3 -m unittest tests.test_modifications -v
```