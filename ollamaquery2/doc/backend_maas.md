# MaaS backend — OpenShift AI / KServe in front of vLLM

Status: **implemented** (`0.4.4`, `-b maas`), verified offline; the gateway was
measured live during the investigation (2026-11-19). The measurements and the
reference wrapper that preceded this backend are archived outside git history;
this file is the durable reference, written from what the code actually does.

## 1. What the endpoint is

An OpenShift route in front of an `LLMInferenceService` (KServe), itself in
front of vLLM. It exposes exactly two OpenAI-compatible routes:

| Route                    | Notes                                                                 |
| ------------------------ | --------------------------------------------------------------------- |
| `GET /v1/models`         | `{"data":[{"id":…,"ready":true,"kind":…}],"object":"list"}`           |
| `POST /v1/chat/completions` | OpenAI request/response shape, including native `tool_calls`       |

Everything else (`/health`, `/metrics`, `/version`, `/tokenize`, `/slots`,
`/v1/completions`, `/v1/embeddings`, …) is a **404**. That single fact drives
several design choices below.

## 2. Configuration

| Setting              | How                                                                  |
| -------------------- | -------------------------------------------------------------------- |
| Origin               | `-H https://<route-host>` or `MAAS_HOST` (a trailing `/v1` is stripped) |
| Key                  | `MAAS_API_KEY`, or `MAAS_API_KEY_FILE` (a `0600` file, first line)   |
| Model                | `-m publishers/<publisher>/models/<name>`, or a short name (resolved) |
| Context window       | `--context-size <tokens>`, persisted for the session                 |
| Listing cache        | `MAAS_MODEL_TTL` seconds (default `300`; `0` disables the cache)     |

```bash
export MAAS_HOST=https://maas.example.internal
export MAAS_API_KEY_FILE="$HOME/.config/maas/token"

python3 ollamaquery2.py -b maas -l                       # full ids
python3 ollamaquery2.py -b maas -m glm-53-flash -I "hi"  # short name resolved
python3 ollamaquery2.py -b maas --context-size 32768 -c
```

`-k/--api-key` works but puts the key in `argv` (`ps`, shell history); prefer
the environment or the file. TLS verification stays on — the route serves a
valid certificate, and disabling it is never needed.

## 3. Failure classification

Only the listing is retried. The classification lives in
`maas_classify_status()` and is the single reading used by the gateway:

| Status                        | Decision   | Why                                                          |
| ----------------------------- | ---------- | ------------------------------------------------------------ |
| `200` with models             | success    | —                                                            |
| `200` with `{"data":[]}`      | retry      | Observed as a disguised rate limit; accepted on the last try |
| `401`                         | **fatal**  | Key rejected/expired — looping cannot help                   |
| `403`, `429`, `500`–`504`     | retry      | Rate limit / token-mint failures                             |
| transport error (no response) | retry      | DNS, timeout, reset                                          |
| any other (incl. `404`)       | error      | A `404` means a wrong origin or an incomplete model id       |

Backoff is exponential with a local jitter (`0.5 s` doubling, capped at `8 s`,
6 attempts) because the gateway sends **no** `Retry-After`. On exhaustion the
error names the last failure and the server's `exceptionCode`/`refId` when the
body carries them. If a cached listing exists, it is used instead of failing —
one stale inventory is better than no session.

**Chat is deliberately not retried**: replaying a chat request would re-run a
generation. Chat failures surface as errors, and a bare `404`/`500` is reported
without a body.

## 4. Models

Ids are **`publishers/<publisher>/models/<name>`**. A short name sent to the
gateway is a *silent* 404, so the client never guesses:

- `-m glm-53-flash` is resolved against the (cached) listing: exact id, then
  `/glm-53-flash` suffix, then substring — each step must be unique;
- an ambiguous or unknown name is an explicit local error listing candidates;
- a name containing `/` is treated as a full id and costs no request.

`-l` prints the full ids to pass to `-m`; `/listmodel` and `/switchmodel` use
the same inventory (and `/switchmodel` resolves short names too).

The listing is cached for `MAAS_MODEL_TTL` seconds in-process, so a probe, a
listing and a model resolution in one session cost **one** round-trip. The
cache holds ids only — never the key.

## 5. Chat behaviour

- **Reasoning is always on** and cannot be disabled (seven payload flags were
  tried; several are silently ignored). `/thinkingoff` is therefore a no-op
  for this backend, by design.
- **`max_tokens` includes reasoning tokens.** With a small cap the reply comes
  back `content: null`, `finish_reason: length`, and **no error**. The backend
  therefore raises the cap to at least `MAAS_MIN_OUTPUT_TOKENS` (1024) at
  request time, after the usual window clamp. A slightly over-budget request is
  preferable to a silent empty answer.
- **Native tool calling** is used. `glm-53-flash` and `gemma4` were falling
  through `TOOL_FORMAT_REGISTRY` to `inline` even though the endpoint serves
  native calls; both are now `openai`. The other `glm-4`/`glm4` entries stay
  `inline` (they target different checkpoints).
- **Inference parameters**: vLLM accepts the wide profile (`temperature`,
  `top_p`, `top_k`, `min_p`, `presence_penalty`).

## 6. Context window

The gateway exposes no way to learn it (no `/health`, no `/slots`). There is no
fabricated default: `refresh_context_window_size()` reports "unknown", prints
one hint, and every place that needs a window (the context bar, `/tokens`,
auto-compaction, the output clamp) keeps working with the unknown value. Set it
explicitly with `--context-size` or `/contextsizeset <tokens>`; an
`--context-size` override survives `/switchmodel` resetting the window.

## 7. Known limits

- **No exact token counting**: no `/tokenize`, and `/v1/messages/count_tokens`
  is not routed. Counts fall back to the heuristic estimator (~2× for
  code-heavy text) unless the backend exposes a counter.
- **No `/health`-style probe**: reachability is "`/v1/models` returns at least
  one model", with the retry policy above. An empty inventory is treated as
  "not ready" at startup.
- **Chat has no retry** (§3) and a bare 404 is not "offline" — it is a wrong
  origin or an incomplete model id.

## 8. Verification

Offline suite: `python3 -m unittest tests.test_maas -v` (69 tests). It covers
the classification/backoff/classification table, the cache TTL and stale
fallback, key resolution (env/file/argv), model resolution precedence, the
request shaping (URL, Bearer header, output floor, inference params), the
context-window override, the backend registries (`CLOUD_BACKENDS`, env maps,
`-b` choices, `LABELS`), and one end-to-end replay of the flaky listing
(`403 → 500 → 200-empty → 200-full`) through the real `urllib` transport.

The live facts this implementation is built on (only two routes, silent chat
errors, short-name 404, always-on reasoning, `max_tokens` sharing) were
measured against the target gateway before implementation. Re-check them
against a new gateway before assuming the same behaviour.
