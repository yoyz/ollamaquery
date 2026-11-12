# Refactor history: v0.3.5 → v0.4.0 (the local `dev` branch)

The branch spans `551e58d` (v0.3.4) to `d3c3f5e`; `v0.3.5` is the version the
code reported throughout, and `v0.4.0` is what the release commit finally
stamped.

Archive, not a live handoff. The live open-item list is `TODO_refactor.md`
(untracked, local-only); this file exists so the fine-grained history is not
lost when the `dev` branch is retired.

## 1. Why this file exists

The work between v0.3.4 and v0.4.0 was done on a local `dev` branch as **43
small commits** (one item per commit: refactor, fix, test, docs). Upstream only
ever received **3 rolled-up commits** — the content is identical, the
granularity is not. `dev` was never pushed (it had no upstream), so deleting
the branch would have made those 43 commit messages, diffs and per-item
boundaries unreachable.

This document records what upstream does not: the item-by-item decomposition,
the commit hashes, and the size of each step.

## 2. Shape: two histories, one tree

| Item                | Value                                                                 |
| ------------------- | --------------------------------------------------------------------  |
| Common base         | `551e58d` — v0.3.4 (2026-10-23), also upstream's parent commit        |
| Local `dev` tip     | `d3c3f5e` — 43 commits, all authored 2026-10-27 (02:43 → 17:32 +0100) |
| Upstream tip        | `c119d00` — v0.4.0 (2026-10-27), tag `0.4.0`, branch `main`           |
| Authors             | `agent <agent@local>` (commits 1–7), `Johann Peyrard` (8–43)          |
| Tree difference     | `TODO_refactor.md` only in `dev`; `__version__` line only upstream    |

Both trees are the same code. Measured at the fast-forward:

```
$ git diff --stat dev c119d00
 ollamaquery2/TODO_refactor.md | 707 ------------------------------------------
 ollamaquery2/ollamaquery2.py  |   2 +-
 2 files changed, 1 insertion(+), 708 deletions(-)
```

What those two entries are:

- `TODO_refactor.md` — the local handoff note, never tracked upstream (708
  deletions here because a one-line change elsewhere is counted as an insertion).
- `ollamaquery2.py` — one line: `__version__ = "0.3.5"` on `dev` versus
  `"0.4.0"` on `main`, bumped by the release commit itself.

`git show c119d00:ollamaquery2/ollamaquery2.py` and the working file hash
identically (md5 `fc680ef09e557b9080bae9dbd831eaea`), so `main` and `dev` ship
the same program.

Cumulative size of the range (base → tip):

| Range                  | Files   | Insertions   | Deletions   |
| ---------------------- | ------- | ------------ | ----------- |
| `551e58d` → `dev`      | 17      | 12 809       | 4 723       |
| `551e58d` → `main`     | 16      | 12 102       | 4 723       |

The 17-vs-16 file gap is `TODO_refactor.md` (707 lines, local-only), and it is
the whole 707-line insertion gap as well: `ollamaquery2.py` contributes the
same 7 044 / 4 550 to both ranges, and the version line differs only in content
(`0.3.5` on `dev`, `0.4.0` on `main`), not in count.

**Versioning note.** The refactor passes were tagged `v0.3.6`–`v0.3.21` in the
handoff notes and in `AGENTS.md`, but those were **session labels, not code
versions** — `__version__` stayed at `0.3.5` through all 43 commits and was
bumped to `0.4.0` only by the upstream release commit. That is why `dev`'s tip
still says `0.3.5` while `main` says `0.4.0`; the release commit is the only
content difference between them.

## 3. The 3 upstream roll-up commits (what the public history shows)

| Commit      | Date         | Subject                                                                            | Files   | +       | −       |
| ----------- | ------------ | -------------------------------------------------------------------------------    | ------- | ------- | ------- |
| `c065010`   | 2026-10-23   | doc/agentic_git_inspect.md                                                         | 1       | 155     | 0       |
| `c4e4150`   | 2026-10-26   | v0.3.5: /promptshow, git_inspect blame + file history, an ACL that explains itself | 10      | 3 777   | 97      |
| `c119d00`   | 2026-10-27   | release: v0.4.0 — component extractions, review fixes, bounded reads               | 14      | 8 397   | 4 853   |

Mapping to the local history:

- `c4e4150` + `c065010` ≈ commit 1 (`794ace3`), the pre-checkpoint rollup. That
  commit also carries the v0.3.6 refactor passes (ToolCallParser,
  TimeoutEvent/AgenticTimeoutPolicy, StepFeedback, ToolExecution) and the
  v0.3.7 spill/probe fixes, because none of that work was committed
  individually — hence its 7 700/2 775 size.
- `c119d00` ≈ commits 2–43 plus the version bump: the four bounded reads, the
  three component extractions out of `ModelQuery`, `WireView` /
  `ConversationStore` out of `ChatLoop`, `AgenticRunner` + `AgenticFinalizer`,
  the three review findings (P1 curl bound, P2/P3 container workdir), the 4.2
  exception-handler narrowing, and the accompanying handoff docs.

## 4. The 43 commits, oldest first

`kind` is the commit-message prefix; `+`/`−` are line counts from
`git show --shortstat`. One item = one commit, so the sequence is also the
order in which the units of work were landed and tested.

| #    | commit    | kind          | change                                                                             | files   | +      | −      |
| ---- | --------- | ------------- | ---------------------------------------------------------------------------------- | ------- | ------ | ------ |
| 1    | 794ace3   | checkpoint    | v0.3.7 working state before 4.3/4.4/1.4                                            | 14      | 7700   | 2775   |
| 2    | 5340fa2   | agentic+mcp   | logging is best-effort, failed spill says so                                       | 3       | 225    | 12     |
| 3    | 8779ae5   | grep          | stop lying about absence, and bound the scan                                       | 3       | 515    | 30     |
| 4    | ec4a078   | sandbox       | CLI-selected image, exec into a running container, memoized pull                   | 6       | 591    | 55     |
| 5    | 1511b9c   | sandbox       | cap throwaway containers at 1 cpu / 1 GiB and show it                              | 5       | 138    | 35     |
| 6    | 91838ee   | docs          | record the web-server E2E's weak assertion (kept on purpose)                       | 1       | 23     | 0      |
| 7    | b80578f   | docs          | rewrite TODO_refactor.md as a handoff for the next session                         | 1       | 255    | 406    |
| 8    | 95ed563   | cluster       | validate resource names against the live API, never rewrite them                   | 4       | 455    | 50     |
| 9    | 214515c   | docs          | refresh the handoff numbers after 1.2                                              | 1       | 10     | 9      |
| 10   | d276c16   | net           | bound the four unbounded reads (16 MiB page, 256 KiB probe/stderr)                 | 6       | 240    | 33     |
| 11   | 1b53a10   | docs          | record the 1.1 commit in the handoff                                               | 1       | 3      | 2      |
| 12   | f79d1a9   | refactor      | extract the outbound-wire cluster into WireView (TODO 3.3)                         | 3       | 569    | 442    |
| 13   | d84104b   | docs          | record the 3.3 commit in the handoff                                               | 1       | 3      | 2      |
| 14   | 2338107   | refactor      | extract ConversationStore (messages + session + compact) (TODO 3.4)                | 3       | 577    | 391    |
| 15   | 11f13eb   | docs          | record the 3.4 commit in the handoff                                               | 1       | 3      | 2      |
| 16   | 9f254c0   | refactor      | move the ReAct loop into AgenticRunner (TODO 3.1)                                  | 2       | 877    | 803    |
| 17   | ee871df   | docs          | record the 3.1 commit in the handoff                                               | 1       | 3      | 2      |
| 18   | 9727538   | tests         | drop the hasattr(messages) checks the property made vacuous                        | 4       | 6      | 18     |
| 19   | c104d71   | docs          | record the test-cleanup commit in the handoff                                      | 1       | 3      | 2      |
| 20   | 6691726   | docs          | handoff resume pass — state, commits, reachability, next steps                     | 1       | 78     | 27     |
| 21   | 876df46   | refactor      | extract RequestBuilder (payload shaping) from ModelQuery (TODO 4.1)                | 1       | 389    | 311    |
| 22   | 8f774f3   | refactor      | extract StatsCalculator (timing + stats line) from ModelQuery (TODO 4.1)           | 1       | 230    | 186    |
| 23   | efdd64d   | docs          | record the 4.1a/4.1b commits in the handoff                                        | 1       | 86     | 40     |
| 24   | 33f7c37   | refactor      | extract StreamParser (chunk shapes + aggregation) from ModelQuery (TODO 4.1c)      | 2       | 466    | 321    |
| 25   | 78fe3bf   | docs          | record the 4.1c commit in the handoff (4.1 closed)                                 | 1       | 57     | 26     |
| 26   | 5d5ba6d   | tests         | re-pin the /switchmodel token tests to the v0.2.22 context bar                     | 1       | 31     | 15     |
| 27   | af5b8dc   | docs          | record the Ollama live pass and the 5d5ba6d test re-pin in the handoff             | 1       | 55     | 20     |
| 28   | 98809e3   | refactor      | delete the dead catch in looks_like_truncated_tool_call (TODO 4.2, 1/N)            | 2       | 33     | 6      |
| 29   | 4278042   | refactor      | narrow the context-bar fallback to real count failures (TODO 4.2, 2/N)             | 2       | 24     | 3      |
| 30   | 9883d5c   | refactor      | narrow WireView._promptshow_tool_defs (TODO 4.2, 3/N)                              | 2       | 22     | 1      |
| 31   | c8ce7dd   | tests         | bare-loop fixtures stub the context bar (4.2 fallout)                              | 2       | 17     | 3      |
| 32   | 02c0992   | docs          | record the 4.2 first pass (classification + 3 fixes) in the handoff                | 1       | 69     | 19     |
| 33   | 2fb33d2   | fix           | map a CWD in a subdirectory of a mounted volume (review P2)                        | 2       | 56     | 7      |
| 34   | b3bc719   | fix           | bound the curl fallback by reading its stdout, not by --max-filesize (review P1)   | 2       | 121    | 24     |
| 35   | c2560f7   | docs          | record the two review findings (P1 curl bound, P2 container workdir)               | 1       | 18     | 9      |
| 36   | 2147352   | refactor      | extract AgenticFinalizer from AgenticRunner (review follow-up)                     | 1       | 953    | 870    |
| 37   | 0c3993f   | docs          | record the AgenticRunner split (TODO 3.7) in the handoff                           | 1       | 51     | 18     |
| 38   | 5dd0286   | fix           | a mount whose source is / must match every CWD (review P3)                         | 2       | 35     | 1      |
| 39   | a539696   | tests         | skip the gzip-bomb curl test when a local port cannot be bound                     | 1       | 5      | 1      |
| 40   | 4d5e7a5   | docs          | record review P3 and the test portability fix in the handoff                       | 1       | 12     | 8      |
| 41   | 23e3c54   | docs          | correct the "commits ahead of main" figure in the handoff                          | 1       | 4      | 2      |
| 42   | a00c1df   | refactor      | split AgenticRunner.run into setup / step / exit (TODO 3.7)                        | 1       | 204    | 155    |
| 43   | d3c3f5e   | docs          | record the run split (TODO 3.7 second slice) in the handoff                        | 1       | 32     | 16     |

Distribution by prefix: `docs` 18, `refactor` 11, `tests` 4, `fix` 3, and 7
one-offs (`sandbox` 2, `checkpoint`/`net`/`grep`/`cluster`/`agentic+mcp` 1
each). The handoff note was updated after every landed item, so the sequence
doubles as a work log; the item numbers in the subjects (`TODO 3.1`, `3.3`,
`3.4`, `3.7`, `4.1`, `4.2`) point at `TODO_refactor.md` §3. The structural
content of the release is the `ModelQuery`/`ChatLoop`/`AgenticRunner`
extractions (12, 14, 16, 21, 22, 24, 36, 42), the three review fixes
(33, 34, 38) and the portability guard for the P1 test (39).

## 5. Invariants worth re-checking

These are cheap and would have caught a bad squash or a lost file. Run them
from the repository root (`/mnt/…/ollamaquery`, one level above `ollamaquery2/`),
with a clean worktree on `main`:

```bash
# 1. The code is the same on both lines (only the version line should appear).
git diff main dev -- ':!ollamaquery2/TODO_refactor.md'

# 2. The working file is byte-identical to the tagged release.
git show 0.4.0:ollamaquery2/ollamaquery2.py | md5sum
md5sum ollamaquery2/ollamaquery2.py

# 3. Only the handoff file was added on dev (plus the version line).
git diff --name-status main dev
```

## 6. Restoring or replaying the history

`dev` is due for deletion. The commits are only reachable through the branch
(or the reflog, ~90 days by default), so keep one anchor:

```bash
git tag -a dev-0.3.5-to-0.4.0 d3c3f5e -m "43-commit history behind v0.4.0"
git branch -D dev
```

With that tag:

```bash
git log --reverse dev-0.3.5-to-0.4.0 ^551e58d     # the sequence above
git show 2147352                                   # one step's diff
git format-patch -o /tmp/dev-history 551e58d..dev-0.3.5-to-0.4.0
```

Do **not** try `git cherry-pick 551e58d..dev` onto `main`: upstream already
contains that content, so every patch would conflict. The history is for
reading (why a change was made, in which order, at what size), not for
re-applying.

If the branch is ever deleted without a tag, `git reflog` is the only recovery
path until the objects are pruned — hence the tag above.
