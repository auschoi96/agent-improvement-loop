---
name: token-efficient-execution
description: >-
  Use throughout any coding or shell task. Cut wasted tokens by never re-reading
  a file already read this session, batching related shell commands into one
  call, and dropping repeated cd/setup boilerplate. Reuse what is already in
  context instead of re-fetching it.
---

# Token-efficient execution

Most of the token cost in a long coding session is **re-fetched work**: the same
file read again, the same `cd <repo> && source .venv/...` prologue replayed
before every command, the same directory listed repeatedly. Every one of those is
input tokens you already paid for. This skill is a standing instruction to stop
paying twice.

These rules are deterministic — they govern *which tool calls you make*, not the
quality of your answer. Following them reduces token usage without changing what
the task asks for. Correctness comes first: never skip a read whose content you
do **not** already have, and never drop a command the task actually needs.

## 1. Never re-read a file you have already read this session

Before calling `Read` (or `cat`/`head`/`tail`/`sed` via `Bash`) on a path, check
whether you have already read it in this session.

- If you read it and **have not changed it since**, do not read it again. The
  contents are already in your context — reference them directly.
- If you only need a specific symbol or line range from a large file you have
  seen, recall it from context rather than re-reading the whole file.
- Re-read **only** when the file has genuinely changed since you last saw it —
  e.g. you just edited it, a command rewrote it, or a tool reported it changed.
  When you do re-read after an edit you made, you already know what you changed,
  so prefer reading just the changed region.

When you catch yourself about to re-read, state briefly what you already know
from the earlier read and proceed without the call.

## 2. Batch related shell commands into a single call

Combine commands that run in the same place and have no decision point between
them, instead of issuing one tool call per command.

- Chain with `&&` (stop on first failure) or `;` (run regardless) in one `Bash`
  call: `ruff check src && ruff format --check src && mypy src && pytest -q`.
- Read several short files at once (`cat a.txt b.txt`) rather than one call each.
- Split into separate calls **only** when you must inspect the output of one step
  before deciding the next, or when isolating a failure.

## 3. Do not replay setup / `cd` boilerplate

Working directory and environment that you have already established persist for
the rest of your turn unless something changed them.

- Establish the working directory and environment **once**. Do not prefix every
  later command with the same `cd <repo>` / `export ...` / `source .venv/...`
  prologue.
- Prefer absolute paths, or `git -C <dir>` and tool-native directory flags, over
  re-`cd`-ing for a single command.
- Re-establish setup only after something actually invalidated it (a new shell,
  a reported directory change, an activated/deactivated environment).

## 4. Reuse context instead of re-deriving it

- Command output you already captured (a directory listing, a test result, a
  `git status`) is in context — read it again from there rather than re-running
  the command, unless the underlying state has changed.
- Cache facts you looked up (a file's location, a function signature, a config
  value) and reuse them for the rest of the session.
- Do not re-run an unchanged probe just to confirm something you already
  observed.

## What this skill does NOT ask you to do

- Do not skip reading something you have **not** read, or whose content changed.
- Do not drop a command the task genuinely requires, or merge commands across a
  real decision point.
- Do not trade correctness for a smaller token count. The goal is to eliminate
  *redundant* work, never *necessary* work.
