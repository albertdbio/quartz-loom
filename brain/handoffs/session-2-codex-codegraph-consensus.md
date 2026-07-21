---
type: handoff
status: archived
session: 2
date: 2026-07-19
description: "Installed the custom Codegraph v1.4 and typed consensus MCPs plus global skills in Codex, then verified both end-to-end from Codex CLI."
branch: main
key_commits: []
prior_handoff: "session-1-h100-gate-and-second-brain"
---

# Session 2 Handoff — Codex Codegraph and consensus

## TL;DR

- Codex CLI now discovers the custom Codegraph v1.4 MCP and global Codegraph
  skill; a real Codex run queried this project's indexed code and memory.
- Added a typed local consensus MCP backed by OpenCode's existing multi-lineage
  `consensus` tool, plus a locked-down OpenCode transport agent and Codex skill.
- The bridge validates inputs, rejects common secret/private-network forms,
  bounds output, serializes panels, handles cancellation/timeouts, and passes
  9/9 Node tests.
- A real scrubbed Codex panel returned successful Kimi-K3 and Grok positions;
  both chose unit tests plus an end-to-end smoke test, with no dissent.
- The experimental state is unchanged: [[ADR-001-quality-qualified-headline]]
  still leaves the quality-qualified headline open.

## What this session worked on

- **Codegraph in Codex** — ran the custom v1.4 install target, then pinned the
  MCP command to `/Users/electric/.local/bin/codegraph-custom` so Codex does not
  resolve the stale PATH-installed v0.9.9 binary.
- **Consensus in Codex** — built `~/.local/share/codex-consensus-mcp`, registered
  it as the `consensus` stdio MCP, and added the global skill at
  `~/.agents/skills/consensus/SKILL.md`.
- **Transport hardening** — OpenCode's `run` command requires a trailing `-` to
  consume the JSON request from stdin. Codex also canceled the tool as
  `user cancelled MCP tool call` until the MCP advertised accurate read-only,
  non-destructive, open-world annotations.

## State at session close

[[State]] is the live project truth. Codex can now start from the indexed brain,
inspect code through Codegraph, and invoke the same external consensus surface
used by OpenCode. No GPU resource was started and no git commit was created.

## Verification evidence

- `npm test` in `~/.local/share/codex-consensus-mcp` passes **9/9** tests,
  including stdin transport, redaction, timeout process-group cleanup, and MCP
  initialize/tool discovery.
- `codex mcp get codegraph` resolves the absolute custom wrapper with
  `serve --mcp`; `codex mcp get consensus` reports the bridge enabled with a
  620-second tool timeout.
- Codegraph smoke: `codex exec --strict-config --ephemeral --json -s read-only
  -C /Users/electric/Documents/areas_of_focus/arrow-agent-workspace/realtime-video
  '<Codegraph-only report_measured_h100_gate query>'` called
  `codegraph_explore` against this project and returned `roofline/roofline.py`.
- Consensus smoke: the same Codex shape with a scrubbed two-lineage request
  called `consensus`, returned `ok: true`, and produced successful `kimi-k3` and
  `grok` voter records in about 32 seconds.
- `scripts/codegraph-local sync .` and `status .` remain the project-local index
  maintenance commands; the global MCP uses the same custom v1.4 build.
- Codex emitted a non-blocking stale model-cache warning on some invocations
  (`supports_reasoning_summaries` missing), but both MCP smokes completed.

## Likely next moves

- Start the next Codex session in this directory and have it read `[[State]]`
  and `PLAN.md` through Codegraph before changing experiments.
- Freeze CF++ 1-step plus TAEHV and pre-register the quality-repair sweep.
- Run the broader multi-family/human quality audit and a sustained ≥60-second
  validation before revisiting any public headline.
- These moves are provisional; new quality evidence may pivot the experiment.

## See also

- [[Handoffs]]
- Prior: [[session-1-h100-gate-and-second-brain]]
