---
type: gotcha
status: active
date: 2026-07-19
description: "A consensus bridge timeout can strand completed voter analyses in child OpenCode sessions; recover them from SQLite, attribute each concrete model, and treat missing voters as a partial panel rather than a consensus verdict."
anchors: ["AGENTS.md", "PLAN.md", "brain/decisions/ADR-002-consensus-review-panel.md"]
related: ["[[ADR-002-consensus-review-panel]]", "[[State]]", "[[session-4-quality-protocol-hardening]]"]
---

# Gotcha — Consensus timeouts preserve voter sessions

The consensus bridge has a bounded orchestration timeout. That timeout can fire
after one or more external voters have already completed, especially when one
lineage stalls before the caller/synthesizer can return a combined result. The
voter work is not necessarily lost.

Recover the parent OpenCode session and its descendants directly from
`~/.local/share/opencode/opencode.db`. Query `session.parent_id` to enumerate
the voter sessions, read assistant `text` parts by joining `message` to `part`,
and use each child session's `model` JSON for concrete attribution. This also
works after live context compaction.

Recovery does not manufacture consensus. Record completed voter positions
individually, record zero-token or missing-final voters as incomplete, and do
not promote a partial panel to a unanimous verdict. In
`ses_083124e38ffeGJLxe1Jh7m0v9Z`, `claude-opus` concretely resolved to
`claude-fable-5` and completed NO-GO, `grok` resolved to `grok-4.5` and
completed NO-GO, while `moonshotai/kimi-k3` produced no final response. That is
two aligned independent positions, not a completed three-voter panel. Fable
completed directly, so no Opus fallback occurred in that session.

The current resolver contract is deliberate: request the lineage alias
`claude-opus`; it seats `claude-fable-5` first and retries concrete
`claude-opus-4-8` once only if Fable is classified as rate-limited. Mere
unavailability, zero output, or a normal timeout does not trigger Opus. Do not
request `claude-fable` and do not manually construct the fallback.

Two later sessions exposed separate timeout modes:

- `ses_08291c0efffebU9NGjv3FxY6zm` asked voters to inspect actual repository
  paths. Their restricted file-tool attempts consumed the whole bridge window;
  none produced a final. A Grok probe also left a two-byte `HELLO2.txt`, which
  was verified and removed.
- `ses_0828769cfffexJxZUYfsxTxWhp` embedded 57,980 characters of complete
  source/schema to remove the file-access blocker. Transport/orchestration used
  roughly 285 seconds before the three voter sessions were created, leaving
  only about 315 seconds under the bridge's 600-second deadline. Grok-4.5
  completed; Fable-5 and Kimi-K3 did not.

The underlying limits are mismatched: the native voter path can use 360 seconds
plus a 180-second active-generation grace, while the OpenCode bridge is killed
at 600 seconds and the Codex MCP waits 620 seconds. Bridge overhead therefore
can terminate a panel before the native voter allowance is exhausted. Diagnose
the failure before retrying: use a smaller actual diff/artifact packet, recover
child sessions, and never count inspection chatter as a final position.
