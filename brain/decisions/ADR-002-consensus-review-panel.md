---
type: decision
status: accepted
date: 2026-07-19
description: "Future reviews request claude-opus, kimi-k3, and grok. The claude-opus lineage targets concrete claude-fable-5 first and automatically retries claude-opus-4-8 after a detected rate limit."
anchors: ["AGENTS.md", "PLAN.md", "bench/quality/quality-repair-v1.protocol.json"]
related: ["[[State]]", "[[ADR-001-quality-qualified-headline]]"]
---

# ADR-002 — Consensus review panel

## Decision

From this decision onward, important project reviews request three independent
voter lineages from the consensus tool:

1. claude-opus;
2. kimi-k3;
3. grok.

`claude-opus` is the request alias, not the concrete first-choice model. In the
rebuilt `../oh-my-openagent`, that lineage targets `claude-fable-5` first; if
the Fable vote is detected as rate-limited, the consensus engine retries
`claude-opus-4-8` automatically. The caller never requests a `claude-fable`
lineage and never constructs the Opus fallback manually. Do not silently
replace any other lineage. Keep the caller/synthesizer separate from the voter
list and preserve each returned concrete model identifier plus any exposed
fallback metadata in the consensus record; aliases alone are not sufficient
attribution.

## Scope

This policy applies to future phase gates, material spend decisions, research
increment selection, target changes, and public claims. It also applies to the
quality-repair artifact review in progress when this decision was issued.

Earlier records are historical evidence, not policy templates: their GPT,
Kimi, and Grok attribution remains unchanged because those models actually
voted.

## Safety and review input

The existing hard rules remain unchanged: scrub credentials, private hosts,
and connection strings before fan-out; give voters the actual artifact rather
than a builder summary; and independently check voter arithmetic before
adopting corrections.

## Operational verification — 2026-07-19

The first completed review under this policy requested exactly
`claude-opus`, `kimi-k3`, and `grok`. The returned concrete voters were
`claude-fable-5`, `moonshotai/kimi-k3`, and `grok-4.5`; Fable completed directly,
so no Opus retry or fallback metadata was exposed. The three voters unanimously
returned NO-GO on the pre-hardening quality protocol.

A second call used the same requested aliases and revised actual artifacts but
timed out after 600 seconds without returning a completed panel. Direct session
recovery from `ses_083124e38ffeGJLxe1Jh7m0v9Z` later found two completed child
votes: concrete `claude-fable-5` and `grok-4.5` independently returned NO-GO.
The `moonshotai/kimi-k3` child recorded zero output tokens and no final text.
This is a partial two-voter recovery, not a completed panel verdict; Fable
completed directly, so no Opus retry occurred. See
[[Gotcha-Consensus-Timeouts-Preserve-Voter-Sessions]].

The correct next action remains to invoke the same aliases after the artifact
is review-ready; do not substitute a `claude-fable` request or manually call
Opus. If the bridge times out, recover child-session output before declaring
the review lost, preserve concrete attribution, and distinguish partial signal
from consensus.

Focused verification in `../oh-my-openagent` passes 55/55 tests across model
lineage, voter resolution, rate-limit classification, and the consensus engine.
The suite proves both routes: Fable-5 is preferred when available, and a
detected Fable rate limit triggers exactly one fresh Opus-4.8 vote; validation,
context, cancellation, and generic provider failures do not spuriously trigger
that retry.
