# realtime-video — agent instructions

## Read the brain first

Start every session from `brain/current/State.md`, then read `PLAN.md` for the
full mission, decision protocol, phase history, budget ledger, consensus record,
and progress log. `State.md` is the compact live truth; `PLAN.md` is the detailed
experimental record. Update both when their facts change, and update PLAN's
Progress Log before ending a session.

Durable knowledge lives in `brain/`: append-only decisions in
`brain/decisions/`, paid-for operational traps in `brain/gotchas/`, research
notes, and slim session event stamps in `brain/handoffs/`. Notes use YAML
frontmatter (`type`, `status`, `date`, a payload-bearing `description`, and
`anchors:` naming code symbols/files) so the custom Codegraph memory layer
returns institutional memory beside the code it explains.

Brain rules: one concern per note; filename is the wikilink target; anchor every
decision/gotcha to at least one code symbol or file; ADRs are append-only
(reversals create a new ADR with `supersedes:`); end every working session with
a handoff based on `brain/_templates/Handoff-Template.md` and add it to
`brain/handoffs/Handoffs.md`. `.codegraph/` is local and gitignored. After brain
or source edits, run `scripts/codegraph-local sync .`; verify with
`scripts/codegraph-local status .` and an actual `codegraph_explore` query that
surfaces the anchored note. Use this wrapper rather than the PATH-installed
`codegraph`: it deliberately runs the workspace's custom `../codegraph/dist`
build whose memory layer indexes these notes.

Current v1.4.0 retrieval quirk: an exact Markdown note path passed to
`codegraph_explore` can still rank nearby Python above the requested note. The
anchored summary path is verified; for note discovery use
`scripts/codegraph-local query "State" --kind note`, then read the named note.

Standing rules (mirrored from PLAN.md):
- Agent operates autonomously here. Important decisions (phase gates, spends ≥$200, Phase-3 increment choice, public claims/headlines, target changes) go through the standalone **`consensus` CLI**, invoked from the shell — NOT an MCP tool (there is no consensus MCP server anymore). Run it via the shell/exec tool: `consensus run --lineages claude-opus,kimi,grok --count 3 [--artifact <file>]... --json < prompt` (prompt on stdin). Lineage routing is handled by the CLI: `claude-opus` → `anthropic/claude-opus-4-8` (this account is out of Fable usage, so the CLI uses Opus directly via the direct-Anthropic route, not opencode's disabled Zen route); `grok`/`gpt`/etc. → opencode. Use `kimi` (→ kimi-k2.6), **NOT** `kimi-k3` (vercel-only; it cannot seat via opencode). The CLI returns a JSON envelope of voter positions for YOU to synthesize; positions are advisory. See [[ADR-002-consensus-review-panel]].
- Never send secrets to consensus voters. Give voters the actual artifact, not a summary — prefer `--artifact <file>` so voters READ the files instead of inlining bytes (bypasses any prompt-size limit). Re-check voter arithmetic before adopting corrections (see Consensus Record precedent).
- The `consensus` CLI has a built-in panel deadline and auto-persists each voter to `~/.local/share/consensus-cli/panels/` as it settles: a slow/stuck voter returns a `timeout` position instead of hanging the whole call, and an identical re-run reuses completed voters from cache (no double-pay). The old opencode.db bridge-timeout recovery is obsolete. A partial panel (some voters `timeout`/`error`) is independent review signal, not a completed verdict.
- Quality is reported next to every throughput number — never fps alone.
- No git commits unless the owner explicitly asks.
- Budget cap $3,000 total; log every spend in PLAN.md's ledger.

## Layout and verification

- `roofline/` — calibrated analytical model plus measured-gate report.
- `bench/` — Phase-1 harness, runbook, metrics, logs, and video artifacts.
- `post/` — public-facing drafts; every throughput claim must carry its quality
  qualifier and apples-to-apples caveat.
- `brain/` — compact durable memory for future agents; do not turn handoffs into
  duplicate warehouses.

Useful local gates:

```bash
python3 bench/harness.py
python3 -m py_compile bench/harness.py roofline/roofline.py
python3 roofline/roofline.py
scripts/codegraph-local sync . && scripts/codegraph-local status .
```

`bench/results/` contains expensive captured evidence. Do not regenerate remote
GPU results merely to test parsing or formatting; write local tests against the
captured responses first. Never print or commit `realtime-video/.env`.

No git commits unless the owner explicitly asks.
