---
type: gotcha
status: active
date: 2026-07-27
description: "Stopping a RunPod pod releases its GPUs to the pool while the volume stays pinned to that host, so restarting can fail for hours with 'not enough free GPUs'; pull anything irreplaceable while you are already inside, never on a planned second visit."
anchors: ["PLAN.md", "studio/infra/README.md"]
related: ["[[Gotcha-Deploy-Green-Log-Not-Shipped]]", "[[State]]"]
---

# Gotcha — Stopping a pod releases its GPU, and the volume can't follow

A stopped RunPod pod keeps its disk, so "stop to save money, start again later"
reads as free. It is not. Stopping returns the GPUs to the pool, and a pod's
disk is **pinned to the specific host machine** it was created on. Restarting
therefore requires *that host* to have capacity again. When it does not, the
API answers:

```
400 - There are not enough free GPUs on the host machine to start this pod.
```

There is no workaround from the outside. Pod-attached storage cannot be
detached and mounted elsewhere (`list-network-volumes` is empty for this
account — these are not network volumes), so the data is unreachable until a
tenant elsewhere happens to release GPUs on that box. For a 2xH100 in a busy
datacenter, that wait was **hours**, and availability across every datacenter
read `LOW`/`NONE` throughout.

## This has now cost the project twice

Day 3 of Phase 1, recorded in PLAN.md at the time: *"host GPU reclaimed the
day-3 pod while stopped (RunPod risk — terminate+recreate, day-3 raw logs lost
but numbers recorded)."*

2026-07-27, the same trap: `sgmd-lsweep-2xh100` was started to inspect its
volume, `ls`-ed, and stopped again to halt a $5.98/hr burn. The GPUs were gone
within minutes. That volume held the only copy of the lambda-sweep student
weights — the run behind the motion x coherence frontier moving from (0.29, 10)
to (2.65, 7.0). They were recovered only because a capacity window happened to
open later. `sgmd-fnorm-2xh100` came closer still: its host was full for hours,
and in the gap the account balance fell to $0.36 so starting it returned
`402 Payment Required`.

## The rule

**Pull irreplaceable data while you are already inside the box.** Not on the
next visit — there may not be one. If you have started a pod and opened an SSH
session, the cost of copying the weights out is already sunk into the hourly
rate you are paying; the cost of *not* copying them is the whole run.

Corollaries:

- Treat an SSH session to a pod holding unbacked artifacts as the retrieval
  window, even if retrieval was not the reason you connected.
- Weights are the only thing worth rescuing from a training volume. Verdicts
  and scores are already in `research/`; base models (`ckpts/Wan2.1-*`) are
  re-downloadable; venvs rebuild from `bootstrap.sh`. Copy the final-step
  `model.pt` and skip the rest — 11GB instead of 146GB.
- Verify with `sha256sum` computed **on the pod** and compared locally. Equal
  file sizes are not equal bytes.
- A measurement run (benchmarks, logs, metrics) leaves nothing unique behind.
  A training run does. Sort volumes by that distinction before deleting any.
