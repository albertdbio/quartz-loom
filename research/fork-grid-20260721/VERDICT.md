# Fork-grid verdict: Stage-3 DMD is where displacement dies

Date: 2026-07-21. Runner: Claude (orchestrating researcher). Pod `rtv-claude-fork-decode` (H100, ~2.3h, ≈$7, stopped after clip pull). Inputs: official released checkpoints from `zhuhz22/Causal-Forcing` (+ `JiaqiFeng/OneForcing`), reference `thu-ml/Causal-Forcing` inference code @ `ac70136`, the 4 P0 object-displacement prompts, seed 0, 81 frames @ 480x832, full Wan VAE decode. Clips: `grid_out/<condition>/*.mp4` (28 total). Montages reviewed by eye per prompt; vehicle prompt (starkest P0 failure) reviewed across all conditions, walker as second read.

## Observations (vehicle prompt unless noted)

| Condition | What the pixels show | Motion | Quality |
|---|---|---|---|
| `cd_4step` (Stage-2 CD init, native 4-step) | camera-follow: car centered, background streams past; late bg degradation | real, moderate | good early |
| `ode_4step` (Stage-2 ODE init, native 4-step) | genuine scene progression, car displaces w/ perspective; walker: heavy camera swing + identity drift | rich | fair, drifts |
| `cd_1step` (Stage-2 CD forced 1-step) | violent motion-blur streaking; car dissolves by mid-clip | huge | collapses |
| `ode_1step` (Stage-2 ODE forced 1-step) | same: strong motion, blur soup late | huge | collapses |
| `final_1step` (official CF++ 1-step = CD init + Stage-3 DMD) | **car dead center, static background, zero traverse** — reproduces OUR Gfinal's P0 failure exactly; walker strides in place under camera-follow, never crosses | ~none (transport) | excellent |
| `final_2step` (official CF++ 2-step) | near-static, same family | ~none | excellent |
| `oneforcing_1step` | noise after frame ~1 — **INVALID: 17GB checkpoint incompatible with this repo's inference path** (One-Forcing has its own fork w/ register heads); not evidence about One-Forcing | n/a | n/a |

File-size corroboration (H.264 bits ≈ pixel change): finals 2.0–2.5MB/condition vs ODE 3.5–4.0MB (oneforcing 8.1MB is noise entropy — excluded).

## Verdict

1. **Stage-3 asymmetric DMD introduces the displacement collapse.** Both Stage-2 inits move (messily); both DMD finals are pristine near-stills. The distillation stage trades transport for fidelity — the mode-seeking reverse-KL settles on the high-quality static/camera-follow mode. This resolves P0's open Cell D.
2. **Our port is exonerated.** The official released CF++ 1-step checkpoint on the reference implementation reproduces our Gfinal's failure on the same prompts (small delta: official walker articulates in place; ours decayed to standing — worth one look later, but the core collapse is the method's).
3. **Init choice is secondary, not primary.** ODE-init Stage-2 shows richer, more coherent scene motion than CD-init Stage-2 (directionally consistent with One-Forcing's DD 52.76-vs-23.61) — but neither init is static; the terminal stillness appears only after DMD. An init swap alone, followed by the same Stage-3, would likely re-collapse.
4. **Reconciliation with CF++'s reported DD≈66:** VBench/VidProM prompts reward any dynamics incl. camera motion; our prompts ban camera motion and command object transport — precisely the mode DMD keeps (camera-follow/in-place articulation) vs the mode it drops (object traverse). Both numbers can be true; ours measures the product-relevant one.

## Implication → next experiment

The fix must target the **Stage-3 objective**, on top of whichever init (ODE preferred as the richer motion prior):
- Motion-preserving Stage-3: SGMD-style score-gradient matching, or DMD with an explicit anti-static term grounded in real-video statistics — NOT endpoint-MSE (known DD→1.30 collapse) and with AAD-1's warning in mind (adversarial stages also eat motion).
- Cheap intermediate probe: fewer Stage-3 steps / earlier checkpoints (motion may die progressively across DMD training — an intermediate-checkpoint sweep localizes the dose-response if intermediate checkpoints can be produced).
- Scoring for all of the above: the calibrated displacement harness (`realtime-video/bench/displacement_metrics.py`, session-21) + its forthcoming batch mode.

## Caveats
- n=4 prompts, seed 0, one run; montage-based eyeball scoring (harness batch-scoring pending).
- Stage-2 "forced 1-step" conditions use a config the checkpoints weren't trained for (expected degradation; included only to separate sampling-collapse from training-collapse).
- `oneforcing_1step` must be re-run in One-Forcing's own repo before any claim about its recipe.
