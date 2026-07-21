# Coherence calibration

## Condition means

| Condition | Expected tier | Mean | Min | Max | Degrades |
| --- | --- | ---: | ---: | ---: | ---: |
| final_1step | most_coherent | 9.631239 | 8.783284 | 10.000000 | 0/4 |
| final_2step | most_coherent | 9.961025 | 9.844100 | 10.000000 | 0/4 |
| cd_4step | mid | 7.302360 | 0.000000 | 10.000000 | 1/4 |
| ode_4step | mid | 4.044224 | 0.000000 | 10.000000 | 2/4 |
| cd_1step | low | 1.938434 | 0.000000 | 7.753737 | 2/4 |
| ode_1step | low | 0.000000 | 0.000000 | 0.000000 | 1/4 |

Expected group means: finals 9.796132 > four-step 5.673292 > one-step 0.969217 — **PASS**.
Strict condition-tier separation — **PASS**.

## SGMD anchors

| Clip | Score | Temporal | Spatial | Early | Middle | Late | Degrades | Expected check |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| ball | 0.000000 | 7.121341 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | false | PASS |
| vehicle | 7.553927 | 9.977530 | 5.719032 | 10.000000 | 8.763047 | 6.935005 | true | PASS |

## Per clip

| Source | Condition | Prompt | Score | Temporal | Spatial | Early | Middle | Late | Degrades |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| fork_grid | final_1step | ball | 9.741670 | 9.998319 | 9.491609 | 9.960631 | 9.832315 | 9.727653 | false |
| fork_grid | final_1step | barrel | 10.000000 | 10.000000 | 10.000000 | 10.000000 | 10.000000 | 10.000000 | false |
| fork_grid | final_1step | walker | 8.783284 | 10.000000 | 7.714607 | 9.920260 | 9.104295 | 8.572127 | false |
| fork_grid | final_1step | vehicle | 10.000000 | 10.000000 | 10.000000 | 10.000000 | 10.000000 | 10.000000 | false |
| fork_grid | final_2step | ball | 10.000000 | 10.000000 | 10.000000 | 10.000000 | 10.000000 | 10.000000 | false |
| fork_grid | final_2step | barrel | 9.844100 | 10.000000 | 9.690631 | 10.000000 | 9.961030 | 9.727068 | false |
| fork_grid | final_2step | walker | 10.000000 | 10.000000 | 10.000000 | 10.000000 | 10.000000 | 10.000000 | false |
| fork_grid | final_2step | vehicle | 10.000000 | 10.000000 | 10.000000 | 10.000000 | 10.000000 | 10.000000 | false |
| fork_grid | cd_4step | ball | 0.000000 | 10.000000 | 0.000000 | 10.000000 | 0.000000 | 0.000000 | true |
| fork_grid | cd_4step | barrel | 9.841885 | 10.000000 | 9.686271 | 10.000000 | 9.800243 | 10.000000 | false |
| fork_grid | cd_4step | walker | 10.000000 | 10.000000 | 10.000000 | 10.000000 | 10.000000 | 10.000000 | false |
| fork_grid | cd_4step | vehicle | 9.367557 | 10.000000 | 8.775112 | 9.340510 | 9.367557 | 9.799117 | false |
| fork_grid | ode_4step | ball | 10.000000 | 10.000000 | 10.000000 | 10.000000 | 10.000000 | 10.000000 | false |
| fork_grid | ode_4step | barrel | 0.000000 | 10.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | false |
| fork_grid | ode_4step | walker | 6.176896 | 10.000000 | 3.815405 | 9.385746 | 9.461512 | 4.235995 | true |
| fork_grid | ode_4step | vehicle | 0.000000 | 10.000000 | 0.000000 | 10.000000 | 0.000000 | 0.000000 | true |
| fork_grid | cd_1step | ball | 0.000000 | 10.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | false |
| fork_grid | cd_1step | barrel | 0.000000 | 9.661094 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | false |
| fork_grid | cd_1step | walker | 0.000000 | 9.800178 | 0.000000 | 8.255573 | 0.000000 | 0.000000 | true |
| fork_grid | cd_1step | vehicle | 7.753737 | 10.000000 | 6.012043 | 10.000000 | 10.000000 | 7.265735 | true |
| fork_grid | ode_1step | ball | 0.000000 | 10.000000 | 0.000000 | 0.000000 | 0.000000 | 10.000000 | false |
| fork_grid | ode_1step | barrel | 0.000000 | 7.061976 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | false |
| fork_grid | ode_1step | walker | 0.000000 | 8.835110 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | false |
| fork_grid | ode_1step | vehicle | 0.000000 | 10.000000 | 0.000000 | 9.889447 | 0.000000 | 0.000000 | true |
| sgmd_pilot | step_000025 | ball | 0.000000 | 7.121341 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | false |
| sgmd_pilot | step_000025 | vehicle | 7.553927 | 9.977530 | 5.719032 | 10.000000 | 8.763047 | 6.935005 | true |

## Ordering violations

- None.
