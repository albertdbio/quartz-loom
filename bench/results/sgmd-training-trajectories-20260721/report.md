# SGMD motion × coherence training trajectories

All motion means below use camera-compensated cells only. A `*` marks a screen-space-only displacement diagnostic; starred cells are counted separately and excluded from motion means. Coherence remains independently usable. Errors and holes are never zero-filled.

## Cross-check

All 11 supplied vehicle anchors match at the reported precision, and their full displacement JSON files are byte-identical to the stored independent spot reports. Fisher-normalized steps 15 and 20 are starred screen-space fallbacks; the supplied numbers match, but they are not camera-compensated despite the task wording.

| Corpus | Step | Expected | Actual | Compensated | Exact report |
| --- | ---: | ---: | ---: | --- | --- |
| sgmd_pilot | 25 | 2.031 | 2.030809 | true | true |
| sgmd_pilot | 50 | 0.352 | 0.352059 | true | true |
| sgmd_pilot | 75 | 0.061 | 0.061183 | true | true |
| sgmd_pilot | 100 | 0.227 | 0.227143 | true | true |
| sgmd_fnorm | 10 | 1.796 | 1.795996 | true | true |
| sgmd_fnorm | 15 | 2.457 | 2.456558 | false | true |
| sgmd_fnorm | 20 | 1.963 | 1.963068 | false | true |
| sgmd_fnorm | 25 | 0.009 | 0.009206 | true | true |
| sgmd_fnorm | 30 | 2.200 | 2.200041 | true | true |
| sgmd_fnorm | 35 | 2.046 | 2.045688 | true | true |
| sgmd_fnorm | 40 | 1.346 | 1.345532 | true | true |

## Raw SGMD (unnormalized Fisher, lambda=0.1)

| Prompt | step_000025 | step_000050 | step_000075 | step_000100 |
| --- | --- | --- | --- | --- |
| ball | 1.248255* / 0.000000 | 0.294174* / 8.278552 | 0.000000 / 9.582932 | 0.006845 / 0.000000 |
| walker | 0.250354* / 0.000000 | 2.028301* / 9.208736 | 1.735440* / 8.760862 | 0.428214* / 5.646459 |
| barrel | 0.635156* / 0.000000 | 0.097570 / 0.000000 | 0.391858 / 0.000000 | 0.000265* / 0.000000 |
| vehicle | 2.030809 / 7.553927 | 0.352059 / 10.000000 | 0.061183 / 9.997474 | 0.227143 / 9.920312 |

Native tree: `/Users/electric/Documents/areas_of_focus/decart-research/sgmd-pilot-20260721/batch_scores_2axis`. Displacement errors: 0; coherence errors: 0.

## Fisher-normalized SGMD (lambda=0.1)

| Prompt | step_000005 | step_000010 | step_000015 | step_000020 | step_000025 | step_000030 | step_000035 | step_000040 | step_000045 | step_000050 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ball | 0.073900 / 0.000000 | 0.351672* / 7.492041 | 0.000149 / 0.000000 | 0.850597 / 0.000000 | 0.023132 / 0.000000 | 0.102603 / 9.348131 | 0.114837 / 9.571482 | 0.112500 / 9.307718 | 0.233788 / 0.000000 | ERROR(D) / 0.000000 |
| walker | 1.531158* / 0.000000 | 1.722579* / 0.000000 | 1.972842* / 8.504357 | ERROR(D) / 9.551582 | 1.208663 / 9.592646 | 0.127174 / 9.583770 | 2.359996 / 8.823126 | 1.911727 / 10.000000 | 2.007346 / 9.927813 | 2.385902 / 9.524040 |
| barrel | 0.894531* / 0.000000 | 0.495163* / 0.000000 | 1.052777* / 9.833283 | 1.320066 / 0.000000 | 1.685362 / 0.000000 | 0.000692 / 0.000000 | 0.666517 / 0.000000 | 0.535877 / 0.000000 | 2.050280* / 9.648732 | 0.026937 / 9.897124 |
| vehicle | ERROR(D) / 9.523172 | 1.795996 / 10.000000 | 2.456558* / 7.038140 | 1.963068* / 9.258775 | 0.009206 / 8.723019 | 2.200041 / 9.789994 | 2.045688 / 10.000000 | 1.345532 / 9.995361 | 0.274560 / 9.878787 | 1.415723 / 10.000000 |

Native tree: `/Users/electric/Documents/areas_of_focus/decart-research/sgmd-fnorm-20260721/batch_scores_2axis`. Displacement errors: 3; coherence errors: 0.

## Fisher-normalized lambda sweep

| Prompt | lam0p05_step_000010 | lam0p05_step_000020 | lam0p05_step_000030 | lam0p05_step_000040 | lam0p05_step_000050 | lam0p2_step_000010 | lam0p2_step_000020 | lam0p2_step_000030 | lam0p2_step_000040 | lam0p2_step_000050 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ball | 0.121190 / 8.013441 | 0.053509 / 0.000000 | 0.163784 / 0.000000 | 0.084970 / 0.000000 | 0.007537 / 10.000000 | 0.000000 / 8.271937 | 0.049890 / 0.000000 | 3.390282 / 0.000000 | 0.000000 / 0.000000 | 0.007777 / 0.000000 |
| walker | 1.802767* / 0.000000 | 1.942057* / 9.435822 | 1.708425 / 9.520328 | 2.203550 / 9.776704 | 0.525879 / 9.642470 | 1.995545* / 6.700500 | 2.163393* / 9.931757 | 1.976812* / 10.000000 | 0.008818 / 9.529210 | 0.079579 / 9.692033 |
| barrel | 2.061829 / 7.518474 | 1.425950 / 8.233047 | 1.988213* / 0.000000 | 2.324739 / 9.647989 | 0.018394 / 0.000000 | 2.375568 / 0.000000 | 0.621961 / 8.040782 | 0.078711 / 9.115878 | 2.242683* / 9.768347 | 4.207528 / 9.923532 |
| vehicle | 1.887101* / 9.888786 | 2.250500* / 9.523650 | 0.835635 / 9.601890 | 2.254947 / 9.977661 | 2.388566 / 7.697103 | 1.997489* / 10.000000 | 7.277543 / 10.000000 | 1.706538 / 9.697369 | 6.019282 / 10.000000 | 2.479276 / 10.000000 |

Native tree: `/Users/electric/Documents/areas_of_focus/decart-research/sgmd-lsweep-20260721/batch_scores_2axis`. Displacement errors: 0; coherence errors: 0.

## Trajectory table

| Run | Step | Mean motion | Motion n/4 | Mean coherence | Coherence n/4 | Degraded | Errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw_sgmd_lam0p1 | 25 | 2.030809 | 1/4 | 1.888482 | 4/4 | 3 | 0 |
| raw_sgmd_lam0p1 | 50 | 0.224815 | 2/4 | 6.871822 | 4/4 | 2 | 0 |
| raw_sgmd_lam0p1 | 75 | 0.151014 | 3/4 | 7.085317 | 4/4 | 1 | 0 |
| raw_sgmd_lam0p1 | 100 | 0.116994 | 2/4 | 3.891693 | 4/4 | 2 | 0 |
| fnorm_lam0p1 | 5 | 0.073900 | 1/4 | 2.380793 | 4/4 | 2 | 1 |
| fnorm_lam0p1 | 10 | 1.795996 | 1/4 | 4.373010 | 4/4 | 3 | 0 |
| fnorm_lam0p1 | 15 | 0.000149 | 1/4 | 6.343945 | 4/4 | 3 | 0 |
| fnorm_lam0p1 | 20 | 1.085332 | 2/4 | 4.702589 | 4/4 | 1 | 1 |
| fnorm_lam0p1 | 25 | 0.731591 | 4/4 | 4.578916 | 4/4 | 0 | 0 |
| fnorm_lam0p1 | 30 | 0.607627 | 4/4 | 7.180474 | 4/4 | 0 | 0 |
| fnorm_lam0p1 | 35 | 1.296760 | 4/4 | 7.098652 | 4/4 | 0 | 0 |
| fnorm_lam0p1 | 40 | 0.976409 | 4/4 | 7.325770 | 4/4 | 0 | 0 |
| fnorm_lam0p1 | 45 | 0.838565 | 3/4 | 7.363833 | 4/4 | 1 | 0 |
| fnorm_lam0p1 | 50 | 1.276187 | 3/4 | 7.355291 | 4/4 | 0 | 1 |
| fnorm_lam0p05 | 10 | 1.091509 | 2/4 | 6.355175 | 4/4 | 2 | 0 |
| fnorm_lam0p05 | 20 | 0.739730 | 2/4 | 6.798130 | 4/4 | 2 | 0 |
| fnorm_lam0p05 | 30 | 0.902615 | 3/4 | 4.780554 | 4/4 | 1 | 0 |
| fnorm_lam0p05 | 40 | 1.717051 | 4/4 | 7.350589 | 4/4 | 0 | 0 |
| fnorm_lam0p05 | 50 | 0.735094 | 4/4 | 6.834893 | 4/4 | 0 | 0 |
| fnorm_lam0p2 | 10 | 1.187784 | 2/4 | 6.243109 | 4/4 | 2 | 0 |
| fnorm_lam0p2 | 20 | 2.649798 | 3/4 | 6.993135 | 4/4 | 1 | 0 |
| fnorm_lam0p2 | 30 | 1.725177 | 3/4 | 7.203312 | 4/4 | 1 | 0 |
| fnorm_lam0p2 | 40 | 2.009367 | 3/4 | 7.324389 | 4/4 | 1 | 0 |
| fnorm_lam0p2 | 50 | 1.693540 | 4/4 | 7.403891 | 4/4 | 0 | 0 |

## What the lambda sweep says

The data contradicts the proposed monotonic framing. Lower lambda=0.05 does not hold motion longer or blur less: across its 15 compensated cells it averages 1.078594 motion and 6.423868 coherence across all 20 cells. Lambda=0.2 averages 1.886850 motion across the same 15 compensated-cell coverage and 7.033567 coherence across all 20 cells, finishing at 1.693540 / 7.403891 with zero degraded cells.

On prompts camera-compensated under both lambdas, lambda=0.2 wins motion at four of five matched checkpoints but loses at step 20. The advantage is prompt-specific: at step 50 its walker is lower than lambda=0.05 while its barrel is much higher.

| Step | Common compensated prompts | lambda=0.05 motion | lambda=0.2 motion | High - low |
| ---: | --- | ---: | ---: | ---: |
| 10 | ball, barrel | 1.091509 | 1.187784 | +0.096275 |
| 20 | ball, barrel | 0.739730 | 0.335925 | -0.403804 |
| 30 | ball, vehicle | 0.499710 | 2.548410 | +2.048700 |
| 40 | ball, walker, vehicle | 1.514489 | 2.009367 | +0.494878 |
| 50 | ball, walker, barrel, vehicle | 0.735094 | 1.693540 | +0.958446 |

Displacement plus coherence does not establish prompt fidelity or correct physics. In particular, lambda=0.2's very large vehicle scores and its step-30 ball motion should still receive frame/blind review before promotion.

## Fisher-normalized vehicle step-25 anomaly

| Step | Motion | Coherence | Temporal | Spatial | Compensated | Degraded |
| ---: | ---: | ---: | ---: | ---: | --- | --- |
| 20 | 1.963068 | 9.258775 | 10.000000 | 8.572492 | false | true |
| 25 | 0.009206 | 8.723019 | 10.000000 | 7.609106 | true | false |
| 30 | 2.200041 | 9.789994 | 10.000000 | 9.584399 | true | false |

**Diagnosis:** The file is a healthy 81-frame decode and the learned coherence axis remains 8.723019 with temporal=10. The clip therefore is neither a broken decode nor a coherence collapse. Its requested transport is weaker and reverses mid-clip; the displacement score is further amplified downward by only 0.078189 track survival, so 0.009206 should not be read as literal absence of all screen motion.

**Re-decode:** Do not re-decode the same artifact as a corruption repair. If checkpoint-level promotion hinges on this isolated dip, add another deterministic seed/sample to measure variance rather than replacing this valid observation.

Step-25 coherence segments:

| Segment | Frames | Coherence | Temporal | Spatial | DINO median | Warp median | Warp max | Raw flow px |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| early | 0-20 | 9.812063 | 10.000000 | 9.627658 | 0.978429 | 0.013461 | 0.023756 | 2.941552 |
| middle | 20-50 | 8.581635 | 10.000000 | 7.364446 | 0.982817 | 0.020073 | 0.026125 | 7.113162 |
| late | 50-80 | 8.846127 | 10.000000 | 7.825397 | 0.978875 | 0.019356 | 0.025978 | 6.724857 |

## Interpretation limits

- Early checkpoint means can have only 1-2 compensated prompts; compare the `n/4` denominator, not only the scalar.
- `degrades_over_time=false` does not imply perfect structure; step 25 has an 8.58 middle-segment dip but misses the registered two-point degradation threshold.
- Motion × coherence still does not score prompt fidelity, direction, bounce/roll semantics, or physical plausibility.
- No expected condition was absent when scoring; `condition_holes` is therefore empty.
