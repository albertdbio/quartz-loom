# Quality rubric v1

This rubric scores a complete video against the original prompt. Judge the
visible result, not the system name, presumed method, or generation speed.
Every item receives five integer scores from 1 to 10 and a short rationale.
Watch the full video at least twice: once normally and once while checking the
first and final thirds for accumulated drift.

## Dimensions

1. **Prompt adherence** — requested subject, action, count, setting, camera,
   and attributes are visibly present. Extra detail is not rewarded when it
   contradicts the original prompt.
2. **Spatial/anatomical fidelity** — bodies, faces, hands, rigid objects,
   contact points, perspective, and scene geometry are plausible within each
   frame.
3. **Identity/geometry consistency** — subjects, object counts, colors,
   shapes, and background layout persist across time without unexplained
   replacement, duplication, or morphing.
4. **Motion naturalness** — articulated motion, trajectories, interactions,
   camera motion, fluids, particles, and deformable materials evolve smoothly
   and physically plausibly. Static output does not score highly merely for
   being clean.
5. **Temporal artifacts** — score high when flicker, popping, tearing,
   sudden camera jumps, blur accumulation, texture crawl, and late-rollout
   degradation are absent.

The per-rating quality score is the unweighted mean of these five dimensions.
Do not provide a separate discretionary “overall” number.

## Anchors

- **1 — unusable:** the requested scene is missing or the video is dominated
  by collapse, severe deformation, discontinuity, or near-static noise.
- **3 — poor:** the scene is recognizable, but major prompt elements fail or
  recurrent structural/temporal errors make the result unsuitable.
- **5 — mixed:** recognizable and watchable, with several obvious failures;
  useful as development evidence but below a public quality bar.
- **7 — good:** the request is fulfilled throughout, motion reads naturally,
  and only minor artifacts are visible on close inspection. This is the
  absolute gate anchor.
- **9 — excellent:** coherent, detailed, natural, and stable throughout, with
  no material failure and only negligible imperfections.

Scores 2, 4, 6, 8, and 10 interpolate between anchors. A 10 should be rare and
requires essentially flawless output for this resolution and duration.

## Required response fields

For each blind asset return:

```json
{
  "blind_id": "opaque id supplied with the asset",
  "scores": {
    "prompt_adherence": 1,
    "spatial_fidelity": 1,
    "identity_consistency": 1,
    "motion_naturalness": 1,
    "temporal_artifacts": 1
  },
  "first_third_quality": 1,
  "final_third_quality": 1,
  "failure_tags": ["none or concise controlled tags"],
  "rationale": "Brief observations grounded in visible events."
}
```

Do not guess the generating system. Do not inspect source filenames or an
unblinding key. If media cannot be decoded or reviewed in full, report the
asset as invalid rather than assigning a quality score.
