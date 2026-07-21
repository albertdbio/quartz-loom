# Displacement batch matrix

| Prompt | cd\_1step score | cd\_1step span | cd\_1step survival | cd\_1step displaced | cd\_4step score | cd\_4step span | cd\_4step survival | cd\_4step displaced | final\_1step score | final\_1step span | final\_1step survival | final\_1step displaced | final\_2step score | final\_2step span | final\_2step survival | final\_2step displaced | ode\_1step score | ode\_1step span | ode\_1step survival | ode\_1step displaced | ode\_4step score | ode\_4step span | ode\_4step survival | ode\_4step displaced | oneforcing\_1step score | oneforcing\_1step span | oneforcing\_1step survival | oneforcing\_1step displaced |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ball | 0.000000 | 0.007352 | 0.168724 | false | 1.924407 | 0.321235 | 0.679012 | false | ERROR | ERROR | ERROR | ERROR | 8.133922 | 0.370101 | 0.992248 | true | ERROR | ERROR | ERROR | ERROR | 0.890132 | 0.093568 | 0.995679 | false | INVALID | INVALID | INVALID | INVALID |
| barrel | ERROR | ERROR | ERROR | ERROR | 0.081487 | 0.039785 | 0.997531 | false | 0.380199 | 0.069816 | 0.995014 | false | 3.967300 | 0.201537 | 0.984636 | false | ERROR | ERROR | ERROR | ERROR | 1.150276 | 0.231108 | 0.638889 | false | INVALID | INVALID | INVALID | INVALID |
| walker | ERROR | ERROR | ERROR | ERROR | 0.057702 | 0.051681 | 0.846914 | false | 0.000000 | 0.009210 | 0.978395 | false | 0.000669 | 0.017005 | 1.000000 | false | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | INVALID | INVALID | INVALID | INVALID |
| vehicle | 0.037849 | 0.043813 | 0.974009 | false | 1.232953 | 0.240493 | 0.518519 | false | 0.286952 | 0.056049 | 1.000000 | false | 0.055885 | 0.032886 | 1.000000 | false | 0.006619 | 0.033091 | 0.049383 | false | 2.128855 | 0.660843 | 0.663580 | false | INVALID | INVALID | INVALID | INVALID |

## Comparison: final\_1step vs cd\_4step

| Prompt | A score | B score | B - A | Verdict |
| --- | ---: | ---: | ---: | --- |
| ball | ERROR | 1.924407 | ERROR | error: final\_1step |
| barrel | 0.380199 | 0.081487 | -0.298712 | final\_1step higher by 0.298712 |
| walker | 0.000000 | 0.057702 | 0.057702 | cd\_4step higher by 0.057702 |
| vehicle | 0.286952 | 1.232953 | 0.946001 | cd\_4step higher by 0.946001 |

Mean B - A (3 comparable): 0.234997 — cd\_4step higher on mean by 0.234997

## Comparison: final\_1step vs ode\_4step

| Prompt | A score | B score | B - A | Verdict |
| --- | ---: | ---: | ---: | --- |
| ball | ERROR | 0.890132 | ERROR | error: final\_1step |
| barrel | 0.380199 | 1.150276 | 0.770077 | ode\_4step higher by 0.770077 |
| walker | 0.000000 | ERROR | ERROR | error: ode\_4step |
| vehicle | 0.286952 | 2.128855 | 1.841903 | ode\_4step higher by 1.841903 |

Mean B - A (2 comparable): 1.305990 — ode\_4step higher on mean by 1.305990

## Comparison: cd\_4step vs ode\_4step

| Prompt | A score | B score | B - A | Verdict |
| --- | ---: | ---: | ---: | --- |
| ball | 1.924407 | 0.890132 | -1.034275 | cd\_4step higher by 1.034275 |
| barrel | 0.081487 | 1.150276 | 1.068789 | ode\_4step higher by 1.068789 |
| walker | 0.057702 | ERROR | ERROR | error: ode\_4step |
| vehicle | 1.232953 | 2.128855 | 0.895902 | ode\_4step higher by 0.895902 |

Mean B - A (3 comparable): 0.310139 — ode\_4step higher on mean by 0.310139
