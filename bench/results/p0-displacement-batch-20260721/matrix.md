# Displacement batch matrix

| Prompt | d0-full-wan-vae score | d0-full-wan-vae span | d0-full-wan-vae survival | d0-full-wan-vae displaced | d1-rolling-taehv score | d1-rolling-taehv span | d1-rolling-taehv survival | d1-rolling-taehv displaced | d2-reset-every-block score | d2-reset-every-block span | d2-reset-every-block survival | d2-reset-every-block displaced | ring-off score | ring-off span | ring-off survival | ring-off displaced | ring-on score | ring-on span | ring-on survival | ring-on displaced |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ball | 4.000687 | 0.217463 | 0.974747 | false | 3.889719 | 0.216092 | 0.982363 | false | 2.115336 | 0.205699 | 0.901796 | false | 3.889719 | 0.216092 | 0.982363 | false | 0.000000 | 0.009082 | 1.000000 | false |
| walker | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | 0.050011 | 0.048802 | 0.956790 | false | 0.004881 | 0.020146 | 1.000000 | false |
| rolling-object | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | 0.211627 | 0.056676 | 0.980012 | false | 0.000000 | 0.012754 | 0.979424 | false |
| vehicle | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | 0.000000 | 0.006685 | 0.977208 | false | 0.071057 | 0.040267 | 0.983539 | false |

## Comparison: ring-off vs ring-on

| Prompt | A score | B score | B - A | Verdict |
| --- | ---: | ---: | ---: | --- |
| ball | 3.889719 | 0.000000 | -3.889719 | ring-off higher by 3.889719 |
| walker | 0.050011 | 0.004881 | -0.045130 | ring-off higher by 0.045130 |
| rolling-object | 0.211627 | 0.000000 | -0.211627 | ring-off higher by 0.211627 |
| vehicle | 0.000000 | 0.071057 | 0.071057 | ring-on higher by 0.071057 |

Mean B - A (4 comparable): -1.018855 — ring-off higher on mean by 1.018855
