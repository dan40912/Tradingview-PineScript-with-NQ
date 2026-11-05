"""POC-definition sensitivity of A3_C3 (lesson from history step 51: a POC_R signal vanished when POC moved
2.5 pts from bin lower edge to bin midpoint). features.value_area returns the POC bin's LOWER edge.
Descriptive robustness check; the frozen candidate is NOT changed."""
import numpy as np, pandas as pd
from a3lib import load, i2_entries, with_exit, run, stats, noprog_exit_slot

K = ("n", "win%", "avg", "total", "maxdd", "ret/dd", "sharpe_ann")
pd.set_option("display.width", 250)
g, lv = load("NQ")
sigs = i2_entries(g, lv)
xs = lambda g_, r, s, px: noprog_exit_slot(g_, s, px, 30, 20)
rows = []
for lab, shift_price, extra_dist, floor in [
        ("frozen (POC = bin lower edge)", 0.0, 0.0, 20), ("POC = bin midpoint (+2.5 price)", 2.5, 0.0, 20),
        ("POC = bin upper edge (+5 price)", 5.0, 0.0, 20), ("POC -2.5 price", -2.5, 0.0, 20),
        ("target 5 pts shorter", 0.0, -5.0, 20), ("target 2.5 shorter", 0.0, -2.5, 20),
        ("target 2.5 longer", 0.0, 2.5, 20), ("target 5 longer", 0.0, 5.0, 20),
        ("midpoint POC, floor 15", 2.5, 0.0, 15), ("midpoint POC, floor 25", 2.5, 0.0, 25)]:
    tp = lambda g_, r, s, px, a=shift_price, b=extra_dist, f=floor: max(s["dir"] * (r.poc + a - px) + b, f)
    tr = run(g, with_exit(g, lv, sigs, tp=tp, sl=60, exit_slot=xs))
    for sp in ["discovery", "validation"]:
        st = stats(tr, g, sp)
        rows.append({"variant": lab, "split": sp, **{k: st.get(k) for k in K}})
df = pd.DataFrame(rows)
df.to_csv("out_c3_poc_sensitivity.csv", index=False)
print(df.to_string(index=False))
