"""03: Descriptive (discovery only): outside-value days where I2 has NOT triggered by T.
Forward points to 15:29 following the move-from-open, split by toward/away-from-value and |move|/ADR."""
import numpy as np, pandas as pd
import c2lib as L
g, lv, i2 = L.load()
i2d = dict(zip(i2.i, i2.entry_slot - L.RTH))
rows = []
for r in lv.itertuples():
    if not r.has_prev or r.split != "discovery" or r.i2_dir == 0 or not np.isfinite(r.adr10):
        continue
    O, H, Lo, C = L.rth(g, r.i)
    if not np.isfinite(C[359]):
        continue
    for T in [30, 45, 60, 90, 120, 150, 180]:
        t = T - 1
        c = C[t]
        if not np.isfinite(c) or (r.i in i2d and i2d[r.i] <= t):
            continue
        mv = c - r.open_0930
        tow = np.sign(mv) == r.i2_dir
        rows.append({"T": T, "h": L.half(r.date), "tow": "toward" if tow else "away",
                     "mag": abs(mv) / r.adr10, "fwd": np.sign(mv) * (C[359] - c)})
D = pd.DataFrame(rows)
D["magb"] = pd.cut(D.mag, [0, .25, .5, .75, 5])
for by in (["T", "tow"], ["T", "magb"], ["T", "tow", "magb"]):
    t = D.groupby(by, observed=True).fwd.agg(["count", "mean"]).round(1)
    t2 = D.groupby(by + ["h"], observed=True).fwd.mean().unstack().round(0)
    print(pd.concat([t, t2], axis=1).to_string()); print()
