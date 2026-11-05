"""15: Look-ahead check: for each signal, blank that day's data from the entry minute on and regenerate."""
import copy, numpy as np
import c2lib as L
from candidates import CANDIDATES, REJECTED
g, lv, i2 = L.load()
for name, f in {**CANDIDATES, **REJECTED}.items():
    if name == "C2_BOOK":
        continue
    sigs = f(g, lv, sized=False)
    bad = 0
    for s in sigs:
        i, e = s["i"], s["entry_slot"]
        saved = [a[i, e:].copy() for a in (g.O, g.H, g.L, g.C, g.V)]
        for a in (g.O, g.H, g.L, g.C, g.V):
            a[i, e:] = np.nan
        s2 = [x for x in f(g, lv, sized=False) if x["i"] == i]
        for a, sv in zip((g.O, g.H, g.L, g.C, g.V), saved):
            a[i, e:] = sv
        if not s2 or s2[0]["entry_slot"] != e or s2[0]["dir"] != s["dir"] or abs(s2[0]["sl_pts"] - s["sl_pts"]) > 1e-9:
            bad += 1
    print(name, f"{len(sigs) - bad}/{len(sigs)} unchanged")
