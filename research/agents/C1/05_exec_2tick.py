"""C1 step 5 (after validation, description only): stricter fill test -- target must trade 2 ticks (0.5 pt) through.
Also counts how many trades exit at a target < 25 pts."""
import warnings; warnings.filterwarnings("ignore")
import os, importlib.util, numpy as np, pandas as pd
import c1lib as c
g, lv = c.load()
sp = importlib.util.spec_from_file_location("c1cand", os.path.join(c.HERE, "candidates.py")); C1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(C1)
rows = []
for k, f in C1.CANDIDATES.items():
    s = f(g, lv)
    a = c.run(g, s); t1 = c.run_through(g, s, 0.25); t2 = c.run_through(g, s, 0.5)
    for pn, lo, hi in [("discovery", *c.DISC), ("validation", *c.VAL)]:
        x = a[(a.date >= lo) & (a.date <= hi)]
        tpx = x[(x.reason == "tp")]
        small = (np.abs(tpx.exit - tpx.entry) < 25).sum()
        rows.append(dict(strategy=k, period=pn, tp_exits=len(tpx), tp_exits_lt25=int(small), base=c.stats(a, g, lo, hi)["avg"],
                         through_1tick=c.stats(t1, g, lo, hi)["avg"], through_2tick=c.stats(t2, g, lo, hi)["avg"],
                         sharpe_2tick=c.stats(t2, g, lo, hi)["sharpe"], dd_2tick=c.stats(t2, g, lo, hi)["maxdd"]))
pd.set_option("display.width", 250)
print(pd.DataFrame(rows).to_string(index=False))
