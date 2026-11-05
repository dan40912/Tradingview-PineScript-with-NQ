"""01: Why did the '區邊空/多' (inside-day VA-edge fade) reference lose? Discovery only."""
import numpy as np, pandas as pd
import c2lib as L
from lib import sim
from edge import edge_signals
g, lv, i2 = L.load()
fam = "01_edge_ref"
tr = sim.run_signals(g, edge_signals(g, lv))
L.disc(g, tr, "edge_ref +55/-145 12:29", fam)
for d in (-1, 1):
    L.disc(g, tr[tr.dir == d], f"edge_ref dir={d}", fam)
D = tr[tr.split == "discovery"].copy()
D["vaw_adr"] = D.va_w / D.adr
D["exc_adr"] = D.exc / D.adr
D["tod"] = D.sig_idx
print(D.reason.value_counts())
def bucket(col, q=3):
    D["b"] = pd.qcut(D[col], q, duplicates="drop")
    print(D.groupby("b", observed=True).pnl.agg(["count", "mean"]).round(0).T)
for c in ["vaw_adr", "exc_adr", "tod", "open_pos", "nexc"]:
    print("---", c); bucket(c)
print("--- on_in"); print(D.groupby("on_in").pnl.agg(["count", "mean"]).round(0))
print("--- dir x on_in"); print(D.groupby(["dir", "on_in"]).pnl.agg(["count", "mean"]).round(0))
D["h"] = D.date.map(L.half)
print(D.groupby(["h", "dir"]).pnl.agg(["count", "mean"]).round(0))
