"""Screen: does the sign of an earlier window predict a later window (momentum >0, reversal <0)?
Discovery only, descriptive, no costs. Prior-close = last RTH close of previous day (lv.pdc)."""
import numpy as np, pandas as pd
from a2lib import load, sim
g, lv = load("NQ")
D = (g.split == "discovery") & g.tradable & lv.has_prev.values
O = pd.DataFrame(g.O).ffill(axis=1).values
C = pd.DataFrame(g.C).ffill(axis=1).values
S = sim.slot_of
def P(t): return O[:, S(t)]
pdc = lv.pdc.values
preds = {"pdc->10:00": P("10:00")-pdc, "pdc->09:30": P("09:30")-pdc, "09:30->10:00": P("10:00")-P("09:30"),
         "09:30->11:30": P("11:30")-P("09:30"), "09:30->12:00": P("12:00")-P("09:30"), "09:30->14:00": P("14:00")-P("09:30"),
         "09:30->15:00": P("15:00")-P("09:30"), "09:30->15:30": P("15:30")-P("09:30"), "pdc->15:30": P("15:30")-pdc,
         "12:00->15:30": P("15:30")-P("12:00"), "18:00->09:30": P("09:30")-O[:,0], "14:00->15:00": P("15:00")-P("14:00")}
tgts = {"10:00->11:30": C[:,S("11:29")]-P("10:00"), "11:30->14:00": C[:,S("13:59")]-P("11:30"),
        "12:00->15:59": C[:,S("15:59")]-P("12:00"), "14:00->15:59": C[:,S("15:59")]-P("14:00"),
        "15:00->15:59": C[:,S("15:59")]-P("15:00"), "15:30->15:59": C[:,S("15:59")]-P("15:30"),
        "09:30->12:29": C[:,S("12:29")]-P("09:30"), "10:00->15:59": C[:,S("15:59")]-P("10:00")}
rows=[]
for pn,p in preds.items():
    for tn,t in tgts.items():
        a = S(pn.split("->")[1]) if ":" in pn.split("->")[1] else 0
        b = S(tn.split("->")[0])
        if b < a: continue
        m = D & np.isfinite(p) & np.isfinite(t)
        x = np.sign(p[m])*t[m]
        rows.append((pn, tn, m.sum(), x.mean().round(1), (x.mean()/x.std()*np.sqrt(len(x))).round(2)))
r = pd.DataFrame(rows, columns=["pred","tgt","n","mean_signed_pts","t"])
print(r.sort_values("t").to_string())
print("screens:", len(r))
