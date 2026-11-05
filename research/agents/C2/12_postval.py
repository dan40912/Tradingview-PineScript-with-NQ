"""12: After the one-shot validation (descriptive only, nothing here changes the frozen candidates).
(a) risk-budget sensitivity of I2 + C2_TE_OUTSIDE; (b) equal-drawdown view; (c) equity-curve stats;
(d) CONTAMINATED: TE_OUTSIDE split by whether I2 had already entered (A2 published the on/off-I2 split)."""
import numpy as np, pandas as pd
import c2lib as L
from lib import sim
import candidates as C
import sig
g, lv, i2 = L.load()
te1 = C.te_outside(g, lv, sized=False)

def eq_stats(trs, sp=None):
    t = pd.concat([x[["date", "pnl", "split"]] for x in trs])
    if sp:
        t = t[t.split == sp]
    d = L.daily(g, t)
    if sp:
        d = d[[L.data.split_of(x) == sp for x in d.index]]
    eq = d.cumsum().values
    peak = np.maximum.accumulate(np.concatenate([[0], eq]))[1:]
    dd = eq - peak
    # longest stretch under water (trading days)
    longest = cur = 0
    for x in dd:
        cur = cur + 1 if x < 0 else 0
        longest = max(longest, cur)
    m = d.groupby([x[:7] for x in d.index]).sum()
    return {"total": round(eq[-1]), "maxdd": round(dd.min()), "worst_day": round(d.min()), "longest_uw_days": longest,
            "neg_months": f"{(m < 0).sum()}/{len(m)}", "sharpe": round(d.mean() / d.std(ddof=1) * np.sqrt(252), 2)}

print("=== (a) risk budget R per TE trade (I2 fixed at 1 contract)")
rows = []
for R in [0, 500, 750, 1000, 1500]:
    C.RISK_USD = float(R) if R else 1000.0
    te = sim.run_signals(g, C._size(te1, g, True)) if R else None
    trs = [i2] + ([te] if R else [])
    for sp in ["discovery", "validation", None]:
        s = eq_stats(trs, sp)
        rows.append({"R": R, "period": sp or "2024-2025", **s})
C.RISK_USD = 1000.0
A = pd.DataFrame(rows); print(A.to_string(index=False)); A.to_csv("out_12_risk_budget.csv", index=False)

print("\n=== (b) equal-drawdown view, 2024-2025: scale the book so its maxDD equals I2 alone")
te = sim.run_signals(g, C.te_outside(g, lv))
b = eq_stats([i2, te]); a = eq_stats([i2])
k = a["maxdd"] / b["maxdd"]
print(f"I2 alone total {a['total']} dd {a['maxdd']}; book total {b['total']} dd {b['maxdd']} -> scale {k:.2f} -> total {b['total']*k:.0f} at dd {a['maxdd']}")
for sp in ["discovery", "validation"]:
    a = eq_stats([i2], sp); b = eq_stats([i2, te], sp)
    print(sp, "I2", a, "\n          book", b)

print("\n=== (d) CONTAMINATED post-hoc: TE_OUTSIDE (1ct) by I2 status at signal time")
im = sig.i2_entry_idx(g, lv)
tr = sim.run_signals(g, te1)
tr["i2_before"] = [(i in im and im[i] < es - L.RTH) for i, es in zip(tr.i, tr.entry_slot)]
tr["i2_day"] = tr.i.isin(set(i2.i))
lvi = lv.set_index("i")
tr["rel"] = np.where(tr.dir == tr.i.map(lvi.i2_dir), "toward", "away")
for col in ["i2_before", "i2_day", "rel"]:
    print(tr.groupby([col, "split"]).pnl.agg(["count", "mean"]).round(0).unstack().to_string())
# I2 status by 11:30 for signals after 11:30
late = tr[tr.entry_slot - L.RTH > 120]
print("signals after 11:30:", late.groupby(["i2_day", "split"]).pnl.agg(["count", "mean"]).round(0).unstack().to_string())
