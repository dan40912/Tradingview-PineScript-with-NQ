"""Discovery-period robustness for the 3 shortlisted families: ES, excluding Apr-2025, half-years, ADR-n surface."""
import numpy as np, pandas as pd
from a2lib import load, sim, disc_only, summ
from sweeps import sweep_signals
from ext import ext_follow, adr10
S = sim.slot_of
def moc(g, lv, sig="15:29", sl=None):
    adr = adr10(g); out = []
    for r in lv.itertuples():
        i = r.i
        if not r.has_prev or not np.isfinite(adr[i]): continue
        c = g.C[i, S(sig)]; b = r.pdc
        if not (np.isfinite(c) and np.isfinite(b)) or c == b: continue
        out.append({"i": i, "entry_slot": S(sig)+1, "dir": -int(np.sign(c-b)), "tp_pts": None, "sl_pts": sl, "exit_slot": S("15:59")})
    return out
def halves(tr):
    d = tr[tr.split=="discovery"].copy()
    h = d.date.str[:4] + "H" + ((d.date.str[5:7].astype(int) > 6) + 1).astype(str)
    x = d.groupby(h).pnl.agg(["count", "mean"]).round(0)
    noapr = d[~d.date.str.startswith("2025-04")].pnl.mean()
    return " | ".join(f"{k}:{int(v['count'])}/{v['mean']:+.0f}" for k, v in x.iterrows()) + f" | exAprq25 {noapr:+.0f}"
gN, lvN = load("NQ"); gE, lvE = load("ES")
sc = 0.275
for sym, g, lv, f in [("NQ", gN, lvN, 1.0), ("ES", gE, lvE, sc)]:
    t1 = sim.run_signals(g, sweep_signals(g, lv, level="on", tgt="R", R=2.0, last_idx=120, exit_idx=389, buf=2*f, stop_buf=5*f, min_risk=10*f, max_risk=80*f))
    disc_only(g, t1, f"{sym} ON sweep 2R", "robust_disc"); print("   ", halves(t1))
    t2 = sim.run_signals(g, ext_follow(g, lv, k=0.75, s=0.35))
    disc_only(g, t2, f"{sym} ext follow k.75 s.35", "robust_disc"); print("   ", halves(t2))
    t3 = sim.run_signals(g, moc(g, lv))
    disc_only(g, t3, f"{sym} MOC rev", "robust_disc"); print("   ", halves(t3))
# ADR-n x k surface for ext follow on NQ (discovery avg per trade)
rows = []
for n in [5, 10, 15, 20]:
    for k in [0.6, 0.75, 0.9]:
        tr = sim.run_signals(gN, ext_follow(gN, lvN, k=k, s=0.35, n=n))
        s = summ(gN, tr, "discovery"); rows.append((n, k, s["n"], s["avg"], s["sharpe_ann"]))
print(pd.DataFrame(rows, columns=["adr_n", "k", "n", "avg", "sh"]).pivot(index="adr_n", columns="k", values="avg"))
