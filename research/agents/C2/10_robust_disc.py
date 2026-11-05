"""10: Robustness of the two leads on discovery only (ADR window, long/short, ex-2025-04, top trades, entry delay, ES)."""
import numpy as np, pandas as pd
import c2lib as L
from lib import sim, data, features
import sig
g, lv, i2 = L.load()
im = sig.i2_entry_idx(g, lv)
fam = "10_robust"
run = lambda s: sim.run_signals(g, s)
def te(gg, ll, s=0.35, **kw):
    return sig.trend_ext(gg, ll, day="outside", s=s, i2map=sig.i2_entry_idx(gg, ll), **kw)
for name, fn in [("TE_out_s35", lambda gg, ll: te(gg, ll, 0.35)), ("TE_out_s50", lambda gg, ll: te(gg, ll, 0.5)),
                 ("PMP_s35", lambda gg, ll: sig.pm_pullback(gg, ll, T=180, s=0.35))]:
    print("=====", name)
    for n in [5, 15, 20]:
        lv2 = lv.copy(); lv2["adr10"] = L.adr(g, n)
        L.disc(g, run(fn(g, lv2)), f"{name} ADR{n}", fam)
    tr = run(fn(g, lv))
    for d in (1, -1):
        L.disc(g, tr[tr.dir == d], f"{name} dir={d}", fam)
    L.disc(g, tr[~tr.date.str.startswith("2025-04")], f"{name} ex-2025-04", fam)
    D = tr[tr.split == "discovery"].sort_values("pnl")
    for k in (1, 3, 5):
        print(f"   drop top{k}: avg={D.pnl.iloc[:-k].mean():.0f}  drop worst{k}: avg={D.pnl.iloc[k:].mean():.0f}")
    sd = [dict(x, entry_slot=x["entry_slot"] + 1) for x in fn(g, lv)]
    L.disc(g, run(sd), f"{name} entry+1min", fam)
    tr2 = tr.copy(); tr2["pnl"] = tr2.pnl - 40
    L.disc(g, tr2, f"{name} cost+$40", fam, log=False)
# ES (all parameters ADR-relative -> no scaling needed)
ges = data.load_grid("ES"); assert all(d < "2026-01-01" for d in ges.dates)
les = features.daily_levels(ges); les["adr10"] = L.adr(ges, 10)
for name, fn in [("TE_out_s35", lambda gg, ll: te(gg, ll, 0.35)), ("TE_out_s50", lambda gg, ll: te(gg, ll, 0.5)),
                 ("PMP_s35", lambda gg, ll: sig.pm_pullback(gg, ll, T=180, s=0.35))]:
    tr = sim.run_signals(ges, fn(ges, les))
    r = L.disc(ges, tr, f"ES {name}", fam)
    tr["gross"] = tr.pnl + ges.cost
    print("    ES gross avg (disc):", round(tr[tr.split == "discovery"].gross.mean(), 0))
