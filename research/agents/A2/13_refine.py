"""Refinement checks (discovery only): MOC stop plateau; ext-follow with 15:29 exit across ADR-n x k; TOM long RTH."""
import numpy as np, pandas as pd
from a2lib import load, sim, disc_only, summ
from ext import ext_follow
import importlib; R = importlib.import_module("12_robust_disc".replace(".py","")) if False else None
from lib.data import RTH_OFF
g, lv = load("NQ"); S = sim.slot_of
def moc(sl, sig="15:29"):
    out = []
    for r in lv.itertuples():
        i = r.i
        if not r.has_prev: continue
        c = g.C[i, S(sig)]; b = r.pdc
        if not (np.isfinite(c) and np.isfinite(b)) or c == b: continue
        out.append({"i": i, "entry_slot": S(sig)+1, "dir": -int(np.sign(c-b)), "tp_pts": None, "sl_pts": sl, "exit_slot": S("15:59")})
    return out
for sl in [10.0, 30.0, 60.0, 80.0]:
    disc_only(g, sim.run_signals(g, moc(sl)), f"MOC sl={sl}", "refine")
for sig in ["15:19", "15:24"]:
    disc_only(g, sim.run_signals(g, moc(30.0, sig)), f"MOC sl=30 sig={sig}", "refine")
rows = []
for n in [5, 10, 15, 20]:
    for k in [0.6, 0.75, 0.9]:
        for s in [0.35, 0.5]:
            tr = sim.run_signals(g, ext_follow(g, lv, k=k, s=s, n=n, exit_idx=359))
            x = summ(g, tr, "discovery"); rows.append((n, k, s, x["n"], x["avg"], x["sharpe_ann"]))
            pd.DataFrame([{"family": "refine", "name": f"extF ex359 n={n} k={k} s={s}", "d_n": x["n"], "d_avg": x["avg"], "d_sharpe_ann": x["sharpe_ann"]}]).to_csv("variant_log.csv", mode="a", header=False, index=False)
df = pd.DataFrame(rows, columns=["adr_n", "k", "s", "n", "avg", "sh"])
print(df.pivot_table(index="adr_n", columns=["s", "k"], values="avg"))
print(df.pivot_table(index="adr_n", columns=["s", "k"], values="sh"))
