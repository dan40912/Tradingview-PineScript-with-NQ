"""09: Risk-normalised versions and combination with I2 v1 (discovery only)."""
import numpy as np, pandas as pd
import c2lib as L
from lib import sim
import sig
g, lv, i2 = L.load()
im = sig.i2_entry_idx(g, lv)
fam = "09_combo"
run = lambda s: sim.run_signals(g, s)
lvi = lv.set_index("i")
base = {
    "TE_out_s50": sig.trend_ext(g, lv, day="outside", s=0.5, i2map=im),
    "TE_out_s35": sig.trend_ext(g, lv, day="outside", s=0.35, i2map=im),
    "PMP_s35": sig.pm_pullback(g, lv, T=180, s=0.35),
    "PMP_s50": sig.pm_pullback(g, lv, T=180, s=0.5),
}
for k, s in base.items():
    tr = run(s)
    tr["typ"] = np.where(tr.i.map(lvi.i2_dir) == 0, "inside", np.where(tr.date.isin(set(i2.date)), "I2day", "out_noI2"))
    print(k, "corr", L.corr_i2(g, tr, i2, "discovery"),
          tr[tr.split == "discovery"].groupby("typ").pnl.agg(["count", "mean"]).round(0).to_dict())
print()
I = i2[i2.split == "discovery"]
s0 = L.summ(g, I, "discovery")
print("I2 alone", s0)
rows = []
for k, s in base.items():
    for R in [500, 750, 1000, 1500, 2000]:
        tr = run(L.size_risk(s, R))
        a = L.summ(g, tr, "discovery")
        c = L.combo(g, [I, tr], "discovery")
        rows.append({"strat": k, "R": R, "n": a["n"], "avg": a["avg"], "tot": a["total"], "dd": a["maxdd"],
                     "sh": a["sharpe_ann"], "avg_ct": round(tr[tr.split == "discovery"].ct.mean(), 2),
                     "combo_tot": c["total"], "combo_dd": c["maxdd"], "combo_rdd": c["ret/dd"], "combo_sh": c["sharpe_ann"]})
# both together
for R in [500, 750, 1000, 1500]:
    t1 = run(L.size_risk(base["TE_out_s35"], R)); t2 = run(L.size_risk(base["PMP_s35"], R))
    c = L.combo(g, [I, t1, t2], "discovery"); c2 = L.combo(g, [t1, t2], "discovery")
    rows.append({"strat": "TE35+PMP35", "R": R, "n": c2["n"], "avg": c2["avg"], "tot": c2["total"], "dd": c2["maxdd"],
                 "sh": c2["sharpe_ann"], "combo_tot": c["total"], "combo_dd": c["maxdd"], "combo_rdd": c["ret/dd"], "combo_sh": c["sharpe_ann"]})
print(pd.DataFrame(rows).to_string())
