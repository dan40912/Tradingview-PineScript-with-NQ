"""ONE-SHOT validation of the 3 frozen candidates (candidates.py). No parameter changes after this run.
Outputs discovery vs validation tables, I2 overlap, ES check (net and gross), combos with I2, daily correlations."""
import numpy as np, pandas as pd, json
from a2lib import load, sim, summ, evaluate, i2_days, overlap, combo, metrics, n_tradable
from candidates import ALL_FROZEN as CANDIDATES
pd.set_option("display.width", 200)
g, lv = load("NQ")
i2set, i2tr = i2_days(g, lv)
out = {}
F = ["n", "win%", "avg", "total", "maxdd", "ret/dd", "sharpe_ann"]
rows = []
trs = {"I2v1": i2tr}
for name, fn in CANDIDATES.items():
    trs[name] = sim.run_signals(g, fn(g, lv))
for name, tr in trs.items():
    evaluate(g, tr, name, "VALIDATION_final", log=(name != "I2v1"))
    for sp in ["discovery", "validation"]:
        s = summ(g, tr, sp); rows.append({"strategy": name, "split": sp, **{k: s.get(k) for k in F}})
tab = pd.DataFrame(rows); print(tab.to_string(index=False)); tab.to_csv("results_validation.csv", index=False)
print("\n== I2 overlap ==")
for name in CANDIDATES:
    for sp in ["discovery", "validation"]:
        print(name, sp, overlap(g, trs[name], i2set, sp))
print("\n== by direction / reason (validation) ==")
for name in CANDIDATES:
    v = trs[name][trs[name].split == "validation"]
    print(name, v.groupby("dir").pnl.agg(["count", "mean"]).round(0).to_dict(), v.groupby("reason").pnl.agg(["count", "mean"]).round(0).to_dict())
print("\n== daily pnl correlation (all 2024-2025 tradable days, zeros on flat days) ==")
days = g.dates[g.tradable]
M = pd.DataFrame({k: v.groupby("date").pnl.sum().reindex(days).fillna(0) for k, v in trs.items()})
print(M.corr().round(2))
print("\n== combos (1 contract each, daily sum) ==")
crow = []
for combo_name, keys in [("I2v1", ["I2v1"]), ("I2+ON_SWEEP", ["I2v1", "A2_ON_SWEEP_2R"]), ("I2+TREND", ["I2v1", "A2_TREND_EXT_FOLLOW"]),
                         ("I2+CLOSE_REV", ["I2v1", "A2_CLOSE_REVERSAL_1530"]), ("I2+all3", ["I2v1"] + list(CANDIDATES)), ("all3 (no I2)", list(CANDIDATES))]:
    for sp in ["discovery", "validation"]:
        s = combo(g, [trs[k] for k in keys], sp)
        crow.append({"combo": combo_name, "split": sp, **{k: s.get(k) for k in ["n", "avg", "total", "maxdd", "ret/dd", "sharpe_ann"]}})
ct = pd.DataFrame(crow); print(ct.to_string(index=False)); ct.to_csv("results_combos.csv", index=False)
print("\n== ES robustness (points scaled x0.275; ES cost $108) ==")
gE, lvE = load("ES")
erow = []
for name, fn in CANDIDATES.items():
    t = sim.run_signals(gE, fn(gE, lvE))
    t["gross"] = t.pts * gE.point
    for sp in ["discovery", "validation"]:
        s = summ(gE, t, sp); x = t[t.split == sp]
        erow.append({"strategy": name, "split": sp, "n": s["n"], "net_avg": s["avg"], "gross_avg": round(x.gross.mean(), 1),
                     "net_total": s["total"], "maxdd": s["maxdd"], "sharpe": s["sharpe_ann"]})
et = pd.DataFrame(erow); print(et.to_string(index=False)); et.to_csv("results_es.csv", index=False)
for name, tr in trs.items():
    tr.to_csv(f"trades_{name}.csv", index=False)
