"""11: ONE-SHOT validation of the frozen candidates (2025H2). Never loads 2026."""
import os, importlib.util, numpy as np, pandas as pd
import c2lib as L
from lib import sim, metrics
from candidates import CANDIDATES
g, lv, i2 = L.load()
spec = importlib.util.spec_from_file_location("a2c", os.path.join(L.RESEARCH, "agents", "A2", "candidates.py"))
a2c = importlib.util.module_from_spec(spec); spec.loader.exec_module(a2c)
import candidates as C
lvi = lv.set_index("i")
strats = {"I2v1 (1ct)": i2}
for k, f in CANDIDATES.items():
    strats[k + " (1ct)"] = sim.run_signals(g, f(g, lv, sized=False))
    strats[k + " ($1000 risk)"] = sim.run_signals(g, f(g, lv))
a2 = a2c.trend_ext_follow(g, lv)
strats["A2_TREND_EXT ref (1ct)"] = sim.run_signals(g, a2)
strats["A2_TREND_EXT ref ($1000 risk)"] = sim.run_signals(g, C._size(a2, g, True))
rows = []
for name, tr in strats.items():
    for sp in ["discovery", "validation"]:
        s = L.summ(g, tr, sp)
        ov = L.overlap(tr, i2, sp) if not name.startswith("I2") else {"on_I2": "", "off_I2": ""}
        rows.append({"strategy": name, "split": sp, "n": s["n"], "win%": s.get("win%"), "avg": s.get("avg"),
                     "total": s.get("total"), "maxdd": s.get("maxdd"), "ret/dd": s.get("ret/dd"), "sharpe": s.get("sharpe_ann"),
                     "t": s.get("t"), "corr_I2": (L.corr_i2(g, tr, i2, sp) if not name.startswith("I2") else ""),
                     "on_I2(n,avg)": ov["on_I2"], "off_I2(n,avg)": ov["off_I2"]})
T = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(T.to_string(index=False))
T.to_csv("out_11_validation.csv", index=False)
print("\n=== halves (avg per trade)")
for name, tr in strats.items():
    print(f"{name:32s}", " ".join(f"{h}:{L.summ(g, tr, hal=h).get('avg', np.nan):>7}" for h in ["2024H1", "2024H2", "2025H1", "2025H2"]))
print("\n=== combined with I2 v1 (1 contract) -- daily sum")
crow = []
books = {"I2 alone": [i2],
         "I2 + TE_OUTSIDE($1k)": [i2, strats["C2_TE_OUTSIDE ($1000 risk)"]],
         "I2 + PM_PULLBACK($1k)": [i2, strats["C2_PM_PULLBACK ($1000 risk)"]],
         "I2 + C2_BOOK($1k)": [i2, strats["C2_BOOK ($1000 risk)"]],
         "I2 + A2_TREND_EXT ref($1k)": [i2, strats["A2_TREND_EXT ref ($1000 risk)"]],
         "I2 + A2_TREND_EXT ref(1ct)": [i2, strats["A2_TREND_EXT ref (1ct)"]]}
for b, trs in books.items():
    for sp in ["discovery", "validation", None]:
        s = L.combo(g, trs, sp)
        crow.append({"book": b, "period": sp or "2024-2025", "trades": s["n"], "total": s["total"], "maxdd": s["maxdd"],
                     "ret/dd": s["ret/dd"], "sharpe": s["sharpe_ann"]})
CB = pd.DataFrame(crow); print(CB.to_string(index=False)); CB.to_csv("out_11_combined.csv", index=False)
print("\n=== validation monthly (sized)")
for k in ["C2_TE_OUTSIDE ($1000 risk)", "C2_PM_PULLBACK ($1000 risk)", "I2v1 (1ct)"]:
    tr = strats[k]; v = tr[tr.split == "validation"]
    print(k, v.groupby(v.date.str[:7]).pnl.sum().round(0).to_dict())
for k in ["C2_TE_OUTSIDE ($1000 risk)", "C2_PM_PULLBACK ($1000 risk)"]:
    tr = strats[k]; v = tr[tr.split == "validation"]
    print(k, "val by dir:", v.groupby("dir").pnl.agg(["count", "mean"]).round(0).to_dict(),
          "boot CI val:", metrics.bootstrap_ci(v.pnl.values), "boot CI disc:", metrics.bootstrap_ci(tr[tr.split == "discovery"].pnl.values))
v = strats["C2_PM_PULLBACK (1ct)"]; print(v[v.split == "validation"][["date", "dir", "reason", "pts", "pnl"]].to_string())
for k in ["C2_TE_OUTSIDE (1ct)", "C2_PM_PULLBACK (1ct)", "C2_BOOK (1ct)"]:
    strats[k].to_csv(f"trades_{k.split(' ')[0]}.csv", index=False)
