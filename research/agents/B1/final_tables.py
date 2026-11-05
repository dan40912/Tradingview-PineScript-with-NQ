from common import *
import candidates as C
g, lv = load("NQ")
i2 = sim.run_signals(g, strategies.i2_v1_signals(g, lv)); i2d = set(i2.i)
allf = {"I2_v1": lambda g, lv: strategies.i2_v1_signals(g, lv), **C.CANDIDATES, **C.REJECTED, **C.LEADS}
rows = []
for k, fn in allf.items():
    tr = sim.run_signals(g, fn(g, lv))
    for sp in ("discovery", "validation"):
        s = summ(g, tr, sp); t = tr[tr.split == sp]
        on, off = t[t.i.isin(i2d)], t[~t.i.isin(i2d)]
        rows.append({"name": k, "split": sp, **{c: s.get(c) for c in ["n", "win%", "avg", "total", "maxdd", "ret/dd", "sharpe_ann"]},
                     "I2days_n": len(on), "I2days_avg": round(on.pnl.mean()) if len(on) else None,
                     "nonI2_n": len(off), "nonI2_avg": round(off.pnl.mean()) if len(off) else None})
R = pd.DataFrame(rows); print(R.to_string(index=False))
R.to_csv(os.path.join(HERE, "final_tables.csv"), index=False)
