"""Robustness info for C1 (threshold neighbours) on both splits — reported only, not used to re-tune."""
from common import *
import candidates as C
g, lv = load("NQ")
i2 = sim.run_signals(g, strategies.i2_v1_signals(g, lv))
rel = np.array([C._extreme_bin_rel(g, g.prev_rth_day(r.i), +1 if r.dir == -1 else -1) for r in i2.itertuples()])
h1 = np.array([((lv.open_0930[r.i] - lv.vah[r.i]) if r.dir == -1 else (lv.val[r.i] - lv.open_0930[r.i])) <= lv.va_width[r.i] for r in i2.itertuples()])
for th in (0.15, 0.2, 0.25, 0.3, 0.35):
    line = f"th={th}: "
    for sp in ("discovery", "validation"):
        m = (rel < th) & (i2.split == sp).values
        t = i2[m]; s = metrics.summary(t, n_days=ndays(g, sp))
        line += f"{sp[:4]} n={s['n']} avg={s['avg']} dd={s['maxdd']} sh={s['sharpe_ann']} | "
    print(line)
for name, mm in (("H1 only", h1), ("C1&H1", h1 & (rel < 0.25))):
    line = f"{name}: "
    for sp in ("discovery", "validation"):
        t = i2[mm & (i2.split == sp).values]; s = metrics.summary(t, n_days=ndays(g, sp))
        line += f"{sp[:4]} n={s['n']} avg={s['avg']} dd={s['maxdd']} sh={s['sharpe_ann']} | "
    print(line)
# what do dropped trades look like
i2["rel"] = rel
print(i2[rel >= 0.25][["date", "dir", "reason", "pnl", "rel", "split"]].to_string())
