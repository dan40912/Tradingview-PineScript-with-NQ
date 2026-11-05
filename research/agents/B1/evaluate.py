"""Stage B: frozen candidates on discovery + validation (validation looked at ONCE)."""
from common import *
from candidates import CANDIDATES
g, lv = load("NQ")
pf = ind.profile_features(g, lv); ia = ind.intraday_arrays(g)
i2 = sim.run_signals(g, strategies.i2_v1_signals(g, lv)); i2d = set(i2.i)
pd.set_option("display.width", 200)
res = {"I2_v1": i2}
for k, fn in CANDIDATES.items():
    res[k] = sim.run_signals(g, fn(g, lv))
for k, tr in res.items():
    print("=====", k); print(table(g, tr).to_string())
    for sp in ("discovery", "validation"):
        t = tr[tr.split == sp]
        on, off = t[t.i.isin(i2d)], t[~t.i.isin(i2d)]
        print(f"  {sp}: I2-days n={len(on)} avg={on.pnl.mean() if len(on) else float('nan'):.0f} | non-I2-days n={len(off)} avg={off.pnl.mean() if len(off) else float('nan'):.0f}")
    print(metrics.by_period(tr, "year")[["n", "win%", "avg", "total", "maxdd"]].to_string())
# C1 as filter: kept fraction + permutation p per split
print("===== C1 filter stats")
c1 = set(res["B1_I2_volpoor"].i)
for sp in ("discovery", "validation"):
    t = i2[i2.split == sp]; m = t.i.isin(c1).values
    print(f"{sp}: kept {m.sum()}/{len(m)} ({m.mean():.0%}) kept avg {t.pnl[m].mean():.0f} dropped avg {t.pnl[~m].mean():.0f} perm p={perm_p(t.pnl.values, m):.3f}")
# informational (pre-declared 4th look): filter B = price vs 09:30-VWAP <= 2.0 sd in trade direction at entry
print("===== Filter B (informational, not a candidate)")
for sp in ("discovery", "validation"):
    t = i2[i2.split == sp]
    z = np.array([(g.C[r.i, r.entry_slot - 1] - ia["vwap930"][r.i, r.entry_slot - 1]) * r.dir / ia["sd930"][r.i, r.entry_slot - 1] for r in t.itertuples()])
    m = z <= 2.0
    print(f"{sp}: kept {m.sum()}/{len(m)} kept avg {t.pnl[m].mean():.0f} dropped avg {t.pnl[~m].mean():.0f} perm p={perm_p(t.pnl.values, m):.3f}")
    mm = m & t.i.isin(c1).values
    print(f"   B & C1: kept {mm.sum()} avg {t.pnl[mm].mean():.0f}")
import pickle
pickle.dump(res, open(os.path.join(HERE, "_cand_trades.pkl"), "wb"))
