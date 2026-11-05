"""B1: validation of the 3 frozen candidates (run ONCE after freezing). No 2026 data."""
from util import *
from candidates import CANDIDATES
g, lv, rf, ev = load()
i2 = sim.run_signals(g, strategies.i2_v1_signals(g, lv))
i2days = set(i2.date)
res = {"I2v1 (ref)": i2}
for k, f in CANDIDATES.items():
    res[k] = sim.run_signals(g, f(g, lv))
for k, tr in res.items():
    print("==", k); print(table(tr, g).to_string())
    print(metrics.by_period(tr, "year")[["n", "avg", "total", "maxdd"]].T.to_string())
    for sp in ["discovery", "validation"]:
        t = tr[tr.split == sp]
        a = t[t.date.isin(i2days)]; b = t[~t.date.isin(i2days)]
        print(f"  {sp}: on I2 days n={len(a)} avg={a.pnl.mean():.0f} | non-I2 days n={len(b)} avg={b.pnl.mean():.0f}")
    print("  CI(all):", metrics.bootstrap_ci(tr.pnl.values))
    tr.to_pickle(f"/tmp/claude-0/-home-user-Tradingview-PineScript-with-NQ/d4597cec-8294-562a-9e11-5568b74be6dd/scratchpad/val_{k.split()[0]}.pkl")
# portfolios
def port(names):
    t = pd.concat([res[n] for n in names]); return t
for names in [["I2v1 (ref)", "B2_ORB30_830"], ["B2_I2_TrendSize", "B2_ORB30_830"], ["I2v1 (ref)", "B2_ORB30_830", "B2_GAPF_NoNews"],
              ["B2_I2_TrendSize", "B2_ORB30_830", "B2_GAPF_NoNews"]]:
    print("== PORTFOLIO", names); print(table(port(names), g).to_string())
