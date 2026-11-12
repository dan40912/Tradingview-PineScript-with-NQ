"""建議的修正組合壓力測試：基準（Pine 式記帳）＋（C1，否則 TD10）×0.5 口。研究用途，不是投資建議。"""
import os, numpy as np, pandas as pd
import rt_core as R
out = []
def P(s=""): print(s); out.append(str(s))
F = R.td_features(960); dv, _ = R.td_signal(F, 0.8, 3)
def addon(size, slip=0.0, cost=R.COST, entry="open"):
    c1 = R.run(R.c1_signals(), slip_stop=slip, cost=cost, entry=entry); days = set(c1.i)
    td = R.run([s for s in R.td_sigs(dv) if s["i"] not in days], slip_stop=slip, cost=cost, entry=entry)
    U = pd.concat([c1.assign(src="C1"), td.assign(src="TD10")])
    U["pnl"] = (U.pnl - 20 * slip) * size; U["worst"] *= size; U["xsched"] = U.x
    return U
def base(slip=0.0, cost=R.COST, entry="open"):
    B = R.base_trades(slip_stop=slip, cost=cost, entry=entry); B["pnl"] -= 20 * slip; return B
def wk(T):
    w = T.assign(w=pd.to_datetime(T.date).dt.to_period("W")).groupby("w").pnl.sum(); return w.min()
for nm, kw in [("原版", {}), ("進場＋停損各差 2 點", dict(slip=2.0)), ("成本 $96", dict(cost=96.0)), ("差 2 點＋成本 $96", dict(slip=2.0, cost=96.0)),
               ("進場不利價(全部單)", dict(entry="adverse"))]:
    B0 = R.book(base(**kw), "real"); Bx = R.book(pd.concat([base(**kw), addon(0.5, **kw)]), "real")
    P(R.line(B0, f"基準 Pine {nm}")); P(R.line(Bx, f"基準＋聯集×0.5 {nm}"))
    P(f"    最差一週：基準 {wk(B0):+,.0f}；合併 {wk(Bx):+,.0f}")
Bx = R.book(pd.concat([base(), addon(0.5)]), "real"); B0 = R.book(base(), "real")
x = Bx[Bx.date >= "2026-01-01"].assign(m=lambda d: d.date.str[:7]).groupby("m").pnl.sum()
y = B0[B0.date >= "2026-01-01"].assign(m=lambda d: d.date.str[:7]).groupby("m").pnl.sum()
P("2026 月度（基準 → 合併）：" + ", ".join(f"{m} {y.get(m,0):+,.0f}→{x[m]:+,.0f}" for m in x.index))
Bx.to_csv(os.path.join(R.OUT, "rt_trades_proposed.csv"), index=False)
open(os.path.join(R.OUT, "rt_final.txt"), "w").write("\n".join(out))
