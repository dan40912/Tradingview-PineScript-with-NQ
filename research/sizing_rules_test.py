import sys, pickle, numpy as np, pandas as pd
import grid, strat
nq = pickle.load(open("NQ_days2.pkl", "rb"))
imp_es = dict(last=89, h1=False, tp=55, sl=145, xi=379, es=grid.ES)
T1 = dict(ib=29, smin=30, smax=80, rr=2.0, xi=179, cond="outpd", be=1.0)
def gen(f): return {d["date"]: t | {"date": d["date"]} for d in nq for t in [f(d)] if t}
A = gen(lambda d: strat.i2(d, **imp_es)); B = gen(lambda d: strat.trend(d, **T1))
def port(size_fn, allow=lambda t: True, mnq_cost=4.8):
    out = []
    for d in sorted(set(A) | set(B)):
        busy, n, pl, losses = -1, 0, 0.0, 0
        for t in sorted([x[d] for x in (A, B) if d in x], key=lambda t: t["ei"]):
            if not allow(t) or t["ei"] <= busy or n >= 3 or pl <= -3000 or losses >= 2: continue
            q = size_fn(t); pts = (t["pnl"] + 48) / 20                         # 每口 NQ 的點數損益
            p = q * (pts * 2 - mnq_cost); out.append(dict(date=d, pnl=p, q=q, setup=t["setup"]))
            busy = t["xi"]; n += 1; pl += p; losses += p < 0
    return pd.DataFrame(out)
def line(name, df):
    x = df[df.date <= "2026-09-30"]; c = x.pnl.cumsum().values; dd = (np.maximum.accumulate(np.r_[0, c])[1:] - c).max()
    y = x.groupby(x.date.str[:4]).pnl.sum(); o = df[df.date >= "2026-10"]; mo = x.groupby(x.date.str[:7]).pnl.sum()
    print(f"{name:<44} {len(x):3d} 筆 平均口數 {x.q.mean():4.1f}｜合計 {x.pnl.sum():+8,.0f} 回撤 {dd:7,.0f} 比 {x.pnl.sum()/dd:4.1f} 虧錢月 {(mo<0).sum():2d}｜"
          f"{y.get('2024',0):+7,.0f}/{y.get('2025',0):+7,.0f}/{y.get('2026',0):+7,.0f}｜10月 {o.pnl.sum():+,.0f}")
print("— 2) 口數：固定口數 vs 固定每筆風險（MNQ，每口每點 $2，每口成本 $4.8）—")
line("固定 10 口 MNQ（＝1 口 NQ）", port(lambda t: 10))
for R in (1000, 1500, 2000):
    line(f"每筆風險固定 ${R}（口數＝${R}÷停損金額）", port(lambda t, R=R: max(1, int(R // (t["risk"] * 2)))))
print("\n— 3) 套用 day-trader-skill 的預設風控 —")
line("我們的組合 D（基準，10 口 MNQ）", port(lambda t: 10))
line("reward:risk ≥ 1（I2 的 55/145＝0.38 會被擋）", port(lambda t: 10, allow=lambda t: t["setup"] != "I2"))
line("不能放空（allow_short=false）", port(lambda t: 10, allow=lambda t: t["side"] == 1))
line("兩個都套（只剩順勢做多）", port(lambda t: 10, allow=lambda t: t["setup"] != "I2" and t["side"] == 1))
