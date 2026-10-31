import pickle, numpy as np, pandas as pd, grid, strat
nq = pickle.load(open("NQ_days2.pkl", "rb"))
def run(f): return pd.DataFrame([t | {"date": d["date"]} for d in nq for t in [f(d)] if t])
def line(name, df, w=1.0):
    x = df[df.date <= "2026-09-30"].copy(); x["pnl"] *= w
    c = x.pnl.cumsum().values; dd = (np.maximum.accumulate(np.r_[0, c])[1:] - c).max(); y = x.groupby(x.date.str[:4]).pnl.mean()
    o = df[df.date >= "2026-10"]
    print(f"{name:<40} {len(x):3d} 筆 每筆 {x.pnl.mean():+5.0f} 合計 {x.pnl.sum():+7.0f} 回撤 {dd:6,.0f} 比 {x.pnl.sum()/dd:4.1f}｜{y.get('2024',0):+4.0f}/{y.get('2025',0):+4.0f}/{y.get('2026',0):+4.0f}｜10月 {len(o)} 筆 {o.pnl.sum()*w:+,.0f}")
T1 = dict(ib=29, smin=30, smax=80, rr=2.0, xi=179, cond="outpd", be=1.0)
print("— 新假設 1：用前一天的波動大小過濾（前一天振幅 ÷ 過去 20 天中位數）—")
line("順勢30 基準", run(lambda d: strat.trend(d, **T1)))
for lo, hi in ((0, 1.0), (1.0, 99), (0, 0.8), (0.8, 1.3), (1.3, 99)):
    line(f"順勢30，前一天波動 {lo}~{hi}", run(lambda d: strat.trend(d, **T1, vol=(lo, hi))))
I2k = dict(last=89, h1=False, tp=55, sl=145, xi=379, es=grid.ES)
line("I2(ES同向) 基準", run(lambda d: strat.i2(d, **I2k)))
for lo, hi in ((0, 1.0), (1.0, 99)):
    line(f"I2(ES同向)，前一天波動 {lo}~{hi}", run(lambda d: strat.i2(d, **I2k, vol=(lo, hi))))
print("— 新假設 2：FOMC 當天不做順勢 —")
line("順勢30，FOMC 不做", run(lambda d: strat.trend(d, **T1, fomc=False)))
print("— 新假設 3：順勢最晚進場時間 —")
for last in (90, 120, 150, 210):
    line(f"順勢30，最晚 {9 + (30 + last) // 60}:{(30 + last) % 60:02d}", run(lambda d: strat.trend(d, **{**T1, 'last': last})))
print("— 新假設 4：I2 用一半口數（I2 停損 145 點，順勢只有 30–80 點，讓每筆風險比較平均）—")
D_i2 = run(lambda d: strat.i2(d, **I2k)); D_t = run(lambda d: strat.trend(d, **T1))
def port(dfs, ws, max_losses=2):
    allt = pd.concat([df.assign(w=w) for df, w in zip(dfs, ws)]).sort_values(["date", "ei"]); out = []
    for d, g in allt.groupby("date"):
        busy, pl, losses, n = -1, 0, 0, 0
        for _, t in g.iterrows():
            if t.ei <= busy or losses >= max_losses or pl <= -3000 or n >= 3: continue
            out.append(dict(date=d, pnl=t.pnl * t.w)); busy = t.xi; pl += t.pnl * t.w; n += 1; losses += t.pnl < 0
    return pd.DataFrame(out)
line("組合 D（I2 1 口＋順勢 1 口）", port([D_i2, D_t], [1, 1]))
line("組合 D（I2 0.5 口＋順勢 1 口）", port([D_i2, D_t], [0.5, 1]))
line("組合 D，虧 1 筆當天就停", port([D_i2, D_t], [1, 1], max_losses=1))
