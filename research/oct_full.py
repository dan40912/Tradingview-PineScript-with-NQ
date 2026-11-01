"""完整 2026-10（需要先跑 build_cache.py --yahoo，把 10/20～10/30 接上）：每個策略的 10 月逐筆和合計。"""
import pickle, numpy as np, pandas as pd, grid, strat, winrate_opt as W
nq = pickle.load(open("NQ_days2.pkl", "rb")); assert nq[-1]["date"] >= "2026-10-30", "請先跑 build_cache.py --yahoo"
hh = lambda i: f"{9 + (30 + i) // 60:02d}:{(30 + i) % 60:02d}"
T1 = dict(ib=29, smin=30, smax=80, rr=2.0, xi=179, cond="outpd", be=1.0)
cfg = pickle.load(open("winrate_cfg.pkl", "rb"))
S = {"I2 v1": lambda d: strat.i2(d), "前高空": lambda d: strat.pdfade(d, 0), "前低多": lambda d: strat.pdfade(d, 1),
     "D-1 I2（ES同向）": lambda d: strat.i2(d, last=89, tp=55, sl=145, xi=379, es=grid.ES),
     "D-1＋ORB": lambda d: W.i2x(d, **dict(cfg["I2D"], orbf=True)), "D-2 順勢突破": lambda d: strat.trend(d, **T1),
     "I2 高勝率（+30）": lambda d: W.i2x(d, **cfg["I2H"]), "前低多＋VWAP（+30）": lambda d: W.refx(d, **cfg["REF"])}
G = {k: {d["date"]: t | {"date": d["date"]} for d in nq for t in [f(d)] if t} for k, f in S.items()}
def port(names, a="2026-10-01", b="2026-10-31"):
    out = []
    for dt in sorted(set().union(*[G[n].keys() for n in names])):
        if not a <= dt <= b: continue
        busy, k, pl, losses = -1, 0, 0.0, 0
        for t in sorted([G[n][dt] for n in names if dt in G[n]], key=lambda t: t["ei"]):
            if t["ei"] <= busy or k >= 3 or pl <= -3000 or losses >= 2: continue
            out.append(t); busy = t["xi"]; k += 1; pl += t["pnl"]; losses += t["pnl"] < 0
    return pd.DataFrame(out)
print("=== 2026-10 全月（10/1～10/30，22 個交易日）===")
for lab, names in (("現在指標（I2＋前高空＋前低多）", ["I2 v1", "前高空", "前低多"]), ("只做 I2 v1", ["I2 v1"]), ("組合 D", ["D-1 I2（ES同向）", "D-2 順勢突破"]),
                   ("D＋ORB（docs/15 建議）", ["D-1＋ORB", "D-2 順勢突破"]), ("I2 高勝率＋順勢 2R（H2）", ["I2 高勝率（+30）", "D-2 順勢突破"]),
                   ("I2 高勝率＋前低多 VWAP（H3）", ["I2 高勝率（+30）", "前低多＋VWAP（+30）"])):
    o = port(names); h1 = o[o.date <= "2026-10-19"]; h2 = o[o.date >= "2026-10-20"]
    print(f"\n{lab}：{len(o)} 筆 {int((o.pnl > 0).sum()) if len(o) else 0} 勝，合計 {o.pnl.sum() if len(o) else 0:+,.0f}（10/1–19 {h1.pnl.sum() if len(h1) else 0:+,.0f}｜10/20–30 {h2.pnl.sum() if len(h2) else 0:+,.0f}）")
    for _, t in o.iterrows(): print(f"   {t.date} {t.setup:<4} {'多' if t.side == 1 else '空'} {hh(t.ei)} SL {t.risk:.0f} 點 {t.res:<4} 出 {hh(min(t.xi, 389))} {t.pnl:+,.0f}")
