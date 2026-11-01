import pickle, numpy as np, pandas as pd, winrate_opt as W
pd.set_option("display.width", 260); pd.set_option("display.max_colwidth", 90)
R = pd.read_pickle("winrate_grid.pkl").drop_duplicates(subset=["family", "選擇期筆", "選擇期合計", "10月合計"])
R["開發筆"] = R["2024筆"] + R["2025筆"]; R["開發勝率"] = (R["2024勝率"] * R["2024筆"] + R["2025勝率"] * R["2025筆"]) / R["開發筆"]
print("=== 1) 巢狀檢查：只用 2024–2025 挑勝率最高（兩年期望值都為正、≥40 筆），2026-01～09 是真正沒看過的資料 ===")
dev = R[(R["2024每筆"] > 0) & (R["2025每筆"] > 0) & (R["開發筆"] >= 40)]
for fam in ("I2", "順勢突破", "參考點"):
    x = dev[dev.family == fam].sort_values(["開發勝率", "2024合計"], ascending=False).head(4)
    print(f"\n{fam}"); print(x[["name", "開發筆", "開發勝率", "2026驗證筆", "2026驗證勝率", "2026驗證每筆", "10月筆", "10月勝率", "10月合計"]].round(3).to_string(index=False))
print("\n=== 2) 勝率和期望值的取捨（I2，ES 同向＋H1＋ORB 不反向，11:00 截止、15:49 出場；只換 TP/SL）===")
for tp, sl in ((30, 145), (40, 145), (55, 145), (30, 100), (40, 100), (55, 100)):
    r = R[R.name == f"I2 最晚89 h1=1 es=1 vwap=0 orb=1 rvol≥0 +{tp}/-{sl} 出場379 保本=None"]
    if len(r): r = r.iloc[0]; print(f"  +{tp}/−{sl}：{r['選擇期筆']:.0f} 筆 勝率 {r['選擇期勝率']:.1%} 每筆 {r['選擇期每筆']:+.0f} 合計 {r['選擇期合計']:+,.0f} 回撤 {r['選擇期回撤']:,.0f}｜2024/2025/2026 勝率 {r['2024勝率']:.0%}/{r['2025勝率']:.0%}/{r['2026驗證勝率']:.0%}")
# ---------- 3) 滾動歷史回放（Time Machine 的 history）：這條規則最近 N 筆的表現好才做 ----------
nq = W.nq
def trades(fam, kw): return pd.DataFrame([t | {"date": d["date"]} for d in nq for t in [W.FN[fam](d, **kw)] if t])
def stats(df, lab=""):
    x = df[df.date <= "2026-09-30"]; c = x.pnl.cumsum().values; dd = (np.maximum.accumulate(np.r_[0, c])[1:] - c).max() if len(x) else 0
    y = x.groupby(x.date.str[:4]).pnl.apply(lambda s: (s > 0).mean()); o = df[df.date >= "2026-10"]
    return f"{lab:<44} {len(x):3d} 筆 勝率 {(x.pnl > 0).mean():.1%} 每筆 {x.pnl.mean():+5.0f} 合計 {x.pnl.sum():+7,.0f} 回撤 {dd:6,.0f}｜{'/'.join(f'{v:.0%}' for v in y)}｜10月 {len(o)} 筆 {(o.pnl > 0).sum()} 勝 {o.pnl.sum():+,.0f}"
def rolling(df, n=10, rule="mean"):
    keep = []
    for k in range(len(df)):
        past = df.pnl.iloc[max(0, k - n):k]                       # 只用之前已經結束的訊號（不論有沒有真的做）
        ok = len(past) < n or (past.mean() > 0 if rule == "mean" else (past > 0).mean() >= rule)
        keep.append(ok)
    return df[keep]
I2H = dict(last=89, h1=True, es=True, vw=False, orbf=True, vol=0, tp=30, sl=145, xi=379, be=None)
I2D = dict(last=89, h1=False, es=True, vw=False, orbf=False, vol=0, tp=55, sl=145, xi=379, be=None)
TRH = dict(rr=0.5, stop="atr2.5", xi=379, vw=False, vol=0, be=None)
TRD = dict(rr=2.0, stop="ib", xi=179, vw=False, vol=0, be=1.0)
REF = dict(which=1, vw=True, vol=0, tp=30, sl=145, be=None)
print("\n=== 3) 滾動歷史回放：最近 N 個訊號的平均損益 > 0（或勝率 ≥ 門檻）才做 ===")
for lab, fam, kw in (("I2 高勝率（+30）", "I2", I2H), ("順勢高勝率（0.5R）", "順勢突破", TRH), ("順勢 D-2（2R）", "順勢突破", TRD), ("前低多＋VWAP（+30）", "參考點", REF)):
    df = trades(fam, kw).sort_values("date").reset_index(drop=True)
    print(stats(df, lab))
    for n, rule in ((10, "mean"), (20, "mean"), (10, 0.8)):
        print(stats(rolling(df, n, rule), f"   ＋最近 {n} 筆 {'平均>0' if rule == 'mean' else f'勝率≥{rule:.0%}'}"))
pickle.dump(dict(I2H=I2H, I2D=I2D, TRH=TRH, TRD=TRD, REF=REF), open("winrate_cfg.pkl", "wb"))
