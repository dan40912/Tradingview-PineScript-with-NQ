"""檢查 1（偷看／執行假設的敏感度）、4（成本與執行）、5（集中度）。"""
import os, numpy as np, pandas as pd
import rt_core as R
F = R.td_features(960); dv, thr = R.td_signal(F, 0.8, 3)
c1s = R.c1_signals(); tds = R.td_sigs(dv)
rows = []
def rec(name, T):
    print(R.line(T, name))
    s = {P: R.stats(T[T.P == P]) for P in R.PER}; a = R.stats(T)
    rows.append(dict(name=name, **{f"{P}_avg": s[P]["avg"] for P in R.PER}, **{f"{P}_tot": s[P]["tot"] for P in R.PER}, 全期=a["tot"], 回撤=a["mdd"]))
V = [("原版", {}), ("成本加倍 $96", dict(cost=96.0)), ("進場用該分鐘不利價(多 high/空 low)", dict(entry="adverse")),
     ("停損成交再差 2 點", dict(slip_stop=2.0)), ("停損成交再差 5 點", dict(slip_stop=5.0)),
     ("移動停損同分鐘先高後低(另一假設)", dict(pess_trail=True)),
     ("全部壓力：不利進場＋停損差 3 點＋成本 $96", dict(entry="adverse", slip_stop=3.0, cost=96.0))]
for nm, base in [("C1", c1s), ("TD10", tds)]:
    print("=====", nm)
    for vn, kw in V:
        rec(f"{nm} {vn}", R.run(base, **kw))
    for dl in (1, 2, 5):
        rec(f"{nm} 晚 {dl} 分進場", R.run([dict(s, e=s["e"] + dl) for s in base]))
    for xs, xn in [(930 + 380, "15:50"), (930 + 389, "15:59")]:
        rec(f"{nm} {xn} 出場", R.run(base, xs=xs))
pd.DataFrame(rows).to_csv(os.path.join(R.OUT, "rt_robust.csv"), index=False)

# 進場分鐘的波動（10:00 常有數據）：比較 09:35 與 10:00 那一分鐘的 H-L 與不利成交差
for nm, slot in [("09:35", 935), ("10:00", 960)]:
    rng = R.H[:, slot] - R.L[:, slot]; okd = R.OKDAY
    adv = np.abs(R.H[:, slot] - R.O[:, slot]) / 2 + np.abs(R.O[:, slot] - R.L[:, slot]) / 2
    print(f"{nm} 那一分鐘 H−L：中位 {np.nanmedian(rng[okd]):.1f} 點，90% {np.nanquantile(rng[okd], .9):.1f}，平均 {np.nanmean(rng[okd]):.1f}；開盤到不利端平均 {np.nanmean(adv[okd]):.1f} 點")

# ---- 集中度 ----
def conc(T, nm):
    dy = T.groupby("date").pnl.sum().sort_values(ascending=False)
    top5 = dy.head(5).sum(); tot = dy.sum()
    T2 = T[~T.date.isin(dy.head(5).index)]
    T3 = T[~T.date.isin(dy.head(10).index)]
    print(f"\n[{nm}] 合計 {tot:+,.0f}；前 5 大獲利日 {top5:+,.0f}（占 {100*top5/tot:.0f}%）：{', '.join(f'{d} {v:+,.0f}' for d, v in dy.head(5).items())}")
    print(f"  拿掉前 5 天：{T2.pnl.sum():+,.0f}，每筆 {T2.pnl.mean():+.0f}；拿掉前 10 天：{T3.pnl.sum():+,.0f}")
    for P in R.PER:
        x = T[T.P == P]; dp = x.groupby("date").pnl.sum().sort_values(ascending=False)
        print(f"  {P}: 合計 {dp.sum():+,.0f}，前 3 天 {dp.head(3).sum():+,.0f}，拿掉前 3 天 {dp.iloc[3:].sum():+,.0f}")
    # 自助法：每筆平均的 95% 區間（全期與測試）
    rng = np.random.default_rng(1)
    for P in ["測試", None]:
        p = (T if P is None else T[T.P == P]).pnl.values
        bs = [rng.choice(p, len(p)).mean() for _ in range(5000)]
        print(f"  {P or '全期'} 每筆 {p.mean():+.0f}，自助 95% 區間 {np.quantile(bs, .025):+.0f}～{np.quantile(bs, .975):+.0f}，P(每筆≤0)={np.mean(np.array(bs) <= 0):.3f}")
    x = T[T.date >= "2026-01-01"].copy(); x["m"] = x.date.str[:7]
    print("  2026 月度：", x.groupby("m").pnl.agg(["size", "sum"]).astype(int).apply(lambda r: f"{r['size']}筆 {r['sum']:+,}", axis=1).to_dict())
    # 勝率、賺賠比
    w = T[T.pnl > 0].pnl; l = T[T.pnl <= 0].pnl
    print(f"  勝率 {100*len(w)/len(T):.1f}%，平均賺 {w.mean():+.0f}，平均賠 {l.mean():+.0f}，最大單筆 {T.pnl.max():+,.0f}")
c1 = R.run(c1s); td = R.run(tds)
conc(c1, "C1"); conc(td, "TD10 q80k3")
# 做多／做空分開
for nm, T in [("C1", c1), ("TD10", td)]:
    for d in (1, -1):
        x = T[T.d == d]
        print(nm, "做多" if d == 1 else "做空", " | ".join(f"{P} {int((x.P == P).sum())}筆 {x[x.P == P].pnl.mean():+.0f}" for P in R.PER))
