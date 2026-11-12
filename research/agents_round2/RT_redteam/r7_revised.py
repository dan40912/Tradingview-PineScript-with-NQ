"""修正版規則（事先寫下的小網格，只用探索＋驗證挑，測試／近期只報告）。研究用途，不是投資建議。
R_A 開盤動能單一規則：決策時間 T∈{09:35, 09:45, 10:00}；m＝(T−1 分收盤) − 09:30 開盤；|m| ≥ k×ADR10（k∈{0.10,0.15,0.20}）
     → T 開盤順勢進場；止損 0.25×ADR10（上限 145）；出場 {抱到 15:55, 移 100}。共 18 種。
R_B 聯集（不加新參數）：當天有 C1 就做 C1（09:35）；沒有 C1 才看 TD10（10:00）。
R_C 同意版：C1 訊號且 10:00 時 TD10 投票不反向（沒有反向強票 ≥2）才保留——不做；只做 R_A／R_B，避免再加參數。
"""
import os, numpy as np, pandas as pd
import rt_core as R
out = []
def P(s=""): print(s); out.append(str(s))
def pp(T): return "/".join(f"{T[T.P == p].pnl.mean():+.0f}" if (T.P == p).any() else "–" for p in R.PER)
def ra_sigs(T, k, trail):
    s = []
    for i in range(R.N):
        if not R.OKDAY[i] or not np.isfinite(R.ADR10[i]): continue
        c = R.C[i, T - 1]
        if np.isnan(c): continue
        m = c - R.O[i, R.R0]
        if m == 0 or abs(m) < k * R.ADR10[i]: continue
        s.append(dict(i=i, e=T, d=1 if m > 0 else -1, sl=min(max(0.25 * R.ADR10[i], 10), 145), trail=trail, trig=trail))
    return s
rows = []
for T in [935, 945, 960]:
    for k in [0.10, 0.15, 0.20]:
        for tr in [None, 100]:
            X = R.run(ra_sigs(T, k, tr))
            s = {p: X[X.P == p] for p in R.PER}
            rows.append(dict(T=T, k=k, exit="15:55" if tr is None else "移100", **{p + "_n": len(s[p]) for p in R.PER},
                             **{p: s[p].pnl.mean() for p in R.PER}, 全期=X.pnl.sum(), 回撤=R.dd(X.groupby("date").pnl.sum().values)))
Z = pd.DataFrame(rows); Z["ev"] = np.minimum(Z.探索, Z.驗證)
Z.to_csv(os.path.join(R.OUT, "rt_revised_grid.csv"), index=False)
P(Z[["T", "k", "exit", "探索", "驗證", "測試", "近期", "測試_n", "全期", "回撤", "ev"]].to_string(index=False, float_format=lambda v: f"{v:.2f}" if abs(v) < 1 else f"{v:+.0f}"))
P(f"18 種：探索＋驗證都賺 {int(((Z.探索>0)&(Z.驗證>0)).sum())}，三段都賺 {int(((Z.探索>0)&(Z.驗證>0)&(Z.測試>0)).sum())}")
best = Z.sort_values("ev", ascending=False).iloc[0]
P(f"只用探索＋驗證（兩段取小）挑出：T={best['T']} k={best['k']} 出場={best['exit']}，測試 {best['測試']:+.0f}")
# R_B 聯集
F = R.td_features(960); dv, _ = R.td_signal(F, 0.8, 3)
c1 = R.run(R.c1_signals()); days_c1 = set(c1.i)
tdx = R.run([s for s in R.td_sigs(dv) if s["i"] not in days_c1])
U = pd.concat([c1, tdx]).sort_values(["date", "e"])
P(f"C1 天數 {len(days_c1)}，TD10 天數 {int((dv != 0).sum())}，兩者同一天 {len(days_c1 & set(np.where(dv != 0)[0]))}，同天且同方向 {sum(1 for i in days_c1 & set(np.where(dv != 0)[0]) if c1[c1.i == i].d.iloc[0] == dv[i])}")
P(R.line(U, "R_B 聯集 C1→否則 TD10"))
P(R.line(R.run(R.td_sigs(dv)), "TD10 單獨")); P(R.line(c1, "C1 單獨"))
# 選中的 R_A 及其壓力測試
sel = R.run(ra_sigs(int(best["T"]), float(best["k"]), None if best["exit"] == "15:55" else 100))
P(R.line(sel, f"R_A 選中"))
for nm, kw in [("成本加倍", dict(cost=96.0)), ("進場不利價", dict(entry="adverse")), ("停損差 3 點", dict(slip_stop=3.0))]:
    P(R.line(R.run(ra_sigs(int(best["T"]), float(best["k"]), None if best["exit"] == "15:55" else 100), **kw), f"R_A 選中 {nm}"))
dy = sel.groupby("date").pnl.sum().sort_values(ascending=False)
P(f"R_A 選中：前 5 天占 {100*dy.head(5).sum()/dy.sum():.0f}%，拿掉前 5 天 {dy.iloc[5:].sum():+,.0f}")
# 半口版（5 MNQ）＋基準（Pine 式記帳）
B = R.base_trades()
for nm, X in [("C1", c1), ("TD10", R.run(R.td_sigs(dv))), ("R_B 聯集", U), ("R_A 選中", sel)]:
    for size in [1.0, 0.5]:
        Y = X.copy(); Y["pnl"] *= size; Y["worst"] *= size; Y["src"] = nm; Y["xsched"] = Y.x
        for acc in ["sched", "real"]:
            Bk = R.book(pd.concat([B, Y]), acc)
            P(R.line(Bk, f"基準＋{nm}×{size}口 [{'books17' if acc=='sched' else 'Pine'}]"))
open(os.path.join(R.OUT, "rt_revised.txt"), "w").write("\n".join(out))
