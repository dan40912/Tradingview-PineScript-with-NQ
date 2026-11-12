"""高原（獨立重算）、進場時間敏感度、固定點數滑價、修正版規則回測。"""
import os, numpy as np, pandas as pd
import rt_core as R
out = []
def P(s=""): print(s); out.append(str(s))
def pp(T):
    return "/".join(f"{T[T.P == p].pnl.mean():+.0f}" if (T.P == p).any() else "–" for p in R.PER)
F = R.td_features(960); dv, _ = R.td_signal(F, 0.8, 3)
c1s = R.c1_signals(); tds = R.td_sigs(dv)
# ---- 固定點數滑價 ----
P("== 固定滑價（進場、停損各差 s 點；成本 $48）每筆 探索/驗證/測試/近期")
for s in [0, 1, 2, 4, 8]:
    a = R.run([dict(x, e=x["e"]) for x in c1s], slip_stop=s); a["pnl"] -= 20 * s
    b = R.run(tds, slip_stop=s); b["pnl"] -= 20 * s
    P(f"  s={s}: C1 {pp(a)}（全期 {a.pnl.sum():+,.0f}）｜TD10 {pp(b)}（全期 {b.pnl.sum():+,.0f}）")
# ---- 10:00 那一分鐘貢獻 ----
for nm, base in [("C1", c1s), ("TD10", tds)]:
    T = pd.DataFrame(base)
    mv = np.array([s["d"] * (R.O[s["i"], s["e"] + 2] - R.O[s["i"], s["e"]]) for s in base])
    per = np.array([R.PERIOD[s["i"]] for s in base])
    P(f"{nm}：進場價到 2 分鐘後開盤的順向移動（點）各段平均 " + "/".join(f"{np.nanmean(mv[per == p]):+.1f}" for p in R.PER))
# ---- C1 高原（自己算） ----
P("\n== C1 高原：N（分鐘）× k，止損 0.25ADR、移 100；每筆 探索/驗證/測試/近期")
rows = []
for N_ in [3, 4, 5, 6, 7, 10]:
    for k in [0.10, 0.125, 0.15, 0.175, 0.20]:
        T = R.run(R.c1_signals(N_, k)); s = {p: T[T.P == p].pnl.mean() for p in R.PER}
        rows.append(dict(N=N_, k=k, **s, n_test=int((T.P == "測試").sum())))
Z = pd.DataFrame(rows); P(Z.to_string(index=False, float_format=lambda v: f"{v:.3g}" if abs(v)<1 else f"{v:+.0f}"))
P(f"  30 格：探索＋驗證都賺 {int(((Z.探索>0)&(Z.驗證>0)).sum())}，三段都賺 {int(((Z.探索>0)&(Z.驗證>0)&(Z.測試>0)).sum())}")
rows = []
for sa in [0.15, 0.2, 0.25, 0.3, 0.35]:
    for tr in [None, 70, 100, 150, 200]:
        T = R.run([dict(x, sl=min(max(sa * R.ADR10[x["i"]], 10), 145), trail=tr, trig=tr) for x in c1s])
        rows.append(dict(stop_adr=sa, trail=tr or "無", **{p: T[T.P == p].pnl.mean() for p in R.PER}))
Z = pd.DataFrame(rows); P("== C1 止損×移動停損：\n" + Z.to_string(index=False, float_format=lambda v: f"{v:.3g}" if abs(v)<1 else f"{v:+.0f}"))
# ---- TD10 高原 ----
P("\n== TD10 高原：q × k × 停損；每筆 探索/驗證/測試/近期（筆數 測試）")
rows = []
for q in [0.7, 0.75, 0.8, 0.85, 0.9]:
    for k in [2, 3, 4]:
        d2, _ = R.td_signal(F, q, k)
        for sl in [80, 100, 145]:
            T = R.run(R.td_sigs(d2, sl=sl))
            rows.append(dict(q=q, k=k, sl=sl, **{p: T[T.P == p].pnl.mean() for p in R.PER}, n測試=int((T.P == "測試").sum())))
Z = pd.DataFrame(rows); P(Z.to_string(index=False, float_format=lambda v: f"{v:.3g}" if abs(v)<1 else f"{v:+.0f}"))
P(f"  45 格：三段都賺 {int(((Z.探索>0)&(Z.驗證>0)&(Z.測試>0)).sum())}")
P("\n== TD10 決策時間改變（特徵和進場一起移動）")
for t in [950, 955, 960, 965, 970, 975]:
    Ft = R.td_features(t); d2, _ = R.td_signal(Ft, 0.8, 3)
    T = R.run(R.td_sigs(d2, t=t)); P(f"  {R.hh(t) if hasattr(R,'hh') else t}: {pp(T)}（全期 {len(T)} 筆 {T.pnl.sum():+,.0f}）")
open(os.path.join(R.OUT, "rt_plateau.txt"), "w").write("\n".join(out))
