"""挑選（只用探索＋驗證）、基準單依局勢拆開、11/11 當天判斷器輸出、最終四段表。研究用途，不是投資建議。"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import a4_alloc as AL
import a3_trend as A
import cbook as B
D, g = A.D, A.g
pd.set_option("display.width", 260); pd.set_option("display.max_rows", 300)

# 1) 只用探索＋驗證挑 book A 的 (q, k, sl)：探索＋驗證接起來的合計 / 回撤
rows = []
for q in [0.5, 0.67, 0.8]:
    for k in [2, 3, 4, 5]:
        dv = A.td_signal("10", q, k)
        for sl in [100.0, 145.0]:
            T = B.book(extra=AL.td_trades(dv, sl))
            ev = T[T.P.isin(["探索", "驗證"])]
            dy = ev.groupby("date").pnl.sum().sort_index()
            b = B.brief(T)
            rows.append(dict(q=q, k=k, sl=sl, ev_tot=round(ev.pnl.sum()), ev_dd=round(B.dd(dy.values)),
                             ev_rdd=round(ev.pnl.sum() / -B.dd(dy.values), 2),
                             exp=round(b["探索"][0]), val=round(b["驗證"][0]), test=round(b["測試"][0]), test_dd=round(b["測試"][1]),
                             rec=round(b["近期"][0]), full=round(b["全期"][0]), full_dd=round(b["全期"][1])))
S = pd.DataFrame(rows)
T0 = B.book(); ev0 = T0[T0.P.isin(["探索", "驗證"])]; dy0 = ev0.groupby("date").pnl.sum().sort_index()
print("基準 探索＋驗證 合計", round(ev0.pnl.sum()), "回撤", round(B.dd(dy0.values)), "比", round(ev0.pnl.sum() / -B.dd(dy0.values), 2))
print(S.sort_values("ev_rdd", ascending=False).to_string(index=False))
S.to_csv(os.path.join(HERE, "out_a5_select.csv"), index=False)

# 2) 基準單依 10:00 判斷器狀態拆開（只看 10:00 以後進場的）
dv = A.td_signal("10", 0.67, 3)
dmap = dict(zip(g.dates, dv))
T0["td"] = T0.date.map(dmap).fillna(0).astype(int)
late = T0[T0.entry_slot >= 960].copy()
late["state"] = np.where(late.td == 0, "非趨勢日", np.where(late.td == late.dir, "趨勢日同向", "趨勢日逆向"))
print("\n== 基準單（10:00 以後進場）依判斷器狀態，每筆")
print(late.groupby(["state", "P"]).pnl.agg(["size", "mean"]).round(0).unstack().to_string())
early = T0[T0.entry_slot < 960].copy()
early["state"] = np.where(early.td == 0, "非趨勢日", np.where(early.td == early.dir, "趨勢日同向", "趨勢日逆向"))
print("== 基準單（10:00 以前進場，判斷器事後才知道，只是描述）")
print(early.groupby(["state", "P"]).pnl.agg(["size", "mean"]).round(0).unstack().to_string())

# 3) 盤整日只做回歸？——把基準單在「趨勢日」的 10:00 後回歸單拿掉的效果已在 a4（B=A，沒有衝突）。
# 4) 11/11
i = g.dindex["2026-11-11"]
disc = (D.P == "探索").values & A.OK
print("\n== 2026-11-11 判斷器")
for tag in ["10", "1030"]:
    print(" 時間", tag)
    for f in A.VOTE:
        x = D[f"{f}_{tag}"].values
        thr = {q: np.nanquantile(np.abs(x[disc]), q) for q in [0.5, 0.67, 0.8]}
        print(f"   {f:11s} 值 {x[i]:+.3f}  門檻(50/67/80%) {thr[0.5]:.3f}/{thr[0.67]:.3f}/{thr[0.8]:.3f}")
    for q in [0.5, 0.67, 0.8]:
        print("   q", q, "k=2..5 →", [int(A.td_signal(tag, q, k)[i]) for k in [2, 3, 4, 5]])
for tag in ["10", "1030"]:
    print(" 其他", tag, {c: round(D[f"{c}_{tag}"].iloc[i], 3) for c in ["ret", "vwap_d", "nq_es_rs", "same_dir", "es_ret", "ibw", "rvol", "break_pdh", "break_on", "kema_x", "krsi", "kadx"]})
print(" 日線", {c: round(D[c].iloc[i], 3) for c in ["gap", "on_rng", "on_ret", "p_eff", "p_clv", "p_dret", "p_rsi_d", "p_emax_d", "p_ret20_d", "p_adx_d", "open_va", "open_vs_pdhl"]})
print(" 基準當天交易：")
print(T0[T0.date == "2026-11-11"].to_string())
F = B.F
print(" 候選表當天（全部規則）：")
print(F[F.date == "2026-11-11"][["rule", "side", "entry_slot", "to_vwap", "pnl55", "why55"]].to_string())
TT = AL.TT
print(" T1~T3 當天：")
print(TT[TT.date == "2026-11-11"][["rule", "dir", "entry_time", "entry", "exit_time", "exit", "reason", "pnl"]].to_string())
# 每天的判斷器紀錄
rec = pd.DataFrame({"date": D.date, "P": D.P, "TD10_q67k3": dv, "TD10_q80k3": A.td_signal("10", 0.8, 3), "y_10": D.y_10})
rec.to_csv(os.path.join(HERE, "out_td_daily.csv"), index=False)
print("\n== 判斷器 TD10(q.67,k3) 統計：趨勢日比例、方向正確率、剩餘行情 |報酬| 平均（點）")
for p in B.PER:
    m = (rec.P == p) & A.OK
    t = m & (rec.TD10_q67k3 != 0)
    acc = (np.sign(rec.y_10[t]) == rec.TD10_q67k3[t]).mean() * 100
    print(f"  {p} 天數 {m.sum()} 趨勢日 {t.sum()}（{t.sum() / m.sum() * 100:.0f}%）方向正確 {acc:.1f}%｜剩餘|報酬| 趨勢日 {rec.y_10[t].abs().mean():.0f} vs 非趨勢日 {rec.y_10[m & (rec.TD10_q67k3 == 0)].abs().mean():.0f}")
