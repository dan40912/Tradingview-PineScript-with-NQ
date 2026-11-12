"""趨勢日判斷器（單一特徵）：在 10:00／10:30 用已知資料預測「剩下到 15:55」的方向與幅度。
交易化：決策時間下一分鐘開盤進場（=O[t]），15:55 收盤出場，1 口 NQ，扣 $48；不設停損（純方向測量）。
另做 +/− 145 停損版本。只看探索段挑方向（順勢或反向）與門檻，驗證段確認，測試／近期只報告。
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
D = pd.read_pickle(os.path.join(HERE, "days.pkl"))
fm = pd.read_pickle(os.path.join(HERE, "fivemin.pkl"))
key = fm.i.values * 288 + fm.b.values
for tag, t in [("10", 960), ("1030", 990)]:
    want = np.arange(len(D)) * 288 + (t // 5 - 1)
    pos = np.searchsorted(key, want, side="right") - 1
    for c in ["rsi", "macdh", "ema_x", "ema_slope", "adx", "di", "bbw_rel", "bb_pos", "atr5_rel", "rvol30"]:
        D[f"k{c}_{tag}"] = fm[c].values[pos]
D.to_pickle(os.path.join(HERE, "days2.pkl"))
D = D[D.tradable & D.has_prev & D.y_10.notna() & D.y_1030.notna()].copy()
PER = ["探索", "驗證", "測試", "近期"]
DIRF = {  # 有方向性的特徵（正＝偏多）
    "ret": "開盤到現在報酬/ATR", "effi": "方向效率(帶號)", "vwap_d": "價格-VWAP", "above_vw": "在VWAP上方時間比例",
    "last_ret": "最近一段報酬", "clv": "區間內位置-0.5", "break_pdh": "突破前高(+)/前低(-)", "break_on": "突破夜盤高/低",
    "es_ret": "ES開盤後報酬", "nq_es_rs": "NQ相對ES強弱", "same_dir": "NQ/ES同向", "krsi": "5分RSI-50", "kmacdh": "5分MACD柱",
    "kema_x": "5分EMA20-50", "kema_slope": "5分EMA20斜率", "kdi": "5分DI+-DI-", "kbb_pos": "5分布林位置"}
DAYF = {"gap": "開盤缺口", "on_ret": "夜盤報酬", "p_dret": "前日漲跌", "p_clv": "前日收盤位置-0.5", "p_macdh_d": "日MACD柱",
        "p_emax_d": "日EMA20-50", "p_emaslope_d": "日EMA斜率", "p_ret5_d": "5日報酬", "p_ret20_d": "20日報酬", "p_rsi_d": "日RSI-50",
        "p_di_d": "日DI", "open_va": "開在VA下(+)/上(-)", "open_vs_pdhl": "開在前高上(+)/前低下(-)"}
center = {"clv": 0.5, "krsi": 50, "p_clv": 0.5, "p_rsi_d": 50}
rows = []
def stat(dirv, y, mask):
    out = {}
    for p in PER:
        m = mask & (D.P == p).values & (dirv != 0)
        pnl = 20 * dirv[m] * y[m] - 48
        out[f"{p}_n"] = int(m.sum()); out[f"{p}_acc"] = round(100 * (np.sign(y[m]) == dirv[m]).mean(), 1) if m.sum() else np.nan
        out[f"{p}_avg"] = round(pnl.mean(), 0) if m.sum() else np.nan
    return out
ntest = 0
for tag in ["10", "1030"]:
    y = D[f"y_{tag}"].values
    feats = [(f"{k}_{tag}", v, center.get(k, 0)) for k, v in DIRF.items()] + [(k, v, center.get(k, 0)) for k, v in DAYF.items()]
    for col, name, c0 in feats:
        x = D[col].values - c0
        disc = (D.P == "探索").values
        for q in [0.0, 0.33, 0.5, 0.67]:   # |x| 超過探索段分位數才判斷
            thr = np.nanquantile(np.abs(x[disc]), q) if q > 0 else 0
            for mode, s in [("順", 1), ("反", -1)]:
                dirv = np.where(np.abs(x) > thr, s * np.sign(np.nan_to_num(x)), 0)
                r = dict(time=tag, feat=col, name=name, q=q, mode=mode, **stat(dirv, y, np.isfinite(x)))
                rows.append(r); ntest += 1
R = pd.DataFrame(rows)
R["pass_ev"] = (R["探索_avg"] > 0) & (R["驗證_avg"] > 0)
R["pass_all"] = R.pass_ev & (R["測試_avg"] > 0)
R.to_csv(os.path.join(HERE, "out_a1_single.csv"), index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400)
print("測了", ntest, "種單特徵規則")
# 探索段先選：每個特徵挑探索每筆最好的（模式＋門檻），再看驗證、測試
best = R.sort_values("探索_avg", ascending=False).groupby(["time", "feat"]).head(1)
cols = ["time", "feat", "name", "q", "mode"] + [f"{p}_{k}" for p in PER for k in ["n", "acc", "avg"]]
print(best[cols].sort_values("探索_avg", ascending=False).head(40).to_string(index=False))
print("\n探索+驗證都正：", R.pass_ev.sum(), "；三段都正：", R.pass_all.sum())
print(R[R.pass_all][cols].sort_values("驗證_avg", ascending=False).head(40).to_string(index=False))
# 基準：一律做多
for tag in ["10", "1030"]:
    y = D[f"y_{tag}"].values
    print(tag, "一律做多", stat(np.ones(len(D)), y, np.ones(len(D), bool)))
# 幅度：|y|/ATR 和非方向特徵的 Spearman 相關（分段）
print("\n== 幅度預測（|剩餘報酬|/ATR 的 Spearman）")
mag = ["ibw", "rvol", "kadx", "kbbw_rel", "katr5_rel"]
for tag in ["10", "1030"]:
    ya = np.abs(D[f"y_{tag}"]) / D.p_atr10
    for c in [f"{m}_{tag}" for m in mag] + ["p_atr_rel", "p_bbw_d", "p_adx_d", "p_eff", "p_eff3", "on_rng", "p_rvol_d", f"effi_{tag}"]:
        xx = D[c].abs() if c.startswith("effi") else D[c]
        print(tag, f"{c:16s}", " ".join(f"{p}:{xx[D.P == p].corr(ya[D.P == p], method='spearman'):+.2f}" for p in PER))
