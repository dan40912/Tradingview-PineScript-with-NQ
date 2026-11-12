"""(1) 指標過濾既有進場點（基準「順勢多點」六個點）。研究用途，不是投資建議。
每個過濾＝「保留 特徵 >= 門檻」或「保留 特徵 <= 門檻」，門檻＝探索段該組候選的分位數（10/20/33/50/67/80/90%）。
有方向的特徵先乘上交易方向（＝順著交易方向的程度）。
用到 09:30 以後資訊的特徵（缺口、夜盤區間、10:00/10:30 盤中）只套在進場時間已經知道它的交易上，其餘一律保留。
套用範圍：整個組合一起、或只套一個點（其他點不變）。
通過（探索＋驗證）：兩段合計都 >= 基準，且兩段最大回撤都不比基準差（|DD| <= 基準）。
高原：同一特徵、同方向、相鄰分位數也通過（至少一邊）。測試段只報告。
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import cbook as B
F = B.F
DIRF = ["k_rsi", "k_macdh", "k_ema_x", "k_ema_slope", "k_di", "k_bb_pos", "p_rsi_d", "p_macdh_d", "p_emax_d", "p_emaslope_d",
        "p_di_d", "p_dret", "p_clv", "p_ret5_d", "p_ret20_d", "gap", "on_ret", "ret_10", "effi_10", "above_vw_10", "ret_1030", "effi_1030",
        "nq_es_rs_10", "nq_es_rs_1030", "kema_slope_10"]
NDF = ["k_adx", "k_bbw_rel", "k_atr5_rel", "k_rvol30", "p_adx_d", "p_bbw_d", "p_atr_rel", "p_rvol_d", "p_eff", "p_eff3", "on_rng", "vol",
       "ibw_10", "rvol_10", "ibw_1030", "rvol_1030"]
CENTER = {"k_rsi": 50, "p_rsi_d": 50, "p_clv": 0.5, "k_bb_pos": 0}
KNOWN = {"gap": 930, "on_ret": 930, "on_rng": 930}   # 進場時間 >= 這格才知道
for c in DIRF + NDF:
    if c.endswith("_10"): KNOWN[c] = 960
    if c.endswith("_1030"): KNOWN[c] = 990
# kema_slope_10 不在 cand 欄位裡（days2 才有），補上
D2 = pd.read_pickle(os.path.join(HERE, "days2.pkl"))
F["kema_slope_10"] = F.date.map(D2.set_index("date").kema_slope_10)
VAL = {}
for c in DIRF:
    VAL[c] = (F[c] - CENTER.get(c, 0)) * F.dir
for c in NDF:
    VAL[c] = F[c]
base = B.brief(B.book())
print("基準", {k: (round(v[0]), round(v[1]), v[2]) for k, v in base.items()})
QS = [0.1, 0.2, 0.33, 0.5, 0.67, 0.8, 0.9]
rows = []
inbook = np.zeros(len(F), bool)
for n in B.BASE: inbook |= B.PTS[n].fillna(False).values
for scope in ["全部"] + B.BASE:
    smask = inbook if scope == "全部" else B.PTS[scope].fillna(False).values
    for c in DIRF + NDF:
        v = VAL[c].values
        known = F.entry_slot.values >= KNOWN.get(c, 0)
        ref = v[smask & known & (F.P == "探索").values]
        ref = ref[np.isfinite(ref)]
        if len(ref) < 20:
            continue
        for q in QS:
            thr = np.quantile(ref, q)
            for side in [">=", "<="]:
                ok = (v >= thr) if side == ">=" else (v <= thr)
                drop = smask & known & np.isfinite(v) & ~ok
                b = B.brief(B.book(keep=~drop))
                r = dict(scope=scope, feat=c, q=q, side=side, thr=round(thr, 3))
                for p in B.PER + ["全期"]:
                    r[f"{p}_tot"] = round(b[p][0]); r[f"{p}_dd"] = round(b[p][1]); r[f"{p}_n"] = b[p][2]
                rows.append(r)
R = pd.DataFrame(rows)
for p in B.PER + ["全期"]:
    R[f"{p}_dtot"] = R[f"{p}_tot"] - round(base[p][0]); R[f"{p}_ddd"] = R[f"{p}_dd"] - round(base[p][1])
R["pass_ev"] = (R.探索_dtot >= 0) & (R.驗證_dtot >= 0) & (R.探索_ddd >= 0) & (R.驗證_ddd >= 0) & ((R.探索_dtot + R.驗證_dtot) > 0)
R = R.sort_values(["scope", "feat", "side", "q"]).reset_index(drop=True)
# 高原：相鄰分位數也通過
pl = []
for (s, f, sd), x in R.groupby(["scope", "feat", "side"]):
    pv = x.pass_ev.values
    for k in range(len(x)):
        nb = [pv[j] for j in (k - 1, k + 1) if 0 <= j < len(x)]
        pl.append(bool(pv[k] and any(nb)))
R["plateau"] = pl   # 依排序後的順序
R["pass_test"] = R.pass_ev & (R.測試_dtot >= 0) & (R.測試_ddd >= 0)
R.to_csv(os.path.join(HERE, "out_a2_filters.csv"), index=False)
pd.set_option("display.width", 260); pd.set_option("display.max_rows", 300)
print("測了", len(R), "個過濾變體；探索＋驗證通過", int(R.pass_ev.sum()), "；其中有高原", int((R.pass_ev & R.plateau).sum()),
      "；再加測試也改善（合計不少、回撤不大）", int((R.pass_ev & R.plateau & R.pass_test).sum()))
cols = ["scope", "feat", "q", "side", "thr", "探索_dtot", "探索_ddd", "驗證_dtot", "驗證_ddd", "測試_dtot", "測試_ddd", "近期_dtot", "全期_dtot", "全期_dd", "全期_n"]
print(R[R.pass_ev & R.plateau][cols].sort_values("驗證_dtot", ascending=False).to_string(index=False))
