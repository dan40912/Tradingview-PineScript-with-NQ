"""檢查 3（選擇偏誤，C）：
(a) C 自己的單特徵表／過濾表／投票表：『探索＋驗證都賺』的變化，測試期也賺的比例。
(b) 投票指標組合的汙染：從 14 個連續型 10:00 指標中任選 5 個（2,002 組），同樣 q80/k3/停損 100/15:55，
    只用探索＋驗證挑，看測試期分布；C 選的那組排第幾。
特徵值取自 C 的 days2.pkl（資料，不用 C 的程式）；其中 5 個投票指標已經被 rt_core 獨立重算過、交易逐筆一致。"""
import os, itertools, numpy as np, pandas as pd
import rt_core as R
C_DIR = os.path.join(os.path.dirname(R.OUT), "C_indicators")
out = []
def P(s): print(s); out.append(s)
# ---- (a) ----
a1 = pd.read_csv(os.path.join(C_DIR, "out_a1_single.csv"))
for t in [10, 1030]:
    x = a1[a1.time == t]; ev = x[(x.探索_avg > 0) & (x.驗證_avg > 0)]
    P(f"C 單特徵判斷器 {t}：{len(x)} 種，探索＋驗證都賺 {len(ev)}，其中測試也賺 {int((ev.測試_avg > 0).sum())}（{100*(ev.測試_avg > 0).mean():.0f}%）；全部變化測試賺的比例 {100*(x.測試_avg > 0).mean():.0f}%")
a2 = pd.read_csv(os.path.join(C_DIR, "out_a2_filters.csv"))
ev = a2[(a2.探索_dtot > 0) & (a2.驗證_dtot > 0)]
P(f"C 指標過濾 {len(a2)} 種：探索＋驗證都改善 {len(ev)}，測試也改善 {int((ev.測試_dtot > 0).sum())}（{100*(ev.測試_dtot > 0).mean():.0f}%）")
a3 = pd.read_csv(os.path.join(C_DIR, "out_a3_direct.csv"))
for t in [10, 1030]:
    x = a3[a3.time == t]; ev = x[(x.探索_avg > 0) & (x.驗證_avg > 0)]
    P(f"C 投票直接單 {t}：{len(x)} 種，探索＋驗證都賺 {len(ev)}，測試也賺 {int((ev.測試_avg > 0).sum())}；測試每筆中位 {ev.測試_avg.median():+.0f}")
# ---- (b) ----
D2 = pd.read_pickle(os.path.join(C_DIR, "days2.pkl"))
assert (D2.date.values == R.DATES).all()
CEN = {"clv": 0.5, "krsi": 50}
FE = ["ret", "effi", "vwap_d", "above_vw", "last_ret", "clv", "es_ret", "nq_es_rs", "krsi", "kmacdh", "kema_x", "kema_slope", "kdi", "kbb_pos"]
X = {f: D2[f"{f}_10"].values.astype(float) - CEN.get(f, 0) for f in FE}
okd = R.OKDAY & (R.PERIOD == "探索")
TH = {f: np.nanquantile(np.abs(X[f][okd]), 0.8) for f in FE}
UP = {f: np.nan_to_num(X[f]) > TH[f] for f in FE}; DN = {f: np.nan_to_num(X[f]) < -TH[f] for f in FE}
# 每天做多／做空的結果（10:00 進、停損 100、15:55 出）
pl = np.full(R.N, np.nan); ps = np.full(R.N, np.nan)
for i in np.where(R.OKDAY)[0]:
    a = R.sim(i, 960, 1, 100.0); b = R.sim(i, 960, -1, 100.0)
    if a: pl[i] = a["pnl"]
    if b: ps[i] = b["pnl"]
rows = []
for combo in itertools.combinations(FE, 5):
    up = sum(UP[f].astype(int) for f in combo); dn = sum(DN[f].astype(int) for f in combo)
    dv = np.where((up >= 3) & (dn == 0), 1, np.where((dn >= 3) & (up == 0), -1, 0)) * R.OKDAY
    pn = np.where(dv == 1, pl, np.where(dv == -1, ps, np.nan))
    r = dict(combo="+".join(combo))
    for p in R.PER:
        m = (R.PERIOD == p) & np.isfinite(pn) & (dv != 0)
        r[p + "_n"] = int(m.sum()); r[p + "_avg"] = pn[m].mean() if m.sum() else np.nan
    rows.append(r)
Z = pd.DataFrame(rows)
Z.to_csv(os.path.join(R.OUT, "rt_selC_combos.csv"), index=False)
chosen = "effi+above_vw+kmacdh+kema_slope+kdi"
cz = Z[Z.combo.apply(lambda s: set(s.split("+")) == set(chosen.split("+")))].iloc[0]
P(f"C 選的組合：探索 {cz.探索_avg:+.0f}（{cz.探索_n}）／驗證 {cz.驗證_avg:+.0f}（{cz.驗證_n}）／測試 {cz.測試_avg:+.0f}（{cz.測試_n}）")
ok = Z[(Z.探索_n >= 20) & (Z.驗證_n >= 8)]
ev = ok[(ok.探索_avg > 0) & (ok.驗證_avg > 0)]
P(f"2,002 組（筆數夠的 {len(ok)}）：探索＋驗證都賺 {len(ev)}（{100*len(ev)/len(ok):.0f}%），其中測試也賺 {int((ev.測試_avg > 0).sum())}（{100*(ev.測試_avg > 0).mean():.0f}%）；測試每筆中位 {ev.測試_avg.median():+.0f}")
P(f"  全部組合測試每筆中位 {ok.測試_avg.median():+.0f}；測試賺的比例 {100*(ok.測試_avg > 0).mean():.0f}%")
ev = ev.assign(ev_score=np.minimum(ev.探索_avg, ev.驗證_avg)).sort_values("ev_score", ascending=False)
rank_ev = list(ev.combo).index(cz.combo) + 1 if cz.combo in list(ev.combo) else None
P(f"  若只用探索＋驗證（兩段每筆取小者）排名，C 的組合排 {rank_ev}／{len(ev)}；前 10 名的測試每筆：{', '.join(f'{v:+.0f}' for v in ev.head(10).測試_avg)}；前 10 名測試平均 {ev.head(10).測試_avg.mean():+.0f}")
P(f"  C 的組合測試每筆在全部 {len(ok)} 組中的百分位：{100*(ok.測試_avg < cz.測試_avg).mean():.0f}%")
# 不含 C 看過測試欄位的「單特徵表」的選法：只用探索期最好的 5 個單特徵（依探索每筆）
single = []
for f in FE:
    dv = np.where(UP[f], 1, np.where(DN[f], -1, 0)) * R.OKDAY
    pn = np.where(dv == 1, pl, np.where(dv == -1, ps, np.nan))
    single.append(dict(f=f, **{p: np.nanmean(pn[(R.PERIOD == p) & (dv != 0)]) for p in R.PER}))
S = pd.DataFrame(single).sort_values("探索", ascending=False); P("單特徵 q80 順勢（10:00 進、停損 100）：\n" + S.round(0).to_string(index=False))
for crit in ["探索", "ev"]:
    s2 = S.assign(ev=np.minimum(S.探索, S.驗證)).sort_values(crit, ascending=False)
    top = "+".join(s2.f.head(5))
    zz = Z[Z.combo.apply(lambda s: set(s.split("+")) == set(top.split("+")))].iloc[0]
    P(f"  只用{'探索' if crit=='探索' else '探索＋驗證(取小)'}挑前 5 個單特徵＝{top}：探索 {zz.探索_avg:+.0f}／驗證 {zz.驗證_avg:+.0f}／測試 {zz.測試_avg:+.0f}（{zz.測試_n} 筆）／近期 {zz.近期_avg:+.0f}")
open(os.path.join(R.OUT, "rt_selC.txt"), "w").write("\n".join(out))
