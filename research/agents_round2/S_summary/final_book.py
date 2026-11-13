# -*- coding: utf-8 -*-
"""S_summary：重算「最終組合」。研究用途，不是投資建議。

事先寫好的組合規則（跑之前就固定，跑完不改）：
  第一層（可以小量實盤）＝基準「順勢多點」六個點，+55／−145，每筆 1 口 NQ（＝10 MNQ），
      預定出場 美盤 12:29／亞洲 02:59；同一時間只持有一筆基準單（單一部位）。
  第二層（紙上記錄；這裡算「如果它畢業」的樣子）＝動能附加層 M：
      當天有 C1 訊號就做 C1（09:35）；當天沒有 C1 訊號才看 TD10（10:00）。每筆 0.5 口（5 MNQ）。
      M 和基準可以同時持倉（分開的策略），但共用每日風險檢查。
  優先順序：同一分鐘 基準 ＞ C1 ＞ TD10。依 (日期, 進場分鐘, 優先序) 一筆一筆檢查。
  每日風險檢查（平倉即記帳）：今天已實現（只算實際出場分鐘 < 本筆進場分鐘的單）
      − 其他持倉的最壞虧損 − 本筆最壞虧損 < −$3,000 → 不下。最壞虧損＝停損點數×20×口數＋$48×口數。
  壓力（不改變下哪些單，只改損益）：成本加倍（每口再扣 $48）；滑價（每筆進場差 2 點、非止盈出場再差 2 點）。
  逐筆來源：RT_redteam 的 rt_base_candidates.csv（599 個基準候選點，自己模擬出真實出場分鐘）、
      rt_trades_C1.csv、rt_trades_TD10.csv（RT 獨立重算，和 A、C 逐筆一致）。
  重疊分析另讀 R2 的 trades_Q13v2.csv、A 的 C6、R1 的 S02（只描述，不進組合）。
"""
import os, sys
import numpy as np, pandas as pd

AG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(AG, "S_summary")
RES = os.path.join(os.path.dirname(AG), "wt", "research")
sys.path.insert(0, RES)
from lib import data, features  # noqa: E402

PER = ["探索", "驗證", "測試", "近期"]
LIM = 3000.0
log = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s); log.append(s)

def hm(slot):
    slot = int(slot); m = (slot + 18 * 60) % 1440
    return f"{m // 60:02d}:{m % 60:02d}"

# ------------------------------------------------------------------ 資料
g = data.load_grid("NQ", include_holdout=True)
lv = features.daily_levels(g).set_index("i")
N = len(lv)
# ADR10（09:30–15:59 振幅，前 10 個有效日，不含當天）
rng = np.full(N, np.nan)
for i in range(N):
    h = g.H[i][930:1320].copy(); l = g.L[i][930:1320].copy()
    if str(lv.loc[i, "date"]) == "2026-10-30": h[1263 - 930] = np.nan; l[1263 - 930] = np.nan  # Yahoo 壞 K（同 A／RT）
    if np.isfinite(h).sum() > 300 and bool(lv.loc[i, "tradable"]):
        rng[i] = np.nanmax(h) - np.nanmin(l)
ADR10 = np.full(N, np.nan); hist = []
for i in range(N):
    if len(hist) >= 10: ADR10[i] = np.mean(hist[-10:])
    if np.isfinite(rng[i]): hist.append(rng[i])
TDF = pd.read_csv(os.path.join(AG, "RT_redteam", "rt_td_features.csv"))
TH = dict(effi=0.322, above_vw=0.800, slope=1.300, di=21.74, macdh=0.382)
NAMES = dict(effi="效率", above_vw="VWAP時間", slope="EMA斜率", di="DI差", macdh="MACD柱")

B = pd.read_csv(os.path.join(AG, "RT_redteam", "rt_base_candidates.csv"))
B["src"] = "基準"; B["size"] = 1.0; B["prio"] = 0
C1 = pd.read_csv(os.path.join(AG, "RT_redteam", "rt_trades_C1.csv")); C1["src"] = "C1"; C1["key"] = "C1"
TD = pd.read_csv(os.path.join(AG, "RT_redteam", "rt_trades_TD10.csv")); TD["src"] = "TD10"; TD["key"] = "TD10"
c1_days = set(C1.date)
M = pd.concat([C1, TD[~TD.date.isin(c1_days)]], ignore_index=True)
M["prio"] = M.src.map({"C1": 1, "TD10": 2})

KEYNAME = {"I2|空": "I2 空", "I2|多": "I2 多", "AM-PD-k1|空": "前高空·順", "AM-PD-k1|多": "前低多·順",
           "ASIA-PD-k1|多": "亞洲前低多 ES", "ASIA-VA-k1|多": "亞洲VAL多 ES", "AM-ON-k1|多": "早盤夜低多 ES"}

def reason(t):
    r = lv.loc[int(t.i)]
    if t.src == "基準":
        k = t.key
        if k.startswith("I2"):
            side = "上" if t.d == -1 else "下"; tgt = f"VAL {r.val:,.0f}" if t.d == -1 else f"VAH {r.vah:,.0f}"
            return f"I2：開在前日價值區{side}方（開 {r.open_0930:,.2f}），連 2 根 5 分K 收回價值區 → 往 {tgt} 方向"
        if k == "AM-PD-k1|空": return f"強壓力：前日高 {r.pdh:,.2f} 掃過後收回＋價格在 VWAP 下（順）"
        if k == "AM-PD-k1|多": return f"強支撐：前日低 {r.pdl:,.2f} 破底翻＋價格在 VWAP 上（順）"
        if k == "ASIA-PD-k1|多": return f"亞洲盤：前日低 {r.pdl:,.2f} 破底翻＋ES 同步"
        if k == "ASIA-VA-k1|多": return f"亞洲盤：前日 VAL {r.val:,.2f} 破底翻＋ES 同步"
        if k == "AM-ON-k1|多": return f"早盤：夜盤低 {r.onl:,.2f} 破底翻＋ES 同步＋波動≥1.2倍"
        return k
    if t.src == "C1":
        m = g.C[int(t.i)][934] - g.O[int(t.i)][930]; a = ADR10[int(t.i)]
        w = "強勢上攻" if t.d == 1 else "強勢下殺"
        return f"開盤 5 分鐘{w} {m:+.0f} 點（≥0.15×ADR10＝{0.15*a:.0f}）"
    f = TDF.loc[int(t.i)]
    if t.d == 1: v = [NAMES[c] for c in TH if f[c] >= TH[c]]
    else: v = [NAMES[c] for c in TH if f[c] <= -TH[c]]
    return f"10:00 趨勢日（{'多' if t.d == 1 else '空'}方強票 {len(v)}/5：{'、'.join(v)}；反向 0 票）"

# ------------------------------------------------------------------ 記帳
def book(T, base_single=True):
    T = T.sort_values(["date", "e", "prio"], kind="stable").reset_index(drop=True)
    keep = np.zeros(len(T), bool)
    for d, idx in T.groupby("date", sort=False).groups.items():
        h = []  # (x, pnl, worst, is_base)
        for k in idx:
            t = T.loc[k]
            if base_single and t.src == "基準" and any(z[3] and z[0] >= t.e for z in h):
                continue
            if t.src != "基準" and any((not z[3]) and z[0] >= t.e for z in h):
                continue
            real = sum(z[1] for z in h if z[0] < t.e)
            openw = sum(z[2] for z in h if z[0] >= t.e)
            if real - openw - t.worst < -LIM:
                continue
            keep[k] = True; h.append((t.x, t.pnl, t.worst, t.src == "基準"))
    return T[keep].copy()

def scale(X, size):
    X = X.copy(); X["size"] = size; X["pnl"] = X.pnl * size; X["worst"] = X.worst * size; return X

def mdd(v):
    c = np.cumsum(v); pk = np.maximum.accumulate(np.concatenate([[0.0], c]))[1:]
    return float((c - pk).min()) if len(c) else 0.0

def stats(T, col="pnl"):
    if len(T) == 0: return dict(n=0, win=np.nan, avg=np.nan, tot=0, mdd=0, mdd_trade=0, worst_day=0, ratio=np.nan)
    T = T.sort_values(["date", "e"]); p = T[col].values; dy = T.groupby("date")[col].sum()
    dd = mdd(dy.values)
    return dict(n=len(p), win=round(100 * (p > 0).mean(), 1), avg=round(p.mean()), tot=round(p.sum()),
                mdd=round(dd), mdd_trade=round(mdd(p)), worst_day=round(dy.min()),
                ratio=round(p.sum() / -dd, 2) if dd < 0 else np.nan)

rows = []
def report(name, T, col="pnl"):
    P(f"\n【{name}】")
    P(f"{'期間':<4} {'筆數':>5} {'勝率':>6} {'每筆':>7} {'合計':>9} {'最大回撤':>9} {'最差一天':>8} {'報酬/回撤':>8}")
    for per in PER + ["全期"]:
        x = T if per == "全期" else T[T.P == per]
        s = stats(x, col); rows.append(dict(組合=name, 期間=per, **s))
        P(f"{per:<4} {s['n']:>5} {s['win']:>5}% {s['avg']:>+7,.0f} {s['tot']:>+9,} {s['mdd']:>+9,} {s['worst_day']:>+8,} {s['ratio']:>8}")

def stress(T):
    T = T.copy()
    T["pnl_cost2"] = T.pnl - 48.0 * T["size"]
    slip = 2.0 + 2.0 * (T.why != "tp")
    T["pnl_slip2"] = T.pnl - slip * 20.0 * T["size"]
    return T

# ------------------------------------------------------------------ 組合
V0 = book(B.copy())
V1 = book(pd.concat([B, scale(M, 0.5)], ignore_index=True))
P("驗證：基準單獨（平倉即記帳、單一部位）應為 520 筆、+175,110、回撤 −11,851")
report("V0 基準（第一層，10 MNQ）", stress(V0))
report("V1 最終組合＝基準 10 MNQ＋M（C1，否則 TD10）5 MNQ", stress(V1))
S0, S1 = stress(V0), stress(V1)
report("V0 壓力：成本加倍", S0, "pnl_cost2"); report("V1 壓力：成本加倍", S1, "pnl_cost2")
report("V0 壓力：滑價 2 點", S0, "pnl_slip2"); report("V1 壓力：滑價 2 點", S1, "pnl_slip2")
Mone = book(scale(M, 0.5)); report("M 單獨（5 MNQ）", Mone)
Mfull = book(M.copy()); report("M 單獨（10 MNQ，對照）", Mfull)

# 排擠
kb0 = set(zip(V0.date, V0.e, V0.key)); kb1 = set(zip(V1[V1.src == "基準"].date, V1[V1.src == "基準"].e, V1[V1.src == "基準"].key))
lost = V0[np.array([k not in kb1 for k in zip(V0.date, V0.e, V0.key)])]
gain = V1[(V1.src == "基準") & np.array([k not in kb0 for k in zip(V1.date, V1.e, V1.key)])]
mk = V1[V1.src != "基準"]
P(f"\n排擠：M 候選 {len(M)} 筆，進到組合 {len(mk)} 筆（C1 {int((mk.src=='C1').sum())}、TD10 {int((mk.src=='TD10').sum())}）")
P(f"  被 M 擋掉的基準單 {len(lost)} 筆，合計 {lost.pnl.sum():+,.0f}；因 M 先賺而多出的基準單 {len(gain)} 筆 {gain.pnl.sum():+,.0f}")
for per in PER:
    P(f"  {per}：擋掉基準 {int((lost.P==per).sum())} 筆 {lost[lost.P==per].pnl.sum():+,.0f}；M 進場 {int((mk.P==per).sum())} 筆 {mk[mk.P==per].pnl.sum():+,.0f}")
same = mk.merge(V1[V1.src == "基準"][["date", "d"]], on="date", suffixes=("", "_b"))
opp = same[same.d != same.d_b].date.nunique()
P(f"  M 與基準同一天 {same.date.nunique()} 天，其中當天有反向基準單 {opp} 天（不一定同時持倉；同時持倉時實盤單一帳戶會抵銷）")

# ------------------------------------------------------------------ 重疊分析（描述）
P("\n重疊分析（開盤動能是不是同一個效應）")
Q13 = pd.read_csv(os.path.join(AG, "R2_backtest", "trades_Q13v2.csv")).rename(columns={"seg": "P"})
C6 = pd.read_csv(os.path.join(AG, "A_trend", "trades_C6_跳空延續5分.csv"))
S02 = pd.read_csv(os.path.join(AG, "R1_backtest", "trades_S02.csv"))
S02t = S02[S02.combo == "S02_top_off_B55"].rename(columns={"seg": "P"}); S02t["d"] = S02t.dir.map({"多": 1, "空": -1})
sets = {"C1": C1, "TD10": TD, "Q13v2": Q13, "C6": C6, "S02top": S02t}
dirs = {k: dict(zip(v.date, v.d)) for k, v in sets.items()}
names = list(sets)
P("同一天（同方向）天數矩陣：")
P("        " + "".join(f"{n:>14}" for n in names))
for a in names:
    line = f"{a:<8}"
    for b in names:
        both = set(dirs[a]) & set(dirs[b]); sd = sum(dirs[a][x] == dirs[b][x] for x in both)
        line += f"{f'{len(both)}({sd})':>14}"
    P(line + f"   共 {len(dirs[a])} 天")
mom = set(C1.date) | set(TD.date)
for nm, X in [("Q13v2", Q13), ("C6", C6), ("S02top", S02t)]:
    X = X.copy(); X["ov"] = X.date.isin(mom)
    msg = []
    for per in PER + ["全期"]:
        y = X if per == "全期" else X[X.P == per]
        a, b = y[y.ov].pnl, y[~y.ov].pnl
        msg.append(f"{per} 同日 {len(a)} 筆 {a.mean() if len(a) else 0:+.0f}／不同日 {len(b)} 筆 {b.mean() if len(b) else 0:+.0f}")
    P(f"{nm} 和 C1∪TD10 同日 vs 不同日（每筆）：" + "；".join(msg))
mom_basedays = set(V0.date)
for nm, X in [("C1", C1), ("TD10", TD)]:
    o = X.date.isin(mom_basedays)
    P(f"{nm} 在基準有單的日子 {int(o.sum())} 筆每筆 {X[o].pnl.mean():+.0f}；沒有基準單的日子 {int((~o).sum())} 筆每筆 {X[~o].pnl.mean():+.0f}")

# 基準各點（候選層級，+55／−145，每筆 1 口，未經每日檢查）
P("\n基準六點各自（候選層級，每筆；括號筆數）")
for k, x in B.groupby("key"):
    P(f"  {KEYNAME[k]:<14}" + "  ".join(f"{per} {x[x.P==per].pnl.mean():+5.0f}({(x.P==per).sum()})" for per in PER)
      + f"  勝率 {100*(x.pnl>0).mean():.0f}%")

# 年度與月度（V1 vs V0）
for nm, T in [("V0", V0), ("V1", V1)]:
    y = T.groupby(T.date.str[:4]).pnl.sum().round(0)
    P(f"{nm} 年度：" + "、".join(f"{a} {b:+,.0f}" for a, b in y.items()))
m26 = pd.DataFrame({"V0": V0[V0.date >= "2026"].groupby(V0.date.str[:7]).pnl.sum(),
                    "V1": V1[V1.date >= "2026"].groupby(V1.date.str[:7]).pnl.sum()}).fillna(0).round(0)
P("2026 月度：" + "；".join(f"{i[5:]}月 {r.V0:+,.0f}/{r.V1:+,.0f}" for i, r in m26.iterrows()))

# 11/11
P("\n2026-11-11：V1 交易 " + str(int((V1.date == "2026-11-11").sum())) + " 筆；ADR10＝"
  + f"{ADR10[740]:.1f}、門檻 {0.15*ADR10[740]:.1f}、09:30→09:34 動 {g.C[740][934]-g.O[740][930]:+.2f} 點；TD10 票："
  + str({NAMES[c]: round(float(TDF.loc[740, c]), 2) for c in TH}))
r = lv.loc[740]
P(f"  11/11 價位：前日高 {r.pdh}、前日低 {r.pdl}、VAH {r.vah}、VAL {r.val}、夜盤高 {r.onh}、夜盤低 {r.onl}、開盤 {r.open_0930}")
lo = np.nanmin(g.L[740][930:1050]); lo_t = hm(930 + int(np.nanargmin(g.L[740][930:1050])))
hi = np.nanmax(g.H[740][930:1320]); hi_t = hm(930 + int(np.nanargmax(g.H[740][930:1320])))
P(f"  09:30–11:29 最低 {lo}（{lo_t}），全天最高 {hi}（{hi_t}），15:55 收 {g.C[740][1315]}")
# 對照：09:35 直接做多（止損 145，15:55 出）——只描述 11/11，不是新規則
e = g.O[740][935]; sl = e - 145; hit = np.where(g.L[740][935:1316] <= sl)[0]
P(f"  對照（描述用）：09:35 做多 {e}，止損 145 → " + (f"{hm(935+hit[0])} 停損 −$2,948" if len(hit) else f"15:55 出 {(g.C[740][1315]-e)*20-48:+,.0f}"))
e = g.O[740][960]; sl = e - 100; hit = np.where(g.L[740][960:1316] <= sl)[0]
P(f"  對照（描述用）：10:00 做多 {e}，止損 100 → " + (f"{hm(960+hit[0])} 停損 −$2,048" if len(hit) else f"15:55 出 {(g.C[740][1315]-e)*20-48:+,.0f}"))

# ------------------------------------------------------------------ 輸出
F = V1.copy()
F["層"] = np.where(F.src == "基準", "第一層（實盤小量）", "第二層（紙上：M 動能附加）")
F["策略"] = [KEYNAME.get(k, k) for k in F.key]
F["方向"] = F.d.map({1: "多", -1: "空"}); F["MNQ口數"] = (F["size"] * 10).astype(int)
F["進場時間"] = F.e.map(hm); F["出場時間"] = F.x.map(hm)
F["原因"] = [reason(t) for t in F.itertuples()]
F["出場原因"] = F.why.map({"tp": "止盈 +55", "sl": "停損", "time": "時間出場", "trail": "移動停損"}).fillna(F.why)
F = F.rename(columns={"date": "日期", "P": "期間", "entry": "進場價", "exit": "出場價", "pnl": "損益", "worst": "最壞虧損"})
F = F[["日期", "期間", "層", "策略", "方向", "MNQ口數", "進場時間", "進場價", "出場時間", "出場價", "出場原因", "損益", "最壞虧損", "原因"]]
F["損益"] = F["損益"].round(2); F["最壞虧損"] = F["最壞虧損"].round(2)
F.sort_values(["日期", "進場時間"]).to_csv(os.path.join(OUT, "final_trades.csv"), index=False, encoding="utf-8-sig")
pd.DataFrame(rows).to_csv(os.path.join(OUT, "final_summary.csv"), index=False, encoding="utf-8-sig")
open(os.path.join(OUT, "out_final_book.txt"), "w").write("\n".join(log) + "\n")
