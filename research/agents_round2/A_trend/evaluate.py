"""第三步：凍結候選的四段績效、與基準「順勢多點」的日相關、合併後的組合、2026-11-11 的動作。
執行：python evaluate.py > out_evaluate.txt"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tlib
from tlib import fmt_table, by_period, daily_corr, combine, baseline_book, dd
from candidates import CANDIDATES, cfg
from families import ctrl_long, STOPS, EXITS
from lib.sim import hhmm_of

D = tlib.Data()
base = baseline_book()
print("壞價修正的 K 棒數：", D.n_bad_bars)
print("基準「順勢多點」重建：", len(base), "筆，合計", round(base.pnl.sum()), "，最大回撤", round(dd(base.pnl.values)))
print(fmt_table(base))
iday = D.dindex = {d: i for i, d in enumerate(D.dates)}
I11 = iday["2026-11-11"]

res = {}
for name in CANDIDATES:
    fn, p, c = cfg(name)
    T = tlib.run(D, fn(D, **p), c)
    T["cand"] = name
    T.to_csv(os.path.join(tlib.OUT, f"trades_{name}.csv"), index=False)
    res[name] = T
    print(f"\n## {name}  參數 {p}  {CANDIDATES[name][2]} 止損／{CANDIDATES[name][3]} 出場")
    print(fmt_table(T))
    bp = by_period(T)
    allp = bp["全期"]
    print(f"全期 報酬／回撤 = {allp['tot'] / -allp['mdd']:.2f}" if allp["mdd"] < 0 else "")
    long_ = T[T.d == 1]; short = T[T.d == -1]
    print("做多／做空 每筆（各段）：", {P: (round(long_[long_.P == P].pnl.mean()), round(short[short.P == P].pnl.mean()) if (short.P == P).any() else None) for P in tlib.PERIODS})
    why = T.groupby("why").pnl.agg(["size", "mean"]).round(0)
    why.index = why.index.map({0: "時間", 1: "止損", 2: "止盈", 3: "VWAP", 4: "移動停損"})
    print("出場原因：", why.to_dict("index"))
    c1, c2, nb = daily_corr(D, base, T)
    print(f"與基準的日損益相關：全部交易日 {c1:+.2f}；兩者同一天都有交易的 {nb} 天 {c2:+.2f}")
    # 合併
    A = combine(base, T)
    ab = A[A.src == "base"]; am = A[A.src == "mine"]
    print(f"合併後（每日 $3,000 檢查，時間先到先做）：總 {len(A)} 筆，其中基準 {len(ab)}（原 {len(base)}）／本規則 {len(am)}（原 {len(T)}）")
    print(fmt_table(A))
    x = T[T.i == I11]
    if len(x):
        for r in x.itertuples():
            print(f"2026-11-11：{'做多' if r.d == 1 else '做空'} {hhmm_of(r.e + 930)} 進場 {r.entry:.2f}，初始止損 {r.stop0:.2f}（{r.dist:.0f} 點），"
                  f"{hhmm_of(r.x + 930)} 出場 {r.exit:.2f}（原因 {r.why}），損益 {r.pnl:+,.0f}")
    else:
        print("2026-11-11：沒有訊號")

# 對照：每天 09:35 無條件做多（同樣止損、移動停損）——看順勢規則是不是只吃到大盤上漲
for N in (5, 15):
    T = tlib.run(D, ctrl_long(D, N=N), {**STOPS["0.25ADR"], **EXITS["移100"]})
    print(f"\n## 對照：每天 09:30+{N} 分無條件做多（0.25ADR 止損、移100）"); print(fmt_table(T))
    # 隨機方向（固定種子）
rng = np.random.default_rng(0)
sig = ctrl_long(D, N=5)
outs = []
for k in range(20):
    s2 = [dict(s, d=int(rng.choice([1, -1]))) for s in sig]
    T = tlib.run(D, s2, {**STOPS["0.25ADR"], **EXITS["移100"]})
    outs.append({P: T[T.P == P].pnl.mean() for P in tlib.PERIODS})
print("\n## 對照：每天 09:35 隨機方向（20 次平均每筆）", pd.DataFrame(outs).mean().round(0).to_dict())

# 合併所有候選的日損益相關矩陣
days = [D.dates[i] for i in range(D.n) if D.ok[i]]
M = pd.DataFrame({k: v.groupby("date").pnl.sum().reindex(days).fillna(0) for k, v in res.items()})
M["基準"] = base.groupby("date").pnl.sum().reindex(days).fillna(0)
print("\n## 日損益相關矩陣（全部交易日）"); print(M.corr().round(2).to_string())


# ---------- 補充 1：C1 開盤驅動的高原在測試／近期（只報告，不用來挑） ----------
from families import f1_od
print("\n## 補充：開盤驅動（0.25ADR 止損、移100）各 N×k 的每筆（探索/驗證/測試/近期）")
for N in (5, 10, 15):
    line = []
    for k in (0.1, 0.15, 0.2):
        T = tlib.run(D, f1_od(D, N=N, k=k), {**STOPS["0.25ADR"], **EXITS["移100"]})
        line.append(f"k={k}: " + "/".join(f"{T[T.P == P].pnl.mean():+.0f}" for P in tlib.PERIODS) + f"（測試 {int((T.P == '測試').sum())} 筆）")
    print(f"N={N}  " + "｜".join(line))
print("\n## 補充：C1 用其他出場（測試期每筆／合計）")
for xn in ["收盤", "移70", "移100", "移150", "3R", "保本1R", "中午", "VWAP"]:
    T = tlib.run(D, f1_od(D, N=5, k=0.15), {**STOPS["0.25ADR"], **EXITS[xn]})
    print(f"  {xn:6s}", "｜".join(f"{P} {T[T.P == P].pnl.mean():+.0f}/{T[T.P == P].pnl.sum():+,.0f}" for P in tlib.PERIODS))

# ---------- 補充 2：C1 測試期逐月 ----------
T1 = res["C1_開盤驅動5分"]
x = T1[T1.P.isin(["測試", "近期"])].copy(); x["m"] = x.date.str[:7]
print("\n## C1 2026 逐月"); print(x.groupby("m").pnl.agg(["size", "sum", "mean"]).round(0).to_string())
print("C1 逐年：", T1.assign(y=T1.date.str[:4]).groupby("y").pnl.agg(["size", "sum", "mean"]).round(0).to_dict("index"))


# ---------- 補充 3：合併時讓基準優先（基準訊號出現時，若本規則持倉中，先在上一根收盤平掉本規則） ----------
def combine_yield(base, mine, fn, p, c):
    from tlib import sim1
    rows = []
    bdays = dict(tuple(base.groupby("date")))
    for r in mine.itertuples():
        b = bdays.get(r.date)
        cut = None
        if b is not None:
            # 本規則持倉期間第一個基準 RTH 進場
            bb = b[(b.entry_slot > r.e + 930) & (b.entry_slot <= r.x + 930)]
            if len(bb):
                cut = int(bb.entry_slot.min()) - 930 - 1
        if cut is None:
            rows.append(dict(date=r.date, P=r.P, dir=r.d, entry_slot=r.e + 930, exit_slot=r.x + 930, pnl=r.pnl, worst=r.worst, src="mine", key="mine"))
        else:
            i = r.i
            xs, xp, why = sim1(D.O[i], D.H[i], D.L[i], D.C[i], D.vwap[i], r.e, r.d, r.stop0, np.nan if c.get("tp_R") is None else r.entry + r.d * c["tp_R"] * r.dist, cut,
                               (np.nan if c.get("trail") is None else float(c["trail"])), (float(c["trail"]) if c.get("ttrig") == "same" else 0.0), np.nan, np.nan, 0, 0.0)
            pnl = r.d * (xp - r.entry) * 20 - 48
            rows.append(dict(date=r.date, P=r.P, dir=r.d, entry_slot=r.e + 930, exit_slot=xs + 930, pnl=pnl, worst=r.worst, src="mine", key="mine"))
    M = pd.DataFrame(rows)
    A = pd.concat([base, M]).sort_values(["date", "entry_slot", "src"]).reset_index(drop=True)
    keep = []
    for d, xx in A.groupby("date", sort=False):
        closed = []; opened = []
        for t in xx.itertuples():
            real = sum(pp for xs, pp in closed if xs <= t.entry_slot)
            ow = sum(w for xs, w in opened if xs >= t.entry_slot)
            if real - ow - t.worst < -3000:
                keep.append(False); continue
            keep.append(True); closed.append((t.exit_slot, t.pnl)); opened.append((t.exit_slot, t.worst))
    return A[keep].copy()


for nm in ["C1_開盤驅動5分", "C6_跳空延續5分"]:
    fn, p, c = cfg(nm)
    A = combine_yield(base, res[nm], fn, p, c)
    print(f"\n## 合併（基準優先：基準訊號出現時先平掉 {nm}）：基準 {int((A.src == 'base').sum())} 筆／本規則 {int((A.src == 'mine').sum())} 筆")
    print(fmt_table(A))
