import numpy as np, pandas as pd
# 需要 claude/dazzling-euler-endxzx 分支的 research/lib，以及 research/winrate_recent/wr_table.csv.gz；放到該分支 research/ 下執行（見 docs/23）。
F = pd.read_csv("wr_table.csv.gz"); F["key"] = F.rule + "|" + F.side
S = pd.read_csv("swan_shock.csv")[["date", "rule", "side", "entry_slot", "shock"]]
F = F.merge(S, on=["date", "rule", "side", "entry_slot"], how="left")
F["Y"] = np.select([F.date >= "2026-10-01", F.date >= "2026-01-01", F.date >= "2025-01-01"], ["近期", "2026-1~9", "2025"], "2024")
withI = F.to_vwap == False   # 美盤：價格在 VWAP 同一邊（順盤中）
P = {"I2": F.key.str.startswith("I2"), "I2多": F.key == "I2|多",
     "前高空": F.key == "AM-PD-k1|空", "前高空順": (F.key == "AM-PD-k1|空") & withI,
     "前低多順I2": (F.key == "AM-PD-k1|多") & (F.i2 != "逆"), "前低多順": (F.key == "AM-PD-k1|多") & withI,
     "亞洲多ES": (F.key == "ASIA-PD-k1|多") & (F.es_ok == True), "亞洲VAL多ES": (F.key == "ASIA-VA-k1|多") & (F.es_ok == True),
     "美盤早夜盤低多ES": (F.key == "AM-ON-k1|多") & (F.es_ok == True) & (F.vol >= 1.2)}
def book(names, guard=True):
    m = np.zeros(len(F), bool)
    for n in names: m |= P[n].fillna(False).values
    T = F[m].copy()
    if guard: T = T[~(T.shock >= 250)]
    T = T.sort_values(["date", "entry_slot"])
    a = T.key.str.startswith("ASIA") & (T.dir == 1); T = pd.concat([T[~a], T[a].drop_duplicates("date")]).sort_values(["date", "entry_slot"])
    keep = []
    for d, x in T.groupby("date", sort=False):
        closed = []; opened = []
        for t in x.itertuples():
            real = sum(p for xs, p in closed if xs <= t.entry_slot); n_open = sum(1 for xs in opened if xs > t.entry_slot)
            if real - 2948 * (n_open + 1) < -3000: keep.append(False); continue
            keep.append(True); closed.append((t.exit_slot, t.pnl55)); opened.append(t.exit_slot)
    return T[keep]
def dd(p): c = p.cumsum(); return (c - c.cummax()).min()
nd = F.groupby("Y").date.nunique()
BK = {"推薦（v16）": ["I2", "前高空", "前低多順I2", "亞洲多ES"],
      "推薦＋盤中順勢": ["I2", "前高空順", "前低多順", "亞洲多ES"],
      "高勝率（只做多＋ES）": ["I2多", "亞洲多ES", "亞洲VAL多ES", "美盤早夜盤低多ES"],
      "順勢多點（I2＋前高空順＋前低多順＋三個 ES 做多）": ["I2", "前高空順", "前低多順", "亞洲多ES", "亞洲VAL多ES", "美盤早夜盤低多ES"]}
for k, names in BK.items():
    T = book(names); print("==", k)
    for y in ["2024", "2025", "2026-1~9", "近期"]:
        x = T[T.Y == y].pnl55
        print(f"   {y:9s} {len(x):4d} 筆｜每天 {len(x)/nd[y]:.2f}｜勝率 {(x>0).mean()*100:5.1f}%｜每筆 {x.mean():+5.0f}｜合計 {x.sum():+8,.0f}")
    x = T.pnl55; dy = T.groupby("date").pnl55.sum()
    print(f"   全期 {len(x)} 筆｜勝率 {(x>0).mean()*100:.1f}%｜合計 {x.sum():+,.0f}｜最大回撤 {dd(x):+,.0f}｜最差一天 {dy.min():+,.0f}｜做空占 {(T.dir==-1).mean()*100:.0f}%")
