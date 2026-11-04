"""「衝太遠之後反轉」：盤中高點比 VAH 高出 ≥ X 倍區寬（而且在前日高之上），之後第一次 5 分K 收在 VWAP 之下 → 做空。
底部反過來做多。SL＝最近 30 分鐘的極值外 5 點（最多 200 點）；TP＝1R／2R／回到 VAH（VAL）；10:00 後、14:00 前進場；15:49 平倉。"""
import sys, pickle, itertools, numpy as np, pandas as pd
from eng import sim
nq = pickle.load(open("NQ_days2.pkl", "rb"))
def ex(d, x=2.0, tgt="1R", side=-1, maxrisk=200):
    O, H, L, C, vw = d["O"], d["H"], d["L"], d["C"], d["vwap"]; w = d["vah"] - d["val"]
    if np.isnan(O[0]) or w <= 0: return None
    for i in range(34, 270, 5):
        if np.isnan(C[i]): continue
        if side == -1:
            hi = np.nanmax(H[:i + 1]); ok = hi >= d["vah"] + x * w and hi >= d["pdh"] and C[i] < vw[i] and np.nanargmax(H[:i + 1]) < i - 4
        else:
            lo = np.nanmin(L[:i + 1]); ok = lo <= d["val"] - x * w and lo <= d["pdl"] and C[i] > vw[i] and np.nanargmin(L[:i + 1]) < i - 4
        if not ok: continue
        ei = i + 1
        if np.isnan(O[ei]): return None
        e = O[ei]; stop = (np.nanmax(H[i - 29:i + 1]) + 5) if side == -1 else (np.nanmin(L[i - 29:i + 1]) - 5); risk = abs(stop - e)
        if risk > maxrisk or risk < 10: return None
        tp = {"1R": risk, "2R": 2 * risk, "VA": abs(e - (d["vah"] if side == -1 else d["val"]))}[tgt]
        if tp < 10: return None
        return sim(d, ei, side, tp, risk, 379, None, None, "延伸反轉")
    return None
if __name__ == "__main__":
    rows = []
    for x, tgt, side in itertools.product((1.0, 2.0, 3.0), ("1R", "2R", "VA"), (-1, 1)):
        df = pd.DataFrame([t | {"date": d["date"]} for d in nq for t in [ex(d, x, tgt, side)] if t])
        r = dict(方向="頂部做空" if side == -1 else "底部做多", 延伸=f"{x:.0f} 倍區寬", 目標=tgt)
        for lab, (a, b) in {"2024": ("2024", "2024-12-31"), "2025": ("2025", "2025-12-31"), "2026驗證": ("2026", "2026-09-30"), "10月起": ("2026-10", "2026-12-31")}.items():
            s = df[(df.date >= a) & (df.date <= b)] if len(df) else df
            r[lab] = f"{len(s)}筆 {s.pnl.mean():+.0f}" if len(s) else "—"
        s = df[df.date <= "2026-09-30"] if len(df) else df
        r["全期每筆"] = round(s.pnl.mean()) if len(s) else np.nan; r["勝率"] = f"{(s.pnl > 0).mean():.0%}" if len(s) else "—"
        t4 = df[df.date == "2026-11-04"] if len(df) else df
        r["11/4"] = f"{t4.res.iloc[0]} {t4.pnl.iloc[0]:+.0f}" if len(t4) else "—"
        rows.append(r)
    pd.set_option("display.width", 220); print(pd.DataFrame(rows).to_string(index=False))
