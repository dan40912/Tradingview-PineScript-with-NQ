"""事先登記的規則家族（research/multi_entry/PREREG.md）：各時段「掃過價位後收回 → 反向」。
# 需要 claude/dazzling-euler-endxzx 分支的 research/lib；放到該分支 research/ 下執行（見 docs/21 第 10 節）。
+55／−145，每筆扣成本，k 根 5 分K 收回才進場。輸出每條規則每個切分的統計。研究用途，不是投資建議。"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from lib import data, features, sim
SYM = sys.argv[1] if len(sys.argv) > 1 else "NQ"
TP, SL = (55.0, 145.0) if SYM == "NQ" else (15.75, 41.5)   # ES：同比例（NQ/ES ≈ 3.5）
g = data.load_grid(SYM, include_holdout=True); lv = features.daily_levels(g)
S = lambda hhmm: sim.slot_of(hhmm)
SESS = {  # name: (window_start, first_entry, last_entry, exit_slot, levels, blocked)
    "ASIA": (S("18:00"), S("18:05"), S("02:00"), S("02:59"), ["PD", "VA"], ()),
    "LON":  (S("03:00"), S("03:00"), S("08:00"), S("08:29"), ["PD", "VA", "AS"], ()),
    "AM":   (S("09:30"), S("09:35"), S("11:30"), S("12:29"), ["PD", "ON"], (S("09:58"), S("10:02"))),
    "PM":   (S("12:30"), S("12:30"), S("14:30"), S("15:49"), ["PD", "VA", "IB"], ()),
}
def levels(r, i, name):
    H, L = g.H[i], g.L[i]
    if name == "PD": return r.pdh, r.pdl
    if name == "VA": return r.vah, r.val
    if name == "ON": return r.onh, r.onl
    if name == "AS":
        h, l = H[:S("03:00")], L[:S("03:00")]
        return (np.nanmax(h), np.nanmin(l)) if np.isfinite(h).sum() > 300 else (np.nan, np.nan)
    if name == "IB":
        h, l = H[S("09:30"):S("10:30")], L[S("09:30"):S("10:30")]
        return (np.nanmax(h), np.nanmin(l)) if np.isfinite(h).sum() > 50 else (np.nan, np.nan)
def signals():
    out = []
    for r in lv.itertuples():
        if not r.has_prev: continue
        i = r.i; H, L, C = g.H[i], g.L[i], g.C[i]
        for sess, (ws, fe, le, xs, lvls, blk) in SESS.items():
            for lname in lvls:
                hi, lo = levels(r, i, lname)
                if np.isnan(hi) or np.isnan(lo): continue
                for k in (1, 2):
                    st = {-1: [False, 0, False], 1: [False, 0, False]}   # swept, count, done
                    for a in range(ws - ws % 5, le + 1, 5):
                        blkc = C[a:a + 5]; ok = ~np.isnan(blkc)
                        if not ok.any(): continue
                        c = blkc[ok][-1]; e = a + 5
                        bh, bl = np.nanmax(H[a:a + 5]), np.nanmin(L[a:a + 5])
                        for d, lvl in ((-1, hi), (1, lo)):
                            s = st[d]
                            if s[2]: continue
                            beyond_wick = bh > lvl if d == -1 else bl < lvl
                            inside = c < lvl if d == -1 else c > lvl
                            if beyond_wick: s[0] = True
                            if not s[0]: continue
                            s[1] = s[1] + 1 if inside else 0
                            if s[1] >= k and fe <= e <= le and not (blk and blk[0] <= e <= blk[1]):
                                out.append(dict(i=i, entry_slot=e, dir=d, exit_slot=xs,
                                                rule=f"{sess}-{lname}-k{k}", side="空" if d == -1 else "多"))
                                s[2] = True
    return out
def split(d):
    return np.select([d <= "2025-06-30", d <= "2025-12-31", d <= "2026-10-19"], ["disc", "val", "test"], "fwd")
if __name__ == "__main__":
    T = sim.run_signals(g, signals(), tp_pts=TP, sl_pts=SL)
    T["sp"] = split(T.date); T["key"] = T.rule + "|" + T.side
    T.to_csv(f"family_trades_{SYM}.csv", index=False)
    rows = []
    for key, x in T.groupby("key"):
        row = {"key": key}
        for sp in ["disc", "val", "test", "fwd"]:
            y = x[x.sp == sp].pnl
            row[f"{sp}_n"] = len(y); row[f"{sp}_avg"] = y.mean() if len(y) else np.nan
            row[f"{sp}_win"] = (y > 0).mean() * 100 if len(y) else np.nan
            if sp == "disc": row["disc_t"] = y.mean() / (y.std(ddof=1) / np.sqrt(len(y))) if len(y) > 2 else np.nan
        rows.append(row)
    R = pd.DataFrame(rows).round(1); R.to_csv(f"family_summary_{SYM}.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 100)
    print(R[["key","disc_n","disc_avg","disc_t","disc_win","val_n","val_avg","test_n","test_avg","test_win","fwd_n","fwd_avg"]].to_string(index=False))
