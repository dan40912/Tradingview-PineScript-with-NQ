"""第二批（PREREG 第二批）：I2 第二筆、延長時段。研究用途，不是投資建議。"""
# 需要 claude/dazzling-euler-endxzx 分支的 research/lib；放到該分支 research/ 下執行（見 docs/21 第 7 節）。
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from lib import data, features, sim, strategies
from lib.data import RTH_OFF as R0
g = data.load_grid("NQ", include_holdout=True); lv = features.daily_levels(g)
base = sim.run_signals(g, strategies.i2_v1_signals(g, lv)).set_index("i")
def rescan(r, start_idx, last_entry, exit_idx, need_arm):
    C = g.C[r.i]; d = r.i2_dir; armed = not need_arm; cnt = 0
    first = start_idx + (4 - start_idx % 5) % 5          # first 5m close at/after start
    for idx in range(first, last_entry, 5):
        c = C[R0 + idx]
        if np.isnan(c): continue
        inside = r.val <= c <= r.vah
        out_orig = (c < r.val) if d == 1 else (c > r.vah)
        if out_orig: armed = True; cnt = 0; continue
        if not armed: continue
        cnt = cnt + 1 if inside else 0
        if cnt >= 2:
            ei = idx + 1
            if (28 <= ei <= 32) or ei > last_entry: return None
            return dict(i=r.i, entry_slot=R0 + ei, dir=int(d), exit_slot=R0 + exit_idx)
    return None
S = {"E1": [], "E2": [], "E3": [], "E4": []}
for r in lv.itertuples():
    if not r.has_prev or r.i2_dir == 0 or np.isnan(r.open_0930): continue
    if r.i in base.index:
        b = base.loc[r.i]; xi = int(b.exit_slot) - R0
        if xi < 120:
            s = rescan(r, xi + 1, 120, 179, True)
            if s and b.reason == "tp": S["E1"].append(s)
            if s: S["E2"].append(s)
        s = rescan(r, 180, 270, 379, True)
        if s: S["E4"].append(s)
    else:
        s = rescan(r, 120, 270, 379, False)
        if s: S["E3"].append(s)
sp = lambda d: np.select([d <= "2025-06-30", d <= "2025-12-31", d <= "2026-10-19"], ["disc", "val", "test"], "fwd")
rows = []; allT = []
for k, sigs in S.items():
    T = sim.run_signals(g, sigs, tp_pts=55.0, sl_pts=145.0); T["sp"] = sp(T.date); T["rule"] = k; allT.append(T)
    row = {"rule": k}
    for s in ["disc", "val", "test", "fwd"]:
        y = T[T.sp == s].pnl; row[f"{s}_n"] = len(y); row[f"{s}_avg"] = round(y.mean(), 1) if len(y) else np.nan
        row[f"{s}_win"] = round((y > 0).mean() * 100, 1) if len(y) else np.nan
        if s == "disc" and len(y) > 2: row["disc_t"] = round(y.mean() / (y.std(ddof=1) / np.sqrt(len(y))), 2)
    dv = T[T.sp.isin(["disc", "val"])].pnl; row["dv_n"] = len(dv); row["dv_avg"] = round(dv.mean(), 1)
    rows.append(row)
pd.concat(allT).to_csv("e_family_trades.csv", index=False)
pd.set_option("display.width", 250)
print(pd.DataFrame(rows).to_string(index=False))
b = base.reset_index(); b["sp"] = sp(b.date)
print("I2 v1 base:", b.groupby("sp").pnl.agg(["size", "mean"]).round(0).to_dict())
