"""檢查：我的 sim（市價 +55/−145）對基準 599 個候選點，和 B_exits 的 pnl1 逐筆比較。"""
import numpy as np, r1lib as R, run as RU
BR = RU.BR
bad = 0
for t in BR.itertuples():
    i = R.g.dindex[t.date]; e = int(t.entry_slot); d = int(t.dir)
    while np.isnan(R.O[i, e]): e += 1
    px = R.O[i, e]
    r = R.sim(i, d, e, px - 145 * d, px + 55 * d, int(t.xsched))
    if abs(r["pnl"] - t.pnl1) > 0.01:
        bad += 1; print(t.date, t.key, r["pnl"], t.pnl1)
print("候選點", len(BR), "不一致", bad)
