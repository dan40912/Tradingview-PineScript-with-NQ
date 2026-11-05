"""02: Descriptive map (discovery only): forward RTH returns from decision time T to 15:59 (points),
by day type and price location vs prior value area. Sign convention: + = price up."""
import numpy as np, pandas as pd
import c2lib as L
g, lv, i2 = L.load()
i2d = dict(zip(i2.i, i2.entry_slot - L.RTH))
rows = []
for r in lv.itertuples():
    if not r.has_prev or r.split != "discovery" or not np.isfinite(r.open_0930):
        continue
    O, H, Lo, C = L.rth(g, r.i)
    last = np.where(np.isfinite(C))[0]
    if len(last) == 0 or last[-1] < 380:
        continue
    cl = C[last[-1]]
    for T in ["10:30", "11:30", "12:30", "13:30", "14:30"]:
        t = (int(T[:2]) * 60 + int(T[3:])) - 570 - 1   # close of minute before T
        c = C[t]
        if not np.isfinite(c):
            continue
        if r.i2_dir == 0:
            typ = "inside"
        else:
            typ = "out_I2" if (r.i in i2d and i2d[r.i] <= t) else "out_noI2"
        loc = "aboveVAH" if c > r.vah else ("belowVAL" if c < r.val else ("in_up" if c >= r.poc else "in_dn"))
        mv = c - r.open_0930
        rows.append({"T": T, "typ": typ, "loc": loc, "h": L.half(r.date), "fwd": cl - c,
                     "dirmove": np.sign(mv), "fwd_follow": np.sign(mv) * (cl - c) if mv != 0 else 0.0,
                     "away": (-r.i2_dir) * (cl - c) if r.i2_dir != 0 else np.nan})
D = pd.DataFrame(rows)
def tab(df, by, val):
    t = df.groupby(by)[val].agg(["count", "mean"]).round(1)
    t2 = df.groupby(by + ["h"])[val].mean().unstack().round(0)
    print(pd.concat([t, t2], axis=1).to_string())
print("=== fwd to close (+=up) by T, type, loc")
tab(D, ["T", "typ", "loc"], "fwd")
print("=== follow the day's move from open (pts), by T and type")
tab(D, ["T", "typ"], "fwd_follow")
print("=== outside days: forward move AWAY from value (+ = continues away), by T, type")
tab(D[D.typ != "inside"], ["T", "typ"], "away")
D.to_csv("out_02_describe.csv", index=False)
