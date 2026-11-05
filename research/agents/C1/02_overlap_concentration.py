"""C1 step 2 (DISCOVERY only): does an ORB-toward-value entry capture the I2 move earlier?
Overlap with I2 v1 days, entry-time difference, P&L concentration (drop best k trades), hybrid 'first of'."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import c1lib as c
g, lv = c.load()
D = lambda e: {i: s for i, s in e.items() if g.split[i] == "discovery"}
i2 = D(c.ent_i2(g, lv)); o15 = D(c.ent_orb(g, lv, 15, False)); o15f = D(c.ent_orb(g, lv, 15, True))
ud = set(c.i2_days(lv).query("split=='discovery'").i)
print("I2-universe discovery days:", len(ud), " I2v1 entries:", len(i2), " ORB15any:", len(o15), " ORB15first:", len(o15f))
both = set(i2) & set(o15)
dt = np.array([i2[i] - o15[i] for i in both])
print(f"days with both: {len(both)}; ORB earlier on {np.mean(dt>0):.0%}, median lead {np.median(dt):.0f} min")
print("ORB15any days without I2 v1 entry:", len(set(o15) - set(i2)), "; I2 v1 days without ORB15any:", len(set(i2) - set(o15)))
X = {"A3": dict(tp="poc", poc_floor=20, sl=60, np=(30, 20)), "POC_60": dict(tp="poc", poc_floor=20, sl=60),
     "EOD_110_145": dict(tp=110, sl=145, exit="1559"), "FIX55_145": dict(tp=55, sl=145)}
first = {i: min(i2.get(i, 10**9), o15.get(i, 10**9)) for i in set(i2) | set(o15)}
rows = []
for xn, xs in X.items():
    for en, ents in [("I2", i2), ("ORB15any", o15), ("ORB15f", o15f), ("FIRST(I2,ORB15any)", first)]:
        tr = c.run(g, c.make_signals(g, lv, ents, xs))
        on = tr[tr.i.isin(i2.keys())]; off = tr[~tr.i.isin(i2.keys())]
        p = np.sort(tr.pnl.values)[::-1]
        rows.append(dict(exit=xn, entry=en, n=len(tr), avg=round(tr.pnl.mean()), on_I2_n=len(on), on_I2_avg=round(on.pnl.mean()) if len(on) else None,
                         off_I2_n=len(off), off_I2_avg=round(off.pnl.mean()) if len(off) else None,
                         drop1=round(p[1:].mean()), drop5=round(p[5:].mean()), drop10=round(p[10:].mean()),
                         med=round(np.median(p)), tgt_lt25=(np.mean([s["tp_pts"] < 25 for s in c.make_signals(g, lv, ents, xs) if s["tp_pts"]]) if xs.get("tp") == "poc" else None)))
pd.set_option("display.width", 250)
print(pd.DataFrame(rows).to_string(index=False))
# median POC target distance per entry
for en, ents in [("I2", i2), ("ORB15any", o15)]:
    s = c.make_signals(g, lv, ents, X["POC_60"])
    tp = np.array([x["tp_pts"] for x in s]); print(en, "POC target pts: median", np.median(tp), " share <25:", np.mean(tp < 25).round(2), " share==floor:", np.mean(tp == 20).round(2))
