"""Screen: average point change over fixed time windows (discovery only), by window,
day-of-week, turn-of-month, OPEX week. Pure descriptive screen (no costs) to decide what to simulate."""
import numpy as np, pandas as pd
from a2lib import load, sim
g, lv = load("NQ")
D = (g.split == "discovery") & g.tradable
def px(slot, kind="O"):
    a = getattr(g, kind)[:, slot]
    return a
def ret(s0, s1):
    # open of s0 -> close of s1-1 approximated by open s0 -> open s1 (fill forward small gaps)
    O = pd.DataFrame(g.O).ffill(axis=1).values
    return O[:, s1] - O[:, s0]
W = [("18:00","02:00"),("02:00","08:30"),("08:30","09:30"),("18:00","09:30"),("09:30","10:00"),("10:00","11:30"),
     ("11:30","14:00"),("14:00","15:00"),("15:00","15:30"),("15:30","15:59"),("15:00","15:59"),("09:30","15:59"),
     ("03:00","09:30"),("18:00","03:00")]
rows=[]
for a,b in W:
    r = ret(sim.slot_of(a), sim.slot_of(b))[D]
    r = r[np.isfinite(r)]
    rows.append((f"{a}-{b}", len(r), r.mean().round(1), np.median(r).round(1), (r.mean()/r.std()*np.sqrt(len(r))).round(2), (r>0).mean().round(2)))
print(pd.DataFrame(rows, columns=["win","n","mean_pts","med","t","up%"]).to_string())
# day of week / TOM for RTH 09:30-15:59 and overnight 18:00-09:30
dts = pd.to_datetime(g.dates)
dow = dts.dayofweek
rth = ret(930, 1319); on = ret(0, 930)
df = pd.DataFrame({"date":dts,"dow":dow,"rth":rth,"on":on,"D":D})
# trading-day-of-month
td = pd.Series(range(len(dts)), index=dts)
mon = dts.to_period("M")
df["tdm"] = df.groupby(mon).cumcount()+1
df["tdm_rev"] = df.groupby(mon).cumcount(ascending=False)+1
# opex: third friday
def third_fri(d):
    f = pd.Timestamp(d.year, d.month, 1); off=(4-f.dayofweek)%7; return f+pd.Timedelta(days=off+14)
df["opex"] = [third_fri(d) for d in dts]
df["opex_wk"] = [(0 <= (o - d).days <= 4) for d,o in zip(dts, df.opex)]
df["post_opex_wk"] = [(3 <= (d - o).days <= 7) for d,o in zip(dts, df.opex)]
x = df[df.D]
def st(s):
    s=s.dropna(); return f"n={len(s):3d} mean={s.mean():6.1f} t={s.mean()/s.std()*np.sqrt(len(s)):5.2f}"
for k in range(5): print("dow",k,"rth",st(x[x.dow==k].rth),"| on",st(x[x.dow==k].on))
tom = (x.tdm<=3)|(x.tdm_rev==1)
print("TOM rth",st(x[tom].rth),"| on",st(x[tom].on)); print("nonTOM rth",st(x[~tom].rth),"| on",st(x[~tom].on))
print("opexwk rth",st(x[x.opex_wk].rth),"| on",st(x[x.opex_wk].on)); print("postopex rth",st(x[x.post_opex_wk].rth),"| on",st(x[x.post_opex_wk].on))
