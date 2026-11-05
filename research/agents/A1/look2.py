import warnings; warnings.filterwarnings("ignore")
from a1lib import *
from signals import *
for sym in ["NQ", "ES"]:
    g, lv = load(sym); DISC = (g.split == "discovery")
    L = lv.set_index("i")
    outVA = (((L.open_0930 > L.vah) | (L.open_0930 < L.val)) & L.has_prev).values
    i2 = sim.run_signals(g, strategies.i2_v1_signals(g, lv)); i2 = i2[i2.split == "discovery"]
    i2d = set(i2.date)
    print(f"===== {sym}  disc tradable days {int((g.tradable&DISC).sum())}, outVA days {int((outVA&DISC&g.tradable).sum())}, I2 days {len(i2d)}")
    print("I2", summ(i2, g, "discovery"))
    for name, kw in [("OOV-ORB15-EOD", dict(N=15, exit_mode="or_eod")), ("OOV-ORB15-2R", dict(N=15, exit_mode="r_k", k=2)),
                     ("OOV-ORB30-1R", dict(N=30, exit_mode="r_k", k=1)), ("OOV-ORB15-fixed55", dict(N=15, exit_mode="fixed", tp=55 if sym=="NQ" else 13, sl=145 if sym=="NQ" else 35, exit_time="12:29"))]:
        tr = sim.run_signals(g, orb_break(g, lv, cutoff="12:00", days=DISC & outVA, **kw))
        s = summ(tr, g, "discovery")
        print(f"\n{name}: n={s['n']} avg={s['avg']} tot={s['total']} dd={s['maxdd']} r/dd={s['ret/dd']} sh={s['sharpe_ann']} t={s['t']}")
        print("  long", metrics.summary(tr[tr.dir==1])["avg"], len(tr[tr.dir==1]), " short", metrics.summary(tr[tr.dir==-1])["avg"], len(tr[tr.dir==-1]))
        on = tr[tr.date.isin(i2d)]; off = tr[~tr.date.isin(i2d)]
        print("  on I2 days", len(on), round(on.pnl.mean(),0), " off I2 days", len(off), round(off.pnl.mean(),0))
        # direction agreement with I2 on overlap days
        m = on.merge(i2[["date","dir","pnl"]], on="date", suffixes=("","_i2"))
        print("  same dir as I2:", int((m.dir==m.dir_i2).sum()), "avg", round(m[m.dir==m.dir_i2].pnl.mean(),0), "| opp dir:", int((m.dir!=m.dir_i2).sum()), "avg", round(m[m.dir!=m.dir_i2].pnl.mean(),0))
        # daily corr with I2 over all disc tradable days
        days = g.dates[g.tradable & DISC]
        a = tr.groupby("date").pnl.sum().reindex(days, fill_value=0); b = i2.groupby("date").pnl.sum().reindex(days, fill_value=0)
        comb = a + b
        dd = metrics.max_drawdown(comb.values)
        print(f"  daily corr w/ I2 {a.corr(b):.2f}; combined total {comb.sum():.0f} dd {dd:.0f} sharpe {comb.mean()/comb.std()*np.sqrt(252):.2f}")
        print("  reasons", tr.reason.value_counts().to_dict(), "median entry", sim.hhmm_of(int(tr.entry_slot.median())), "avg sl", round(np.mean(tr.pts[tr.reason=='sl']),1))
        if name == "OOV-ORB15-EOD": print(halves(tr).T)
