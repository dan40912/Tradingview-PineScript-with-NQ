from overlap import *
pf_ = pf
def fA(tr, th=0.25):
    m = np.array([(pf_.loc[t.i, "vol_top_rel"] if t.dir == -1 else pf_.loc[t.i, "vol_bot_rel"]) < th for t in tr.itertuples()])
    return m
for name, tr in [("S3 k3m2", sim.run_signals(g, s3(3, 2))), ("ONVA m2", sim.run_signals(g, s10f("on", 2))),
                 ("S3 k2m2", sim.run_signals(g, s3(2, 2))), ("S3 k1m1", sim.run_signals(g, s3(1, 1))), ("ONVA m1", sim.run_signals(g, s10f("on", 1)))]:
    D = tr[tr.split == "discovery"].reset_index(drop=True)
    m = fA(D)
    print(f"{name}: base {D.pnl.mean():.0f} n={len(D)} | keep {m.sum()} avg {D.pnl[m].mean():.0f} | drop {(~m).sum()} avg {D.pnl[~m].mean():.0f} p={perm_p(D.pnl.values, m):.3f}")
