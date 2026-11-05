"""Discovery-only neighbour checks for the 3 shortlisted filters."""
from common import *
T = pd.read_pickle(os.path.join(HERE, "_i2_trades_feats.pkl"))
g, lv = load("NQ")
rH, rL, rV = g.rth(g.H), g.rth(g.L), g.rth(g.V)
D = T[T.split == "discovery"].reset_index(drop=True)
pnl = D.pnl.values
def rep(name, m):
    m = np.asarray(m, bool)
    print(f"{name:40s} keep {m.sum():3d}/{len(m)} kept {pnl[m].mean():7.0f} drop {pnl[~m].mean() if (~m).any() else np.nan:7.0f} p={perm_p(pnl,m):.3f}")
# poor extreme neighbours
def poor(i, d, period, bin_, mincnt=2):
    j = g.prev_rth_day(i)
    k, c = ind.tpo_counts(rH[j], rL[j], period, bin_)
    return (c[-1] if d == -1 else c[0]) >= mincnt
for per in (15, 30, 60):
    for b in (2.5, 5, 10):
        m = np.array([not poor(t.i, t.dir, per, b) for t in D.itertuples()])
        rep(f"no poor (TPO {per}m, bin {b})", m)
# volume poor: extreme 5pt bin volume rel to mean bin
pf = ind.profile_features(g, lv)
for th in (0.15, 0.2, 0.25, 0.3):
    m = np.array([(pf.loc[t.i, "vol_top_rel"] if t.dir == -1 else pf.loc[t.i, "vol_bot_rel"]) < th for t in D.itertuples()])
    rep(f"vol extreme bin rel < {th}", m)
for th in (1.25, 1.5, 1.84, 2.0, 2.5, 3.0):
    rep(f"px_vs_vwap930_sd <= {th}", D.px_vs_vwap930_sd.values <= th)
for th in (1.0, 1.25, 1.59, 2.0, 2.5):
    rep(f"px_vs_vwap18_sd <= {th}", D.px_vs_vwap18_sd.values <= th)
for th in (-100, -80, -62, -40, -20):
    rep(f"dpoc_ahead >= {th}", D.dpoc_ahead.values >= th)
for th in (0.7, 0.8, 0.9, 1.0, 1.1):
    rep(f"or_rvol5 >= {th}", D.or_rvol5.values >= th)
for th in (0.7, 0.8, 0.9, 1.0):
    rep(f"rvol_to_entry >= {th}", D.rvol_to_entry.values >= th)
print("--- combos")
va = np.array([(pf.loc[t.i, "vol_top_rel"] if t.dir == -1 else pf.loc[t.i, "vol_bot_rel"]) < 0.25 for t in D.itertuples()])
vw = D.px_vs_vwap930_sd.values <= 2.0
rep("A volpoor<0.25", va); rep("B vwap930<=2", vw); rep("A&B", va & vw)
print(pd.crosstab(va, vw))
# H1 overlap
h1 = D.open_dist_w.values <= 1.0
rep("H1", h1); rep("A&H1", va & h1)
print(pd.crosstab(va, h1))
