"""Discovery-only vectorized screens (no stops, points, cost 2.4pt). Intraday momentum family."""
import warnings; warnings.filterwarnings("ignore")
from a1lib import *
g, lv = load(); d = daytable(g, lv)
C, O = g.C, g.O
def cl(slot):  # close at slot (fallback previous available within 5 min)
    x = C[:, slot].copy()
    for k in range(1, 6):
        m = np.isnan(x); x[m] = C[m, slot - k]
    return x
def op(slot):
    x = O[:, slot].copy()
    for k in range(1, 6):
        m = np.isnan(x); x[m] = O[m, slot + k]
    return x
S = sim.slot_of
disc = (d.split == "discovery").values & d.tradable.values & d.has_prev.values
COST = 2.4
def rep(name, sig, ret):
    m = disc & np.isfinite(sig) & np.isfinite(ret) & (sig != 0)
    p = np.sign(sig[m]) * ret[m] - COST
    corr = np.corrcoef(sig[m], ret[m])[0, 1]
    t = p.mean() / p.std() * np.sqrt(len(p))
    print(f"{name:45s} n={m.sum():4d} corr={corr:+.3f} avg_pts={p.mean():+6.2f} $={20*p.mean():+7.1f} t={t:+.2f} long_only_avg={(ret[m]-COST).mean():+.2f}")
close = cl(S("15:59"))
pdc = d.pdc.values; o930 = d.open_0930.values
for es, xs in [("15:30", "15:59"), ("15:00", "15:59"), ("14:00", "15:59"), ("12:00", "15:59"), ("13:00", "15:59")]:
    ret = cl(S(xs)) - op(S(es))
    for p_end in ["10:00", "10:30", "11:00", "12:00", es]:
        pe = S(p_end) - 1
        if pe >= S(es): pe = S(es) - 1
        rep(f"pdc->{p_end} predicts {es}-{xs}", cl(pe) - pdc, ret)
        rep(f"open->{p_end} predicts {es}-{xs}", cl(pe) - o930, ret)
    # last half-hour before entry
    rep(f"prev30 predicts {es}-{xs}", cl(S(es) - 1) - cl(S(es) - 31), ret)
