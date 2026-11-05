"""POST-HOC (after seeing validation): OOV-ORB split by breakout direction relative to value.
Not a candidate; reported only as a lead with a contamination warning."""
import warnings; warnings.filterwarnings("ignore")
from a1lib import *
from candidates import oov_orb
for sym in ["NQ", "ES"]:
    g, lv = load(sym); L = lv.set_index("i")
    side = np.where(L.open_0930 > L.vah, 1, np.where(L.open_0930 < L.val, -1, 0))  # +1 open above value
    i2d = i2_days(g, lv)
    for name, kw in [("ORB15_EOD", dict(N=15, k=None)), ("ORB30_1R", dict(N=30, k=1.0))]:
        tr = sim.run_signals(g, oov_orb(g, lv, **kw))
        tr["toward"] = tr.dir == -side[tr.i.values]
        for tw in [True, False]:
            x = tr[tr.toward == tw]
            t = table(x, g)
            for sp in ["discovery", "validation"]:
                y = x[x.split == sp]
                print(f"{sym} {name} toward_value={tw} {sp}: n={t.loc[sp,'n']} avg={t.loc[sp,'avg']} dd={t.loc[sp,'maxdd']} sh={t.loc[sp,'sharpe_ann']}"
                      f" | on-I2 {y.date.isin(i2d).sum()} avg {y[y.date.isin(i2d)].pnl.mean():.0f} off-I2 {(~y.date.isin(i2d)).sum()} avg {y[~y.date.isin(i2d)].pnl.mean():.0f}")
