import warnings; warnings.filterwarnings("ignore")
from a1lib import *
from signals import *
g, lv = load(); DISC = (g.split == "discovery")
i2 = sim.run_signals(g, strategies.i2_v1_signals(g, lv))
print("I2", summ(i2, g, "discovery")); print(halves(i2[i2.split=="discovery"]).T)
for name, sigs in [("ORB15 r_k3 cut13", orb_break(g, lv, N=15, cutoff="13:00", exit_mode="r_k", k=3, days=DISC)),
                   ("ORB30 r_k1 cut11", orb_break(g, lv, N=30, cutoff="11:00", exit_mode="r_k", k=1, days=DISC)),
                   ("ORB30 or_eod cut11", orb_break(g, lv, N=30, cutoff="11:00", exit_mode="or_eod", days=DISC)),
                   ("ORB30 fixed55", orb_break(g, lv, N=30, cutoff="11:00", exit_mode="fixed", tp=55, sl=145, exit_time="12:29", days=DISC)),
                   ("CANDLE15", orb_candle(g, lv, N=15, days=DISC)), ("ORB5 or_eod", orb_break(g, lv, N=5, cutoff="11:00", exit_mode="or_eod", days=DISC))]:
    tr = sim.run_signals(g, sigs)
    print("\n", name, summ(tr, g, "discovery")); print(halves(tr).T)
    print(tr.reason.value_counts().to_dict(), "avg sl pts", round(np.mean([s["sl_pts"] for s in sigs]),1))
