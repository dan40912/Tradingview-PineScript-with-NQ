import warnings; warnings.filterwarnings("ignore")
from a1lib import *
from signals import orb_break
from candidates import *
g, lv = load(); DISC = g.split == "discovery"
L = lv.set_index("i"); outVA = (((L.open_0930 > L.vah) | (L.open_0930 < L.val)) & L.has_prev).values
a = sim.run_signals(g, orb_break(g, lv, N=15, cutoff="12:00", exit_mode="or_eod", days=DISC & outVA))
b = sim.run_signals(g, OOV_ORB15_EOD(g, lv)); b = b[b.split == "discovery"]
print(len(a), len(b), a.pnl.sum(), b.pnl.sum())
