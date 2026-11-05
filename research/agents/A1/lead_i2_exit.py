"""Lead for other agents (discovery only): I2 v1 entries with alternative fixed exits."""
import warnings; warnings.filterwarnings("ignore")
from a1lib import *
g, lv = load()
base = strategies.i2_v1_signals(g, lv)
EOD = sim.slot_of("15:59")
for name, upd in [("I2 v1 (+55/-145, 12:29)", {}), ("I2 +55/-145 EOD", {"exit_slot": EOD}),
                  ("I2 noTP/-145 12:29", {"tp_pts": None}), ("I2 noTP/-145 EOD", {"tp_pts": None, "exit_slot": EOD}),
                  ("I2 +110/-145 EOD", {"tp_pts": 110, "exit_slot": EOD})]:
    sg = [dict(s, **upd) for s in base if g.split[s["i"]] == "discovery"]
    tr = sim.run_signals(g, sg); s = summ(tr, g, "discovery")
    print(f"{name:28s} n={s['n']} avg={s['avg']} tot={s['total']} dd={s['maxdd']} r/dd={s['ret/dd']} sh={s['sharpe_ann']}")
