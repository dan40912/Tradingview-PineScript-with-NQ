"""07: Afternoon value-migration family (entry 12:30 or later, after I2's window). Discovery only."""
import c2lib as L
from lib import sim
import sig
g, lv, i2 = L.load()
fam = "07_pm"
run = lambda s: sim.run_signals(g, s)
for T in [180, 210, 240]:
    for cond in ["accept", "reject"]:
        for day in ["all", "inside", "outside"]:
            tr = run(sig.pm_migration(g, lv, T=T, cond=cond, day=day))
            L.disc(g, tr, f"PMM T={T} {cond} day={day}", fam)
