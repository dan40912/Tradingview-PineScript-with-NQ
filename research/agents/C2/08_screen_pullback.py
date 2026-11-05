"""08: Inverse of the losing 'reject' rule: buy the pullback into old value after the morning migrated. Discovery only."""
import c2lib as L
from lib import sim
import sig
g, lv, i2 = L.load()
fam = "08_pullback"
run = lambda s: sim.run_signals(g, s)
for T in [150, 165, 180, 195, 210]:
    for s in [0.35, 0.5]:
        for ex in [359, 389]:
            tr = run(sig.pm_pullback(g, lv, T=T, s=s, exit_idx=ex))
            L.disc(g, tr, f"PMP T={T} s={s} ex={ex}", fam)
tr = run(sig.pm_pullback(g, lv, T=180, s=0.35, exit_idx=359))
print(tr[tr.split == "discovery"][["date", "dir", "entry_time", "reason", "pnl"]].to_string())
print("corr", L.corr_i2(g, tr, i2, "discovery"))
