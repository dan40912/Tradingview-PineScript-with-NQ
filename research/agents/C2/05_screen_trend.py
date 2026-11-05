"""05: Trend-continuation family with value-area context (discovery only)."""
import numpy as np
import c2lib as L
from lib import sim
import sig
g, lv, i2 = L.load()
im = sig.i2_entry_idx(g, lv)
fam = "05_trend_va"
run = lambda s: sim.run_signals(g, s)
print("== day filter x rel (k .75, s .5, exit 15:29)")
for day in ["all", "outside", "inside", "out_noI2", "out_I2done"]:
    for rel in (["any", "toward", "away"] if day not in ("all", "inside") else ["any"]):
        tr = run(sig.trend_ext(g, lv, day=day, rel=rel, i2map=im))
        L.disc(g, tr, f"TE day={day} rel={rel}", fam)
print("== outside: neighbours")
for k in [0.6, 0.75, 0.9]:
    for s in [0.35, 0.5, 0.7]:
        tr = run(sig.trend_ext(g, lv, k=k, s=s, day="outside", i2map=im))
        L.disc(g, tr, f"TE outside k={k} s={s}", fam)
for ex in [329, 369, 389]:
    tr = run(sig.trend_ext(g, lv, exit_idx=ex, day="outside", i2map=im))
    L.disc(g, tr, f"TE outside exit_idx={ex}", fam)
for st in [0, 60]:
    tr = run(sig.trend_ext(g, lv, start=st, day="outside", i2map=im))
    L.disc(g, tr, f"TE outside start={st}", fam)
for en in [180, 265]:
    tr = run(sig.trend_ext(g, lv, end=en, day="outside", i2map=im))
    L.disc(g, tr, f"TE outside end={en}", fam)
print("== at-time decisions on outside days without I2 so far")
for T in [120, 150, 180]:
    for mode in ["move", "away", "toward"]:
        for mm in [0.0, 0.25, 0.5]:
            tr = run(sig.at_time(g, lv, T=T, mode=mode, min_mag=mm, i2map=im))
            L.disc(g, tr, f"AT T={T} {mode} mag>={mm}", fam)
