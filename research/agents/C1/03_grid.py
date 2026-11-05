"""C1 step 3 (DISCOVERY only): robustness grids around the shortlisted structures.
Shortlist (from 01/02, discovery only): entries I2 / ORB-N toward value (any, first) / FIRST(I2, ORB-N);
exits A3-style POC target, and TP/SL hold to close. Writes out_03_grid.csv."""
import warnings; warnings.filterwarnings("ignore")
import itertools, numpy as np, pandas as pd
import c1lib as c
g, lv = c.load()
D = lambda e: {i: s for i, s in e.items() if g.split[i] == "discovery"}
i2 = D(c.ent_i2(g, lv))
ENT = {"I2": i2}
for N in (10, 15, 20):
    a = D(c.ent_orb(g, lv, N, False)); f = D(c.ent_orb(g, lv, N, True))
    ENT[f"ORB{N}any"] = a; ENT[f"ORB{N}f"] = f
    ENT[f"FIRST(I2,ORB{N}any)"] = {i: min(i2.get(i, 10**9), a.get(i, 10**9)) for i in set(i2) | set(a)}
rows = []
# POC family
for en, ents in ENT.items():
    for fl, sl, npx, cap, ex in itertools.product((15, 20, 25), (45, 60, 75), (None, (20, 15), (30, 20), (45, 25)), (None, 110), ("1229", "1559")):
        xs = dict(tp="poc", poc_floor=fl, sl=sl, np=npx, cap=cap, exit=ex)
        for ov, ok in (("none", {}), ("trend", dict(trend=True))):
            tr = c.run(g, c.overlay(c.make_signals(g, lv, ents, xs), lv, **ok))
            r = c.full_row(f"{en}|POC fl{fl} sl{sl} np{npx} cap{cap} {ex}|{ov}", tr, g)
            r.update(entry=en, fam="POC", fl=fl, sl=sl, np=str(npx), cap=cap, exit=ex, ovl=ov); rows.append(r)
# TP/SL hold family
for en, ents in ENT.items():
    for tp, sl, ex in itertools.product((90, 110, 130), (120, 145, 170), ("15:00", "15:30", "1559")):
        xs = dict(tp=tp, sl=sl, exit=ex)
        for ov, ok in (("none", {}), ("trend", dict(trend=True))):
            tr = c.run(g, c.overlay(c.make_signals(g, lv, ents, xs), lv, **ok))
            r = c.full_row(f"{en}|TP{tp} SL{sl} {ex}|{ov}", tr, g)
            r.update(entry=en, fam="TPSL", tp=tp, sl=sl, exit=ex, ovl=ov); rows.append(r)
df = pd.DataFrame(rows); df.to_csv("out_03_grid.csv", index=False)
print("variants:", len(df))
