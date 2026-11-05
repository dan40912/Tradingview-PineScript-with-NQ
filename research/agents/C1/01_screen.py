"""C1 step 1 (DISCOVERY only): entries x exits x overlays screen for the I2 family.
Writes out_01_screen.csv. Validation columns are NOT computed here."""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
import c1lib as c

g, lv = c.load()
ENTRIES = {
    "I2":       lambda: c.ent_i2(g, lv),
    "I2_1x5":   lambda: c.ent_i2(g, lv, m=1, step=5),
    "R1":       lambda: c.ent_i2(g, lv, m=1, step=1),
    "GAP0931":  lambda: c.ent_gap(g, lv, 1),
    "ORB5f":    lambda: c.ent_orb(g, lv, 5, True),
    "ORB15f":   lambda: c.ent_orb(g, lv, 15, True),
    "ORB30f":   lambda: c.ent_orb(g, lv, 30, True),
    "ORB15any": lambda: c.ent_orb(g, lv, 15, False),
    "DBL":      lambda: c.ent_dbl(g, lv, False),
    "DBLlater": lambda: c.ent_dbl(g, lv, True),
}
EXITS = {
    "FIX55_145":  dict(tp=55, sl=145),
    "A3":         dict(tp="poc", poc_floor=20, sl=60, np=(30, 20)),
    "POC_145":    dict(tp="poc", poc_floor=20, sl=145),
    "POC_60":     dict(tp="poc", poc_floor=20, sl=60),
    "POC_145_NP": dict(tp="poc", poc_floor=20, sl=145, np=(30, 20)),
    "EOD_145":    dict(tp=None, sl=145, exit="1559"),
    "EOD_110_145": dict(tp=110, sl=145, exit="1559"),
    "R2_60_NP_EOD": dict(tp=120, sl=60, np=(30, 20), exit="1559"),
    "EOD_60_NP":  dict(tp=None, sl=60, np=(30, 20), exit="1559"),
}
OVL = {"none": dict(), "volpoor": dict(volpoor=0.25), "trend": dict(trend=True), "both": dict(volpoor=0.25, trend=True)}

rows = []
for en, ef in ENTRIES.items():
    ents = ef()
    ents_d = {i: s for i, s in ents.items() if g.split[i] == "discovery"}
    for xn, xs in EXITS.items():
        base = c.make_signals(g, lv, ents_d, xs)
        for on, ok in OVL.items():
            sigs = c.overlay(base, lv, **ok)
            tr = c.run(g, sigs)
            r = c.full_row(f"{en}|{xn}|{on}", tr, g)
            r.update(entry=en, exit=xn, ovl=on)
            rows.append(r)
df = pd.DataFrame(rows)
df.to_csv("out_01_screen.csv", index=False)
print("variants:", len(df))
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
cols = ["variant", "D_n", "D_win%", "D_avg", "D_total", "D_maxdd", "D_ret/dd", "D_sharpe", "D_worst", "D_lstreak",
        "2024H1_avg", "2024H2_avg", "2025H1_avg", "halves_pos"]
print(df[df.ovl == "none"][cols].to_string(index=False))
