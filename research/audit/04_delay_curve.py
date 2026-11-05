"""Entry-timing sensitivity: per-trade average (2024-25, disc / val) for entry delays 0..5 minutes.
Large swings from a 1-minute shift mean the per-trade differences between candidates are inside timing noise."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib
fr = importlib.import_module("03_fragility") if False else None
import audit_common as A
exec(open(os.path.join(A.HERE, "03_fragility.py")).read().split("STRESS = {")[0])   # reuse helpers only
rows = []
names = ["BASE/I2v1"] + [n for n in C]
from lib import strategies
for name in names:
    rec = {"cand": name}
    for k in range(0, 6):
        if name == "BASE/I2v1":
            sigs = strategies.i2_v1_signals(g, lv)
            sigs = [dict(s, entry_slot=s["entry_slot"] + k) for s in sigs]
            tr = A.run(g, sigs)
        else:
            tr = evaluate(name, delay=k)
        rec[f"D+{k}"] = round(tr[tr.split == "discovery"].pnl.mean(), 0)
        rec[f"V+{k}"] = round(tr[tr.split == "validation"].pnl.mean(), 0)
    D = [rec[f"D+{k}"] for k in range(6)]
    rec["D_range"] = max(D) - min(D)
    rows.append(rec)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(A.HERE, "out_04_delay_curve.csv"), index=False)
with pd.option_context("display.width", 250):
    print(df.to_string(index=False))
