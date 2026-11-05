"""13: CONTAMINATED hypothesis (not a candidate): C2_TE_OUTSIDE only when I2 has not entered before the signal.
It was in the discovery screen (05, 'TE day=out_noI2') before validation, but it is singled out now because the
validation split by I2 status (published by A2 for its parent rule) favours it -> treat as forward-test only."""
import numpy as np, pandas as pd
import c2lib as L
from lib import sim
import candidates as C
import sig
g, lv, i2 = L.load()
im = sig.i2_entry_idx(g, lv)
s1 = [s for s in C.te_outside(g, lv, sized=False) if not (s["i"] in im and im[s["i"]] < s["entry_slot"] - L.RTH)]
for lab, ss in [("1ct", s1), ("$1000 risk", C._size(s1, g, True))]:
    tr = sim.run_signals(g, ss)
    for sp in ["discovery", "validation"]:
        s = L.summ(g, tr, sp); c = L.combo(g, [i2, tr], sp)
        print(f"TE_OUT_NOI2 {lab:10s} {sp:10s} n={s['n']} win={s['win%']} avg={s['avg']} tot={s['total']} dd={s['maxdd']} "
              f"r/dd={s['ret/dd']} sh={s['sharpe_ann']} corr={L.corr_i2(g, tr, i2, sp)} | I2+it: tot={c['total']} dd={c['maxdd']} r/dd={c['ret/dd']} sh={c['sharpe_ann']}")
    print("   halves:", {h: L.summ(g, tr, hal=h).get("avg") for h in ["2024H1", "2024H2", "2025H1", "2025H2"]})
    c = L.combo(g, [i2, tr]); print("   I2+it 2024-2025:", {k: c[k] for k in ["total", "maxdd", "ret/dd", "sharpe_ann"]})
