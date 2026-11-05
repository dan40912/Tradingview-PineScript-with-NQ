"""E1/E2: fixed-point TP x SL grid and exit-time grid on I2 v1 entries, DISCOVERY ONLY."""
import numpy as np, pandas as pd, itertools
from a3lib import load, i2_entries, with_exit, run, stats, RTH, EXIT_1229
from lib import sim

g, lv = load("NQ")
sigs = [s for s in i2_entries(g, lv) if g.split[s["i"]] == "discovery"]
INF = np.inf
TPS = [20, 30, 40, 55, 70, 85, 100, 120, 150, 200, INF]
SLS = [30, 45, 60, 80, 100, 120, 145, 175, 220, INF]
rows = []
for tp, sl in itertools.product(TPS, SLS):
    tr = run(g, with_exit(g, lv, sigs, tp=tp, sl=sl))
    s = stats(tr, g, "discovery")
    rows.append({"tp": tp, "sl": sl, **{k: s[k] for k in ("n", "win%", "avg", "total", "maxdd", "ret/dd", "sharpe_ann")}})
df = pd.DataFrame(rows)
df.to_csv("out_grid_pts.csv", index=False)
pd.set_option("display.width", 250)
for k in ["avg", "sharpe_ann", "maxdd", "ret/dd", "win%"]:
    print(f"\n=== {k} (rows TP, cols SL) exit 12:29 ===")
    print(df.pivot(index="tp", columns="sl", values=k))
# exit-time grid
EX = {"10:30": sim.slot_of("10:30"), "11:00": sim.slot_of("11:00"), "11:30": sim.slot_of("11:30"), "12:29": EXIT_1229,
      "13:30": sim.slot_of("13:30"), "14:30": sim.slot_of("14:30"), "15:55": sim.slot_of("15:55")}
rows = []
for (tp, sl) in [(55, 145), (85, 145), (100, 145), (100, 100), (INF, 145), (INF, INF), (150, 145), (55, INF), (100, INF)]:
    for nm, xs in EX.items():
        tr = run(g, with_exit(g, lv, sigs, tp=tp, sl=sl, exit_slot=xs))
        s = stats(tr, g, "discovery")
        rows.append({"tp": tp, "sl": sl, "exit": nm, **{k: s[k] for k in ("n", "win%", "avg", "maxdd", "ret/dd", "sharpe_ann")}})
dx = pd.DataFrame(rows)
dx.to_csv("out_grid_exit_time.csv", index=False)
print("\n=== exit time grid ===")
print(dx.pivot_table(index=["tp", "sl"], columns="exit", values="avg"))
print(dx.pivot_table(index=["tp", "sl"], columns="exit", values="sharpe_ann"))
print(dx.pivot_table(index=["tp", "sl"], columns="exit", values="maxdd"))
print("variants:", len(df) + len(dx))
