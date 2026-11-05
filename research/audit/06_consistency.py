"""Consistency: numbers written in each REPORT.md (transcribed by the auditor) vs recomputation with the shared
engine (same as research/run_holdout.py --no-holdout, C3 portfolios as true two-leg books). Also: how many
candidates are really distinct (daily P&L correlation, share of trades on I2 v1 days). Output: out_06_*.csv"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_common as A
from lib import strategies, sim

g, lv = A.load("NQ")
C = A.all_candidates()
# (cand, split): reported n, avg, maxdd, sharpe   [from the agents' REPORT.md tables]
REP = {
    ("A1/OOV_ORB15_EOD", "D"): (231, 553, -21986, 1.91), ("A1/OOV_ORB15_EOD", "V"): (86, 73, -10382, 0.42),
    ("A1/OOV_ORB30_1R", "D"): (226, 513, -12322, 2.70), ("A1/OOV_ORB30_1R", "V"): (84, -130, -17265, -0.79),
    ("A1/OOV_ORB30_1R_RISK1500", "D"): (226, 292, -5700, 3.24), ("A1/OOV_ORB30_1R_RISK1500", "V"): (84, -67, -8478, -0.80),
    ("A2/A2_TREND_EXT_FOLLOW", "D"): (134, 443, -15969, 1.29), ("A2/A2_TREND_EXT_FOLLOW", "V"): (49, 125, -16102, 0.41),
    ("A3/A3_C3_POC_SL60_NP30", "D"): (81, 195, -3583, 1.71), ("A3/A3_C3_POC_SL60_NP30", "V"): (36, 211, -1719, 2.10),
    ("B1/B1_I2_volpoor", "D"): (70, 294, -5230, 1.72), ("B1/B1_I2_volpoor", "V"): (32, 43, -5838, 0.26),
    ("B1/B1_ONVA_I2", "D"): (84, 224, -5641, 1.33), ("B1/B1_ONVA_I2", "V"): (23, 22, -8043, 0.10),
    ("B2/B2_I2_TrendSize", "D"): (81, 171, -4111, 1.26), ("B2/B2_I2_TrendSize", "V"): (36, 25, -3166, 0.21),
    ("B3/B3_NQES_RS_0945_T120", "D"): (374, 280, -30943, 1.78), ("B3/B3_NQES_RS_0945_T120", "V"): (128, 87, -24224, 0.52),
    ("C1/C1_I2_A3X_TS", "D"): (81, 197, -2638, 2.14), ("C1/C1_I2_A3X_TS", "V"): (36, 117, -1886, 1.43),
    ("C1/C1_ORB15_POC_A3X_TS", "D"): (143, 218, -5246, 1.83), ("C1/C1_ORB15_POC_A3X_TS", "V"): (52, 248, -4846, 1.28),
    ("C1/C1_ORB15_TP110_TS", "D"): (143, 377, -6054, 2.26), ("C1/C1_ORB15_TP110_TS", "V"): (52, 156, -5896, 0.99),
    ("C2/C2_TE_OUTSIDE", "D"): (90, 218, -5457, 1.82), ("C2/C2_TE_OUTSIDE", "V"): (32, 39, -5008, 0.28),
    ("C2/C2_TE_OUTSIDE_NOI2", "D"): (68, 240, -3022, 1.83), ("C2/C2_TE_OUTSIDE_NOI2", "V"): (20, 312, -1894, 1.58),
    ("C3/C3_ES_I2_40_15", "D"): (76, 154, -3114, 1.14), ("C3/C3_ES_I2_40_15", "V"): (23, 517, -858, 4.13),
    ("C3/C3_PORT_A3C3_ES", "D"): (157, 150, -3076, 1.76), ("C3/C3_PORT_A3C3_ES", "V"): (59, 277, -1547, 3.45),
    ("C3/C3_PORT_I2v1_ES", "D"): (157, 105, -5654, 1.22), ("C3/C3_PORT_I2v1_ES", "V"): (59, 164, -2848, 2.04),
}
rows = []
TR = {}
for name in C:
    tr = A.trades(name, C)
    TR[name] = tr
    for sp, lab in (("discovery", "D"), ("validation", "V")):
        s = A.summ(g, tr, sp)
        r = REP.get((name, lab))
        rec = dict(cand=name, split=lab, n=s["n"], avg=s["avg"], maxdd=s["maxdd"], sharpe=s["sharpe_ann"])
        if r:
            rec.update(rep_n=r[0], rep_avg=r[1], rep_maxdd=r[2], rep_sharpe=r[3])
            rec["match"] = (r[0] == s["n"] and abs(r[1] - s["avg"]) <= 1.5 and abs(r[2] - s["maxdd"]) <= 2
                            and abs(r[3] - s["sharpe_ann"]) <= 0.05)
        rows.append(rec)
d = pd.DataFrame(rows)
d.to_csv(os.path.join(A.HERE, "out_06_consistency.csv"), index=False)
with pd.option_context("display.width", 220):
    print(d.to_string(index=False))

# harness view of C3 portfolios (NQ leg only) vs true portfolio
print("\nHarness scores C3 portfolios on the NQ leg only:")
for name in ("C3/C3_PORT_A3C3_ES", "C3/C3_PORT_I2v1_ES"):
    nq = sim.run_signals(g, C[name](g, lv))
    print(f"  {name}: harness-equivalent NQ leg total {nq.pnl.sum():.0f} (n={len(nq)}) vs two-leg total {TR[name].pnl.sum():.0f} (n={len(TR[name])})")

# distinctness
i2 = sim.run_signals(g, strategies.i2_v1_signals(g, lv))
i2days = set(i2.date)
dv = pd.DataFrame({n: A.daily_vec(g, t) for n, t in TR.items()})
dv["BASE/I2v1"] = A.daily_vec(g, i2)
corr = dv.corr().round(2)
corr.to_csv(os.path.join(A.HERE, "out_06_corr.csv"))
ov = pd.Series({n: round(t.date.isin(i2days).mean(), 2) for n, t in TR.items()}, name="share_of_trades_on_I2_days")
print("\nShare of each candidate's trades that fall on I2 v1 trade days (2024-25):")
print(ov.to_string())
print("\nDaily P&L correlation with I2 v1 and with A3_C3:")
print(corr[["BASE/I2v1", "A3/A3_C3_POC_SL60_NP30"]].to_string())
