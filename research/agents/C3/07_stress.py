"""C3 step 7 (after validation, DESCRIPTIVE ONLY, nothing re-chosen): stress tests of the frozen candidates.
 - extra cost per full contract +$20 / +$40 (scaled for micros), 1-min late entry
 - ES leg: exit-reason mix, long/short, bootstrap CI of avg/trade, profit share by half-year
 - realised daily sd of the portfolios vs 1 NQ I2 v1 (same-risk-budget check), diversification ratio
 - leg-drop: what each portfolio would be without its ES leg"""
import numpy as np, pandas as pd
from c3lib import data, features, run, stats, tradable_days, daily, strategies, metrics
from candidates import _a3c3, _i2v1, es_i2_40_15, WEIGHTS
pd.set_option("display.width", 250)
gN = data.load_grid("NQ"); lvN = features.daily_levels(gN)
gE = data.load_grid("ES"); lvE = features.daily_levels(gE)
DAYS = {s: sorted(set(tradable_days(gN, s)) | set(tradable_days(gE, s))) for s in ["discovery", "validation"]}

def legs(late=0):
    sh = lambda sig: [dict(s, entry_slot=s["entry_slot"] + late) for s in sig]
    return {"I2v1": run(gN, sh(_i2v1(gN, lvN))), "A3C3": run(gN, sh(_a3c3(gN, lvN))),
            "ES_40_15": run(gE, sh(es_i2_40_15(gE, lvE)))}
# NB: for late entry the A3 POC target / no-progress window were computed at the original fill; acceptable approximation
PORT = {"P0 1xI2v1": {"I2v1": 1.0}, "P0b 1xA3C3": {"A3C3": 1.0}, "ES only": {"ES_40_15": 1.0},
        "P1 I2v1.6+ES.8": {"I2v1": 0.6, "ES_40_15": 0.8}, "P2 A3C3.9+ES.8": {"A3C3": 0.9, "ES_40_15": 0.8}}
rows = []
for label, late, extra in [("base", 0, 0), ("+$20/contract", 0, 20), ("+$40/contract", 0, 40), ("late 1 min", 1, 0)]:
    L = legs(late)
    for p, ws in PORT.items():
        tr = pd.concat([L[k].assign(pnl=(L[k].pnl - extra) * w) for k, w in ws.items()])
        for per, D in DAYS.items():
            st = stats(tr[tr.date.isin(D)], D)
            rows.append({"test": label, "portfolio": p, "period": per, **{k: st[k] for k in ("n", "total", "maxdd", "ret/dd", "sharpe", "worst_day")}})
print(pd.DataFrame(rows).to_string(index=False))

L = legs()
es = L["ES_40_15"]
print("\nES leg exit reasons by period:\n", pd.crosstab([es.split], es.reason))
print("ES leg avg by dir:\n", es.groupby(["split", "dir"]).pnl.agg(["count", "mean"]).round(0))
for per in ["discovery", "validation"]:
    p = es[es.split == per].pnl
    print(per, "bootstrap 95% CI avg/trade:", metrics.bootstrap_ci(p.values))
h = es.date.str[:4] + np.where(es.date.str[5:7] <= "06", "H1", "H2")
print("ES profit by half:", es.groupby(h).pnl.sum().round(0).to_dict())
print("ES stop-outs share of trades:", round((es.reason == 'sl').mean(), 2))

# realised risk
print("\nrealised daily sd / diversification")
for per, D in DAYS.items():
    out = {}
    for p, ws in PORT.items():
        comps = [daily(L[k], D) * w for k, w in ws.items()]
        tot = sum(comps)
        out[p] = f"sd {tot.std(ddof=1):.0f} (sum of leg sd {sum(c.std(ddof=1) for c in comps):.0f})"
    print(per, out)
