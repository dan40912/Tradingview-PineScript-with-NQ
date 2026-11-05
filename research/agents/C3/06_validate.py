"""C3 step 6: VALIDATION (2025H2) of the frozen candidates (candidates.py, sha256 in FROZEN_candidates.sha256)
plus pre-registered reference rows. No parameter is changed here. Never loads 2026."""
import sys, os, numpy as np, pandas as pd
from c3lib import data, features, levels, i2_exit_sigs, run, stats, tradable_days, daily, strategies, half_avgs
from candidates import CANDIDATES, run_portfolio, WEIGHTS, _a3c3, _i2v1, es_i2_40_15
HERE = os.path.dirname(os.path.abspath(__file__))
import importlib.util
def _load(agent):
    spec = importlib.util.spec_from_file_location(f"cand_{agent}", os.path.join(HERE, "..", agent, "candidates.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
B1 = _load("B1"); B2 = _load("B2")
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70)

gN = data.load_grid("NQ"); lvN = features.daily_levels(gN)
gE = data.load_grid("ES"); lvE = features.daily_levels(gE)
assert not (gN.split == "holdout").any() and not (gE.split == "holdout").any()
DAYS = {s: sorted(set(tradable_days(gN, s)) | set(tradable_days(gE, s))) for s in ["discovery", "validation"]}
DAYS["2024-25"] = DAYS["discovery"] + DAYS["validation"]

def tbl(streams):
    rows = []
    for name, tr in streams.items():
        for per, D in DAYS.items():
            td = tr[tr.date.isin(D)]
            dv = daily(td, D)
            rows.append({"stream": name, "period": per, **stats(td, D), "avg_day": round(dv.mean(), 1)})
    return pd.DataFrame(rows)

# ---- ES streams ---------------------------------------------------------------------------------
es = {"C3_ES_I2_40_15 (cand)": run(gE, es_i2_40_15(gE, lvE)),
      "ES 22/58 shadow (contaminated)": run(gE, i2_exit_sigs(gE, lvE, tp=22, sl=58)),
      "ES A3-analog POC f10/s20/np30-4": run(gE, i2_exit_sigs(gE, lvE, poc_floor=10, sl=20, np_n=30, np_x=4)),
      "ES I2 v1 55/145 (unscaled)": run(gE, strategies.i2_v1_signals(gE, lvE))}
t_es = tbl(es); print("=== ES streams (1 ES)"); print(t_es.to_string(index=False))
# neighbours of 40/15 (info only)
nb = []
for b in [5.0, 2.5, 1.25]:
    lv = levels(gE, b)
    for T, S in ([(40, 15), (30, 15), (55, 15), (40, 10), (40, 22)] if b == 5.0 else [(40, 15)]):
        tr = run(gE, i2_exit_sigs(gE, lv, tp=T, sl=S))
        for per in ["discovery", "validation"]:
            td = tr[tr.split == per]
            nb.append({"bin": b, "tp": T, "sl": S, "period": per, **{k: v for k, v in stats(td, DAYS[per]).items() if k in ("n", "avg", "maxdd", "sharpe")}})
nb = pd.DataFrame(nb); print("\n=== ES 40/15 neighbours (validation = info only)"); print(nb.to_string(index=False))
# overlap: ES on NQ-I2 days vs not
i2days = set(run(gN, strategies.i2_v1_signals(gN, lvN)).date)
print("\n=== overlap of C3_ES_I2_40_15 with NQ I2 days")
for per in ["discovery", "validation"]:
    t = es["C3_ES_I2_40_15 (cand)"]; t = t[t.split == per]; on = t.date.isin(i2days)
    print(per, f"on NQ-I2 days n={on.sum()} avg={t.pnl[on].mean():.0f} sum={t.pnl[on].sum():.0f} | off n={(~on).sum()} avg={t.pnl[~on].mean():.0f} sum={t.pnl[~on].sum():.0f}")

# ---- portfolios ---------------------------------------------------------------------------------
def scaled(tr, w, sym):
    t = tr.copy(); t["pnl"] = t.pnl * w; t["symbol"] = sym; return t
raw = {"NQ_I2v1": run(gN, _i2v1(gN, lvN)), "NQ_A3C3": run(gN, _a3c3(gN, lvN)),
       "NQ_B1volpoor": run(gN, B1.CANDIDATES["B1_I2_volpoor"](gN, lvN)),
       "NQ_B2trend": run(gN, B2.CANDIDATES["B2_I2_TrendSize"](gN, lvN)),
       "ES_40_15": es["C3_ES_I2_40_15 (cand)"], "ES_22_58": es["ES 22/58 shadow (contaminated)"]}
PW = {"P0 1xNQ_I2v1 (benchmark)": {"NQ_I2v1": 1.0},
      "P0b 1xNQ_A3C3": {"NQ_A3C3": 1.0},
      "P1 C3_PORT_I2v1_ES (cand) 0.6NQ+0.8ES": {"NQ_I2v1": 0.6, "ES_40_15": 0.8},
      "P2 C3_PORT_A3C3_ES (cand) 0.9NQ+0.8ES": {"NQ_A3C3": 0.9, "ES_40_15": 0.8},
      "P2c A3C3 1.0 + ES22/58 0.9 (contam)": {"NQ_A3C3": 1.0, "ES_22_58": 0.9},
      "P3 A3C3 .7+I2v1 .4+ES .6": {"NQ_A3C3": 0.7, "NQ_I2v1": 0.4, "ES_40_15": 0.6},
      "P4 B1volpoor .7+ES .9": {"NQ_B1volpoor": 0.7, "ES_40_15": 0.9},
      "P5 B2trend .8+ES .9": {"NQ_B2trend": 0.8, "ES_40_15": 0.9}}
ports = {p: pd.concat([scaled(raw[k], w, "ES" if k.startswith("ES") else "NQ") for k, w in ws.items()]) for p, ws in PW.items()}
# consistency with candidates.run_portfolio
for p, n in [("P1 C3_PORT_I2v1_ES (cand) 0.6NQ+0.8ES", "C3_PORT_I2v1_ES"), ("P2 C3_PORT_A3C3_ES (cand) 0.9NQ+0.8ES", "C3_PORT_A3C3_ES")]:
    a = run_portfolio(n); assert abs(a.pnl.sum() - ports[p].pnl.sum()) < 1e-6, (p, a.pnl.sum(), ports[p].pnl.sum())
t_p = tbl(ports); print("\n=== portfolios (same discovery risk budget as 1 NQ I2 v1)"); print(t_p.to_string(index=False))
# half-year and year
print("\n=== per half-year total $")
hy = pd.DataFrame({p: tr.groupby(tr.date.str[:4] + np.where(tr.date.str[5:7] <= "06", "H1", "H2")).pnl.sum().round(0) for p, tr in ports.items()})
print(hy.T.to_string())
print("\n=== monthly validation total $ (P0 vs P1 vs P2)")
mo = pd.DataFrame({p: ports[p][ports[p].split == "validation"].groupby(lambda i: None).size() for p in []})
for p in list(PW)[:4]:
    t = ports[p][ports[p].split == "validation"]
    print(p[:40].ljust(40), t.groupby(t.date.str[:7]).pnl.sum().round(0).to_dict())
# daily correlation of the two legs in validation
dv = pd.DataFrame({k: daily(raw[k], DAYS["validation"]) for k in ["NQ_I2v1", "NQ_A3C3", "ES_40_15"]})
print("\nvalidation daily corr:\n", dv.corr().round(2))
t_es.to_csv("out_06_es_streams.csv", index=False); t_p.to_csv("out_06_portfolios.csv", index=False); nb.to_csv("out_06_es_neighbours.csv", index=False)
