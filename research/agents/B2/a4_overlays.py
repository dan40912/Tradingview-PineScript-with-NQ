"""A4: risk overlays (skip / halve / vol-target sizing) on DISCOVERY. Scale-free metrics: ret/dd, Sharpe, avg per unit."""
from util import *
SP = "/tmp/claude-0/-home-user-Tradingview-PineScript-with-NQ/d4597cec-8294-562a-9e11-5568b74be6dd/scratchpad/"
g, lv, rf, ev = load()
ref_atr = pd.Series(rf.atr20.values).rolling(250, min_periods=40).median().shift(1).values
ref_rv = pd.Series(rf.rv_rth10.values).rolling(250, min_periods=40).median().shift(1).values
rf["vt_atr"] = np.clip(np.round(ref_atr / rf.atr20.values, 1), 0.5, 2.0)
rf["vt_rv"] = np.clip(np.round(ref_rv / rf.rv_rth10.values, 1), 0.5, 2.0)
rf["vt_atr_cap1"] = np.clip(np.round(ref_atr / rf.atr20.values, 1), 0.3, 1.0)
for c in ["vt_atr", "vt_rv", "vt_atr_cap1"]:
    rf[c] = rf[c].fillna(1.0)
OVER = {
    "base": lambda t: np.ones(len(t)),
    "VT_atr(0.5-2)": lambda t: t.vt_atr.values,
    "VT_rv(0.5-2)": lambda t: t.vt_rv.values,
    "VT_atr_cap1": lambda t: t.vt_atr_cap1.values,
    "halve_atrpct>0.8": lambda t: np.where(t.atr20_pct > 0.8, 0.5, 1.0),
    "skip_atrpct>0.8": lambda t: np.where(t.atr20_pct > 0.8, 0.0, 1.0),
    "skip_atrpct<0.2": lambda t: np.where(t.atr20_pct < 0.2, 0.0, 1.0),
    "halve_fear>0.7": lambda t: np.where(t.fear > 0.7, 0.5, 1.0),
    "skip_fomc": lambda t: np.where(t.fomc, 0.0, 1.0),
    "skip_fomc+next": lambda t: np.where(t.fomc | t.fomc_next, 0.0, 1.0),
    "skip_tier1": lambda t: np.where(t.any_tier1, 0.0, 1.0),
    "halve_tier1": lambda t: np.where(t.any_tier1, 0.5, 1.0),
    "skip_rel830": lambda t: np.where(t.rel830, 0.0, 1.0),
    "only_rel830": lambda t: np.where(t.rel830, 1.0, 0.0),
    "skip_roll": lambda t: np.where(t.roll_window, 0.0, 1.0),
    "skip_absgap>1atr": lambda t: np.where(t.absgap_atr > 1.0, 0.0, 1.0),
    "skip_onrange>1atr": lambda t: np.where(t.on_range_atr > 1.0, 0.0, 1.0),
}
nd = ndays(g, "discovery")
nvar = 0
for s in ["I2v1", "ORB30_1R", "ORB15_eod", "GAPF_all"]:
    tr = attach(pd.read_pickle(SP + f"tr_{s}.pkl"), rf, ev)
    tr = tr[tr.split == "discovery"].reset_index(drop=True)
    rows = []
    for k, f in OVER.items():
        nvar += 1
        w = f(tr)
        t = tr.assign(pnl=tr.pnl * w)
        t = t[w > 0]
        sm = metrics.summary(t, n_days=nd)
        avg_unit = t.pnl.sum() / w[w > 0].sum() if len(t) else np.nan
        rows.append({"overlay": k, "n": sm.get("n", 0), "units": round(w.sum(), 1), "avg/unit": round(avg_unit, 0),
                     "total": sm.get("total"), "maxdd": sm.get("maxdd"), "ret/dd": sm.get("ret/dd"),
                     "sharpe": sm.get("sharpe_ann"), "worst_day": round(t.groupby('date').pnl.sum().min(), 0) if len(t) else np.nan})
    print("==", s); print(pd.DataFrame(rows).to_string())
print("overlay variants:", nvar)
