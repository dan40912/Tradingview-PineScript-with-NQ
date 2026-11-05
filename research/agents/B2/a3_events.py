"""A3: event-day vs normal-day behaviour (DISCOVERY; validation hidden). Also morning range descriptive."""
from util import *
SP = "/tmp/claude-0/-home-user-Tradingview-PineScript-with-NQ/d4597cec-8294-562a-9e11-5568b74be6dd/scratchpad/"
g, lv, rf, ev = load()
EV = ["fomc", "fomc_eve", "fomc_next", "cpi", "nfp", "rel830", "opex", "quad", "roll_window", "month_end",
      "month_start", "pre_holiday", "post_holiday", "any_tier1"]
# descriptive: morning (09:30-12:29) range and open->12:29 abs move, discovery days
disc = (g.split == "discovery") & g.tradable
H = g.H[:, 930:1110]; L = g.L[:, 930:1110]; C = g.C[:, 930:1110]; O = g.O[:, 930]
rng = np.nanmax(H, 1) - np.nanmin(L, 1)
r30 = np.nanmax(g.H[:, 930:960], 1) - np.nanmin(g.L[:, 930:960], 1)
rows = []
for e in EV:
    m = ev[e].values & disc
    rows.append({"event": e, "days": m.sum(), "rng_med": np.nanmedian(rng[m]), "rng_med_other": np.nanmedian(rng[disc & ~m]),
                 "rng/atr": np.nanmedian(rng[m] / rf.atr20.values[m]), "other": np.nanmedian(rng[disc & ~m] / rf.atr20.values[disc & ~m]),
                 "or30/atr": np.nanmedian(r30[m] / rf.atr20.values[m]), "or30_other": np.nanmedian(r30[disc & ~m] / rf.atr20.values[disc & ~m])})
print(pd.DataFrame(rows).round(3).to_string())
STRATS = ["I2v1", "ORB30_1R", "ORB15_eod", "GAPF_all", "LONG0935_noTP", "LONG0935", "SHORT0935"]
nvar = 0
out = []
for s in STRATS:
    tr = attach(pd.read_pickle(SP + f"tr_{s}.pkl"), rf, ev)
    tr = tr[tr.split == "discovery"]
    tr["h"] = half(tr)
    for e in EV:
        nvar += 1
        a = tr[tr[e]]; b = tr[~tr[e]]
        hs = [round(a[a.h == h].pnl.mean(), 0) if (a.h == h).any() else np.nan for h in ["D1", "D2", "D3"]]
        hb = [round(b[b.h == h].pnl.mean(), 0) for h in ["D1", "D2", "D3"]]
        out.append({"strat": s, "event": e, "n_ev": len(a), "avg_ev": round(a.pnl.mean(), 0), "avg_other": round(b.pnl.mean(), 0),
                    "time%_ev": round(100 * (a.reason == "time").mean(), 0), "time%_oth": round(100 * (b.reason == "time").mean(), 0),
                    "ev_D123": hs, "oth_D123": hb,
                    "worse_all3": all((x < y) for x, y in zip(hs, hb) if x == x) and sum(x == x for x in hs) == 3})
O_ = pd.DataFrame(out)
print(O_.to_string())
print("tests:", nvar, "worse in all 3 halves:", O_.worse_all3.sum(), "(chance ~", round(nvar / 8, 1), ")")
