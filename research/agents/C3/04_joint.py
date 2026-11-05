"""C3 step 4 (DISCOVERY for choosing; validation printed only for the rows frozen afterwards is done in 06).
NQ/ES joint value-area logic, all causal:
 (a) state split: NQ I2 by ES state at 09:30 (ES outside same side / ES inside)  [H3-like, descriptive]
 (b) dual confirmation: trade X only after BOTH NQ and ES have re-entered their own VA (2 consecutive
     5-min closes inside); entry = minute after the later confirmation; same skip window/cutoff as I2.
 (c) relative value on 'both outside same side' days: trade only the one further outside (|open-edge|/ATR14)
     or only the laggard (closer to its edge), with I2 entry/exit.
 (d) NQ-ES pair reversion at the dual confirmation: short the more-stretched / long the less-stretched,
     dollar-neutral in micros, exit 12:29.
"""
import sys, os, numpy as np, pandas as pd
from c3lib import (data, features, levels, add_atr, i2_exit_sigs, run, stats, tradable_days, half_avgs,
                   RTH, EXIT_1229)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "A3"))
from candidates import c3_poc_sl60_np30

SPLIT = sys.argv[1] if len(sys.argv) > 1 else "discovery"
gN = data.load_grid("NQ"); lvN = add_atr(gN, features.daily_levels(gN))
gE = data.load_grid("ES"); lvE = add_atr(gE, levels(gE, 5.0))
DN = tradable_days(gN, SPLIT); DE = tradable_days(gE, SPLIT)


def confirm_idx(g, lv):
    """date -> (dir, ei) first I2 re-entry confirmation (ei = entry idx), no skip/cutoff filter."""
    out = {}
    for r in lv.itertuples():
        if not r.has_prev or r.i2_dir == 0 or np.isnan(r.open_0930):
            continue
        C = g.C[r.i]; cnt = 0
        for idx in range(4, 120, 5):
            c = C[RTH + idx]
            if np.isnan(c):
                continue
            cnt = cnt + 1 if (r.val <= c <= r.vah) else 0
            if cnt >= 2:
                out[r.date] = (int(r.i2_dir), idx + 1); break
    return out


cN, cE = confirm_idx(gN, lvN), confirm_idx(gE, lvE)
ok_ei = lambda ei: not ((28 <= ei <= 32) or ei > 120)
rows = []


def rep(name, g, sigs, days, exitkw):
    tr = run(g, i2_exit_sigs(g, lv_of[g.symbol], sigs=sigs, **exitkw)) if exitkw is not None else run(g, sigs)
    td = tr[tr.date.isin(days)] if len(tr) else tr
    rows.append({"variant": name, **stats(td, days), **(half_avgs(td) if SPLIT == "discovery" else {})})
    return td


lv_of = {"NQ": lvN, "ES": lvE}
EX = {"NQ": {"v1": dict(tp=55, sl=145), "A3C3": None}, "ES": {"22_58": dict(tp=22, sl=58), "40_15": dict(tp=40, sl=15)}}


def i2_sigs(g, own, other=None, mode="all", relkey=None):
    lv = lv_of[g.symbol].set_index("date"); olv = lv_of["ES" if g.symbol == "NQ" else "NQ"].set_index("date")
    out = []
    for d, (dr, ei) in own.items():
        i = g.dindex[d]
        if mode == "all":
            e = ei
        elif mode == "dual":
            if other is None or d not in other or other[d][0] != dr:
                continue
            e = max(ei, other[d][1])
        elif mode in ("other_in", "other_out"):
            od = olv.loc[d, "i2_dir"] if d in olv.index else 0
            if (mode == "other_out") != (od == dr):
                continue
            e = ei
        elif mode in ("further", "laggard"):
            od = olv.loc[d, "i2_dir"] if d in olv.index else 0
            if od != dr:
                continue
            def stretch(L, dd):
                r = L.loc[dd]
                return ((r.open_0930 - r.vah) if dr == -1 else (r.val - r.open_0930)) / r.atr14
            s_own, s_oth = stretch(lv, d), stretch(olv, d)
            if not (np.isfinite(s_own) and np.isfinite(s_oth)):
                continue
            if (mode == "further") != (s_own > s_oth):
                continue
            e = ei
        if not ok_ei(e):
            continue
        out.append({"i": i, "entry_slot": RTH + e, "dir": dr, "tag": mode})
    return out


def a3c3_from(g, sigs):
    """A3 C3 exit (POC floor 20 / SL 60 / NP 30-20) for arbitrary NQ entries."""
    return i2_exit_sigs(g, lvN, sigs=sigs, poc_floor=20.0, sl=60.0, np_n=30, np_x=20.0)


for sym, g, own, oth, D in [("NQ", gN, cN, cE, DN), ("ES", gE, cE, cN, DE)]:
    for mode in ["all", "dual", "other_out", "other_in", "further", "laggard"]:
        sg = i2_sigs(g, own, oth, mode)
        for ex, kw in EX[sym].items():
            if kw is None:
                rep(f"{sym} {mode} x{ex}", g, a3c3_from(g, sg), D, None)
            else:
                rep(f"{sym} {mode} x{ex}", g, sg, D, kw)

# (d) pair: at the dual confirmation, short the more stretched (open distance beyond own edge / ATR) and long
# the other (direction flipped), dollar-neutral: NQ 0.5 (=5 MNQ) vs ES notional-matched in 0.1 steps; exit 12:29
D = sorted(set(DN) & set(DE))
lvn = lvN.set_index("date"); lve = lvE.set_index("date")
for variant in ["fade_stretched_long_laggard", "fade_laggard_long_stretched"]:
    pairs = []
    for d in D:
        if d not in cN or d not in cE or cN[d][0] != cE[d][0]:
            continue
        dr = cN[d][0]; e = max(cN[d][1], cE[d][1])
        if not ok_ei(e):
            continue
        sN = ((lvn.loc[d, "open_0930"] - lvn.loc[d, "vah"]) if dr == -1 else (lvn.loc[d, "val"] - lvn.loc[d, "open_0930"])) / lvn.loc[d, "atr14"]
        sE = ((lve.loc[d, "open_0930"] - lve.loc[d, "vah"]) if dr == -1 else (lve.loc[d, "val"] - lve.loc[d, "open_0930"])) / lve.loc[d, "atr14"]
        if not (np.isfinite(sN) and np.isfinite(sE)):
            continue
        nq_more = sN > sE
        if variant.startswith("fade_laggard"):
            nq_more = not nq_more
        # the 'more' leg is traded in the I2 (reversion) direction, the other leg against it
        dN_, dE_ = (dr, -dr) if nq_more else (-dr, dr)
        iN, iE = gN.dindex[d], gE.dindex[d]
        pxN = gN.O[iN, RTH + e]; pxE = gE.O[iE, RTH + e]
        if not (np.isfinite(pxN) and np.isfinite(pxE)):
            continue
        qN = 0.5; qE = round(qN * pxN * 20 / (pxE * 50), 1)
        a = run(gN, [{"i": iN, "entry_slot": RTH + e, "dir": dN_, "tp_pts": None, "sl_pts": None, "exit_slot": EXIT_1229, "contracts": qN}])
        b = run(gE, [{"i": iE, "entry_slot": RTH + e, "dir": dE_, "tp_pts": None, "sl_pts": None, "exit_slot": EXIT_1229, "contracts": qE}])
        if len(a) and len(b):
            pairs.append({"date": d, "pnl": a.pnl.iloc[0] + b.pnl.iloc[0], "split": SPLIT})
    td = pd.DataFrame(pairs)
    rows.append({"variant": f"PAIR {variant}", **stats(td, D), **(half_avgs(td) if SPLIT == "discovery" else {})})

df = pd.DataFrame(rows)
df.to_csv(f"out_04_joint_{SPLIT}.csv", index=False)
pd.set_option("display.width", 250)
print(SPLIT, "variants:", len(df)); print(df.to_string(index=False))
