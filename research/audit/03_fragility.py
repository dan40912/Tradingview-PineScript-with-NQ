"""Fragility tests on 2024-2025 only (discovery + validation; holdout never loaded).
Stresses: +1 minute entry delay (POC target / no-progress exit recomputed from the delayed fill), +$20 / +$40 cost
per contract, target must trade THROUGH by 1 / 2 ticks, stop slippage 2 pts (=$40/ct, stop exits only),
combined worst case (delay + $40 + 2 ticks), removal of top-5 trades, monthly P&L concentration.
C3 portfolios are evaluated as real two-leg portfolios (NQ leg + ES leg), not only the NQ leg the harness sees.
Output: out_03_fragility.csv, out_03_months.csv, stdout."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_common as A
from lib import sim, metrics

g, lv = A.load("NQ")
es, lves = A.load("ES")
C = A.all_candidates()
TICK = 0.25
es_fn = C["C3/C3_ES_I2_40_15"]


def legs(name):
    if name == "C3/C3_ES_I2_40_15":
        return [("ES", es, lves, es_fn(es, lves))]
    if name in ("C3/C3_PORT_A3C3_ES", "C3/C3_PORT_I2v1_ES"):
        return [("NQ", g, lv, C[name](g, lv)), ("ES", es, lves, es_fn(es, lves, contracts=0.8))]
    return [("NQ", g, lv, C[name](g, lv))]


def fill(G, i, e, last):
    while e <= last and not np.isfinite(G.O[i, e]):
        e += 1
    return e, (G.O[i, e] if e <= last else np.nan)


def delayed(name, sym, G, L, sigs, k=1):
    poc = dict(zip(L.i, L.poc))
    out = []
    for s in sigs:
        s = dict(s, entry_slot=s["entry_slot"] + k)
        if name in A.NOPROG and sym == "NQ":
            base = 930 + 179
            e, px = fill(G, s["i"], s["entry_slot"], base)
            if not np.isfinite(px):
                continue
            d = s["dir"]
            s["tp_pts"] = float(max(d * (poc[s["i"]] - px), 20.0))
            last = e + 29
            if last < base:
                seg = (G.H[s["i"], e:last + 1] - px) if d == 1 else (px - G.L[s["i"], e:last + 1])
                s["exit_slot"] = last if np.nanmax(seg) < 20 else base
            else:
                s["exit_slot"] = base
        out.append(s)
    return out


def through(sigs, ticks):
    return [dict(s, tp_pts=(s["tp_pts"] + ticks * TICK if s.get("tp_pts") is not None else None)) for s in sigs]


def evaluate(name, delay=0, cost=0.0, ticks=0, stop_slip=0.0):
    out = []
    for sym, G, L, sigs in legs(name):
        s2 = delayed(name, sym, G, L, sigs, delay) if delay else sigs
        s2 = through(s2, ticks) if ticks else s2
        tr = A.run(G, s2)
        if len(tr) == 0:
            continue
        if ticks:
            m = tr.reason == "tp"
            tr.loc[m, "pnl"] -= ticks * TICK * G.point * tr.loc[m, "contracts"]
        if stop_slip:
            m = tr.reason.isin(["sl", "stop"])
            tr.loc[m, "pnl"] -= stop_slip * G.point * tr.loc[m, "contracts"]
        tr["pnl"] -= cost * tr.contracts
        tr["sym"] = sym
        out.append(tr)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def stats(tr):
    r = {}
    for sp in ["discovery", "validation", None]:
        s = A.summ(g, tr, sp)
        tag = {"discovery": "D", "validation": "V", None: "ALL"}[sp]
        r[f"{tag}_avg"] = s.get("avg", np.nan)
        r[f"{tag}_sh"] = s.get("sharpe_ann", np.nan)
        r[f"{tag}_dd"] = s.get("maxdd", np.nan)
    return r


STRESS = {"base": {}, "delay1": {"delay": 1}, "cost20": {"cost": 20}, "cost40": {"cost": 40},
          "thru1": {"ticks": 1}, "thru2": {"ticks": 2}, "slip2pt": {"stop_slip": 2.0},
          "worst": {"delay": 1, "cost": 40, "ticks": 2}}
rows, mrows = [], []
for name in C:
    base = None
    rec = {"cand": name}
    for sk, kw in STRESS.items():
        tr = evaluate(name, **kw)
        if sk == "base":
            base = tr
        st = stats(tr)
        for k, v in st.items():
            rec[f"{sk}:{k}"] = v
    # concentration (2024-25 all, per trade-day aggregated for portfolios)
    daily = base.groupby("date").pnl.sum().sort_values(ascending=False)
    p = base.pnl.values
    top5 = np.sort(p)[::-1][:5]
    rest = np.sort(p)[::-1][5:]
    rec["n"] = len(base)
    rec["total"] = round(p.sum(), 0)
    rec["top5_share"] = round(top5.sum() / p.sum(), 2) if p.sum() > 0 else np.nan
    rec["ex_top5_avg"] = round(rest.mean(), 1)
    rec["ex_top5_t"] = round(rest.mean() / (rest.std(ddof=1) / np.sqrt(len(rest))), 2)
    vb = base[base.split == "validation"].pnl.sort_values(ascending=False).values
    rec["V_ex_top1_avg"] = round(vb[1:].mean(), 1) if len(vb) > 1 else np.nan
    rec["V_ex_top3_avg"] = round(vb[3:].mean(), 1) if len(vb) > 3 else np.nan
    mon = base.assign(m=base.date.str[:7]).groupby("m").pnl.sum()
    rec["months_pos"] = f"{int((mon > 0).sum())}/{len(mon)}"
    rec["best_month"] = mon.idxmax()
    rec["best_month_share"] = round(mon.max() / mon.sum(), 2) if mon.sum() > 0 else np.nan
    rec["top3_month_share"] = round(mon.sort_values(ascending=False)[:3].sum() / mon.sum(), 2) if mon.sum() > 0 else np.nan
    rec["ex_best_month_total"] = round(mon.sum() - mon.max(), 0)
    for m, v in mon.items():
        mrows.append({"cand": name, "month": m, "pnl": round(v, 0)})
    # robustness score: stresses (excluding base) with BOTH split averages > 0
    ok = [sk for sk in STRESS if sk != "base" and rec[f"{sk}:D_avg"] > 0 and rec[f"{sk}:V_avg"] > 0]
    rec["stress_pass"] = f"{len(ok)}/{len(STRESS)-1}"
    rows.append(rec)
    print(name, "done", flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(A.HERE, "out_03_fragility.csv"), index=False)
pd.DataFrame(mrows).to_csv(os.path.join(A.HERE, "out_03_months.csv"), index=False)
cols = ["cand", "n", "base:D_avg", "base:V_avg", "base:ALL_sh", "delay1:D_avg", "delay1:V_avg", "cost40:V_avg",
        "thru2:D_avg", "thru2:V_avg", "slip2pt:V_avg", "worst:D_avg", "worst:V_avg", "worst:ALL_sh", "stress_pass"]
cols2 = ["cand", "total", "top5_share", "ex_top5_avg", "ex_top5_t", "V_ex_top1_avg", "V_ex_top3_avg", "months_pos",
         "best_month", "best_month_share", "top3_month_share", "ex_best_month_total"]
with pd.option_context("display.width", 250, "display.max_columns", 50):
    print(df[cols].to_string(index=False))
    print(df[cols2].to_string(index=False))
