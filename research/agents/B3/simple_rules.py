"""B3: transparent one-feature rules derived from the IC study (intraday momentum family).

Family (36 variants, fixed BEFORE looking at the P&L):
  decision time T in {09:45, 10:00, 10:30}
  direction = sign of one feature in {s_vwap (price - RTH VWAP), s_since_open (price - 09:30 open),
                                      x_es_rel_60 (NQ minus ES return over the last 60 min)}
  exit in {time 60m, time 120m, +-40 pts first-hit (15:59 cap), +-80 pts first-hit}
Reported per period: 2024, 2025H1 (discovery), 2025H2 (validation).
Plus a neighbour grid over decision time (09:35..10:45 step 5) for the variants that pass discovery.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..")); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
from lib import data, features, sim, metrics
import features_ml as fm

OUT = os.path.join(HERE, "results")
EOD = sim.slot_of("15:59")
FEATS = ["s_vwap", "s_since_open", "x_es_rel_60"]
EXITS = {"t60": ("time", 60), "t120": ("time", 120), "x40": ("hit", 40), "x80": ("hit", 80)}


def per(date):
    return "2024" if date < "2025-01-01" else ("2025H1" if date <= "2025-06-30" else "2025H2")


def signals(ev, X, feat, exit_key, sign=1):
    kind, v = EXITS[exit_key]
    out = []
    for (k, r), f in zip(ev.iterrows(), X[feat].values):
        if not np.isfinite(f) or f == 0:
            continue
        s = {"i": int(r.i), "entry_slot": int(r.entry_slot), "dir": int(np.sign(f)) * sign}
        if kind == "time":
            s.update(tp_pts=None, sl_pts=None, exit_slot=int(r.entry_slot) + v - 1)
        else:
            s.update(tp_pts=v, sl_pts=v, exit_slot=EOD)
        out.append(s)
    return out


def table(g, tr):
    tr = tr.copy(); tr["per"] = tr.date.map(per)
    nd = {p: int(sum(1 for d, t in zip(g.dates, g.tradable) if t and per(d) == p)) for p in ["2024", "2025H1", "2025H2"]}
    res = {}
    for p in ["2024", "2025H1", "2025H2"]:
        res[p] = metrics.summary(tr[tr.per == p], n_days=nd[p])
    nd_disc = nd["2024"] + nd["2025H1"]
    res["disc"] = metrics.summary(tr[tr.per != "2025H2"], n_days=nd_disc)
    return res


def main():
    g = data.load_grid("NQ"); lv = features.daily_levels(g); es = data.load_grid("ES")
    ctx = fm.Context(g, lv, es=es)
    rows = []
    times = ["09:45", "10:00", "10:30"]
    ev = fm.fixed_time_events(g, lv, times)
    X = fm.build(ctx, ev)
    for t in times:
        m = (ev.t == t).values
        for f in FEATS:
            for ek in EXITS:
                tr = sim.run_signals(g, signals(ev[m], X[m], f, ek))
                r = table(g, tr)
                rows.append({"time": t, "feat": f, "exit": ek,
                             **{f"{p}_{k}": r[p].get(k) for p in ["2024", "2025H1", "disc", "2025H2"] for k in ["n", "avg", "total", "maxdd", "sharpe_ann", "win%"]}})
    R = pd.DataFrame(rows)
    R["disc_pass"] = (R["disc_avg"] >= 50) & (R["2024_avg"] > 0) & (R["2025H1_avg"] > 0) & (R["disc_n"] >= 40)
    R.to_csv(os.path.join(OUT, "simple_rules.csv"), index=False)
    pd.set_option("display.width", 250)
    print(R[["time", "feat", "exit", "2024_n", "2024_avg", "2025H1_avg", "disc_avg", "disc_maxdd", "disc_sharpe_ann", "disc_pass", "2025H2_n", "2025H2_avg", "2025H2_total", "2025H2_maxdd"]].to_string())
    # neighbour grid for discovery passers
    P = R[R.disc_pass]
    grid_t = [sim.hhmm_of(s) for s in range(sim.slot_of("09:35"), sim.slot_of("10:46"), 5)]
    evg = fm.fixed_time_events(g, lv, grid_t)
    Xg = fm.build(ctx, evg)
    nb = []
    for r in P.itertuples():
        for t in grid_t:
            m = (evg.t == t).values
            tr = sim.run_signals(g, signals(evg[m], Xg[m], r.feat, r.exit))
            rr = table(g, tr)
            nb.append({"base": f"{r.time}|{r.feat}|{r.exit}", "time": t, "2024": rr["2024"].get("avg"), "2025H1": rr["2025H1"].get("avg"),
                       "disc": rr["disc"].get("avg"), "2025H2": rr["2025H2"].get("avg")})
    NB = pd.DataFrame(nb); NB.to_csv(os.path.join(OUT, "simple_rules_neighbours.csv"), index=False)
    if len(NB):
        print(NB.pivot_table(index="time", columns="base", values="disc").round(0).to_string())
        print(NB.pivot_table(index="time", columns="base", values="2025H2").round(0).to_string())


if __name__ == "__main__":
    main()
