"""E8: position sizing / risk overlays on I2 v1 entries, DISCOVERY ONLY.
Sizing uses only pre-entry info (ATR14 of previous full RTH sessions, prior-day VA width, overnight range).
Contracts rounded to 0.1 (MNQ granularity), clipped to [0.3, 2.0]. Reference level = median of that
measure over discovery I2 days (so average size ~1 contract)."""
import numpy as np, pandas as pd, itertools
from a3lib import load, i2_entries, with_exit, run, stats, noprog_exit_slot

g, lv = load("NQ")
sigs = [s for s in i2_entries(g, lv) if g.split[s["i"]] == "discovery"]
lvi = lv.set_index("i")
K = ("n", "win%", "avg", "total", "maxdd", "ret/dd", "sharpe_ann")
rows = []


def poc_tp(floor):
    return lambda g_, r, s, px: max(s["dir"] * (r.poc - px), floor)


NP = lambda N, X: (lambda g_, r, s, px: noprog_exit_slot(g_, s, px, N, X))
STRUCT = {"base55/145": dict(tp=55, sl=145), "C1 100/60": dict(tp=100, sl=60),
          "C2 100/60+NP30/20": dict(tp=100, sl=60, exit_slot=NP(30, 20)),
          "C3 POCf20/60+NP30/20": dict(tp=poc_tp(20), sl=60, exit_slot=NP(30, 20))}
MEAS = {m: float(np.nanmedian(lvi.loc[[s["i"] for s in sigs], m])) for m in ["atr14", "va_width", "on_range"]}
print("reference medians", MEAS)


def size_fn(m, power=1.0):
    ref = MEAS[m]
    return lambda g_, r, s, px: float(np.clip(np.round((ref / r[m]) ** power, 1), 0.3, 2.0))


def overlay_consec(tr, N, M):
    """after N consecutive losing trades, skip the next M signals (skipped pnl=0). Uses only past results."""
    tr = tr.sort_values("date").copy()
    keep, streak, skip = [], 0, 0
    for p in tr.pnl.values:
        if skip > 0:
            keep.append(False); skip -= 1
            continue
        keep.append(True)
        streak = streak + 1 if p < 0 else 0
        if streak >= N:
            skip, streak = M, 0
    return tr[np.array(keep)]


def overlay_ddhalf(tr, dd_usd):
    """if running drawdown from equity peak exceeds dd_usd, trade half size until a new equity high."""
    tr = tr.sort_values("date").copy()
    eq = peak = 0.0
    mult = []
    for p in tr.pnl.values:
        m = 0.5 if peak - eq > dd_usd else 1.0
        mult.append(m)
        eq += p * m
        peak = max(peak, eq)
    tr["pnl"] = tr.pnl.values * np.array(mult)
    tr["contracts"] = np.array(mult)
    return tr


def rec(struct, rule, tr, avgc=1.0):
    st = stats(tr, g, "discovery")
    rows.append({"struct": struct, "rule": rule, "avg_contracts": round(avgc, 2), **{k: st.get(k) for k in K}})


for sn, kw in STRUCT.items():
    base_sigs = with_exit(g, lv, sigs, **kw)
    tr0 = run(g, base_sigs)
    rec(sn, "1 contract", tr0)
    for m, pw in itertools.product(MEAS, [0.5, 1.0]):
        sz = with_exit(g, lv, sigs, contracts=size_fn(m, pw), **kw)
        tr = run(g, sz)
        avgc = float(np.mean([s["contracts"] for s in sz]))
        rec(sn, f"size ~ (ref/{m})^{pw}", tr, avgc)
    for N, M in [(2, 1), (2, 2), (3, 1), (3, 3)]:
        rec(sn, f"skip {M} after {N} losses", overlay_consec(tr0, N, M))
    for dd in [3000, 5000]:
        t = overlay_ddhalf(tr0, dd)
        rec(sn, f"half size while DD>{dd}", t, float(t.contracts.mean()))
df = pd.DataFrame(rows)
df.to_csv("out_sizing.csv", index=False)
pd.set_option("display.width", 220)
for s, grp in df.groupby("struct", sort=False):
    print(f"\n== {s}")
    print(grp.drop(columns="struct").to_string(index=False))
print("variants:", len(df))
