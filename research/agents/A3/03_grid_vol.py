"""E3/E4: volatility-scaled TP/SL (units of prior-day VA width, ATR14 of RTH ranges, overnight range)
and structural targets (POC, opposite VA edge, RTH VWAP at entry). DISCOVERY ONLY."""
import numpy as np, pandas as pd, itertools
from a3lib import load, i2_entries, with_exit, run, stats, RTH, EXIT_1229

g, lv = load("NQ")
sigs = [s for s in i2_entries(g, lv) if g.split[s["i"]] == "discovery"]
K = ("n", "win%", "avg", "total", "maxdd", "ret/dd", "sharpe_ann")
nvar = 0
pd.set_option("display.width", 250)


def grid(unit, A, B, label):
    global nvar
    rows = []
    for a, b in itertools.product(A, B):
        tp = (lambda g_, r, s, px, a=a: a * r[unit]) if np.isfinite(a) else np.inf
        sl = (lambda g_, r, s, px, b=b: b * r[unit]) if np.isfinite(b) else np.inf
        tr = run(g, with_exit(g, lv, sigs, tp=tp, sl=sl))
        st = stats(tr, g, "discovery")
        rows.append({"a": a, "b": b, **{k: st[k] for k in K}})
        nvar += 1
    df = pd.DataFrame(rows)
    df.to_csv(f"out_grid_{label}.csv", index=False)
    for k in ["avg", "sharpe_ann", "maxdd", "ret/dd"]:
        print(f"\n=== {label}: {k} (rows TP mult, cols SL mult) ===")
        print(df.pivot(index="a", columns="b", values=k))
    return df


print("median va_width / atr14 / on_range in discovery entries:",
      lv.set_index("i").loc[[s["i"] for s in sigs], ["va_width", "atr14", "on_range"]].median().round(0).to_dict())
grid("va_width", [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, np.inf], [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, np.inf], "vaw")
grid("atr14", [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6], [0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8], "atr")
grid("on_range", [0.3, 0.5, 0.75, 1.0, 1.25, 1.5], [0.5, 0.75, 1.0, 1.25, 1.5, 2.0], "onr")


# structural targets ---------------------------------------------------------
def vwap_at(g_, i, e):
    h, l, c, v = g_.H[i, RTH:e], g_.L[i, RTH:e], g_.C[i, RTH:e], g_.V[i, RTH:e]
    tp_ = (h + l + c) / 3
    m = np.isfinite(tp_) & np.isfinite(v)
    return np.sum(tp_[m] * v[m]) / max(np.sum(v[m]), 1)


def target(kind, floor=None, cap=None):
    def f(g_, r, s, px):
        d = s["dir"]
        if kind == "poc":
            x = d * (r.poc - px)
        elif kind == "opp":
            x = (r.vah - px) if d == 1 else (px - r.val)
        elif kind == "vwap":
            x = d * (vwap_at(g_, s["i"], s["entry_slot"]) - px)
        elif kind == "opp_beyond":   # opposite edge + 0.25 VA width
            x = ((r.vah - px) if d == 1 else (px - r.val)) + 0.25 * r.va_width
        if floor is not None:
            x = max(x, floor)
        if cap is not None:
            x = min(x, cap)
        return x if x > 1 else np.nan   # target already passed -> treat as 'skip' (counted)
    return f


rows = []
for kind, floor, cap in [("poc", None, None), ("poc", 20, None), ("poc", 55, None), ("opp", None, None), ("opp", 55, None),
                         ("opp", 85, None), ("opp", None, 100), ("opp", 55, 150), ("vwap", None, None), ("vwap", 55, None),
                         ("opp_beyond", None, None)]:
    for sl in [60, 100, 145, "1.0vaw", np.inf]:
        slv = (lambda g_, r, s, px: 1.0 * r.va_width) if sl == "1.0vaw" else sl
        sg = with_exit(g, lv, sigs, tp=target(kind, floor, cap), sl=slv)
        tr = run(g, sg)
        st = stats(tr, g, "discovery")
        rows.append({"target": kind, "floor": floor, "cap": cap, "sl": sl, **{k: st[k] for k in K}})
        nvar += 1
ds = pd.DataFrame(rows)
ds.to_csv("out_struct_targets.csv", index=False)
print("\n=== structural targets (trades with target already passed are skipped) ===")
print(ds.to_string())
print("variants:", nvar)
