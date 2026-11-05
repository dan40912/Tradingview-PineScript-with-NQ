"""Engine audit: an independent trade simulator (no research/lib/sim.py) + independent re-derivation of the
signals of A3_C3, C1_ORB15_POC_A3X_TS and C2_TE_OUTSIDE from their written rules. Reconciles trade by trade,
then re-simulates EVERY candidate with the independent simulator, and quantifies stop gaps
(bar opening beyond the stop -> realistic fill at that open instead of the stop price).
Output: out_02_engine.txt (stdout) and out_02_gaps.csv"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_common as A
from lib import sim

g, lv = A.load("NQ")
es, lves = A.load("ES")
C = A.all_candidates()
RTH = 930


# ------------------------------------------------------------------ independent simulator
def mysim(G, sig, gap_fill=False):
    i, d = sig["i"], int(sig["dir"])
    X = int(sig.get("exit_slot", RTH + 179))
    ct = float(sig.get("contracts", 1.0))
    O, H, L, Cl = G.O[i], G.H[i], G.L[i], G.C[i]
    e = int(sig["entry_slot"])
    while e <= X and not np.isfinite(O[e]):
        e += 1
    if e > X:
        return None
    px = O[e]
    tpp, slp = sig.get("tp_pts"), sig.get("sl_pts")
    tp = px + d * tpp if tpp is not None and np.isfinite(tpp) else None
    sl = px - d * slp if slp is not None and np.isfinite(slp) else None
    seg = np.arange(e, X + 1)
    ok = np.isfinite(H[seg])
    seg = seg[ok]
    hs, ls = H[seg], L[seg]
    s_hit = np.where((ls <= sl) if d == 1 else (hs >= sl))[0] if sl is not None else np.array([], int)
    t_hit = np.where((hs >= tp) if d == 1 else (ls <= tp))[0] if tp is not None else np.array([], int)
    ks = s_hit[0] if len(s_hit) else 10 ** 9
    kt = t_hit[0] if len(t_hit) else 10 ** 9
    gapped = False
    if ks <= kt and ks < 10 ** 9:            # stop first (ties -> stop)
        xs = seg[ks]; xp = sl; why = "sl"
        o = O[xs]
        if xs > e and np.isfinite(o) and ((d == 1 and o < sl) or (d == -1 and o > sl)):
            gapped = True
            if gap_fill:
                xp = o
    elif kt < 10 ** 9:
        xs = seg[kt]; xp = tp; why = "tp"
    else:
        f = np.where(np.isfinite(Cl[e:X + 1]))[0]
        xs = e + f[-1]; xp = Cl[xs]; why = "time"
    pts = d * (xp - px)
    return dict(date=G.dates[i], i=i, dir=d, entry_slot=e, entry=px, exit_slot=xs, exit=xp, reason=why, pts=pts,
                pnl=(pts * G.point - G.cost) * ct, contracts=ct, gapped=gapped, split=G.split[i])


def mysim_all(G, sigs, gap_fill=False):
    return pd.DataFrame([r for r in (mysim(G, s, gap_fill) for s in sigs) if r is not None])


# ------------------------------------------------------------------ independent signals
def prev_rth(G, i):
    j = i - 1
    while j >= 0 and not G.has_rth[j]:
        j -= 1
    return j


def i2_universe():
    """days with prior same-contract VA, tradable, open outside VA; dir toward value"""
    out = []
    for r in lv.itertuples():
        if not (r.tradable and r.has_prev) or not np.isfinite(r.open_0930):
            continue
        if r.open_0930 > r.vah:
            out.append((r.i, -1, r))
        elif r.open_0930 < r.val:
            out.append((r.i, 1, r))
    return out


def blackout(ei):
    return 28 <= ei <= 32 or ei > 120


def fill_of(i, es_, last=RTH + 179):
    e = es_
    while e <= last and not np.isfinite(g.O[i, e]):
        e += 1
    return e, g.O[i, e]


def noprog(i, d, e, px, base=RTH + 179):
    last = e + 29
    if last >= base:
        return base
    seg = (g.H[i, e:last + 1] - px) if d == 1 else (px - g.L[i, e:last + 1])
    return last if np.nanmax(seg) < 20 else base


def a3x(i, d, ent, r):
    e, px = fill_of(i, ent)
    tp = max(d * (r.poc - px), 20.0)
    return dict(i=i, entry_slot=ent, dir=d, tp_pts=tp, sl_pts=60.0, exit_slot=noprog(i, d, e, px))


def my_a3c3():
    out = []
    for i, d, r in i2_universe():
        cnt = 0
        for idx in range(4, 120, 5):
            c = g.C[i, RTH + idx]
            if not np.isfinite(c):
                continue
            cnt = cnt + 1 if r.val <= c <= r.vah else 0
            if cnt == 2:
                if not blackout(idx + 1):
                    out.append(a3x(i, d, RTH + idx + 1, r))
                break
    return out


def my_ret20():
    """sum of the last 20 grid rows' same-contract RTH log returns (row j return uses closes of j and its
    previous RTH day; roll -> 0), known before day i."""
    n = len(g)
    rc = np.full(n, np.nan)
    for i in range(n):
        if g.has_rth[i]:
            c = g.C[i, RTH:RTH + 390]
            f = np.where(np.isfinite(c))[0]
            rc[i] = c[f[-1]]
    lr = np.zeros(n)
    for i in range(n):
        j = prev_rth(g, i)
        if j >= 0 and np.isfinite(rc[i]) and np.isfinite(rc[j]):
            lr[i] = np.log(rc[i] / rc[j]) if g.contract[i] == g.contract[j] else 0.0
    out = np.full(n, np.nan)
    for i in range(20, n):
        out[i] = 100 * lr[i - 20:i].sum()
    return out


def my_orb15_poc():
    r20 = my_ret20()
    out = []
    for i, d, r in i2_universe():
        h = np.nanmax(g.H[i, RTH:RTH + 15]); l = np.nanmin(g.L[i, RTH:RTH + 15])
        for idx in range(15, 120):
            c = g.C[i, RTH + idx]
            if np.isfinite(c) and ((d == 1 and c > h) or (d == -1 and c < l)):
                if not blackout(idx + 1):
                    s = a3x(i, d, RTH + idx + 1, r)
                    s["contracts"] = 0.5 if (np.isfinite(r20[i]) and np.sign(r20[i]) == -d) else 1.0
                    out.append(s)
                break
    return out


def my_te_outside():
    rng_ = np.array([np.nanmax(g.H[i, RTH:RTH + 390]) - np.nanmin(g.L[i, RTH:RTH + 390]) if g.has_rth[i] else np.nan
                     for i in range(len(g))])
    hist, adr = [], np.full(len(g), np.nan)
    for i in range(len(g)):
        if len(hist) >= 10:
            adr[i] = np.mean(hist[-10:])
        if g.has_rth[i]:
            hist.append(rng_[i])
    out = []
    for i, d0, r in i2_universe():
        if not np.isfinite(adr[i]):
            continue
        for idx in range(34, 325, 5):
            c = g.C[i, RTH + idx]
            if np.isfinite(c) and abs(c - r.open_0930) > 0.75 * adr[i]:
                sl = 0.5 * adr[i]
                ct = min(np.floor(1000 / (sl * 20) / 0.1 + 1e-9) * 0.1, 3.0)
                if ct >= 0.1:
                    out.append(dict(i=i, entry_slot=RTH + idx + 1, dir=int(np.sign(c - r.open_0930)), tp_pts=None,
                                    sl_pts=sl, exit_slot=RTH + 359, contracts=round(ct, 1)))
                break
    return out


def cmp_sigs(name, mine, theirs):
    k = lambda s: (s["i"], int(s["entry_slot"]), int(s["dir"]),
                   None if s.get("tp_pts") is None else round(s["tp_pts"], 4), round(s["sl_pts"], 4),
                   int(s["exit_slot"]), round(float(s.get("contracts", 1.0)), 4))
    a, b = set(map(k, mine)), set(map(k, theirs))
    print(f"[signals] {name}: mine={len(a)} candidate={len(b)} identical={len(a & b)} only_mine={len(a-b)} only_cand={len(b-a)}")
    for x in sorted(a - b)[:5]:
        print("    only mine:", x)
    for x in sorted(b - a)[:5]:
        print("    only cand:", x)


print("=" * 100)
print("1) Independent signal derivation vs candidate code")
cmp_sigs("A3_C3", my_a3c3(), C["A3/A3_C3_POC_SL60_NP30"](g, lv))
cmp_sigs("C1_ORB15_POC_A3X_TS", my_orb15_poc(), C["C1/C1_ORB15_POC_A3X_TS"](g, lv))
cmp_sigs("C2_TE_OUTSIDE", my_te_outside(), C["C2/C2_TE_OUTSIDE"](g, lv))

print("=" * 100)
print("2) Trade-by-trade reconciliation: independent sim (my signals) vs lib.sim (candidate signals)")
for name, mine in [("A3/A3_C3_POC_SL60_NP30", my_a3c3()), ("C1/C1_ORB15_POC_A3X_TS", my_orb15_poc()),
                   ("C2/C2_TE_OUTSIDE", my_te_outside())]:
    a = mysim_all(g, mine).set_index("date")
    b = sim.run_signals(g, C[name](g, lv)).set_index("date")
    j = a.join(b, lsuffix="_me", rsuffix="_lib", how="outer")
    diff = (j.pnl_me - j.pnl_lib).abs()
    print(f"  {name}: trades me={len(a)} lib={len(b)}  max|pnl diff|={diff.max():.6f}  "
          f"reason mismatches={(j.reason_me != j.reason_lib).sum()}  total me={a.pnl.sum():.0f} lib={b.pnl.sum():.0f}")

print("=" * 100)
print("3) All candidates: lib.sim vs independent sim on the candidates' own signals; stop-gap realism")
rows = []
for name, fn in C.items():
    G, L = (es, lves) if name in A.ES_ONLY else (g, lv)
    sigs = fn(G, L)
    lib = sim.run_signals(G, sigs)
    me = mysim_all(G, sigs)
    mg = mysim_all(G, sigs, gap_fill=True)
    d = np.abs(lib.pnl.values - me.pnl.values).max() if len(lib) == len(me) else np.nan
    nsl = int((me.reason == "sl").sum())
    ngap = int(me.gapped.sum())
    extra = float((mg.pnl - me.pnl).sum())
    same_bar = 0
    # how many TP exits happened on a bar that also touched the stop? (engine books stop -> count stop-first ties)
    for s, r in zip(sigs, me.to_dict("records")):
        pass
    rows.append(dict(cand=name, n=len(lib), max_abs_pnl_diff=d, sl_exits=nsl, gapped_stops=ngap,
                     gap_cost_total=round(extra, 0), gap_cost_per_trade=round(extra / max(len(me), 1), 2),
                     total=round(lib.pnl.sum(), 0)))
df = pd.DataFrame(rows)
with pd.option_context("display.width", 200):
    print(df.to_string(index=False))
df.to_csv(os.path.join(A.HERE, "out_02_gaps.csv"), index=False)

print("=" * 100)
print("4) Same-minute TP/SL ambiguity: trades whose exit bar touched both (engine books the stop)")
for name in ["A3/A3_C3_POC_SL60_NP30", "C1/C1_ORB15_POC_A3X_TS", "C2/C2_TE_OUTSIDE", "C1/C1_I2_A3X_TS",
             "A1/OOV_ORB30_1R", "C1/C1_ORB15_TP110_TS"]:
    sigs = C[name](g, lv)
    tr = sim.run_signals(g, sigs)
    amb = 0
    for s, r in zip(sigs, tr.to_dict("records")):
        if r["reason"] in ("sl", "stop") and s.get("tp_pts") is not None:
            tp = r["entry"] + r["dir"] * s["tp_pts"]
            x = r["exit_slot"]
            if (r["dir"] == 1 and g.H[r["i"], x] >= tp) or (r["dir"] == -1 and g.L[r["i"], x] <= tp):
                amb += 1
    print(f"  {name}: ambiguous bars booked as stop = {amb} of {len(tr)}")

print("=" * 100)
print("5) Entry-bar checks: entry price equals open of entry minute; cost scales with contracts")
for name in ["A3/A3_C3_POC_SL60_NP30", "C1/C1_ORB15_POC_A3X_TS", "C2/C2_TE_OUTSIDE"]:
    sigs = C[name](g, lv)
    tr = A.run(g, sigs)
    ok_px = np.allclose(tr.entry.values, g.O[tr.i.values, tr.entry_slot.values])
    recon = (tr.pts * 20 - 48) * tr.contracts
    print(f"  {name}: entry==O[entry_slot] {ok_px};  pnl==(pts*20-48)*contracts {np.allclose(recon, tr.pnl)};"
          f"  contracts used: {sorted(tr.contracts.unique())[:8]}")
