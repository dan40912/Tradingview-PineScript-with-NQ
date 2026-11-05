"""Look-ahead audit by data corruption (all candidates, NQ + ES legs).

Scenario (day i, cutoff slot S, mode):
  mode A "fill known":   day i  H,L,C,V at slots >= S and O at slots >= S+1 replaced by garbage
                         (so a decision for an entry at the OPEN of S may use bars <= S-1 and O[S] = the fill).
  mode B "open unknown": day i  O,H,L,C,V at slots >= S replaced by garbage.
In both modes EVERY later day (> i) is fully replaced by garbage, for NQ and ES alike (catches cross-day leaks:
ADR/ATR/ret20/percentiles/IC windows etc.). lv = features.daily_levels() is recomputed on the corrupted grid.

Requirements checked (per candidate, signals of day i only):
  A: every signal with entry_slot <= S is unchanged in (entry_slot, dir, contracts, tp, sl), and no new signal with
     entry_slot <= S appears.  exit_slot must also be unchanged, except for 'no-progress' candidates when S <= e+29
     (their exit decision legitimately uses bars e..e+29; with S >= e+30 it must be identical).
  B: signals with entry_slot < S fully unchanged; signals with entry_slot == S unchanged in (entry_slot, dir);
     a contracts/tp/sl change at entry_slot == S is reported as 'uses fill price' (legit for brackets, flagged for sizing).
Output: research/audit/out_01_lookahead.csv (+ summary printed).
"""
import os, sys, copy, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_common as A
from lib import data, features

rng = np.random.default_rng(20261105)
g0, lv0 = A.load("NQ")
es0, lves0 = A.load("ES")
C = A.all_candidates()
_orig_load = data.load_grid
_CUR = {"ES": es0}


def _patched(symbol="NQ", include_holdout=False, drop_exclude=False):
    assert not include_holdout, "holdout must stay locked"
    if symbol == "ES":
        return _CUR["ES"]
    return _orig_load(symbol, include_holdout=False, drop_exclude=drop_exclude)


data.load_grid = _patched   # B3 loads ES inside its candidate -> give it the corrupted ES grid


def garbage_like(row_o, row_c, start, seed_px):
    """random-walk garbage with ~0.5% per-minute moves, keeps NaN pattern."""
    n = len(row_o) - start
    if n <= 0:
        return None
    steps = rng.normal(0, 0.005, n) * seed_px
    c = seed_px + np.cumsum(steps)
    o = np.r_[seed_px, c[:-1]] + rng.normal(0, 0.002, n) * seed_px
    h = np.maximum(o, c) + np.abs(rng.normal(0, 0.003, n)) * seed_px
    l = np.minimum(o, c) - np.abs(rng.normal(0, 0.003, n)) * seed_px
    v = rng.lognormal(6, 1.5, n)
    return o, h, l, c, v


def corrupt(g, i, S, mode, date_map=None):
    """Return a corrupted shallow copy of grid g. For ES, i is the index of the same date (or None)."""
    G = copy.copy(g)
    O, H, L, Cc, V = g.O.copy(), g.H.copy(), g.L.copy(), g.C.copy(), g.V.copy()
    if i is not None:
        fin = np.where(np.isfinite(g.C[i, :S]))[0]
        px = g.C[i, fin[-1]] if len(fin) else np.nanmean(g.C[i])
        gb = garbage_like(O[i], Cc[i], S, px)
        if gb is not None:
            o, h, l, c, v = gb
            nanm = ~np.isfinite(g.C[i, S:])
            for arr, val in ((H, h), (L, l), (Cc, c), (V, v)):
                seg = val.copy(); seg[nanm] = np.nan; arr[i, S:] = seg
            so = 1 if mode == "A" else 0
            seg = o.copy(); seg[nanm] = np.nan
            O[i, S + so:] = seg[so:]
        later = range(i + 1, len(g))
    else:
        later = []
    for j in later:
        fin = np.isfinite(g.C[j])
        if not fin.any():
            continue
        px = np.nanmean(g.C[j]) * (1 + rng.normal(0, 0.05))
        o, h, l, c, v = garbage_like(O[j], Cc[j], 0, px)
        nanm = ~fin
        for arr, val in ((O, o), (H, h), (L, l), (Cc, c), (V, v)):
            seg = val.copy(); seg[nanm] = np.nan; arr[j] = seg
    G.O, G.H, G.L, G.C, G.V = O, H, L, Cc, V
    return G


def key(s):
    f = lambda x: None if x is None else round(float(x), 6)
    return (int(s["entry_slot"]), int(s["dir"]), f(s.get("contracts", 1.0)), f(s.get("tp_pts")), f(s.get("sl_pts")))


def sigs_by_day(sigs):
    out = {}
    for s in sigs:
        out.setdefault(s["i"], []).append(s)
    return out


# baseline signals
BASE = {}
for name, fn in C.items():
    if name in A.ES_ONLY:
        BASE[name] = sigs_by_day(fn(es0, lves0))
    else:
        BASE[name] = sigs_by_day(fn(g0, lv0))
multi = {k: sum(len(v) > 1 for v in d.values()) for k, d in BASE.items()}
print("days with >1 signal per candidate:", {k: v for k, v in multi.items() if v})

# sample days: signal days of each candidate (up to 6 per candidate) + random days
days = set()
for name, d in BASE.items():
    ks = sorted(d)
    if name in A.ES_ONLY:   # map ES day index -> NQ index by date
        ks = [g0.dindex[es0.dates[k]] for k in ks if es0.dates[k] in g0.dindex]
    for k in rng.choice(ks, size=min(6, len(ks)), replace=False):
        days.add(int(k))
days |= set(int(x) for x in rng.choice(np.where(g0.tradable)[0], 15, replace=False))
days = sorted(days)
print("sample days:", len(days))

rows = []
t0 = time.time()
for n_i, i in enumerate(days):
    date = g0.dates[i]
    ies = es0.dindex.get(date)
    # cutoffs: every candidate's entry slot that day (+ e+30 for no-progress), plus 2 random RTH cutoffs
    Ss = {}
    for name, d in BASE.items():
        ii = ies if name in A.ES_ONLY else i
        for s in d.get(ii, []):
            e = int(s["entry_slot"])
            Ss.setdefault(e, set()).update({"A", "B"})
            if name in A.NOPROG:
                Ss.setdefault(e + 30, set()).add("A")
                Ss.setdefault(e + 29, set()).add("A")
    for S in rng.integers(931, 1320, 2):
        Ss.setdefault(int(S), set()).add("A")
    for S, modes in sorted(Ss.items()):
        for mode in sorted(modes):
            gN = corrupt(g0, i, S, mode)
            gE = corrupt(es0, ies, S, mode) if ies is not None else corrupt(es0, None, S, mode)
            if ies is None:  # still corrupt ES days after this date
                later = np.where(es0.dates > date)[0]
                gE = corrupt(es0, later[0] - 1, 1440, mode) if len(later) else es0
            _CUR["ES"] = gE
            lvN, lvE = features.daily_levels(gN), features.daily_levels(gE)
            for name, fn in C.items():
                es_leg = name in A.ES_ONLY
                ii = ies if es_leg else i
                if ii is None:
                    continue
                try:
                    new = sigs_by_day(fn(gE, lvE) if es_leg else fn(gN, lvN)).get(ii, [])
                except Exception as ex:
                    rows.append(dict(cand=name, date=date, S=S, mode=mode, status="ERROR", detail=repr(ex)[:120]))
                    continue
                base = BASE[name].get(ii, [])
                bmap = {int(s["entry_slot"]): s for s in base}
                nmap = {int(s["entry_slot"]): s for s in new}
                status, detail = "ok", ""
                for e, s in bmap.items():
                    if e < S or (e == S and mode == "A"):
                        t = nmap.get(e)
                        if t is None:
                            status, detail = "VIOLATION", f"signal at {e} vanished"; break
                        if key(s) != key(t):
                            status, detail = "VIOLATION", f"changed {key(s)} -> {key(t)}"; break
                        if s.get("exit_slot") != t.get("exit_slot"):
                            if name in A.NOPROG and S <= e + 29:
                                pass
                            else:
                                status, detail = "VIOLATION", f"exit_slot {s.get('exit_slot')} -> {t.get('exit_slot')}"; break
                    elif e == S and mode == "B":
                        t = nmap.get(e)
                        if t is None or int(t["dir"]) != int(s["dir"]):
                            status, detail = "VIOLATION", f"entry/dir depends on open of entry minute ({e})"; break
                        if key(s) != key(t) and status == "ok":
                            status, detail = "uses_fill_price", f"{key(s)} -> {key(t)}"
                if status in ("ok", "uses_fill_price"):
                    for e, t in nmap.items():
                        if e not in bmap and (e < S or (e == S and mode == "A")):
                            status, detail = "VIOLATION", f"new signal at {e} appeared after corrupting >= {S}"; break
                        if e not in bmap and e == S and mode == "B":
                            status, detail = "VIOLATION", f"new signal at {e} depends on open of entry minute"; break
                rows.append(dict(cand=name, date=date, S=S, mode=mode, status=status, detail=detail,
                                 has_base_sig=len(base) > 0))
    if n_i % 10 == 0:
        print(f"  day {n_i+1}/{len(days)} {date}  {time.time()-t0:.0f}s", flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(A.HERE, "out_01_lookahead.csv"), index=False)
summ = df.groupby(["cand", "status"]).size().unstack(fill_value=0)
with pd.option_context("display.width", 200):
    print(summ)
    print(df[df.status.isin(["VIOLATION", "ERROR"])].head(40).to_string())
    print(df[df.status == "uses_fill_price"].groupby("cand").size())
