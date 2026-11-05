"""Stage A (discovery only): standalone signal generators built on B1 indicators.
All signals: entry at open of entry_slot using bars < entry_slot; default 55/145, exit 12:29, 1/day."""
from common import *
import itertools, sys

g, lv = load("NQ")
pf = ind.profile_features(g, lv)
ia = ind.intraday_arrays(g)
EXIT = RTH_OFF + 179
NEWS = lambda ei: 28 <= ei <= 32   # 09:58-10:02 blackout (idx from 09:30)

def sig(i, slot, d, tp=55, sl=145, tag=""):
    return {"i": i, "entry_slot": slot, "dir": int(d), "tp_pts": tp, "sl_pts": sl, "exit_slot": EXIT, "tag": tag}

def ok_day(i):
    return bool(lv.has_prev[i]) and np.isfinite(lv.open_0930[i])

# S1 value-acceptance breakout on open-inside days: n consecutive 5m closes outside prior VA
def s1_accept(n=2, last=120, start=14, side=0, tp=55, sl=145, require_inside_open=True):
    out = []
    for r in lv.itertuples():
        if not ok_day(r.i) or (require_inside_open and r.i2_dir != 0): continue
        C = g.C[r.i]; cu = cd = 0
        for idx in range(4, last, 5):
            c = C[RTH_OFF + idx]
            if np.isnan(c): continue
            cu = cu + 1 if c > r.vah else 0
            cd = cd + 1 if c < r.val else 0
            ei = idx + 1
            if ei < start or NEWS(ei): continue
            if cu >= n and side >= 0:
                out.append(sig(r.i, RTH_OFF + ei, 1, tp, sl)); break
            if cd >= n and side <= 0:
                out.append(sig(r.i, RTH_OFF + ei, -1, tp, sl)); break
    return out

# S3 composite-VA I2: open outside k-day composite VA, wait m 5m closes inside -> revert
def s3_composite(k=3, m=2, last=120, tp=55, sl=145):
    out = []
    for r in lv.itertuples():
        if not ok_day(r.i): continue
        p = pf.loc[r.i]; va_, vh = p.get(f"c{k}_val"), p.get(f"c{k}_vah")
        if not np.isfinite(va_): continue
        o = r.open_0930
        d = -1 if o > vh else (1 if o < va_ else 0)
        if d == 0: continue
        C = g.C[r.i]; cnt = 0
        for idx in range(4, last, 5):
            c = C[RTH_OFF + idx]
            if np.isnan(c): continue
            cnt = cnt + 1 if va_ <= c <= vh else 0
            if cnt >= m:
                ei = idx + 1
                if not NEWS(ei): out.append(sig(r.i, RTH_OFF + ei, d, tp, sl))
                break
    return out

# S4 naked-POC magnet: at 09:30+w, nearest naked POC (incl. prior-day POC) between lo..hi pts away -> trade toward it
def s4_naked(w=5, lo=60, hi=300, tp=55, sl=145, need_momentum=True):
    out = []
    for r in lv.itertuples():
        if not ok_day(r.i): continue
        p = pf.loc[r.i]; i = r.i
        s = RTH_OFF + w
        c = g.C[i, s - 1]
        if not np.isfinite(c): continue
        cands = []
        if p.prev_poc_naked == 1: cands.append(p.poc_mid)
        o = r.open_0930
        if np.isfinite(p.get("naked_up_dist", np.nan)): cands.append(o + p.naked_up_dist)
        if np.isfinite(p.get("naked_dn_dist", np.nan)): cands.append(o - p.naked_dn_dist)
        # still untouched since 09:30?
        hh, ll = np.nanmax(g.H[i, RTH_OFF:s]), np.nanmin(g.L[i, RTH_OFF:s])
        cands = [x for x in cands if not (ll <= x <= hh)]
        if not cands: continue
        dist = np.array(cands) - c
        k = np.argmin(np.abs(dist)); dd = dist[k]
        if not (lo <= abs(dd) <= hi): continue
        d = int(np.sign(dd))
        if need_momentum and (c - o) * d <= 0: continue
        out.append(sig(i, s, d, tp, sl))
    return out

# S5 ETH-VWAP band fade: close beyond vwap18 +- z*sd, then a close back inside -> fade toward VWAP
def s5_vwapband(z=2.0, start=15, last=120, anchor="18", tp=55, sl=145):
    vw, sd = ia["vwap" + anchor], ia["sd" + anchor]
    out = []
    for r in lv.itertuples():
        if not ok_day(r.i): continue
        i = r.i; armed = 0
        for idx in range(0, last):
            s = RTH_OFF + idx
            c = g.C[i, s]
            if np.isnan(c) or np.isnan(sd[i, s]) or sd[i, s] <= 0: continue
            zz = (c - vw[i, s]) / sd[i, s]
            if zz > z: armed = -1
            elif zz < -z: armed = 1
            elif armed != 0 and abs(zz) < z:
                ei = idx + 1
                if ei >= start and not NEWS(ei):
                    out.append(sig(i, s + 1, armed, tp, sl)); break
                armed = 0
    return out

# S6 delta-proxy divergence at w minutes: price below open (beyond prior VA edge optional) but cum delta>0 -> long
def s6_divergence(w=30, thr=0.05, need_outside_va=False, tp=55, sl=145, follow=False):
    out = []
    for r in lv.itertuples():
        if not ok_day(r.i): continue
        i = r.i; s = RTH_OFF + w
        c = g.C[i, s - 1]; o = r.open_0930
        cdn = ia["cd930"][i, s - 1] / max(np.nansum(g.V[i, RTH_OFF:s]), 1)
        mv = c - o
        if not np.isfinite(mv): continue
        d = 0
        if mv < 0 and cdn > thr: d = 1
        elif mv > 0 and cdn < -thr: d = -1
        if d == 0: continue
        if need_outside_va and not ((d == 1 and c < r.val) or (d == -1 and c > r.vah)): continue
        if follow: d = -d
        out.append(sig(i, s, d, tp, sl))
    return out

# S8 developing value acceptance: at w min, developing VA (09:30..) fully above prior VAH -> long / below VAL -> short
def s8_dva_accept(w=60, tp=55, sl=145, fade=False, mode="full"):
    out = []
    for r in lv.itertuples():
        if not ok_day(r.i): continue
        i = r.i; s = RTH_OFF + w
        dv = ind.developing_va(g, i, s)
        if not np.isfinite(dv[0]): continue
        if mode == "full":
            d = 1 if dv[0] > r.vah else (-1 if dv[2] < r.val else 0)
        else:  # developing POC outside prior VA
            pm = dv[1] + 2.5
            d = 1 if pm > r.vah else (-1 if pm < r.val else 0)
        if d == 0: continue
        if fade: d = -d
        out.append(sig(i, s, d, tp, sl))
    return out

# S9 poor-extreme repair: prior day poor high (vol_top_rel>=th) and open within [lo, hi] pts below it -> long at 09:30+w (mirror for lows)
def s9_poor_repair(th=0.25, lo=55, hi=250, w=1, tp=55, sl=145):
    out = []
    for r in lv.itertuples():
        if not ok_day(r.i): continue
        p = pf.loc[r.i]; i = r.i; s = RTH_OFF + w
        c = g.C[i, s - 1] if w > 0 else r.open_0930
        cand = []
        if p.vol_top_rel >= th and lo <= r.pdh - c <= hi: cand.append((r.pdh - c, 1))
        if p.vol_bot_rel >= th and lo <= c - r.pdl <= hi: cand.append((c - r.pdl, -1))
        if not cand: continue
        d = min(cand)[1]
        out.append(sig(i, max(s, RTH_OFF), d, tp, sl))
    return out

def run(name, sigs):
    tr = sim.run_signals(g, sigs)
    if len(tr) == 0:
        print(f"{name:55s} no trades"); return tr
    s = summ(g, tr, "discovery")
    print(f"{name:55s} n={s.get('n',0):4} win={s.get('win%',0):5} avg={s.get('avg',0):7} tot={s.get('total',0):8} dd={s.get('maxdd',0):8} r/dd={s.get('ret/dd',0):5} sh={s.get('sharpe_ann',0)}")
    return tr

if __name__ == "__main__":
    # restrict to discovery for speed: sims include validation but we only print discovery
    which = sys.argv[1:] or ["s1", "s3", "s4", "s5", "s6", "s8", "s9"]
    if "s1" in which:
        for n, side, start in itertools.product((1, 2, 3), (0, 1, -1), (14,)):
            run(f"S1 accept n={n} side={side}", s1_accept(n=n, side=side, start=start))
        run("S1 accept n=2 any-open-day", s1_accept(n=2, require_inside_open=False))
    if "s3" in which:
        for k, m in itertools.product((2, 3, 5), (1, 2, 3)):
            run(f"S3 composite k={k} m={m}", s3_composite(k, m))
    if "s4" in which:
        for w, (lo, hi), mom in itertools.product((1, 5, 15), ((60, 200), (60, 400), (100, 600)), (True, False)):
            run(f"S4 naked w={w} {lo}-{hi} mom={mom}", s4_naked(w, lo, hi, need_momentum=mom))
    if "s5" in which:
        for anc, z in itertools.product(("18", "930"), (1.5, 2.0, 2.5)):
            run(f"S5 vwapband anchor={anc} z={z}", s5_vwapband(z, anchor=anc))
    if "s6" in which:
        for w, thr, va_, fol in itertools.product((15, 30, 60), (0.0, 0.05), (False, True), (False, True)):
            run(f"S6 div w={w} thr={thr} outVA={va_} follow={fol}", s6_divergence(w, thr, va_, follow=fol))
    if "s8" in which:
        for w, fade, mode in itertools.product((30, 60, 90), (False, True), ("full", "poc")):
            run(f"S8 dva w={w} fade={fade} mode={mode}", s8_dva_accept(w, fade=fade, mode=mode))
    if "s9" in which:
        for th, (lo, hi), w in itertools.product((0.2, 0.3), ((55, 150), (55, 300)), (1, 15)):
            run(f"S9 poor repair th={th} {lo}-{hi} w={w}", s9_poor_repair(th, lo, hi, w))
