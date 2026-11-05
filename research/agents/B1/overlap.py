from s10_refva import *
import sys
i2 = sim.run_signals(g, strategies.i2_v1_signals(g, lv)); i2d = set(i2.i)
for kind in ("on",):
    for frac in (0.6, 0.8):
        CL[(kind, frac)] = None
def s10f(kind, m, frac=0.7, last=120):
    if CL.get((kind, frac)) is None:
        out = {}
        for r in lv.itertuples():
            if not ok_day(r.i): continue
            on = ind.profile(g.H[r.i, :RTH_OFF], g.L[r.i, :RTH_OFF], g.V[r.i, :RTH_OFF])
            out[r.i] = ind.va_from_profile(*on, frac)
        CL[(kind, frac)] = out
    CLk = CL[(kind, frac)]; res = []
    for r in lv.itertuples():
        if r.i not in CLk: continue
        va_, _, vh = CLk[r.i]; o = r.open_0930
        d = -1 if o > vh else (1 if o < va_ else 0)
        if d == 0: continue
        C = g.C[r.i]; cnt = 0
        for idx in range(4, last, 5):
            c = C[RTH_OFF + idx]
            if np.isnan(c): continue
            cnt = cnt + 1 if va_ <= c <= vh else 0
            if cnt >= m:
                ei = idx + 1
                if not NEWS(ei): res.append(sig(r.i, RTH_OFF + ei, d))
                break
    return res
for m in (1, 2, 3):
    for frac in (0.6, 0.7, 0.8):
        run(f"ONVA m={m} frac={frac}", s10f("on", m, frac))
for last in (60, 90, 150):
    run(f"ONVA m=2 last={last}", s10f("on", 2, 0.7, last))
def ov(name, tr):
    D = tr[tr.split == "discovery"]
    on = D[D.i.isin(i2d)]; off = D[~D.i.isin(i2d)]
    # same direction as I2 on overlap days
    same = on.merge(i2[["i", "dir"]], on="i", suffixes=("", "_i2"))
    print(f"{name}: on-I2-days n={len(on)} avg={on.pnl.mean():.0f} (same dir {(same.dir==same.dir_i2).mean():.0%}) | non-I2 days n={len(off)} avg={off.pnl.mean():.0f}")
ov("S3 k3m2", sim.run_signals(g, s3(3, 2)))
ov("ONVA m2", sim.run_signals(g, s10f("on", 2)))
ov("ONVA m1", sim.run_signals(g, s10f("on", 1)))
ov("C3+ON m2", sim.run_signals(g, s10("c3_plus_on", 2)))
