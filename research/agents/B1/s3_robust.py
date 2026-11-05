from standalone import *
profs = ind._rth_profiles(g)
def comp_levels(k, frac=0.70, same_contract=True):
    out = {}
    for r in lv.itertuples():
        if not ok_day(r.i): continue
        prev = ind._prev_rth_list(g, r.i, k)
        if len(prev) < k: continue
        kc, pc = ind.add_profiles([profs[x] for x in prev])
        out[r.i] = ind.va_from_profile(kc, pc, frac)
    return out
CL = {}
def s3(k=3, m=2, frac=0.70, last=120, tp=55, sl=145, start_idx=4, step=5):
    key = (k, frac)
    if key not in CL: CL[key] = comp_levels(k, frac)
    out = []
    for r in lv.itertuples():
        if r.i not in CL[key]: continue
        va_, _, vh = CL[key][r.i]; o = r.open_0930
        d = -1 if o > vh else (1 if o < va_ else 0)
        if d == 0: continue
        C = g.C[r.i]; cnt = 0
        for idx in range(step - 1, last, step):
            c = C[RTH_OFF + idx]
            if np.isnan(c): continue
            cnt = cnt + 1 if va_ <= c <= vh else 0
            if cnt >= m:
                ei = idx + 1
                if not NEWS(ei) and ei <= last: out.append(sig(r.i, RTH_OFF + ei, d, tp, sl))
                break
    return out
if __name__ == "__main__":
    for k in (1, 2, 3, 4, 5):
        for m in (1, 2, 3):
            run(f"S3 k={k} m={m}", s3(k, m))
    for frac in (0.6, 0.65, 0.75, 0.8):
        for m in (1, 2, 3):
            run(f"S3 k=3 m={m} frac={frac}", s3(3, m, frac))
    for last in (60, 90, 150):
        run(f"S3 k=3 m=2 last={last}", s3(3, 2, last=last))
    for step in (3, 10):
        run(f"S3 k=3 m=2 step={step}m", s3(3, 2, step=step))
    for tp, sl in ((40, 145), (70, 145), (55, 100), (55, 200), (100, 100)):
        run(f"S3 k=3 m=2 tp={tp} sl={sl}", s3(3, 2, tp=tp, sl=sl))
