from s3_robust import *
def refva_levels(kind):
    out = {}
    for r in lv.itertuples():
        if not ok_day(r.i): continue
        i = r.i; j = g.prev_rth_day(i)
        on = ind.profile(g.H[i, :RTH_OFF], g.L[i, :RTH_OFF], g.V[i, :RTH_OFF])
        if kind == "on": kp = on
        elif kind == "full_prev": kp = ind.profile(g.H[j], g.L[j], g.V[j])
        elif kind == "rth_plus_on": kp = ind.add_profiles([profs[j], on])
        elif kind == "c3_plus_on":
            prev = ind._prev_rth_list(g, i, 3)
            if len(prev) < 3: continue
            kp = ind.add_profiles([profs[x] for x in prev] + [on])
        out[i] = ind.va_from_profile(*kp)
    return out
for kind in ("on", "full_prev", "rth_plus_on", "c3_plus_on"):
    CL[(kind, 0.7)] = refva_levels(kind)
def s10(kind, m):
    CLk = CL[(kind, 0.7)]
    out = []
    for r in lv.itertuples():
        if r.i not in CLk: continue
        va_, _, vh = CLk[r.i]; o = r.open_0930
        d = -1 if o > vh else (1 if o < va_ else 0)
        if d == 0: continue
        C = g.C[r.i]; cnt = 0
        for idx in range(4, 120, 5):
            c = C[RTH_OFF + idx]
            if np.isnan(c): continue
            cnt = cnt + 1 if va_ <= c <= vh else 0
            if cnt >= m:
                ei = idx + 1
                if not NEWS(ei): out.append(sig(r.i, RTH_OFF + ei, d))
                break
    return out
for kind in ("on", "full_prev", "rth_plus_on", "c3_plus_on"):
    for m in (1, 2):
        run(f"S10 refVA={kind} m={m}", s10(kind, m))
