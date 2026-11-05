from standalone import *
def fade_all(w, cond="all"):
    out = []
    for r in lv.itertuples():
        if not ok_day(r.i): continue
        i = r.i; s = RTH_OFF + w
        c = g.C[i, s - 1]; o = r.open_0930
        cdn = ia["cd930"][i, s - 1] / max(np.nansum(g.V[i, RTH_OFF:s]), 1)
        mv = c - o
        if not np.isfinite(mv) or mv == 0: continue
        d = -int(np.sign(mv))
        agree = np.sign(cdn) == np.sign(mv)
        if cond == "agree" and not agree: continue
        if cond == "disagree" and agree: continue
        out.append(sig(i, s, d))
    return out
for w in (10, 15, 20, 30, 45):
    for cond in ("all", "agree", "disagree"):
        run(f"fade open move w={w} {cond}", fade_all(w, cond))
