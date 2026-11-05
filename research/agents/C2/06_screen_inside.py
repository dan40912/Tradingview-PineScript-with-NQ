"""06: Inside-value-day family (balance-day behaviour). Discovery only."""
import numpy as np
import c2lib as L
from lib import sim
import sig
from edge import edge_signals
g, lv, i2 = L.load()
fam = "06_inside"
run = lambda s: sim.run_signals(g, s)
print("== edge fade (区邊) structural variants")
for exc in [0.0, 0.1, 0.2]:
    for tp in [55.0, "poc", "opp", None]:
        for sl in [145.0, "ext"]:
            for ex in [179, 389]:
                if tp is None and ex == 179 and sl == 145.0:
                    pass
                ss = [s for s in edge_signals(g, lv, tp=tp, sl=sl, exit_idx=ex) if s["exc"] >= exc * s["adr"]]
                tr = run(ss)
                L.disc(g, tr, f"EDGE exc>={exc} tp={tp} sl={sl} ex={ex}", fam)
print("== edge fade: on_in (overnight inside VA) / open position")
for cond, f in [("on_in", lambda s: s["on_in"]), ("not_on_in", lambda s: not s["on_in"]),
                ("short_only", lambda s: s["dir"] == -1), ("long_only", lambda s: s["dir"] == 1),
                ("vaw<0.4adr", lambda s: s["va_w"] < 0.4 * s["adr"]), ("vaw>0.6adr", lambda s: s["va_w"] > 0.6 * s["adr"])]:
    tr = run([s for s in edge_signals(g, lv) if f(s)])
    L.disc(g, tr, f"EDGE ref cond={cond}", fam)
print("== inside acceptance outside VA (follow)")
for n in [2, 3, 6]:
    for st in [30, 60]:
        for ex in [359, 389]:
            tr = run(sig.inside_accept(g, lv, n_out=n, start=st, exit_idx=ex))
            L.disc(g, tr, f"ACC n={n} start={st} ex={ex}", fam)
print("== inside extension fade")
for k in [0.5, 0.75]:
    for s in [0.35, 0.5]:
        for tp in ["open", "poc", None]:
            tr = run(sig.inside_ext_fade(g, lv, k=k, s=s, tp=tp))
            L.disc(g, tr, f"IXF k={k} s={s} tp={tp}", fam)
