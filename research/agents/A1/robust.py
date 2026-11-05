"""Discovery-only neighbour / stress / ES checks for the OOV-ORB family."""
import warnings, sys; warnings.filterwarnings("ignore")
from a1lib import *
from candidates import oov_orb
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "discovery"
assert SPLIT == "discovery"
rows = []
def ev(sym, name, sigs, **kw):
    g, lv = load(sym)
    sigs = [s for s in sigs if g.split[s["i"]] == SPLIT]
    tr = sim.run_signals(g, sigs, **kw)
    s = summ(tr, g, SPLIT); s.update(sym=sym, name=name)
    if len(tr):
        h = halves(tr); s["h"] = "/".join(str(int(x)) for x in h["mean"].values)
    rows.append(s); return tr
g, lv = load("NQ")
base = {"C1": dict(N=15, cutoff="12:00", stop_mult=1.0, k=None, exit_time="15:59"),
        "C2": dict(N=30, cutoff="12:00", stop_mult=1.0, k=1.0, exit_time="15:59")}
grid = {"N": {"C1": [10, 20], "C2": [20, 45]}, "cutoff": ["11:00", "13:00"], "stop_mult": [0.75, 1.25],
        "k": {"C1": [3.0], "C2": [0.75, 1.5]}, "exit_time": ["15:00", "15:30"], "regime": ["outPR"]}
for cn, b in base.items():
    ev("NQ", f"{cn} base", oov_orb(g, lv, **b))
    for p, vals in grid.items():
        vals = vals[cn] if isinstance(vals, dict) else vals
        for v in vals:
            kw = dict(b); kw[p] = v
            ev("NQ", f"{cn} {p}={v}", oov_orb(g, lv, **kw))
    # stress: entry 1 minute later, cost +$40
    sg = oov_orb(g, lv, **b)
    for s in sg: s["entry_slot"] += 1
    ev("NQ", f"{cn} stress late1m", sg)
    tr = ev("NQ", f"{cn} base", oov_orb(g, lv, **b)); rows.pop()
    t2 = tr.copy(); t2["pnl"] -= 40; s = summ(t2, g, SPLIT); s.update(sym="NQ", name=f"{cn} stress cost+40"); rows.append(s)
    for R in [1000, 1500, 2000]:
        ev("NQ", f"{cn} risk${R}", oov_orb(g, lv, risk_usd=R, **b))
    ge, le = load("ES")
    ev("ES", f"{cn} ES", oov_orb(ge, le, **b))
df = pd.DataFrame(rows).set_index("name")
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
print(df[["sym","n","win%","avg","total","maxdd","ret/dd","sharpe_ann","t","h"]].to_string())
log_variants("oov_robust", df.reset_index()[["name","n","avg","sharpe_ann"]].to_dict("records"))
df.to_csv("robust_discovery.csv")
