import warnings; warnings.filterwarnings("ignore")
from a1lib import *
from signals import *
g, lv = load(); d = daytable(g, lv); DISC = (g.split == "discovery")
rows = []
def go(name, sigs):
    tr = sim.run_signals(g, sigs)
    s = summ(tr, g, "discovery"); s["name"] = name
    if len(tr):
        s["L_avg"] = round(tr[tr.dir == 1].pnl.mean(), 0); s["S_avg"] = round(tr[tr.dir == -1].pnl.mean(), 0)
        h = halves(tr); s["h_pos"] = int((h["mean"].iloc[:3] > 0).sum())
    rows.append(s)
buckets = [(0.05, 0.2), (0.2, 0.4), (0.4, 0.7), (0.7, 3)]
for ref in ["pdc", "pdr"]:
    for lo, hi in buckets:
        for st in [0.3, 0.6]:
            go(f"fill {ref} [{lo},{hi}) st{st} tpRef 12:00", gap_signals(g, lv, d, "fill", lo, hi, ref, stop_atr=st, exit_time="12:00", days=DISC))
            go(f"fill {ref} [{lo},{hi}) st{st} tpRef EOD", gap_signals(g, lv, d, "fill", lo, hi, ref, stop_atr=st, days=DISC))
            go(f"go {ref} [{lo},{hi}) st{st} EOD", gap_signals(g, lv, d, "go", lo, hi, ref, stop_atr=st, target=None, days=DISC))
            go(f"go {ref} [{lo},{hi}) st{st} conf10:00 EOD", gap_signals(g, lv, d, "go", lo, hi, ref, stop_atr=st, target=None, confirm="10:00", days=DISC))
df = pd.DataFrame(rows).set_index("name")
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
print(df[["n","win%","avg","total","maxdd","ret/dd","sharpe_ann","t","L_avg","S_avg","h_pos"]].to_string())
log_variants("gap", df.reset_index()[["name","n","avg","sharpe_ann"]].to_dict("records"))
