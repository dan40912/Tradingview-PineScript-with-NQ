"""Phase E: run every frozen candidate (research/agents/*/candidates.py) on ALL data incl. the 2026 holdout.
Usage: python3 research/run_holdout.py [agent ...]   -> writes research/results/holdout_summary.csv and per-candidate trades.
"""
import sys, os, importlib.util, traceback
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, pandas as pd
from lib import data, features, sim, metrics, strategies

OUT = os.path.join(HERE, "results"); os.makedirs(os.path.join(OUT, "trades"), exist_ok=True)
g = data.load_grid("NQ", include_holdout=True)
lv = features.daily_levels(g)
ndays = pd.Series(g.split[g.tradable]).value_counts().to_dict()


def load_candidates(agent):
    p = os.path.join(HERE, "agents", agent, "candidates.py")
    if not os.path.exists(p):
        return {}
    sys.path.insert(0, os.path.dirname(p))
    spec = importlib.util.spec_from_file_location(f"cand_{agent}", p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    sys.path.pop(0)
    return getattr(m, "CANDIDATES", {})


def evaluate(name, tr):
    rows = []
    for sp in ["discovery", "validation", "holdout"]:
        s = metrics.summary(tr[tr.split == sp], n_days=ndays.get(sp)) if len(tr) else {"n": 0}
        s.update({"candidate": name, "split": sp}); rows.append(s)
    s = metrics.summary(tr, n_days=sum(ndays.values())) if len(tr) else {"n": 0}
    s.update({"candidate": name, "split": "ALL"}); rows.append(s)
    return rows


agents = sys.argv[1:] or sorted(os.listdir(os.path.join(HERE, "agents")))
rows = []
base = sim.run_signals(g, strategies.i2_v1_signals(g, lv))
base.to_csv(os.path.join(OUT, "trades", "BASE__I2v1.csv"), index=False)
rows += evaluate("BASE/I2v1", base)
for a in agents:
    try:
        cands = load_candidates(a)
    except Exception:
        traceback.print_exc(); continue
    for name, fn in cands.items():
        key = f"{a}/{name}"
        try:
            tr = sim.run_signals(g, fn(g, lv))
        except Exception:
            print("FAILED", key); traceback.print_exc(); continue
        tr.to_csv(os.path.join(OUT, "trades", key.replace("/", "__") + ".csv"), index=False)
        rows += evaluate(key, tr)
        print("ok", key, len(tr))
df = pd.DataFrame(rows)
cols = ["candidate", "split", "n", "win%", "avg", "total", "pf", "maxdd", "ret/dd", "sharpe_ann", "t"]
df = df[[c for c in cols if c in df]]
df.to_csv(os.path.join(OUT, "holdout_summary.csv"), index=False)
with pd.option_context("display.width", 200, "display.max_rows", 500):
    print(df)
