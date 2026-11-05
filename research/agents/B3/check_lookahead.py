"""Lookahead test: scramble all data at/after the decision minute and verify features unchanged."""
import sys, copy; sys.path.insert(0, "research"); sys.path.insert(0, "research/agents/B3")
import numpy as np, pandas as pd
from lib import data, features
import features_ml as fm
g = data.load_grid("NQ"); lv = features.daily_levels(g); es = data.load_grid("ES")
ev = pd.DataFrame({"i": [100, 200, 300, 400], "entry_slot": [947, 960, 991, 936]})
X0 = fm.build(fm.Context(g, lv, es=es), ev)
rng = np.random.default_rng(0)
for n, (i, e) in enumerate(zip(ev.i, ev.entry_slot)):
    g2 = copy.deepcopy(g); es2 = copy.deepcopy(es)
    for G in (g2, es2):
        k = i if G is g2 else es.dindex[g.dates[i]]
        for A in (G.O, G.H, G.L, G.C):
            A[k, e:] = A[k, e:] + rng.normal(0, 50, A[k, e:].shape)
            A[k + 1:, :] = A[k + 1:, :] * 1.1
        G.V[k, e:] *= 7
    X1 = fm.build(fm.Context(g2, features.daily_levels(g2), es=es2), ev.iloc[[n]])
    a, b = X0.iloc[n].values, X1.iloc[0].values
    bad = ~(np.isclose(a, b, equal_nan=True))
    print(i, e, "changed features:", list(X0.columns[bad]))
