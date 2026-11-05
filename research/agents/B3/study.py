"""B3 main study: walk-forward ML on classic + structural features (discovery), then one
final fit on all discovery evaluated on validation.  Never touches 2026.

Run:  python research/agents/B3/study.py      (writes research/agents/B3/results/*)
"""
from __future__ import annotations
import os, sys, json, time, warnings
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..")); sys.path.insert(0, HERE)
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
import lightgbm as lgb
from lib import data, features, sim, metrics, strategies
import features_ml as fm

warnings.filterwarnings("ignore")
OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)
DECISION = ["09:45", "10:00", "10:30"]
HORIZONS = [60, 120]
XS = [40, 80]
EXIT_EOD = sim.slot_of("15:59")
WF_START = "2024-07"          # first walk-forward test month (>= 6 months of training)
DELTAS = [0.0, 0.05, 0.10]    # trade when |p-0.5| > delta  (chosen on discovery WF only)
META_Q = [0.55, 0.65, 0.75]   # keep I2 trade when p >= q   (chosen on discovery WF only)


# ------------------------------------------------------------------ models
def model_l1(C=0.05):
    # intercept_scaling large so the (penalised) liblinear intercept can reach the base rate
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(penalty="l1", C=C, solver="liblinear", max_iter=500, intercept_scaling=100.0))


class L1CV:
    """L1-logistic whose C is picked inside the training window by an expanding time-series CV
    (3 splits, AUC).  No information from the test month is used."""
    CS = [0.01, 0.03, 0.1, 0.3]

    def fit(self, X, y):
        n = len(y); cuts = [int(n * f) for f in (0.55, 0.7, 0.85, 1.0)]
        best, bestC = -1, self.CS[0]
        for C in self.CS:
            aucs = []
            for a, b in zip(cuts[:-1], cuts[1:]):
                if len(np.unique(y[:a])) < 2 or len(np.unique(y[a:b])) < 2:
                    continue
                m = model_l1(C).fit(X[:a], y[:a])
                aucs.append(roc_auc_score(y[a:b], m.predict_proba(X[a:b])[:, 1]))
            sc = np.mean(aucs) if aucs else -1
            if sc > best:
                best, bestC = sc, C
        self.C_ = bestC
        self.m_ = model_l1(bestC).fit(X, y)
        return self

    def predict_proba(self, X):
        return self.m_.predict_proba(X)

    def __getitem__(self, k):
        return self.m_[k]


def model_gbm():
    return lgb.LGBMClassifier(n_estimators=150, learning_rate=0.03, max_depth=2, num_leaves=3,
                              min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                              reg_lambda=5.0, n_jobs=1, verbose=-1, random_state=0)


def model_gbm_small():
    m = model_gbm(); m.set_params(min_child_samples=8, n_estimators=80); return m


MODELS = {"L1": model_l1, "L1CV": L1CV, "GBM": model_gbm, "GBMs": model_gbm_small}
MODELS_FOR = {"dir": ["L1", "L1CV", "GBM"], "hit": ["L1", "L1CV", "GBM"], "meta": ["L1", "L1CV", "GBM", "GBMs"]}


def fit_predict(mname, Xtr, ytr, Xte):
    if len(np.unique(ytr)) < 2:
        return np.full(len(Xte), float(np.mean(ytr))), None
    m = MODELS[mname]()
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1], m


# ------------------------------------------------------------------ data
def load_all():
    g = data.load_grid("NQ"); lv = features.daily_levels(g); es = data.load_grid("ES")
    assert g.dates[-1] < "2026-01-01"
    ctx = fm.Context(g, lv, es=es)
    return g, lv, ctx


def build_tasks(g, lv, ctx):
    """Returns dict task_name -> DataFrame(events with i, entry_slot, date, split, y, + trade spec)."""
    tasks = {}
    ev = fm.fixed_time_events(g, lv, DECISION)
    X = fm.build(ctx, ev)
    ev = pd.concat([ev, X], axis=1)
    for t in DECISION:
        e = ev[ev.t == t].copy()
        for h in HORIZONS:
            fr = fm.forward_return(g, e.i.values, e.entry_slot.values, h)
            d = e.assign(fwd=fr)
            d = d[np.isfinite(d.fwd) & (d.fwd != 0)].copy()
            d["y"] = (d.fwd > 0).astype(int)
            d["kind"] = "dir"; d["h"] = h
            tasks[f"dir_{t}_{h}"] = d
        for x in XS:
            ys = []
            for r in e.itertuples():
                tr = sim.simulate_one(g, r.i, r.entry_slot, 1, tp_pts=x, sl_pts=x, exit_slot=EXIT_EOD)
                ys.append(np.nan if tr is None else (1.0 if tr["reason"] == "tp" else 0.0 if tr["reason"] != "time" else float(tr["pts"] > 0)))
            d = e.assign(y=ys)
            d = d[np.isfinite(d.y)].copy(); d["y"] = d.y.astype(int)
            d["kind"] = "hit"; d["x"] = x
            tasks[f"hit_{t}_{x}"] = d
    # I2 meta-label
    sigs = strategies.i2_v1_signals(g, lv)
    tr = sim.run_signals(g, sigs)
    e = tr[["i", "entry_slot", "date", "split", "year", "pnl", "dir"]].copy()
    Xm = fm.build(ctx, e)
    d = pd.concat([e.reset_index(drop=True), Xm.reset_index(drop=True)], axis=1)
    d["y"] = (d.pnl > 0).astype(int); d["kind"] = "meta"
    # direction-aware versions of signed features (so a "long" and a "short" trade look alike)
    tasks["meta_i2"] = d
    return tasks, list(X.columns), sigs


def feature_sets(cols):
    struct = [c for c in cols if c.startswith(("s_", "x_"))]
    classic = [c for c in cols if not c.startswith(("s_", "x_"))]
    return {"ALL": cols, "STRUCT": struct, "CLASSIC": classic}


def meta_signed(d, cols):
    """For the meta task multiply direction-dependent features by trade dir so that the
    features read 'in favour of the trade'.  Unsigned features are left alone."""
    unsigned = ("rsi", "bbp", "stoch", "donch", "adx", "atr_rel", "bbw", "rng_rel", "vaw", "onrng", "pdrng",
                "rth_rng", "or15_rng", "rvol", "mins", "dow", "in_va", "open_in_va", "i2dir", "rth_pos")
    X = d[cols].copy()
    for c in cols:
        if any(c.endswith(u) or u in c for u in unsigned):
            if c.endswith(("rsi", "stoch")):
                X[c] = (X[c] - 50) * d["dir"].values
            elif c.endswith(("bbp", "donch", "rth_pos")):
                X[c] = (X[c] - 0.5) * d["dir"].values
            continue
        X[c] = X[c] * d["dir"].values
    return X


# ------------------------------------------------------------------ walk-forward
def months_of(dates):
    return pd.Series(dates).str[:7].values


def walk_forward(d, cols, mname, Xfun=None):
    disc = d[d.split == "discovery"].reset_index(drop=True)
    X = (Xfun(disc, cols) if Xfun else disc[cols]).values
    y = disc.y.values
    mon = months_of(disc.date.values)
    test_months = sorted(m for m in set(mon) if m >= WF_START)
    p = np.full(len(disc), np.nan)
    base = np.full(len(disc), np.nan)
    fold_models = []
    for m in test_months:
        tr, te = mon < m, mon == m
        if te.sum() == 0:
            continue
        p[te], mdl = fit_predict(mname, X[tr], y[tr], X[te])
        base[te] = y[tr].mean()
        fold_models.append((m, mdl, np.where(te)[0]))
    disc["p"] = p; disc["base"] = base
    return disc, fold_models, X


def final_fit(d, cols, mname, Xfun=None):
    disc = d[d.split == "discovery"].reset_index(drop=True)
    val = d[d.split == "validation"].reset_index(drop=True)
    Xd = (Xfun(disc, cols) if Xfun else disc[cols]).values
    Xv = (Xfun(val, cols) if Xfun else val[cols]).values
    val["p"], mdl = fit_predict(mname, Xd, disc.y.values, Xv)
    val["base"] = disc.y.mean()
    return val, mdl


def scores(df):
    df = df[np.isfinite(df.p)]
    y, p = df.y.values, df.p.values
    if len(np.unique(y)) < 2:
        return {}
    auc = roc_auc_score(y, p)
    acc = ((p > 0.5) == y).mean()
    maj = (df.base.values > 0.5).astype(int)
    acc_b = (maj == y).mean()
    ll = log_loss(y, np.clip(p, 1e-4, 1 - 1e-4)); llb = log_loss(y, np.clip(df.base.values, 1e-4, 1 - 1e-4))
    # bootstrap CI of AUC by day
    rng = np.random.default_rng(0)
    aucs = []
    for _ in range(300):
        k = rng.integers(0, len(y), len(y))
        if len(np.unique(y[k])) == 2:
            aucs.append(roc_auc_score(y[k], p[k]))
    lo, hi = np.quantile(aucs, [0.025, 0.975])
    return {"n": len(y), "base_rate": round(y.mean(), 3), "auc": round(auc, 3), "auc_lo": round(lo, 3), "auc_hi": round(hi, 3),
            "acc": round(acc, 3), "acc_naive": round(acc_b, 3), "logloss": round(ll, 4), "logloss_naive": round(llb, 4)}


# ------------------------------------------------------------------ trading rules
def trade_signals(df, task_kind, delta, spec):
    sigs = []
    for r in df.itertuples():
        if not np.isfinite(r.p):
            continue
        if r.p > 0.5 + delta:
            dr = 1
        elif r.p < 0.5 - delta:
            dr = -1
        else:
            continue
        s = {"i": int(r.i), "entry_slot": int(r.entry_slot), "dir": dr}
        if task_kind == "dir":
            s.update(tp_pts=None, sl_pts=None, exit_slot=int(r.entry_slot) + spec - 1)
        else:
            s.update(tp_pts=spec, sl_pts=spec, exit_slot=EXIT_EOD)
        sigs.append(s)
    return sigs


def ndays(g, split):
    return int(((g.split == split) & g.tradable).sum())


def ndays_wf(g):
    m = (g.split == "discovery") & g.tradable & (pd.Series(g.dates).str[:7].values >= WF_START)
    return int(m.sum())


def summ(tr, nd):
    s = metrics.summary(tr, n_days=nd) if len(tr) else {"n": 0}
    return s


# ------------------------------------------------------------------ permutation importance
def perm_importance(disc, fold_models, X, cols, mname, reps=2, seed=0):
    rng = np.random.default_rng(seed)
    ok = np.isfinite(disc.p.values)
    y = disc.y.values
    base_auc = roc_auc_score(y[ok], disc.p.values[ok])
    drops = np.zeros(len(cols))
    for j in range(len(cols)):
        acc = []
        for _ in range(reps):
            pp = disc.p.values.copy()
            for m, mdl, idx in fold_models:
                if mdl is None:
                    continue
                Xt = X[idx].copy()
                Xt[:, j] = rng.permutation(Xt[:, j])
                pp[idx] = mdl.predict_proba(Xt)[:, 1]
            acc.append(roc_auc_score(y[ok], pp[ok]))
        drops[j] = base_auc - np.mean(acc)
    return pd.Series(drops, index=cols).sort_values(ascending=False)


def l1_stability(fold_models, cols):
    rows = []
    for m, mdl, idx in fold_models:
        if mdl is None:
            continue
        coef = (mdl.m_ if hasattr(mdl, "m_") else mdl)[-1].coef_[0]
        rows.append(pd.Series(coef, index=cols, name=m))
    if not rows:
        return pd.DataFrame()
    C = pd.DataFrame(rows)
    st = pd.DataFrame({"sel_frac": (C != 0).mean(), "pos_frac": (C > 0).sum() / (C != 0).sum().replace(0, np.nan),
                       "mean_coef": C.mean()})
    return st.sort_values("sel_frac", ascending=False)


def gbm_stability(fold_models, cols, k=10):
    tops = []
    for m, mdl, idx in fold_models:
        if mdl is None:
            continue
        imp = pd.Series(mdl.booster_.feature_importance("gain"), index=cols)
        tops.append(set(imp.sort_values(ascending=False).index[:k]))
    if len(tops) < 2:
        return np.nan, pd.Series(dtype=float)
    jac = [len(a & b) / len(a | b) for n, a in enumerate(tops) for b in tops[n + 1:]]
    freq = pd.Series([c for t in tops for c in t]).value_counts() / len(tops)
    return float(np.mean(jac)), freq


# ------------------------------------------------------------------ main
def main():
    t0 = time.time()
    g, lv, ctx = load_all()
    tasks, cols, i2sigs = build_tasks(g, lv, ctx)
    fsets = feature_sets(cols)
    pd.to_pickle((tasks, cols), os.path.join(OUT, "tasks.pkl"))
    print("tasks built", {k: len(v) for k, v in tasks.items()}, f"{time.time()-t0:.0f}s", flush=True)
    i2_tr = sim.run_signals(g, i2sigs)
    rows, trade_rows, imp_store, stab_store = [], [], {}, {}
    nd_wf, nd_val = ndays_wf(g), ndays(g, "validation")
    n_variants = 0
    for tname, d in tasks.items():
        kind = d.kind.iloc[0]
        spec = d.h.iloc[0] if kind == "dir" else (d.x.iloc[0] if kind == "hit" else None)
        for fs, fcols in fsets.items():
            for mname in MODELS_FOR[kind]:
                n_variants += 1
                Xfun = meta_signed if kind == "meta" else None
                disc, folds, X = walk_forward(d, fcols, mname, Xfun)
                val, final_mdl = final_fit(d, fcols, mname, Xfun)
                s_wf, s_val = scores(disc), scores(val)
                row = {"task": tname, "fset": fs, "model": mname}
                row.update({f"wf_{k}": v for k, v in s_wf.items()})
                row.update({f"val_{k}": v for k, v in s_val.items()})
                # trading rule, threshold chosen on discovery WF only
                wfd = disc[np.isfinite(disc.p)]
                best, best_tot = None, -np.inf
                if kind in ("dir", "hit"):
                    for dl in DELTAS:
                        tr = sim.run_signals(g, trade_signals(wfd, kind, dl, spec))
                        s = summ(tr, nd_wf)
                        trade_rows.append({"task": tname, "fset": fs, "model": mname, "period": "wf_disc", "rule": f"delta={dl}", **s})
                        if s.get("n", 0) >= 40 and s["total"] > best_tot:
                            best, best_tot = dl, s["total"]
                    if best is not None:
                        tr = sim.run_signals(g, trade_signals(val, kind, best, spec))
                        s = summ(tr, nd_val)
                        trade_rows.append({"task": tname, "fset": fs, "model": mname, "period": "validation", "rule": f"delta={best}", **s})
                        row["chosen"] = f"delta={best}"; row["wf_trade_avg"] = round(best_tot / max(1, summ(sim.run_signals(g, trade_signals(wfd, kind, best, spec)), nd_wf)["n"]), 1)
                        row["val_trade_n"] = s.get("n", 0); row["val_trade_avg"] = s.get("avg", np.nan); row["val_trade_total"] = s.get("total", np.nan)
                else:  # meta: keep I2 trade if p >= q
                    base_wf = wfd.pnl
                    for q in META_Q:
                        kept = wfd[wfd.p >= q]
                        trade_rows.append({"task": tname, "fset": fs, "model": mname, "period": "wf_disc", "rule": f"keep p>={q}",
                                           "n": len(kept), "avg": round(kept.pnl.mean(), 1) if len(kept) else np.nan,
                                           "total": round(kept.pnl.sum(), 0), "unfiltered_avg": round(base_wf.mean(), 1)})
                        if len(kept) >= 0.4 * len(wfd) and kept.pnl.sum() > best_tot:
                            best, best_tot = q, kept.pnl.sum()
                    if best is not None:
                        kept = val[val.p >= best]
                        trade_rows.append({"task": tname, "fset": fs, "model": mname, "period": "validation", "rule": f"keep p>={best}",
                                           "n": len(kept), "avg": round(kept.pnl.mean(), 1) if len(kept) else np.nan,
                                           "total": round(kept.pnl.sum(), 0), "unfiltered_avg": round(val.pnl.mean(), 1)})
                        row["chosen"] = f"keep p>={best}"; row["val_trade_n"] = len(kept)
                        row["val_trade_avg"] = round(kept.pnl.mean(), 1) if len(kept) else np.nan
                        row["val_unfiltered_avg"] = round(val.pnl.mean(), 1)
                rows.append(row)
                # importance & stability on the ALL feature set only
                if fs == "ALL":
                    imp_store[(tname, mname)] = perm_importance(disc, folds, X, fcols, mname, reps=2)
                    if mname in ("L1", "L1CV"):
                        stab_store[(tname, mname)] = l1_stability(folds, fcols)
                    else:
                        jac, freq = gbm_stability(folds, fcols)
                        stab_store[(tname, mname)] = (jac, freq)
                print(f"{tname:16s} {fs:7s} {mname:3s} wfAUC={s_wf.get('auc')} valAUC={s_val.get('auc')} "
                      f"chosen={row.get('chosen')} val_avg={row.get('val_trade_avg')}  {time.time()-t0:.0f}s", flush=True)
    # naive trading baselines: always long / always short at the same decision points
    for tname, d in tasks.items():
        kind = d.kind.iloc[0]
        if kind == "meta":
            for per, sub in [("wf_disc", d[(d.split == "discovery") & (d.date.str[:7] >= WF_START)]), ("validation", d[d.split == "validation"])]:
                trade_rows.append({"task": tname, "fset": "-", "model": "baseline", "period": per, "rule": "I2 unfiltered",
                                   "n": len(sub), "avg": round(sub.pnl.mean(), 1), "total": round(sub.pnl.sum(), 0)})
            continue
        spec = d.h.iloc[0] if kind == "dir" else d.x.iloc[0]
        for per, sub, nd in [("wf_disc", d[(d.split == "discovery") & (d.date.str[:7] >= WF_START)], nd_wf), ("validation", d[d.split == "validation"], nd_val)]:
            for dr, nm in [(0.99, "always long"), (0.01, "always short")]:
                tr = sim.run_signals(g, trade_signals(sub.assign(p=dr), kind, 0.0, spec))
                trade_rows.append({"task": tname, "fset": "-", "model": "baseline", "period": per, "rule": nm, **summ(tr, nd)})
    res = pd.DataFrame(rows); res.to_csv(os.path.join(OUT, "model_results.csv"), index=False)
    pd.DataFrame(trade_rows).to_csv(os.path.join(OUT, "trade_results.csv"), index=False)
    pd.to_pickle((imp_store, stab_store), os.path.join(OUT, "importance.pkl"))
    json.dump({"n_model_variants": n_variants, "n_threshold_variants": len(DELTAS), "tasks": list(tasks)},
              open(os.path.join(OUT, "variants.json"), "w"))
    print("done", n_variants, f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
