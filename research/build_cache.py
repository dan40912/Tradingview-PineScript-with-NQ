import pickle, eng
for sym, sc in (("NQ", 1.0), ("ES", 3.7)):
    ds = eng.build(sym, sc); pickle.dump(ds, open(f"{sym}_days2.pkl", "wb")); print(sym, len(ds))
