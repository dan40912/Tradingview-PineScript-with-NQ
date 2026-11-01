"""建立每日快取。加上 --yahoo 會把 data/*_1m_yahoo_20261016-20261030.csv.gz（10/20～10/30）接在主檔後面。"""
import sys, pickle, eng
ext = "--yahoo" in sys.argv
for sym, sc in (("NQ", 1.0), ("ES", 3.7)):
    ds = eng.build(sym, sc, f"{eng.REPO}/data/{sym}_1m_yahoo_20261016-20261030.csv.gz" if ext else None)
    pickle.dump(ds, open(f"{sym}_days2.pkl", "wb")); print(sym, len(ds), "天，最後一天", ds[-1]["date"])
