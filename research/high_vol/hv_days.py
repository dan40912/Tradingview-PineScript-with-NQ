import pandas as pd, numpy as np
a=pd.read_csv("tv/nq_full.csv.gz"); b=pd.read_csv("tv/nq.csv.gz")
d=pd.concat([a,b]).drop_duplicates("timestamp_utc").sort_values("timestamp_utc").reset_index(drop=True)
d["contract"]=d.contract.fillna("NQZ26"); d["t"]=pd.to_datetime(d.timestamp_utc); et=d.t.dt.tz_convert("America/New_York")
d["td"]=(et+pd.Timedelta(hours=6)).dt.strftime("%Y-%m-%d"); d["m"]=et.dt.hour*60+et.dt.minute
d.to_pickle("full_clean.pkl")
r=d[(d.m>=570)&(d.m<960)].groupby("td").agg(h=("high","max"),l=("low","min"),n=("m","size"))
r=r[r.n>=300]; r["rr"]=r.h-r.l; r["adr10"]=r.rr.rolling(10).mean().shift(1); r["vm"]=r.adr10/255.2
r.to_csv("daily_vm.csv")
for th in [1.5,1.8,2.0,2.5]:
    x=r[r.vm>=th]; print(f"波動倍數 ≥{th}: {len(x)} 天 | 2024: {(x.index<'2025').sum()}  2025: {((x.index>='2025')&(x.index<'2026')).sum()}  2026-01~09: {((x.index>='2026')&(x.index<'2026-10')).sum()}  10/01後: {(x.index>='2026-10').sum()}")
print(r[r.vm>=2.0].groupby(r[r.vm>=2.0].index.str[:7]).size().to_dict())
