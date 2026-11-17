exec(open('entry_test.py').read().split("P=['1探索'")[0])
# 確認進場：價格先往有利方向走 k 點（停損買單 entry+k），成交後 TP/SL 從成交價算；有效 30 分鐘
def simc(r,k,win=30):
    i=idx[np.datetime64(r.et.tz_convert(None),'m').astype(np.int64)]; asia=r.kind in ASIA; D=deadline(i,asia); s=r.dir
    trig=r.entry+s*k; f=None
    for j in range(i,min(D,i+win)+1):
        if (s==1 and H[j]>=trig) or (s==-1 and L[j]<=trig): px=max(trig,O[j]) if s==1 else min(trig,O[j]); f=j; break
    if f is None: return None
    tp=px+s*55; sl=px-s*145
    for j in range(f,D+1):
        if (s==1 and L[j]<=sl) or (s==-1 and H[j]>=sl): return -145*20-48
        if j>f and ((s==1 and H[j]>=tp) or (s==-1 and L[j]<=tp)): return 55*20-48
    return s*(C[D]-px)*20-48
P=['1探索','2驗證','3測試','4近期10/1起']
for k in [5,10,20,30]:
    A['v']=[simc(r,k) for r in A.itertuples()]
    print(f'先漲 {k} 點才追', ' | '.join(f"{p} {len(A[A.P==p].v.dropna())}筆 {(A[A.P==p].v.dropna()>0).mean():.0%} {A[A.P==p].v.dropna().sum():+,.0f}" for p in P))
