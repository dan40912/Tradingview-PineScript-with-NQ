"""檢查 6（和基準合併）＋ B 的記帳主張（a）＋ 同時持倉。全部用 rt_core 自己的模擬器與風控。"""
import os, numpy as np, pandas as pd
import rt_core as R
out = []
def P(s=""): print(s); out.append(str(s))
B = R.base_trades()
P(f"自己模擬的基準候選點 {len(B)} 筆；和 wr_table pnl55 不一致的筆數：{int(((B.pnl - B.pnl55).abs() > 1).sum())}")
B.to_csv(os.path.join(R.OUT, "rt_base_candidates.csv"), index=False)
bs = R.book(B, "sched"); br = R.book(B, "real"); b1 = R.book(B, "real", single=True)
P(R.line(bs, "基準 books17 記帳(預定出場才算)")); P(R.line(br, "基準 平倉即記帳(Pine 式)")); P(R.line(b1, "基準 平倉即記帳＋單一部位"))
# 同時持倉：平倉即記帳下，有幾筆是在另一筆還沒平倉時進場？方向相反的？
def overlaps(T):
    n = n_opp = 0; ex = []
    for d, x in T.groupby("date"):
        x = x.sort_values("e")
        for a in range(len(x)):
            for b in range(a):
                if x.iloc[b].x >= x.iloc[a].e:
                    n += 1
                    if x.iloc[b].d != x.iloc[a].d:
                        n_opp += 1; ex.append(d)
    return n, n_opp, ex
for nm, T in [("books17", bs), ("Pine 式", br)]:
    n, no, ex = overlaps(T); P(f"{nm}：進場時另一筆仍持倉 {n} 筆，其中方向相反（實際帳戶會互相抵銷）{no} 筆 {ex[:8]}")
# 和 books17 比：Pine 式多出來的筆數
extra = br.merge(bs[["date", "e", "key"]], on=["date", "e", "key"], how="left", indicator=True)
ex = extra[extra._merge == "left_only"]
P(f"Pine 式比 books17 多出 {len(ex)} 筆，這些筆合計 {ex.pnl.sum():+,.0f}，每筆 {ex.pnl.mean():+.0f}，勝率 {100*(ex.pnl>0).mean():.0f}%")
miss = bs.merge(br[["date", "e", "key"]], on=["date", "e", "key"], how="left", indicator=True); miss = miss[miss._merge == "left_only"]
P(f"books17 有但 Pine 式沒有：{len(miss)} 筆 合計 {miss.pnl.sum():+,.0f}")

# ---- C1、TD10 合併 ----
F = R.td_features(960); dv, _ = R.td_signal(F, 0.8, 3)
c1 = R.run(R.c1_signals()); c1["src"] = "C1"; c1["xsched"] = c1.x
td = R.run(R.td_sigs(dv)); td["src"] = "TD10"; td["xsched"] = td.x
def wk(T):
    w = T.assign(w=pd.to_datetime(T.date).dt.to_period("W")).groupby("w").pnl.sum()
    return w.min(), str(w.idxmin())
def dd_len(T):
    dy = T.groupby("date").pnl.sum().sort_index(); c = dy.cumsum().values; pk = np.maximum.accumulate(np.concatenate([[0], c]))[1:]
    under = c < pk - 1e-9; best = cur = 0; start = None; bstart = bend = None; dts = list(dy.index)
    for k, u in enumerate(under):
        if u:
            if cur == 0: start = k
            cur += 1
            if cur > best: best, bstart, bend = cur, start, k
        else:
            cur = 0
    if bstart is None: return 0, "", ""
    return best, dts[max(bstart - 1, 0)], dts[bend]
rows = []
def rep(nm, T):
    P(R.line(T, nm)); w, wd = wk(T); n, a, b = dd_len(T)
    P(f"    最差一週 {w:+,.0f}（{wd}）；最長水下 {n} 個交易日（{a}→{b}）；最差一天 {T.groupby('date').pnl.sum().min():+,.0f}；筆數來源 {T.src.value_counts().to_dict()}")
    s = {p: R.stats(T[T.P == p]) for p in R.PER}; a_ = R.stats(T)
    rows.append(dict(name=nm, **{p: s[p]["tot"] for p in R.PER}, 全期=a_["tot"], 回撤=a_["mdd"], 比=round(a_["tot"] / -a_["mdd"], 1), 最差週=w, 水下日=n))
P("\n=== 合併（每日 $3,000：已實現 − 持倉最壞 − 本筆最壞 < −3,000 不下；同一分鐘基準優先）")
for acc in ["sched", "real"]:
    for single in [False, True]:
        tag = f"[{'books17記帳' if acc=='sched' else 'Pine式記帳'}{'＋單一部位' if single else ''}]"
        rep(f"基準 {tag}", R.book(B, acc, single=single))
        rep(f"基準＋C1 {tag}", R.book(pd.concat([B, c1]), acc, single=single))
        rep(f"基準＋TD10 {tag}", R.book(pd.concat([B, td]), acc, single=single))
        rep(f"基準＋C1＋TD10 {tag}", R.book(pd.concat([B, c1, td]), acc, single=single))
pd.DataFrame(rows).to_csv(os.path.join(R.OUT, "rt_combo.csv"), index=False)
# 增量：合併後多賺的錢，來自幾天？
b0 = R.book(B, "sched"); bc = R.book(pd.concat([B, c1]), "sched"); bt = R.book(pd.concat([B, td]), "sched")
for nm, X in [("C1", bc), ("TD10", bt)]:
    d0 = b0.groupby("date").pnl.sum(); d1 = X.groupby("date").pnl.sum()
    inc = d1.subtract(d0, fill_value=0).sort_values(ascending=False)
    P(f"合併 {nm} 的增量 {inc.sum():+,.0f}：前 5 天 {inc.head(5).sum():+,.0f}，拿掉前 5 天 {inc.iloc[5:].sum():+,.0f}；被擋掉的基準單 {len(b0) - int((X.src=='base').sum())} 筆")
open(os.path.join(R.OUT, "rt_combo.txt"), "w").write("\n".join(out))
