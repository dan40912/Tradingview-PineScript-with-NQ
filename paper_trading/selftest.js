// 自我回測：引擎 vs v18 模擬器（harness.js）的逐筆結果
const E = require("./engine.js"), days = require("./days.json");
let n = 0, ok = 0, entryOk = 0, skipped = 0, tot = 0, totRef = 0;
const bad = [];
for (const d of days) {
  const bars = E.decode(d);
  for (const s of d.sig) {
    if (!s.plan) continue;
    if (s.m < 575 || s.m > 900) { skipped++; continue; }
    const b = bars.find(x => x.m === s.m);
    if (!b) { skipped++; continue; }
    n++;
    if (Math.abs(b.o - s.entry) < 1e-9) entryOk++;
    const t = E.runTrade(d, { dir: s.dir, entry: s.entry, entryM: s.m, tp: s.tp, sl: s.sl, tEnd: 749, qty: 10 });
    tot += t.pnl; totRef += s.pnl;
    if (Math.abs(t.pnl - s.pnl) < 0.5 && t.exitM === s.xm) ok++;
    else bad.push([d.td, E.hhmm(s.m), s.kind, s.why, s.pnl, t.why, Math.round(t.pnl), E.hhmm(t.exitM), s.xm == null ? "-" : E.hhmm(s.xm)]);
  }
}
console.log(`計畫內訊號（圖內）${n} 筆；進場價一致 ${entryOk}/${n}；出場時間＋損益一致 ${ok}/${n}；亞洲盤略過 ${skipped}`);
console.log(`引擎合計 $${tot.toFixed(0)}，v18 模擬器合計 $${totRef.toFixed(0)}`);
bad.slice(0, 12).forEach(r => console.log(r.join(" | ")));
