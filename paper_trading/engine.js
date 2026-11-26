// 紙上交易撮合引擎（頁面和回測共用）。研究用途，不是投資建議。
// 時間 m：美東分鐘數（420 = 07:00）。bars: [m-420, o, h, l, c, v]，價格是 (價-base)*4 的整數。
(function (root) {
  const PT_MNQ = 2;          // MNQ 每點 $2
  const COST_MNQ = 4.8;      // 每口 MNQ 來回成本（= 1 口 NQ $48 ÷ 10）

  function decode(day) {
    if (day._dec) return day._dec;
    const b = day.base;
    day._dec = day.bars.map(r => ({ m: r[0] + 420, o: b + r[1] / 4, h: b + r[2] / 4, l: b + r[3] / 4, c: b + r[4] / 4, v: r[5] }));
    return day._dec;
  }

  function hhmm(m) {
    const h = Math.floor(m / 60), mm = m % 60;
    return String(h).padStart(2, "0") + ":" + String(mm).padStart(2, "0");
  }

  // 在第 i 根 K 棒上推進一筆持倉；回傳 null（還在場內）或 {exitM, exit, why}
  // 規則和 v18 模擬器一樣：同一根先檢查止損、再檢查止盈；到 tEnd 那根收盤平倉。
  function stepTrade(t, bar) {
    if (bar.m < t.entryM) return null;
    const d = t.dir;
    if (t.sl != null && (d === 1 ? bar.l <= t.sl : bar.h >= t.sl)) return { exitM: bar.m, exit: t.sl, why: "止損" };
    if (t.tp != null && (d === 1 ? bar.h >= t.tp : bar.l <= t.tp)) return { exitM: bar.m, exit: t.tp, why: "止盈" };
    if (bar.m >= t.tEnd) return { exitM: bar.m, exit: bar.c, why: "時間到" };
    return null;
  }

  function pnlUSD(t) {
    return (t.exit - t.entry) * t.dir * PT_MNQ * t.qty - COST_MNQ * t.qty;
  }

  // 整筆跑完（回測用）
  function runTrade(day, t) {
    const bars = decode(day);
    for (const bar of bars) {
      const r = stepTrade(t, bar);
      if (r) return Object.assign(t, r, { pnl: pnlUSD(Object.assign({}, t, r)) });
    }
    const last = bars[bars.length - 1];
    return Object.assign(t, { exitM: last.m, exit: last.c, why: "收盤", pnl: pnlUSD(Object.assign({}, t, { exit: last.c })) });
  }

  // 建議口數：每筆最多虧 risk 美元，止損 slPts 點
  function sizing(risk, slPts, vm) {
    const raw = Math.floor(risk / (slPts * PT_MNQ + COST_MNQ));
    const halved = vm >= 2 ? Math.floor(raw / 2) : raw;
    return { raw, qty: Math.max(0, halved), halved: vm >= 2 };
  }

  // I2 方向：09:30 開盤在前日價值區外
  function i2Direction(day) {
    const bars = decode(day);
    const open = bars.find(b => b.m === 570);
    if (!open) return { dir: 0, text: "沒有 09:30 開盤價" };
    const { VAH, VAL } = day.lv;
    if (open.o > VAH) return { dir: -1, open: open.o, text: "開在 VAH 之上 → 等回到價值區才空" };
    if (open.o < VAL) return { dir: 1, open: open.o, text: "開在 VAL 之下 → 等回到價值區才多" };
    return { dir: 0, open: open.o, text: "開在價值區內 → I2 今天不做" };
  }

  // 下單時的規則檢查（不擋單，只記錄；當天停損鎖由頁面處理）
  function ruleCheck(day, order, ctx) {
    const v = [];
    const m = order.entryM;
    const plan = day.sig.find(s => s.plan && s.dir === order.dir && Math.abs(s.m - m) <= 2);
    if (!plan) v.push("不是計畫內訊號");
    if (m < 575 || m > 690) v.push("不在 09:35～11:30");
    if (ctx.suggest > 0 && order.qty > ctx.suggest) v.push("口數超過建議（" + ctx.suggest + " 口）");
    if (order.sl == null) v.push("沒有止損");
    else if (Math.abs(order.entry - order.sl) > 145.01) v.push("止損比 145 點寬");
    return { plan: plan || null, violations: v };
  }

  const api = { decode, hhmm, stepTrade, pnlUSD, runTrade, sizing, i2Direction, ruleCheck, PT_MNQ, COST_MNQ };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.PaperEngine = api;
})(typeof window !== "undefined" ? window : globalThis);
