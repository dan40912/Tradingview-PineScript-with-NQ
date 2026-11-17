// 用 engine.js 跑 days_full.json：(1) 和 backtest/ 的凍結逐筆紀錄核對 (2) 算每種策略每年的統計 → stats.json
// 用法：node strategy_map/build_stats.js
const fs = require('fs'), path = require('path');
const HERE = __dirname, ROOT = path.join(HERE, '..');
const E = require(path.join(HERE, 'engine.js'));
const days = JSON.parse(fs.readFileSync(path.join(HERE, 'days_full.json'), 'utf8'));
const csv = f => fs.readFileSync(path.join(ROOT, f), 'utf8').replace(/^﻿/, '').trim().split('\n').slice(1).map(l => l.split(','));

const results = days.map(r => ({ d: r.d, res: E.runAll(E.decode(r)) }));
const first = days[0].d, last = days[days.length - 1].d;

// ---- 核對 ----
function check(rows, id, label) {
  const want = {}; for (const r of rows) want[r[0]] = { e: +r[4], net: +r[10] };
  let same = 0, extra = [], miss = [];
  for (const { d, res } of results) {
    const t = res[id].trades[0], w = want[d];
    if (t && w && Math.abs(t.entry - w.e) < 0.01 && Math.abs(t.net - w.net) < 0.6) same++;
    else if (t && !w) extra.push(d); else if (w) miss.push(d);
  }
  const total = Object.keys(want).filter(d => d >= first && d <= last).length;
  console.log(`${label}：凍結紀錄 ${total} 筆，一致 ${same}，多出 ${extra.length}，缺少／不同 ${miss.length}`, extra.slice(0, 5), miss.slice(0, 5));
  return { label, total, same, extra: extra.length, miss: miss.length };
}
const ref = csv('backtest/reference_points_trades.csv');
const checks = [
  check(csv('backtest/i2_v1_trades.csv'), 'i2v1', 'I2 v1'),
  check(ref.filter(r => r[1].startsWith('前高')), 'pdhF', '前高假突破 空'),
  check(ref.filter(r => r[1].startsWith('前低')), 'pdlF', '前低假突破 多'),
];

// ---- 每年統計：[筆數, 止盈, 止損, 時間到, 合計$] ----
const stats = {};
for (const { d, res } of results) {
  const y = d.slice(0, 4);
  for (const s of E.STRATS) {
    const st = ((stats[s.id] = stats[s.id] || {})[y] = stats[s.id][y] || [0, 0, 0, 0, 0]);
    for (const t of res[s.id].trades) {
      st[0]++; st[t.reason === 'tp' ? 1 : t.reason === 'sl' ? 2 : 3]++; st[4] += t.net;
    }
  }
}
for (const k in stats) for (const y in stats[k]) stats[k][y][4] = Math.round(stats[k][y][4]);
fs.writeFileSync(path.join(HERE, 'stats.json'), JSON.stringify({ range: [first, last], days: days.length, checks, stats }));
for (const s of E.STRATS) console.log(s.id.padEnd(5), Object.entries(stats[s.id]).map(([y, v]) => `${y} ${v[0]}筆 每筆${v[0] ? Math.round(v[4] / v[0]) : 0}`).join(' | '));
