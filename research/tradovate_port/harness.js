// 模擬 Tradovate 執行環境：把 1 分K 一根一根餵給指標，輸出指標內部的交易紀錄。
// 用法：node harness.js <指標.js> <輸出.csv> <商品 NQ|ES> <資料1.csv.gz>[,<資料2.csv.gz>...]
// 環境變數：LIVE=1 每根重複呼叫 3 次（模擬即時盤）；SHIFT=60 時間戳改成 K 棒收盤時間並打開 tsIsClose。
const fs = require("fs"), zlib = require("zlib");
const [, , indPath, outPath, product, dataList] = process.argv;
const ind = require(require("path").resolve(indPath));
const shift = (+process.env.SHIFT || 0) * 1000;
const seen = new Set(), bars = [];
for (const p of dataList.split(",")) {
    const lines = zlib.gunzipSync(fs.readFileSync(p)).toString("utf8").split("\n");
    const hdr = lines[0].replace(/^﻿/, "").split(","), ix = (n) => hdr.indexOf(n);
    for (let k = 1; k < lines.length; k++) {
        if (!lines[k]) continue;
        const f = lines[k].split(","), key = f[ix("timestamp_utc")];
        if (seen.has(key)) continue;
        seen.add(key);
        const ts = new Date(new Date(key).getTime() + shift);
        const o = +f[ix("open")], h = +f[ix("high")], l = +f[ix("low")], c = +f[ix("close")], v = +f[ix("volume")];
        bars.push({ ts, timestamp: () => ts, open: () => o, high: () => h, low: () => l, close: () => c, volume: () => v, value: () => c });
    }
}
bars.sort((a, b) => a.ts - b.ts);
bars.forEach((b, j) => { const last = j === bars.length - 1; b.isLast = () => last; });
const history = { get: (j) => bars[j], size: () => bars.length };
const calc = new ind.calculator();
calc.props = {};
for (const [k, p] of Object.entries(ind.params)) calc.props[k] = p.def;
if (shift) calc.props.tsIsClose = true;
calc.chartDescription = { underlyingType: "MinuteBar", elementSize: 1, elementSizeUnit: "UnderlyingUnits", withHistogram: false };
calc.contractInfo = { product: product, contract: product + "Z6", tickSize: 0.25 };
calc.init();
let items = 0, noKey = 0;
for (let i = 0; i < bars.length; i++) {
    for (let r = 0; r < (process.env.LIVE ? 3 : 1); r++) {
        const out = calc.map(bars[i], i, history);
        if (r === 0 && out && out.graphics) { items += out.graphics.items.length; noKey += out.graphics.items.filter((it) => !it.key).length; }
    }
}
calc.cur = bars.length;
calc.step(bars[bars.length - 1], bars.length - 1);   // 最後一根當作已收盤
const iso = (ms) => (ms ? new Date(ms).toISOString() : "");
const rows = calc.log.map((t) => [t.kind, t.tier, t.dir, t.inPlan ? 1 : 0, JSON.stringify(t.skipWhy || ""), iso(t.entryMs), t.entry, iso(t.exitMs), t.exit, t.why || "", t.pnl].join(","));
fs.writeFileSync(outPath, "kind,tier,dir,inPlan,skipWhy,entry_utc,entry,exit_utc,exit,why,pnl\n" + rows.join("\n") + "\n");
console.log("bars", bars.length, "log rows", calc.log.length, "graphic items", items, "items without key", noKey);
