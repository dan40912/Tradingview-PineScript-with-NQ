// 端到端測試：在真的瀏覽器裡跑完一天的紙上交易流程
const { chromium } = require("playwright");
const http = require("http"), fs = require("fs"), path = require("path");
const ROOT = __dirname;
const srv = http.createServer((req, res) => {
  let p = decodeURIComponent(req.url.split("?")[0]); if (p === "/") p = "/index.html";
  const f = path.join(ROOT, p);
  if (p === "/index.html") {
    const body = '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"></head><body>' + fs.readFileSync(path.join(ROOT, "NQ紙上交易練習.html"), "utf8") + "</body></html>";
    res.writeHead(200, { "content-type": "text/html; charset=utf-8" }); return res.end(body);
  }
  if (!fs.existsSync(f)) { res.writeHead(404); return res.end(); }
  res.writeHead(200, { "content-type": f.endsWith(".json") ? "application/json" : f.endsWith(".js") ? "text/javascript" : "text/plain" }); res.end(fs.readFileSync(f));
}).listen(8765);
(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1360, height: 900 } });
  const errors = []; page.on("pageerror", e => errors.push(e.message)); page.on("console", m => { if (m.type() === "error") errors.push(m.text()); });
  await page.goto("http://localhost:8765/");
  await page.waitForSelector("#btnStart");
  await page.screenshot({ path: path.join(__dirname, "shot_pre.png") });
  let followed = false;
  for (let attempt = 0; attempt < 25 && !followed; attempt++) {
    if (attempt) { await page.click("#btnNew"); await page.waitForSelector("#btnStart"); }
    for (const c of await page.$$("[data-chk]")) await c.check();
    await page.click("#btnStart");
    await page.click("#btnPlay");            // 暫停
    for (let k = 0; k < 8; k++) {
      await page.click("#btnNextSig");
      if (await page.$("#btnFollow")) { await page.click("#btnFollow"); followed = true; break; }
      const txt = await page.textContent(".log");
      if (txt.includes("後面沒有訊號")) break;
      if (await page.$("#btnSkip")) await page.click("#btnSkip"), await page.click("#btnPlay");
    }
    if (!followed) { await page.click("#btnEnd"); }
  }
  console.log("followed a plan signal:", followed);
  await page.screenshot({ path: path.join(__dirname, "shot_play.png") });
  if (await page.isEnabled("#btnEnd")) await page.click("#btnEnd");
  await page.waitForSelector("#btnSave");
  const rows = await page.$$eval("#review table", ts => ts.map(t => [...t.querySelectorAll("tbody tr")].map(r => [...r.children].map(c => c.textContent.trim()))));
  console.log("你的單:", JSON.stringify(rows[0]));
  console.log("v18 訊號:", JSON.stringify(rows[1]));
  await page.fill("#note", "測試：照訊號下單");
  await page.click("#btnSave");
  await page.waitForFunction(() => document.querySelector("#saveMsg").textContent.includes("已存"));
  console.log("save:", await page.textContent("#saveMsg"));
  console.log("journal tiles:", (await page.$$eval("#journal .tile", ts => ts.map(t => t.textContent.replace(/\s+/g, " ").trim()))).join(" | "));
  await page.screenshot({ path: path.join(__dirname, "shot_review.png"), fullPage: true });
  const mob = await browser.newPage({ viewport: { width: 400, height: 860 } });
  await mob.goto("http://localhost:8765/"); await mob.waitForSelector("#btnStart");
  const ov = await mob.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  console.log("phone horizontal overflow px:", ov);
  console.log("errors:", errors.length ? errors : "none");
  await browser.close(); srv.close();
})().catch(e => { console.error(e); srv.close(); process.exit(1); });
