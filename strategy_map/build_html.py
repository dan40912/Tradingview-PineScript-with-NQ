"""把 engine.js、importer.js、days.json、stats.json 塞進 template.html → 策略進出場地圖.html（單一檔案，雙擊就能開）。"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
html = (HERE / "template.html").read_text(encoding="utf-8")
html = html.replace("/*__ENGINE__*/", (HERE / "engine.js").read_text(encoding="utf-8"))
html = html.replace("/*__IMPORTER__*/", (HERE / "importer.js").read_text(encoding="utf-8"))
html = html.replace("/*__DAYS__*/", (HERE / "days.json").read_text(encoding="utf-8"))
html = html.replace("/*__STATS__*/", (HERE / "stats.json").read_text(encoding="utf-8"))
# 單獨開檔時補上文件骨架（發佈成 Artifact 時平台會自己包）
page = '<!doctype html>\n<html lang="zh-Hant">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n</head>\n<body>\n' + html + '\n</body>\n</html>\n'
out = HERE / "策略進出場地圖.html"
out.write_text(page, encoding="utf-8")
print(out, f"{out.stat().st_size/1e6:.2f} MB")
