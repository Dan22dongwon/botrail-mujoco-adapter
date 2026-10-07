"""사이트 카드용 썸네일: 각 예제의 사이클 중간 장면을 3D 뷰어로 찍는다 (헤드리스 Chrome).

  botrail-mujoco-view out/survey --port 8790 &      # 뷰어 서버
  python tools/make_thumbnails.py out/site --base http://127.0.0.1:8790 -j 4
"""
import argparse
import json
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image


def shot(base, slug, dst, chrome):
    url = f"{base}/index.html?thumb=1&t=mid&data=data/{slug}/mj/web"
    with tempfile.TemporaryDirectory() as d:
        png = Path(d) / "s.png"
        subprocess.run([chrome, "--headless=new", "--no-sandbox", f"--user-data-dir={d}/p", "--use-angle=swiftshader",
                        "--enable-unsafe-swiftshader", "--hide-scrollbars", "--window-size=960,600",
                        "--virtual-time-budget=25000", f"--screenshot={png}", url],
                       capture_output=True, timeout=180)
        if png.exists():
            Image.open(png).convert("RGB").resize((640, 400), Image.LANCZOS).save(dst, quality=82)
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("site")
    ap.add_argument("--base", default="http://127.0.0.1:8790")
    ap.add_argument("-j", type=int, default=4)
    ap.add_argument("--chrome", default="google-chrome")
    a = ap.parse_args()
    site = Path(a.site)
    items = [x for x in json.loads((site / "data" / "index.json").read_text())["items"] if x.get("ready")]
    (site / "thumbs").mkdir(exist_ok=True)
    def one(x):
        slug = x["data"].split("/")[-1]
        ok = shot(a.base, slug, site / x["thumb"], a.chrome)
        print(("ok  " if ok else "FAIL") + " " + slug, flush=True)
    with ThreadPoolExecutor(a.j) as pool:
        list(pool.map(one, items))


if __name__ == "__main__":
    main()
