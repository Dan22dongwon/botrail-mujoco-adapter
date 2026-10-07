"""결과를 웹으로 보기:  botrail-mujoco-view OUT_DIR [--port 8790] [--host 127.0.0.1]

OUT_DIR 이 한 셀의 결과(web.json)면 3D 뷰어, 예제 일괄 시험 폴더(survey.json)면 갤러리.
/            → 뷰어 또는 갤러리 (botrail_mujoco/viewer/)
/data/...    → OUT_DIR
"""
import argparse
import functools
import http.server
from pathlib import Path

VIEWER = Path(__file__).resolve().parent / "viewer"


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, out: Path, **kw):
        self.out = out
        super().__init__(*a, directory=str(VIEWER), **kw)

    def translate_path(self, path):
        p = path.split("?", 1)[0].split("#", 1)[0]
        if p == "/" and (self.out / "survey.json").exists():
            return str(VIEWER / "gallery.html")
        if p.startswith("/data/"):
            target = (self.out / p[len("/data/"):]).resolve()
            if self.out in target.parents or target == self.out:
                return str(target)
            return str(self.out / "__forbidden__")
        return super().translate_path(path)

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out")
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    if not (out / "web.json").exists() and not (out / "survey.json").exists():
        raise SystemExit(f"{out} 에 web.json(셀 결과)도 survey.json(일괄 시험 결과)도 없습니다")
    srv = http.server.ThreadingHTTPServer((a.host, a.port), functools.partial(Handler, out=out))
    kind = "갤러리" if (out / "survey.json").exists() else "3D 뷰어"
    print(f"MuJoCo {kind}: http://{a.host}:{a.port}/  (데이터 {out})  Ctrl-C 로 종료")
    srv.serve_forever()


if __name__ == "__main__":
    main()
