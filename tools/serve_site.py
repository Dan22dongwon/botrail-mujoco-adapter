"""갤러리 로컬 미리보기 서버 — HTTP Range 지원 (영상 탐색에 필요; python -m http.server 는 미지원).

  python tools/serve_site.py out/site [--port 8791] [--bind 0.0.0.0]
"""
import argparse
import functools
import os
import re
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class RangeHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        rng = self.headers.get("Range")
        path = self.translate_path(self.path)
        if not rng or not os.path.isfile(path):
            return super().send_head()
        m = re.match(r"bytes=(\d*)-(\d*)", rng)
        size = os.path.getsize(path)
        start = int(m.group(1)) if m and m.group(1) else 0
        end = int(m.group(2)) if m and m.group(2) else size - 1
        if m and not m.group(1) and m.group(2):  # bytes=-N (끝에서 N)
            start, end = max(0, size - int(m.group(2))), size - 1
        end = min(end, size - 1)
        if start > end:
            self.send_error(416)
            return None
        f = open(path, "rb")
        f.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        self._remaining = end - start + 1
        return f

    def copyfile(self, src, dst):
        n = getattr(self, "_remaining", None)
        if n is None:
            return super().copyfile(src, dst)
        while n > 0:
            b = src.read(min(1 << 16, n))
            if not b:
                break
            dst.write(b)
            n -= len(b)
        self._remaining = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--port", type=int, default=8791)
    ap.add_argument("--bind", default="127.0.0.1")
    a = ap.parse_args()
    h = functools.partial(RangeHandler, directory=a.dir)
    print(f"http://{a.bind}:{a.port}/", flush=True)
    ThreadingHTTPServer((a.bind, a.port), h).serve_forever()


if __name__ == "__main__":
    main()
