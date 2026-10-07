#!/usr/bin/env bash
# out/site → 공개 GitHub Pages 저장소(botrail-mujoco-gallery) 로 배포
#   tools/build_site.py out/survey out/site && tools/make_thumbnails.py out/site 를 먼저 실행
set -euo pipefail
cd "$(dirname "$0")/.."
DEST="${DEST:-$HOME/botrail-mujoco-gallery}"
[ -d "$DEST/.git" ] || { echo "$DEST 가 없습니다: gh repo clone Dan22dongwon/botrail-mujoco-gallery $DEST"; exit 1; }
rsync -a --delete out/site/data/ "$DEST/data/"
rsync -a --delete out/site/thumbs/ "$DEST/thumbs/"
rsync -a --delete out/site/video/ "$DEST/video/"
python3 - "$DEST" <<'PY'
import sys
from pathlib import Path
import time
dest = Path(sys.argv[1]); body = Path("botrail_mujoco/viewer/site.html").read_text().replace("__BUILD__", time.strftime("%Y%m%d%H%M%S"))
title, rest = body.split("\n", 1)
old = (dest / "index.html").read_text()
head = old[:old.index("<body>") + len("<body>")]
head = head.replace(head[head.index("<title>"):head.index("</title>") + 8], title)
(dest / "index.html").write_text(head + "\n" + rest + "\n</body>\n</html>\n")
PY
cd "$DEST" && git add -A && git commit -qm "Update gallery" && git push -q && echo "pushed → https://dan22dongwon.github.io/botrail-mujoco-gallery/"
