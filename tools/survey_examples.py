"""botrail 예제 전체에 어댑터를 돌려 호환성 표를 만든다.

  python tools/survey_examples.py EXAMPLES_DIR OUT_DIR [-j 4] [--timeout 900]
결과: OUT_DIR/survey.json, OUT_DIR/survey.md
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def classify(log: str, code: int) -> tuple[str, str]:
    if code == 0 and '"verdict": "pass"' in log:
        return "ok", "통과"
    if code == 0 and '"verdict": "collision"' in log:
        return "ok-collision", "변환·실행됨, MuJoCo 에서 충돌 검출"
    if code == 0 and '"verdict": "tracking"' in log:
        return "ok-tracking", "변환·실행됨, MuJoCo 에서 계획을 못 따라감 (TCP 오차 > 허용치)"
    pats = [
        (r"\[no-sequence\]", "no-sequence", "시퀀스 베이크 없음 (계획/물리/스윕/RL/내보내기 데모)"),
        (r"\[unsupported\] (.+)", "unsupported", "{0}"),
        (r"TIMEOUT", "timeout", "시간 초과"),
        (r"policy `[^`]+` is not registered|waiting in step 0 \(`wait`\)", "external-control",
         "외부 제어(RL 정책) 입력을 기다리는 시퀀스 — 정책 없이 단독 실행 불가"),
        (r"planning failed[^\n]*", "botrail-plan", "botrail 롤아웃 중 계획 실패"),
        (r"ModuleNotFoundError: No module named '([\w.]+)'", "missing-dep", "의존성 없음: {0}"),
    ]
    for p, k, msg in pats:
        m = re.search(p, log)
        if m:
            return k, msg.format(*m.groups())
    last = [l for l in log.strip().splitlines() if l.strip()][-1:] or ["?"]
    return "error", last[0][:200]


def run_one(path: Path, out: Path, timeout: int):
    name = str(path.relative_to(path.parents[1]))
    od = (out / name.replace("/", "__").removesuffix(".py")).resolve()
    od.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    # 예제가 현재 폴더에 쓰는 산출물(USD, BOM 등)이 저장소를 더럽히지 않도록 결과 폴더 안에서 실행
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    try:
        p = subprocess.run([sys.executable, "-m", "botrail_mujoco", str(path.resolve()), "--out", str(od / "mj")],
                           capture_output=True, text=True, timeout=timeout, cwd=od, env=env)
        log, code = p.stdout + p.stderr, p.returncode
    except subprocess.TimeoutExpired as e:
        log, code = (e.stdout or "") + str(e.stderr or "") + "\nTIMEOUT", -1
        log = log if isinstance(log, str) else log.decode(errors="ignore")
    od.mkdir(parents=True, exist_ok=True)
    (od / "log.txt").write_text(log)
    kind, why = classify(log, code)
    rep = {}
    if (od / "mj" / "report.json").exists() and kind.startswith("ok"):
        r = json.loads((od / "mj" / "report.json").read_text())
        rep = {k: r.get(k) for k in ("cycle_s", "max_tcp_err_mm", "mean_tcp_err_mm", "verdict")}
        rep["collisions"] = len(r.get("collisions", {}))
        rep["moving"] = len(r.get("moving_objects", []))
        rep["robots"] = len(r.get("robots", {}))
        rep["warnings"] = len(r.get("warnings", []))
        rep["plan_source"] = r.get("plan_source")
    res = dict(example=name, status=kind, detail=why, seconds=round(time.time() - t0, 1), **rep)
    print(json.dumps(res, ensure_ascii=False), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("examples")
    ap.add_argument("out")
    ap.add_argument("-j", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--only", nargs="*", default=None, help="이 예제들만 (예: assembly/cover_bolting_demo.py)")
    a = ap.parse_args()
    ex = sorted(p for p in Path(a.examples).glob("*/*.py") if not p.name.startswith("_"))
    if a.only:
        ex = [p for p in ex if str(p.relative_to(p.parents[1])) in a.only]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(a.j) as pool:
        results = list(pool.map(lambda p: run_one(p, out, a.timeout), ex))
    (out / "survey.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    rows = ["| 예제 | 결과 | 내용 | 사이클 s | TCP 오차 최대 mm |", "|---|---|---|---|---|"]
    for r in results:
        rows.append(f"| {r['example']} | {r['status']} | {r['detail']} | {r.get('cycle_s', '')} | "
                    f"{'' if r.get('max_tcp_err_mm') is None else round(r['max_tcp_err_mm'], 2)} |")
    (out / "survey.md").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
