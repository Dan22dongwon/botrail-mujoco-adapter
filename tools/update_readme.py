"""survey.json → README 의 <!-- SURVEY --> ... <!-- /SURVEY --> 구간에 호환성 표를 쓴다.

  python tools/update_readme.py out/survey/survey.json
"""
import collections
import json
import re
import sys
from pathlib import Path

LABEL = {
    "ok": "✅ 통과", "ok-collision": "⚠️ 판정 차이", "ok-tracking": "⚠️ 추종 실패", "unsupported": "⛔ 미지원", "external-control": "⛔ 외부 정책 필요",
    "no-sequence": "— 시퀀스 없음", "error": "❌ 오류", "timeout": "❌ 시간 초과", "missing-dep": "❌ 의존성",
    "botrail-plan": "❌ botrail 계획 실패",
}
ORDER = list(LABEL)

# 판정 차이로 남은 예제의 원인 (수동 분석 결과)
NOTES = {
    "assembly/cover_bolting_demo.py": "체결 직후 비트가 나사 구멍에 남아 있는 구간 — 커버의 구멍이 충돌 모델에서 꽉 찬 박스",
    "welding/nimak_spot_welding_demo.py": "평행 링크(닫힌 고리) 로봇을 트리+미믹 제약으로 근사 — 동작 중 링크가 처져 건이 강판에 닿음",
}


def main():
    rows = json.loads(Path(sys.argv[1]).read_text())
    c = collections.Counter(r["status"] for r in rows)
    conv = c["ok"] + c["ok-collision"] + c["ok-tracking"]
    rest = ", ".join(f"{LABEL[k]} {c[k]}" for k in ORDER if not k.startswith("ok") and c[k])
    head = (f"botrail 0.13 예제 {len(rows)}개를 수정 없이 돌린 결과 (`tools/survey_examples.py`): "
            f"**{conv}개 변환·실행** (통과 {c['ok']}, 판정 차이 {c['ok-collision']}, 추종 실패 {c['ok-tracking']}) · "
            f"{rest}.\n\n")
    lines = ["| 예제 | 결과 | 사이클 s | TCP 오차 최대 mm | 비고 |", "|---|---|---:|---:|---|"]
    for r in sorted(rows, key=lambda r: (ORDER.index(r["status"]) if r["status"] in ORDER else 99, r["example"])):
        tcp = "" if r.get("max_tcp_err_mm") is None else f"{r['max_tcp_err_mm']:.2f}"
        cyc = "" if r.get("cycle_s") is None else f"{r['cycle_s']:.1f}"
        note = NOTES.get(r["example"], "" if r["status"].startswith("ok") else r["detail"])
        if r["status"].startswith("ok") and r.get("robots", 1) > 1:
            note = (f"로봇 {r['robots']}대. " + note).strip()
        lines.append(f"| `{r['example']}` | {LABEL.get(r['status'], r['status'])} | {cyc} | {tcp} | {note} |")
    block = "<!-- SURVEY -->\n" + head + "\n".join(lines) + "\n<!-- /SURVEY -->"
    readme = Path(__file__).resolve().parents[1] / "README.md"
    text = readme.read_text()
    text = re.sub(r"<!-- SURVEY -->.*?(<!-- /SURVEY -->|$)", lambda m: block, text, count=1, flags=re.S) \
        if "<!-- /SURVEY -->" in text else text.replace("<!-- SURVEY -->", block)
    readme.write_text(text)
    print(head.strip())


if __name__ == "__main__":
    main()
