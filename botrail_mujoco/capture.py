"""예제 스크립트를 수정 없이 실행해, 처음 베이크되는 셀(Scene)과 시퀀스를 가로챈다.

botrail 예제 대부분은 셀을 돌려주는 함수 없이 main() 안에서 만들고 바로
`scene.simulate_sequence(s)(...)` 를 부른다. 그 호출을 잠시 바꿔치기해 (scene, 시퀀스, 인자)를 잡고 중단한다.
"""
from __future__ import annotations

import runpy
import sys
from dataclasses import dataclass, field
from pathlib import Path


class _Captured(BaseException):
    """BaseException: 예제/라이브러리의 `except Exception` 에 삼켜지지 않게."""


@dataclass
class Capture:
    scene: object
    sequences: list
    kwargs: dict = field(default_factory=dict)


MIN_CYCLE_S = 1.0  # 이보다 짧은 베이크는 본 사이클이 아니라 탐색용으로 본다
ROLLOUT_KW = ("dt", "max_duration", "plan_resolution", "scenario", "toolpath_spin", "physics")


def capture_from_script(path, argv=None) -> Capture:
    import botrail as bt
    path = Path(path).resolve()
    got: dict = {}
    cls = bt.Scene
    orig = {n: getattr(cls, n) for n in ("simulate_sequence", "simulate_sequences", "simulate_scenarios", "open_rollout")}
    orig_studio = bt.studio

    def grab(kind):
        def f(self, names, *a, **kw):
            seqs = [names] if isinstance(names, str) else list(names)
            pos = ("scenarios", "dt", "max_duration", "plan_resolution") if kind == "simulate_scenarios" else \
                ("dt", "max_duration", "plan_resolution", "scenario", "toolpath_spin", "physics")
            params = dict(zip(pos, a))
            params.update({k: v for k, v in kw.items() if k in ROLLOUT_KW})
            params = {k: v for k, v in params.items() if k in ROLLOUT_KW}
            if kind in ("simulate_sequence", "simulate_sequences"):
                # 실제로 베이크해 보고, 탐색용 짧은 베이크(예: 자세 확인용 probe)는 통과시킨다.
                tl = orig[kind](self, names, *a, **kw)
                if getattr(tl, "duration", MIN_CYCLE_S) < MIN_CYCLE_S:
                    return tl
            got["c"] = Capture(self, seqs, params)
            raise _Captured
        return f

    old_argv, old_path = sys.argv, list(sys.path)
    try:
        for n in orig:
            setattr(cls, n, grab(n))
        bt.studio = lambda *a, **k: None
        sys.argv = [str(path), *(argv or [])]
        sys.path.insert(0, str(path.parent))
        from .allowances import record_allowances
        try:
            with record_allowances():
                runpy.run_path(str(path), run_name="__main__")
        except _Captured:
            pass
        except SystemExit:
            pass
    finally:
        for n, f in orig.items():
            setattr(cls, n, f)
        bt.studio = orig_studio
        sys.argv, sys.path[:] = old_argv, old_path
    if "c" not in got:
        raise LookupError(f"{path.name}: 시퀀스 베이크(simulate_sequence/s, simulate_scenarios, open_rollout) 호출이 없음 — 시퀀스 없는 예제")
    return got["c"]


@dataclass
class StudioCapture:
    scene: object
    view: object = None  # 예제가 지정한 (eye, target)
    source: str = "studio"  # studio | sequences | scene


def studio_scene_from_script(path, prefer_sequences=False) -> StudioCapture:
    """botrail 이 그 예제를 스튜디오에 띄울 때의 장면을 그대로 얻는다.
    1) `--studio` 로 실행해 예제의 `bt.studio(scene, ...)` 호출을 가로챔 (예제가 고른 타임라인·물리 베이크 그대로)
    2) 없으면 시퀀스 베이크를 가로채 일괄 베이크 타임라인을 올림
    3) 그것도 없으면(도구용 예제) 내보내기·저장 호출 시점의 장면"""
    import botrail as bt
    path = Path(path).resolve()
    got: dict = {}
    orig_studio = bt.studio

    class _Server:  # block=False 로 띄우고 계속 진행하는 예제(애니메이션을 나중에 싣는 경우)용
        url = ""

        def stop(self):
            pass

    def fake_studio(scene, *a, **kw):
        got["c"] = StudioCapture(scene, kw.get("view"))
        if kw.get("block", True):
            raise _Captured
        return _Server()

    old_argv, old_path = sys.argv, list(sys.path)
    try:
        bt.studio = fake_studio
        sys.argv = [str(path), "--studio"]
        sys.path.insert(0, str(path.parent))
        from .allowances import record_allowances
        try:
            with record_allowances():
                runpy.run_path(str(path), run_name="__main__")
        except (_Captured, SystemExit):
            pass
        except Exception:
            if "c" not in got:  # 스튜디오를 띄운 뒤의 대기(input() 등)에서 난 오류는 무시
                raise
    finally:
        bt.studio = orig_studio
        sys.argv, sys.path[:] = old_argv, old_path
    if "c" in got and not prefer_sequences:
        return got["c"]
    try:
        cap = capture_from_script(path)
        try:
            tl = cap.scene.simulate_sequences(cap.sequences, **cap.kwargs)
        except ValueError:  # 외부 정책·신호를 기다리는 시퀀스(RL 환경)는 단독 베이크가 안 됨 → 셀만
            return got.get("c") or StudioCapture(cap.scene, None, "scene")
        cap.scene.show_timeline(tl)
        return StudioCapture(cap.scene, None, "sequences")
    except LookupError:
        if "c" in got:
            return got["c"]
    # 도구용 예제: 내보내기/저장하는 순간의 장면
    cls = bt.Scene
    names = [n for n in dir(cls) if n.startswith(("export_", "save_")) and callable(getattr(cls, n))]
    orig = {n: getattr(cls, n) for n in names}
    orig_cell = bt.export_cell

    def grab(n):
        def f(self, *a, **kw):
            got["c"] = StudioCapture(self, None, "scene")
            raise _Captured
        return f

    def grab_cell(scene, *a, **kw):
        got["c"] = StudioCapture(scene, None, "scene")
        raise _Captured

    try:
        for n in names:
            setattr(cls, n, grab(n))
        bt.export_cell = grab_cell
        sys.argv = [str(path)]
        sys.path.insert(0, str(path.parent))
        try:
            runpy.run_path(str(path), run_name="__main__")
        except (_Captured, SystemExit):
            pass
    finally:
        for n, f in orig.items():
            setattr(cls, n, f)
        bt.export_cell = orig_cell
        sys.argv, sys.path[:] = old_argv, old_path
    if "c" not in got:
        raise LookupError(f"{path.name}: 스튜디오에 올릴 장면을 찾지 못함")
    return got["c"]
