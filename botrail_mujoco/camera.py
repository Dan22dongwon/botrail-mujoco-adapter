"""세 엔진 영상(botrail 스튜디오 · MuJoCo · Isaac RTX)이 같은 화면 구도를 쓰도록 하는 공용 카메라 규칙.
pxr 와 표준 라이브러리만 쓴다 (Isaac 파이썬에서도 import 가능)."""
import json
from pathlib import Path

VFOV_DEG = 36.0  # 웹 3D 뷰어와 같은 세로 화각


def cell_view(usd_path):
    """셀 USD 의 보이는 형상 경계 상자(50 m 넘는 바닥 등 제외)에서 (eye, target) 을 정한다. 웹 뷰어 frame() 과 같은 규칙."""
    from pxr import Usd, UsdGeom, Gf
    stage = Usd.Stage.Open(str(usd_path)) if not hasattr(usd_path, "Traverse") else usd_path
    bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    box = Gf.Range3d()
    for p in stage.Traverse():
        if p.IsA(UsdGeom.Gprim) and UsdGeom.Imageable(p).ComputeVisibility() != UsdGeom.Tokens.invisible:
            r = bb.ComputeWorldBound(p).ComputeAlignedRange()
            if not r.IsEmpty() and r.GetSize().GetLength() < 50 and r.GetMin()[2] > -5:
                box.UnionWith(r)
    if box.IsEmpty():
        c, rad = (0.0, 0.0, 0.5), 2.0
    else:
        m = box.GetMidpoint()
        c, rad = (m[0], m[1], m[2]), max(1.2, box.GetSize().GetLength() / 2)
    eye = (c[0] + rad * 1.3, c[1] - rad * 1.6, c[2] + rad * 1.2)
    return [round(v, 4) for v in eye], [round(v, 4) for v in c]


def view_for(job_dir):
    """결과 폴더의 view.json (없으면 cell_physics.usda 로 계산해 저장)"""
    job_dir = Path(job_dir)
    f = job_dir / "view.json"
    if f.exists():
        v = json.loads(f.read_text())
        return v["eye"], v["target"]
    eye, target = cell_view(job_dir / "cell_physics.usda")
    f.write_text(json.dumps(dict(eye=eye, target=target, vfov_deg=VFOV_DEG)))
    return eye, target
