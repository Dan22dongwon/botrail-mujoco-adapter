"""CLI:  python -m botrail_mujoco CELL [--sequences a,b] [--out DIR] [--record] [--view ex,ey,ez,tx,ty,tz]

CELL = "파일.py:함수" — 함수가 botrail Scene (또는 첫 원소가 Scene 인 튜플)을 돌려준다.
CELL = "파일.py"      — 스크립트를 그대로 실행해 처음 베이크하는 셀/시퀀스를 가로챈다 (예제 수정 불필요).
예)   python -m botrail_mujoco examples/machining/machine_tending_demo.py:bake --out out/mj_tending --record
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

from .adapter import MujocoAdapter, UnsupportedCell


def load_cell(spec: str):
    path, _, fn = spec.partition(":")
    path = Path(path).resolve()
    sys.path.insert(0, str(path.parent))
    mod_spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(mod_spec)
    from .allowances import record_allowances
    with record_allowances():
        mod_spec.loader.exec_module(mod)
        out = getattr(mod, fn or "build")()
    return out[0] if isinstance(out, tuple) else out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cell")
    ap.add_argument("--sequences", default=None, help="쉼표 구분 (기본: 셀의 모든 시퀀스)")
    ap.add_argument("--out", default="out/mujoco")
    ap.add_argument("--timestep", type=float, default=0.001)
    ap.add_argument("--payload", type=float, default=0.0, help="TCP 에 더할 질량 kg (모델에 빠진 그리퍼+공작물)")
    ap.add_argument("--no-torque-limits", action="store_true", help="모델 토크 한계 무시 (기구학·충돌만 검증)")
    ap.add_argument("--plan-only", action="store_true",
                    help="동역학 없이 botrail 계획 동작만 plan.json/.bin 으로 (기존 결과 폴더에 추가할 때)")
    ap.add_argument("--record", action="store_true", help="out/video.mp4")
    ap.add_argument("--view", default="2.9,-3.4,2.6,0.2,0.2,1.0")
    args = ap.parse_args()
    if args.record:  # 오프스크린 렌더링이 필요할 때만 (EGL 없는 CI 에서도 import 되도록)
        os.environ.setdefault("MUJOCO_GL", "egl")
    v = [float(x) for x in args.view.split(",")]
    t0 = time.time()
    rk, seqs = {}, args.sequences.split(",") if args.sequences else None
    if ":" in args.cell:
        scene = load_cell(args.cell)
    else:
        from .capture import capture_from_script
        try:
            cap = capture_from_script(args.cell)
        except LookupError as e:
            print(f"[no-sequence] {e}")
            sys.exit(4)
        scene, rk = cap.scene, cap.kwargs
        seqs = seqs or cap.sequences
        print(f"[capture] sequences={cap.sequences} kwargs={ {k: v for k, v in rk.items() if k != 'physics'} }")
    try:
        ad = MujocoAdapter(scene, seqs, args.out, timestep=args.timestep, view=(v[:3], v[3:]),
                           tcp_payload_kg=args.payload, rollout_kwargs=rk,
                           torque_limits=not args.no_torque_limits)
        ad.prepare()
    except UnsupportedCell as e:
        print(f"[unsupported] {e}")
        sys.exit(3)
    t1 = time.time()
    if args.plan_only:
        ad.export_plan()
        ad.export_isaac_job()
        print(f"[time] prepare {t1 - t0:.1f}s · plan {time.time() - t1:.1f}s → {args.out}/plan.json")
        return
    rep = ad.run(record=args.record)
    ad.export_isaac_job()
    t2 = time.time()
    show = {k: rep[k] for k in ("engine", "sequences", "cycle_s", "tcp_payload_kg", "botrail_min_clearance_mm", "max_tcp_err_mm",
                                "worst_t", "mean_tcp_err_mm", "max_joint_err_rad", "verdict", "warnings", "collisions", "pregrasp_contacts", "robots")}
    print(json.dumps(show, indent=2, ensure_ascii=False))
    print(f"[time] prepare {t1 - t0:.1f}s · mujoco {t2 - t1:.1f}s (sim {rep['cycle_s']:.1f}s) → {args.out}")


if __name__ == "__main__":
    main()
