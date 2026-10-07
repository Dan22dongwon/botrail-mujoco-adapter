"""MuJoCo 결과(web.json/.bin 의 바디 포즈)를 MuJoCo 자체 렌더러로 영상화 — 갤러리 'MuJoCo' 모드용.
시뮬레이션을 다시 돌리지 않는다: 기록된 바디 포즈로 지오메트리 위치를 직접 채워 mjv 장면을 만든다.

  MUJOCO_GL=egl python tools/mujoco_render.py OUT/mj OUT.mp4 [--fps 15]
카메라는 botrail_mujoco.camera 공통 구도 (botrail 스튜디오 · Isaac RTX 영상과 같은 시점).
"""
import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")


def quat_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mj_dir")
    ap.add_argument("out")
    ap.add_argument("--fps", type=float, default=15.0)
    a = ap.parse_args()
    import mujoco
    from botrail_mujoco.camera import view_for

    mj = Path(a.mj_dir)
    model = mujoco.MjModel.from_xml_path(str(next((mj / "mjcf").glob("*.xml"))))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    meta = json.loads((mj / "web.json").read_text())
    raw = (mj / meta.get("bin", "web.bin")).read_bytes()
    T = np.array(meta["t"])
    mv = meta["moving"]
    P = np.frombuffer(raw, np.float32, meta["pos"][1], meta["pos"][0]).reshape(len(T), len(mv), 3)
    Q = np.frombuffer(raw, np.float32, meta["quat"][1], meta["quat"][0]).reshape(len(T), len(mv), 4)
    static = {int(b): (np.array(s["p"]), np.array(s["q"])) for b, s in meta["static"].items()}
    col = {b: k for k, b in enumerate(mv)}

    W, H = model.vis.global_.offwidth, model.vis.global_.offheight
    renderer = mujoco.Renderer(model, H, W)
    eye, target = (np.array(v, float) for v in view_for(mj))
    cam = mujoco.MjvCamera()
    f = target - eye  # MuJoCo 방위각·고도 = 보는 방향
    cam.lookat[:] = target
    cam.distance = float(np.linalg.norm(f))
    cam.azimuth = math.degrees(math.atan2(f[1], f[0]))
    cam.elevation = math.degrees(math.asin(f[2] / np.linalg.norm(f)))

    gb = model.geom_bodyid
    gpos, gquat = model.geom_pos.copy(), model.geom_quat.copy()
    gR = np.array([quat_mat(q) for q in gquat])
    enc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                            "-r", str(a.fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "28",
                            "-preset", "veryfast", "-movflags", "+faststart", a.out], stdin=subprocess.PIPE)
    n = int(T[-1] * a.fps) + 1
    for k in range(n):
        t = k / a.fps
        i = min(int(np.searchsorted(T, t, side="right")) - 1, len(T) - 2)
        i = max(i, 0)
        w = float(np.clip((t - T[i]) / max(T[i + 1] - T[i], 1e-9), 0, 1))
        bp = data.xpos.copy()
        bR = data.xmat.reshape(-1, 3, 3).copy()
        for b in range(model.nbody):
            if b in col:
                j = col[b]
                p = (1 - w) * P[i, j] + w * P[i + 1, j]
                q0, q1 = Q[i, j], Q[i + 1, j]
                q = (1 - w) * q0 + w * (q1 if np.dot(q0, q1) >= 0 else -q1)
                bp[b], bR[b] = p, quat_mat(q / np.linalg.norm(q))
            elif b in static:
                bp[b], bR[b] = static[b][0], quat_mat(static[b][1])
        data.geom_xpos[:] = bp[gb] + np.einsum("gij,gj->gi", bR[gb], gpos)
        data.geom_xmat[:] = np.einsum("gij,gjk->gik", bR[gb], gR).reshape(-1, 9)
        data.xpos[:], data.xmat[:] = bp, bR.reshape(-1, 9)
        renderer.update_scene(data, cam)
        enc.stdin.write(renderer.render().tobytes())
    enc.stdin.close()
    enc.wait()
    print(f"wrote {a.out} · {n} frames", flush=True)


if __name__ == "__main__":
    main()
