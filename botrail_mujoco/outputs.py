"""어댑터 결과물: 웹 뷰어 데이터(web/mujoco.html 이 읽는 json+bin), mp4."""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def export_web(model, t, pos, quat, tcp_mm, steps, prefix: Path):
    """MuJoCo 가 계산한 바디 포즈 + 보이는 geom + 메쉬 → <prefix>.json / <prefix>.bin."""
    import mujoco
    geoms, bodies = [], set()
    for g in range(model.ngeom):
        if model.geom_group[g] == 3 or model.geom_rgba[g, 3] == 0:
            continue
        b = int(model.geom_bodyid[g])
        geoms.append(dict(b=b, type=int(model.geom_type[g]), size=np.round(model.geom_size[g], 5).tolist(),
                          pos=np.round(model.geom_pos[g], 5).tolist(), quat=np.round(model.geom_quat[g], 6).tolist(),
                          rgba=np.round(model.geom_rgba[g], 3).tolist(), mesh=int(model.geom_dataid[g])))
        bodies.add(b)
    chunks, off = [], 0

    def put(arr):
        nonlocal off
        buf = np.ascontiguousarray(arr).tobytes()
        chunks.append(buf)
        start, off = off, off + len(buf)
        return [start, int(np.asarray(arr).size)]

    meshes = {}
    for k in sorted({g["mesh"] for g in geoms if g["type"] == int(mujoco.mjtGeom.mjGEOM_MESH)}):
        va, vn = model.mesh_vertadr[k], model.mesh_vertnum[k]
        fa, fn = model.mesh_faceadr[k], model.mesh_facenum[k]
        uniq, inv = np.unique(np.round(model.mesh_vert[va:va + vn], 5), axis=0, return_inverse=True)
        f = inv.reshape(-1)[model.mesh_face[fa:fa + fn]]
        meshes[k] = dict(v=put(uniq.astype(np.float32)), f=put(f.astype(np.uint32)))
    moving = [b for b in sorted(bodies) if np.ptp(pos[:, b], axis=0).max() > 1e-6 or np.ptp(quat[:, b], axis=0).max() > 1e-6]
    static = {b: dict(p=np.round(pos[0, b], 5).tolist(), q=np.round(quat[0, b], 6).tolist()) for b in bodies if b not in moving}
    data = dict(t=np.asarray(t).tolist(), geoms=geoms, meshes=meshes, static=static, moving=moving,
                pos=put(pos[:, moving].astype(np.float32)), quat=put(quat[:, moving].astype(np.float32)),
                tcp_mm=list(tcp_mm), steps=steps, bin=prefix.name + ".bin")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".bin").write_bytes(b"".join(chunks))
    prefix.with_suffix(".json").write_text(json.dumps(data, separators=(",", ":")))


def write_video(frames, fps, path: Path):
    from PIL import Image
    with tempfile.TemporaryDirectory() as d:
        for i, f in enumerate(frames):
            Image.fromarray(f).save(f"{d}/{i:05d}.png")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{fps:.6f}", "-i", f"{d}/%05d.png",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-movflags", "+faststart", str(path)],
                       check=True)


def export_poses(t, pos, quat, prefix: Path):
    """바디 포즈 시계열만 (<prefix>.json / .bin): 움직인 바디는 프레임별, 나머지는 한 번."""
    moving = [b for b in range(pos.shape[1]) if np.ptp(pos[:, b], axis=0).max() > 1e-6 or np.ptp(quat[:, b], axis=0).max() > 1e-6]
    static = {b: dict(p=np.round(pos[0, b], 5).tolist(), q=np.round(quat[0, b], 6).tolist())
              for b in range(pos.shape[1]) if b not in moving}
    P = pos[:, moving].astype(np.float32).tobytes()
    Q = quat[:, moving].astype(np.float32).tobytes()
    prefix.with_suffix(".bin").write_bytes(P + Q)
    prefix.with_suffix(".json").write_text(json.dumps(dict(
        t=np.asarray(t).tolist(), moving=moving, static=static,
        pos=[0, len(moving) * len(t) * 3], quat=[len(P), len(moving) * len(t) * 4]), separators=(",", ":")))
