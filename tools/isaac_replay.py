"""Isaac Sim(PhysX) 재현기: botrail 셀의 계획을 Isaac Sim 에서 물리로 따라가게 하고, 결과 바디 포즈를
MuJoCo 웹 데이터와 같은 바디 순서로 저장한다 → 웹 뷰어에서 botrail / MuJoCo / Isaac Sim 을 같은 화면으로 비교.

  # botrail_mujoco 가 결과 폴더에 isaac_job.json(.npz) 를 남긴다 (run 또는 --plan-only)
  OMNI_KIT_ACCEPT_EULA=YES <isaac python> tools/isaac_replay.py OUT/mj/isaac_job.json

결과: OUT/mj/isaac.json/.bin (바디 포즈), OUT/mj/isaac_report.json (관절 추종 오차)
- 로봇: botrail 물리 USD 의 아티큘레이션을 USD 드라이브(강성·감쇠·토크 한계) 그대로, 위치+속도 목표로 구동.
- 움직이는 장애물(문·부품·장치): botrail 계획 포즈를 그대로 따름 (충돌 끔).
- 이동형 베이스 로봇(차량·보행·드론): botrail 로봇의 첫 링크를 플로팅 루트로 떼어내 계획 포즈(plan.json)로 매 스텝 옮기고,
  팔·다리 관절은 위와 같이 물리로 구동. 바퀴·발 ↔ 바닥·걸을 수 있는 면 접촉은 끈다 (MuJoCo 쪽 FLOOR_BIT 와 같은 규칙).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def load_plan_track(path, bodies):
    """plan.json/.bin (botrail 계획의 바디 포즈) 에서 주어진 바디들의 시계열: {body: (t, pos[N,3], quat[N,4] wxyz)}"""
    if not bodies:
        return {}
    meta = json.loads(path.read_text())
    raw = path.with_suffix(".bin").read_bytes()
    t = np.array(meta["t"])
    n, m = len(t), len(meta["moving"])
    P = np.frombuffer(raw, np.float32, n * m * 3, meta["pos"][0]).reshape(n, m, 3)
    Q = np.frombuffer(raw, np.float32, n * m * 4, meta["quat"][0]).reshape(n, m, 4)
    out = {}
    for b in bodies:
        if b in meta["moving"]:
            j = meta["moving"].index(b)
            out[b] = (t, P[:, j].astype(float), Q[:, j].astype(float))
        else:
            s = meta["static"][str(b)]
            out[b] = (t[:1], np.array([s["p"]], float), np.array([s["q"]], float))
    return out


def sample(track, t):
    """시각 t 의 위치·자세(선형 보간 + 쿼터니언 정규화)와 선속도·각속도(유한차분)"""
    T, P, Q = track
    if len(T) == 1:
        return P[0], Q[0], np.zeros(3), np.zeros(3)
    f = float(np.clip(np.interp(t, T, np.arange(len(T))), 0, len(T) - 1.000001))
    i, w = int(f), f - int(f)
    q0, q1 = Q[i], Q[i + 1] if np.dot(Q[i], Q[i + 1]) >= 0 else -Q[i + 1]
    q = (1 - w) * q0 + w * q1
    h = T[i + 1] - T[i]
    lin = (P[i + 1] - P[i]) / h
    dq = q1 - q0  # ω = 2 · dq/dt ⊗ q*  (wxyz)
    w0, v0 = q0[0], -q0[1:]
    d0, dv = dq[0], dq[1:]
    ang = 2 * (d0 * v0 + w0 * dv + np.cross(dv, v0)) / h
    return (1 - w) * P[i] + w * P[i + 1], q / np.linalg.norm(q), lin, ang


def detach_mobile_root(stage, r, floor_paths):
    """이동 로봇: 가상 베이스 관절·차량 쪽 연결을 끊고 botrail 첫 링크를 플로팅 아티큘레이션 루트로.
    바퀴·발 ↔ 바닥 접촉은 필터링. 반환: 새 루트 경로."""
    from pxr import PhysxSchema, Sdf, Usd, UsdPhysics
    rigid = lambda p: p.IsValid() and p.HasAPI(UsdPhysics.RigidBodyAPI)
    joints = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)]
    tgt = lambda j, k: [str(x) for x in j.GetRelationship(f"physics:body{k}").GetTargets()]
    edges = [(tgt(j, 0)[0] if tgt(j, 0) else None, tgt(j, 1)[0], j) for j in joints if tgt(j, 1)]
    root = r["root_link"]
    sub, frontier = {root}, [root]
    while frontier:
        cur = frontier.pop()
        for b0, b1, _ in edges:
            if b0 == cur and b1 not in sub and rigid(stage.GetPrimAtPath(b1)):
                sub.add(b1)
                frontier.append(b1)
    # 서브트리 밖과 이어진 관절(가상 베이스 x/y/z/yaw…, 차량 mount, 월드 고정)은 끈다
    upstream = set()
    for b0, b1, j in edges:
        if (b0 in sub) != (b1 in sub):
            UsdPhysics.Joint(j).CreateJointEnabledAttr(False)
            upstream.add(b0 if b1 in sub else b1)
    # 끊긴 위쪽 링크(가상 캐리어 등)는 강체를 꺼서 따로 떨어지지 않게 (움직이는 장애물은 이미 꺼져 있음)
    frontier = [u for u in upstream if u]
    while frontier:
        u = frontier.pop()
        prim = stage.GetPrimAtPath(u)
        if rigid(prim) and u not in sub:
            UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
            for b0, b1, j in edges:
                if u in (b0, b1):
                    UsdPhysics.Joint(j).CreateJointEnabledAttr(False)
                    o = b0 if b1 == u else b1
                    if o and o not in sub and o not in upstream:
                        upstream.add(o)
                        frontier.append(o)
    # 아티큘레이션 루트 옮기기
    for p in Usd.PrimRange(stage.GetPrimAtPath(r["root"])):
        if p.HasAPI(UsdPhysics.ArticulationRootAPI) and str(p.GetPath()) != root:
            p.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            p.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
    for a in stage.GetPrimAtPath(root).GetPath().GetAncestorsRange():
        ap = stage.GetPrimAtPath(a)
        if ap.IsValid() and ap.HasAPI(UsdPhysics.ArticulationRootAPI) and str(a) != root:
            ap.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            ap.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
    rp = stage.GetPrimAtPath(root)
    UsdPhysics.ArticulationRootAPI.Apply(rp)
    PhysxSchema.PhysxArticulationAPI.Apply(rp).CreateEnabledSelfCollisionsAttr(False)
    # 바닥·걸을 수 있는 면 ↔ 로봇 링크 접촉 끄기
    for fp in floor_paths:
        f = stage.GetPrimAtPath(fp)
        if f.IsValid():
            rel = UsdPhysics.FilteredPairsAPI.Apply(f).CreateFilteredPairsRel()
            for s_ in sorted(sub):
                rel.AddTarget(Sdf.Path(s_))
    return root


def setup_view(stage, job_dir):
    """RTX 렌더용 조명·카메라. 구도는 botrail_mujoco.camera (세 엔진 영상 공통 규칙)."""
    import math
    from pxr import Gf, UsdGeom, UsdLux
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from botrail_mujoco.camera import VFOV_DEG, view_for
    eye, target = view_for(job_dir)
    if not any(p.HasAPI(UsdLux.LightAPI) for p in stage.Traverse()):
        dome = UsdLux.DomeLight.Define(stage, "/World/ViewLights/Dome")
        dome.CreateIntensityAttr(900.0)
        sun = UsdLux.DistantLight.Define(stage, "/World/ViewLights/Sun")
        sun.CreateIntensityAttr(2500.0)
        UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-40, 25, 0))
    cam = UsdGeom.Camera.Define(stage, "/World/ViewCam")
    vap = 20.955 * 9 / 16
    cam.CreateHorizontalApertureAttr(20.955)
    cam.CreateVerticalApertureAttr(vap)
    cam.CreateFocalLengthAttr(vap / 2 / math.tan(math.radians(VFOV_DEG / 2)))
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 2000.0))
    UsdGeom.Xformable(cam).AddTransformOp().Set(
        Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)).GetInverse())
    return "/World/ViewCam"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job")
    ap.add_argument("--physics-hz", type=float, default=120.0)
    ap.add_argument("--until", type=float, default=None, help="디버그: 이 시각(s)까지만")
    ap.add_argument("--video", default=None, help="RTX 렌더 영상 경로 (mp4)")
    ap.add_argument("--video-fps", type=float, default=15.0)
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--trace", default=None, help="디버그: 이 로봇의 관절값을 1초마다 출력")
    a = ap.parse_args()
    job_path = Path(a.job)
    job = json.loads(job_path.read_text())
    arr = np.load(job_path.with_suffix(".npz"))
    out = job_path.with_name("isaac")
    rep_path = job_path.with_name("isaac_report.json")
    if any(r["mobile"] and "root_link" not in r for r in job["robots"]):
        rep_path.write_text(json.dumps({"supported": False, "reason": "이동 로봇 작업 파일이 옛 형식이에요 (--plan-only 로 다시 만드세요)"},
                                       ensure_ascii=False))
        print("skip: old job without mobile root")
        return
    root_track = load_plan_track(job_path.with_name("plan.json"), [r["root_body"] for r in job["robots"] if r["mobile"]])

    sys.argv = sys.argv[:1]
    from isaacsim import SimulationApp
    W, H = map(int, a.size.split("x"))
    app = SimulationApp({"headless": True, "width": W, "height": H, "renderer": "RaytracedLighting",
                         "extra_args": ["--enable", "omni.replicator.core"] if a.video else []})
    t_start = time.time()
    try:
        import omni.usd
        from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

        omni.usd.get_context().open_stage(job["usd"])
        for _ in range(10):
            app.update()
        stage = omni.usd.get_context().get_stage()

        # 접촉 오프셋을 mm 단위 여유에 맞게
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                c = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                c.CreateContactOffsetAttr(0.001)
                c.CreateRestOffsetAttr(0.0)

        # 움직이는 장애물: 강체·충돌 해제, 단일 transform op (로컬 스케일은 유지)
        cache = UsdGeom.XformCache()
        movers = []
        for k, path in enumerate(job["moving"]):
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                continue
            for sub in Usd.PrimRange(prim):
                if sub.HasAPI(UsdPhysics.RigidBodyAPI):
                    UsdPhysics.RigidBodyAPI(sub).CreateRigidBodyEnabledAttr(False)
                if sub.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(sub).CreateCollisionEnabledAttr(False)
            # 상위 그룹이 강체이고 부품만 움직이는 장애물로 잡힌 경우(팔레트 등): 그 강체도 끈다 — 충돌체를 잃고 떨어지지 않게
            anc = prim.GetParent()
            while anc.IsValid() and str(anc.GetPath()) not in ("/World/Env", "/World", "/"):
                if anc.HasAPI(UsdPhysics.RigidBodyAPI):
                    UsdPhysics.RigidBodyAPI(anc).CreateRigidBodyEnabledAttr(False)
                anc = anc.GetParent()
            xf = UsdGeom.Xformable(prim)
            local = xf.GetLocalTransformation()
            scale = Gf.Vec3d(Gf.Transform(local).GetScale())
            parent_inv = cache.GetLocalToWorldTransform(prim.GetParent()).GetInverse()
            xf.ClearXformOpOrder()
            op = xf.AddTransformOp(UsdGeom.XformOp.PrecisionDouble)
            movers.append((k, op, Gf.Matrix4d().SetScale(scale), parent_inv))

        if os.environ.get("ISAAC_NOCOLLIDE"):  # 디버그: 이 접두사 아래 충돌 끄기
            for prim in stage.Traverse():
                if str(prim.GetPath()).startswith(os.environ["ISAAC_NOCOLLIDE"]) and prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
        floor = [str(p.GetPath()) for p in stage.GetPrimAtPath("/World/Env").GetChildren()
                 if p.GetName().startswith("Ground")] + list(job.get("floor", []))
        for r in job["robots"]:
            if r["mobile"]:
                r["art_path"] = detach_mobile_root(stage, r, floor)

        # 정적 구조물(셀 베드·포스트·레일·용접 컨트롤러 등)은 botrail 에선 바닥에 고정된 집기다.
        # PhysX 는 RigidBodyAPI 가 켜져 있고 고정 조인트가 없으면 중력에 무너뜨린다 → 로봇 링크가 아닌
        # 강체는 모두 정적(rigidBodyEnabled=False)으로 돌려 충돌체로만 남긴다. 이동체는 그 위에 xform op 로
        # 운동학적 이동을 그대로 한다. (MuJoCo 변환은 이런 집기를 처음부터 정적 geom 으로 둬 문제가 없었다.)
        robot_roots = [r["root"] for r in job["robots"]]
        n_static = 0
        for prim in stage.Traverse():
            p = str(prim.GetPath())
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            if any(p == rr or p.startswith(rr + "/") for rr in robot_roots):
                continue  # 로봇 아티큘레이션 링크(모바일 루트 포함)는 건드리지 않는다
            api = UsdPhysics.RigidBodyAPI(prim)
            en = api.GetRigidBodyEnabledAttr()
            if en.Get() if en.HasAuthoredValue() else True:
                api.CreateRigidBodyEnabledAttr(False)
                n_static += 1

        # 질량이 없는 로봇 링크: PhysX 는 충돌체 부피 × 1000 kg/m³ 로 질량을 만들어 큰 팔이 수백 kg 이 된다.
        # MuJoCo 변환과 같은 규칙(botrail mass_floor): 형상이 있으면 0.2 kg·반지름 5 cm 구 관성, 없으면 0.001 kg
        n_mass = 0
        for r in job["robots"]:
            for prim in Usd.PrimRange(stage.GetPrimAtPath(r["root"])):
                if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    continue
                m = UsdPhysics.MassAPI.Apply(prim)
                mass = m.GetMassAttr().Get() if m.GetMassAttr().HasAuthoredValue() else None
                if mass is not None and mass > 0.001:
                    continue
                shaped = any(c.IsA(UsdGeom.Gprim) for c in Usd.PrimRange(prim) if c != prim)
                if mass is None or shaped:
                    mm = 0.2 if shaped else 0.001
                    m.CreateMassAttr(mm)
                    m.CreateDiagonalInertiaAttr(Gf.Vec3f(*([0.4 * mm * 0.05 ** 2] * 3)))
                    n_mass += 1

        # 미믹 관절(평행 링크 등)은 PhysX 가 유한한 관절 한계를 요구한다 — 없으면 미믹이 통째로 빠져 링크가 무너짐
        n_mimic_lim = 0
        mimic = set()
        for prim in stage.Traverse():
            if not prim.IsA(UsdPhysics.Joint):
                continue
            for at in prim.GetAttributes():
                if at.GetName().startswith("physxMimicJoint:"):
                    mimic.add(prim.GetPath())
            for rel in prim.GetRelationships():
                if rel.GetName().startswith("physxMimicJoint:") and rel.GetName().endswith(":referenceJoint"):
                    mimic.add(prim.GetPath())
                    mimic.update(rel.GetTargets())
        if os.environ.get("ISAAC_NOMIMIC"):  # 디버그
            for prim in stage.Traverse():
                for api in list(prim.GetAppliedSchemas()):
                    if api.startswith("PhysxMimicJointAPI"):
                        prim.RemoveAppliedSchema(api)
        for path in mimic:
            j = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(path))
            if not j:
                continue
            lo, hi = j.GetLowerLimitAttr().Get(), j.GetUpperLimitAttr().Get()
            if lo is None or hi is None or not np.isfinite(lo) or not np.isfinite(hi):
                j.CreateLowerLimitAttr(-359.0)
                j.CreateUpperLimitAttr(359.0)
                n_mimic_lim += 1

        # botrail 이 허용한 링크 ↔ 장애물 접촉 쌍은 거른다 (트랙에 박힌 셔틀처럼 처음부터 겹쳐 있으면 PhysX 가 밀어내 끼어 버림)
        n_filtered = 0
        for e in job.get("allowed", []):
            obst = stage.GetPrimAtPath(e["obstacle"])
            if not obst.IsValid():
                continue
            links = [prim.GetPath() for r in job["robots"] if e["robot"] in (None, r["name"])
                     for prim in Usd.PrimRange(stage.GetPrimAtPath(r["root"]))
                     if prim.GetName() == e["link"] and prim.HasAPI(UsdPhysics.RigidBodyAPI)]
            if not links:
                continue
            for o_ in Usd.PrimRange(obst):  # 장애물 그룹이면 그 아래 충돌체마다
                if o_.HasAPI(UsdPhysics.CollisionAPI) or o_.HasAPI(UsdPhysics.RigidBodyAPI):
                    rel = UsdPhysics.FilteredPairsAPI.Apply(o_).CreateFilteredPairsRel()
                    for lp in links:
                        rel.AddTarget(lp)
                    n_filtered += len(links)

        P, Q = arr["obj_pos"], arr["obj_quat"]
        T = arr["t"]
        plan_dt = float(job["dt"])

        def set_env(t):
            f = min(t / plan_dt, len(T) - 1.000001)
            i, w = int(f), f - int(f)
            for k, op, S, parent_inv in movers:
                pos = (1 - w) * P[i, k] + w * P[i + 1, k]
                q0, q1 = Q[i, k], Q[i + 1, k]
                q = (1 - w) * q0 + w * (q1 if np.dot(q0, q1) >= 0 else -q1)
                q = q / np.linalg.norm(q)
                W = Gf.Matrix4d().SetRotate(Gf.Quatd(float(q[0]), Gf.Vec3d(*map(float, q[1:])))) \
                    .SetTranslateOnly(Gf.Vec3d(*map(float, pos)))
                op.Set(S * W * parent_inv)

        # 자리채움 토크 한계 해제 (MuJoCo 쪽과 같은 규칙): 시뮬레이션 시작 전에 USD 드라이브 maxForce 를 올린다
        for r in job["robots"]:
            want = set(r.get("unlimit", []))
            if not want:
                continue
            for prim in Usd.PrimRange(stage.GetPrimAtPath(r["root"])):
                if prim.GetName() in want and prim.IsA(UsdPhysics.Joint):
                    for kind in ("angular", "linear"):
                        if prim.HasAPI(UsdPhysics.DriveAPI, kind):
                            UsdPhysics.DriveAPI(prim, kind).GetMaxForceAttr().Set(1e6)

        # 관절 속도 한계(physxJoint:maxJointVelocity, 회전은 deg/s): PhysX 는 지키고 MuJoCo 모델에는 없다 → 계획이 한계를
        # 넘는 관절(예: 차량 바퀴·마스트)은 Isaac 에서만 뒤처진다. 리포트에 남긴다
        vel_over = {}
        for k, r in enumerate(job["robots"]):
            qv = np.abs(np.gradient(arr[f"q{k}"], arr["t"], axis=0)).max(0) if len(arr["t"]) > 1 else np.zeros(len(r["dofs"]))
            jp = {p.GetName(): p for p in Usd.PrimRange(stage.GetPrimAtPath(r["root"])) if p.IsA(UsdPhysics.Joint)}
            for name, peak in zip(r["dofs"], qv):
                prim = jp.get(name, stage.GetPrimAtPath("/"))
                at = prim.GetAttribute("physxJoint:maxJointVelocity") if prim.IsValid() else None
                if not at or not at.HasAuthoredValue():
                    continue
                lim = float(at.Get())
                rev = prim.IsA(UsdPhysics.RevoluteJoint)
                lim_si = np.radians(lim) if rev else lim
                if peak > lim_si * 1.05 and not (rev and peak > 20 * lim_si):  # 순간 튐(조향 점프 등)은 제외
                    vel_over.setdefault(r["name"], {})[prim.GetName()] = dict(plan_peak=round(float(peak), 3),
                                                                              limit=round(float(lim_si), 3),
                                                                              unit="rad/s" if rev else "m/s")

        cam_path = setup_view(stage, job_path.parent) if a.video else None

        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.types import ArticulationAction

        # 물리 스텝을 웹 프레임 간격의 정수분의 1 로: 프레임이 계획 프레임과 같은 시각에 찍혀야 순간이동하는 물체
        # (숨겨 뒀다 꺼내는 랩 필름 등)가 중간 위치로 잡히지 않는다
        frame_dt = float(job["frame_dt"])
        substeps = max(1, int(round(frame_dt * a.physics_hz)))
        dt = frame_dt / substeps
        world = World(physics_dt=dt, rendering_dt=dt * 4, stage_units_in_meters=1.0)
        arts = []
        for k, r in enumerate(job["robots"]):
            art = SingleArticulation(r.get("art_path", r["root"]), name=f"r{k}")
            world.scene.add(art)
            arts.append(art)
        mobiles = [(art, root_track[r["root_body"]]) for art, r in zip(arts, job["robots"]) if r["mobile"]]

        def set_bases(t):
            for art, tr in mobiles:
                p_, q_, v_, w_ = sample(tr, t)
                art.set_world_pose(position=p_, orientation=q_)
                art.set_linear_velocity(v_)
                art.set_angular_velocity(w_)

        set_env(0.0)
        world.reset()
        set_bases(0.0)

        plans = []
        revolute = {p.GetName() for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)}
        for k, (art, r) in enumerate(zip(arts, job["robots"])):
            names = list(art.dof_names)
            idx = [names.index(n) for n in r["dofs"] if n in names]
            cols = [i for i, n in enumerate(r["dofs"]) if n in names]
            q = arr[f"q{k}"][:, cols]
            v = np.gradient(q, T, axis=0) if len(T) > 1 else np.zeros_like(q)
            full = art.get_joint_positions()
            full[idx] = q[0]
            art.set_joint_positions(full)
            art.set_joint_velocities(np.zeros(len(names)))
            # 연속 회전 관절(바퀴·로터): PhysX 는 각도를 감아(wrap) 돌려주고 계획 각도는 계속 커진다 → 목표·오차를 감김 기준으로
            spin = np.array([np.ptp(q[:, c]) > np.pi and n in revolute for c, n in enumerate(np.array(r["dofs"])[cols])], bool) \
                if len(idx) else np.zeros(0, bool)
            plans.append(dict(art=art, idx=np.array(idx, int), q=q, v=v, spin=spin,
                              missing=[n for n in r["dofs"] if n not in names]))

        body_ids = sorted(int(b) for b in job["bodies"])
        body_prims = [stage.GetPrimAtPath(job["bodies"][str(b)]) for b in body_ids]
        world.play()
        enc = None
        if a.video:
            import subprocess
            import omni.replicator.core as rep_
            rp_ = rep_.create.render_product(cam_path, (W, H))
            rgb = rep_.AnnotatorRegistry.get_annotator("rgb")
            rgb.attach([rp_])
            for _ in range(40):  # 렌더러 워밍업 (물리 진행 없음)
                world.render()
            enc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba",
                                    "-s", f"{W}x{H}", "-r", str(a.video_fps), "-i", "-", "-c:v", "libx264",
                                    "-pix_fmt", "yuv420p", "-crf", "28", "-preset", "veryfast", "-movflags", "+faststart",
                                    a.video], stdin=subprocess.PIPE)
            video_every = max(1, int(round(1.0 / a.video_fps / dt)))
        steps = int(round(min(job["t_end"], a.until or 1e9) / dt)) + 1
        many = len(movers) > 30
        FT, FP, FQ = [], [], []
        err_max = [0.0] * len(plans)
        err_joint = [np.zeros(len(pl["idx"])) for pl in plans]
        err_sum = [0.0] * len(plans)
        n_err = 0
        for i in range(steps):
            t = i * dt
            f = min(t / plan_dt, len(T) - 1.000001)
            k0, w = int(f), f - int(f)
            for pl in plans:
                q = (1 - w) * pl["q"][k0] + w * pl["q"][k0 + 1]
                v = (1 - w) * pl["v"][k0] + w * pl["v"][k0 + 1]
                if not len(pl["idx"]):
                    continue
                if pl["spin"].any():
                    cur = pl["art"].get_joint_positions()[pl["idx"]]
                    q = np.where(pl["spin"], cur + wrap(q - cur), q)
                pl["art"].apply_action(ArticulationAction(joint_positions=q, joint_velocities=v, joint_indices=pl["idx"]))
            if not many or i % substeps == 0:  # 장애물이 많으면 프레임 시각에만 (기록되는 포즈는 정확)
                set_env(t)
            set_bases(t - dt)  # 한 스텝 동안 계획 속도로 움직여 t 에 계획 포즈에 닿도록
            world.step(render=False)
            if enc and i % video_every == 0:
                world.render()
                img = np.asarray(rgb.get_data())
                if img.size == W * H * 4:
                    enc.stdin.write(np.ascontiguousarray(img.reshape(H, W, 4)).tobytes())
            if i % substeps == 0:
                c = UsdGeom.XformCache()
                pos, quat = [], []
                for prim in body_prims:
                    tr = Gf.Transform(c.GetLocalToWorldTransform(prim))
                    p_ = tr.GetTranslation()
                    q_ = tr.GetRotation().GetQuat()
                    pos.append([p_[0], p_[1], p_[2]])
                    quat.append([q_.GetReal(), *q_.GetImaginary()])
                FT.append(round(t, 4))
                FP.append(pos)
                FQ.append(quat)
                for j, pl in enumerate(plans):
                    q = (1 - w) * pl["q"][k0] + w * pl["q"][k0 + 1]
                    d_ = pl["art"].get_joint_positions()[pl["idx"]] - q if len(pl["idx"]) else np.zeros(0)
                    ej = np.abs(np.where(pl["spin"], wrap(d_), d_))
                    err_joint[j] = np.maximum(err_joint[j], ej)
                    e = float(ej.max()) if len(ej) else 0.0
                    err_max[j] = max(err_max[j], e)
                    err_sum[j] += e
                n_err += 1
            if a.trace and i % int(round(1 / dt)) == 0:
                for pl, r in zip(plans, job["robots"]):
                    if r["name"] == a.trace:
                        print(f"[trace] t={t:.2f} plan={np.round(pl['q'][k0], 3)} isaac={np.round(pl['art'].get_joint_positions()[pl['idx']], 3)}", flush=True)
            if i and i % max(1, steps // 10) == 0:
                print(f"[isaac] {100 * i // steps}%", flush=True)

        if enc:
            enc.stdin.close()
            enc.wait()
            print("wrote", a.video, flush=True)
        FP, FQ = np.array(FP, np.float32), np.array(FQ, np.float32)
        moving = [b for j, b in enumerate(body_ids) if np.ptp(FP[:, j], axis=0).max() > 1e-6 or np.ptp(FQ[:, j], axis=0).max() > 1e-6]
        mj = [body_ids.index(b) for b in moving]
        static = {b: dict(p=FP[0, j].round(5).tolist(), q=FQ[0, j].round(6).tolist())
                  for j, b in enumerate(body_ids) if b not in moving}
        Pb, Qb = FP[:, mj].tobytes(), FQ[:, mj].tobytes()
        out.with_suffix(".bin").write_bytes(Pb + Qb)
        out.with_suffix(".json").write_text(json.dumps(dict(
            t=FT, moving=moving, static=static, pos=[0, len(moving) * len(FT) * 3],
            quat=[len(Pb), len(moving) * len(FT) * 4]), separators=(",", ":")))
        rep = dict(supported=True, engine="Isaac Sim (PhysX)", physics_dt=dt, filtered_allowed_pairs=n_filtered,
                   mimic_limits_added=n_mimic_lim, mass_assumed_links=n_mass, pinned_static_bodies=n_static,
                   seconds=round(time.time() - t_start, 1),
                   robots={r["name"]: dict(max_joint_err_rad=round(err_max[j], 5),
                                           mean_joint_err_rad=round(err_sum[j] / max(n_err, 1), 6),
                                           missing_dofs=plans[j]["missing"],
                                           joint_err_rad={n: round(float(v), 5) for n, v in
                                                          zip([d for d in r["dofs"] if d not in plans[j]["missing"]], err_joint[j])},
                                           max_force_raised=r.get("unlimit", []),
                                           mobile_base=bool(r["mobile"]),
                                           plan_exceeds_velocity_limit=vel_over.get(r["name"], {}))
                           for j, r in enumerate(job["robots"])})
        rep_path.write_text(json.dumps(rep, ensure_ascii=False, indent=2))
        print(json.dumps(rep, ensure_ascii=False), flush=True)
    except BaseException:
        import traceback
        err = traceback.format_exc()
        print(err, flush=True)
        rep_path.write_text(json.dumps({"supported": False, "reason": "Isaac 재현 중 오류", "error": err[-2000:]}, ensure_ascii=False))
        app.close()
        os._exit(1)
    app.close()


if __name__ == "__main__":
    main()
