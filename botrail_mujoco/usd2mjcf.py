"""botrail 물리 스테이지(USD, `scene.export_usd(physics=...)`) → MuJoCo MJCF.

botrail 이 쓰는 UsdPhysics 어휘만 다룬다:
- 아티큘레이션(PhysicsArticulationRootAPI) → 바디 트리. 링크 질량/관성(MassAPI), 관절 프레임(localPos/Rot0·1),
  축·한계, 아마추어(physxJoint:armature), 드라이브(stiffness/damping/maxForce) → general 액추에이터
  (affine bias 로 kp(q*-q)+kv(v*-qd)), PhysxMimicJointAPI → equality joint.
- /World/Env 의 Cube/Cylinder/Sphere/Mesh → geom. 움직이는 장애물(`moving`)은 mocap 바디.
- 충돌 그룹: 로봇·로봇이 든 물체 = contype 1 / conaffinity 2, 환경 = 2 / 1 → 로봇↔환경만 검사.
  잡는 동안에는 adapter 가 mocap geom 의 그룹을 실행 중에 바꾼다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

ENV_BITS = (2, 1)
OFF_BITS = (0, 0)


def robot_bits(k: int, n: int) -> tuple[int, int]:
    """로봇 k 의 (contype, conaffinity). 환경과 다른 로봇과는 충돌, 자기 자신과는 무시.
    env = (2, 1); 로봇 k = (1 | 4<<k, 2 | 다른 로봇 비트). 든 물체는 든 로봇의 비트를 받는다."""
    own = 4 << k
    others = sum(4 << j for j in range(n) if j != k)
    return 1 | own, 2 | others


ROBOT_BITS = robot_bits(0, 1)


# ------------------------------------------------------------------ 수학 (열벡터 4x4)
def gf_to_np(m) -> np.ndarray:
    return np.array(m, dtype=float).T


def H(pos, quat_wxyz) -> np.ndarray:
    w, x, y, z = quat_wxyz
    T = np.eye(4)
    T[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    T[:3, 3] = pos
    return T


def quat_of(R) -> np.ndarray:
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def split_scale(M):
    s = np.linalg.norm(M[:3, :3], axis=0)
    T = M.copy()
    T[:3, :3] = M[:3, :3] / s
    return T, s


def fmt(v) -> str:
    return " ".join(f"{x:.9g}" for x in v)


def pq(T) -> str:
    return f'pos="{fmt(T[:3, 3])}" quat="{fmt(quat_of(T[:3, :3]))}"'


def jframe(joint, side):
    p = joint.GetAttribute(f"physics:localPos{side}").Get() or Gf.Vec3f(0)
    r = joint.GetAttribute(f"physics:localRot{side}").Get() or Gf.Quatf(1)
    return H(np.array(p), [r.GetReal(), *r.GetImaginary()])


def mj_name(path: str) -> str:
    """USD prim 경로 → MJCF 이름 (봇레일 장애물 이름 그대로: /World/Env/a/b → a/b)."""
    return path.replace("/World/Env/", "").replace("/World/", "")


@dataclass
class Conversion:
    xml_path: Path
    robots: dict = field(default_factory=dict)      # USD 루트 → {"joints": [...], "actuators": {joint: info}}
    mocap: dict = field(default_factory=dict)       # 장애물 이름 → mocap 바디 이름
    mocap_geoms: dict = field(default_factory=dict)  # 장애물 이름 → [(geom 이름, 원래 충돌 여부)]


class _Geoms:
    def __init__(self, out_dir: Path):
        self.cache = UsdGeom.XformCache()
        self.out_dir = out_dir
        self.meshes: list[tuple[str, str]] = []
        self.n = 0

    @staticmethod
    def color(prim):
        if prim.HasAttribute("primvars:displayColor"):
            dc = prim.GetAttribute("primvars:displayColor").Get()
            if dc:
                return [*dc[0], 1.0]
        mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if mat:
            for sh in Usd.PrimRange(mat.GetPrim()):
                a = sh.GetAttribute("inputs:diffuseColor")
                if a and a.Get() is not None:
                    return [*a.Get(), 1.0]
        return [0.75, 0.77, 0.8, 1.0]

    def emit(self, root, body_world, bits_collide, skip=(), out_names=None, prune_bodies=False):
        out = []
        it = iter(Usd.PrimRange(root))
        for prim in it:
            path = str(prim.GetPath())
            if prim != root and (path in skip or (prune_bodies and prim.HasAPI(UsdPhysics.RigidBodyAPI))):
                it.PruneChildren()  # mocap 으로 따로 만들 장애물 / 로봇의 다른 링크
                continue
            if not prim.IsA(UsdGeom.Gprim):
                continue
            img = UsdGeom.Imageable(prim)
            if img.ComputeVisibility() == UsdGeom.Tokens.invisible:
                continue
            hidden = img.GetPurposeAttr().Get() == UsdGeom.Tokens.guide
            collide = prim.HasAPI(UsdPhysics.CollisionAPI) and \
                UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False
            if hidden and not collide:
                continue
            M = np.linalg.inv(body_world) @ gf_to_np(self.cache.GetLocalToWorldTransform(prim))
            T, s = split_scale(M)
            ct, ca = bits_collide if collide else OFF_BITS
            rgba = self.color(prim)
            group = 3 if hidden else 0
            if hidden:
                rgba[3] = 0.0
            name = mj_name(path)
            common = f'name="{name}" contype="{ct}" conaffinity="{ca}" group="{group}" rgba="{fmt(rgba)}"'
            xml = None
            if prim.IsA(UsdGeom.Cube):
                size = prim.GetAttribute("size").Get() or 2.0
                xml = f'<geom type="box" {common} size="{fmt(s * size / 2)}" {pq(T)}/>'
            elif prim.IsA(UsdGeom.Cylinder):
                r, h = prim.GetAttribute("radius").Get(), prim.GetAttribute("height").Get()
                ax = prim.GetAttribute("axis").Get() or "Z"
                c = math.cos(math.pi / 4)
                if ax == "X":
                    T = T @ H([0, 0, 0], [c, 0, c, 0])
                elif ax == "Y":
                    T = T @ H([0, 0, 0], [c, -c, 0, 0])
                xml = f'<geom type="cylinder" {common} size="{r * s[0]:.9g} {h * s[2] / 2:.9g}" {pq(T)}/>'
            elif prim.IsA(UsdGeom.Sphere):
                xml = f'<geom type="sphere" {common} size="{prim.GetAttribute("radius").Get() * s[0]:.9g}" {pq(T)}/>'
            elif prim.IsA(UsdGeom.Plane):
                xml = f'<geom type="plane" {common} size="0 0 0.05" {pq(T)} material="floor"/>'
            elif prim.IsA(UsdGeom.Mesh):
                f = self.mesh_file(prim, M)
                if f:
                    mname = f"m{self.n}"
                    self.meshes.append((mname, f))
                    xml = f'<geom type="mesh" mesh="{mname}" {common}/>'
            if xml:
                out.append(xml)
                if out_names is not None:
                    out_names.append((name, collide))
            self.n += 1
        return out

    def mesh_file(self, prim, M):
        mesh = UsdGeom.Mesh(prim)
        pts = np.array(mesh.GetPointsAttr().Get() or [], dtype=float)
        if len(pts) < 4:
            return None
        counts = mesh.GetFaceVertexCountsAttr().Get()
        idx = mesh.GetFaceVertexIndicesAttr().Get()
        pts = (M[:3, :3] @ pts.T).T + M[:3, 3]
        d = self.out_dir / "meshes"
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"m{self.n}.obj"
        with open(path, "w") as fh:
            fh.writelines(f"v {p[0]:.7g} {p[1]:.7g} {p[2]:.7g}\n" for p in pts)
            k = 0
            for c in counts:
                for j in range(1, c - 1):
                    fh.write(f"f {idx[k] + 1} {idx[k + j] + 1} {idx[k + j + 1] + 1}\n")
                k += c
        return f"meshes/{path.name}"


RESERVED = {"world"}
MASS_UNSET = 0.0011   # botrail 은 관성 정보가 없는 링크를 0.001 kg 으로 내보낸다
MASS_FLOOR = 0.2      # botrail set_robot_physics 의 mass_floor 기본값
# PhysX 는 아주 큰 드라이브 강성을 암시적으로 풀지만 MuJoCo 는 강성을 명시적으로 적분한다 → 안정 범위로 조정
KP_MAX = 1e6          # N·m/rad 또는 N/m
KP_SAFE, KV_SAFE = 1e5, 1e4
ARMATURE_MIN = 0.05   # 구동 관절 최소 반사 관성 (kg·m²)
INERTIA_R = 0.02      # 링크 관성 하한: 반지름 2 cm 구


def body_name(prefix: str, name: str) -> str:
    """MJCF 바디 이름. MuJoCo 예약어(world)와 겹치는 URDF 링크 이름은 robot_ 를 붙인다."""
    n = prefix + name
    return "robot_" + n if n in RESERVED else n


def _robot_xml(stage, root, g: _Geoms, info: dict, eqs: list, prefix: str, bits, all_joints, root_link=None):
    """아티큘레이션 하나의 바디 트리. 링크는 폴더 구조가 아니라 관절 연결을 따라 모은다
    (botrail 은 다리 링크를 루트의 형제로, 차량 몸체를 /World/Env 에 두기도 한다).
    root_link: 이동형 베이스일 때 트리를 시작할 링크 이름(botrail 로봇의 첫 링크) — 그 위의 가상 베이스 관절·차량은 건너뜀."""
    rootp = str(root.GetPath())
    rigid = lambda p: p.IsValid() and p.HasAPI(UsdPhysics.RigidBodyAPI)
    edges = []  # (body0, body1, joint)
    for j in all_joints:
        b0 = [str(x) for x in j.GetRelationship("physics:body0").GetTargets()]
        b1 = [str(x) for x in j.GetRelationship("physics:body1").GetTargets()]
        if b1:
            edges.append((b0[0] if b0 else None, b1[0], j))
    # 시작 링크: 루트가 링크면 그것, 아니면 루트 아래에서 월드에 붙는 링크
    if rigid(root):
        start = rootp
    else:
        cands = [b1 for b0, b1, _ in edges if (b0 is None or not rigid(stage.GetPrimAtPath(b0)))
                 and b1.startswith(rootp + "/") and rigid(stage.GetPrimAtPath(b1))]
        start = cands[0] if cands else next(str(p.GetPath()) for p in Usd.PrimRange(root) if rigid(p))
    # 관절 연결을 따라 링크 수집
    links, frontier = {start: stage.GetPrimAtPath(start)}, [start]
    while frontier:
        cur = frontier.pop()
        for b0, b1, _ in edges:
            if b0 == cur and b1 not in links and rigid(stage.GetPrimAtPath(b1)):
                links[b1] = stage.GetPrimAtPath(b1)
                frontier.append(b1)
    if root_link:  # 이동형 베이스: botrail 로봇의 첫 링크에서 트리를 시작
        hit = [p for p in links if p.rsplit("/", 1)[-1] == root_link.rsplit("/", 1)[-1]]
        if hit:
            start = hit[0]
    world_of = {k: split_scale(gf_to_np(g.cache.GetLocalToWorldTransform(p)))[0] for k, p in links.items()}
    children = {}
    for b0, b1, j in edges:
        if b0 in links and b1 in links:
            children.setdefault(b0, []).append((b1, j))
    base = (start, None)

    def body(path, joint, T_rel, depth):
        prim = links[path]
        ind = "  " * depth
        xml = [f'{ind}<body name="{body_name(prefix, prim.GetName())}" {pq(T_rel)}>']
        info.setdefault("body_paths", {})[body_name(prefix, prim.GetName())] = path  # MJCF 바디 → USD 링크 경로
        m = UsdPhysics.MassAPI(prim)
        mass = m.GetMassAttr().Get() or 0.001
        com = np.array(m.GetCenterOfMassAttr().Get() or Gf.Vec3f(0), dtype=float)
        if not np.all(np.isfinite(com)):  # USD 의 "미지정" 센티널 (-inf)
            com = np.zeros(3)
        di = m.GetDiagonalInertiaAttr().Get() or Gf.Vec3f(1e-6)
        di = [v if v is not None and math.isfinite(v) and v > 0 else 1e-6 for v in di]
        pa = m.GetPrincipalAxesAttr().Get() or Gf.Quatf(1)
        paq = np.array([pa.GetReal(), *pa.GetImaginary()])
        paq = paq / np.linalg.norm(paq) if np.linalg.norm(paq) > 1e-9 else np.array([1.0, 0, 0, 0])
        geoms = g.emit(prim, world_of[path], bits, prune_bodies=True)
        if mass <= MASS_UNSET and any('type="mesh"' in x or 'type="box"' in x or 'type="cylinder"' in x for x in geoms):
            # 형상은 있는데 질량이 미지정(내보내기 기본 0.001 kg): botrail 의 mass_floor 와 같은 0.2 kg,
            # 반지름 5 cm 구 관성. 이게 없으면 큰 PD 게인 아래 동역학이 불안정해진다.
            mass, di = MASS_FLOOR, [0.4 * MASS_FLOOR * 0.05 ** 2] * 3
            info["mass_floored"] = info.get("mass_floored", 0) + 1
        info["mass"] = info.get("mass", 0.0) + mass
        floor = 0.4 * mass * INERTIA_R ** 2
        if mass > 0.05 and min(di) < floor:
            di = [max(v, floor) for v in di]
            info["inertia_floored"] = info.get("inertia_floored", 0) + 1
        xml.append(f'{ind}  <inertial pos="{fmt(com)}" quat="{fmt(paq)}" mass="{max(mass, 1e-4):.6g}" '
                   f'diaginertia="{fmt([max(v, 1e-7) for v in di])}"/>')
        if joint is not None and (joint.IsA(UsdPhysics.RevoluteJoint) or joint.IsA(UsdPhysics.PrismaticJoint)):
            rev = joint.IsA(UsdPhysics.RevoluteJoint)
            J1 = jframe(joint, 1)
            axis = J1[:3, :3] @ np.eye(3)["XYZ".index(joint.GetAttribute("physics:axis").Get() or "X")]
            lo, hi = joint.GetAttribute("physics:lowerLimit").Get(), joint.GetAttribute("physics:upperLimit").Get()
            limited = lo is not None and hi is not None and math.isfinite(lo) and math.isfinite(hi)
            if rev and limited:
                lo, hi = math.radians(lo), math.radians(hi)
            arm = joint.GetAttribute("physxJoint:armature").Get() or 0.0
            jn = prefix + joint.GetName()
            rng = f'range="{lo:.6g} {hi:.6g}" limited="true"' if limited else 'limited="false"'
            xml.append(f'{ind}  <joint name="{jn}" type="{"hinge" if rev else "slide"}" pos="{fmt(J1[:3, 3])}" '
                       f'axis="{fmt(axis)}" {rng} armature="{arm:.6g}"/>')
            info["joints"].append(dict(name=jn, prim=joint.GetName(),
                                       rel=str(joint.GetPath())[len(rootp):]))
            mimic = False
            # usd-core 에는 PhysxSchema 가 등록돼 있지 않아 GetAppliedSchemas() 가 PhysxMimicJointAPI 를
            # 돌려주지 않는다 → 속성 이름(physxMimicJoint:<축>:referenceJoint)으로 직접 찾는다.
            mimic_axes = {pr.GetName().split(":")[1] for pr in joint.GetProperties()
                          if pr.GetName().startswith("physxMimicJoint:") and pr.GetName().endswith(":referenceJoint")}
            for ax in sorted(mimic_axes):
                if ax.startswith("rot") != rev:  # botrail 과 같이: 관절 종류와 축이 안 맞는 미믹은 무시
                    info.setdefault("ignored_mimic", []).append(jn)
                    continue
                ref = joint.GetRelationship(f"physxMimicJoint:{ax}:referenceJoint").GetTargets()
                gear = joint.GetAttribute(f"physxMimicJoint:{ax}:gearing").Get() or 0.0
                off = joint.GetAttribute(f"physxMimicJoint:{ax}:offset").Get() or 0.0
                if ref:  # PhysX: q + gearing*q_ref + offset = 0
                    eqs.append((jn, prefix + ref[0].name, -off, -gear))
                    mimic = True
            kind = "angular" if rev else "linear"
            if joint.HasAPI(UsdPhysics.DriveAPI, kind) and not mimic:
                d = UsdPhysics.DriveAPI(joint, kind)
                k, c = d.GetStiffnessAttr().Get() or 0.0, d.GetDampingAttr().Get() or 0.0
                if rev:  # USD 각 드라이브는 deg 단위
                    k, c = k * 180 / math.pi, c * 180 / math.pi
                fmax = d.GetMaxForceAttr().Get()
                if k > KP_MAX:
                    info.setdefault("gains_clamped", []).append(jn)
                    k, c = KP_SAFE, KV_SAFE
                if k > 0:
                    info["actuators"][jn] = dict(kp=k, kv=c, fmax=fmax if fmax and math.isfinite(fmax) else None)
        if joint is not None and info["actuators"].get(prefix + joint.GetName()) is not None:
            for i, line in enumerate(xml):  # 구동 관절의 아마추어 하한
                if f'<joint name="{prefix + joint.GetName()}"' in line:
                    a0 = float(line.split('armature="')[1].split('"')[0])
                    if a0 < ARMATURE_MIN:
                        xml[i] = line.replace(f'armature="{a0:.6g}"', f'armature="{ARMATURE_MIN}"')
                        info["armature_floored"] = info.get("armature_floored", 0) + 1
        xml += [f"{ind}  {x}" for x in geoms]
        for child, cj in children.get(path, []):
            xml += body(child, cj, jframe(cj, 0) @ np.linalg.inv(jframe(cj, 1)), depth + 1)
        xml.append(f"{ind}</body>")
        return xml

    bpath, bj = base
    T = world_of[bpath]  # 루트 링크의 내보낸 시점 월드 포즈
    info["root_link"] = body_name(prefix, links[bpath].GetName())
    xml = body(bpath, None, T, 2)
    if info.get("mobile"):  # 이동형 베이스: 루트를 mocap 으로 — botrail 이 계획한 차량/몸통 경로를 그대로 따른다
        xml[0] = xml[0].replace("<body ", '<body mocap="true" ', 1)
    return xml


def robot_name_of(root_path: str) -> str:
    """아티큘레이션 루트 경로 → 로봇 인스턴스 이름 (/World/<이름>/...)."""
    parts = root_path.strip("/").split("/")
    return parts[1] if len(parts) > 1 else parts[0]


def articulation_roots(usd_path: Path) -> list[str]:
    stage = Usd.Stage.Open(str(usd_path))
    return [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]


def convert(usd_path: Path, out_dir: Path, moving: list[str], *, timestep=0.001,
            size=(1280, 720), fovy=36.0, mobile_roots=()) -> Conversion:
    """USD → MJCF. `moving` = botrail 장애물 이름 (mocap 으로 만들 것)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.Open(str(usd_path))
    g = _Geoms(out_dir)
    conv = Conversion(xml_path=out_dir / "model.xml")
    eqs: list = []

    robot_xml = []
    roots = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    all_joints = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)]
    mobile_roots = dict(mobile_roots) if not isinstance(mobile_roots, dict) else mobile_roots
    for k, prim in enumerate(roots):
        prefix = "" if len(roots) == 1 else robot_name_of(str(prim.GetPath())) + "/"
        info = conv.robots.setdefault(str(prim.GetPath()), {"joints": [], "actuators": {}, "prefix": prefix,
                                                            "index": k, "bits": robot_bits(k, len(roots)),
                                                            "mobile": str(prim.GetPath()) in mobile_roots})
        robot_xml += _robot_xml(stage, prim, g, info, eqs, prefix, info["bits"], all_joints,
                                root_link=mobile_roots.get(str(prim.GetPath())))

    # 움직이는 장애물: 이름 → USD 경로. 상위 그룹이 움직이면 그 그룹 하나만 mocap 으로.
    env = stage.GetPrimAtPath("/World/Env")
    moving_paths = {}
    for name in sorted(moving, key=len):
        path = "/World/Env/" + name
        if not stage.GetPrimAtPath(path).IsValid():
            continue
        if any(path.startswith(p + "/") for p in moving_paths.values()):
            continue
        moving_paths[name] = path

    static_xml = [f"    {x}" for x in g.emit(env, np.eye(4), ENV_BITS, skip=set(moving_paths.values()))]
    ground = stage.GetPrimAtPath("/World/Ground")
    if ground.IsValid():
        static_xml += [f"    {x}" for x in g.emit(ground, np.eye(4), ENV_BITS)]

    mocap_xml = []
    for name, path in moving_paths.items():
        prim = stage.GetPrimAtPath(path)
        Tw = split_scale(gf_to_np(g.cache.GetLocalToWorldTransform(prim)))[0]
        names: list = []
        gs = g.emit(prim, Tw, ENV_BITS, out_names=names)
        bname = "mc:" + name
        conv.mocap[name] = bname
        conv.mocap_geoms[name] = names
        mocap_xml.append(f'    <body name="{bname}" mocap="true" {pq(Tw)}>')
        mocap_xml += [f"      {x}" for x in gs]
        mocap_xml.append("    </body>")

    act = []
    for info in conv.robots.values():
        for jn, d in info["actuators"].items():
            fr = f'forcerange="{-d["fmax"]:.6g} {d["fmax"]:.6g}" forcelimited="true"' if d["fmax"] else ""
            act.append(f'    <general name="a:{jn}" joint="{jn}" gainprm="{d["kp"]:.6g}" biastype="affine" '
                       f'biasprm="0 {-d["kp"]:.6g} {-d["kv"]:.6g}" {fr}/>')
    # 미믹은 기구학적 결합(평행 링크·그리퍼)이라 단단해야 한다: 기본 solref(20 ms)면 그 관절에 매달린
    # 하위 체인 전체가 움직이는 동안 처진다. 시간상수를 스텝 2배로.
    eq_xml = [f'    <joint joint1="{a}" joint2="{b}" polycoef="{c0:.6g} {c1:.6g} 0 0 0" '
              f'solref="{2 * timestep:.6g} 1" solimp="0.99 0.999 0.0001"/>' for a, b, c0, c1 in eqs]

    w, h = size
    xml = f"""<mujoco model="{usd_path.stem}">
  <compiler angle="radian" inertiafromgeom="false" meshdir="."/>
  <option timestep="{timestep}" integrator="implicitfast" gravity="0 0 -9.81">
    <flag filterparent="disable"/>
  </option>
  <visual>
    <global offwidth="{w}" offheight="{h}" fovy="{fovy}"/>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.5 0.5 0.5" specular="0.1 0.1 0.1"/>
    <quality shadowsize="4096"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="flat" rgb1="0.86 0.86 0.85" rgb2="0.86 0.86 0.85" width="32" height="32"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.82 0.82 0.8" rgb2="0.78 0.78 0.76" width="512" height="512"/>
    <material name="floor" texture="grid" texrepeat="8 8" reflectance="0.05"/>
{chr(10).join(f'    <mesh name="{n}" file="{f}"/>' for n, f in g.meshes)}
  </asset>
  <worldbody>
    <light directional="true" pos="2 -3 6" dir="-0.3 0.4 -1" diffuse="0.6 0.6 0.6" castshadow="true"/>
{chr(10).join(static_xml)}
{chr(10).join(mocap_xml)}
{chr(10).join(robot_xml)}
  </worldbody>
  <equality>
{chr(10).join(eq_xml)}
  </equality>
  <actuator>
{chr(10).join(act)}
  </actuator>
</mujoco>
"""
    conv.xml_path.write_text(xml)
    return conv
