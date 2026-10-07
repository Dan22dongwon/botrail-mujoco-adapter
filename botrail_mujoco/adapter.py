"""botrail 셀 → MuJoCo 어댑터.

    adapter = MujocoAdapter(scene, ["tend", "vmc"], "out/mj")
    adapter.prepare()          # botrail 롤아웃 → 계획/움직이는 물체/파지 구간, 물리 USD → MJCF
    report = adapter.run()     # MuJoCo 가 계획을 물리적으로 추종: 오차, 충돌, 토크 여유

역할 분담
- botrail: PLC 시퀀스·장치·센서·모션 계획 (open_rollout, 10 ms 스캔). 셀의 "논리"와 "계획".
- MuJoCo: 로봇 동역학(질량/관성/아마추어/토크 한계/PD), 로봇·든 물체 ↔ 환경·다른 로봇 접촉. 셀의 "물리".
  장치가 움직이는 물체(문, 컨베이어 위 부품, 액추에이터)와 로봇이 든 물체는 mocap 으로 botrail 포즈를 따른다.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import usd2mjcf


FLOOR_BIT = 1 << 30
SCAN_TICKS = 10      # 움직이는 장애물을 찾을 때 훑는 간격 (틱)  # 바닥·걸을 수 있는 면 (이동 로봇과는 충돌 계산 제외)


class UnsupportedCell(Exception):
    """어댑터가 아직 다루지 않는 셀 (이유를 메시지로)."""


@dataclass
class RobotPlan:
    name: str
    joint_names: list
    tips: list           # TCP 링크들 (팔이 여럿이면 팔마다 하나)
    q: np.ndarray        # (n, dof)
    tcp: np.ndarray      # (n, ntips, 3)
    base_pos: np.ndarray = None   # (n, 3) 첫 링크(베이스) 월드 위치
    base_quat: np.ndarray = None  # (n, 4) wxyz
    mobile: bool = False          # 베이스가 움직이는 로봇 (차량·보행·드론)


def tips_of(robot) -> list:
    """로봇의 TCP 링크 목록. 다중 팔 로봇(groups)은 팔마다 tip."""
    try:
        return [robot.tcp_link]
    except ValueError:
        return [robot.group(g).tip for g in robot.groups]


@dataclass
class Plan:
    """botrail 롤아웃 한 번의 기록 (스캔 틱마다)."""
    dt: float
    t: np.ndarray
    robots: list = field(default_factory=list)   # [RobotPlan]
    obj_pos: np.ndarray = None    # (n, k, 3)
    obj_quat: np.ndarray = None   # (n, k, 4) wxyz
    moving: list = field(default_factory=list)
    grasps: list = field(default_factory=list)   # [{"object", "robot", "start", "end"}]
    steps: list = field(default_factory=list)
    clearance_mm: float | None = None
    ramps: list = field(default_factory=list)    # [(robot, t0, t1)] botrail 이 충돌 검사하지 않는 램프 구간


class MujocoAdapter:
    def __init__(self, scene, sequences=None, out_dir="out/mujoco", *, timestep=0.001,
                 grasp_margin=0.6, approach_window=1.5, max_duration=300.0, tcp_payload_kg=0.0,
                 rollout_kwargs=None, allowances=(), torque_limits=True, track_tol_mm=20.0,
                 view=((2.9, -3.4, 2.6), (0.2, 0.2, 1.0))):
        self.scene = scene
        self.sequences = list(sequences or scene.sequence_names)
        if not self.sequences:
            raise UnsupportedCell("시퀀스가 없는 셀")
        if not scene.robots:
            raise UnsupportedCell("로봇이 없는 셀")
        self.out = Path(out_dir)
        self.timestep = timestep
        self.grasp_margin = grasp_margin
        self.approach_window = approach_window
        self.tcp_payload_kg = tcp_payload_kg
        self.view = view
        self.allowances = list(allowances)  # 추가 허용 접촉 (allowances.py 형식)
        self.torque_limits = torque_limits  # False: 모델 토크 한계 무시 (기구학·충돌만 검증)
        self.track_tol_mm = track_tol_mm    # TCP 추종 오차가 이보다 크면 verdict = "tracking"
        # 원래 베이크 인자 (dt, scenario, physics, ...) 그대로. botrail 물리 베이크로 설계된 셀은
        # 부품이 내려앉은 상태를 전제로 계획되므로 physics 를 빼면 계획이 달라진다(실패할 수 있다).
        self.rollout_kwargs = {k: v for k, v in (rollout_kwargs or {}).items() if v is not None}
        self.rollout_kwargs.setdefault("max_duration", max_duration)
        self.plan: Plan | None = None
        self.conv = None

    # -------------------------------------------------------------- 1. botrail 쪽
    def record_plan(self) -> Plan:
        """botrail 롤아웃(틱 단위)으로 계획 기록. 롤아웃이 계획에 실패하면(일괄 베이크와 다르게 동작하는 경우가 있음)
        일괄 베이크 타임라인에서 샘플링하는 경로로 대체한다."""
        # 실패한 롤아웃은 장면 상태를 바꿔 놓으므로, 일괄 베이크(기준 결과)를 먼저 확보해 둔다.
        tl = self.scene.simulate_sequences(self.sequences, **self.rollout_kwargs)
        try:
            plan = self._record_rollout()
        except ValueError as e:
            if "planning failed" not in str(e) and "collision" not in str(e):
                raise
            print(f"[botrail] rollout 실패 → 일괄 베이크 타임라인으로 대체: {str(e)[:160]}")
            plan = self._record_timeline(tl)
        # 램프(공구 스트로크 등)는 botrail 이 충돌 검사하지 않는 구간 — 그 동안의 접촉은 따로 분류
        for r in self.scene.robots:
            try:
                plan.ramps += [(r, a, b) for label, a, b in tl.moves(r) if label == "ramp"]
            except Exception:
                pass
        return plan

    def _record_timeline(self, tl) -> Plan:
        sc = self.scene
        kw = dict(self.rollout_kwargs)
        dt = kw.get("dt", 0.01)
        T = np.arange(0.0, tl.duration + 1e-9, dt)
        names = list(sc.obstacle_names)
        robots = list(sc.robots)
        rob = {r: sc.robot_of(r) for r in robots}
        BP, BQ = {r: [] for r in robots}, {r: [] for r in robots}
        for r in robots:
            for t in T:
                bpq = tl.base_pose(t, robot=r)
                if bpq is None:
                    bpq = sc.link_pose(rob[r].link_names[0], robot=r)
                BP[r].append(list(bpq[0]))
                BQ[r].append([bpq[1][3], bpq[1][0], bpq[1][1], bpq[1][2]])
        Q = {r: np.array([list(tl.sample(t, robot=r)) for t in T]) for r in robots}
        tips = {r: tips_of(rob[r]) for r in robots}
        TCP = {r: [] for r in robots}
        for k, t in enumerate(T):
            for r in robots:
                sc.set_joint_positions(list(Q[r][k]), robot=r)
            for r in robots:
                TCP[r].append([list(sc.link_pose(tip, robot=r)[0]) for tip in tips[r]])
        tracked, P, R = [], [], []
        for n in names:
            try:
                poses = [tl.object_pose(n, t) for t in T]
            except Exception:
                continue
            if any(p is None for p in poses):
                continue
            pos = np.array([p[0] for p in poses])
            quat = np.array([(p[1][3], p[1][0], p[1][1], p[1][2]) for p in poses])
            if np.ptp(pos, axis=0).max() > 1e-6 or np.ptp(quat, axis=0).max() > 1e-6:
                tracked.append(n)
                P.append(pos)
                R.append(quat)
        grasps = [{"object": g["object"], "robot": g.get("robot"), "start": g["start"], "end": g["end"]}
                  for g in tl.grasp_report()]
        try:
            clr = float(tl.min_clearance()) * 1e3
        except Exception:
            clr = None
        self.plan = Plan(
            dt=dt, t=T,
            robots=[_with_base(RobotPlan(r, list(rob[r].joint_names), tips[r], Q[r], np.array(TCP[r])), BP[r], BQ[r])
                    for r in robots],
            obj_pos=np.stack(P, axis=1) if P else np.zeros((len(T), 0, 3)),
            obj_quat=np.stack(R, axis=1) if R else np.zeros((len(T), 0, 4)),
            moving=tracked, grasps=grasps, steps=[[n, a, b] for n, a, b in tl.step_spans], clearance_mm=clr)
        self.plan_source = "timeline"
        return self.plan

    def _record_rollout(self) -> Plan:
        self.plan_source = "rollout"
        sc = self.scene
        names = list(sc.obstacle_names)
        robots = list(sc.robots)
        rob = {r: sc.robot_of(r) for r in robots}
        tips = {r: tips_of(rob[r]) for r in robots}
        # 1차: 10 틱마다 모든 장애물을 훑어 움직이는 것만 찾는다 (장애물이 수천 개인 셀에서 매 틱 전부 묻으면 너무 느림)
        ro = sc.open_rollout(self.sequences, **self.rollout_kwargs)
        p0 = {n: ro.object_pose(n) for n in names}
        moved = set()
        def scan():
            for n in names:
                if n in moved:
                    continue
                pos, quat = ro.object_pose(n)
                if max(abs(a - b) for a, b in zip(pos, p0[n][0])) > 1e-6 or \
                        max(abs(a - b) for a, b in zip(quat, p0[n][1])) > 1e-6:
                    moved.add(n)
        while not ro.finished:
            ro.tick(SCAN_TICKS)
            scan()
        ro.finish()
        movers = sorted(moved)
        # 2차: 같은 롤아웃(결정론적)을 다시 돌며 로봇과 움직이는 물체만 매 틱 기록
        ro = sc.open_rollout(self.sequences, **self.rollout_kwargs)
        dt = ro.dt
        T, P, R = [], [], []
        Q = {r: [] for r in robots}
        TCP = {r: [] for r in robots}
        BP = {r: [] for r in robots}
        BQ = {r: [] for r in robots}
        while True:
            T.append(ro.t)
            for r in robots:
                Q[r].append(list(ro.joint_positions(robot=r)))
                TCP[r].append([list(ro.link_pose(tip, robot=r)[0]) for tip in tips[r]])
                bp, bq = ro.link_pose(rob[r].link_names[0], robot=r)
                BP[r].append(list(bp))
                BQ[r].append([bq[3], bq[0], bq[1], bq[2]])
            poses = [ro.object_pose(n) for n in movers]
            P.append([p[0] for p in poses])
            R.append([(q[3], q[0], q[1], q[2]) for _, q in poses])  # xyzw → wxyz
            if ro.finished:
                break
            ro.tick()
        tl = ro.finish()
        idx = list(range(len(movers)))
        grasps = [{"object": g["object"], "robot": g.get("robot"), "start": g["start"], "end": g["end"]}
                  for g in tl.grasp_report()]
        try:
            clr = float(tl.min_clearance()) * 1e3
        except Exception:
            clr = None
        P, R = np.array(P), np.array(R)
        self.plan = Plan(
            dt=dt, t=np.array(T),
            robots=[_with_base(RobotPlan(r, list(rob[r].joint_names), tips[r], np.array(Q[r]), np.array(TCP[r])),
                               BP[r], BQ[r]) for r in robots],
            obj_pos=P[:, idx] if idx else np.zeros((len(T), 0, 3)),
            obj_quat=R[:, idx] if idx else np.zeros((len(T), 0, 4)),
            moving=sorted(moved), grasps=grasps, steps=[[n, a, b] for n, a, b in tl.step_spans], clearance_mm=clr)
        return self.plan

    def export_stage(self):
        """계획 시작 자세로 물리 USD 를 쓰고 MJCF 로 변환."""
        import botrail as bt
        self.out.mkdir(parents=True, exist_ok=True)
        for rp in self.plan.robots:
            self.scene.set_joint_positions(list(rp.q[0]), robot=rp.name)
        usd = self.out / "cell_physics.usda"
        for w in self.scene.export_usd(str(usd), physics=bt.Physics(world=True)):
            print("export warning:", w)
        roots = usd2mjcf.articulation_roots(usd)
        mobile = {}  # 아티큘레이션 루트 경로 → 트리를 시작할 링크 (botrail 로봇의 첫 링크)
        for k, rp in enumerate(self.plan.robots):
            if not rp.mobile:
                continue
            hit = [p for p in roots if usd2mjcf.robot_name_of(p) == rp.name] or \
                ([roots[k]] if len(roots) == len(self.plan.robots) else [])
            for h in hit:
                mobile[h] = self.scene.robot_of(rp.name).link_names[0]
        self.conv = usd2mjcf.convert(usd, self.out / "mjcf", self.plan.moving, timestep=self.timestep,
                                     mobile_roots=mobile)
        return self.conv

    def prepare(self):
        self.record_plan()
        self.export_stage()
        p = self.plan
        print(f"[botrail] {len(p.t)} ticks · {p.t[-1]:.2f}s · robots {[r.name + ('(mobile)' if r.mobile else '') for r in p.robots]} · "
              f"moving {len(p.moving)} · grasps {len(p.grasps)}")
        return self

    # -------------------------------------------------------------- 이름 대응
    def _articulation_of(self, name: str, k: int):
        roots = list(self.conv.robots.items())
        for path, info in roots:
            if usd2mjcf.robot_name_of(path) == name:
                return info
        if len(roots) == len(self.plan.robots):
            return roots[k][1]
        raise UnsupportedCell(f"로봇 '{name}' 의 아티큘레이션을 USD 에서 찾지 못함")

    @staticmethod
    def _joint_of(info, bt_name: str):
        """botrail 관절 이름 → MJCF 관절 이름. URDF('a/b' → 'a_b'), USD('/root/link/joint' → 상대경로)."""
        js = info["joints"]
        flat = bt_name.strip("/").replace("/", "_")
        tail = bt_name.strip("/").split("/", 1)[-1]
        last = bt_name.rsplit("/", 1)[-1]
        for test in (lambda j: j["prim"] == flat, lambda j: j["rel"].strip("/") == tail,
                     lambda j: j["rel"].endswith("/" + tail), lambda j: j["prim"] == last):
            hit = [j for j in js if test(j)]
            if len(hit) == 1:
                return hit[0]["name"]
        return None

    def _body_of(self, model, info, link: str):
        import mujoco
        for cand in (link.strip("/").replace("/", "_"), link.rsplit("/", 1)[-1]):
            b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, usd2mjcf.body_name(info["prefix"], cand))
            if b >= 0:
                return b
        return -1

    # -------------------------------------------------------------- 2. MuJoCo 쪽
    def run(self, record=False, web=True, fps=30):
        import mujoco
        p, conv = self.plan, self.conv
        model = mujoco.MjModel.from_xml_path(str(conv.xml_path))
        data = mujoco.MjData(model)
        oid = lambda kind, n: mujoco.mj_name2id(model, kind, n)

        R = []  # 로봇별 대응표
        for k, rp in enumerate(p.robots):
            info = self._articulation_of(rp.name, k)
            mj = [self._joint_of(info, n) for n in rp.joint_names]
            cols = [i for i, n in enumerate(mj) if n is not None]
            qadr = np.array([model.jnt_qposadr[oid(mujoco.mjtObj.mjOBJ_JOINT, mj[i])] for i in cols], int)
            act = np.array([oid(mujoco.mjtObj.mjOBJ_ACTUATOR, "a:" + mj[i]) for i in cols], int)
            tcp_bodies = [self._body_of(model, info, tip) for tip in rp.tips]
            for tb in tcp_bodies:
                if self.tcp_payload_kg and tb >= 0:
                    model.body_mass[tb] += self.tcp_payload_kg
                    model.body_inertia[tb] += 1e-3 * self.tcp_payload_kg
            R.append(dict(plan=rp, info=info, cols=np.array(cols, int), names=[mj[i] for i in cols], qadr=qadr,
                          act=act, has=act >= 0, tcp=tcp_bodies, bits=info["bits"],
                          vel=np.gradient(rp.q, p.t, axis=0) if len(p.t) > 1 else np.zeros_like(rp.q),
                          unmapped=[rp.joint_names[i] for i, n in enumerate(mj) if n is None]))
        for r in R:  # 이동형 베이스: 루트 mocap 을 계획 베이스 경로로 (첫 링크 → 루트 고정 오프셋)
            r["mocap"] = -1
            if r["info"].get("mobile"):
                rb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, r["info"]["root_link"])
                r["mocap"] = int(model.body_mocapid[rb])
                T_root0 = _T(model.body_pos[rb], model.body_quat[rb])
                T_b0 = _T(r["plan"].base_pos[0], r["plan"].base_quat[0])
                r["base_off"] = np.linalg.inv(T_b0) @ T_root0
        if self.tcp_payload_kg:
            mujoco.mj_setConst(model, data)
        placeholder = self._unhold_placeholder_limits(model, R) if self.torque_limits else {}
        if not self.torque_limits:
            model.actuator_forcelimited[:] = 0
        kvkp = np.zeros(model.nu)
        nz = model.actuator_gainprm[:, 0] != 0
        kvkp[nz] = -model.actuator_biasprm[nz, 2] / model.actuator_gainprm[nz, 0]

        robot_of_body = {}
        for k, r in enumerate(R):
            pre = r["info"]["prefix"]
            for b in range(1, model.nbody):
                nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
                if not nm.startswith("mc:") and (nm.startswith(pre) if pre else True):
                    robot_of_body.setdefault(b, k)
        moc = {n: model.body_mocapid[oid(mujoco.mjtObj.mjOBJ_BODY, conv.mocap[n])] for n in p.moving if n in conv.mocap}
        kidx = {n: p.moving.index(n) for n in moc}
        mgeoms = {n: [oid(mujoco.mjtObj.mjOBJ_GEOM, g) for g, col in conv.mocap_geoms[n] if col] for n in moc}
        rindex = {r["plan"].name: k for k, r in enumerate(R)}
        carried_of = {}  # mocap 이름 → [(t0, t1, 로봇 index)]
        for g in p.grasps:
            for n in moc:
                if g["object"] == n or g["object"].startswith(n + "/") or n.startswith(g["object"] + "/"):
                    carried_of.setdefault(n, []).append((g["start"], g["end"], rindex.get(g["robot"], 0)))

        # 접촉 분류용: geom → (로봇 index, 링크 이름) / 장애물 이름, 허용 접촉, 물리 동적 물체
        from .allowances import allowances_of
        geom_robot, geom_link, geom_obst = {}, {}, {}
        for g in range(model.ngeom):
            b = int(model.geom_bodyid[g])
            bn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
            if b in robot_of_body:
                pre = R[robot_of_body[b]]["info"]["prefix"]
                geom_robot[g] = robot_of_body[b]
                geom_link[g] = bn[len(pre):] if pre and bn.startswith(pre) else bn
            elif bn.startswith("mc:"):
                geom_obst[g] = bn[3:]
            else:
                geom_obst[g] = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        dynamic = set()
        if self.rollout_kwargs.get("physics"):
            try:
                dynamic = {str(n) for n in self.scene.physics_plan(self.rollout_kwargs["physics"]).dynamic}
            except Exception:
                dynamic = set()
        ctx = dict(robot=geom_robot, link=geom_link, obst=geom_obst, robots=[r["plan"].name for r in R],
                   allow=allowances_of(self.scene) + list(self.allowances), dynamic=dynamic, ramps=p.ramps,
                   mobile={k for k, r in enumerate(R) if r["mocap"] >= 0},
                   walkable=[e["name"] for e in allowances_of(self.scene)
                             if e["kind"] == "set_obstacle_walkable" and e.get("walkable", True)])
        initial = set()
        # 이동 로봇은 베이스가 경로를 강제로 따르므로 바닥·디딤판과의 접촉은 물리적으로 의미가 없다(바퀴·발이 마찰에
        # 붙잡혀 관절이 끌려감). 바닥류 geom 에 별도 비트를 줘서 고정 로봇과만 충돌하게 한다.
        if ctx["mobile"]:
            fixed_bits = sum(4 << k for k in range(len(R)) if k not in ctx["mobile"])
            def floorish(o):
                return o.startswith("Ground") or any(o == w or o.startswith(w + "/") for w in ctx["walkable"])
            for g, o in geom_obst.items():
                if model.geom_contype[g] and floorish(o):
                    model.geom_contype[g], model.geom_conaffinity[g] = FLOOR_BIT, fixed_bits

        for r in R:
            data.qpos[r["qadr"]] = r["plan"].q[0][r["cols"]]
        for a_, b_, c0, c1 in _equalities(model):
            data.qpos[a_] = c0 + c1 * data.qpos[b_]
        mujoco.mj_forward(model, data)
        hold = data.qpos.copy()
        for i in range(model.nu):  # botrail 이 명령하지 않는 구동 관절은 시작 위치 유지
            if model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT:
                data.ctrl[i] = hold[model.jnt_qposadr[model.actuator_trnid[i, 0]]]

        dt = model.opt.timestep
        steps = int(round(p.t[-1] / dt)) + 1
        frame_every = max(1, int(round(1 / fps / dt)))
        lim = np.where(model.actuator_forcelimited.astype(bool), model.actuator_forcerange[:, 1], np.inf)
        per = [dict(max_joint_err_rad=0.0, max_tcp_err_mm=0.0, worst_t=0.0, sum=0.0, n=0,
                    joint_err=np.zeros(len(r["cols"]))) for r in R]
        series, contacts = [], {}
        web_t, web_pos, web_quat, web_err = [], [], [], []
        renderer = cam = None
        frames = []
        if record:
            renderer = mujoco.Renderer(model, model.vis.global_.offheight, model.vis.global_.offwidth)
            cam = _free_camera(mujoco, *self.view)
        mode = {}

        for i in range(steps):
            if steps > 200_000 and i % (steps // 10) == 0:
                print(f"[mujoco] {100 * i // steps}% ({i * dt:.0f}/{p.t[-1]:.0f}s)", flush=True)
            t = i * dt
            f = min(t / p.dt, len(p.t) - 1.000001)
            k, a = int(f), f - int(f)
            for r in R:
                q = (1 - a) * r["plan"].q[k] + a * r["plan"].q[k + 1]
                v = (1 - a) * r["vel"][k] + a * r["vel"][k + 1]
                ai = r["act"][r["has"]]
                data.ctrl[ai] = (q[r["cols"]] + kvkp[r["act"]] * v[r["cols"]])[r["has"]] if len(ai) else data.ctrl[ai]
            for r in R:
                if r["mocap"] >= 0:
                    rp = r["plan"]
                    bq0, bq1 = rp.base_quat[k], rp.base_quat[k + 1]
                    bq = (1 - a) * bq0 + a * (bq1 if np.dot(bq0, bq1) >= 0 else -bq1)
                    Tb = _T((1 - a) * rp.base_pos[k] + a * rp.base_pos[k + 1], bq / np.linalg.norm(bq)) @ r["base_off"]
                    data.mocap_pos[r["mocap"]] = Tb[:3, 3]
                    data.mocap_quat[r["mocap"]] = usd2mjcf.quat_of(Tb[:3, :3])
            for n, mid in moc.items():
                j = kidx[n]
                data.mocap_pos[mid] = (1 - a) * p.obj_pos[k, j] + a * p.obj_pos[k + 1, j]
                q0, q1 = p.obj_quat[k, j], p.obj_quat[k + 1, j]
                qq = (1 - a) * q0 + a * (q1 if np.dot(q0, q1) >= 0 else -q1)
                data.mocap_quat[mid] = qq / np.linalg.norm(qq)
                holder = next((ri for t0, t1, ri in carried_of.get(n, [])
                               if t0 - self.grasp_margin <= t <= t1 + self.grasp_margin), None)
                if mode.get(n, "x") != holder:  # 잡는 동안: 든 로봇의 충돌 그룹 (그 로봇과는 무시, 환경과는 검사)
                    bits = usd2mjcf.ENV_BITS if holder is None else R[holder]["bits"]
                    for g in mgeoms[n]:
                        model.geom_contype[g], model.geom_conaffinity[g] = bits
                    mode[n] = holder
            mujoco.mj_step(model, data)

            for c in data.contact[:data.ncon]:
                if c.dist >= 0:
                    continue
                pair = (min(c.geom1, c.geom2), max(c.geom1, c.geom2))
                if t < 0.05:
                    initial.add(pair)  # 시작부터 겹친 쌍: 계획이 아니라 모델(볼록 껍질·장착) 문제
                kind = self._classify(c.geom1, c.geom2, ctx, mode, t, carried_of)
                if kind and kind.startswith("collision") and pair in initial:
                    kind = "initial_overlap"
                if kind:
                    g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or "?"
                    g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or "?"
                    e = contacts.setdefault(f"{kind}: " + " x ".join(sorted((g1, g2))), {"t": [], "max_depth_mm": 0.0})
                    e["max_depth_mm"] = round(max(e["max_depth_mm"], -c.dist * 1e3), 2)
                    if len(e["t"]) < 10 and (not e["t"] or t - e["t"][-1] > 0.5):
                        e["t"].append(round(t, 2))

            if i % frame_every == 0:
                row = [round(t, 2)]
                tcp_mm_all = []
                for r, st in zip(R, per):
                    q = (1 - a) * r["plan"].q[k] + a * r["plan"].q[k + 1]
                    h = r["has"]
                    ej = np.abs(data.qpos[r["qadr"]] - q[r["cols"]])
                    st["joint_err"] = np.maximum(st["joint_err"], ej)
                    err = float(ej[h].max()) if h.any() else 0.0
                    tcp_mm = 0.0
                    for ti, tb in enumerate(r["tcp"]):
                        if tb >= 0:
                            ref = (1 - a) * r["plan"].tcp[k, ti] + a * r["plan"].tcp[k + 1, ti]
                            tcp_mm = max(tcp_mm, float(np.linalg.norm(data.xpos[tb] - ref) * 1e3))
                    st["n"] += 1
                    st["sum"] += tcp_mm
                    st["max_joint_err_rad"] = max(st["max_joint_err_rad"], err)
                    if tcp_mm > st["max_tcp_err_mm"]:
                        st["max_tcp_err_mm"], st["worst_t"] = tcp_mm, round(t, 2)
                    tcp_mm_all.append(tcp_mm)
                    row += [round(err, 5), round(tcp_mm, 2)]
                if i % (frame_every * 3) == 0:
                    series.append(row if len(R) > 1 else row[:3])
                web_t.append(round(t, 4))
                web_pos.append(data.xpos.copy())
                web_quat.append(data.xquat.copy())
                web_err.append(round(max(tcp_mm_all), 2))
                if renderer:
                    renderer.update_scene(data, camera=cam)
                    frames.append(renderer.render().copy())

        robots_rep = {}
        for r, st in zip(R, per):
            robots_rep[r["plan"].name] = dict(
                max_tcp_err_mm=st["max_tcp_err_mm"], worst_t=st["worst_t"], mean_tcp_err_mm=st["sum"] / max(st["n"], 1),
                max_joint_err_rad=st["max_joint_err_rad"], unmapped_joints=r["unmapped"],
                joint_err_rad={n: round(float(e), 5) for n, e in zip(r["names"], st["joint_err"])},
                mobile=r["mocap"] >= 0,
                required_torque=self.inverse_dynamics(model, r, lim))
        warnings = []
        for r in R:
            nf = r["info"].get("mass_floored", 0)
            if nf:
                warnings.append(f"로봇 '{r['plan'].name}' 링크 {nf}개에 질량 정보가 없어 {usd2mjcf.MASS_FLOOR} kg 로 보정 — "
                                "필요 토크·동역학은 참고용 (URDF <inertial> / USD MassAPI 를 넣으면 정확해짐)")
            if any(tb < 0 for tb in r["tcp"]):
                warnings.append(f"로봇 '{r['plan'].name}' TCP 링크를 MJCF 에서 못 찾음: {r['plan'].tips}")
            if r["mocap"] >= 0:
                warnings.append(f"로봇 '{r['plan'].name}' 이동형 베이스: 베이스는 botrail 경로를 그대로 따름(차량·보행 동역학 없음). "
                                "바닥·계단 디딤판과의 접촉은 계산 제외, 역동역학 필요 토크에 베이스 가속도는 미포함")
            fixes = []
            if r["info"].get("gains_clamped"):
                fixes.append(f"PhysX 전용 고강성 드라이브 {len(r['info']['gains_clamped'])}개 → kp {usd2mjcf.KP_SAFE:g}")
            if r["info"].get("inertia_floored"):
                fixes.append(f"비현실적으로 작은 링크 관성 {r['info']['inertia_floored']}개 → 하한 적용")
            if r["info"].get("armature_floored"):
                fixes.append(f"아마추어 0 인 구동 관절 {r['info']['armature_floored']}개 → {usd2mjcf.ARMATURE_MIN}")
            if fixes:
                warnings.append(f"로봇 '{r['plan'].name}' MuJoCo 안정화 보정: " + "; ".join(fixes))
            if r["info"].get("ignored_mimic"):
                warnings.append(f"로봇 '{r['plan'].name}' 관절 종류와 축이 안 맞는 미믹 무시: {r['info']['ignored_mimic']}")
            if r["unmapped"]:
                warnings.append(f"로봇 '{r['plan'].name}' MJCF 에서 못 찾은 관절: {r['unmapped']}")
        for name, x in robots_rep.items():
            over = [f"{j} {v['ratio'] * 100:.0f}%" for j, v in x["required_torque"].items() if v["ratio"] and v["ratio"] > 1.0]
            if over:
                warnings.append(f"로봇 '{name}' 계획이 모델 토크 한계를 넘음 ({', '.join(over)}) — 추종 오차·충돌은 그 결과일 수 있음. "
                                "한계가 실제 값인지 확인하거나 --no-torque-limits 로 기구학만 검증")
        for name, joints in placeholder.items():
            warnings.append(f"로봇 '{name}' 관절 {joints} 의 토크 한계가 시작 자세의 중력 토크보다 작음 — 모델의 자리채움 값으로 "
                            "보고 한계 없이 실행 (URDF <limit effort> 확인 필요)")
        n_init = sum(1 for k in contacts if k.startswith("initial_overlap"))
        if n_init:
            warnings.append(f"시작부터 겹친 접촉 쌍 {n_init}개 — 오목 메쉬의 볼록 껍질화나 장착부 겹침일 가능성 (initial_overlaps 참고)")
        worst = max(robots_rep.values(), key=lambda x: x["max_tcp_err_mm"])
        hard = {k: v for k, v in contacts.items() if k.startswith("collision")}
        report = {
            "engine": f"MuJoCo {mujoco.__version__}",
            "sequences": self.sequences,
            "cycle_s": round(float(p.t[-1]), 3),
            "botrail_min_clearance_mm": p.clearance_mm,
            "tcp_payload_kg": self.tcp_payload_kg,
            "torque_limits": self.torque_limits,
            "physics_dt": dt,
            "max_tcp_err_mm": worst["max_tcp_err_mm"],
            "worst_t": worst["worst_t"],
            "mean_tcp_err_mm": float(np.mean([x["mean_tcp_err_mm"] for x in robots_rep.values()])),
            "max_joint_err_rad": max(x["max_joint_err_rad"] for x in robots_rep.values()),
            "verdict": "collision" if hard else ("tracking" if worst["max_tcp_err_mm"] > self.track_tol_mm else "pass"),
            "track_tol_mm": self.track_tol_mm,
            "plan_source": getattr(self, "plan_source", "rollout"),
            "warnings": warnings,
            "collisions": hard,
            "pregrasp_contacts": {k: v for k, v in contacts.items() if k.startswith("pregrasp")},
            "handling_contacts": {k: v for k, v in contacts.items() if k.startswith("handling")},
            "allowed_contacts": {k: v for k, v in contacts.items() if k.startswith("allowed")},
            "physics_contacts": {k: v for k, v in contacts.items() if k.startswith("physics")},
            "initial_overlaps": {k: v for k, v in contacts.items() if k.startswith("initial_overlap")},
            "process_ramp_contacts": {k: v for k, v in contacts.items() if k.startswith("process_ramp")},
            "support_contacts": {k: v for k, v in contacts.items() if k.startswith("support")},
            "robots": robots_rep,
            "required_torque": robots_rep[p.robots[0].name]["required_torque"] if len(R) == 1 else None,
            "moving_objects": p.moving,
            "grasps": p.grasps,
            "series_t_jointerr_tcpmm": series,
            "steps": p.steps,
        }
        (self.out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
        if web:
            from .outputs import export_web
            export_web(model, np.array(web_t), np.array(web_pos), np.array(web_quat), web_err, p.steps, self.out / "web")
            self.export_plan(fps)
        if frames:
            from .outputs import write_video
            write_video(frames, 1 / (frame_every * dt), self.out / "video.mp4")
        return report

    def _unhold_placeholder_limits(self, model, R):
        """시작 자세에서 중력만 버티는 데 필요한 토크보다 작은 토크 한계는 모델의 자리채움 값(URDF effort="1" 등)으로
        본다: 그 한계로는 로봇이 제 무게도 못 든다. 그런 액추에이터는 한계를 끄고 이름을 돌려준다."""
        import mujoco
        d = mujoco.MjData(model)
        out = {}
        for r in R:
            d.qpos[r["qadr"]] = r["plan"].q[0][r["cols"]]
            for a_, b_, c0, c1 in _equalities(model):
                d.qpos[a_] = c0 + c1 * d.qpos[b_]
            mujoco.mj_forward(model, d)
            for j, ai in enumerate(r["act"]):
                if ai < 0 or not model.actuator_forcelimited[ai]:
                    continue
                dof = model.jnt_dofadr[model.actuator_trnid[ai, 0]]
                if abs(d.qfrc_bias[dof]) > model.actuator_forcerange[ai, 1]:
                    out.setdefault(r["plan"].name, []).append(r["names"][j])
            # 중력 부하가 없는 관절(수평 트랙 셔틀 등): 1 N(·m) 이하 한계가 계획을 따르는 데 필요한 힘의 1/10 도 안 되면 자리채움
            limited = [j for j, ai in enumerate(r["act"]) if ai >= 0 and model.actuator_forcelimited[ai]
                       and model.actuator_forcerange[ai, 1] <= 1.0 and r["names"][j] not in out.get(r["plan"].name, [])]
            if limited:
                peak, _ = self._plan_peak_force(model, r)
                for j in limited:
                    if peak[j] > 10 * model.actuator_forcerange[r["act"][j], 1]:
                        out.setdefault(r["plan"].name, []).append(r["names"][j])
            # 같은 로봇에서 자리채움으로 판정된 값과 똑같은 한계(예: 전부 effort="1")도 자리채움으로 본다
            bad = {model.actuator_forcerange[r["act"][r["names"].index(n)], 1] for n in out.get(r["plan"].name, [])}
            for j, ai in enumerate(r["act"]):
                if ai >= 0 and model.actuator_forcelimited[ai] and model.actuator_forcerange[ai, 1] in bad:
                    model.actuator_forcelimited[ai] = 0
                    if r["names"][j] not in out.get(r["plan"].name, []):
                        out.setdefault(r["plan"].name, []).append(r["names"][j])
        return out

    def export_plan(self, fps=30, prefix=None):
        """botrail 계획 동작을 같은 MuJoCo 모델로 재생 데이터화 (동역학 없이 기구학만): 계획 관절값 + 계획된
        물체·베이스 포즈 → 바디 포즈. 웹 뷰어에서 'botrail 계획' 과 'MuJoCo 물리' 를 같은 화면에서 비교하는 데 쓴다.
        프레임 시각은 run() 의 웹 출력과 같다."""
        import mujoco
        from .outputs import export_poses
        p, conv = self.plan, self.conv
        model = mujoco.MjModel.from_xml_path(str(conv.xml_path))
        d = mujoco.MjData(model)
        oid = lambda kind, n: mujoco.mj_name2id(model, kind, n)
        maps = []
        for k, rp in enumerate(p.robots):
            info = self._articulation_of(rp.name, k)
            mj = [self._joint_of(info, n) for n in rp.joint_names]
            cols = [i for i, n in enumerate(mj) if n is not None]
            qadr = np.array([model.jnt_qposadr[oid(mujoco.mjtObj.mjOBJ_JOINT, mj[i])] for i in cols], int)
            m = dict(rp=rp, cols=np.array(cols, int), qadr=qadr, mocap=-1)
            if info.get("mobile"):
                rb = oid(mujoco.mjtObj.mjOBJ_BODY, info["root_link"])
                m["mocap"] = int(model.body_mocapid[rb])
                m["off"] = np.linalg.inv(_T(rp.base_pos[0], rp.base_quat[0])) @ _T(model.body_pos[rb], model.body_quat[rb])
            maps.append(m)
        moc = {n: model.body_mocapid[oid(mujoco.mjtObj.mjOBJ_BODY, conv.mocap[n])] for n in p.moving if n in conv.mocap}
        eqs = _equalities(model)
        dt = model.opt.timestep
        frame_every = max(1, int(round(1 / fps / dt)))
        steps = int(round(p.t[-1] / dt)) + 1
        T, POS, QUAT = [], [], []
        for i in range(0, steps, frame_every):
            t = i * dt
            f = min(t / p.dt, len(p.t) - 1.000001)
            k, a = int(f), f - int(f)
            for m in maps:
                rp = m["rp"]
                d.qpos[m["qadr"]] = ((1 - a) * rp.q[k] + a * rp.q[k + 1])[m["cols"]]
                if m["mocap"] >= 0:
                    bq0, bq1 = rp.base_quat[k], rp.base_quat[k + 1]
                    bq = (1 - a) * bq0 + a * (bq1 if np.dot(bq0, bq1) >= 0 else -bq1)
                    Tb = _T((1 - a) * rp.base_pos[k] + a * rp.base_pos[k + 1], bq / np.linalg.norm(bq)) @ m["off"]
                    d.mocap_pos[m["mocap"]] = Tb[:3, 3]
                    d.mocap_quat[m["mocap"]] = usd2mjcf.quat_of(Tb[:3, :3])
            for a_, b_, c0, c1 in eqs:
                d.qpos[a_] = c0 + c1 * d.qpos[b_]
            for n, mid in moc.items():
                j = p.moving.index(n)
                d.mocap_pos[mid] = (1 - a) * p.obj_pos[k, j] + a * p.obj_pos[k + 1, j]
                q0, q1 = p.obj_quat[k, j], p.obj_quat[k + 1, j]
                qq = (1 - a) * q0 + a * (q1 if np.dot(q0, q1) >= 0 else -q1)
                d.mocap_quat[mid] = qq / np.linalg.norm(qq)
            mujoco.mj_kinematics(model, d)
            T.append(round(t, 4))
            POS.append(d.xpos.copy())
            QUAT.append(d.xquat.copy())
        export_poses(np.array(T), np.array(POS), np.array(QUAT), prefix or self.out / "plan")

    def export_isaac_job(self, path=None, fps=30):
        """Isaac Sim 재현기(tools/isaac_replay.py)에 넘길 작업: 물리 USD, 로봇별 아티큘레이션·관절 계획,
        움직이는 장애물 포즈, 결과를 MuJoCo 바디 순서로 돌려받기 위한 바디 ↔ USD 경로 대응표."""
        import mujoco
        from .allowances import allowances_of
        p, conv = self.plan, self.conv
        model = mujoco.MjModel.from_xml_path(str(conv.xml_path))
        oid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
        robots, arrays, R = [], {}, []
        for k, rp in enumerate(p.robots):
            info = self._articulation_of(rp.name, k)
            root = next(r for r, i in conv.robots.items() if i is info)
            mj = [self._joint_of(info, n) for n in rp.joint_names]
            cols = [i for i, n in enumerate(mj) if n is not None]
            prims = {j["name"]: j["prim"] for j in info["joints"]}
            robots.append(dict(name=rp.name, root=root, mobile=bool(info.get("mobile")),
                               dofs=[prims[mj[i]] for i in cols], prim_of=prims))
            if info.get("mobile"):  # 이동형 베이스: Isaac 에서 이 링크를 플로팅 루트로 두고 계획 포즈(plan.json)로 옮긴다
                robots[-1].update(root_body=int(oid(info["root_link"])),
                                  root_link=info["body_paths"][info["root_link"]])
            arrays[f"q{k}"] = rp.q[:, cols]
            jid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, mj[i]) for i in cols]
            act = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "a:" + mj[i]) for i in cols], int)
            R.append(dict(plan=rp, cols=np.array(cols, int), names=[mj[i] for i in cols], act=act,
                          qadr=np.array([model.jnt_qposadr[j] for j in jid], int)))
        # MuJoCo 와 같은 규칙: 시작 자세 중력도 못 버티는 토크 한계(URDF effort="1" 같은 자리채움)는 Isaac 에서도 푼다
        placeholder = self._unhold_placeholder_limits(model, R)
        for r in robots:
            r["unlimit"] = [r["prim_of"][n] for n in placeholder.get(r["name"], [])]
            del r["prim_of"]
        bodies = {}
        for info in conv.robots.values():
            for name, usd in info.get("body_paths", {}).items():
                if oid(name) >= 0:
                    bodies[int(oid(name))] = usd
        for name, bname in conv.mocap.items():
            if oid(bname) >= 0:
                bodies[int(oid(bname))] = "/World/Env/" + name
        dt = model.opt.timestep
        frame_every = max(1, int(round(1 / fps / dt)))
        job = dict(usd=str((self.out / "cell_physics.usda").resolve()), dt=p.dt, t_end=float(p.t[-1]),
                   frame_dt=frame_every * dt, robots=robots, moving=[("/World/Env/" + n) for n in p.moving],
                   bodies=bodies, nbody=int(model.nbody),
                   floor=[("/World/Env/" + e["name"]) for e in allowances_of(self.scene)
                          if e["kind"] == "set_obstacle_walkable" and e.get("walkable", True)],
                   # botrail 이 허용한 링크 ↔ 장애물 접촉 (트랙에 박힌 셔틀, 공정 접촉): Isaac 에서는 접촉 쌍을 거른다
                   allowed=[dict(robot=e.get("robot"), link=str(e["link"]).rstrip("/").rsplit("/", 1)[-1],
                                 obstacle="/World/Env/" + str(e["obstacle"]))
                            for e in allowances_of(self.scene) + list(self.allowances)
                            if e["kind"] == "allow_link_obstacle_contact"])
        arrays.update(t=p.t, obj_pos=p.obj_pos, obj_quat=p.obj_quat)
        path = Path(path or self.out / "isaac_job.json")
        np.savez_compressed(path.with_suffix(".npz"), **arrays)
        path.write_text(json.dumps(job, ensure_ascii=False))
        return path

    def _classify(self, g1, g2, ctx, mode, t, carried_of):
        """접촉 종류. None = 무시. collision* 만 판정(verdict)에 들어간다.
        collision          로봇 ↔ 환경
        collision(robots)  로봇 ↔ 다른 로봇 (또는 다른 로봇이 든 물체)
        collision(carried) 든 물체 ↔ 환경, 운반 중
        allowed            셀이 botrail 에 허용한 접촉 (allow_link_obstacle_contact 등: 공정 접촉, 끼워넣기)
        physics            botrail 물리 베이크의 동적 물체와의 접촉 (botrail 도 충돌로 보지 않음)
        pregrasp           로봇 ↔ 곧 잡을/방금 놓은 물체 (파지 구간 ± approach_window)
        handling           든 물체 ↔ 놓인 자리 (집기/놓기 순간 ± grasp_margin)
        process_ramp       botrail 이 충돌 검사하지 않는 램프(공구 스트로크 등) 중 접촉
        support            이동 로봇(바퀴·발) ↔ 바닥·걸을 수 있는 면
        initial_overlap    t=0 부터 겹친 쌍 — 모델(볼록 껍질·장착) 문제, 경고만"""
        def side(g):
            if g in ctx["robot"]:
                return "robot", ctx["robot"][g], None
            o = ctx["obst"].get(g, "")
            h = mode.get(o, None) if o in mode else None
            return ("env", None, o) if h is None else ("carried", h, o)

        def is_obst(name, o):
            return bool(o) and (name == o or name.startswith(o + "/"))

        def short(link):
            return str(link).rstrip("/").rsplit("/", 1)[-1]

        (s1, r1, o1), (s2, r2, o2) = side(g1), side(g2)
        kinds = {s1, s2}
        allow = ctx["allow"]
        if kinds <= {"robot", "carried"}:
            if r1 == r2:
                return None
            la, lb = ctx["link"].get(g1, o1), ctx["link"].get(g2, o2)
            na, nb = ctx["robots"][r1], ctx["robots"][r2]
            for e in allow:
                if e["kind"] == "allow_inter_robot_collision":
                    x = (e["robot_a"], short(e["link_a"]), e["robot_b"], short(e["link_b"]))
                    if x in ((na, short(la), nb, short(lb)), (nb, short(lb), na, short(la))):
                        return "allowed"
            return "collision(robots)"
        env_o = o1 if s1 == "env" else o2
        if any(is_obst(env_o, d) for d in ctx["dynamic"]):
            return "physics"
        if kinds == {"robot", "env"}:
            g_r = g1 if s1 == "robot" else g2
            ri = r1 if s1 == "robot" else r2
            link, rname = ctx["link"].get(g_r, ""), ctx["robots"][ri]
            if ri in ctx["mobile"] and (env_o.startswith("Ground") or any(is_obst(env_o, w) for w in ctx["walkable"])):
                return "support"  # 이동 로봇의 바퀴·발 ↔ 바닥/계단 디딤판
            for e in allow:
                if e["kind"] == "allow_link_obstacle_contact" and short(e["link"]) == short(link) \
                        and is_obst(env_o, e["obstacle"]) and e.get("robot") in (None, rname):
                    return "allowed"
            if env_o and any(t0 - self.approach_window <= t <= t1 + self.approach_window
                             for t0, t1, _ in carried_of.get(env_o, [])):
                return "pregrasp"
            if any(rr == rname and a <= t <= b for rr, a, b in ctx["ramps"]):
                return "process_ramp"
            return "collision"
        if kinds == {"carried", "env"}:
            o = o1 if s1 == "carried" else o2
            for e in allow:
                if e["kind"] == "allow_object_obstacle_contact" and is_obst(o, e["object"]) and is_obst(env_o, e["obstacle"]):
                    return "allowed"
            m = self.grasp_margin + 0.3
            near = any(abs(t - t0) <= m or abs(t - t1) <= m for t0, t1, _ in carried_of.get(o, []))
            if near:
                return "handling"
            holder = ctx["robots"][r1 if s1 == "carried" else r2]
            if any(rr == holder and a <= t <= b for rr, a, b in ctx["ramps"]):
                return "process_ramp"
            return "collision(carried)"
        return None

    def _plan_peak_force(self, model, r):
        """계획 궤적을 정확히 따르는 데 필요한 관절별 최대 힘/토크와 그 시각 (mj_inverse, 제약력 제외)"""
        import mujoco
        p = self.plan
        rp = r["plan"]
        d = mujoco.MjData(model)
        v = np.gradient(rp.q, p.t, axis=0)
        acc = np.gradient(v, p.t, axis=0)
        jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in r["names"]]
        dofadr = np.array([model.jnt_dofadr[j] for j in jids], int)
        peak = np.zeros(len(jids))
        at = np.zeros(len(jids))
        eqs = _equalities(model)
        saved = model.opt.disableflags
        model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
        try:
            for k in range(len(p.t)):
                d.qpos[r["qadr"]] = rp.q[k][r["cols"]]
                d.qvel[dofadr] = v[k][r["cols"]]
                d.qacc[dofadr] = acc[k][r["cols"]]
                for a_, b_, c0, c1 in eqs:
                    d.qpos[a_] = c0 + c1 * d.qpos[b_]
                mujoco.mj_inverse(model, d)
                tau = np.abs(d.qfrc_inverse[dofadr])
                better = tau > peak
                peak[better], at[better] = tau[better], p.t[k]
        finally:
            model.opt.disableflags = saved
        return peak, at

    def inverse_dynamics(self, model, r, lim):
        """계획 궤적(q, q̇, q̈)을 정확히 따르는 데 필요한 관절 토크 (mj_inverse, 제약력 제외).
        PD 튜닝과 무관한 모터 요구치. 든 물체 질량은 mocap 이라 포함되지 않음 (--payload 로 보정)."""
        import mujoco
        jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in r["names"]]
        peak, at = self._plan_peak_force(model, r)
        out = {}
        for j, (n, jid) in enumerate(zip(r["names"], jids)):
            if not r["has"][j] or model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
                continue
            L = lim[r["act"][j]]
            out[n] = dict(peak=round(float(peak[j]), 2), at_s=round(float(at[j]), 2),
                          limit=None if math.isinf(L) else float(L),
                          ratio=None if math.isinf(L) else round(float(peak[j] / L), 3))
        return out


# ------------------------------------------------------------------ 도우미
def _with_base(rp, bp, bq):
    rp.base_pos, rp.base_quat = np.array(bp, float), np.array(bq, float)
    rp.mobile = bool(np.ptp(rp.base_pos, axis=0).max() > 1e-3 or np.ptp(rp.base_quat, axis=0).max() > 1e-4)
    return rp


def _T(pos, quat_wxyz):
    return usd2mjcf.H(np.asarray(pos, float), np.asarray(quat_wxyz, float))


def _equalities(model):
    import mujoco
    out = []
    for e in range(model.neq):
        if model.eq_type[e] == mujoco.mjtEq.mjEQ_JOINT:
            a = model.jnt_qposadr[model.eq_obj1id[e]]
            b = model.jnt_qposadr[model.eq_obj2id[e]]
            out.append((a, b, model.eq_data[e, 0], model.eq_data[e, 1]))
    return out


def _free_camera(mujoco, eye, target):
    eye, target = np.array(eye, float), np.array(target, float)
    d = target - eye
    cam = mujoco.MjvCamera()
    cam.lookat[:] = target
    cam.distance = float(np.linalg.norm(d))
    cam.azimuth = math.degrees(math.atan2(d[1], d[0]))
    cam.elevation = math.degrees(math.asin(d[2] / np.linalg.norm(d)))
    return cam
