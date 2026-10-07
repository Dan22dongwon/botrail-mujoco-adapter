"""일괄 시험 결과(out/survey) → 공유용 정적 사이트(out/site).

  python tools/build_site.py out/survey out/site [--fps 10] [--max-mb 14]

- 각 예제의 MuJoCo 재생 데이터(web.json/.bin)를 공유용으로 줄인다 (프레임 수 축소, 파일당 max-mb 이하).
- data/index.json: 예제 목록 (분야, 한 줄 설명, 쉬운 말 결과, 핵심 수치, 썸네일 경로)
- index.html: 사이트 페이지 (botrail_mujoco/viewer/site.html 를 감싸서), artifact.html: 감싸지 않은 본문
썸네일(thumbs/<key>.jpg)은 tools/make_thumbnails.py 가 만든다.
"""
import argparse
import base64
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

CATS = [
    ("머신 텐딩", ["machining/machine_tending_demo", "machining/two_machine_cell_demo"]),
    ("로봇 가공", ["machining/machining_demo", "machining/ati_deburring_demo", "machining/spindle_mounting_demo"]),
    ("조립", ["assembly/cover_bolting_demo", "assembly/shuttle_line_demo"]),
    ("용접 (BIW)", ["welding/weld_station_demo", "welding/weld_line_demo", "welding/line_balance_sweep",
                  "welding/nimak_spot_welding_demo", "welding/spot_gun_mounting_demo"]),
    ("도장", ["painting/painting_demo", "painting/painting_hood_demo"]),
    ("팔레타이징·컨베이어", ["palletizing/palletizing_line_demo", "basics/sequence_demo", "basics/physics_conveyor"]),
    ("물류 (AMR·AGV·AGF)", ["vehicles/warehouse_demo", "vehicles/agv_cell_demo", "vehicles/agv_sweep_demo",
                          "vehicles/forklift_demo", "vehicles/lift_demo", "vehicles/amr_demo",
                          "vehicles/semi_humanoid_demo", "drone/drone_survey_demo"]),
    ("레그드·휴머노이드", ["legged/legged_patrol_demo", "legged/stairs_delivery_demo", "legged/building_delivery_demo",
                     "legged/humanoid_carry_demo", "legged/wash_inspect_ship_demo"]),
    ("멀티 로봇", ["multi_robot/dual_arm_demo", "multi_robot/dual_cell_demo"]),
    ("그리핑", ["basics/gripper_pick_demo", "basics/friction_grasp_demo", "basics/hand_grasp_demo",
             "basics/physics_pick_place", "basics/physics_drop", "basics/physics_world_demo"]),
    ("셀 엔지니어링", ["basics/demo", "basics/sfc_chart_demo", "basics/sweep_demo", "engineering/cell_deliverables_demo",
                   "engineering/equipment_cell_demo", "engineering/urplus_products"]),
    ("강화학습", ["rl/reach_control_demo", "rl/reach_env", "rl/tabletop_env", "rl/torque_env", "rl/depth_pick_env",
              "rl/policy_cell_demo"]),
    ("Export", ["export/export_urscript", "export/export_animation", "export/isaaclab_cell",
                "export/isaaclab_tabletop", "export/play_record"]),
]
TITLE = {
    "machining/machine_tending_demo": "CNC 머신 텐딩 (VMC 로딩/언로딩)",
    "machining/two_machine_cell_demo": "1로봇 2설비 머신 텐딩",
    "machining/machining_demo": "로봇 엣지 트리밍",
    "machining/ati_deburring_demo": "ATI 디버링 툴 알루미늄 디버링",
    "machining/spindle_mounting_demo": "스핀들 마운팅 검증",
    "assembly/cover_bolting_demo": "커버 압입·볼트 체결",
    "assembly/shuttle_line_demo": "리니어 셔틀 트랙 조립 라인",
    "welding/weld_station_demo": "BIW 스폿 용접 스테이션",
    "welding/weld_line_demo": "BIW 용접 라인",
    "welding/line_balance_sweep": "용접 라인 밸런싱 (택트 스윕)",
    "welding/nimak_spot_welding_demo": "NIMAK 스폿건 2타점 용접",
    "welding/spot_gun_mounting_demo": "스폿건 마운팅 검증",
    "painting/painting_demo": "패널 스프레이 도장",
    "painting/painting_hood_demo": "후드 곡면 도장",
    "palletizing/palletizing_line_demo": "EOL 팔레타이징·스트레치 랩핑",
    "basics/sequence_demo": "컨베이어 트래킹 픽앤플레이스",
    "basics/physics_conveyor": "마찰 컨베이어 반송",
    "vehicles/warehouse_demo": "창고 AMR 리프트 반송",
    "vehicles/agv_cell_demo": "AGV 도킹·셀 반송",
    "vehicles/agv_sweep_demo": "AGV 도착시간 파라미터 스윕",
    "vehicles/forklift_demo": "AGF 랙 입출고",
    "vehicles/lift_demo": "AMR 엘리베이터 층간 반송",
    "vehicles/amr_demo": "AMR 모바일 매니퓰레이터",
    "vehicles/semi_humanoid_demo": "휠베이스 휴머노이드 선반 피킹",
    "drone/drone_survey_demo": "드론 재고 실사",
    "legged/legged_patrol_demo": "4족 보행 순찰",
    "legged/stairs_delivery_demo": "4족 보행 계단 반송",
    "legged/building_delivery_demo": "4족 보행 건물 내 배송",
    "legged/humanoid_carry_demo": "휴머노이드 부품 반송",
    "legged/wash_inspect_ship_demo": "휴머노이드 세척·검사·출하",
    "multi_robot/dual_arm_demo": "듀얼암 키팅",
    "multi_robot/dual_cell_demo": "2로봇 공유 투입구 인터록",
    "basics/gripper_pick_demo": "평행 그리퍼 픽앤플레이스",
    "basics/friction_grasp_demo": "마찰 파지",
    "basics/hand_grasp_demo": "다지 핸드 파지",
    "basics/physics_pick_place": "동역학 픽앤플레이스",
    "basics/physics_drop": "강체 낙하·적재",
    "basics/physics_world_demo": "셀 전역 중력 동역학",
    "basics/demo": "IK·경로 계획",
    "basics/sfc_chart_demo": "SFC 시퀀스 픽 셀",
    "basics/sweep_demo": "사이클타임 파라미터 스윕",
    "engineering/cell_deliverables_demo": "셀 산출물 (도면·BOM·I/O)",
    "engineering/equipment_cell_demo": "카탈로그 설비 셀 구성",
    "engineering/urplus_products": "UR+ 주변기기 장착 검토",
    "rl/reach_control_demo": "Reach 제어기",
    "rl/reach_env": "Reach RL 환경",
    "rl/tabletop_env": "테이블탑 RL 환경",
    "rl/torque_env": "토크 제어 RL 환경",
    "rl/depth_pick_env": "Depth 기반 피킹 RL",
    "rl/policy_cell_demo": "학습 정책 셀 배치",
    "export/export_urscript": "URScript 오프라인 프로그램",
    "export/export_animation": "USD 애니메이션",
    "export/isaaclab_cell": "Isaac Lab 셀",
    "export/isaaclab_tabletop": "Isaac Lab 테이블탑",
    "export/play_record": "기록 재생",
}
WHY_NOT = {
    "unsupported": "로봇 없음 (강체 동역학 단독)",
    "external-control": "외부 정책 제어 (모션 시퀀스 없음)",
    "no-sequence": "모션 시퀀스 없음",
    "timeout": "변환 시간 초과",
}


def key_of(example):
    return example.removesuffix(".py")


def cat_of(example):
    k = key_of(example)
    return next((c for c, xs in CATS if k in xs), "기타")


def plain_summary(r, rep):
    """판정 → (레벨, 라벨, 지표 한 줄)"""
    tcp, mean = rep["max_tcp_err_mm"], rep["mean_tcp_err_mm"]
    coll = rep.get("collisions", {})
    depth = max((v["max_depth_mm"] for v in coll.values()), default=0)
    line = f"TCP 추종오차 평균 {mean:.1f} / 최대 {tcp:.1f} mm · 간섭 {len(coll)}건" + (f" (최대 관입 {depth:.1f} mm)" if coll else "")
    if r["status"] == "ok":
        return "good", "PASS", line
    if r["status"] == "ok-collision":
        return "check", "간섭", line
    return "check", "추종 이탈", line


def decimate(v, f, cell_mm):
    """정점 클러스터링: cell_mm 격자 안의 정점을 하나로 합치고 퇴화한 면을 버린다 (보기용 단순화)."""
    if cell_mm <= 0 or len(v) < 200:
        return v, f
    q = np.floor(v / (cell_mm / 1000.0)).astype(np.int64)
    _, inv, counts = np.unique(q, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    nv = np.zeros((len(counts), 3))
    np.add.at(nv, inv, v)
    nv /= counts[:, None]
    nf = inv[f]
    keep = (nf[:, 0] != nf[:, 1]) & (nf[:, 1] != nf[:, 2]) & (nf[:, 0] != nf[:, 2])
    nf = nf[keep]
    if len(nf) == 0:
        return v, f
    return nv, nf


def shrink(src_json, src_bin, dst_json, dst_bin, fps, max_mb, cell_mm=3.0):
    meta = json.loads(src_json.read_text())
    buf = src_bin.read_bytes()
    t = np.array(meta["t"])
    nb = len(meta["moving"])
    pos = np.frombuffer(buf, np.float32, meta["pos"][1], meta["pos"][0]).reshape(len(t), nb * 3)
    quat = np.frombuffer(buf, np.float32, meta["quat"][1], meta["quat"][0]).reshape(len(t), nb * 4)
    src_fps = (len(t) - 1) / max(t[-1], 1e-9)
    mesh_bytes = sum(m["v"][1] * 4 + m["f"][1] * 4 for m in meta["meshes"].values()) * 0.9  # 단순화가 잘 안 되는 메쉬(차체 패널) 대비 보수적으로
    # 비교 트랙(plan/isaac) 바디 수. 차체(BIW)처럼 리지드 바디가 수백 패널로 쪼개진 셀은 여기가 폭증한다.
    cmp_nb = sum(len(json.loads(tj.read_text()).get("moving", []))
                 for tj in (src_json.with_name("plan.json"), src_json.with_name("isaac.json")) if tj.exists())
    drop_cmp = cmp_nb > 80  # 초복잡 셀: 비교 트랙 생략 → 바디 수·용량 급감, MuJoCo 3D 는 정상 렌더 (deviation 히트맵만 없음)
    track_nb = nb + (0 if drop_cmp else cmp_nb)
    while True:
        step = max(1, int(round(src_fps / fps)))
        idx = np.arange(0, len(t), step)
        if idx[-1] != len(t) - 1:
            idx = np.append(idx, len(t) - 1)
        size = mesh_bytes + len(idx) * track_nb * 7 * 4  # MuJoCo + botrail 계획 + Isaac (실제 바디 수 합)
        if size <= max_mb * 1e6 or fps <= 2:
            break
        fps /= 2
    chunks, off = [], 0

    def put(arr):
        nonlocal off
        b = np.ascontiguousarray(arr).tobytes()
        chunks.append(b)
        start, off = off, off + len(b)
        return [start, int(np.asarray(arr).size)]

    meshes = {}
    for k, m in meta["meshes"].items():
        v = np.frombuffer(buf, np.float32, m["v"][1], m["v"][0]).reshape(-1, 3)
        f = np.frombuffer(buf, np.uint32, m["f"][1], m["f"][0]).reshape(-1, 3)
        v, f = decimate(v, f, cell_mm)
        meshes[k] = dict(v=put(v.astype(np.float32).ravel()), f=put(f.astype(np.uint32).ravel()))
    out = dict(meta, t=[round(float(x), 3) for x in t[idx]], meshes=meshes,
               pos=put(pos[idx].astype(np.float32)), quat=put(quat[idx].astype(np.float32)),
               tcp_mm=[meta["tcp_mm"][i] for i in idx])
    # 비교용 트랙: botrail 계획(plan) · Isaac Sim(isaac). 같은 MuJoCo 바디 순서, 웹 프레임 시각으로 맞춘다.
    for key, fname in (("plan", "plan.json"), ("isaac", "isaac.json")):
        if drop_cmp:
            break
        tj = src_json.with_name(fname)
        if not tj.exists():
            continue
        tm = json.loads(tj.read_text())
        tb = tj.with_suffix(".bin").read_bytes()
        n = len(tm["moving"])
        if not n:
            continue
        tt = np.array(tm["t"])
        tp = np.frombuffer(tb, np.float32, tm["pos"][1], tm["pos"][0]).reshape(len(tt), n, 3)
        tq = np.frombuffer(tb, np.float32, tm["quat"][1], tm["quat"][0]).reshape(len(tt), n, 4)
        want = t[idx]
        f = np.clip(np.interp(want, tt, np.arange(len(tt))), 0, len(tt) - 1.000001)
        k = f.astype(int); w = (f - k)[:, None, None]
        k1 = np.minimum(k + 1, len(tt) - 1)
        rp = (1 - w) * tp[k] + w * tp[k1]
        q0, q1 = tq[k], tq[k1]
        q1 = np.where((np.sum(q0 * q1, axis=2, keepdims=True) < 0), -q1, q1)
        rq = (1 - w) * q0 + w * q1
        rq /= np.linalg.norm(rq, axis=2, keepdims=True)
        out[key] = dict(moving=tm["moving"], static=tm["static"],
                        pos=put(rp.reshape(len(want), n * 3).astype(np.float32)),
                        quat=put(rq.reshape(len(want), n * 4).astype(np.float32)))
    out.pop("bin", None)
    # 공유 페이지(claude.ai 아티팩트)는 바이너리 파일을 받지 않으므로 base64 텍스트로 쓴다
    dst_bin.with_suffix(".bin.txt").write_text(base64.b64encode(b"".join(chunks)).decode())
    dst_json.write_text(json.dumps(out, separators=(",", ":")))
    return dst_bin.with_suffix(".bin.txt").stat().st_size, round(fps, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("survey")
    ap.add_argument("site")
    ap.add_argument("--fps", type=float, default=10)
    ap.add_argument("--max-mb", type=float, default=8)  # 무거운 멀티로봇 셀(차체 패널 다수)은 자동 fps↓ (브라우저 3D 과부하 방지)
    ap.add_argument("--cell-mm", type=float, default=3.0, help="메쉬 단순화 격자 (0 이면 원본)")
    a = ap.parse_args()
    survey, site = Path(a.survey), Path(a.site)
    (site / "data").mkdir(parents=True, exist_ok=True)
    (site / "thumbs").mkdir(exist_ok=True)
    rows = json.loads((survey / "survey.json").read_text())
    items, total = [], 0
    for r in rows:
        k = key_of(r["example"])
        slug = k.replace("/", "__")
        item = dict(id=slug.split("__")[-1], key=k, category=cat_of(r["example"]),
                    title=TITLE.get(k, k.split("/")[-1]), example=r["example"])
        mj = survey / slug / "mj"
        if r["status"].startswith("ok") and (mj / "web.json").exists():
            rep = json.loads((mj / "report.json").read_text())
            size, fps = shrink(mj / "web.json", mj / "web.bin", site / "data" / f"{slug}.json",
                               site / "data" / f"{slug}.bin", a.fps, a.max_mb, a.cell_mm)
            total += size
            level, label, sentence = plain_summary(r, rep)
            isaac = {"supported": False, "reason": "Isaac Sim 결과가 아직 없어요"}
            if (mj / "isaac_report.json").exists():
                ir = json.loads((mj / "isaac_report.json").read_text())
                if ir.get("supported"):
                    # 계속 도는 관절(바퀴·로터)은 각도 차이가 의미 없어 제외 (계획 범위가 반 바퀴 넘는 관절)
                    spin = set()
                    if (mj / "isaac_job.json").exists():
                        job = json.loads((mj / "isaac_job.json").read_text())
                        arr = np.load(mj / "isaac_job.npz")
                        for k, rj in enumerate(job["robots"]):
                            q = arr[f"q{k}"]
                            spin |= {(rj["name"], n) for c, n in enumerate(rj["dofs"]) if c < q.shape[1] and np.ptp(q[:, c]) > np.pi}
                    errs = [e for rn, x in ir["robots"].items() for n, e in x.get("joint_err_rad", {}).items() if (rn, n) not in spin]
                    err = max(errs, default=0.0)
                    over = [dict(joint=n, plan=v["plan_peak"], limit=v["limit"], unit=v.get("unit", ""))
                            for x in ir["robots"].values() for n, v in x.get("plan_exceeds_velocity_limit", {}).items()
                            if (n not in {sn for _, sn in spin})]
                    isaac = {"supported": True, "max_joint_err_deg": round(float(np.degrees(err)), 3),
                             "mobile": any(x.get("mobile_base") for x in ir["robots"].values()),
                             "spin_excluded": len(spin), "over_speed": over}
                else:
                    isaac = {"supported": False, "reason": ir.get("reason", "")}
            item.update(
                ready=True, level=level, label=label, summary=sentence, data=f"data/{slug}", fps=fps, isaac=isaac,
                thumb=f"thumbs/{slug}.jpg", cycle_s=rep["cycle_s"], robots=len(rep["robots"]),
                mobile=any(x.get("mobile") for x in rep["robots"].values()),
                detail=dict(
                    max_tcp_err_mm=round(rep["max_tcp_err_mm"], 2), mean_tcp_err_mm=round(rep["mean_tcp_err_mm"], 2),
                    collisions={kk.split(": ", 1)[1]: vv for kk, vv in list(rep["collisions"].items())[:8]},
                    torque={n: {j: v for j, v in x["required_torque"].items() if v.get("ratio") is not None}
                            for n, x in rep["robots"].items()},
                    warnings=rep.get("warnings", []),
                ))
        else:
            item.update(ready=False, reason="MuJoCo·Isaac 미변환 — " + WHY_NOT.get(r["status"], r["detail"]))
        # 엔진별 자체 렌더 영상: botrail 스튜디오 녹화 · MuJoCo 렌더러 · Isaac RTX
        vids = {}
        for eng in ("botrail", "mujoco", "isaac"):
            src = survey / slug / "video" / f"{eng}.mp4"
            if src.exists() and src.stat().st_size > 10_000:
                dst = site / "video" / slug / f"{eng}.mp4"
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
                    shutil.copy2(src, dst)
                vids[eng] = f"video/{slug}/{eng}.mp4"
                total += src.stat().st_size
        item["videos"] = vids
        if vids:
            meta = survey / slug / "video" / "botrail.json"
            item["video_s"] = json.loads(meta.read_text())["duration"] if meta.exists() else item.get("cycle_s")
            src = survey / slug / "video" / ("isaac.mp4" if "isaac" in vids else "botrail.mp4")
            thumb = site / "thumbs" / f"{slug}.jpg"
            dur = item.get("video_s") or 6
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{dur * 0.45:.2f}", "-i", str(src), "-frames:v", "1",
                            "-vf", "scale=480:-2", "-q:v", "4", str(thumb)], check=False)
            item["thumb"] = f"thumbs/{slug}.jpg"
        items.append(item)
    order = [c for c, _ in CATS] + ["기타"]
    items.sort(key=lambda x: (order.index(x["category"]), not x["ready"], x["title"]))
    (site / "data" / "index.json").write_text(json.dumps(dict(categories=order, items=items), ensure_ascii=False))
    body = (ROOT / "botrail_mujoco" / "viewer" / "site.html").read_text()
    (site / "artifact.html").write_text(body)
    (site / "index.html").write_text(
        '<!doctype html>\n<html lang="ko"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"></head><body>\n'
        + body + "\n</body></html>\n")
    n = sum(1 for x in items if x.get("ready"))
    print(f"{n}/{len(items)} 예제, 재생 데이터 {total / 1e6:.1f} MB → {site}")


if __name__ == "__main__":
    main()
