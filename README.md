# botrail-mujoco

**botrail 로봇 셀을 MuJoCo 물리로 돌리는 어댑터.**
botrail 이 만든 셀(PLC 시퀀스·장치·센서·모션 계획)을 그대로 두고, 그 계획을 MuJoCo 가 질량·관성·토크 한계를 가진
로봇으로 실제로 따라가게 해서 *"계획이 물리적으로도 되는가"* 를 검사합니다.

> Run [botrail](https://github.com/botrail/botrail) robot cells on [MuJoCo](https://mujoco.org) physics:
> botrail keeps the logic and the plan, MuJoCo plays the plant — tracking error, collisions (robot↔scene,
> robot↔robot, carried part↔scene), and the joint torques the plan really needs.

```
botrail (논리·계획)                        botrail-mujoco                         MuJoCo (물리)
─────────────────────                     ─────────────────────                 ─────────────────────
Scene · 시퀀스 · 장치 · 센서 ──bake/rollout──▶ 계획 기록 (10 ms 틱)                 로봇 = 아티큘레이션
                                           · 로봇 관절값 / TCP                     (질량·관성·아마추어,
scene.export_usd(physics=…) ──────────────▶ · 움직이는 장애물 포즈        ───────▶  PD + 토크 한계)
                                           · 파지 구간 (grasp_report)              문·부품·액추에이터 = mocap
                                           USD → MJCF 변환                         1 ms 스텝
                                                                                   ▼
                                           report.json · web 3D · video.mp4  ◀─── 추종 오차 · 충돌 · 필요 토크
```

## 설치

```bash
pip install botrail mujoco usd-core numpy pillow
pip install -e .            # 이 저장소
```

## 사용

```bash
# 1) botrail 예제/셀 스크립트를 수정 없이: 처음 베이크하는 셀과 시퀀스를 가로채 변환
botrail-mujoco path/to/botrail/examples/machining/machine_tending_demo.py --out out/tending --record

# 2) 셀을 돌려주는 함수 지정 (Scene 또는 (Scene, ...) 반환)
botrail-mujoco my_cell.py:build_cell --sequences cycle --out out/my_cell --payload 1.5

# 결과를 브라우저 3D 로 보기
botrail-mujoco-view out/tending        # → http://127.0.0.1:8790/

# botrail 예제 전체를 변환하고 갤러리로 보기
python tools/survey_examples.py path/to/botrail/examples out/survey -j 4
botrail-mujoco-view out/survey --host 0.0.0.0   # 예제 목록 + MuJoCo 3D (전문가용 상세 갤러리)

# 누구나 보는 공유용 사이트 (썸네일 카드 + 3D 플레이어, 쉬운 말 결과)
python tools/build_site.py out/survey out/site          # 재생 데이터 축소(10 fps, 메쉬 단순화) + 목록
python tools/make_thumbnails.py out/site -j 4           # 카드 썸네일 (위 뷰어 서버가 떠 있어야 함)
python -m http.server -d out/site 8791                  # 로컬 확인 — out/site/artifact.html 은 claude.ai 공유 페이지용 본문
tools/publish_pages.sh                                  # 공개 사이트 배포 → https://dan22dongwon.github.io/botrail-mujoco-gallery/
```

Python:

```python
from botrail_mujoco import MujocoAdapter
report = MujocoAdapter(scene, ["cycle"], "out/cell", tcp_payload_kg=1.0).prepare().run(record=True)
print(report["verdict"], report["max_tcp_err_mm"], report["collisions"])
```

| 옵션 | 뜻 |
|---|---|
| `--sequences a,b` | 돌릴 시퀀스 (기본: 셀의 전부, 또는 스크립트가 베이크한 것) |
| `--payload KG` | TCP 에 더할 질량. 그리퍼·공작물 질량이 모델에 없을 때 보정 |
| `--timestep S` | MuJoCo 스텝 (기본 0.001) |
| `--record` | `video.mp4` (오프스크린 EGL 렌더) |
| `--no-torque-limits` | 모델 토크 한계 무시 — 기구학·충돌만 검증 |
| `--view ex,ey,ez,tx,ty,tz` | 영상 카메라 |

## 결과 (`<out>/`)

| 파일 | 내용 |
|---|---|
| `report.json` | `verdict`, 로봇별 TCP/관절 추종 오차, `collisions`, `pregrasp_contacts`, `handling_contacts`, 로봇별 `required_torque`(역동역학, 한계 대비 비율), `warnings` |
| `mjcf/model.xml` | 변환된 MuJoCo 모델 (메쉬 포함) — `python -m mujoco.viewer --mjcf ...` 로 열 수 있음 |
| `cell_physics.usda` | botrail 이 쓴 물리 USD (변환 원본) |
| `web.json/.bin` | 웹 3D 뷰어 데이터 (MuJoCo 가 계산한 바디 포즈) |
| `video.mp4` | `--record` 시 |

접촉 분류

| 종류 | 뜻 |
|---|---|
| `collision` | 로봇 ↔ 환경 |
| `collision(robots)` | 로봇 ↔ 다른 로봇 (또는 다른 로봇이 든 물체) |
| `collision(carried)` | 운반 중인 물체 ↔ 환경 |
| `pregrasp` | 로봇 ↔ 곧 잡을/방금 놓은 물체 (접근·이탈 중), 침투 깊이 기록 |
| `handling` | 든 물체 ↔ 놓인 자리 (집기/놓기 순간) |

| `allowed` | 셀이 botrail 에 허용한 접촉 (`allow_link_obstacle_contact` 등 — 커터↔소재, 비트↔나사) |
| `process_ramp` | botrail 이 충돌 검사하지 않는 램프(공구 스트로크) 중 접촉 |
| `physics` | botrail 물리 베이크의 동적 물체와의 접촉 |
| `initial_overlap` | 시작부터 겹친 쌍 — 오목 메쉬의 볼록 껍질화·장착부 (경고) |

`verdict`: `collision*` 이 있으면 `"collision"`, 아니고 TCP 추종 오차가 `track_tol_mm`(기본 20) 를 넘으면
`"tracking"`, 둘 다 아니면 `"pass"`.

## 동작 방식

1. **계획 기록** — `scene.simulate_sequences(...)` 로 기준 결과를 확보하고, `scene.open_rollout(...)` 을 틱(10 ms)마다
   진행하며 로봇 관절값·TCP, 모든 장애물 포즈를 기록. 움직인 장애물과 파지 구간(`grasp_report`)을 자동 판별.
   롤아웃이 계획에 실패하는 셀은 일괄 베이크 타임라인(`tl.sample`, `tl.object_pose`)에서 샘플링.
   원래 베이크 인자(`physics=` 포함)를 그대로 써서 botrail 물리 베이크용 셀도 같은 계획이 나옴.
2. **USD → MJCF** — `scene.export_usd(physics=bt.Physics(world=True))` 결과를 변환:
   아티큘레이션(링크 질량/관성, 관절 프레임·축·한계, 아마추어), 드라이브(stiffness/damping/maxForce → `general`
   액추에이터 `kp(q*-q)+kv(v*-q̇)`), `PhysxMimicJointAPI` → equality, URDF/USD 출처 로봇 모두. 로봇 여러 대면
   이름에 `<로봇>/` 접두사.
3. **MuJoCo 실행** — 1 ms 스텝, `implicitfast`. 관절 목표 = 계획 위치 + 속도 피드포워드. 움직이는 장애물은 mocap.
   베이스가 움직이는 로봇(차량·보행·드론)은 첫 링크를 mocap 루트로 두고 botrail 베이스 경로를 따르게 한다.
   잡는 동안 그 물체를 든 로봇의 충돌 그룹으로 옮겨 그리퍼와는 무시, 환경·다른 로봇과는 검사.
4. **역동역학** — 계획 (q, q̇, q̈) 에 대해 `mj_inverse` (제약력 제외) → 관절별 필요 토크 최대값과 한계 대비 비율.
5. **모델 보정 (경고로 기록)** — 질량 미지정 링크 0.2 kg, 비현실적으로 작은 관성·아마추어 0 에 하한,
   PhysX 전용 초고강성 드라이브(예: NVIDIA Franka 10⁸ N·m/rad)는 MuJoCo 안정 범위로, 시작 자세 중력도 못 버티는
   토크 한계(URDF `effort="1"` 같은 자리채움)는 해제, 관절 종류와 축이 안 맞는 미믹은 무시.

## 호환성 (botrail 예제)

<!-- SURVEY -->
botrail 0.13 예제 55개를 수정 없이 돌린 결과 (`tools/survey_examples.py`): **38개 변환·실행** (통과 25, 판정 차이 8, 추종 실패 5) · ⛔ 미지원 2, ⛔ 외부 정책 필요 5, — 시퀀스 없음 10.

| 예제 | 결과 | 사이클 s | TCP 오차 최대 mm | 비고 |
|---|---|---:|---:|---|
| `basics/friction_grasp_demo.py` | ✅ 통과 | 8.8 | 1.34 |  |
| `basics/gripper_pick_demo.py` | ✅ 통과 | 8.8 | 1.35 |  |
| `basics/hand_grasp_demo.py` | ✅ 통과 | 9.2 | 1.84 |  |
| `basics/physics_pick_place.py` | ✅ 통과 | 8.4 | 5.23 |  |
| `basics/sequence_demo.py` | ✅ 통과 | 12.4 | 0.89 |  |
| `basics/sfc_chart_demo.py` | ✅ 통과 | 19.3 | 1.63 |  |
| `basics/sweep_demo.py` | ✅ 통과 | 7.1 | 4.56 |  |
| `drone/drone_survey_demo.py` | ✅ 통과 | 89.0 | 4.63 | 로봇 2대. |
| `engineering/cell_deliverables_demo.py` | ✅ 통과 | 8.8 | 4.14 |  |
| `export/export_urscript.py` | ✅ 통과 | 8.8 | 4.14 |  |
| `legged/humanoid_carry_demo.py` | ✅ 통과 | 40.7 | 9.35 |  |
| `legged/legged_patrol_demo.py` | ✅ 통과 | 37.2 | 9.12 | 로봇 2대. |
| `legged/stairs_delivery_demo.py` | ✅ 통과 | 18.8 | 4.87 |  |
| `machining/ati_deburring_demo.py` | ✅ 통과 | 19.9 | 1.55 |  |
| `machining/machine_tending_demo.py` | ✅ 통과 | 56.8 | 3.31 |  |
| `machining/machining_demo.py` | ✅ 통과 | 99.2 | 1.67 |  |
| `machining/two_machine_cell_demo.py` | ✅ 통과 | 154.4 | 3.42 |  |
| `multi_robot/dual_arm_demo.py` | ✅ 통과 | 15.7 | 0.85 |  |
| `multi_robot/dual_cell_demo.py` | ✅ 통과 | 75.4 | 1.19 | 로봇 2대. |
| `painting/painting_demo.py` | ✅ 통과 | 16.5 | 0.11 |  |
| `painting/painting_hood_demo.py` | ✅ 통과 | 31.6 | 0.11 |  |
| `rl/reach_control_demo.py` | ✅ 통과 | 8.0 | 0.00 |  |
| `vehicles/amr_demo.py` | ✅ 통과 | 21.0 | 2.46 |  |
| `vehicles/lift_demo.py` | ✅ 통과 | 16.2 | 0.00 |  |
| `vehicles/semi_humanoid_demo.py` | ✅ 통과 | 25.0 | 4.40 |  |
| `assembly/cover_bolting_demo.py` | ⚠️ 판정 차이 | 56.9 | 3.52 | 체결 직후 비트가 나사 구멍에 남아 있는 구간 — 커버의 구멍이 충돌 모델에서 꽉 찬 박스 |
| `legged/building_delivery_demo.py` | ⚠️ 판정 차이 | 436.4 | 334.76 |  |
| `legged/wash_inspect_ship_demo.py` | ⚠️ 판정 차이 | 64.1 | 18.93 |  |
| `vehicles/agv_cell_demo.py` | ⚠️ 판정 차이 | 19.0 | 1.04 |  |
| `vehicles/agv_sweep_demo.py` | ⚠️ 판정 차이 | 19.0 | 1.04 |  |
| `vehicles/forklift_demo.py` | ⚠️ 판정 차이 | 105.4 | 153.77 |  |
| `vehicles/warehouse_demo.py` | ⚠️ 판정 차이 | 303.7 | 6.86 | 로봇 3대. |
| `welding/nimak_spot_welding_demo.py` | ⚠️ 판정 차이 | 4.7 | 65.96 | 평행 링크(닫힌 고리) 로봇을 트리+미믹 제약으로 근사 — 동작 중 링크가 처져 건이 강판에 닿음 |
| `assembly/shuttle_line_demo.py` | ⚠️ 추종 실패 | 37.1 | 7029.47 | 로봇 19대. |
| `palletizing/palletizing_line_demo.py` | ⚠️ 추종 실패 | 94.3 | 69.32 | 로봇 4대. |
| `welding/line_balance_sweep.py` | ⚠️ 추종 실패 | 114.3 | 97.49 | 로봇 4대. |
| `welding/weld_line_demo.py` | ⚠️ 추종 실패 | 108.7 | 97.49 | 로봇 4대. |
| `welding/weld_station_demo.py` | ⚠️ 추종 실패 | 108.7 | 273.93 | 로봇 4대. |
| `basics/physics_conveyor.py` | ⛔ 미지원 |  |  | 로봇이 없는 셀 |
| `basics/physics_drop.py` | ⛔ 미지원 |  |  | 로봇이 없는 셀 |
| `rl/depth_pick_env.py` | ⛔ 외부 정책 필요 |  |  | 외부 제어(RL 정책) 입력을 기다리는 시퀀스 — 정책 없이 단독 실행 불가 |
| `rl/policy_cell_demo.py` | ⛔ 외부 정책 필요 |  |  | 외부 제어(RL 정책) 입력을 기다리는 시퀀스 — 정책 없이 단독 실행 불가 |
| `rl/reach_env.py` | ⛔ 외부 정책 필요 |  |  | 외부 제어(RL 정책) 입력을 기다리는 시퀀스 — 정책 없이 단독 실행 불가 |
| `rl/tabletop_env.py` | ⛔ 외부 정책 필요 |  |  | 외부 제어(RL 정책) 입력을 기다리는 시퀀스 — 정책 없이 단독 실행 불가 |
| `rl/torque_env.py` | ⛔ 외부 정책 필요 |  |  | 외부 제어(RL 정책) 입력을 기다리는 시퀀스 — 정책 없이 단독 실행 불가 |
| `basics/demo.py` | — 시퀀스 없음 |  |  | 시퀀스 베이크 없음 (계획/물리/스윕/RL/내보내기 데모) |
| `basics/physics_world_demo.py` | — 시퀀스 없음 |  |  | 시퀀스 베이크 없음 (계획/물리/스윕/RL/내보내기 데모) |
| `engineering/equipment_cell_demo.py` | — 시퀀스 없음 |  |  | 시퀀스 베이크 없음 (계획/물리/스윕/RL/내보내기 데모) |
| `engineering/urplus_products.py` | — 시퀀스 없음 |  |  | 시퀀스 베이크 없음 (계획/물리/스윕/RL/내보내기 데모) |
| `export/export_animation.py` | — 시퀀스 없음 |  |  | 시퀀스 베이크 없음 (계획/물리/스윕/RL/내보내기 데모) |
| `export/isaaclab_cell.py` | — 시퀀스 없음 |  |  | 시퀀스 베이크 없음 (계획/물리/스윕/RL/내보내기 데모) |
| `export/isaaclab_tabletop.py` | — 시퀀스 없음 |  |  | 시퀀스 베이크 없음 (계획/물리/스윕/RL/내보내기 데모) |
| `export/play_record.py` | — 시퀀스 없음 |  |  | 시퀀스 베이크 없음 (계획/물리/스윕/RL/내보내기 데모) |
| `machining/spindle_mounting_demo.py` | — 시퀀스 없음 |  |  | 시퀀스 베이크 없음 (계획/물리/스윕/RL/내보내기 데모) |
| `welding/spot_gun_mounting_demo.py` | — 시퀀스 없음 |  |  | 시퀀스 베이크 없음 (계획/물리/스윕/RL/내보내기 데모) |
<!-- /SURVEY -->

## 한계

- **단방향 결합**: botrail 논리 → MuJoCo 물리. MuJoCo 상태가 botrail 센서/PLC 판단으로 되돌아가지 않음
  (botrail 에 외부 물리 상태를 넣는 API 가 없음).
- **이동형 베이스**(AGV·AMR·지게차·보행·드론)는 *베이스 경로를 botrail 계획 그대로* 따른다(mocap). 위의 팔·마스트·다리
  관절은 물리 추종하지만 차량 구동·보행 동역학(접지력, 균형)은 시뮬레이션하지 않으며, 바닥·계단 디딤판과의 접촉은 계산 제외.
- **파지**는 물리 파지가 아니라 botrail 포즈를 따르는 mocap. 마찰 파지·슬립 검증은 로드맵.
- 모델에 관성 정보가 없는 로봇(URDF `<inertial>` 없음)은 토크 결과가 참고용 — `warnings` 에 표시.
- 메쉬 충돌은 MuJoCo 볼록 껍질. 오목 형상은 botrail 의 VHACD 분해 결과가 USD 에 있으면 그대로 사용.
- 닫힌 고리 기구(평행 링크 로봇)는 트리 + 미믹 제약 근사라 큰 하중에서 처짐이 생길 수 있음.

## 로드맵

- [ ] 물리 파지 (equality weld → 마찰 파지, 슬립/낙하 검증)
- [x] 이동형 베이스 — 경로 추종(mocap) 방식
- [ ] 차량 구동 동역학(바퀴 접지) · 보행 제어기(외부 정책) 로 베이스까지 물리화
- [ ] MJX/Warp 병렬 실행 (파라미터 스윕·강화학습)
- [ ] 닫힌 고리 기구 (connect/weld 제약)
- [ ] botrail 쪽 외부 물리 훅이 생기면 폐루프 결합

## 테스트

```bash
pytest -q        # 저장소 자체 셀(tests/cell_small.py): 기구학 일치, 추종, 캡처, 다중 로봇 충돌
python tools/survey_examples.py path/to/botrail/examples out/survey -j 4   # 예제 전체 호환성 표
```

## 라이선스

이 저장소: MIT. botrail 코드나 예제를 포함하지 않으며 의존성으로만 사용합니다.
botrail 은 별도 라이선스(PolyForm Noncommercial / Small Business, 상용 라이선스)를 따르므로 사용 조건은
botrail 쪽 라이선스를 확인하세요. MuJoCo 는 Apache-2.0.
