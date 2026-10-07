"""어댑터 테스트 (저장소 자체 셀 tests/cell_small.py 사용, botrail 예제 불필요)."""

import sys
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("botrail")
sys.path.insert(0, str(Path(__file__).parent))

from botrail_mujoco import MujocoAdapter  # noqa: E402
from botrail_mujoco.capture import capture_from_script  # noqa: E402
from cell_small import build_cell  # noqa: E402


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    ad = MujocoAdapter(build_cell(), ["cycle"], tmp_path_factory.mktemp("mj"))
    return ad.prepare()


def test_plan_records_moving_conveyor_part(prepared):
    p = prepared.plan
    assert "crate" in p.moving                      # 컨베이어가 옮긴 상자 → mocap
    assert p.t[-1] > 1.0 and p.robots[0].q.shape[1] == 4


def test_forward_kinematics_matches_botrail(prepared):
    """MJCF 기구학이 botrail 기구학과 같다: 여러 시각의 계획 관절값에서 링크 위치 비교."""
    import botrail as bt  # noqa: F401
    model = mujoco.MjModel.from_xml_path(str(prepared.conv.xml_path))
    data = mujoco.MjData(model)
    sc = prepared.scene
    names = ["j1", "j2", "j3", "j4"]
    adr = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names]
    for k in np.linspace(0, len(prepared.plan.t) - 1, 6).astype(int):
        q = prepared.plan.robots[0].q[k]
        data.qpos[adr] = q
        mujoco.mj_forward(model, data)
        sc.set_joint_positions(list(q))
        for link in ("l2", "l4"):
            want = np.array(sc.link_pose(link)[0])
            got = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link)]
            assert np.linalg.norm(got - want) < 1e-4, (k, link, got, want)


def test_run_tracks_plan_without_collision(prepared):
    rep = prepared.run(web=True)
    assert rep["verdict"] == "pass"
    assert rep["max_tcp_err_mm"] < 5.0
    assert set(rep["required_torque"]) == {"j1", "j2", "j3", "j4"}
    assert all(v["ratio"] < 1.0 for v in rep["required_torque"].values())
    assert (prepared.out / "web.json").exists() and (prepared.out / "mjcf" / "model.xml").exists()


def test_capture_from_unmodified_script():
    cap = capture_from_script(Path(__file__).parent / "cell_small.py")
    assert cap.sequences == ["cycle"]
    assert cap.scene.robot.joint_names == ["j1", "j2", "j3", "j4"]


def test_two_robots_collide_with_each_other(tmp_path):
    """다중 로봇: 이름 접두사, 로봇끼리 충돌 검사."""
    from cell_small import build_two_robot_cell
    ad = MujocoAdapter(build_two_robot_cell(), ["a", "b"], tmp_path).prepare()
    rep = ad.run(web=False)
    assert set(rep["robots"]) == {"left", "right"}
    assert any(k.startswith("collision(robots)") for k in rep["collisions"]), rep["collisions"].keys()


def test_registered_allowances_are_not_collisions(tmp_path):
    """셀이 allow_inter_robot_collision 으로 허용한 쌍은 collision 이 아니라 allowed."""
    from botrail_mujoco.allowances import record_allowances
    from cell_small import build_two_robot_cell
    with record_allowances():
        scene = build_two_robot_cell()
    rep = MujocoAdapter(scene, ["a", "b"], tmp_path).prepare().run(web=False)
    assert not rep["collisions"], rep["collisions"].keys()  # 허용된 쌍은 충돌로 세지 않는다
    assert rep["allowed_contacts"]
    # 허용돼도 MuJoCo 에서는 실제로 밀어내므로 추종은 깨질 수 있다 (verdict == "tracking")
    assert rep["verdict"] in ("pass", "tracking")
