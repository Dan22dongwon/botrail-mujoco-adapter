"""테스트용 소형 셀: 컨베이어가 상자를 보내고, 빔이 감지하면 멈추고, 팔이 다가갔다 돌아온다."""
from pathlib import Path

import botrail as bt

HERE = Path(__file__).resolve().parent


def build_cell(velocity: float = 0.25, lane_y: float = 0.6) -> bt.Scene:
    scene = bt.Scene(bt.Robot.from_urdf(str(HERE / "assets" / "arm4.urdf")))
    scene.add_box("crate", size=(0.04, 0.04, 0.04), position=(-0.5, lane_y, 0.3))
    scene.add_box("table", size=(0.6, 0.4, 0.02), position=(0.55, 0.0, 0.2))
    scene.add_conveyor("belt", zone_position=(-0.2, lane_y, 0.3), zone_size=(1.2, 0.3, 0.3),
                       velocity=(velocity, 0.0, 0.0), running=False)
    scene.add_beam_sensor("eye", frm=(0.0, lane_y - 0.2, 0.3), to=(0.0, lane_y + 0.2, 0.3))
    scene.add_segment("approach", goal=[0.9, 0.5, 0.6, 0.4])
    scene.add_segment("home", goal=[0.0, 0.0, 0.0, 0.0])
    sq = scene.sequence("cycle")
    sq.step("feed", actions=[bt.seq.start("belt")], transition=bt.seq.signal("eye"))
    sq.step("stop", actions=[bt.seq.stop("belt")])
    sq.step("approach", actions=[bt.seq.motion("approach")])
    sq.step("work", transition=bt.seq.elapsed(0.5))
    sq.step("home", actions=[bt.seq.motion("home")])
    return scene


def build_two_robot_cell() -> bt.Scene:
    """두 팔이 서로를 향해 뻗어 부딪히는 셀. botrail 에는 로봇끼리 접촉을 허용해 두어 베이크는 통과하고,
    MuJoCo 쪽에서 그 충돌이 잡히는지 본다."""
    urdf = str(HERE / "assets" / "arm4.urdf")
    scene = bt.Scene(bt.Robot.from_urdf(urdf), name="left")
    scene.add_robot(bt.Robot.from_urdf(urdf), "right", base_position=(0.9, 0.0, 0.0),
                    base_quaternion=(0.0, 0.0, 1.0, 0.0))
    links = ["l1", "l2", "l3", "l4"]
    for la in links:
        for lb in links:
            scene.allow_inter_robot_collision("left", la, "right", lb)
    scene.add_segment("reach_l", goal=[0.0, 1.2, 0.3, 0.0], robot="left")
    scene.add_segment("reach_r", goal=[0.0, 1.2, 0.3, 0.0], robot="right")
    scene.sequence("a").step("go", actions=[bt.seq.motion("reach_l")])
    scene.sequence("b").step("go", actions=[bt.seq.motion("reach_r")])
    return scene


if __name__ == "__main__":
    tl = build_cell().simulate_sequence("cycle")
    print(f"cycle {tl.duration:.2f}s")
