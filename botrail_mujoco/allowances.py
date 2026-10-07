"""셀이 botrail 에 등록한 '허용 접촉'을 기록한다 (botrail 에는 조회 API 가 없어 등록 호출을 가로챈다).

- allow_link_obstacle_contact(link, obstacle, robot)   공정 접촉 (커터↔소재, 비트↔나사)
- allow_object_obstacle_contact(object, obstacle, ...)  든 물체를 끼워 넣는 접촉 (커버↔하우징)
- allow_inter_robot_collision(robot_a, link_a, robot_b, link_b)
- set_obstacle_walkable(name)                            걸을 수 있는 면 (바닥·계단 디딤판) — 이동 로봇의 지지 접촉
"""
from __future__ import annotations

import contextlib

_REG: dict = {}  # id(scene) -> [dict]
_NAMES = ("allow_link_obstacle_contact", "allow_object_obstacle_contact", "allow_inter_robot_collision",
          "set_obstacle_walkable")


def _arg(names, a, kw):
    out = dict(zip(names, a))
    out.update({k: v for k, v in kw.items() if k in names})
    return out


@contextlib.contextmanager
def record_allowances():
    import botrail as bt
    cls = bt.Scene
    orig = {n: getattr(cls, n) for n in _NAMES}
    params = {
        "allow_link_obstacle_contact": ("link", "obstacle", "robot"),
        "allow_object_obstacle_contact": ("object", "obstacle", "window"),
        "allow_inter_robot_collision": ("robot_a", "link_a", "robot_b", "link_b"),
        "set_obstacle_walkable": ("name", "walkable"),
    }

    def wrap(n):
        def f(self, *a, **kw):
            _REG.setdefault(id(self), []).append(dict(kind=n, **_arg(params[n], a, kw)))
            return orig[n](self, *a, **kw)
        return f

    try:
        for n in _NAMES:
            setattr(cls, n, wrap(n))
        yield
    finally:
        for n, f in orig.items():
            setattr(cls, n, f)


def allowances_of(scene) -> list:
    return [{k: v for k, v in e.items() if k != "window"} for e in _REG.get(id(scene), [])]
