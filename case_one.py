from dataclasses import dataclass
from typing import Optional, Dict, List

# 논문에서 정의한 클래스 집합
CLASS_SET = (
    "ship", "buoy", "tss_entrance", "land",
    "bridge", "crane", "fishing_gear", "tire", "unknown"
)

@dataclass
class ObjObs:
    obj_id: int
    t: int
    x: float
    y: float
    vx: float
    vy: float
    heading: float   # radians
    size: float

    # 관측값: detector가 주는 클래스별 confidence (확률/점수)
    # 예: {"ship":0.8, "buoy":0.1, "unknown":0.1}
    det_conf: Optional[Dict[str, float]] = None

# 전체 관계 집합 R (논문에서의 𝓡)
REL_SET = (
    # ship-ship (운동 의도 기반)
    "give_way", "stand_on", "overtaking", "overtaken",

    # ship-goal (예: tss_entrance)
    "approaching", "passing",

    # ship-obstacle
    "colliding", "mission_operating",

    # ship-buoy
    "avoiding_left", "avoiding_right", "well_clear",

    # ship-part (부속품)
    "on",

    # fallback
    "none"
)

def allowed_relations(ci: str, cj: str) -> List[str]:
    """
    클래스 조합 (Ci, Cj)에 따라 가능한 관계 라벨 subset을 반환.
    (논문: 객체 클래스 조합에 의해 사전에 제한)
    """
    # ship-ship
    if (ci == "ship" and cj
            == "ship"):
        return ["give_way", "stand_on", "overtaking", "overtaken", "none"]

    # ship-goal (tss_entrance를 목적지로 취급)
    if ci == "ship" and cj == "tss_entrance":
        return ["approaching", "passing", "none"]
    if ci == "tss_entrance" and cj == "ship":
        # 방향성 있을 경우 반대로도 정의하거나, (i,j)만 쓰면 생략 가능
        return ["approaching", "passing", "none"]

    # ship-buoy
    if (ci == "ship" and cj == "buoy") or (ci == "buoy" and cj == "ship"):
        return ["avoiding_left", "avoiding_right", "well_clear", "none"]

    # ship-obstacle (여기서는 land/bridge/crane/fishing_gear/tire를 장애물 범주로)
    obstacle = {"land", "bridge", "crane", "fishing_gear", "tire"}
    if (ci == "ship" and cj in obstacle) or (cj == "ship" and ci in obstacle):
        return ["colliding", "mission_operating", "none"]

    # ship-part: 예시는 fishing_gear/tire를 part로 볼 수도 있는데,
    # 논문에서 "선박 부속품"을 별도 클래스라고 했다면 클래스 분리를 추천.
    # 일단 예시로 fishing_gear/tire를 part 취급하고 ship과의 관계를 on으로 제한:
    part = {"fishing_gear", "tire"}
    if (ci == "ship" and cj in part) or (cj == "ship" and ci in part):
        return ["on", "none"]

    return ["none"]

