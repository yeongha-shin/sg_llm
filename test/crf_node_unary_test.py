
#######################################################################################################################
#                                                       Utils
#######################################################################################################################
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


########################################################################################################################
#                                                   CRF Model
########################################################################################################################
import math

class CRFSceneGraph:
    def __init__(self):
        # node unary parameter
        # --- physical thresholds ---
        self.moving_obj_speed = 1.0

        # --- motion unary parameters ---
        self.penalty_stopping = 2.0
        self.scale_static_speed = 0.5

        # --- unary weights ---
        self.weight_det = 1.0
        self.weight_motion = 0.5

        # --- numerical stability ---
        self.detector_eps = 1e-6



    def unary_from_detector(self, o: ObjObs, c: str) -> float:
        """
        Detector confidence 기반 unary energy
        """
        if o.det_conf is None:
            return 0.0  # detector 없으면 neutral

        p = o.det_conf.get(c, 0.0)
        return -math.log(p + self.detector_eps)

    def unary_from_motion(self, o: ObjObs, c: str) -> float:
        speed = math.hypot(o.vx, o.vy)

        if c == "ship":
            return 0.0 if speed > self.moving_obj_speed else self.penalty_stopping

        if c in {"buoy", "land", "bridge", "crane"}:
            return self.scale_static_speed * speed

        return 0.0

    def node_unary_energy(
            self,
            o: ObjObs,
            c: str
    ) -> float:
        E = 0.0
        E += self.weight_det * self.unary_from_detector(o, c)
        E += self.weight_motion * self.unary_from_motion(o, c)
        return E

    def node_belief(self, o: ObjObs, class_set: List[str]) -> Dict[str, float]:
        energies = {
            c: self.node_unary_energy(o, c)
            for c in class_set
        }
        # softmax over -energy
        mx = min(energies.values())
        probs = {c: math.exp(-(E - mx)) for c, E in energies.items()}
        Z = sum(probs.values()) + 1e-12
        return {c: p / Z for c, p in probs.items()}


def main():
    crf = CRFSceneGraph()

    # -----------------------------
    # 테스트 객체 정의
    # -----------------------------

    # 1) 이동 중인 선박 (ship)
    ship_obs = ObjObs(
        obj_id=1,
        t=0,
        x=0.0,
        y=0.0,
        vx=5.0,
        vy=0.0,
        heading=0.0,
        size=10.0,
        det_conf={
            "ship": 0.50,
            "buoy": 0.50,
            "unknown": 0.10
        }
    )

    # 2) 거의 정지한 부표 (buoy)
    buoy_obs = ObjObs(
        obj_id=2,
        t=0,
        x=20.0,
        y=5.0,
        vx=0.05,
        vy=0.02,
        heading=0.0,
        size=2.0,
        det_conf={
            "buoy": 0.80,
            "ship": 0.10,
            "unknown": 0.10
        }
    )

    # 3) detector가 애매한 정적 장애물
    static_obs = ObjObs(
        obj_id=3,
        t=0,
        x=-10.0,
        y=15.0,
        vx=0.0,
        vy=0.0,
        heading=0.0,
        size=30.0,
        det_conf={
            "land": 0.40,
            "bridge": 0.30,
            "unknown": 0.30
        }
    )

    obs_list = [ship_obs, buoy_obs, static_obs]

    # -----------------------------
    # Node belief 출력
    # -----------------------------
    for o in obs_list:
        print("=" * 80)
        print(f"Object ID {o.obj_id}")
        print(f"  position = ({o.x:.1f}, {o.y:.1f})")
        print(f"  velocity = ({o.vx:.2f}, {o.vy:.2f}) | speed = {math.hypot(o.vx, o.vy):.2f}")

        beliefs = crf.node_belief(o, CLASS_SET)

        # 에너지 값도 같이 출력 (논문 디버깅용)
        energies = {
            c: crf.node_unary_energy(o, c)
            for c in CLASS_SET
        }

        print("\n  [Unary energies]")
        for c, E in sorted(energies.items(), key=lambda x: x[1]):
            print(f"    {c:15s}: {E:.3f}")

        print("\n  [Node belief]")
        for c, p in sorted(beliefs.items(), key=lambda x: -x[1]):
            print(f"    {c:15s}: {p:.3f}")

        print(f"  -> MAP class: {max(beliefs, key=beliefs.get)}")

    print("=" * 80)


if __name__ == "__main__":
    main()
