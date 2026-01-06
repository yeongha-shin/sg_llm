import math
import numpy as np

class ColregDecision:

    def calc_dcpa_tcpa(self, ego_ship, target_ship):
        # 상대 위치 (dx, dy)를 미터 단위로 변환
        dx = (target_ship.x - ego_ship.x)
        dy = (target_ship.y - ego_ship.y)

        # 선박의 속도 벡터 계산 (속도는 이미 m/s 단위이므로 변환 불필요)
        ego_vx = ego_ship.speed * math.sin(math.radians(ego_ship.heading))
        ego_vy = ego_ship.speed * math.cos(math.radians(ego_ship.heading))
        target_vx = target_ship.speed * math.sin(math.radians(target_ship.heading))
        target_vy = target_ship.speed * math.cos(math.radians(target_ship.heading))

        # 상대 속도 벡터
        dvx = target_vx - ego_vx
        dvy = target_vy - ego_vy

        # 상대 속도 크기
        dv = dvx ** 2 + dvy ** 2

        if dv == 0:
            # 상대 속도가 0이면 충돌이 없고, DCPA는 현재 거리로 설정
            dcpa = math.sqrt(dx ** 2 + dy ** 2)
            tcpa = float('inf')
            return dcpa, tcpa

        # TCPA 계산 (시간)
        tcpa = -(dx * dvx + dy * dvy) / dv

        # TCPA에서의 두 선박 위치 계산 (픽셀 정보를 미터 단위로 변환)
        closest_x_ego = ego_ship.x + ego_vx * tcpa
        closest_y_ego = ego_ship.y + ego_vy * tcpa
        closest_x_target = target_ship.x + target_vx * tcpa
        closest_y_target = target_ship.y + target_vy * tcpa

        # DCPA 계산 (거리)
        dcpa = math.sqrt((closest_x_target - closest_x_ego) ** 2 + (closest_y_target - closest_y_ego) ** 2)

        return dcpa, tcpa

    def calc_rel_bearing(self, ego_x, ego_y, target_x, target_y, ego_heading):
        dx = target_x - ego_x
        dy = (target_y) - ego_y
        angle_to_target = math.degrees(math.atan2(dy, dx))

        # 상대적인 각도와 heading(방향)에 따른 조우 상황 판단
        relative_bearing = (90 - (angle_to_target)) - ego_heading

        if relative_bearing > 360:
            relative_bearing -= 360
        if relative_bearing < 0:
            relative_bearing += 360

        return relative_bearing

    def calc_abs_bearing(self, ego_x, ego_y, target_x, target_y):
        dx = target_x - ego_x
        dy = (target_y) - ego_y
        angle_to_target = math.degrees(math.atan2(-dy, dx))

        # 상대적인 각도와 heading(방향)에 따른 조우 상황 판단
        # abs_bearing = angle_to_target
        abs_bearing = 90 -(angle_to_target)

        if abs_bearing > 360:
            abs_bearing -= 360
        if abs_bearing < 0:
            abs_bearing += 360

        return abs_bearing

    def calc_encounter_role(self, ego_ship, target_ship):
        # 입력값을 2π 범위 내로 조정

        course_diff = target_ship.heading - ego_ship.heading
        rel_bearing = self.calc_rel_bearing(ego_ship.x, ego_ship.y,
                                            target_ship.x, target_ship.y, ego_ship.heading)

        if course_diff < 0:
            course_diff += 360

        psi_ot_deg = course_diff
        beta_ot_deg = rel_bearing

        psi_ot = np.radians(psi_ot_deg)
        beta_ot = np.radians(beta_ot_deg)

        psi_ot = psi_ot % (2 * np.pi)
        beta_ot = beta_ot % (2 * np.pi)

        # 절차 2: 안전(Safe) 조건 확인
        if target_ship.speed <= 0.1:
            return "Safe"

        if (psi_ot > np.pi / 2) and (beta_ot < 3 * np.pi / 2) and (abs(beta_ot - psi_ot) < np.pi / 2):
            return "Safe"

        # 절차 5-11: 정면, 스타보드 교차, 포트 교차 확인
        if (psi_ot >= 7 * np.pi / 8) and (psi_ot < 9 * np.pi / 8):
            return "Headon"  # okay
        elif (psi_ot >= 9 * np.pi / 8) and (psi_ot < 13 * np.pi / 8):
            return "StarboardCrossing"  # okay
        elif (psi_ot >= 3 * np.pi / 8) and (psi_ot < 7 * np.pi / 8):
            return "PortCrossing"  # okay

        # 절차 12-21: 추월, 추월 당함, 기타 상황 확인
        psi_to = (2 * np.pi - psi_ot) % (2 * np.pi)
        beta_to = (np.pi + beta_ot - psi_ot) % (2 * np.pi)

        # TODO: overtaking, overtaken 용어 정리
        # print("after")

        # if (beta_ot >= 5 * np.pi / 8) and (beta_ot < 11 * np.pi / 8) and (ego_ship.speed > target_ship.speed):
        #     return "Overtaking"
        # elif (beta_to >= 5 * np.pi / 8) and (beta_to < 11 * np.pi / 8) and (ego_ship.speed < target_ship.speed):
        #     return "Overtaken"
        #
        if (beta_ot >= 5 * np.pi / 8) and (beta_ot < 11 * np.pi / 8) and (ego_ship.speed < target_ship.speed):
            return "Overtaken"
        elif (beta_to >= 5 * np.pi / 8) and (beta_to < 11 * np.pi / 8) and (ego_ship.speed > target_ship.speed):
            return "Overtaking"
        elif beta_ot < np.pi:
            return "StarboardCrossing"
        else:
            return "PortCrossing"

        return "Safe"

#######################################################################################################################
#                                                       Utils
#######################################################################################################################
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

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

@dataclass
class ShipStateForColreg:
    x: float
    y: float
    speed: float       # m/s
    heading: float     # degrees

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

        # edge unary parameter
        self.colreg = ColregDecision()

        self.weight_colreg = 1.5

    def _to_colreg_state(self, o: ObjObs) -> ShipStateForColreg:
        speed = math.hypot(o.vx, o.vy)
        heading_deg = o.heading  # rad -> deg
        return ShipStateForColreg(x=o.x, y=o.y, speed=speed, heading=heading_deg)

    #####################################################################################################################
    #                                               Node Unary
    #####################################################################################################################

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

    #####################################################################################################################
    #                                               Edge Unary
    #####################################################################################################################

    def edge_features(self, oi: ObjObs, oj: ObjObs):
        dx = oj.x - oi.x
        dy = oj.y - oi.y
        dist = math.hypot(dx, dy) + 1e-6

        rel_bearing = math.atan2(dy, dx) - oi.heading
        rel_bearing = (rel_bearing + math.pi) % (2 * math.pi) - math.pi

        rvx = oj.vx - oi.vx
        rvy = oj.vy - oi.vy
        closing = -(dx * rvx + dy * rvy) / dist

        heading_diff = abs(
            (oj.heading - oi.heading + math.pi) % (2 * math.pi) - math.pi
        )

        return {
            "distance": dist,
            "bearing": rel_bearing,
            "closing": closing,
            "heading_diff": heading_diff
        }

    def colreg_edge_unary(self, ego: ObjObs, target: ObjObs, r: str) -> float:
        """
        COLREG-based soft prior for ship-ship relations
        (ObjObs -> ColregDecision input adapter)
        """
        ego_s = self._to_colreg_state(ego)
        tgt_s = self._to_colreg_state(target)

        role = self.colreg.calc_encounter_role(ego_s, tgt_s)

        E = 0.0
        if role == "StarboardCrossing":
            if r == "give_way":
                E -= 2.0
            elif r == "stand_on":
                E += 2.0

        elif role == "PortCrossing":
            if r == "stand_on":
                E -= 1.5
            elif r == "give_way":
                E += 1.5

        elif role == "Headon":
            if r in {"give_way"}:
                E -= 2.0

        elif role == "Overtaking":
            if r == "overtaking":
                E -= 2.0
            else:
                E += 1.0

        elif role == "Overtaken":
            if r == "overtaken":
                E -= 2.0
            else:
                E += 1.0

        return self.weight_colreg * E

    def edge_unary_ship_ship(
            self,
            oi: ObjObs,
            oj: ObjObs,
            ci: str,
            cj: str,
            r: str
    ) -> float:
        # hard compatibility
        if r not in allowed_relations(ci, cj):
            return 50.0

        feat = self.edge_features(oi, oj)

        E = 0.0
        E += self.colreg_edge_unary(oi, oj, r)

        return E

    def edge_unary_ship_goal(self, feat, r):
        E = 0.0
        closing = feat["closing"]

        if r == "approaching":
            if closing > 0:
                E -= 2.0
            else:
                E += 2.0

        elif r == "passing":
            if closing < 0:
                E -= 1.5
            else:
                E += 1.5

        return E

    def edge_unary_ship_buoy(self, feat, r):
        E = 0.0
        bearing = feat["bearing"]

        # if r == "avoiding_left":
        #     if bearing < 0:
        #         E -= 1.0
        #     else:
        #         E += 1.0

        # if r == "avoiding_right":
        #     if bearing > 0:
        #         E -= 1.0
        #     else:
        #         E += 1.0
        #
        # elif r == "well_clear":
        #     if abs(bearing) > math.pi / 2:
        #         E -= 0.5

        return E

    def edge_unary_ship_obstacle(self, feat, r):
        E = 0.0
        dist = feat["distance"]

        # if r == "colliding":
        #     if dist < 20.0:
        #         E -= 2.0
        #     else:
        #         E += 2.0
        #
        # elif r == "mission_operating":
        #     if dist > 20.0:
        #         E -= 1.0

        return E

    def edge_unary_energy(
            self,
            oi: ObjObs,
            oj: ObjObs,
            ci: str,
            cj: str,
            r: str
    ) -> float:
        # hard compatibility constraint
        if r not in allowed_relations(ci, cj):
            return 50.0  # +∞ 근사

        feat = self.edge_features(oi, oj)
        E = 0.0

        # ship -> ship
        if ci == "ship" and cj == "ship":
            E += self.colreg_edge_unary(oi, oj, r)

        # ship -> tss
        elif ci == "ship" and cj == "tss_entrance":
            E += self.edge_unary_ship_goal(feat, r)

        # ship -> buoy # TODO
        elif ci == "ship" and cj == "buoy":
            E += self.edge_unary_ship_buoy(feat, r)

        # ship -> obstacle # TODO
        elif ci == "ship" and cj in {"land", "bridge", "crane", "fishing_gear", "tire"}:
            E += self.edge_unary_ship_obstacle(feat, r)

        return E

    def _softmax_energy(self, energies: Dict[str, float]) -> Dict[str, float]:
        """
        Convert energy dict {label: E} to probability via softmax(-E).
        """
        # numerical stability: subtract min energy (equiv to shifting energies)
        Emin = min(energies.values())
        exps = {k: math.exp(-(E - Emin)) for k, E in energies.items()}
        Z = sum(exps.values()) + 1e-12
        return {k: v / Z for k, v in exps.items()}

    def edge_belief_given_classes(
            self,
            oi: ObjObs,
            oj: ObjObs,
            ci: str,
            cj: str,
            rel_set: Tuple[str, ...] = REL_SET
    ) -> Dict[str, float]:
        """
        q(R_{i->j} | Ci=ci, Cj=cj, X) ∝ exp(-E(r))
        with hard compatibility via allowed_relations.
        """
        allowed = set(allowed_relations(ci, cj))
        energies = {}

        for r in rel_set:
            if r not in allowed:
                # forbidden relations effectively get zero prob
                continue
            energies[r] = self.edge_unary_energy(oi, oj, ci, cj, r)

        # if nothing allowed (shouldn't happen), fall back to none
        if not energies:
            energies = {"none": 0.0}

        return self._softmax_energy(energies)

    # -----------------------------
    # (B) Mean-field edge belief marginalizing class uncertainty
    # -----------------------------
    def edge_belief_mf(
            self,
            oi: ObjObs,
            oj: ObjObs,
            node_beliefs: Dict[int, Dict[str, float]],
            rel_set: Tuple[str, ...] = REL_SET,
            class_set: Tuple[str, ...] = CLASS_SET,
            min_pair_prob: float = 1e-4
    ) -> Dict[str, float]:
        """
        Mean-field approximation:
        q_ij(r) ∝ exp( - E_{Ci~q_i, Cj~q_j}[ edge_unary_energy(r|Ci,Cj) ] )

        We approximate expected energy by summing over class pairs.
        """
        qi = node_beliefs[oi.obj_id]
        qj = node_beliefs[oj.obj_id]

        # expected energy for each relation
        expE = {r: 0.0 for r in rel_set}

        for ci in class_set:
            pi = qi.get(ci, 0.0)
            if pi <= 0.0:
                continue
            for cj in class_set:
                pj = qj.get(cj, 0.0)
                pij = pi * pj
                if pij < min_pair_prob:
                    continue

                allowed = set(allowed_relations(ci, cj))
                for r in rel_set:
                    if r not in allowed:
                        continue
                    expE[r] += pij * self.edge_unary_energy(oi, oj, ci, cj, r)

                # (선택) forbidden 관계는 확률 0이 되게 하고 싶으면 expE에 따로 반영하지 않음

        # 관계 중에서 "어떤 클래스 조합에서도 허용되지 않은" 관계는 제거
        energies = {}
        for r, E in expE.items():
            # expE가 0.0인 게 실제로 "가능해서 0"인지 "전혀 누적이 안돼서 0"인지 구분이 필요
            # 안전하게: 최소한 한 번이라도 허용된 관계만 남기자
            # -> 간단히: allowed_relations가 존재할 때만 남기는 방식으로 재계산
            energies[r] = E

        # 만약 모든 에너지가 0으로만 남는 특수 케이스(누적이 거의 없을 때) 대비:
        # 여기서는 softmax가 알아서 균등에 가깝게 됨. 필요하면 "none"에 작은 bias 줄 수 있음.

        return self._softmax_energy(energies)

    # -----------------------------
    # (C) Frame-level edge beliefs for all directed pairs
    # -----------------------------
    def infer_edge_beliefs_mf(
            self,
            obs: List[ObjObs],
            node_beliefs: Dict[int, Dict[str, float]]
    ) -> Dict[Tuple[int, int], Dict[str, float]]:
        """
        Compute q(R_{i->j}) for all directed pairs i != j.
        """
        edges = {}
        for oi in obs:
            for oj in obs:
                if oi.obj_id == oj.obj_id:
                    continue
                edges[(oi.obj_id, oj.obj_id)] = self.edge_belief_mf(oi, oj, node_beliefs)
        return edges


def main():
    crf = CRFSceneGraph()

    # 예시: ship, buoy, tss
    ship_one = ObjObs(
        obj_id=1, t=0, x=0.0, y=0.0, vx=5.0, vy=0.0, heading=90.0, size=10.0,
        det_conf={"ship": 0.7, "buoy": 0.2, "unknown": 0.1}
    )

    ship_two = ObjObs(
        obj_id=2, t=0, x=20.0, y=0.0, vx=-5.0, vy=0.0, heading=270.0, size=10.0,
        det_conf={"ship": 0.8, "buoy": 0.1, "unknown": 0.1}
    )

    # buoy = ObjObs(
    #     obj_id=2, t=0, x=20.0, y=5.0, vx=0.05, vy=0.02, heading=0.0, size=2.0,
    #     det_conf={"buoy": 0.8, "ship": 0.1, "unknown": 0.1}
    # )

    # tss = ObjObs(
    #     obj_id=3, t=0, x=0.0, y=80.0, vx=0.0, vy=0.0, heading=0.0, size=3.0,
    #     det_conf={"tss_entrance": 0.9, "unknown": 0.1}
    # )
    #
    # obs = [ship, buoy, tss]

    obs = [ship_one, ship_two]

    # 1) node beliefs
    node_beliefs = {o.obj_id: crf.node_belief(o, list(CLASS_SET)) for o in obs}

    print("\n" + "=" * 80)
    print("[Node beliefs]")
    for oid, b in node_beliefs.items():
        top = sorted(b.items(), key=lambda x: -x[1])[:3]
        print(f"Obj {oid} top-3: {top}")

    # 2) directed edge beliefs (mean-field)
    edge_beliefs = crf.infer_edge_beliefs_mf(obs, node_beliefs)

    print("\n" + "=" * 80)
    print("[Directed edge beliefs: top-3 relations per edge]")
    for (i, j), pR in edge_beliefs.items():
        top = sorted(pR.items(), key=lambda x: -x[1])[:3]
        print(f"R_{i}->{j}: {top}")

    print("=" * 80)

    viz = SceneGraphVisualizer()
    viz.plot(obs, node_beliefs, edge_beliefs)


import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


class SceneGraphVisualizer:
    def __init__(self):
        self.node_colors = {
            "ship": "tab:red",
            "buoy": "tab:blue",
            "tss_entrance": "tab:green",
            "land": "saddlebrown",
            "bridge": "purple",
            "crane": "darkorange",
            "fishing_gear": "olive",
            "tire": "gray",
            "unknown": "black"
        }

        self.edge_colors = {
            "give_way": "red",
            "stand_on": "green",
            "overtaking": "orange",
            "overtaken": "orange",
            "approaching": "blue",
            "passing": "cyan",
            "avoiding_left": "purple",
            "avoiding_right": "purple",
            "well_clear": "gray",
            "colliding": "black",
            "mission_operating": "brown",
            "on": "pink",
            "none": "lightgray"
        }

    def plot(
        self,
        obs: List[ObjObs],
        node_beliefs: Dict[int, Dict[str, float]],
        edge_beliefs: Dict[Tuple[int, int], Dict[str, float]],
        top_k_edge: int = 1
    ):
        fig, ax = plt.subplots(figsize=(8, 8))

        # -------------------------
        # Plot nodes
        # -------------------------
        for o in obs:
            beliefs = node_beliefs[o.obj_id]
            cls, p = max(beliefs.items(), key=lambda x: x[1])

            ax.scatter(
                o.x, o.y,
                s=300,
                color=self.node_colors.get(cls, "black"),
                alpha=0.8,
                zorder=3
            )

            ax.text(
                o.x + 0.5, o.y + 0.5,
                f"{o.obj_id}\n{cls}\n{p:.2f}",
                fontsize=9,
                bbox=dict(boxstyle="round", fc="white", alpha=0.8)
            )

            # velocity arrow
            ax.arrow(
                o.x, o.y, o.vx, o.vy,
                head_width=0.8,
                length_includes_head=True,
                alpha=0.5
            )

        # -------------------------
        # Plot directed edges
        # -------------------------
        for (i, j), pR in edge_beliefs.items():
            oi = next(o for o in obs if o.obj_id == i)
            oj = next(o for o in obs if o.obj_id == j)

            top_rel = sorted(pR.items(), key=lambda x: -x[1])[:top_k_edge]

            for k, (r, p) in enumerate(top_rel):
                if p < 0.05:
                    continue  # 너무 약한 관계는 생략

                dx = oj.x - oi.x
                dy = oj.y - oi.y

                arrow = FancyArrowPatch(
                    (oi.x, oi.y),
                    (oj.x, oj.y),
                    arrowstyle="->",
                    linewidth=1 + 5 * p,
                    color=self.edge_colors.get(r, "black"),
                    alpha=0.7,
                    zorder=2
                )
                ax.add_patch(arrow)

                ax.text(
                    oi.x + 0.5 * dx,
                    oi.y + 0.5 * dy + 1.0 * k,
                    f"{r}\n{p:.2f}",
                    fontsize=8,
                    color=self.edge_colors.get(r, "black")
                )

        xlim = (-50, 50)
        ylim = (-50, 50)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        ax.set_aspect("equal")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.grid(True)
        ax.set_title("CRF Scene Graph (Node & Edge Beliefs)")
        plt.show()


if __name__ == "__main__":
    main()


