from dataclasses import dataclass
from typing import Optional



#############################################################################################
#                                               Utils
#############################################################################################
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
    det_label: Optional[str] = None
    det_score: Optional[float] = None


@dataclass
class VHFMessage:
    sender_id: int
    text: str

#############################################################################################
#                                           LLM module
#############################################################################################
class VHFIntentParser:
    @staticmethod
    def parse_goal_intent(text: str) -> float:
        text = text.lower()
        keywords = [
            "entering tss",
            "intend to enter tss",
            "request cooperation",
            "proceeding to traffic separation scheme",
        ]
        return 1.0 if any(k in text for k in keywords) else 0.0

#############################################################################################
#                                           CRF Module
#############################################################################################

import math
from typing import Dict, List, Optional
class DynamicCRF:
    def __init__(self):
        self.relations = [
            "head_on", "crossing", "overtaking",
            "A_stand_on_B", "A_give_way_B",
            "B_stand_on_A", "B_give_way_A",
            "approaching", "passing", "near", "none"
        ]

        self.near_dist = 30.0
        self.vhf_parser = VHFIntentParser()
        self.prev_edge_beliefs = {}
        self.temporal_weight = 2.0

        self.intent_memory = {}  # sender_id -> strength
        self.intent_decay = 0.9

        self.coop = {}  # ship_id -> p(cooperative)
        self.coop_decay = 0.98

    # ---------- utils ----------
    def _wrap(self, a):
        return (a + math.pi) % (2 * math.pi) - math.pi

    def _softmax(self, scores: Dict[str, float]):
        mx = max(scores.values())
        exps = {k: math.exp(v - mx) for k, v in scores.items()}
        s = sum(exps.values()) + 1e-12
        return {k: v / s for k, v in exps.items()}

    # ---------- node ----------
    def node_belief(self, o: ObjObs):
        if o.det_label == "tss_entrance":
            return {
                "ship": 0.0,
                "buoy": 0.0,
                "bridge_mark": 0.0,
                "tss_entrance": 1.0,
                "unknown": 0.0,
            }

        speed = math.hypot(o.vx, o.vy)
        scores = {
            "ship": 2.0 if speed > 1.0 else -2.0,
            "buoy": 2.0 if speed < 0.5 else -2.0,
            "bridge_mark": -2.0,
            "tss_entrance": -3.0,
            "unknown": -1.0,
        }

        if o.det_label in scores:
            scores[o.det_label] += 3.0

        return self._softmax(scores)

    # ---------- geometry ----------
    def rel(self, oi: ObjObs, oj: ObjObs):
        dx, dy = oj.x - oi.x, oj.y - oi.y
        d = math.hypot(dx, dy) + 1e-6
        bearing = self._wrap(math.atan2(dy, dx) - oi.heading)
        rvx, rvy = oj.vx - oi.vx, oj.vy - oi.vy
        closing = -(dx * rvx + dy * rvy) / d
        return d, bearing, closing

    def apply_relation_validity_mask(self, scores, oi, oj, pCi, pCj):
        """
        Soft mask impossible relations based on object types.
        This does NOT change relation space, only energies.
        """

        VERY_NEG = -100.0  # soft -∞

        oi_is_ship = pCi.get("ship", 0.0) > 0.6
        oj_is_ship = pCj.get("ship", 0.0) > 0.6

        oi_is_tss = oi.det_label == "tss_entrance"
        oj_is_tss = oj.det_label == "tss_entrance"

        # --------------------------------------------------
        # Ship-only relations
        # --------------------------------------------------
        ship_only = [
            "head_on", "crossing", "overtaking",
            "A_stand_on_B", "A_give_way_B",
            "B_stand_on_A", "B_give_way_A",
        ]

        if not (oi_is_ship and oj_is_ship):
            for r in ship_only:
                scores[r] += VERY_NEG

        # --------------------------------------------------
        # Approaching: only ship → TSS
        # --------------------------------------------------
        if not (oi_is_ship and oj_is_tss):
            scores["approaching"] += VERY_NEG

        if not (oi_is_ship and oj_is_tss):
            scores["passing"] += VERY_NEG

        return scores

    # ---------- edge unary ----------
    def edge_unary(self, oi, oj, pCi, pCj, goal_intent):
        d, bearing, closing = self.rel(oi, oj)
        scores = {r: -4.0 for r in self.relations}
        scores["none"] = -1.5

        oi_is_ship = pCi.get("ship", 0.0) > 0.6
        oj_is_ship = pCj.get("ship", 0.0) > 0.6
        oj_is_tss = (oj.det_label == "tss_entrance")

        # near는 타입 무관하게 써도 됨
        if d < self.near_dist:
            scores["near"] += 2.0

        # ✅ ship–ship일 때만 head_on / crossing
        if oi_is_ship and oj_is_ship:
            if abs(bearing) < math.radians(15) and closing > 0:
                scores["head_on"] += 3.0
            if abs(abs(bearing) - math.pi / 2) < math.radians(30) and closing > 0:
                scores["crossing"] += 3.0

        # ✅ ship → tss일 때만 approaching/passing
        if oi_is_ship and oj_is_tss:
            scores["approaching"] = -2.0  # instead of -4
            scores["passing"] = -2.0

            geom_strength = max(0.0, 1.0 - d / 120.0)
            speed_strength = min(abs(closing) / 3.0, 1.0)
            vhf_strength = self.intent_memory.get(oi.obj_id, 0.0)

            w_geom = 2.0
            w_vhf = 0.1

            eps = 0.1
            if closing > eps:
                scores["approaching"] += (
                        w_geom * (geom_strength + speed_strength)
                        + w_vhf * vhf_strength * (geom_strength + speed_strength)
                )
            else:
                # closing <= eps : 유지 or 멀어짐 → passing
                scores["passing"] += (
                        w_geom * (geom_strength + speed_strength)
                )

            scores["none"] -= 0.5 * (geom_strength + speed_strength + vhf_strength)

        # 마스크는 “안전벨트”로만 남기고, 핵심은 위에서 게이팅하는 것
        scores = self.apply_relation_validity_mask(scores, oi, oj, pCi, pCj)

        return scores

    def apply_temporal_potential(self, key, scores):
        if key not in self.prev_edge_beliefs:
            return scores

        prev = self.prev_edge_beliefs[key]
        for r in scores:
            scores[r] += self.temporal_weight * prev.get(r, 0.0)
        return scores

    def apply_node_edge_coupling(self, node_beliefs, edge_beliefs):
        for (i, j), pR in edge_beliefs.items():
            if pR.get("approaching", 0.0) > 0.5:
                # i는 ship 쪽으로 강화
                node_beliefs[i]["ship"] += 2.0

                # j는 tss / fixed 쪽으로 강화
                if "tss_entrance" in node_beliefs[j]:
                    node_beliefs[j]["tss_entrance"] += 2.0
                    node_beliefs[j]["ship"] -= 2.0  # ❗ ship 억제 (중요)

            ship_rel_strength = (
                    pR.get("crossing", 0.0)
                    + pR.get("head_on", 0.0)
                    + pR.get("A_stand_on_B", 0.0)
                    + pR.get("A_give_way_B", 0.0)
                    + pR.get("B_stand_on_A", 0.0)
                    + pR.get("B_give_way_A", 0.0)
            )
            if ship_rel_strength > 0.6:
                node_beliefs[i]["ship"] += 1.0
                node_beliefs[j]["ship"] += 1.0

        for i in node_beliefs:
            node_beliefs[i] = self._softmax(node_beliefs[i])
        return node_beliefs

    def update_coop_from_action(self, ship_id, delta_heading, p_app):
        # p_app: 1번->TSS approaching 확률
        # delta_heading: 이번 프레임 heading 변화량(abs)
        evidence = min(abs(delta_heading) / math.radians(5), 1.0)  # 0~1
        # approaching 상황일수록 evidence를 더 신뢰
        w = 0.5 * p_app

        # Bayesian-ish update (간단한 EMA)
        prior = self.coop.get(ship_id, 0.5)
        post = (1 - w) * prior + w * evidence
        self.coop[ship_id] = post

    def apply_relation_mutex(self, scores):
        """
        Penalize mutually exclusive relations (soft constraint)
        """
        # mutex strength (hyperparameter)
        w_strong = 3.0
        w_medium = 2.0

        # head_on ↔ overtaking
        scores["overtaking"] -= w_strong * max(0.0, scores["head_on"])
        scores["head_on"] -= w_strong * max(0.0, scores["overtaking"])

        # approaching ↔ overtaking
        scores["overtaking"] -= w_medium * max(0.0, scores["approaching"])
        scores["approaching"] -= w_medium * max(0.0, scores["overtaking"])

        # scores["approaching"] -= w_strong * max(0.0, scores["passing"])
        # scores["passing"] -= w_strong * max(0.0, scores["approaching"])


        return scores

    # ---------- inference ----------
    def infer_frame(
            self,
            obs: List[ObjObs],
            vhf_messages: Optional[List[VHFMessage]] = None,
            delta_heading: Optional[Dict[int, float]] = None,
            mf_iters: int = 3,
    ):
        # --------------------------------------------------
        # 1) Node unary initialization
        # --------------------------------------------------
        node_beliefs = {o.obj_id: self.node_belief(o) for o in obs}

        # --------------------------------------------------
        # 2) Parse VHF intent (1-shot + memory)
        # --------------------------------------------------
        goal_intent: Dict[int, float] = {}
        if vhf_messages:
            for msg in vhf_messages:
                gi = float(self.vhf_parser.parse_goal_intent(msg.text))
                if gi > 0.0:
                    goal_intent[msg.sender_id] = max(
                        goal_intent.get(msg.sender_id, 0.0), gi
                    )
                    self.intent_memory[msg.sender_id] = max(
                        self.intent_memory.get(msg.sender_id, 0.0), gi
                    )

        # decay intent memory
        for k in list(self.intent_memory.keys()):
            self.intent_memory[k] *= self.intent_decay
            if self.intent_memory[k] < 0.05:
                del self.intent_memory[k]

        # --------------------------------------------------
        # 3) Mean-field iteration
        # --------------------------------------------------
        for _ in range(mf_iters):
            edge_beliefs = {}

            # ---------- (A) edge update ----------
            for i in range(len(obs)):
                for j in range(i + 1, len(obs)):
                    oi, oj = obs[i], obs[j]

                    gi = goal_intent.get(oi.obj_id, 0.0)

                    scores = self.edge_unary(
                        oi, oj,
                        node_beliefs[oi.obj_id],
                        node_beliefs[oj.obj_id],
                        gi
                    )

                    # ship–ship VHF bias (logit space)
                    a, b = oi.obj_id, oj.obj_id
                    if (
                            node_beliefs[a].get("ship", 0.0) > 0.6
                            and node_beliefs[b].get("ship", 0.0) > 0.6
                    ):
                        if gi > 0.8:
                            scores["A_stand_on_B"] += 8.0
                            scores["B_give_way_A"] += 8.0
                            scores["crossing"] -= 6.0
                            scores["head_on"] -= 6.0
                            scores["overtaking"] -= 6.0
                            scores["none"] -= 1.0

                    key = (a, b)
                    scores = self.apply_temporal_potential(key, scores)
                    scores = self.apply_relation_mutex(scores)

                    edge_beliefs[key] = self._softmax(scores)

            # ==================================================
            # (B) 🔗 relation–relation coupling
            # A → TSS approaching  ⇒  others give-way to A
            # ==================================================
            for (i, j), pR in list(edge_beliefs.items()):
                oi = next(o for o in obs if o.obj_id == i)
                oj = next(o for o in obs if o.obj_id == j)

                # i = ship, j = TSS
                if oi.det_label == "ship" and oj.det_label == "tss_entrance":
                    p_app = pR.get("approaching", 0.0)

                    if p_app > 0.3:
                        for ok in obs:
                            if ok.obj_id == i:
                                continue
                            if ok.det_label != "ship":
                                continue

                            k_id = ok.obj_id

                            # ----------------------------------
                            # ✅ (1) coop 업데이트 (행동 기반)
                            # ----------------------------------
                            if delta_heading is not None:
                                dh = delta_heading.get(k_id, 0.0)
                                self.update_coop_from_action(
                                    ship_id=k_id,
                                    delta_heading=dh,
                                    p_app=p_app
                                )

                            # ----------------------------------
                            # (2) coop 반영하여 relation 보정
                            # ----------------------------------
                            key2 = tuple(sorted((k_id, i)))
                            if key2 not in edge_beliefs:
                                continue

                            p2 = dict(edge_beliefs[key2])
                            p_coop = self.coop.get(k_id, 0.5)

                            p2["B_give_way_A"] += 3.0 * p_app * p_coop
                            p2["A_stand_on_B"] += 3.0 * p_app * p_coop

                            p2["crossing"] -= 2.0 * p_app * p_coop
                            p2["head_on"] -= 2.0 * p_app
                            p2["overtaking"] -= 2.0 * p_app

                            edge_beliefs[key2] = self._softmax(p2)

            # ---------- (C) node update ----------
            node_beliefs = self.apply_node_edge_coupling(
                node_beliefs, edge_beliefs
            )

        # --------------------------------------------------
        # 4) Store temporal memory & return
        # --------------------------------------------------
        self.prev_edge_beliefs = edge_beliefs
        return node_beliefs, edge_beliefs

#############################################################################################
#                                     Visualization Module
#############################################################################################


class SceneGraphVisualizer:
    def visualize(self, obs, node_beliefs, edge_beliefs, top_k=2, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        node_colors = {
            "ship": "red",
            "buoy": "blue",
            "tss_entrance": "green",
            "unknown": "gray",
        }

        edge_colors = {
            "approaching": "green",
            "A_stand_on_B": "red",
            "A_give_way_B": "blue",
            "B_stand_on_A": "red",
            "B_give_way_A": "blue",
            "crossing": "orange",
            "head_on": "purple",
            "near": "gray",
        }

        for o in obs:
            belief = node_beliefs[o.obj_id]

            # 가장 확률 높은 label
            lbl = max(belief, key=belief.get)
            conf = belief[lbl]

            # node 시각화 (confidence 반영)
            ax.scatter(
                o.x, o.y,
                s=200,
                color=node_colors.get(lbl, "black"),
                alpha=0.3 + 0.7 * conf
            )

            ax.arrow(o.x, o.y, o.vx * 2, o.vy * 2, head_width=1.0)

            # --- belief 텍스트 생성 ---
            sorted_beliefs = sorted(
                belief.items(),
                key=lambda x: -x[1]
            )

            belief_text = f"{o.obj_id}: {lbl}\n"
            for k, v in sorted_beliefs:
                belief_text += f"{k}: {v:.2f}\n"

            ax.text(
                o.x + 1.5,
                o.y + 1.5,
                belief_text.strip(),
                fontsize=8,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
            )

        for (i, j), pR in edge_beliefs.items():
            oi = next(o for o in obs if o.obj_id == i)
            oj = next(o for o in obs if o.obj_id == j)

            items = sorted(
                [(r, p) for r, p in pR.items() if r != "none"],
                key=lambda x: -x[1]
            )[:top_k]

            for k, (rel, prob) in enumerate(items):
                ax.plot(
                    [oi.x, oj.x],
                    [oi.y, oj.y],
                    linewidth=1 + 5 * prob,
                    color=edge_colors.get(rel, "black"),
                )
                ax.text(
                    (oi.x + oj.x) / 2,
                    (oi.y + oj.y) / 2 + 3 * k,
                    f"{rel} {prob:.2f}"
                )

        ax.set_aspect("equal")
        ax.grid(True)
        return ax

#############################################################################################
#                                           Main Code
#############################################################################################


import math
import os
import imageio
import matplotlib.pyplot as plt

def update_position(o: ObjObs, dt: float):
    o.x += o.vx * dt
    o.y += o.vy * dt


def gradual_turn_towards(o: ObjObs, target_heading, max_turn_rate, dt):
    # shortest angular difference
    diff = (target_heading - o.heading + math.pi) % (2 * math.pi) - math.pi
    turn = max(-max_turn_rate * dt, min(max_turn_rate * dt, diff))
    o.heading += turn

    speed = math.hypot(o.vx, o.vy)
    o.vx = speed * math.cos(o.heading)
    o.vy = speed * math.sin(o.heading)

class SimulationRunner:
    def __init__(self, crf, visualizer):
        self.crf = crf
        self.viz = visualizer

    def run_gif(self, obs, vhf, tss, steps=30, dt=0.5):
        os.makedirs("frames", exist_ok=True)
        frames = []

        red = obs[0]
        blue = obs[1]

        # ✅ 프레임 간 heading 기억
        prev_heading = {o.obj_id: o.heading for o in obs}

        for t in range(steps):
            for o in obs:
                o.t = t

            # -------------------------
            # motion update
            # -------------------------
            target_heading = math.atan2(tss.y - red.y, tss.x - red.x)

            if t > 8:
                gradual_turn_towards(
                    red,
                    target_heading=target_heading,
                    max_turn_rate=math.radians(6),
                    dt=dt
                )

            update_position(red, dt)

            blue.x += blue.vx * dt
            blue.y += blue.vy * dt

            # -------------------------
            # ✅ delta_heading 계산 (프레임 1회)
            # -------------------------
            delta_heading = {}
            for o in obs:
                delta_heading[o.obj_id] = abs(o.heading - prev_heading[o.obj_id])
                prev_heading[o.obj_id] = o.heading

            # -------------------------
            # CRF inference
            # -------------------------
            nodes, edges = self.crf.infer_frame(
                obs,
                vhf_messages=vhf,
                delta_heading=delta_heading
            )

            # -------------------------
            # visualization
            # -------------------------
            fig, ax = plt.subplots(figsize=(7, 7))
            self.viz.visualize(obs, nodes, edges, ax=ax)
            ax.set_xlim(-80, 80)
            ax.set_ylim(-80, 120)

            fname = f"frames/frame_{t:03d}.png"
            plt.savefig(fname)
            plt.close(fig)

            frames.append(imageio.imread(fname))

        imageio.mimsave("scene_graph.gif", frames, duration=0.3)


import math
import matplotlib.pyplot as plt


def main():
    crf = DynamicCRF()
    viz = SceneGraphVisualizer()
    sim = SimulationRunner(crf, viz)

    red  = ObjObs(1, 0, -60, 20, 6, 0, 0.0, 10.0, "ship")
    blue = ObjObs(2, 0, 0, -40, 0, 6, math.pi/2, 10.0, "ship")
    tss  = ObjObs(4, 0, 0, 80, 0, 0, 0.0, 3.0, "tss_entrance")

    obs = [red, blue, tss]

    vhf = [VHFMessage(
        sender_id=1,
        text="This is red ship, we are entering TSS from south, request cooperation."
    )]

    sim.run_gif(obs, vhf, tss)


if __name__ == "__main__":
    main()
