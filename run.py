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
    def edge_unary_energy(self, oi, oj, pCi, pCj):
        """
        Unary energy for relation R_ij.
        Lower energy = more plausible relation.
        """
        d, bearing, closing = self.rel(oi, oj)

        E = {r: 0.0 for r in self.relations}

        oi_is_ship = pCi.get("ship", 0.0) > 0.6
        oj_is_ship = pCj.get("ship", 0.0) > 0.6
        oj_is_tss = (oj.det_label == "tss_entrance")

        # ----------------------------
        # proximity
        # ----------------------------
        if d < self.near_dist:
            E["near"] -= 1.0

        # ----------------------------
        # ship–ship geometry
        # ----------------------------
        if oi_is_ship and oj_is_ship:
            if abs(bearing) < math.radians(15) and closing > 0:
                E["head_on"] -= 2.0
            if abs(abs(bearing) - math.pi / 2) < math.radians(30) and closing > 0:
                E["crossing"] -= 2.0
        else:
            # ship-only relations invalid
            for r in [
                "head_on", "crossing", "overtaking",
                "A_stand_on_B", "A_give_way_B",
                "B_stand_on_A", "B_give_way_A"
            ]:
                E[r] += 5.0

        # ----------------------------
        # ship → TSS
        # ----------------------------
        if oi_is_ship and oj_is_tss:
            if closing > 0:
                E["approaching"] -= 2.0
            else:
                E["passing"] -= 1.0
        else:
            E["approaching"] += 5.0
            E["passing"] += 5.0

        return E

    def temporal_energy(self, key, E):
        """
        Temporal consistency potential.
        Penalize relation changes across frames.
        """
        if key not in self.prev_edge_beliefs:
            return E

        prev_r = max(
            self.prev_edge_beliefs[key],
            key=self.prev_edge_beliefs[key].get
        )

        for r in E:
            if r == prev_r:
                E[r] -= self.temporal_weight
            else:
                E[r] += self.temporal_weight

        return E

    def contextual_energy(self, Rij, Rik):
        """
        Higher-order contextual potential between relations.
        """
        if Rik == "approaching":
            if Rij in ["crossing", "head_on"]:
                return +2.0  # conflict
            if Rij in ["A_stand_on_B", "B_give_way_A"]:
                return -2.0  # consistent
        return 0.0

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
        # 2) Parse VHF intent (memory update only)
        # --------------------------------------------------
        if vhf_messages:
            for msg in vhf_messages:
                gi = float(self.vhf_parser.parse_goal_intent(msg.text))
                if gi > 0.0:
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

            # ==============================
            # (A) Edge update (ENERGY)
            # ==============================
            for i in range(len(obs)):
                for j in range(i + 1, len(obs)):
                    oi, oj = obs[i], obs[j]
                    a, b = oi.obj_id, oj.obj_id

                    # ---- unary energy ----
                    E = self.edge_unary_energy(
                        oi, oj,
                        node_beliefs[a],
                        node_beliefs[b]
                    )

                    # ---- temporal energy ----
                    E = self.temporal_energy((a, b), E)

                    # ---- energy → probability ----
                    edge_beliefs[(a, b)] = self._softmax(
                        {r: -E[r] for r in E}
                    )

            # ==================================================
            # (B) Higher-order contextual coupling (ENERGY STYLE)
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
                            if ok.obj_id == i or ok.det_label != "ship":
                                continue

                            k_id = ok.obj_id
                            key2 = tuple(sorted((k_id, i)))
                            if key2 not in edge_beliefs:
                                continue

                            # --- rebuild energy from belief ---
                            E2 = {
                                r: -math.log(edge_beliefs[key2].get(r, 1e-12))
                                for r in self.relations
                            }

                            # contextual energy
                            for r in E2:
                                E2[r] += self.contextual_energy(
                                    Rij=r,
                                    Rik="approaching"
                                )

                            # update belief
                            edge_beliefs[key2] = self._softmax(
                                {r: -E2[r] for r in E2}
                            )

                            # optional: update cooperation belief
                            if delta_heading is not None:
                                dh = delta_heading.get(k_id, 0.0)
                                self.update_coop_from_action(
                                    ship_id=k_id,
                                    delta_heading=dh,
                                    p_app=p_app
                                )

            # ==============================
            # (C) Node update
            # ==============================
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
