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
            "approaching", "near", "fixed", "none"
        ]
        self.near_dist = 30.0
        self.vhf_parser = VHFIntentParser()
        self.prev_edge_beliefs = {}
        self.temporal_weight = 2.0

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

    # ---------- edge unary ----------
    def edge_unary(self, oi, oj, pCi, goal_intent):
        d, bearing, closing = self.rel(oi, oj)
        scores = {r: -4.0 for r in self.relations}
        scores["none"] = -1.5

        # -----------------------------------------
        # 기존 geometry-based relation evidence
        # -----------------------------------------
        if d < self.near_dist:
            scores["near"] += 2.0
        if abs(bearing) < math.radians(15) and closing > 0:
            scores["head_on"] += 3.0
        if abs(abs(bearing) - math.pi / 2) < math.radians(30) and closing > 0:
            scores["crossing"] += 3.0

        # ============================================================
        # ✅ OR 조건 approaching: geometry OR VHF (soft, balanced)
        # ============================================================
        if pCi.get("ship", 0) > 0.6 and oj.det_label == "tss_entrance":
            # 1) Geometry evidence for "approaching" (soft continuous)
            # - closing > 0 (approaching speed)
            # - distance small-ish
            geom_app = 0.0

            # closing term: normalize (튜닝 가능)
            if closing > 0:
                geom_app += min(closing / 3.0, 1.0)  # 0~1

            # distance term: 가까울수록 1에 가까움 (튜닝 가능)
            geom_app += max(0.0, 1.0 - d / 120.0)  # 0~1

            # 2) VHF evidence (already 0~1)
            vhf_app = float(goal_intent)  # 0 or 1 in your parser

            # 3) OR 방식 합성: 둘 중 하나만 커도 올라감, 둘 다면 더 올라감
            w_geom = 2.0  # geometry weight (튜닝)
            w_vhf = 2.0  # vhf weight (튜닝)

            scores["approaching"] += w_geom * geom_app + w_vhf * vhf_app

            # none도 조금 눌러주되, 과하게 누르지 않음 (100% 방지)
            scores["none"] -= 0.5 * (geom_app + vhf_app)

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
                node_beliefs[i]["ship"] += 1.5
                node_beliefs[j]["ship"] += 0.5

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

        # fixed ↔ crossing
        scores["crossing"] -= w_medium * max(0.0, scores["fixed"])
        scores["fixed"] -= w_medium * max(0.0, scores["crossing"])

        return scores

    # ---------- inference ----------
    def infer_frame(
            self,
            obs: List[ObjObs],
            vhf_messages: Optional[List[VHFMessage]] = None,
            mf_iters: int = 3,  # ⭐ mean-field iteration 횟수
    ):
        # --------------------------------------------------
        # 1. Node unary initialization
        # --------------------------------------------------
        node_beliefs = {o.obj_id: self.node_belief(o) for o in obs}

        # --------------------------------------------------
        # 2. Parse VHF intent
        # --------------------------------------------------
        goal_intent = {}
        if vhf_messages:
            for msg in vhf_messages:
                gi = self.vhf_parser.parse_goal_intent(msg.text)
                if gi > 0:
                    goal_intent[msg.sender_id] = gi

        # --------------------------------------------------
        # 3. Mean-field iteration
        # --------------------------------------------------
        for _ in range(mf_iters):
            edge_beliefs = {}

            # ---------- edge update ----------
            for i in range(len(obs)):
                for j in range(i + 1, len(obs)):
                    oi, oj = obs[i], obs[j]
                    gi = goal_intent.get(oi.obj_id, 0.0)

                    scores = self.edge_unary(
                        oi, oj,
                        node_beliefs[oi.obj_id],
                        gi
                    )

                    key = (oi.obj_id, oj.obj_id)
                    scores = self.apply_temporal_potential(key, scores)
                    scores = self.apply_relation_mutex(scores)

                    edge_beliefs[key] = self._softmax(scores)

            # ---------- ship–ship bias (기존 로직 유지) ----------
            for (a, b), pR in edge_beliefs.items():
                if (
                        node_beliefs[a].get("ship", 0) > 0.6
                        and node_beliefs[b].get("ship", 0) > 0.6
                ):
                    gi = goal_intent.get(a, 0.0)
                    if gi > 0.8:
                        p = dict(pR)

                        # a가 VHF로 협조 요청 → a는 stand-on, b는 give-way
                        p["A_stand_on_B"] += 4.0
                        p["B_give_way_A"] += 4.0

                        # conflicting relations suppress
                        p["crossing"] *= 0.05
                        p["head_on"] *= 0.05
                        p["none"] *= 0.05

                        edge_beliefs[(a, b)] = self._softmax(p)

            # ---------- node update ----------
            node_beliefs = self.apply_node_edge_coupling(
                node_beliefs, edge_beliefs
            )

        # --------------------------------------------------
        # 4. Store temporal memory & return
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

        for t in range(steps):
            for o in obs:
                o.t = t

            # red → TSS
            # dx, dy = tss.x - red.x, tss.y - red.y
            # red.heading = math.atan2(dy, dx)
            # red.vx = 6 * math.cos(red.heading)
            # red.vy = 6 * math.sin(red.heading)
            # red.x += red.vx * dt
            # red.y += red.vy * dt

            target_heading = math.atan2(tss.y - red.y, tss.x - red.x)

            if t > 8:  # ⏱️ 일정 시간 이후부터 회두 시작
                gradual_turn_towards(
                    red,
                    target_heading=target_heading,
                    max_turn_rate=math.radians(6),  # deg/sec
                    dt=dt
                )

            update_position(red, dt)

            # blue straight
            blue.x += blue.vx * dt
            blue.y += blue.vy * dt

            nodes, edges = self.crf.infer_frame(obs, vhf)

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
