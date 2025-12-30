from dataclasses import dataclass
from typing import Optional


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
import math
from typing import Dict, List, Optional
# from vhf_parser import VHFIntentParser
# from observations import ObjObs, VHFMessage


class LearningFreeDynamicCRF:
    def __init__(self):
        self.relations = [
            "head_on", "crossing", "overtaking",
            "stand_on", "yielding",
            "approaching", "near", "fixed", "none"
        ]
        self.near_dist = 30.0
        self.vhf_parser = VHFIntentParser()

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

        if d < self.near_dist:
            scores["near"] += 2.0
        if abs(bearing) < math.radians(15) and closing > 0:
            scores["head_on"] += 3.0
        if abs(abs(bearing) - math.pi / 2) < math.radians(30) and closing > 0:
            scores["crossing"] += 3.0

        # HARD TSS + VHF PRIOR
        if pCi.get("ship", 0) > 0.6 and oj.det_label == "tss_entrance":
            scores["approaching"] += 4.0
            if goal_intent > 0.8:
                scores["approaching"] += 10.0
                scores["none"] -= 8.0

        return scores

    # ---------- inference ----------
    def infer_frame(
        self,
        obs: List[ObjObs],
        vhf_messages: Optional[List[VHFMessage]] = None,
    ):
        node_beliefs = {o.obj_id: self.node_belief(o) for o in obs}

        goal_intent = {}
        if vhf_messages:
            for msg in vhf_messages:
                gi = self.vhf_parser.parse_goal_intent(msg.text)
                if gi > 0:
                    goal_intent[msg.sender_id] = gi

        edge_beliefs = {}
        for i in range(len(obs)):
            for j in range(i + 1, len(obs)):
                oi, oj = obs[i], obs[j]
                gi = goal_intent.get(oi.obj_id, 0.0)
                scores = self.edge_unary(
                    oi, oj,
                    node_beliefs[oi.obj_id],
                    gi
                )
                edge_beliefs[(oi.obj_id, oj.obj_id)] = self._softmax(scores)

        for (a, b), pR in edge_beliefs.items():
            if (
                    node_beliefs[a].get("ship", 0) > 0.6
                    and node_beliefs[b].get("ship", 0) > 0.6
            ):
                gi = goal_intent.get(a, 0.0)
                if gi > 0.8:
                    p = dict(pR)
                    p["stand_on"] += 4.0
                    p["yielding"] += 3.0
                    p["crossing"] *= 0.05
                    p["none"] *= 0.05
                    edge_beliefs[(a, b)] = self._softmax(p)

        return node_beliefs, edge_beliefs
############################################################################################
import matplotlib.pyplot as plt


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
            "stand_on": "red",
            "yielding": "blue",
            "crossing": "orange",
            "head_on": "purple",
            "near": "gray",
        }

        for o in obs:
            lbl = max(node_beliefs[o.obj_id], key=node_beliefs[o.obj_id].get)
            ax.scatter(o.x, o.y, s=200, color=node_colors.get(lbl, "black"))
            ax.arrow(o.x, o.y, o.vx * 2, o.vy * 2, head_width=1.0)
            ax.text(o.x + 1, o.y + 1, f"{o.obj_id}:{lbl}")

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

############################################################################################
import math
import matplotlib.pyplot as plt
# from observations import ObjObs, VHFMessage
# from dynamic_crf import LearningFreeDynamicCRF
# from visualizer import SceneGraphVisualizer
#

def main():
    crf = LearningFreeDynamicCRF()
    viz = SceneGraphVisualizer()
    sim = SimulationRunner(crf, viz)

    red  = ObjObs(1, 0, -60, 20, 6, 0, 0.0, 10.0, "ship")
    blue = ObjObs(2, 0, 0, -40, 0, 6, math.pi/2, 10.0, "ship")
    buoy = ObjObs(3, 0, 25, 20, 0, 0, 0.0, 1.2, "buoy")
    tss  = ObjObs(4, 0, 0, 80, 0, 0, 0.0, 3.0, "tss_entrance")

    obs = [red, blue, buoy, tss]

    vhf = [VHFMessage(
        sender_id=1,
        text="This is red ship, we are entering TSS from south, request cooperation."
    )]

    sim.run_gif(obs, vhf, tss)


if __name__ == "__main__":
    main()
