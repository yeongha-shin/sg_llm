import math
from dataclasses import dataclass
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import imageio
import os

# =====================================================
# Observation structures
# =====================================================
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


# =====================================================
# VHF intent parser (rule-based)
# =====================================================
class VHFIntentParser:
    @staticmethod
    def parse_goal_intent(vhf: VHFMessage) -> float:
        text = vhf.text.lower()
        keywords = [
            "entering tss",
            "intend to enter tss",
            "request cooperation",
            "proceeding to traffic separation scheme",
        ]
        return 1.0 if any(k in text for k in keywords) else 0.0


# =====================================================
# Learning-free CRF with HARD TSS + VHF PRIOR
# =====================================================
class LearningFreeDynamicCRF:
    def __init__(self):
        self.relations = [
            "head_on",
            "crossing",
            "overtaking",
            "stand_on",
            "yielding",
            "approaching",
            "avoiding_right",
            "near",
            "fixed",
            "none",
        ]
        self.near_dist = 30.0
        self.vhf_parser = VHFIntentParser()

    # ---------------- utils ----------------
    def _wrap(self, a):
        return (a + math.pi) % (2 * math.pi) - math.pi

    def _softmax(self, scores: Dict[str, float]):
        mx = max(scores.values())
        exps = {k: math.exp(v - mx) for k, v in scores.items()}
        s = sum(exps.values()) + 1e-12
        return {k: v / s for k, v in exps.items()}

    # ---------------- node belief ----------------
    def node_belief(self, o: ObjObs):
        # 🔒 TSS is a FACT
        if o.det_label == "tss_entrance":
            return {
                "ship": 0.0,
                "buoy": 0.0,
                "bridge_mark": 0.0,
                "tss_entrance": 1.0,
                "unknown": 0.0,
            }

        speed = math.hypot(o.vx, o.vy)
        s = {
            "ship": 2.0 if speed > 1.0 else -2.0,
            "buoy": 2.0 if speed < 0.5 else -2.0,
            "bridge_mark": -2.0,
            "tss_entrance": -3.0,
            "unknown": -1.0,
        }

        if o.det_label in s:
            s[o.det_label] += 3.0

        return self._softmax(s)

    # ---------------- geometry ----------------
    def rel(self, oi, oj):
        dx, dy = oj.x - oi.x, oj.y - oi.y
        d = math.hypot(dx, dy) + 1e-6
        bearing = self._wrap(math.atan2(dy, dx) - oi.heading)
        rvx, rvy = oj.vx - oi.vx, oj.vy - oi.vy
        closing = -(dx * rvx + dy * rvy) / d
        return d, bearing, closing

    # ---------------- edge unary (LOGIT SPACE) ----------------
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

        # ---------- HARD ship -> TSS ----------
        if pCi.get("ship", 0) > 0.6 and oj.det_label == "tss_entrance":
            scores["approaching"] += 4.0  # factual relation

            if goal_intent > 0.8:
                # 🔥 STRONG PRIOR OVERRIDE
                scores["approaching"] += 10.0
                scores["none"] -= 8.0
                scores["near"] -= 3.0

        return scores

    # ---------------- inference ----------------
    def infer_frame(
        self,
        obs: List[ObjObs],
        vhf_messages: Optional[List[VHFMessage]] = None,
    ):
        # node beliefs
        node_beliefs = {o.obj_id: self.node_belief(o) for o in obs}

        # VHF intent
        goal_intent = {}
        if vhf_messages:
            for msg in vhf_messages:
                gi = self.vhf_parser.parse_goal_intent(msg)
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
                    goal_intent=gi
                )
                edge_beliefs[(oi.obj_id, oj.obj_id)] = self._softmax(scores)

        # ---------- ship-ship override ----------
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


# =====================================================
# Visualization (top-K)
# =====================================================
def visualize_scene_graph(obs, node_beliefs, edge_beliefs, top_k=2, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 9))

    node_colors = {
        "ship": "tab:red",
        "buoy": "tab:blue",
        "tss_entrance": "tab:green",
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

    # ---------- nodes ----------
    for o in obs:
        lbl = max(node_beliefs[o.obj_id], key=node_beliefs[o.obj_id].get)
        ax.scatter(o.x, o.y, s=250, color=node_colors.get(lbl, "black"), zorder=3)
        ax.arrow(o.x, o.y, o.vx * 2, o.vy * 2, head_width=1.2, zorder=2)
        ax.text(o.x + 1, o.y + 1, f"{o.obj_id}:{lbl}", fontsize=9)

    # ---------- edges ----------
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
                linewidth=1 + 6 * prob,
                color=edge_colors.get(rel, "black"),
                alpha=0.8,
                zorder=1
            )
            mx, my = (oi.x + oj.x) / 2, (oi.y + oj.y) / 2
            ax.text(mx, my + 3 * k, f"{rel} {prob:.2f}", fontsize=8)

    ax.set_aspect("equal")
    ax.grid(True)

    return ax

# =====================================================
# Demo
# =====================================================
def demo():
    crf = LearningFreeDynamicCRF()

    obs = [
        ObjObs(1, 0, -60, 20, 6, 0, 0.0, 10.0, "ship"),          # 🔴 red
        ObjObs(2, 0, 0, -40, 0, 6, math.pi / 2, 10.0, "ship"),  # 🔵 blue
        ObjObs(3, 0, 25, 20, 0, 0, 0.0, 1.2, "buoy"),
        ObjObs(4, 0, 0, 80, 0, 0, 0.0, 3.0, "tss_entrance"),
    ]

    vhf = [
        VHFMessage(
            sender_id=1,
            text="This is red ship, we are entering TSS from south, request cooperation."
        )
    ]

    nodes, edges = crf.infer_frame(obs, vhf)
    visualize_scene_graph(obs, nodes, edges)


def demo_gif():
    crf = LearningFreeDynamicCRF()

    red  = ObjObs(1, 0, -60, 20, 6, 0, 0.0, 10.0, "ship")
    blue = ObjObs(2, 0, 0, -40, 0, 6, math.pi / 2, 10.0, "ship")
    buoy = ObjObs(3, 0, 25, 20, 0, 0, 0.0, 1.2, "buoy")
    tss  = ObjObs(4, 0, 0, 80, 0, 0, 0.0, 3.0, "tss_entrance")

    vhf = [
        VHFMessage(
            sender_id=1,
            text="This is red ship, we are entering TSS from south, request cooperation."
        )
    ]

    obs = [red, blue, buoy, tss]

    out_dir = "frames"
    os.makedirs(out_dir, exist_ok=True)
    frame_files = []

    dt = 0.5
    speed_red = 6.0
    speed_blue = 6.0

    for t in range(30):
        for o in obs:
            o.t = t

        # ---------- red moves toward TSS ----------
        dx = tss.x - red.x
        dy = tss.y - red.y
        red.heading = math.atan2(dy, dx)
        red.vx = speed_red * math.cos(red.heading)
        red.vy = speed_red * math.sin(red.heading)
        red.x += red.vx * dt
        red.y += red.vy * dt

        # ---------- blue straight ----------
        blue.vx = speed_blue * math.cos(blue.heading)
        blue.vy = speed_blue * math.sin(blue.heading)
        blue.x += blue.vx * dt
        blue.y += blue.vy * dt

        nodes, edges = crf.infer_frame(obs, vhf)

        fig, ax = plt.subplots(figsize=(7, 7))
        visualize_scene_graph(obs, nodes, edges, top_k=2, ax=ax)

        ax.set_title(f"t = {t}")
        ax.set_xlim(-80, 80)
        ax.set_ylim(-80, 120)

        fname = f"{out_dir}/frame_{t:03d}.png"
        plt.savefig(fname, dpi=120)
        plt.close(fig)

        frame_files.append(fname)

    images = [imageio.imread(f) for f in frame_files]
    imageio.mimsave("scene_graph.gif", images, duration=0.3)

    print("✅ scene_graph.gif generated (no plt.show)")

if __name__ == "__main__":
    # demo()
    demo_gif()
