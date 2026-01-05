from dataclasses import dataclass
from typing import Optional, Dict, List
import math
import os
import imageio
import matplotlib.pyplot as plt

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


#############################################################################################
#                                           CRF Module
#############################################################################################

class DynamicCRF:
    def __init__(self):
        self.relations = [
            "head_on", "crossing", "overtaking",
            "A_stand_on_B", "A_give_way_B",
            "B_stand_on_A", "B_give_way_A",
            "approaching", "passing", "near", "none"
        ]

        self.near_dist = 30.0
        self.prev_edge_beliefs = {}
        self.temporal_weight = 2.0

        # 행동 기반 협조성 추정
        self.coop = {}          # ship_id -> p(cooperative)
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

    # ---------- edge unary ----------
    def edge_unary_energy(self, oi, oj, pCi, pCj):
        d, bearing, closing = self.rel(oi, oj)
        E = {r: 0.0 for r in self.relations}

        oi_is_ship = pCi.get("ship", 0.0) > 0.6
        oj_is_ship = pCj.get("ship", 0.0) > 0.6
        oj_is_tss = (oj.det_label == "tss_entrance")

        # proximity
        if d < self.near_dist:
            E["near"] -= 1.0

        # ship–ship geometry
        if oi_is_ship and oj_is_ship:
            if abs(bearing) < math.radians(15) and closing > 0:
                E["head_on"] -= 2.0
            if abs(abs(bearing) - math.pi / 2) < math.radians(30) and closing > 0:
                E["crossing"] -= 2.0
        else:
            for r in [
                "head_on", "crossing", "overtaking",
                "A_stand_on_B", "A_give_way_B",
                "B_stand_on_A", "B_give_way_A"
            ]:
                E[r] += 5.0

        # ship → TSS
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
        if key not in self.prev_edge_beliefs:
            return E

        prev_r = max(self.prev_edge_beliefs[key],
                     key=self.prev_edge_beliefs[key].get)

        for r in E:
            if r == prev_r:
                E[r] -= self.temporal_weight
            else:
                E[r] += self.temporal_weight
        return E

    def contextual_energy(self, Rij, Rik):
        if Rik == "approaching":
            if Rij in ["crossing", "head_on"]:
                return +2.0
            if Rij in ["A_stand_on_B", "B_give_way_A"]:
                return -2.0
        return 0.0

    def apply_node_edge_coupling(self, node_beliefs, edge_beliefs):
        for (i, j), pR in edge_beliefs.items():
            if pR.get("approaching", 0.0) > 0.5:
                node_beliefs[i]["ship"] += 2.0
                if "tss_entrance" in node_beliefs[j]:
                    node_beliefs[j]["tss_entrance"] += 2.0
                    node_beliefs[j]["ship"] -= 2.0

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
        evidence = min(abs(delta_heading) / math.radians(5), 1.0)
        w = 0.5 * p_app
        prior = self.coop.get(ship_id, 0.5)
        self.coop[ship_id] = (1 - w) * prior + w * evidence

    # ---------- inference ----------
    def infer_frame(
        self,
        obs: List[ObjObs],
        delta_heading: Optional[Dict[int, float]] = None,
        mf_iters: int = 3,
    ):
        node_beliefs = {o.obj_id: self.node_belief(o) for o in obs}

        for _ in range(mf_iters):
            edge_beliefs = {}

            # (A) Edge update
            for i in range(len(obs)):
                for j in range(i + 1, len(obs)):
                    oi, oj = obs[i], obs[j]
                    a, b = oi.obj_id, oj.obj_id

                    E = self.edge_unary_energy(
                        oi, oj,
                        node_beliefs[a],
                        node_beliefs[b]
                    )
                    E = self.temporal_energy((a, b), E)

                    edge_beliefs[(a, b)] = self._softmax(
                        {r: -E[r] for r in E}
                    )

            # (B) Contextual coupling + cooperation update
            for (i, j), pR in list(edge_beliefs.items()):
                oi = next(o for o in obs if o.obj_id == i)
                oj = next(o for o in obs if o.obj_id == j)

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

                            E2 = {
                                r: -math.log(edge_beliefs[key2].get(r, 1e-12))
                                for r in self.relations
                            }
                            for r in E2:
                                E2[r] += self.contextual_energy(
                                    Rij=r,
                                    Rik="approaching"
                                )
                            edge_beliefs[key2] = self._softmax(
                                {r: -E2[r] for r in E2}
                            )

                            if delta_heading is not None:
                                self.update_coop_from_action(
                                    ship_id=k_id,
                                    delta_heading=delta_heading.get(k_id, 0.0),
                                    p_app=p_app
                                )

            node_beliefs = self.apply_node_edge_coupling(
                node_beliefs, edge_beliefs
            )

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
            lbl = max(belief, key=belief.get)
            conf = belief[lbl]

            ax.scatter(o.x, o.y, s=200,
                       color=node_colors.get(lbl, "black"),
                       alpha=0.3 + 0.7 * conf)
            ax.arrow(o.x, o.y, o.vx * 2, o.vy * 2, head_width=1.0)

            txt = f"{o.obj_id}: {lbl}\n" + "\n".join(
                f"{k}: {v:.2f}" for k, v in sorted(
                    belief.items(), key=lambda x: -x[1]
                )
            )
            ax.text(o.x + 1.5, o.y + 1.5, txt, fontsize=8,
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

        for (i, j), pR in edge_beliefs.items():
            oi = next(o for o in obs if o.obj_id == i)
            oj = next(o for o in obs if o.obj_id == j)

            items = sorted(
                [(r, p) for r, p in pR.items() if r != "none"],
                key=lambda x: -x[1]
            )[:top_k]

            for k, (rel, prob) in enumerate(items):
                ax.plot([oi.x, oj.x], [oi.y, oj.y],
                        linewidth=1 + 5 * prob,
                        color=edge_colors.get(rel, "black"))
                ax.text((oi.x + oj.x) / 2,
                        (oi.y + oj.y) / 2 + 3 * k,
                        f"{rel} {prob:.2f}")

        ax.set_aspect("equal")
        ax.grid(True)
        return ax


#############################################################################################
#                                           Simulation
#############################################################################################

def update_position(o: ObjObs, dt: float):
    o.x += o.vx * dt
    o.y += o.vy * dt


def gradual_turn_towards(o: ObjObs, target_heading, max_turn_rate, dt):
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

    def run_gif(self, obs, tss, steps=30, dt=0.5):
        os.makedirs("frames", exist_ok=True)
        frames = []

        prev_heading = {o.obj_id: o.heading for o in obs}

        for t in range(steps):
            for o in obs:
                o.t = t

            red, blue = obs[0], obs[1]
            target_heading = math.atan2(tss.y - red.y, tss.x - red.x)

            if t > 8:
                gradual_turn_towards(
                    red,
                    target_heading,
                    max_turn_rate=math.radians(6),
                    dt=dt
                )

            update_position(red, dt)
            update_position(blue, dt)

            delta_heading = {}
            for o in obs:
                delta_heading[o.obj_id] = abs(o.heading - prev_heading[o.obj_id])
                prev_heading[o.obj_id] = o.heading

            nodes, edges = self.crf.infer_frame(
                obs,
                delta_heading=delta_heading
            )

            fig, ax = plt.subplots(figsize=(7, 7))
            self.viz.visualize(obs, nodes, edges, ax=ax)
            ax.set_xlim(-80, 80)
            ax.set_ylim(-80, 120)

            fname = f"frames/frame_{t:03d}.png"
            plt.savefig(fname)
            plt.close(fig)
            frames.append(imageio.imread(fname))

        imageio.mimsave("scene_graph.gif", frames, duration=0.3)


#############################################################################################
#                                           Main
#############################################################################################

def main():
    crf = DynamicCRF()
    viz = SceneGraphVisualizer()
    sim = SimulationRunner(crf, viz)

    red  = ObjObs(1, 0, -60, 20, 6, 0, 0.0, 10.0, "ship")
    blue = ObjObs(2, 0, 0, -40, 0, 6, math.pi / 2, 10.0, "ship")
    tss  = ObjObs(4, 0, 0, 80, 0, 0, 0.0, 3.0, "tss_entrance")

    obs = [red, blue, tss]
    sim.run_gif(obs, tss)


if __name__ == "__main__":
    main()
