import math
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter


# =====================================================
# Observation structure
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
    text: Optional[str] = None


# =====================================================
# COLREG decision logic
# =====================================================
class ColregDecision:

    def calc_rel_bearing(self, ego_x, ego_y, target_x, target_y, ego_heading):
        dx = target_x - ego_x
        dy = target_y - ego_y
        angle_to_target = math.degrees(math.atan2(dy, dx))
        relative_bearing = (90 - angle_to_target) - ego_heading
        return relative_bearing % 360

    def calc_encounter_role(self, ego_ship, target_ship):
        course_diff = (target_ship.heading - ego_ship.heading) % 360
        rel_bearing = self.calc_rel_bearing(
            ego_ship.x, ego_ship.y,
            target_ship.x, target_ship.y,
            ego_ship.heading
        )

        psi = np.radians(course_diff)
        beta = np.radians(rel_bearing)

        if target_ship.speed < 0.1:
            return "safe"

        if 7 * np.pi / 8 <= psi < 9 * np.pi / 8:
            return "head_on"

        if 9 * np.pi / 8 <= psi < 13 * np.pi / 8:
            return "starboard_crossing"
        if 3 * np.pi / 8 <= psi < 7 * np.pi / 8:
            return "port_crossing"

        if 5 * np.pi / 8 <= beta < 11 * np.pi / 8:
            return "overtaking" if ego_ship.speed > target_ship.speed else "overtaken"

        return "safe"


# =====================================================
# Helper Ship class
# =====================================================
@dataclass
class RuleShip:
    x: float
    y: float
    heading: float  # degrees
    speed: float


# =====================================================
# Rule-based Scene Graph Generator
# =====================================================
class RuleBasedSceneGraph:

    def __init__(self):
        self.colreg = ColregDecision()

    def infer(self, obs: List[ObjObs]):
        nodes: Dict[int, str] = {}
        edges: Dict[Tuple[int, int], str] = {}

        for o in obs:
            nodes[o.obj_id] = o.det_label if o.det_label else "unknown"

        for i in range(len(obs)):
            for j in range(i + 1, len(obs)):
                oi, oj = obs[i], obs[j]
                key = (oi.obj_id, oj.obj_id)

                ti, tj = nodes[oi.obj_id], nodes[oj.obj_id]

                if ti == "ship" and tj == "ship":
                    ego = RuleShip(
                        x=oi.x, y=oi.y,
                        heading=math.degrees(oi.heading),
                        speed=math.hypot(oi.vx, oi.vy),
                    )
                    tgt = RuleShip(
                        x=oj.x, y=oj.y,
                        heading=math.degrees(oj.heading),
                        speed=math.hypot(oj.vx, oj.vy),
                    )
                    edges[key] = self.colreg.calc_encounter_role(ego, tgt)

                elif ti == "ship" and tj == "tss_entrance":
                    edges[key] = "approaching" if oi.text and "approaching" in oi.text.lower() else "none"

                elif ti == "tss_entrance" and tj == "ship":
                    edges[key] = "approaching" if oj.text and "approaching" in oj.text.lower() else "none"

                else:
                    edges[key] = "none"

        return nodes, edges


# =====================================================
# Visualization
# =====================================================
def visualize_rule_based_scene_graph(
    obs: List[ObjObs],
    nodes: Dict[int, str],
    edges: Dict[Tuple[int, int], str],
    ax,
    title=""
):
    ax.set_aspect("equal")
    ax.set_xlim(-120, 120)
    ax.set_ylim(-120, 120)
    ax.grid(True)
    ax.set_title(title)

    node_colors = {
        "ship": "tab:red",
        "tss_entrance": "tab:green",
        "unknown": "gray",
    }

    edge_colors = {
        "head_on": "purple",
        "starboard_crossing": "orange",
        "port_crossing": "orange",
        "overtaking": "brown",
        "overtaken": "brown",
        "approaching": "green",
        "safe": "gray",
    }

    for o in obs:
        color = node_colors.get(nodes[o.obj_id], "black")
        ax.scatter(o.x, o.y, s=220, color=color, zorder=3)
        ax.arrow(o.x, o.y, o.vx * 2, o.vy * 2,
                 head_width=1.2, head_length=2.2,
                 fc=color, ec=color)

        if nodes[o.obj_id] == "tss_entrance":
            ax.add_patch(
                patches.Circle((o.x, o.y), radius=8,
                               fill=False, linestyle="--", color="green")
            )

    for (i, j), rel in edges.items():
        if rel == "none":
            continue
        oi = next(o for o in obs if o.obj_id == i)
        oj = next(o for o in obs if o.obj_id == j)
        ax.plot([oi.x, oj.x], [oi.y, oj.y],
                linestyle="--", linewidth=2,
                color=edge_colors.get(rel, "black"))
        ax.text((oi.x + oj.x) / 2, (oi.y + oj.y) / 2,
                rel, fontsize=9,
                bbox=dict(facecolor="white", alpha=0.7))


# =====================================================
# Motion models
# =====================================================
def update_ship_state(o: ObjObs, dt):
    o.x += o.vx * dt
    o.y += o.vy * dt


def gradual_turn(o: ObjObs, target_heading, max_turn_rate, dt):
    diff = (target_heading - o.heading + math.pi) % (2 * math.pi) - math.pi
    turn = np.clip(diff, -max_turn_rate * dt, max_turn_rate * dt)
    o.heading += turn
    speed = math.hypot(o.vx, o.vy)
    o.vx = speed * math.cos(o.heading)
    o.vy = speed * math.sin(o.heading)


# =====================================================
# Simulation + GIF
# =====================================================
def run_simulation_and_save_gif():
    dt = 0.5
    steps = 40

    ship1 = ObjObs(
        1, 0, -80, 20, 6, 0, 0.0, 10.0,
        "ship", "own ship is approaching traffic separation scheme"
    )

    ship2 = ObjObs(
        2, 0, 0, -80, 0, 6, math.pi / 2, 10.0,
        "ship", "crossing vessel"
    )

    tss = ObjObs(
        4, 0, 0, 80, 0, 0, 0.0, 3.0,
        "tss_entrance"
    )

    obs = [ship1, ship2, tss]
    sg = RuleBasedSceneGraph()

    fig, ax = plt.subplots(figsize=(8, 8))

    def animate(frame):
        if frame > 10:
            ship1.text = "own ship is turning to port and approaching TSS"
            gradual_turn(ship1, math.pi / 2, math.radians(5), dt)

        update_ship_state(ship1, dt)
        update_ship_state(ship2, dt)

        ax.clear()
        nodes, edges = sg.infer(obs)
        visualize_rule_based_scene_graph(
            obs, nodes, edges, ax,
            title=f"t = {frame}"
        )

    ani = FuncAnimation(fig, animate, frames=steps)
    ani.save("colreg_scene.gif", writer=PillowWriter(fps=4))
    plt.close()


# =====================================================
# Main
# =====================================================
if __name__ == "__main__":
    run_simulation_and_save_gif()
