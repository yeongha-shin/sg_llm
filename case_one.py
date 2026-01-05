from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Set
import math

import matplotlib.pyplot as plt
import imageio
import os

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
    # (optional) detector per-class confidence dictionary, if you have it:
    det_scores: Optional[Dict[str, float]] = None


#############################################################################################
#                                           Taxonomy
#############################################################################################
NODE_LABELS: List[str] = [
    "ship", "buoy", "tss_entrance", "land",
    "bridge", "crane", "fishing_gear", "tire", "unknown"
]

# Relation sets by class-pair category
SHIP_SHIP_REL: Set[str] = {"give_way", "stand_on", "overtaking", "overtaken", "none"}
SHIP_DEST_REL: Set[str] = {"approaching", "passing", "none"}  # ship <-> tss_entrance
SHIP_OBS_REL: Set[str] = {"colliding", "mission_operating", "none"}  # land/bridge/crane as obstacles
SHIP_BUOY_REL: Set[str] = {"avoiding_left", "avoiding_right", "well_clear", "none"}
SHIP_PART_REL: Set[str] = {"on", "none"}  # fishing_gear / tire etc attached/on


def is_obstacle(cls: str) -> bool:
    return cls in {"land", "bridge", "crane"}


def is_part(cls: str) -> bool:
    return cls in {"fishing_gear", "tire"}


def allowed_relations(ci: str, cj: str) -> Set[str]:
    """
    Directed relation R_{i->j}.
    Allowed set depends on (C_i, C_j).
    """
    if ci == "ship" and cj == "ship":
        return SHIP_SHIP_REL
    if ci == "ship" and cj == "tss_entrance":
        return SHIP_DEST_REL
    if ci == "ship" and cj == "buoy":
        return SHIP_BUOY_REL
    if ci == "ship" and is_obstacle(cj):
        return SHIP_OBS_REL
    if ci == "ship" and is_part(cj):
        return SHIP_PART_REL

    # Non-ship as subject: by default only "none"
    return {"none"}

class SceneGraphCRF:
    def __init__(self):
        self.node_labels = NODE_LABELS

        # global relation vocabulary (저장/시각화 편의용)
        self.all_relations = sorted({
            *SHIP_SHIP_REL, *SHIP_DEST_REL, *SHIP_OBS_REL, *SHIP_BUOY_REL, *SHIP_PART_REL
        })

        # 하이퍼파라미터(나중에 학습 가능)
        self.lambda_incompat = 50.0  # 호환 안 되는 relation이면 큰 패널티
        self.lambda_T = 2.0          # 시간 일관성(temporal smoothness)

        self.prev_edge_beliefs: Dict[Tuple[int, int], Dict[str, float]] = {}

        # 거리 기준들
        self.near_dist = 30.0
        self.collision_dist = 8.0

    @staticmethod
    def _wrap(a: float) -> float:
        return (a + math.pi) % (2 * math.pi) - math.pi

    @staticmethod
    def _softmax_from_energy(E: Dict[str, float]) -> Dict[str, float]:
        """
        p(k) ∝ exp(-E(k))
        """
        m = min(E.values())
        exps = {k: math.exp(-(v - m)) for k, v in E.items()}
        s = sum(exps.values()) + 1e-12
        return {k: v / s for k, v in exps.items()}

    def rel(self, oi: ObjObs, oj: ObjObs):
        """
        i 기준 상대 기하:
        - distance d
        - bearing beta (i heading 좌표계)
        - closing speed (positive=closing)
        """
        dx, dy = oj.x - oi.x, oj.y - oi.y
        d = math.hypot(dx, dy) + 1e-6
        beta = self._wrap(math.atan2(dy, dx) - oi.heading)
        rvx, rvy = oj.vx - oi.vx, oj.vy - oi.vy
        closing = -(dx * rvx + dy * rvy) / d
        return d, beta, closing

    def node_unary_energy(self, o: ObjObs) -> Dict[str, float]:
        """
        φ_i(C_i): lower is better
        """
        E = {c: 1.0 for c in self.node_labels}
        E["unknown"] = 0.8

        # (1) detector single-label evidence
        if o.det_label in E:
            s = float(o.det_score) if o.det_score is not None else 1.0
            s = max(0.0, min(s, 1.0))
            E[o.det_label] -= 2.5 * s  # w_det=2.5 정도

        # (2) optional per-class scores (있으면 더 좋음)
        if o.det_scores:
            for c, sc in o.det_scores.items():
                if c in E:
                    sc = max(0.0, min(float(sc), 1.0))
                    E[c] -= 1.0 * sc  # 약하게 추가 반영

        # (3) simple motion prior (optional)
        speed = math.hypot(o.vx, o.vy)
        # 빠르면 ship, 매우 느리면 buoy 경향
        E["ship"] -= 1.5 if speed > 1.0 else 0.0
        E["buoy"] -= 1.0 if speed < 0.5 else 0.0

        return E

    def init_node_beliefs(self, obs: List[ObjObs]) -> Dict[int, Dict[str, float]]:
        qC = {}
        for o in obs:
            E = self.node_unary_energy(o)
            qC[o.obj_id] = self._softmax_from_energy(E)
        return qC

    def compat_energy(self, ci: str, cj: str, r: str) -> float:
        """
        ψ_CR(Ci,Cj,Rij): taxonomy gating
        """
        if r not in allowed_relations(ci, cj):
            return self.lambda_incompat
        return 0.0

    def edge_unary_energy(self, oi: ObjObs, oj: ObjObs) -> Dict[str, float]:
        """
        ψ_ij(R_{i->j}; Xi,Xj): geometry-only energy
        """
        d, beta, closing = self.rel(oi, oj)

        # 기본값: 다 높은 에너지(별로)
        E = {r: 2.0 for r in self.all_relations}

        # 멀면 none이 유리
        if d > 2.5 * self.near_dist:
            E["none"] = 0.7

        # ship->tss: approaching/passing는 closing sign으로
        if closing > 0:
            E["approaching"] = 0.9
        else:
            E["passing"] = 1.0

        # ship->obstacle: 가까우면 colliding
        if d < self.collision_dist:
            E["colliding"] = 0.5
        else:
            E["mission_operating"] = 1.2

        # ship->buoy: near면 avoiding, far면 well_clear
        if d < self.near_dist:
            if beta > 0:
                E["avoiding_left"] = 1.0
                E["avoiding_right"] = 1.6
            else:
                E["avoiding_right"] = 1.0
                E["avoiding_left"] = 1.6
        else:
            E["well_clear"] = 0.9

        # ship->ship: give_way/stand_on/overtaking/overtaken(아주 단순 버전)
        if abs(beta) < math.radians(30) and closing > 0.2:
            E["overtaking"] = 0.9
        if abs(beta) > math.radians(150) and closing > 0.2:
            E["overtaken"] = 1.0

        if closing > 0.2:
            if beta < 0:  # target on starboard -> give way tendency
                E["give_way"] = 1.0
                E["stand_on"] = 1.4
            else:
                E["stand_on"] = 1.0
                E["give_way"] = 1.4

        # ship->part: on
        E["on"] = 1.0

        return E

    def temporal_expected_energy(self, key: Tuple[int, int], r: str) -> float:
        if key not in self.prev_edge_beliefs:
            return 0.0
        qprev = self.prev_edge_beliefs[key]
        return self.lambda_T * (1.0 - float(qprev.get(r, 0.0)))

    def infer_frame(self, obs: List[ObjObs], mf_iters: int = 5):
        id2obs = {o.obj_id: o for o in obs}
        ids = [o.obj_id for o in obs]

        # (1) init node beliefs qC
        qC = self.init_node_beliefs(obs)

        # (2) init edge beliefs qR for directed edges i->j
        qR: Dict[Tuple[int, int], Dict[str, float]] = {}
        for i in ids:
            for j in ids:
                if i == j:
                    continue
                qR[(i, j)] = {r: 1.0 / len(self.all_relations) for r in self.all_relations}

        # cache unary energies (for speed & clarity)
        phi = {o.obj_id: self.node_unary_energy(o) for o in obs}

        # (3) mean-field iterations
        for _ in range(mf_iters):
            # --- update edges ---
            for (i, j) in list(qR.keys()):
                oi, oj = id2obs[i], id2obs[j]
                psi_geom = self.edge_unary_energy(oi, oj)

                E_r = {}
                for r in self.all_relations:
                    E = psi_geom[r]

                    # expected compat: E_{qCi,qCj}[ψ_CR]
                    Ec = 0.0
                    for ci, pci in qC[i].items():
                        for cj, pcj in qC[j].items():
                            Ec += pci * pcj * self.compat_energy(ci, cj, r)
                    E += Ec

                    # temporal
                    E += self.temporal_expected_energy((i, j), r)

                    E_r[r] = E

                qR[(i, j)] = self._softmax_from_energy(E_r)

            # --- update nodes ---
            for i in ids:
                E_c = {}
                for c in self.node_labels:
                    E = phi[i][c]

                    # outgoing edges (i->j)
                    for j in ids:
                        if j == i:
                            continue
                        for cj, pcj in qC[j].items():
                            for r, pr in qR[(i, j)].items():
                                E += pcj * pr * self.compat_energy(c, cj, r)

                        # incoming edges (j->i)
                        for cj, pcj in qC[j].items():
                            for r, pr in qR[(j, i)].items():
                                E += pcj * pr * self.compat_energy(cj, c, r)

                    E_c[c] = E

                qC[i] = self._softmax_from_energy(E_c)

        # (4) store temporal memory
        self.prev_edge_beliefs = {k: v.copy() for k, v in qR.items()}

        return qC, qR

class SceneGraphVisualizer:
    def __init__(self):
        self.node_colors = {
            "ship": "red",
            "buoy": "blue",
            "tss_entrance": "green",
            "land": "brown",
            "bridge": "purple",
            "crane": "orange",
            "fishing_gear": "cyan",
            "tire": "gray",
            "unknown": "black",
        }

        self.edge_colors = {
            "approaching": "green",
            "passing": "olive",
            "give_way": "red",
            "stand_on": "blue",
            "overtaking": "purple",
            "overtaken": "purple",
            "avoiding_left": "orange",
            "avoiding_right": "orange",
            "well_clear": "gray",
            "colliding": "black",
            "mission_operating": "brown",
            "on": "cyan",
            "none": "lightgray",
        }

    def visualize(self, obs, qC, qR, ax, top_k=1):
        # =========================
        # 1) Draw nodes
        # =========================
        for o in obs:
            belief = qC[o.obj_id]
            label = max(belief, key=belief.get)
            conf = belief[label]

            ax.scatter(
                o.x, o.y,
                s=300,
                color=self.node_colors.get(label, "black"),
                alpha=0.3 + 0.7 * conf,
                edgecolors="k",
                zorder=3
            )

            ax.arrow(
                o.x, o.y,
                o.vx * 2, o.vy * 2,
                head_width=1.5,
                length_includes_head=True,
                alpha=0.6,
                zorder=4
            )

            ax.text(
                o.x + 1.5, o.y + 1.5,
                f"{o.obj_id}\n{label}",
                fontsize=8,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
                zorder=5
            )

        # =========================
        # 2) Draw graph skeleton (ALL edges)
        # =========================
        for (i, j) in qR.keys():
            oi = next(o for o in obs if o.obj_id == i)
            oj = next(o for o in obs if o.obj_id == j)

            ax.plot(
                [oi.x, oj.x],
                [oi.y, oj.y],
                linestyle="--",
                linewidth=0.6,
                color="lightgray",
                alpha=0.6,
                zorder=1
            )

        # =========================
        # 3) Draw semantic relations (top-1, non-none)
        # =========================
        for (i, j), dist in qR.items():
            oi = next(o for o in obs if o.obj_id == i)
            oj = next(o for o in obs if o.obj_id == j)

            # top-1 relation
            rel, prob = max(dist.items(), key=lambda x: x[1])
            if rel == "none":
                continue

            color = self.edge_colors.get(rel, "black")

            # edge thickness ∝ confidence
            lw = 1.0 + 6.0 * prob

            # perpendicular offset (to reduce overlap)
            dx = oj.y - oi.y
            dy = -(oj.x - oi.x)
            norm = math.hypot(dx, dy) + 1e-6
            offset = 1.2
            ox = dx / norm * offset
            oy = dy / norm * offset

            ax.annotate(
                "",
                xy=(oj.x + ox, oj.y + oy),
                xytext=(oi.x + ox, oi.y + oy),
                arrowprops=dict(
                    arrowstyle="->",
                    color=color,
                    linewidth=lw,
                    alpha=0.85
                ),
                zorder=2
            )

            # relation label at midpoint
            mx = (oi.x + oj.x) / 2 + ox
            my = (oi.y + oj.y) / 2 + oy
            ax.text(
                mx, my,
                f"{rel}\n{prob:.2f}",
                fontsize=7,
                color=color,
                ha="center",
                va="center",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
                zorder=5
            )

        ax.set_aspect("equal")
        ax.grid(True)

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

class CRFSimulationRunner:
    def __init__(self, crf: SceneGraphCRF, visualizer: SceneGraphVisualizer):
        self.crf = crf
        self.viz = visualizer

    def run_gif(self, obs: List[ObjObs], steps=30, dt=0.5, out_name="scene_graph.gif"):
        os.makedirs("frames", exist_ok=True)
        frames = []

        ship = obs[0]
        tss = next(o for o in obs if o.det_label == "tss_entrance")

        for t in range(steps):
            for o in obs:
                o.t = t

            # --- simple motion ---
            target_heading = math.atan2(tss.y - ship.y, tss.x - ship.x)
            if t > 5:
                gradual_turn_towards(
                    ship,
                    target_heading=target_heading,
                    max_turn_rate=math.radians(5),
                    dt=dt
                )

            for o in obs:
                update_position(o, dt)

            # --- CRF inference ---
            qC, qR = self.crf.infer_frame(obs, mf_iters=5)

            # --- visualization ---
            fig, ax = plt.subplots(figsize=(7, 7))
            self.viz.visualize(obs, qC, qR, ax)

            ax.set_xlim(-80, 80)
            ax.set_ylim(-80, 120)
            ax.set_title(f"Frame {t}")

            fname = f"frames/frame_{t:03d}.png"
            plt.savefig(fname)
            plt.close(fig)

            frames.append(imageio.imread(fname))

        imageio.mimsave(out_name, frames, duration=0.3)
        print(f"[✓] Saved GIF to {out_name}")

def main():
    crf = SceneGraphCRF()
    viz = SceneGraphVisualizer()
    sim = CRFSimulationRunner(crf, viz)

    # --- objects ---
    ship1 = ObjObs(
        obj_id=1, t=0,
        x=-60, y=20,
        vx=6, vy=0,
        heading=0.0,
        size=10.0,
        det_label="ship",
        det_score=0.9
    )

    ship2 = ObjObs(
        obj_id=2, t=0,
        x=0, y=-40,
        vx=0, vy=6,
        heading=math.pi / 2,
        size=10.0,
        det_label="ship",
        det_score=0.9
    )

    tss = ObjObs(
        obj_id=3, t=0,
        x=0, y=80,
        vx=0, vy=0,
        heading=0.0,
        size=5.0,
        det_label="tss_entrance",
        det_score=0.95
    )

    obs = [ship1, ship2, tss]

    sim.run_gif(obs, steps=35, dt=0.5, out_name="scene_graph.gif")


if __name__ == "__main__":
    main()