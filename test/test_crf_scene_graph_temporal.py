import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# -----------------------------
# Dummy observation structure
# -----------------------------
@dataclass
class ObjObs:
    obj_id: int
    t: int
    x: float
    y: float
    vx: float
    vy: float
    heading: float   # radians
    size: float      # rough length / scale
    det_label: Optional[str] = None
    det_score: Optional[float] = None


# -----------------------------
# CRF Scene Graph (learning-free)
# -----------------------------
class LearningFreeDynamicCRF:
    def __init__(
        self,
        classes=("ship", "buoy", "bridge_mark", "unknown"),
        relations=("head_on", "crossing", "overtaking", "near", "fixed", "none"),
        # semantic: head_on and overtaking shouldn't both be high
        mutex_pairs=(("head_on", "overtaking"),),
        near_dist=25.0,
        lambda_mutex=6.0,
        lambda_temp_node=2.0,
        lambda_temp_edge=2.0,
        iters=5,
    ):
        self.C = list(classes)
        self.R = list(relations)
        self.near_dist = near_dist
        self.lambda_mutex = lambda_mutex
        self.lambda_temp_node = lambda_temp_node
        self.lambda_temp_edge = lambda_temp_edge
        self.iters = iters
        self.mutex_pairs = {tuple(sorted(p)) for p in mutex_pairs}

        # previous beliefs for temporal smoothing
        self.prev_node_beliefs: Dict[int, Dict[str, float]] = {}
        self.prev_edge_beliefs: Dict[Tuple[int, int], Dict[str, float]] = {}

    # ---------- helpers ----------
    def _wrap(self, a: float) -> float:
        return (a + math.pi) % (2 * math.pi) - math.pi

    def _softmax(self, scores: Dict[str, float]) -> Dict[str, float]:
        keys = list(scores.keys())
        mx = max(scores.values())
        exps = [math.exp(scores[k] - mx) for k in keys]
        s = sum(exps) + 1e-12
        return {k: exps[i] / s for i, k in enumerate(keys)}

    def _normalize(self, p: Dict[str, float]) -> Dict[str, float]:
        s = sum(p.values()) + 1e-12
        return {k: v / s for k, v in p.items()}

    # ---------- rule-based unaries ----------
    def node_unary(self, o: ObjObs) -> Dict[str, float]:
        speed = math.hypot(o.vx, o.vy)
        s = {c: -2.0 for c in self.C}
        s["unknown"] = -0.5

        # optional detector hint
        if o.det_label in s and o.det_score is not None:
            s[o.det_label] += 2.0 * float(o.det_score)

        # speed prior
        if speed > 1.0:
            s["ship"] += 2.0
            s["buoy"] -= 1.0
            s["bridge_mark"] -= 1.0
        else:
            s["buoy"] += 1.5
            s["bridge_mark"] += 1.0

        # size prior
        if o.size < 2.0:
            s["buoy"] += 1.0
        elif o.size > 8.0:
            s["ship"] += 0.5

        return s

    def edge_unary(self, oi: ObjObs, oj: ObjObs) -> Dict[str, float]:
        dx, dy = (oj.x - oi.x), (oj.y - oi.y)
        dist = math.hypot(dx, dy) + 1e-9

        # bearing of j in i-frame
        bearing = self._wrap(math.atan2(dy, dx) - oi.heading)

        # relative velocity (j - i)
        rvx, rvy = (oj.vx - oi.vx), (oj.vy - oi.vy)
        closing = -(dx * rvx + dy * rvy) / dist  # >0 => approaching

        scores = {r: -3.0 for r in self.R}
        scores["none"] = -0.5

        # near
        if dist < self.near_dist:
            scores["near"] += 2.0

        # head-on: in front and approaching
        if abs(bearing) < math.radians(15) and closing > 0.2:
            scores["head_on"] += 2.5

        # overtaking: behind and approaching
        if abs(abs(bearing) - math.pi) < math.radians(25) and closing > 0.2:
            scores["overtaking"] += 2.0

        # crossing: side and approaching
        if abs(abs(bearing) - math.pi / 2) < math.radians(25) and closing > 0.2:
            scores["crossing"] += 2.0

        # fixed: low relative speed + close
        rel_speed = math.hypot(rvx, rvy)
        if rel_speed < 0.3 and dist < self.near_dist:
            scores["fixed"] += 1.8

        return scores

    # ---------- semantic constraints (soft) ----------
    def apply_mutex(self, pR: Dict[str, float]) -> Dict[str, float]:
        p = dict(pR)
        for a, b in self.mutex_pairs:
            if a in p and b in p:
                # if both high, suppress both
                penalty = math.exp(-self.lambda_mutex * (p[a] * p[b]))
                p[a] *= penalty
                p[b] *= penalty
        return self._normalize(p)

    def apply_type_relation_consistency(
        self, pCi: Dict[str, float], pCj: Dict[str, float], pR: Dict[str, float]
    ) -> Dict[str, float]:
        # head_on/crossing/overtaking are meaningful mostly for ship-ship
        shipship = pCi.get("ship", 0.0) * pCj.get("ship", 0.0)
        p = dict(pR)
        for r in ("head_on", "crossing", "overtaking"):
            if r in p:
                p[r] *= (0.1 + 0.9 * shipship)  # if not ship-ship => downweight
        return self._normalize(p)

    # ---------- temporal smoothing (prior injection) ----------
    def temporal_smooth_node(self, obj_id: int, p: Dict[str, float]) -> Dict[str, float]:
        if obj_id not in self.prev_node_beliefs:
            return p
        prev = self.prev_node_beliefs[obj_id]
        out = {c: p[c] * (prev.get(c, 1e-6) ** self.lambda_temp_node) for c in p}
        return self._normalize(out)

    def temporal_smooth_edge(self, key: Tuple[int, int], p: Dict[str, float]) -> Dict[str, float]:
        if key not in self.prev_edge_beliefs:
            return p
        prev = self.prev_edge_beliefs[key]
        out = {r: p[r] * (prev.get(r, 1e-6) ** self.lambda_temp_edge) for r in p}
        return self._normalize(out)

    # ---------- mean-field-ish inference per frame ----------
    def infer_frame(self, obs: List[ObjObs]):
        # initialize node beliefs from unaries
        node_beliefs: Dict[int, Dict[str, float]] = {}
        for o in obs:
            p = self._softmax(self.node_unary(o))
            p = self.temporal_smooth_node(o.obj_id, p)
            node_beliefs[o.obj_id] = p

        # initialize edge beliefs from unaries
        edge_beliefs: Dict[Tuple[int, int], Dict[str, float]] = {}
        for i in range(len(obs)):
            for j in range(i + 1, len(obs)):
                oi, oj = obs[i], obs[j]
                key = (oi.obj_id, oj.obj_id)
                p = self._softmax(self.edge_unary(oi, oj))
                p = self.temporal_smooth_edge(key, p)
                # apply semantic constraints using current node beliefs
                p = self.apply_type_relation_consistency(node_beliefs[oi.obj_id], node_beliefs[oj.obj_id], p)
                p = self.apply_mutex(p)
                edge_beliefs[key] = p

        # iterate: refine with constraints (simple loop; no learning)
        for _ in range(self.iters):
            # update edges using current nodes (type consistency + mutex)
            for i in range(len(obs)):
                for j in range(i + 1, len(obs)):
                    oi, oj = obs[i], obs[j]
                    key = (oi.obj_id, oj.obj_id)
                    p = edge_beliefs[key]
                    p = self.apply_type_relation_consistency(node_beliefs[oi.obj_id], node_beliefs[oj.obj_id], p)
                    p = self.apply_mutex(p)
                    edge_beliefs[key] = p

            # optional: update nodes using incident edges (very light message passing)
            # Idea: if object participates in strong ship-ship COLREG relations, boost ship a bit.
            for o in obs:
                obj_id = o.obj_id
                ship_boost = 0.0
                for (a, b), pR in edge_beliefs.items():
                    if obj_id in (a, b):
                        ship_boost += (pR.get("head_on", 0.0) + pR.get("crossing", 0.0) + pR.get("overtaking", 0.0))
                # convert boost to a mild multiplicative factor on ship
                pC = dict(node_beliefs[obj_id])
                pC["ship"] *= math.exp(0.3 * ship_boost)
                node_beliefs[obj_id] = self._normalize(pC)

        # store for next frame temporal smoothing
        self.prev_node_beliefs = {k: dict(v) for k, v in node_beliefs.items()}
        self.prev_edge_beliefs = {k: dict(v) for k, v in edge_beliefs.items()}

        return node_beliefs, edge_beliefs


# -----------------------------
# Demo: dummy observations -> CRF -> scene graph
# -----------------------------
def top1(dist: Dict[str, float]) -> Tuple[str, float]:
    k = max(dist, key=dist.get)
    return k, dist[k]


def demo():
    crf = LearningFreeDynamicCRF(iters=5)

    # Frame t=0 (dummy)
    # ship1 moving north, ship2 moving south in front (head-on-ish)
    # buoy static near them
    obs0 = [
        ObjObs(obj_id=1, t=0, x=0,  y=0,  vx=0, vy=3, heading=math.radians(90), size=10.0, det_label="ship", det_score=0.6),
        ObjObs(obj_id=2, t=0, x=0,  y=60, vx=0, vy=-3, heading=math.radians(-90), size=10.0, det_label="ship", det_score=0.6),
        ObjObs(obj_id=3, t=0, x=10, y=20, vx=0, vy=0, heading=0.0, size=1.0, det_label=None, det_score=None),  # unknown -> buoy likely
    ]

    # Frame t=1 (dummy): moved a bit closer
    obs1 = [
        ObjObs(obj_id=1, t=1, x=0,  y=3,  vx=0, vy=3,  heading=math.radians(90),  size=10.0),
        ObjObs(obj_id=2, t=1, x=0,  y=57, vx=0, vy=-3, heading=math.radians(-90), size=10.0),
        ObjObs(obj_id=3, t=1, x=10, y=20, vx=0, vy=0, heading=0.0, size=1.0),
    ]

    for obs in (obs0, obs1):
        node_beliefs, edge_beliefs = crf.infer_frame(obs)
        t = obs[0].t
        print(f"\n=== Scene Graph @ t={t} ===")

        print("\n[NODES]")
        for o in obs:
            label, conf = top1(node_beliefs[o.obj_id])
            print(f"  obj {o.obj_id}: top={label:12s} conf={conf:.3f}  full={node_beliefs[o.obj_id]}")

        print("\n[EDGES]")
        for (i, j), pR in edge_beliefs.items():
            rel, conf = top1(pR)
            print(f"  ({i},{j}): top={rel:12s} conf={conf:.3f}  full={pR}")

if __name__ == "__main__":
    demo()
