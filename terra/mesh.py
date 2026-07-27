"""Self-healing sensor mesh — routing and sensing resilience.

A Terra site is a low-power wireless mesh of TLK nodes reporting to one or more
gateways. Two kinds of healing matter, and Terra does both:

  * routing self-heal (packets): when a link or node drops, neighbours re-route
    to a gateway over the best surviving path, with no central controller. This
    is standard mesh behaviour, modelled here as least-cost routing over link
    quality with on-failure re-computation.

  * sensing self-heal (data): when a node or its probe drops, the site is not
    blind at that location. Because the engine models a spatially coupled field,
    a missing reading is imputed from correlated neighbours (quality/distance
    weighted) until the node returns. This is the differentiator: the network
    heals its data, not just its packets.

Pure numpy, no radios required — this is the model used to design and test the
behaviour before it runs on hardware.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class MeshNetwork:
    pos: dict            # node id -> (x, y) metres
    gateways: set        # node ids that are gateways (sinks)
    adj: dict = field(default_factory=dict)   # id -> {neighbour: quality in (0,1]}

    # ---- topology ----------------------------------------------------------
    def add_link(self, a, b, quality: float) -> None:
        self.adj.setdefault(a, {})[b] = quality
        self.adj.setdefault(b, {})[a] = quality

    def neighbours(self, n) -> dict:
        return {k: v for k, v in self.adj.get(n, {}).items() if v > 0}

    def fail_link(self, a, b) -> None:
        self.adj.get(a, {}).pop(b, None)
        self.adj.get(b, {}).pop(a, None)

    def fail_node(self, n) -> None:
        for m in list(self.adj.get(n, {})):
            self.fail_link(n, m)
        self.adj[n] = {}

    # ---- routing self-heal -------------------------------------------------
    def routes(self) -> dict:
        """Least-cost next hop to the nearest gateway for every node. Link cost
        is -log(quality) so chaining good links stays cheap. Recompute after any
        failure to get the healed topology. Returns id -> {next_hop, cost, hops}
        or None if the node is currently partitioned from all gateways."""
        # multi-source Dijkstra outward from the gateways
        best: dict = {g: (0.0, 0, g) for g in self.gateways if g in self.pos}
        pq = [(0.0, 0, g, g) for g in best]
        heapq.heapify(pq)
        while pq:
            cost, hops, node, nxt = heapq.heappop(pq)
            if cost > best[node][0]:
                continue
            for nb, q in self.neighbours(node).items():
                c = cost + (-math.log(max(q, 1e-6)))
                if nb not in best or c < best[nb][0]:
                    best[nb] = (c, hops + 1, node)   # toward gateway via `node`
                    heapq.heappush(pq, (c, hops + 1, nb, node))
        out: dict = {}
        for n in self.pos:
            if n in self.gateways:
                out[n] = {"next_hop": n, "cost": 0.0, "hops": 0}
            elif n in best:
                cost, hops, nh = best[n]
                out[n] = {"next_hop": nh, "cost": cost, "hops": hops}
            else:
                out[n] = None
        return out

    def connectivity(self) -> float:
        """Fraction of *live* nodes (a gateway, or still holding a link) that can
        reach a gateway. Nodes taken fully offline are no longer participants and
        are excluded — the metric is how well the survivors stay connected."""
        r = self.routes()
        live = [n for n in self.pos if n in self.gateways or self.neighbours(n)]
        return float(np.mean([r[n] is not None for n in live])) if live else 0.0

    # ---- sensing self-heal -------------------------------------------------
    def impute(self, node, readings: dict, power: float = 2.0) -> float | None:
        """Estimate a dropped node's signal from its still-connected neighbours,
        inverse-distance and link-quality weighted. Returns None if isolated."""
        nbrs = [(m, q) for m, q in self.neighbours(node).items()
                if m in readings and readings[m] is not None]
        if not nbrs:
            return None
        px = np.array(self.pos[node], float)
        num = den = 0.0
        for m, q in nbrs:
            d = np.linalg.norm(np.array(self.pos[m], float) - px) + 1e-6
            w = q / d ** power
            num += w * readings[m]
            den += w
        return num / den if den else None


# ---- helpers ------------------------------------------------------------------

def build_grid(rows: int, cols: int, spacing: float = 30.0,
               link_range: float = 45.0, seed: int = 0) -> MeshNetwork:
    """A grid of nodes; a link exists between nodes within `link_range`, with
    quality decaying with distance. Gateway at the corner (0,0)."""
    rng = np.random.default_rng(seed)
    pos = {(r, c): (c * spacing, r * spacing) for r in range(rows) for c in range(cols)}
    net = MeshNetwork(pos=pos, gateways={(0, 0)})
    ids = list(pos)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            d = math.dist(pos[a], pos[b])
            if d <= link_range:
                q = max(0.05, math.exp(-d / link_range) * (0.9 + 0.1 * rng.random()))
                net.add_link(a, b, q)
    return net


def smooth_field(pos: dict, seed: int = 1) -> dict:
    """A spatially smooth latent signal over the node positions (e.g. a DO or
    temperature gradient), so neighbouring nodes are genuinely correlated."""
    rng = np.random.default_rng(seed)
    ax, ay, bx = rng.uniform(-0.03, 0.03, 3)
    base = 8.0
    return {n: base + ax * x + ay * y + 0.4 * math.sin(bx * (x + y))
            for n, (x, y) in pos.items()}
