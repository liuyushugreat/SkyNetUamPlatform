"""Partition-quality metrics reported in the paper."""

from __future__ import annotations

import numpy as np

from .baseline import Partition


def partition_metrics(part: Partition, entities, cells_per_side: int = 16) -> dict:
    """Compute:

    * ``edge_cut``          — # entity-entity cell-neighbor pairs that cross
                              an edge boundary (normalized by total pairs)
    * ``load_imbalance``    — max(size) / mean(size)
    * ``spatial_compactness`` — mean squared intra-edge distance in cell units
    """
    side = int(cells_per_side)
    C = side * side
    cells = np.array([e.home_cell for e in entities], dtype=np.int64)
    assignment = part.assignment
    cell_to_edge: dict[int, int] = {}
    for c in range(C):
        mask = cells == c
        if mask.any():
            # dominant edge (majority vote) for that cell
            vals, cnts = np.unique(assignment[mask], return_counts=True)
            cell_to_edge[c] = int(vals[int(np.argmax(cnts))])

    # Edge-cut via 4-neighbor graph of populated cells.
    cut = 0
    tot = 0
    for c, e_c in cell_to_edge.items():
        r0, c0 = c // side, c % side
        for dr, dc in [(1, 0), (0, 1)]:
            nr, nc = r0 + dr, c0 + dc
            if 0 <= nr < side and 0 <= nc < side:
                nb = nr * side + nc
                if nb in cell_to_edge:
                    tot += 1
                    if cell_to_edge[nb] != e_c:
                        cut += 1
    edge_cut = cut / tot if tot else 0.0

    sizes = part.sizes.astype(np.float64)
    load_imb = float(sizes.max() / max(1.0, sizes.mean()))

    # Spatial compactness per edge: mean squared cell-centroid distance.
    rows = (cells // side).astype(np.float64)
    cols = (cells %  side).astype(np.float64)
    compact = 0.0
    for k in range(part.num_edges):
        mask = assignment == k
        if mask.sum() <= 1:
            continue
        cx = float(cols[mask].mean())
        cy = float(rows[mask].mean())
        d2 = (cols[mask] - cx) ** 2 + (rows[mask] - cy) ** 2
        compact += float(d2.mean())
    compact /= max(1, part.num_edges)

    return {
        "edge_cut": float(edge_cut),
        "load_imbalance": float(load_imb),
        "spatial_compactness": float(compact),
        "sizes": sizes.tolist(),
    }
