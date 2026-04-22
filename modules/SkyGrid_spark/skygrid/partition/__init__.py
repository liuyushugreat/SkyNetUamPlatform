from .stp import SpatioTemporalPartitioner
from .baseline import (
    HashPartitioner,
    RandomPartitioner,
    LDGPartitioner,
    Partitioner,
    Partition,
)
from .rebalance import FMRebalancer
from .metrics import partition_metrics

__all__ = [
    "Partitioner",
    "Partition",
    "SpatioTemporalPartitioner",
    "HashPartitioner",
    "RandomPartitioner",
    "LDGPartitioner",
    "FMRebalancer",
    "partition_metrics",
]


def build_partitioner(name: str, num_edges: int, **kwargs):
    name = name.lower()
    if name == "stp":
        return SpatioTemporalPartitioner(num_edges=num_edges, **kwargs)
    if name == "ldg":
        return LDGPartitioner(num_edges=num_edges, **kwargs)
    if name == "hash":
        return HashPartitioner(num_edges=num_edges, **kwargs)
    if name == "random":
        return RandomPartitioner(num_edges=num_edges, **kwargs)
    raise ValueError(f"unknown partitioner: {name}")
