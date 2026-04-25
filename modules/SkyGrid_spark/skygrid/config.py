"""Typed, YAML-loadable configuration dataclasses.

Every numeric knob the paper reports on is exposed here; the YAML loader
preserves nested defaults so reviewers can override a single field with
``--override foo.bar=42``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------- workload


@dataclass
class MobilityConfig:
    model: str = "corridor"
    speed_mps: float = 12.0
    corridor_sigma: float = 0.02


@dataclass
class JitterConfig:
    cov: float = 0.35


@dataclass
class WorkloadConfig:
    name: str = "city_medium"
    num_entities: int = 10_000
    area_km: float = 8.0
    duration_s: float = 60.0
    event_rate_per_entity_s: float = 1.5
    mobility: MobilityConfig = field(default_factory=MobilityConfig)
    jitter: JitterConfig = field(default_factory=JitterConfig)


# ---------------------------------------------------------------------- DAG


@dataclass
class OpSpec:
    name: str
    kind: str                   # "nn" or "symbolic"
    cost_flops: float
    input_bytes: int
    output_bytes: int
    prefers: str                # "cloud" or "edge"
    batchable: bool = False
    batch_sweet: int = 1
    state_refs: int = 0         # spatial state lookups per invocation


@dataclass
class DAGConfig:
    ops: list[OpSpec] = field(default_factory=list)
    edges: list[tuple[str, str]] = field(default_factory=list)


# ---------------------------------------------------------------------- fabric


@dataclass
class CloudConfig:
    num_devices: int = 1
    tflops: float = 80.0
    queue_capacity: int = 20_000


@dataclass
class EdgeConfig:
    num_nodes: int = 4
    tflops: float = 1.0
    queue_capacity: int = 2048
    memory_gb: float = 128.0        # DGX Spark unified memory
    storage_iops: int = 1_500_000   # GP Spark NVMe-oF peak random-read
    rdma_bw_gbps: float = 100.0     # ConnectX-7 per-port
    # Optional per-edge TFLOPS override used for heterogeneous or
    # degraded-mode experiments. When None, every edge uses ``tflops``.
    # When a list of length ``num_nodes`` is provided, each edge gets
    # its own compute capacity, enabling hetero / graceful-degradation
    # studies without changing the rest of the fabric.
    tflops_per_node: list[float] | None = None


@dataclass
class StateTierConfig:
    """Three-tier state access: hot (unified mem) / warm (NVMe) / cold (cloud)."""
    hot_capacity: int = 10_000
    hot_latency_ms: float = 0.001       # DGX Spark unified memory (~1 µs)
    warm_capacity: int = 100_000
    warm_latency_ms: float = 0.020      # GP Spark NVMe-oF / RDMA (~20 µs)
    cold_latency_ms: float = 12.0       # edge→cloud RTT
    hot_hit_rate: float = 0.70
    warm_hit_rate: float = 0.25
    enabled: bool = True


@dataclass
class NetworkConfig:
    edge_cloud_latency_ms: float = 12.0
    edge_cloud_bw_gbps: float = 1.0
    edge_edge_latency_ms: float = 4.0
    edge_edge_bw_gbps: float = 0.5
    jitter_ms: float = 1.5


@dataclass
class FabricConfig:
    cloud: CloudConfig = field(default_factory=CloudConfig)
    edge: EdgeConfig = field(default_factory=EdgeConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    state_tier: StateTierConfig = field(default_factory=StateTierConfig)


# ---------------------------------------------------------------------- STP / COP / ABP


@dataclass
class GridConfig:
    cells_per_side: int = 16


@dataclass
class STPConfig:
    alpha_spatial: float = 1.0
    beta_ruledep: float = 0.6
    gamma_balance: float = 0.4
    refine_iters: int = 2


@dataclass
class RebalanceConfig:
    trigger_imbalance: float = 1.25
    amortize_every_s: float = 10.0


@dataclass
class PartitionConfig:
    method: str = "stp"
    grid: GridConfig = field(default_factory=GridConfig)
    stp: STPConfig = field(default_factory=STPConfig)
    rebalance: RebalanceConfig = field(default_factory=RebalanceConfig)


@dataclass
class COPConfig:
    epsilon: float = 0.10
    max_swaps: int = 64


@dataclass
class PlacementConfig:
    method: str = "cop"
    cop: COPConfig = field(default_factory=COPConfig)


@dataclass
class ABPConfig:
    microbatch_max: int = 64
    microbatch_timeout_ms: float = 8.0
    staleness_bound: int = 2
    backpressure_watermark_high: float = 0.85
    backpressure_watermark_low: float = 0.55


@dataclass
class PipelineConfig:
    method: str = "abp"
    abp: ABPConfig = field(default_factory=ABPConfig)


# ---------------------------------------------------------------------- top-level


@dataclass
class SkyGridConfig:
    seed: int = 20260928
    workload: WorkloadConfig = field(default_factory=WorkloadConfig)
    dag: DAGConfig = field(default_factory=DAGConfig)
    fabric: FabricConfig = field(default_factory=FabricConfig)
    partition: PartitionConfig = field(default_factory=PartitionConfig)
    placement: PlacementConfig = field(default_factory=PlacementConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    output_dir: str = "outputs"

    # --------------------------------------------------------- loaders

    @staticmethod
    def load(path: str | Path) -> "SkyGridConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return SkyGridConfig.from_dict(raw)

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "SkyGridConfig":
        wl = raw.get("workload", {})
        mob = wl.get("mobility", {})
        jit = wl.get("jitter", {})
        workload = WorkloadConfig(
            name=wl.get("name", "default"),
            num_entities=int(wl.get("num_entities", 10_000)),
            area_km=float(wl.get("area_km", 8.0)),
            duration_s=float(wl.get("duration_s", 60.0)),
            event_rate_per_entity_s=float(wl.get("event_rate_per_entity_s", 1.5)),
            mobility=MobilityConfig(**mob),
            jitter=JitterConfig(**jit),
        )

        dag_raw = raw.get("dag", {})
        ops = [OpSpec(**o) for o in dag_raw.get("ops", [])]
        edges = [tuple(e) for e in dag_raw.get("edges", [])]
        dag = DAGConfig(ops=ops, edges=edges)

        fab = raw.get("fabric", {})
        edge_raw = fab.get("edge", {})
        tflops_per_node = edge_raw.get("tflops_per_node")
        if tflops_per_node is not None:
            tflops_per_node = [float(x) for x in tflops_per_node]
        edge = EdgeConfig(
            num_nodes=int(edge_raw.get("num_nodes", 4)),
            tflops=float(edge_raw.get("tflops", 1.0)),
            queue_capacity=int(edge_raw.get("queue_capacity", 2048)),
            memory_gb=float(edge_raw.get("memory_gb", 128.0)),
            storage_iops=int(edge_raw.get("storage_iops", 1_500_000)),
            rdma_bw_gbps=float(edge_raw.get("rdma_bw_gbps", 100.0)),
            tflops_per_node=tflops_per_node,
        )
        st_raw = fab.get("state_tier", {})
        state_tier = StateTierConfig(
            hot_capacity=int(st_raw.get("hot_capacity", 10_000)),
            hot_latency_ms=float(st_raw.get("hot_latency_ms", 0.001)),
            warm_capacity=int(st_raw.get("warm_capacity", 100_000)),
            warm_latency_ms=float(st_raw.get("warm_latency_ms", 0.020)),
            cold_latency_ms=float(st_raw.get("cold_latency_ms", 12.0)),
            hot_hit_rate=float(st_raw.get("hot_hit_rate", 0.70)),
            warm_hit_rate=float(st_raw.get("warm_hit_rate", 0.25)),
            enabled=bool(st_raw.get("enabled", True)),
        )
        fabric = FabricConfig(
            cloud=CloudConfig(**fab.get("cloud", {})),
            edge=edge,
            network=NetworkConfig(**fab.get("network", {})),
            state_tier=state_tier,
        )

        part = raw.get("partition", {})
        partition = PartitionConfig(
            method=part.get("method", "stp"),
            grid=GridConfig(**part.get("grid", {})),
            stp=STPConfig(**part.get("stp", {})),
            rebalance=RebalanceConfig(**part.get("rebalance", {})),
        )

        plc = raw.get("placement", {})
        placement = PlacementConfig(
            method=plc.get("method", "cop"),
            cop=COPConfig(**plc.get("cop", {})),
        )

        pipe = raw.get("pipeline", {})
        pipeline = PipelineConfig(
            method=pipe.get("method", "abp"),
            abp=ABPConfig(**pipe.get("abp", {})),
        )

        return SkyGridConfig(
            seed=int(raw.get("seed", 20260928)),
            workload=workload,
            dag=dag,
            fabric=fabric,
            partition=partition,
            placement=placement,
            pipeline=pipeline,
            output_dir=raw.get("output_dir", "outputs"),
        )

    # --------------------------------------------------------- helpers

    def with_overrides(self, **overrides: Any) -> "SkyGridConfig":
        """Return a shallow copy with top-level fields overridden.

        Useful for scripted sweeps: ``cfg.with_overrides(partition=...)``.
        """
        return replace(self, **overrides)
