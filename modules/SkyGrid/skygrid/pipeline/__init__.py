from .executor import PipelineExecutor, ABPExecutor, SyncExecutor
from .micro_batch import MicroBatcher
from .backpressure import BackpressureController

__all__ = [
    "PipelineExecutor",
    "ABPExecutor",
    "SyncExecutor",
    "MicroBatcher",
    "BackpressureController",
    "build_pipeline",
]


def build_pipeline(name: str, **kwargs):
    name = name.lower()
    if name == "abp":
        return ABPExecutor(**kwargs)
    if name == "sync":
        return SyncExecutor(**kwargs)
    raise ValueError(f"unknown pipeline: {name}")
