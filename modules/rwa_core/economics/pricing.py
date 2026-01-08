from dataclasses import dataclass

@dataclass
class VoxelParams:
    x: int
    y: int
    z: int
    congestion_level: float

class CongestionPricingModel:
    """
    Classic congestion pricing logic (non-neural).
    """
    def __init__(self, base_rate: float = 1.0):
        self.base_rate = base_rate

    def get_price(self, voxel: VoxelParams) -> float:
        return self.base_rate * (1 + voxel.congestion_level)

