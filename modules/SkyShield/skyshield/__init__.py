"""SkyShield: radar-guided real-time counter-UAV interception artifact (RTSS 2026).

Pure-Python deterministic discrete-event implementation that takes the
authors' 10 real interception sorties as ground truth and replays /
extends them under a controlled scenario regenerator.  All numeric
results in the paper are reproduced from this package by the scripts
in ``modules/SkyShield/scripts/``.
"""

from .config import SkyShieldConfig

__all__ = ["SkyShieldConfig"]
__version__ = "0.1.0"
