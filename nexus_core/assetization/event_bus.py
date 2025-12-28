import asyncio
from typing import Callable, Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger("AssetEventBus")

@dataclass
class TelemetryEvent:
    """Standardized event packet for flight data."""
    event_id: str
    event_type: str  # e.g., "TELEMETRY_UPDATE", "MISSION_COMPLETE"
    source_id: str   # UAV ID or DID
    payload: Dict[str, Any]
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = datetime.now().timestamp()

class AssetEventBus:
    """
    Central Event Bus for the Assetization Layer.
    Decouples data providers (UAVs/Sim) from data consumers (Pricing/Minting).
    Singleton Pattern.
    """
    _instance = None
    _subscribers: Dict[str, List[Callable[[TelemetryEvent], None]]] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AssetEventBus, cls).__new__(cls)
            cls._subscribers = {}
        return cls._instance

    def subscribe(self, event_type: str, callback: Callable[[TelemetryEvent], None]):
        """
        Register a callback for a specific event type.
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)
        logger.info(f"Subscribed to {event_type} with {callback.__name__}")

    def publish(self, event: TelemetryEvent):
        """
        Publish an event to all subscribers.
        Non-blocking dispatch (logic dependent on implementation, here synchronous for simplicity 
        or async hook if needed).
        """
        if event.event_type in self._subscribers:
            for callback in self._subscribers[event.event_type]:
                try:
                    # In a real async system, we might use asyncio.create_task here
                    callback(event)
                except Exception as e:
                    logger.error(f"Error in subscriber {callback.__name__}: {e}")

    @classmethod
    def get_instance(cls):
        return cls()

