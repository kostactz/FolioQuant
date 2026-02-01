"""
Market Data Models: Type-safe representations of Coinbase WebSocket messages

This module defines dataclasses for the three primary message types from
the Coinbase WebSocket API:
- SnapshotMessage: Initial book state
- L2UpdateMessage: Incremental book changes
- HeartbeatMessage: Connection keepalive

These models provide type safety, validation, and a clean interface for
the rest of the application to consume market data.

Author: FolioQuant Team
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from decimal import Decimal
import logging

logger = logging.getLogger(__name__)


@dataclass
class SnapshotMessage:
    """
    Represents a Coinbase snapshot message containing the full order book state.
    
    This is the first message received after subscribing to level2_batch.
    It provides a complete snapshot of all price levels on both sides of the book.
    
    Attributes:
        product_id: Trading pair identifier (e.g., "BTC-USD")
        bids: List of [price, size] tuples for bid side
        asks: List of [price, size] tuples for ask side
    """
    product_id: str
    bids: List[Tuple[str, str]]
    asks: List[Tuple[str, str]]
    
    @classmethod
    def from_dict(cls, data: dict) -> "SnapshotMessage":
        """
        Create a SnapshotMessage from a raw dictionary.
        
        Args:
            data: Raw message dictionary from WebSocket
        
        Returns:
            SnapshotMessage instance
        
        Raises:
            ValueError: If required fields are missing
        """
        if data.get("type") != "snapshot":
            raise ValueError(f"Expected snapshot message, got {data.get('type')}")
        
        return cls(
            product_id=data["product_id"],
            bids=[(price, size) for price, size in data.get("bids", [])],
            asks=[(price, size) for price, size in data.get("asks", [])]
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary format compatible with OrderBook."""
        return {
            "type": "snapshot",
            "product_id": self.product_id,
            "bids": self.bids,
            "asks": self.asks
        }
    
    def __repr__(self) -> str:
        return (
            f"SnapshotMessage({self.product_id}, "
            f"bids={len(self.bids)}, asks={len(self.asks)})"
        )


@dataclass
class L2UpdateMessage:
    """
    Represents a Coinbase l2update message with incremental book changes.
    
    These messages arrive continuously after the snapshot and contain
    arrays of changes to apply to the book. Each change is a tuple of
    [side, price, size].
    
    Critical: size is an ABSOLUTE value, not a delta:
    - size == 0: DELETE the price level
    - size > 0: SET the price level to this size
    
    Attributes:
        product_id: Trading pair identifier
        time: ISO timestamp of the update
        changes: List of [side, price, size] change tuples
    """
    product_id: str
    time: str
    changes: List[Tuple[str, str, str]]
    
    @classmethod
    def from_dict(cls, data: dict) -> "L2UpdateMessage":
        """
        Create an L2UpdateMessage from a raw dictionary.
        
        Args:
            data: Raw message dictionary from WebSocket
        
        Returns:
            L2UpdateMessage instance
        
        Raises:
            ValueError: If required fields are missing
        """
        if data.get("type") != "l2update":
            raise ValueError(f"Expected l2update message, got {data.get('type')}")
        
        return cls(
            product_id=data["product_id"],
            time=data["time"],
            changes=[(side, price, size) for side, price, size in data.get("changes", [])]
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary format compatible with OrderBook."""
        return {
            "type": "l2update",
            "product_id": self.product_id,
            "time": self.time,
            "changes": self.changes
        }
    
    def get_bid_changes(self) -> List[Tuple[str, str]]:
        """
        Extract only bid-side changes.
        
        Returns:
            List of (price, size) tuples for buy side
        """
        return [(price, size) for side, price, size in self.changes if side == "buy"]
    
    def get_ask_changes(self) -> List[Tuple[str, str]]:
        """
        Extract only ask-side changes.
        
        Returns:
            List of (price, size) tuples for sell side
        """
        return [(price, size) for side, price, size in self.changes if side == "sell"]
    
    def __repr__(self) -> str:
        return (
            f"L2UpdateMessage({self.product_id}, "
            f"changes={len(self.changes)}, time={self.time})"
        )


@dataclass
class HeartbeatMessage:
    """
    Represents a Coinbase heartbeat message for connection monitoring.
    
    Heartbeats are sent periodically to verify the WebSocket connection
    is still active. They also include sequence numbers for gap detection.
    
    Attributes:
        product_id: Trading pair identifier
        sequence: Monotonically increasing sequence number
        last_trade_id: ID of the most recent trade
        time: ISO timestamp
    """
    product_id: str
    sequence: int
    last_trade_id: int
    time: str
    
    @classmethod
    def from_dict(cls, data: dict) -> "HeartbeatMessage":
        """
        Create a HeartbeatMessage from a raw dictionary.
        
        Args:
            data: Raw message dictionary from WebSocket
        
        Returns:
            HeartbeatMessage instance
        
        Raises:
            ValueError: If required fields are missing
        """
        if data.get("type") != "heartbeat":
            raise ValueError(f"Expected heartbeat message, got {data.get('type')}")
        
        return cls(
            product_id=data["product_id"],
            sequence=data["sequence"],
            last_trade_id=data["last_trade_id"],
            time=data["time"]
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        return {
            "type": "heartbeat",
            "product_id": self.product_id,
            "sequence": self.sequence,
            "last_trade_id": self.last_trade_id,
            "time": self.time
        }
    
    def __repr__(self) -> str:
        return (
            f"HeartbeatMessage({self.product_id}, "
            f"seq={self.sequence}, time={self.time})"
        )


def parse_message(data: dict) -> Optional[SnapshotMessage | L2UpdateMessage | HeartbeatMessage]:
    """
    Parse a raw WebSocket message into the appropriate typed message.
    
    This is a convenience function that determines the message type
    and delegates to the appropriate from_dict() method.
    
    Args:
        data: Raw message dictionary from WebSocket
    
    Returns:
        Typed message object or None if type is not recognized
    
    Examples:
        >>> msg = parse_message({"type": "snapshot", ...})
        >>> isinstance(msg, SnapshotMessage)
        True
    """
    msg_type = data.get("type")
    
    try:
        if msg_type == "snapshot":
            return SnapshotMessage.from_dict(data)
        elif msg_type == "l2update":
            return L2UpdateMessage.from_dict(data)
        elif msg_type == "heartbeat":
            return HeartbeatMessage.from_dict(data)
        else:
            logger.warning(f"Unknown message type: {msg_type}")
            return None
    except (KeyError, ValueError) as e:
        logger.error(f"Failed to parse {msg_type} message: {e}")
        return None
