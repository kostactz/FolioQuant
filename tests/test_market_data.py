"""
Unit tests for market data message models.

Tests cover:
- SnapshotMessage parsing and creation
- L2UpdateMessage parsing and creation
- HeartbeatMessage parsing and creation
- parse_message() factory function
- Edge cases and error handling
"""

import pytest
from src.models.market_data import (
    SnapshotMessage,
    L2UpdateMessage,
    HeartbeatMessage,
    parse_message
)


class TestSnapshotMessage:
    """Test SnapshotMessage parsing and functionality."""
    
    def test_from_dict_valid_snapshot(self):
        """Should parse a valid snapshot message."""
        data = {
            "type": "snapshot",
            "product_id": "BTC-USD",
            "bids": [
                ["50000.00", "1.5"],
                ["49999.00", "2.0"]
            ],
            "asks": [
                ["50001.00", "1.0"],
                ["50002.00", "1.5"]
            ]
        }
        
        msg = SnapshotMessage.from_dict(data)
        
        assert msg.product_id == "BTC-USD"
        assert len(msg.bids) == 2
        assert len(msg.asks) == 2
        assert msg.bids[0] == ("50000.00", "1.5")
        assert msg.asks[0] == ("50001.00", "1.0")
    
    def test_from_dict_empty_sides(self):
        """Should handle snapshots with empty bids or asks."""
        data = {
            "type": "snapshot",
            "product_id": "ETH-USD",
            "bids": [],
            "asks": []
        }
        
        msg = SnapshotMessage.from_dict(data)
        
        assert msg.product_id == "ETH-USD"
        assert msg.bids == []
        assert msg.asks == []
    
    def test_from_dict_wrong_type(self):
        """Should raise ValueError for wrong message type."""
        data = {
            "type": "l2update",
            "product_id": "BTC-USD",
            "bids": [],
            "asks": []
        }
        
        with pytest.raises(ValueError, match="Expected snapshot message"):
            SnapshotMessage.from_dict(data)
    
    def test_from_dict_missing_fields(self):
        """Should raise KeyError for missing required fields."""
        data = {
            "type": "snapshot"
            # Missing product_id
        }
        
        with pytest.raises(KeyError):
            SnapshotMessage.from_dict(data)
    
    def test_to_dict(self):
        """Should convert back to dictionary format."""
        msg = SnapshotMessage(
            product_id="BTC-USD",
            bids=[("50000.00", "1.0")],
            asks=[("50001.00", "1.0")]
        )
        
        data = msg.to_dict()
        
        assert data["type"] == "snapshot"
        assert data["product_id"] == "BTC-USD"
        assert data["bids"] == [("50000.00", "1.0")]
        assert data["asks"] == [("50001.00", "1.0")]
    
    def test_repr(self):
        """Should have a useful string representation."""
        msg = SnapshotMessage(
            product_id="BTC-USD",
            bids=[("50000.00", "1.0"), ("49999.00", "2.0")],
            asks=[("50001.00", "1.5")]
        )
        
        repr_str = repr(msg)
        
        assert "SnapshotMessage" in repr_str
        assert "BTC-USD" in repr_str
        assert "bids=2" in repr_str
        assert "asks=1" in repr_str


class TestL2UpdateMessage:
    """Test L2UpdateMessage parsing and functionality."""
    
    def test_from_dict_valid_update(self):
        """Should parse a valid l2update message."""
        data = {
            "type": "l2update",
            "product_id": "BTC-USD",
            "time": "2024-01-28T12:00:00.000000Z",
            "changes": [
                ["buy", "50000.00", "1.5"],
                ["sell", "50001.00", "0"]
            ]
        }
        
        msg = L2UpdateMessage.from_dict(data)
        
        assert msg.product_id == "BTC-USD"
        assert msg.time == "2024-01-28T12:00:00.000000Z"
        assert len(msg.changes) == 2
        assert msg.changes[0] == ("buy", "50000.00", "1.5")
        assert msg.changes[1] == ("sell", "50001.00", "0")
    
    def test_from_dict_empty_changes(self):
        """Should handle updates with no changes."""
        data = {
            "type": "l2update",
            "product_id": "ETH-USD",
            "time": "2024-01-28T12:00:00.000000Z",
            "changes": []
        }
        
        msg = L2UpdateMessage.from_dict(data)
        
        assert msg.changes == []
    
    def test_from_dict_wrong_type(self):
        """Should raise ValueError for wrong message type."""
        data = {
            "type": "snapshot",
            "product_id": "BTC-USD",
            "time": "2024-01-28T12:00:00.000000Z",
            "changes": []
        }
        
        with pytest.raises(ValueError, match="Expected l2update message"):
            L2UpdateMessage.from_dict(data)
    
    def test_get_bid_changes(self):
        """Should extract only bid-side changes."""
        msg = L2UpdateMessage(
            product_id="BTC-USD",
            time="2024-01-28T12:00:00.000000Z",
            changes=[
                ("buy", "50000.00", "1.5"),
                ("sell", "50001.00", "2.0"),
                ("buy", "49999.00", "0"),
                ("sell", "50002.00", "1.0")
            ]
        )
        
        bid_changes = msg.get_bid_changes()
        
        assert len(bid_changes) == 2
        assert bid_changes[0] == ("50000.00", "1.5")
        assert bid_changes[1] == ("49999.00", "0")
    
    def test_get_ask_changes(self):
        """Should extract only ask-side changes."""
        msg = L2UpdateMessage(
            product_id="BTC-USD",
            time="2024-01-28T12:00:00.000000Z",
            changes=[
                ("buy", "50000.00", "1.5"),
                ("sell", "50001.00", "2.0"),
                ("buy", "49999.00", "0"),
                ("sell", "50002.00", "1.0")
            ]
        )
        
        ask_changes = msg.get_ask_changes()
        
        assert len(ask_changes) == 2
        assert ask_changes[0] == ("50001.00", "2.0")
        assert ask_changes[1] == ("50002.00", "1.0")
    
    def test_to_dict(self):
        """Should convert back to dictionary format."""
        msg = L2UpdateMessage(
            product_id="BTC-USD",
            time="2024-01-28T12:00:00.000000Z",
            changes=[("buy", "50000.00", "1.5")]
        )
        
        data = msg.to_dict()
        
        assert data["type"] == "l2update"
        assert data["product_id"] == "BTC-USD"
        assert data["time"] == "2024-01-28T12:00:00.000000Z"
        assert data["changes"] == [("buy", "50000.00", "1.5")]
    
    def test_repr(self):
        """Should have a useful string representation."""
        msg = L2UpdateMessage(
            product_id="BTC-USD",
            time="2024-01-28T12:00:00.000000Z",
            changes=[("buy", "50000.00", "1.5"), ("sell", "50001.00", "2.0")]
        )
        
        repr_str = repr(msg)
        
        assert "L2UpdateMessage" in repr_str
        assert "BTC-USD" in repr_str
        assert "changes=2" in repr_str
        assert "2024-01-28T12:00:00.000000Z" in repr_str


class TestHeartbeatMessage:
    """Test HeartbeatMessage parsing and functionality."""
    
    def test_from_dict_valid_heartbeat(self):
        """Should parse a valid heartbeat message."""
        data = {
            "type": "heartbeat",
            "product_id": "BTC-USD",
            "sequence": 12345,
            "last_trade_id": 67890,
            "time": "2024-01-28T12:00:00.000000Z"
        }
        
        msg = HeartbeatMessage.from_dict(data)
        
        assert msg.product_id == "BTC-USD"
        assert msg.sequence == 12345
        assert msg.last_trade_id == 67890
        assert msg.time == "2024-01-28T12:00:00.000000Z"
    
    def test_from_dict_wrong_type(self):
        """Should raise ValueError for wrong message type."""
        data = {
            "type": "snapshot",
            "product_id": "BTC-USD",
            "sequence": 12345,
            "last_trade_id": 67890,
            "time": "2024-01-28T12:00:00.000000Z"
        }
        
        with pytest.raises(ValueError, match="Expected heartbeat message"):
            HeartbeatMessage.from_dict(data)
    
    def test_from_dict_missing_fields(self):
        """Should raise KeyError for missing required fields."""
        data = {
            "type": "heartbeat",
            "product_id": "BTC-USD"
            # Missing sequence, last_trade_id, time
        }
        
        with pytest.raises(KeyError):
            HeartbeatMessage.from_dict(data)
    
    def test_to_dict(self):
        """Should convert back to dictionary format."""
        msg = HeartbeatMessage(
            product_id="BTC-USD",
            sequence=12345,
            last_trade_id=67890,
            time="2024-01-28T12:00:00.000000Z"
        )
        
        data = msg.to_dict()
        
        assert data["type"] == "heartbeat"
        assert data["product_id"] == "BTC-USD"
        assert data["sequence"] == 12345
        assert data["last_trade_id"] == 67890
        assert data["time"] == "2024-01-28T12:00:00.000000Z"
    
    def test_repr(self):
        """Should have a useful string representation."""
        msg = HeartbeatMessage(
            product_id="BTC-USD",
            sequence=12345,
            last_trade_id=67890,
            time="2024-01-28T12:00:00.000000Z"
        )
        
        repr_str = repr(msg)
        
        assert "HeartbeatMessage" in repr_str
        assert "BTC-USD" in repr_str
        assert "seq=12345" in repr_str
        assert "2024-01-28T12:00:00.000000Z" in repr_str


class TestParseMessage:
    """Test the parse_message factory function."""
    
    def test_parse_snapshot(self):
        """Should parse snapshot messages."""
        data = {
            "type": "snapshot",
            "product_id": "BTC-USD",
            "bids": [["50000.00", "1.0"]],
            "asks": [["50001.00", "1.0"]]
        }
        
        msg = parse_message(data)
        
        assert isinstance(msg, SnapshotMessage)
        assert msg.product_id == "BTC-USD"
    
    def test_parse_l2update(self):
        """Should parse l2update messages."""
        data = {
            "type": "l2update",
            "product_id": "BTC-USD",
            "time": "2024-01-28T12:00:00.000000Z",
            "changes": [["buy", "50000.00", "1.5"]]
        }
        
        msg = parse_message(data)
        
        assert isinstance(msg, L2UpdateMessage)
        assert msg.product_id == "BTC-USD"
    
    def test_parse_heartbeat(self):
        """Should parse heartbeat messages."""
        data = {
            "type": "heartbeat",
            "product_id": "BTC-USD",
            "sequence": 12345,
            "last_trade_id": 67890,
            "time": "2024-01-28T12:00:00.000000Z"
        }
        
        msg = parse_message(data)
        
        assert isinstance(msg, HeartbeatMessage)
        assert msg.sequence == 12345
    
    def test_parse_unknown_type(self):
        """Should return None for unknown message types."""
        data = {
            "type": "unknown_type",
            "product_id": "BTC-USD"
        }
        
        msg = parse_message(data)
        
        assert msg is None
    
    def test_parse_malformed_message(self):
        """Should return None for malformed messages."""
        # Missing required fields
        data = {
            "type": "snapshot"
            # Missing product_id, bids, asks
        }
        
        msg = parse_message(data)
        
        assert msg is None
    
    def test_parse_all_types_roundtrip(self):
        """Should correctly parse and convert all message types."""
        messages = [
            {
                "type": "snapshot",
                "product_id": "BTC-USD",
                "bids": [["50000.00", "1.0"]],
                "asks": [["50001.00", "1.0"]]
            },
            {
                "type": "l2update",
                "product_id": "BTC-USD",
                "time": "2024-01-28T12:00:00.000000Z",
                "changes": [["buy", "50000.00", "1.5"]]
            },
            {
                "type": "heartbeat",
                "product_id": "BTC-USD",
                "sequence": 12345,
                "last_trade_id": 67890,
                "time": "2024-01-28T12:00:00.000000Z"
            }
        ]
        
        for data in messages:
            msg = parse_message(data)
            assert msg is not None
            
            # Round-trip: message -> dict -> message
            converted = msg.to_dict()
            msg2 = parse_message(converted)
            
            assert type(msg) == type(msg2)
            assert msg.product_id == msg2.product_id
