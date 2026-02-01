"""
Unit tests for the OrderBook class.

Tests cover:
- Snapshot initialization
- L2 update application
- Price level deletion (size=0)
- Best bid/ask calculation
- Mid-price calculation
- Spread calculation
- Micro-price calculation
- Edge cases (empty book, crossed market)
"""

import pytest
from decimal import Decimal
from src.models.order_book import OrderBook


class TestOrderBookInitialization:
    """Test order book initialization and basic state."""
    
    def test_empty_book_creation(self):
        """Empty book should have no bids or asks."""
        book = OrderBook("BTC-USD")
        
        assert book.product_id == "BTC-USD"
        assert len(book.bids) == 0
        assert len(book.asks) == 0
        assert book.best_bid is None
        assert book.best_ask is None
    
    def test_empty_book_properties(self):
        """Empty book should return None for all calculated properties."""
        book = OrderBook("ETH-USD")
        
        assert book.mid_price is None
        assert book.spread is None
        assert book.micro_price is None
        assert not book.is_valid()


class TestSnapshotApplication:
    """Test snapshot message processing."""
    
    def test_apply_simple_snapshot(self):
        """Should correctly initialize from a snapshot."""
        book = OrderBook("BTC-USD")
        
        snapshot = {
            "type": "snapshot",
            "product_id": "BTC-USD",
            "bids": [
                ["50000.00", "1.5"],
                ["49999.00", "2.0"],
                ["49998.00", "0.5"]
            ],
            "asks": [
                ["50001.00", "1.0"],
                ["50002.00", "1.5"],
                ["50003.00", "2.0"]
            ]
        }
        
        book.apply_snapshot(snapshot)
        
        assert len(book.bids) == 3
        assert len(book.asks) == 3
        assert book.product_id == "BTC-USD"
    
    def test_snapshot_bids_sorted_descending(self):
        """Bids should be sorted in descending order (highest first)."""
        book = OrderBook("BTC-USD")
        
        snapshot = {
            "type": "snapshot",
            "product_id": "BTC-USD",
            "bids": [
                ["49998.00", "0.5"],  # Lowest
                ["50000.00", "1.5"],  # Highest
                ["49999.00", "2.0"],  # Middle
            ],
            "asks": [["50001.00", "1.0"]]
        }
        
        book.apply_snapshot(snapshot)
        
        # Best bid should be the highest price
        assert book.best_bid == (Decimal("50000.00"), Decimal("1.5"))
        
        # Check order of all bids
        bid_prices = list(book.bids.keys())
        assert bid_prices == [
            Decimal("50000.00"),
            Decimal("49999.00"),
            Decimal("49998.00")
        ]
    
    def test_snapshot_asks_sorted_ascending(self):
        """Asks should be sorted in ascending order (lowest first)."""
        book = OrderBook("BTC-USD")
        
        snapshot = {
            "type": "snapshot",
            "product_id": "BTC-USD",
            "bids": [["50000.00", "1.0"]],
            "asks": [
                ["50003.00", "2.0"],  # Highest
                ["50001.00", "1.0"],  # Lowest
                ["50002.00", "1.5"],  # Middle
            ]
        }
        
        book.apply_snapshot(snapshot)
        
        # Best ask should be the lowest price
        assert book.best_ask == (Decimal("50001.00"), Decimal("1.0"))
        
        # Check order of all asks
        ask_prices = list(book.asks.keys())
        assert ask_prices == [
            Decimal("50001.00"),
            Decimal("50002.00"),
            Decimal("50003.00")
        ]
    
    def test_snapshot_ignores_zero_sizes(self):
        """Snapshot should ignore entries with size=0."""
        book = OrderBook("BTC-USD")
        
        snapshot = {
            "type": "snapshot",
            "product_id": "BTC-USD",
            "bids": [
                ["50000.00", "1.5"],
                ["49999.00", "0"],  # Should be ignored
            ],
            "asks": [
                ["50001.00", "0"],  # Should be ignored
                ["50002.00", "1.0"],
            ]
        }
        
        book.apply_snapshot(snapshot)
        
        assert len(book.bids) == 1
        assert len(book.asks) == 1
        assert Decimal("49999.00") not in book.bids
        assert Decimal("50001.00") not in book.asks
    
    def test_snapshot_clears_previous_state(self):
        """Applying a new snapshot should clear the previous book state."""
        book = OrderBook("BTC-USD")
        
        # First snapshot
        snapshot1 = {
            "type": "snapshot",
            "product_id": "BTC-USD",
            "bids": [["50000.00", "1.0"]],
            "asks": [["50001.00", "1.0"]]
        }
        book.apply_snapshot(snapshot1)
        
        assert len(book.bids) == 1
        assert len(book.asks) == 1
        
        # Second snapshot with different data
        snapshot2 = {
            "type": "snapshot",
            "product_id": "BTC-USD",
            "bids": [["49000.00", "2.0"], ["48999.00", "1.0"]],
            "asks": [["49001.00", "2.0"]]
        }
        book.apply_snapshot(snapshot2)
        
        # Old data should be gone
        assert Decimal("50000.00") not in book.bids
        assert Decimal("50001.00") not in book.asks
        
        # New data should be present
        assert len(book.bids) == 2
        assert len(book.asks) == 1
        assert book.best_bid == (Decimal("49000.00"), Decimal("2.0"))
    
    def test_invalid_snapshot_type(self):
        """Should raise ValueError for non-snapshot messages."""
        book = OrderBook("BTC-USD")
        
        invalid = {
            "type": "l2update",
            "bids": [],
            "asks": []
        }
        
        with pytest.raises(ValueError, match="Expected snapshot message"):
            book.apply_snapshot(invalid)


class TestL2UpdateApplication:
    """Test l2update message processing."""
    
    def test_apply_update_new_bid(self):
        """Should add a new bid price level."""
        book = OrderBook("BTC-USD")
        
        # Initialize with snapshot
        snapshot = {
            "type": "snapshot",
            "product_id": "BTC-USD",
            "bids": [["50000.00", "1.0"]],
            "asks": [["50001.00", "1.0"]]
        }
        book.apply_snapshot(snapshot)
        
        # Add a new bid level
        update = {
            "type": "l2update",
            "product_id": "BTC-USD",
            "time": "2024-01-28T12:00:00.000000Z",
            "changes": [
                ["buy", "49999.00", "2.0"]
            ]
        }
        book.apply_update(update)
        
        assert len(book.bids) == 2
        assert book.bids[Decimal("49999.00")] == Decimal("2.0")
    
    def test_apply_update_modify_existing_size(self):
        """Should update size at existing price level (absolute, not delta)."""
        book = OrderBook("BTC-USD")
        
        snapshot = {
            "type": "snapshot",
            "product_id": "BTC-USD",
            "bids": [["50000.00", "1.0"]],
            "asks": [["50001.00", "1.0"]]
        }
        book.apply_snapshot(snapshot)
        
        # Update existing bid size to 3.0 (absolute)
        update = {
            "type": "l2update",
            "product_id": "BTC-USD",
            "time": "2024-01-28T12:00:00.000000Z",
            "changes": [
                ["buy", "50000.00", "3.0"]
            ]
        }
        book.apply_update(update)
        
        # Size should be set to 3.0, not increased by 3.0
        assert book.bids[Decimal("50000.00")] == Decimal("3.0")
    
    def test_apply_update_delete_level_with_zero_size(self):
        """Size=0 should DELETE the price level (critical Coinbase protocol)."""
        book = OrderBook("BTC-USD")
        
        snapshot = {
            "type": "snapshot",
            "product_id": "BTC-USD",
            "bids": [
                ["50000.00", "1.0"],
                ["49999.00", "2.0"]
            ],
            "asks": [["50001.00", "1.0"]]
        }
        book.apply_snapshot(snapshot)
        
        # Delete the 49999 bid level
        update = {
            "type": "l2update",
            "product_id": "BTC-USD",
            "time": "2024-01-28T12:00:00.000000Z",
            "changes": [
                ["buy", "49999.00", "0"]
            ]
        }
        book.apply_update(update)
        
        # Level should be completely removed
        assert len(book.bids) == 1
        assert Decimal("49999.00") not in book.bids
        assert book.best_bid == (Decimal("50000.00"), Decimal("1.0"))
    
    def test_apply_update_multiple_changes(self):
        """Should handle multiple changes in a single update."""
        book = OrderBook("BTC-USD")
        
        snapshot = {
            "type": "snapshot",
            "product_id": "BTC-USD",
            "bids": [["50000.00", "1.0"]],
            "asks": [["50001.00", "1.0"]]
        }
        book.apply_snapshot(snapshot)
        
        # Multiple simultaneous changes
        update = {
            "type": "l2update",
            "product_id": "BTC-USD",
            "time": "2024-01-28T12:00:00.000000Z",
            "changes": [
                ["buy", "49999.00", "2.0"],   # New bid
                ["buy", "50000.00", "0"],      # Delete existing bid
                ["sell", "50002.00", "1.5"],   # New ask
                ["sell", "50001.00", "2.0"]    # Update ask
            ]
        }
        book.apply_update(update)
        
        # Check results
        assert len(book.bids) == 1
        assert Decimal("50000.00") not in book.bids
        assert book.best_bid == (Decimal("49999.00"), Decimal("2.0"))
        
        assert len(book.asks) == 2
        assert book.asks[Decimal("50001.00")] == Decimal("2.0")
        assert book.asks[Decimal("50002.00")] == Decimal("1.5")
    
    def test_apply_update_ask_side(self):
        """Should correctly handle ask-side updates."""
        book = OrderBook("BTC-USD")
        
        snapshot = {
            "type": "snapshot",
            "product_id": "BTC-USD",
            "bids": [["50000.00", "1.0"]],
            "asks": [["50001.00", "1.0"]]
        }
        book.apply_snapshot(snapshot)
        
        # Update ask side
        update = {
            "type": "l2update",
            "product_id": "BTC-USD",
            "time": "2024-01-28T12:00:00.000000Z",
            "changes": [
                ["sell", "50001.00", "5.0"],  # Update existing
                ["sell", "50002.00", "2.0"]   # Add new
            ]
        }
        book.apply_update(update)
        
        assert len(book.asks) == 2
        assert book.asks[Decimal("50001.00")] == Decimal("5.0")
        assert book.asks[Decimal("50002.00")] == Decimal("2.0")
    
    def test_update_time_tracking(self):
        """Should track the timestamp of the last update."""
        book = OrderBook("BTC-USD")
        
        snapshot = {
            "type": "snapshot",
            "product_id": "BTC-USD",
            "bids": [["50000.00", "1.0"]],
            "asks": [["50001.00", "1.0"]]
        }
        book.apply_snapshot(snapshot)
        
        update = {
            "type": "l2update",
            "product_id": "BTC-USD",
            "time": "2024-01-28T12:30:45.123456Z",
            "changes": [["buy", "49999.00", "1.0"]]
        }
        book.apply_update(update)
        
        assert book.last_update_time == "2024-01-28T12:30:45.123456Z"
    
    def test_invalid_update_type(self):
        """Should raise ValueError for non-update messages."""
        book = OrderBook("BTC-USD")
        
        invalid = {
            "type": "snapshot",
            "changes": []
        }
        
        with pytest.raises(ValueError, match="Expected l2update message"):
            book.apply_update(invalid)


class TestBookProperties:
    """Test calculated properties of the order book."""
    
    def test_best_bid_and_ask(self):
        """Should return correct best bid and ask."""
        book = OrderBook("BTC-USD")
        
        snapshot = {
            "type": "snapshot",
            "product_id": "BTC-USD",
            "bids": [
                ["50000.00", "1.5"],
                ["49999.00", "2.0"],
                ["49998.00", "0.5"]
            ],
            "asks": [
                ["50001.00", "1.0"],
                ["50002.00", "1.5"],
                ["50003.00", "2.0"]
            ]
        }
        book.apply_snapshot(snapshot)
        
        # Best bid is highest price
        assert book.best_bid == (Decimal("50000.00"), Decimal("1.5"))
        
        # Best ask is lowest price
        assert book.best_ask == (Decimal("50001.00"), Decimal("1.0"))
    
    def test_mid_price_calculation(self):
        """Mid-price should be average of best bid and ask."""
        book = OrderBook("BTC-USD")
        
        snapshot = {
            "type": "snapshot",
            "product_id": "BTC-USD",
            "bids": [["50000.00", "1.0"]],
            "asks": [["50002.00", "1.0"]]
        }
        book.apply_snapshot(snapshot)
        
        # (50000 + 50002) / 2 = 50001
        assert book.mid_price == Decimal("50001.00")
    
    def test_spread_calculation(self):
        """Spread should be best_ask - best_bid."""
        book = OrderBook("BTC-USD")
        
        snapshot = {
            "type": "snapshot",
            "product_id": "BTC-USD",
            "bids": [["50000.00", "1.0"]],
            "asks": [["50005.00", "1.0"]]
        }
        book.apply_snapshot(snapshot)
        
        # 50005 - 50000 = 5
        assert book.spread == Decimal("5.00")
    
    def test_micro_price_calculation(self):
        """Micro-price should be volume-weighted."""
        book = OrderBook("BTC-USD")
        
        snapshot = {
            "type": "snapshot",
            "product_id": "BTC-USD",
            "bids": [["50000.00", "2.0"]],  # bid_size = 2
            "asks": [["50004.00", "1.0"]]   # ask_size = 1
        }
        book.apply_snapshot(snapshot)
        
        # P_micro = (q_bid * P_ask + q_ask * P_bid) / (q_bid + q_ask)
        # P_micro = (2 * 50004 + 1 * 50000) / (2 + 1)
        # P_micro = (100008 + 50000) / 3 = 150008 / 3 = 50002.666...
        
        expected = (Decimal("2.0") * Decimal("50004.00") + 
                   Decimal("1.0") * Decimal("50000.00")) / Decimal("3.0")
        
        assert book.micro_price == expected
    
    def test_micro_price_with_equal_sizes(self):
        """Micro-price with equal sizes should equal mid-price."""
        book = OrderBook("BTC-USD")
        
        snapshot = {
            "type": "snapshot",
            "product_id": "BTC-USD",
            "bids": [["50000.00", "1.0"]],
            "asks": [["50002.00", "1.0"]]
        }
        book.apply_snapshot(snapshot)
        
        # When bid_size == ask_size, micro_price == mid_price
        assert book.micro_price == book.mid_price


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_book_is_invalid(self):
        """Empty book should not be valid."""
        book = OrderBook("BTC-USD")
        assert not book.is_valid()
    
    def test_book_with_only_bids_is_invalid(self):
        """Book with only bids should not be valid."""
        book = OrderBook("BTC-USD")
        
        snapshot = {
            "type": "snapshot",
            "product_id": "BTC-USD",
            "bids": [["50000.00", "1.0"]],
            "asks": []
        }
        book.apply_snapshot(snapshot)
        
        assert not book.is_valid()
    
    def test_book_with_only_asks_is_invalid(self):
        """Book with only asks should not be valid."""
        book = OrderBook("BTC-USD")
        
        snapshot = {
            "type": "snapshot",
            "product_id": "BTC-USD",
            "bids": [],
            "asks": [["50001.00", "1.0"]]
        }
        book.apply_snapshot(snapshot)
        
        assert not book.is_valid()
    
    def test_crossed_market_detection(self):
        """Should detect when bid >= ask (crossed market)."""
        book = OrderBook("BTC-USD")
        
        # Intentionally create a crossed market
        snapshot = {
            "type": "snapshot",
            "product_id": "BTC-USD",
            "bids": [["50002.00", "1.0"]],  # Bid higher than ask!
            "asks": [["50001.00", "1.0"]]
        }
        book.apply_snapshot(snapshot)
        
        assert not book.is_valid()
    
    def test_get_depth(self):
        """Should return top N levels from each side."""
        book = OrderBook("BTC-USD")
        
        snapshot = {
            "type": "snapshot",
            "product_id": "BTC-USD",
            "bids": [
                ["50000.00", "1.0"],
                ["49999.00", "2.0"],
                ["49998.00", "3.0"],
                ["49997.00", "4.0"]
            ],
            "asks": [
                ["50001.00", "1.0"],
                ["50002.00", "2.0"],
                ["50003.00", "3.0"]
            ]
        }
        book.apply_snapshot(snapshot)
        
        depth = book.get_depth(num_levels=2)
        
        assert len(depth["bids"]) == 2
        assert len(depth["asks"]) == 2
        
        # Check correct prices (top 2)
        assert depth["bids"][0] == ("50000.00", "1.0")
        assert depth["bids"][1] == ("49999.00", "2.0")
        assert depth["asks"][0] == ("50001.00", "1.0")
        assert depth["asks"][1] == ("50002.00", "2.0")
    
    def test_repr(self):
        """Should have a useful string representation."""
        book = OrderBook("BTC-USD")
        
        snapshot = {
            "type": "snapshot",
            "product_id": "BTC-USD",
            "bids": [["50000.00", "1.5"]],
            "asks": [["50001.00", "2.0"]]
        }
        book.apply_snapshot(snapshot)
        
        repr_str = repr(book)
        
        assert "BTC-USD" in repr_str
        assert "50000.00" in repr_str
        assert "50001.00" in repr_str
        assert "1.5" in repr_str
        assert "2.0" in repr_str
