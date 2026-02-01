"""
OrderBook: Efficient Level 2 Order Book Implementation

This module implements a high-performance order book using SortedDict for 
maintaining price levels in sorted order. The book supports snapshot initialization
and incremental l2update messages from the Coinbase WebSocket feed.

Key Design Decisions:
- SortedDict provides O(log n) insert/delete and O(1) access to best bid/ask
- Bids sorted in descending order (highest price first)
- Asks sorted in ascending order (lowest price first)
- Size of 0 means DELETE the price level (Coinbase protocol)
- Size > 0 means SET the absolute size at that price (not a delta)

Author: FolioQuant Team
"""

from decimal import Decimal
from typing import Dict, List, Tuple, Optional
from sortedcontainers import SortedDict
import logging

logger = logging.getLogger(__name__)


class OrderBook:
    """
    Level 2 Order Book with efficient sorted price level management.
    
    The order book maintains two sorted dictionaries:
    - bids: {price: size} sorted in descending order
    - asks: {price: size} sorted in ascending order
    
    This allows O(1) access to best bid/ask and O(log n) updates.
    
    Attributes:
        bids: SortedDict of bid price levels (price -> size)
        asks: SortedDict of ask price levels (price -> size)
        product_id: Trading pair identifier (e.g., "BTC-USD")
        last_update_time: Timestamp of last update
    """
    
    def __init__(self, product_id: str = ""):
        """
        Initialize an empty order book.
        
        Args:
            product_id: The trading pair this book tracks (e.g., "BTC-USD")
        """
        # Bids: descending order (highest price first)
        # Use negative key function to reverse sort
        self.bids: SortedDict = SortedDict(lambda x: -x)
        
        # Asks: ascending order (lowest price first)
        self.asks: SortedDict = SortedDict()
        
        self.product_id: str = product_id
        self.last_update_time: Optional[str] = None
        
        logger.debug(f"Initialized OrderBook for {product_id}")
    
    def apply_snapshot(self, snapshot: Dict) -> None:
        """
        Initialize the order book from a Coinbase snapshot message.
        
        A snapshot provides the complete state of the book at a point in time.
        Format:
        {
            "type": "snapshot",
            "product_id": "BTC-USD",
            "bids": [["price", "size"], ...],
            "asks": [["price", "size"], ...]
        }
        
        Args:
            snapshot: Snapshot message dictionary from Coinbase
        
        Raises:
            ValueError: If snapshot format is invalid
        """
        if snapshot.get("type") != "snapshot":
            raise ValueError(f"Expected snapshot message, got {snapshot.get('type')}")
        
        # Clear existing state
        self.bids.clear()
        self.asks.clear()
        
        # Process bids
        bids_data = snapshot.get("bids", [])
        for price_str, size_str in bids_data:
            price = Decimal(price_str)
            size = Decimal(size_str)
            if size > 0:  # Ignore zero-size entries
                self.bids[price] = size
        
        # Process asks
        asks_data = snapshot.get("asks", [])
        for price_str, size_str in asks_data:
            price = Decimal(price_str)
            size = Decimal(size_str)
            if size > 0:  # Ignore zero-size entries
                self.asks[price] = size
        
        self.product_id = snapshot.get("product_id", self.product_id)
        
        logger.info(
            f"Applied snapshot for {self.product_id}: "
            f"{len(self.bids)} bids, {len(self.asks)} asks"
        )
    
    def apply_update(self, update: Dict) -> None:
        """
        Apply an incremental l2update message to the order book.
        
        Format:
        {
            "type": "l2update",
            "product_id": "BTC-USD",
            "time": "2024-01-28T...",
            "changes": [
                ["side", "price", "size"],
                ...
            ]
        }
        
        Critical Coinbase Protocol Detail:
        - If size == 0: DELETE the price level
        - If size > 0: SET the price level to this absolute size (not a delta)
        
        Args:
            update: L2 update message dictionary from Coinbase
        
        Raises:
            ValueError: If update format is invalid
        """
        if update.get("type") != "l2update":
            raise ValueError(f"Expected l2update message, got {update.get('type')}")
        
        changes = update.get("changes", [])
        self.last_update_time = update.get("time")
        
        for change in changes:
            if len(change) != 3:
                logger.warning(f"Invalid change format: {change}")
                continue
            
            side, price_str, size_str = change
            price = Decimal(price_str)
            size = Decimal(size_str)
            
            # Select the appropriate side of the book
            book_side = self.bids if side == "buy" else self.asks
            
            # Apply the change
            if size == 0:
                # Delete the price level
                if price in book_side:
                    del book_side[price]
                    logger.debug(f"Deleted {side} level at {price}")
            else:
                # Set to new size (absolute, not delta)
                book_side[price] = size
                logger.debug(f"Updated {side} level at {price} to size {size}")
    
    @property
    def best_bid(self) -> Optional[Tuple[Decimal, Decimal]]:
        """
        Get the best (highest) bid price and size.
        
        Returns:
            Tuple of (price, size) or None if no bids exist
        """
        if not self.bids:
            return None
        # First item in descending order is the highest price
        price = self.bids.keys()[0]
        return (price, self.bids[price])
    
    @property
    def best_ask(self) -> Optional[Tuple[Decimal, Decimal]]:
        """
        Get the best (lowest) ask price and size.
        
        Returns:
            Tuple of (price, size) or None if no asks exist
        """
        if not self.asks:
            return None
        # First item in ascending order is the lowest price
        price = self.asks.keys()[0]
        return (price, self.asks[price])
    
    @property
    def mid_price(self) -> Optional[Decimal]:
        """
        Calculate the mid-price: (best_bid + best_ask) / 2
        
        Returns:
            Mid-price as Decimal or None if either side is empty
        """
        bid = self.best_bid
        ask = self.best_ask
        
        if bid is None or ask is None:
            return None
        
        return (bid[0] + ask[0]) / 2
    
    @property
    def spread(self) -> Optional[Decimal]:
        """
        Calculate the bid-ask spread: best_ask - best_bid
        
        Returns:
            Spread as Decimal or None if either side is empty
        """
        bid = self.best_bid
        ask = self.best_ask
        
        if bid is None or ask is None:
            return None
        
        return ask[0] - bid[0]
    
    @property
    def micro_price(self) -> Optional[Decimal]:
        """
        Calculate the volume-weighted micro-price.
        
        The micro-price incorporates the relative depths at the BBO:
        P_micro = (q_bid * P_ask + q_ask * P_bid) / (q_bid + q_ask)
        
        This provides a more accurate estimate of the "true" price than
        the simple mid-price, as it weights by liquidity.
        
        Returns:
            Micro-price as Decimal or None if either side is empty
        """
        bid = self.best_bid
        ask = self.best_ask
        
        if bid is None or ask is None:
            return None
        
        bid_price, bid_size = bid
        ask_price, ask_size = ask
        
        if bid_size + ask_size == 0:
            return None
        
        numerator = bid_size * ask_price + ask_size * bid_price
        denominator = bid_size + ask_size
        
        return numerator / denominator
    
    def get_depth(self, num_levels: int = 10) -> Dict[str, List[Tuple[str, str]]]:
        """
        Get the top N levels of the order book.
        
        Args:
            num_levels: Number of price levels to retrieve from each side
        
        Returns:
            Dictionary with 'bids' and 'asks' arrays of [price, size] tuples
        """
        bids_list = [
            (str(price), str(size))
            for price, size in list(self.bids.items())[:num_levels]
        ]
        
        asks_list = [
            (str(price), str(size))
            for price, size in list(self.asks.items())[:num_levels]
        ]
        
        return {
            "bids": bids_list,
            "asks": asks_list
        }
    
    def is_valid(self) -> bool:
        """
        Check if the order book is in a valid state.
        
        A valid book has:
        - At least one bid and one ask
        - Best bid < best ask (no crossed market)
        
        Returns:
            True if book is valid, False otherwise
        """
        if not self.bids or not self.asks:
            return False
        
        bid = self.best_bid
        ask = self.best_ask
        
        if bid is None or ask is None:
            return False
        
        # Check for crossed market
        if bid[0] >= ask[0]:
            logger.warning(
                f"Crossed market detected: bid={bid[0]}, ask={ask[0]}"
            )
            return False
        
        return True
    
    def __repr__(self) -> str:
        """String representation of the order book."""
        bid = self.best_bid
        ask = self.best_ask
        
        bid_str = f"{bid[0]} x {bid[1]}" if bid else "None"
        ask_str = f"{ask[0]} x {ask[1]}" if ask else "None"
        
        return (
            f"OrderBook({self.product_id}, "
            f"bid={bid_str}, ask={ask_str}, "
            f"levels={len(self.bids)}x{len(self.asks)})"
        )
