"""
Time utility functions for FolioQuant.
"""

from typing import Tuple
from datetime import datetime

def calculate_latency(message: dict, process_timestamp: float) -> Tuple[float, float, float]:
    """
    Calculate latency components from message timestamps.
    
    Args:
        message: WebSocket message dictionary
        process_timestamp: Current timestamp (processing time)
        
    Returns:
        Tuple of (total_latency, network_latency, system_latency) in milliseconds
    """
    # 1. Exchange Time: When the event happened at Coinbase
    exchange_time_str = message.get("time")
    exchange_ts = None
    if exchange_time_str:
        try:
            # Parse ISO 8601 string safely
            exchange_ts = datetime.fromisoformat(exchange_time_str.replace('Z', '+00:00')).timestamp()
        except ValueError:
            pass
    
    # 2. Ingest Time: When we received it from the socket
    # Fallback to process_timestamp if missing (implies 0 system latency)
    ingest_ts = message.get("_ingest_time", process_timestamp)
    
    # Calculate components
    total_latency = (process_timestamp - exchange_ts) * 1000 if exchange_ts else 0.0
    network_latency = (ingest_ts - exchange_ts) * 1000 if exchange_ts else 0.0
    system_latency = (process_timestamp - ingest_ts) * 1000
    
    return total_latency, network_latency, system_latency
