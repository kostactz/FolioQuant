"""
Unit tests for Latency Calculation Logic.

Tests the MetricsService.calculate_latency static method.
"""

import pytest
from datetime import datetime, timezone
from src.utils.time_utils import calculate_latency

class TestLatencyLogic:
    
    def test_calculate_latency_happy_path(self):
        """Test calculation with valid timestamps."""
        # Use simple fixed timestamp to avoid system clock confusion
        process_ts = 1700000000.0 # Arbitrary timestamp
        
        ingest_delay_ms = 10
        network_delay_ms = 90 # Total 100ms from exchange
        
        ingest_ts = process_ts - (ingest_delay_ms / 1000.0)
        exchange_ts = process_ts - ((ingest_delay_ms + network_delay_ms) / 1000.0)
        
        # Create UTC-aware datetime for exchange string
        # fromtimestamp takes timestamp and returns naive local or aware if tz provided
        exchange_dt = datetime.fromtimestamp(exchange_ts, tz=timezone.utc)
        
        # isoformat() with 'Z' is strictly for naive, but aware objects behave differently
        # We manually construct correct string ending in Z or +00:00
        exchange_str = exchange_dt.isoformat().replace('+00:00', 'Z')
        
        message = {
            "time": exchange_str,
            "_ingest_time": ingest_ts
        }
        
        total, network, system = calculate_latency(message, process_ts)
        
        # Verify within small margin of error (floating point)
        assert abs(total - 100.0) < 0.1
        assert abs(network - 90.0) < 0.1
        assert abs(system - 10.0) < 0.1

    def test_calculate_latency_missing_ingest_timestamp(self):
        """Test fallback when _ingest_time is missing."""
        process_ts = 1700000000.0
        exchange_ts = process_ts - 0.05 # 50ms ago
        
        exchange_dt = datetime.fromtimestamp(exchange_ts, tz=timezone.utc)
        exchange_str = exchange_dt.isoformat().replace('+00:00', 'Z')
        
        message = {
            "time": exchange_str
            # No _ingest_time
        }
        
        total, network, system = calculate_latency(message, process_ts)
        
        # Ingest defaults to Process time -> System Latency 0
        assert abs(total - 50.0) < 0.1
        assert abs(network - 50.0) < 0.1
        assert system == 0.0

    def test_calculate_latency_missing_exchange_timestamp(self):
        """Test behavior when exchange time is missing."""
        process_ts = 1700000000.0
        message = {
            "_ingest_time": process_ts - 0.01
        }
        
        total, network, system = calculate_latency(message, process_ts)
        
        # Cannot calc network or total
        assert total == 0.0
        assert network == 0.0
        # Can still calc system?
        # Logic: system = (process - ingest) * 1000
        assert abs(system - 10.0) < 0.1

    def test_calculate_latency_invalid_exchange_timestamp(self):
        """Test behavior with malformed exchange time."""
        process_ts = 1700000000.0
        message = {
            "time": "not-a-timestamp",
            "_ingest_time": process_ts - 0.01
        }
        
        total, network, system = calculate_latency(message, process_ts)
        
        assert total == 0.0
        assert network == 0.0
        assert abs(system - 10.0) < 0.1

    def test_negative_latency_clocks_skewed(self):
        """Test when ingest time is BEFORE exchange time (clock skew)."""
        process_ts = 1000.0
        ingest_ts = 999.0
        exchange_ts = 1001.0 # Future timestamp from exchange perspective
        
        # Manually construct valid ISO string for 1001.0
        # 1970-01-01 00:00:01.000 UTC... actually simpler
        exchange_dt = datetime.fromtimestamp(exchange_ts, tz=timezone.utc)
        exchange_str = exchange_dt.isoformat().replace('+00:00', 'Z')
        
        message = {
            "time": exchange_str,
            "_ingest_time": ingest_ts
        }
        
        total, network, system = calculate_latency(message, process_ts)
        
        # Total: 1000 - 1001 = -1s (-1000ms)
        # Network: 999 - 1001 = -2s (-2000ms)
        # System: 1000 - 999 = 1s (1000ms)
        
        assert total == -1000.0
        assert network == -2000.0
        assert system == 1000.0

if __name__ == "__main__":
    pytest.main([__file__])
