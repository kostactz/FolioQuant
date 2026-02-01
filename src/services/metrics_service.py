"""
Metrics Service - Application Layer

This service calculates performance metrics for the OFI signal to evaluate its
predictive power and risk-adjusted returns. It subscribes to the OFICalculator
and tracks signal performance over time.

Key Metrics Calculated:
1. Sharpe Ratio - Risk-adjusted return metric
2. Hit Rate - Directional accuracy (% of correct predictions)
3. Maximum Drawdown - Largest peak-to-trough decline
4. Win/Loss Ratio - Average winning vs losing predictions
5. Signal Persistence - How long signals remain in same direction
6. Price Correlation - Linear relationship with price changes

Mathematical Foundation:
    Sharpe Ratio: S = (E[R_p] - R_f) / σ_p
        - Annualized: S_annual = sqrt(N) * S_period
        - For crypto (24/7): N = periods_per_year
    
    Hit Rate: HR = (correct_predictions / total_predictions) * 100
    
    Max Drawdown: MDD = max((peak - trough) / peak)
    
    Win/Loss Ratio: WLR = avg(winning_predictions) / avg(losing_predictions)
"""

import logging
import math
import statistics
from collections import deque
from decimal import Decimal
from typing import Optional, List, Tuple
from datetime import datetime

from ..models.signals import OFISignal, MetricsSnapshot


# Configure logging
logger = logging.getLogger(__name__)


class MetricsService:
    """
    Performance metrics calculator for OFI signals.
    
    This service subscribes to OFI signal updates and calculates rolling
    performance metrics including Sharpe ratio, hit rate, drawdown, and
    correlation with price changes.
    
    Architecture:
        - Subscribes to OFICalculator signal updates
        - Maintains history of signals and price changes
        - Calculates metrics over rolling window
        - Provides real-time performance snapshot
    
    Attributes:
        window_size: Number of observations for rolling metrics
        risk_free_rate: Annual risk-free rate (default: 0.0 for crypto)
        periods_per_year: Trading periods per year for annualization
        signal_history: Rolling window of OFI signals
        price_history: Rolling window of mid-prices
        prediction_results: History of prediction outcomes (correct/incorrect)
        cumulative_returns: Track cumulative performance
        peak_value: Track peak for drawdown calculation
    """

    
    def __init__(
        self,
        window_size: int = 100,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 525600,  # Legacy parameter
        use_dynamic_scaling: bool = True,
        log_metrics: bool = False,
        trading_fee_bps: float = 0.0,
        signal_threshold: float = 5.0
    ):
        """
        Initialize metrics service.
        
        Args:
            window_size: Number of observations for rolling metrics (default: 100)
            risk_free_rate: Annual risk-free rate (default: 0.0 for crypto)
            periods_per_year: Trading periods per year (legacy)
            use_dynamic_scaling: whether to calculate annualization factor from data frequency
            log_metrics: Whether to log metric calculations (verbose)
            trading_fee_bps: Trading fee in basis points per trade (default: 0.0)
            signal_threshold: Min abs(OFI) to trigger a trade (default: 5.0)
        """
        if window_size <= 1:
            raise ValueError(f"window_size must be > 1, got {window_size}")
        if periods_per_year <= 0:
            raise ValueError(f"periods_per_year must be positive, got {periods_per_year}")
        
        self.window_size = window_size
        self.risk_free_rate = Decimal(str(risk_free_rate))
        self.periods_per_year = periods_per_year
        self.use_dynamic_scaling = use_dynamic_scaling
        self.log_metrics = log_metrics
        self.trading_fee_bps = Decimal(str(trading_fee_bps))
        self.signal_threshold = Decimal(str(signal_threshold))
        
        # Rolling windows for signal and price history
        self.signal_history: deque[Tuple[datetime, Decimal, Decimal, Decimal]] = deque(maxlen=window_size)
        # Tuple: (timestamp, ofi_value, mid_price, spread)
        
        # Prediction tracking
        self.prediction_results: deque[bool] = deque(maxlen=window_size)
        # True if prediction was correct, False otherwise
        
        # Performance tracking
        self.cumulative_returns: deque[Decimal] = deque(maxlen=window_size)
        self.peak_value = Decimal('1.0')  # Start at 1.0 for percentage calculations
        self.max_drawdown = Decimal('0.0')
        
        # Statistics
        self.total_predictions = 0
        self.correct_predictions = 0
        
        # Execution Tracking
        self.current_position = Decimal('0.0')
        self.recent_trades: deque[dict] = deque(maxlen=window_size)
        
        logger.info(
            f"MetricsService initialized: window_size={window_size}, "
            f"risk_free_rate={risk_free_rate}, fee_bps={trading_fee_bps}, threshold={signal_threshold}"
        )
    
    async def on_signal_update(self, signal: OFISignal) -> None:
        """
        Callback for OFI signal updates (observer pattern).
        
        This method is registered as a subscriber to OFICalculator and receives
        notifications whenever a new OFI signal is computed.
        
        Args:
            signal: OFISignal object with current OFI value and market state
        """
        # Validate signal has required data
        if signal.ofi_value is None or signal.mid_price is None:
            logger.debug("Skipping signal update: missing ofi_value or mid_price")
            return
        
        # Add to signal history
        self.signal_history.append((
            signal.timestamp,
            signal.ofi_value,
            signal.mid_price,
            signal.spread
        ))
        
        # Evaluate prediction if we have previous price
        if len(self.signal_history) >= 2:
            self._evaluate_prediction()
            self._evaluate_trade(signal)
        
        # Update cumulative returns for drawdown calculation
        if len(self.signal_history) >= 2:
            self._update_returns()
        
        if self.log_metrics and len(self.signal_history) >= self.window_size // 2:
            snapshot = self.get_metrics_snapshot()
            if snapshot.sharpe_ratio:
                logger.debug(
                    f"Metrics: Sharpe={float(snapshot.sharpe_ratio):.3f}, "
                    f"HitRate={float(snapshot.hit_rate):.1f}%"
                )
    
    def _evaluate_prediction(self) -> None:
        """
        Evaluate whether the previous OFI signal correctly predicted price direction.
        
        This compares the sign of OFI[t-1] with the sign of (price[t] - price[t-1])
        to determine if the signal correctly predicted the direction of price movement.
        
        Interpretation:
            - OFI > 0 predicts price increase (bullish)
            - OFI < 0 predicts price decrease (bearish)
            - OFI = 0 is neutral (not counted as prediction)
        """
        if len(self.signal_history) < 2:
            return
        
        # Get previous and current data
        # Note: History stores 4 items now
        prev_data = self.signal_history[-2]
        curr_data = self.signal_history[-1]
        
        prev_timestamp, prev_ofi, prev_price = prev_data[0], prev_data[1], prev_data[2]
        curr_timestamp, curr_ofi, curr_price = curr_data[0], curr_data[1], curr_data[2]
        
        # Calculate price change
        price_change = curr_price - prev_price
        
        # OFI should predict the NEXT price change, so we use prev_ofi to predict curr_price
        # Check if signs match (both positive or both negative)
        if prev_ofi != 0 and price_change != 0:  # Only evaluate active moves
            correct = (prev_ofi > 0 and price_change > 0) or (prev_ofi < 0 and price_change < 0)
            
            self.prediction_results.append(correct)
            self.total_predictions += 1
            if correct:
                self.correct_predictions += 1
    
    def _evaluate_trade(self, signal: OFISignal) -> None:
        """
        Evaluate if a trade should be executed based on the new signal.
        """
        if len(self.signal_history) < 2:
            return

        # Use previous signal for decision (simulate reality where we act on available info)
        # But here we are processing the *current* signal update. 
        # Actually, self.signal_history[-1] is the current signal.
        # We trade based on the current signal value for the *next* price move.
        
        current_ofi = signal.ofi_value
        current_price = signal.mid_price
        timestamp = signal.timestamp
        
        # Determine target position
        target_pos = self.current_position
        
        if current_ofi > self.signal_threshold:
            target_pos = Decimal('1.0')
        elif current_ofi < -self.signal_threshold:
            target_pos = Decimal('-1.0')
        
        # Check for trade
        if target_pos != self.current_position:
            side = 'buy' if target_pos > self.current_position else 'sell'
            size = abs(target_pos - self.current_position)
            
            trade = {
                'timestamp': timestamp,
                'side': side,
                'price': float(current_price),
                'size': float(size),
                'ofi': float(current_ofi)
            }
            logger.warning(f"[EXECUTION] Trade Executed: {side.upper()} {size:.2f} @ {current_price} | OFI={current_ofi:.2f}")
            self.recent_trades.append(trade)
            self.current_position = target_pos
    
    def _update_returns(self) -> None:
        """
        Update cumulative returns based on OFI signal strategy.
        
        This simulates a simple strategy where:
            - If OFI > 0: Go long (profit if price increases)
            - If OFI < 0: Go short (profit if price decreases)
            - If OFI = 0: No position (no profit/loss)
        
        The cumulative return tracks the hypothetical P&L of this strategy.
        """
        if len(self.signal_history) < 2:
            return
        
        # Get previous and current data
        prev_data = self.signal_history[-2]
        curr_data = self.signal_history[-1]
        
        prev_timestamp, prev_ofi, prev_price = prev_data[0], prev_data[1], prev_data[2]
        curr_timestamp, curr_ofi, curr_price = curr_data[0], curr_data[1], curr_data[2]
        
        # Calculate price return
        if prev_price == 0:
            return  # Avoid division by zero
        
        price_return = (curr_price - prev_price) / prev_price
        
        # Strategy: position = sign(OFI)
        # Return = position * price_return
        if prev_ofi > self.signal_threshold:
            position = Decimal('1.0')
        elif prev_ofi < -self.signal_threshold:
            position = Decimal('-1.0')
        else:
            position = Decimal('0.0')

        strategy_return = position * price_return
        
        # Update cumulative returns
        if self.cumulative_returns:
            new_cumulative = self.cumulative_returns[-1] * (Decimal('1.0') + strategy_return)
        else:
            new_cumulative = Decimal('1.0') + strategy_return
        
        self.cumulative_returns.append(new_cumulative)
        
        # Update peak and drawdown
        if new_cumulative > self.peak_value:
            self.peak_value = new_cumulative
        
        if self.peak_value > 0:
            current_drawdown = (self.peak_value - new_cumulative) / self.peak_value
            if current_drawdown > self.max_drawdown:
                self.max_drawdown = current_drawdown
    
    def calculate_sharpe_ratio(self) -> Optional[Decimal]:
        """
        Calculate rolling Sharpe ratio of the OFI strategy.
        
        Formula:
            Sharpe = (E[R_p] - R_f) / σ_p
            Annualized: S_annual = sqrt(periods_per_year) * S_period
        
        Where:
            R_p: Strategy returns
            R_f: Risk-free rate
            σ_p: Standard deviation of returns
        
        Returns:
            Annualized Sharpe ratio, or None if insufficient data
        """
        if len(self.signal_history) < 2:
            return None
        
        # 1. Calculate raw tick returns
        # Group them by 1-second buckets
        bucketed_returns: dict[int, Decimal] = {}
        
        # Track previous position for cost calculation
        # Initialize assuming neutral start
        current_pos = Decimal('0.0')
        
        for i in range(1, len(self.signal_history)):
            prev_timestamp, prev_ofi, prev_price, prev_spread = self.signal_history[i-1]
            curr_timestamp, curr_ofi, curr_price, curr_spread = self.signal_history[i]
            
            if prev_price == 0:
                continue
            
            # Price return
            price_return = (curr_price - prev_price) / prev_price
            
            # Hysteresis Logic:
            # Only change position if signal exceeds threshold
            # Otherwise, hold current position
            if prev_ofi > self.signal_threshold:
                target_pos = Decimal('1.0')
            elif prev_ofi < -self.signal_threshold:
                target_pos = Decimal('-1.0')
            else:
                target_pos = current_pos
            
            # Calculate cost return using helper
            cost_return = self._calculate_trade_cost(
                current_pos, target_pos, curr_spread, curr_price
            )
            
            # Strategy return = (position * price_return) - cost
            gross_return = target_pos * price_return
            strategy_return = gross_return - cost_return
            
            # Update state
            current_pos = target_pos
            
            # Bucket by second
            bucket_ts = int(curr_timestamp.timestamp())
            if bucket_ts not in bucketed_returns:
                bucketed_returns[bucket_ts] = Decimal('0')
            bucketed_returns[bucket_ts] += strategy_return
            
        # 2. Calculate Sharpe on buckets
        # We need at least 2 buckets to calculate std dev
        if len(bucketed_returns) < 2:
            return None
            
        returns_list = [float(r) for r in bucketed_returns.values()]
        
        try:
            mean_return = Decimal(str(statistics.mean(returns_list)))
            std_return = Decimal(str(statistics.stdev(returns_list)))
        except statistics.StatisticsError:
            return None
        
        if std_return == 0:
            return None
            
        # 3. Annualize
        # 1-second buckets -> 31,536,000 buckets per year
        seconds_per_year = 31536000
        annualization_factor = Decimal(str(math.sqrt(seconds_per_year)))
        
        # Assuming risk-free rate is roughly 0 per second
        sharpe_annual = (mean_return / std_return) * annualization_factor
        
        # Store for diagnostics
        self.last_volatility = std_return * annualization_factor
        
        return sharpe_annual

    def _calculate_trade_cost(
        self, 
        current_pos: Decimal, 
        target_pos: Decimal, 
        spread: Decimal, 
        price: Decimal
    ) -> Decimal:
        """
        Calculate total trading cost (spread + fees) as a return percentage.
        
        Args:
            current_pos: Check current position size
            target_pos: Target position size
            spread: Current bid-ask spread
            price: Current mid price
            
        Returns:
            Cost as a fraction of price (e.g. 0.0005 for 5 bps)
        """
        turnover = abs(target_pos - current_pos)
        
        if turnover == 0:
            return Decimal('0.0')
            
        # Spread cost (in price units)
        # Use current spread for immediate execution cost
        spread_cost = turnover * (spread / Decimal('2.0'))
        
        # Fee cost (in price units)
        # Fee is bps of the transaction value (approx price * turnover)
        # trading_fee_bps is e.g. 1.0.  1 bps = 0.0001
        fee_cost = (turnover * price) * (self.trading_fee_bps / Decimal('10000.0'))
        
        total_cost = spread_cost + fee_cost
        
        # Convert cost to return %
        return total_cost / price
    
    def calculate_hit_rate(self) -> Optional[Decimal]:
        """
        Calculate hit rate (directional accuracy).
        
        Formula:
            Hit Rate = (correct_predictions / total_predictions) * 100
        
        Returns:
            Hit rate percentage (0-100), or None if no predictions
        """
        if not self.prediction_results:
            return None
        
        correct = sum(1 for result in self.prediction_results if result)
        total = len(self.prediction_results)
        
        if total == 0:
            return None
        
        hit_rate = Decimal(str(correct)) / Decimal(str(total)) * Decimal('100')
        return hit_rate
    
    def calculate_max_drawdown(self) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        """
        Calculate maximum and current drawdown.
        
        Formula:
            Drawdown = (peak - current) / peak
            Max Drawdown = max(drawdown over all periods)
        
        Returns:
            Tuple of (max_drawdown, current_drawdown) as percentages, or (None, None)
        """
        if not self.cumulative_returns:
            return None, None
        
        current_value = self.cumulative_returns[-1]
        
        if self.peak_value == 0:
            return None, None
        
        current_drawdown = (self.peak_value - current_value) / self.peak_value * Decimal('100')
        max_drawdown = self.max_drawdown * Decimal('100')
        
        return max_drawdown, current_drawdown
    
    def calculate_win_loss_ratio(self) -> Tuple[Optional[Decimal], Optional[Decimal], Optional[Decimal]]:
        """
        Calculate win/loss ratio and average win/loss magnitudes.
        
        This tracks the average magnitude of winning vs losing predictions
        based on the actual price changes that occurred.
        
        Returns:
            Tuple of (win_loss_ratio, avg_win, avg_loss), or (None, None, None)
        """
        if len(self.signal_history) < 2:
            return None, None, None
        
        winning_magnitudes: List[Decimal] = []
        losing_magnitudes: List[Decimal] = []
        
        for i in range(1, len(self.signal_history)):
            prev_timestamp, prev_ofi, prev_price, _ = self.signal_history[i-1]
            curr_timestamp, curr_ofi, curr_price, _ = self.signal_history[i]
            
            if prev_price == 0 or prev_ofi == 0:
                continue
            
            # Price return
            price_return = (curr_price - prev_price) / prev_price
            
            # Strategy position
            if prev_ofi > 0:
                position = Decimal('1.0')
            elif prev_ofi < 0:
                position = Decimal('-1.0')
            else:
                continue
            
            strategy_return = position * price_return
            
            if strategy_return > 0:
                winning_magnitudes.append(abs(strategy_return))
            elif strategy_return < 0:
                losing_magnitudes.append(abs(strategy_return))
        
        if not winning_magnitudes or not losing_magnitudes:
            return None, None, None
        
        avg_win = Decimal(str(statistics.mean([float(w) for w in winning_magnitudes])))
        avg_loss = Decimal(str(statistics.mean([float(l) for l in losing_magnitudes])))
        
        if avg_loss == 0:
            return None, avg_win, avg_loss
        
        win_loss_ratio = avg_win / avg_loss
        
        return win_loss_ratio, avg_win, avg_loss
    
    def calculate_signal_persistence(self) -> Optional[Decimal]:
        """
        Calculate average signal persistence (consecutive periods in same direction).
        
        This measures how long OFI signals tend to remain positive or negative,
        which can indicate trend strength.
        
        Returns:
            Average consecutive periods in same direction, or None
        """
        if len(self.signal_history) < 2:
            return None
        
        runs: List[int] = []
        current_run = 1
        
        for i in range(1, len(self.signal_history)):
            prev_ofi = self.signal_history[i-1][1]
            curr_ofi = self.signal_history[i][1]
            
            # Check if same sign
            same_sign = (prev_ofi > 0 and curr_ofi > 0) or (prev_ofi < 0 and curr_ofi < 0)
            
            if same_sign:
                current_run += 1
            else:
                if current_run > 0:
                    runs.append(current_run)
                current_run = 1
        
        # Add final run
        if current_run > 0:
            runs.append(current_run)
        
        if not runs:
            return None
        
        avg_persistence = Decimal(str(statistics.mean(runs)))
        return avg_persistence
    
    def calculate_information_coefficient(self, lag: int = 1) -> Optional[Decimal]:
        """
        Calculate Information Coefficient (IC).
        
        IC is the correlation between the signal at time t and the return at time t+lag.
        
        Args:
            lag: Number of periods forward to correlate (default: 1)
            
        Returns:
            Correlation coefficient (-1 to 1), or None
        """
        if len(self.signal_history) < lag + 2:
            return None
        
        signal_values: List[float] = []
        future_returns: List[float] = []
        
        # We need signal[t] and return[t+lag]
        # Return[t+lag] is (Price[t+lag] - Price[t]) / Price[t] (or log returns)
        # Here we use simple price chagne for consistency with existing price_correlation
        # or better: return relative to mid price
        
        for i in range(len(self.signal_history) - lag):
            curr_data = self.signal_history[i]
            future_data = self.signal_history[i+lag]
            
            curr_timestamp, curr_ofi, curr_price = curr_data[0], curr_data[1], curr_data[2]
            future_timestamp, future_ofi, future_price = future_data[0], future_data[1], future_data[2]
            
            if curr_price == 0:
                continue
                
            # Signal at t
            signal_values.append(float(curr_ofi))
            
            # Return over next 'lag' periods
            # (P_{t+lag} - P_t) / P_t
            ret = float((future_price - curr_price) / curr_price)
            future_returns.append(ret)
        
        if len(signal_values) < 2:
            return None
        
        # Calculate correlation coefficient
        try:
            ic = Decimal(str(self._pearson_correlation(signal_values, future_returns)))
            return ic
        except (ValueError, ZeroDivisionError):
            return None

    def calculate_alpha_decay(self, max_lag: int = 10) -> dict:
        """
        Calculate Alpha Decay (IC at various lags).
        
        Returns:
            Dictionary mapping lag (int) to IC (float)
        """
        decay_profile = {}
        for lag in range(1, max_lag + 1):
            ic = self.calculate_information_coefficient(lag=lag)
            if ic is not None:
                decay_profile[lag] = float(ic)
        return decay_profile

    # Alias for backward compatibility
    calculate_price_correlation = lambda self: self.calculate_information_coefficient(lag=1)
    
    def _pearson_correlation(self, x: List[float], y: List[float]) -> float:
        """
        Calculate Pearson correlation coefficient.
        
        Formula:
            r = cov(X, Y) / (std(X) * std(Y))
        
        Args:
            x: First variable
            y: Second variable
        
        Returns:
            Correlation coefficient
        
        Raises:
            ValueError: If arrays have different lengths or insufficient data
            ZeroDivisionError: If std is zero
        """
        if len(x) != len(y) or len(x) < 2:
            raise ValueError("Arrays must have same length and at least 2 elements")
        
        n = len(x)
        mean_x = statistics.mean(x)
        mean_y = statistics.mean(y)
        
        # Calculate covariance
        cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n)) / (n - 1)
        
        # Calculate standard deviations
        std_x = statistics.stdev(x)
        std_y = statistics.stdev(y)
        
        if std_x == 0 or std_y == 0:
            raise ZeroDivisionError("Standard deviation is zero")
        
        correlation = cov / (std_x * std_y)
        return correlation
    
    def get_scatter_data(self, history_len: int = 1000) -> List[Tuple[float, float]]:
        """
        Get OFI vs Price Change data for scatter plot.
        
        Returns:
            List of (ofi, price_change_bps) tuples
        """
        data = []
        # Need at least 2 points to calculate change
        if len(self.signal_history) < 2:
            return data
            
        # We need OFI[t] and PriceChange[t+1]
        # PriceChange[t+1] = (P[t+1] - P[t]) / P[t]
        
        for i in range(len(self.signal_history) - 1):
            curr_data = self.signal_history[i]
            next_data = self.signal_history[i+1]
            
            curr_ofi = float(curr_data[1])
            curr_price = float(curr_data[2])
            next_price = float(next_data[2])
            
            if curr_price == 0:
                continue
                
            # Calculate next period return in Basis Points
            price_change_bps = ((next_price - curr_price) / curr_price) * 10000
            
            data.append((curr_ofi, price_change_bps))
            
        return data[-history_len:]
    
    def get_metrics_snapshot(self) -> MetricsSnapshot:
        """
        Get current performance metrics snapshot.
        
        This calculates all available metrics and returns them in a structured
        snapshot for monitoring and visualization.
        
        Returns:
            MetricsSnapshot with current metrics
        """
        # Calculate all metrics
        sharpe_ratio = self.calculate_sharpe_ratio()
        hit_rate = self.calculate_hit_rate()
        max_dd, current_dd = self.calculate_max_drawdown()
        win_loss, avg_win, avg_loss = self.calculate_win_loss_ratio()
        persistence = self.calculate_signal_persistence()
        ic = self.calculate_information_coefficient()
        alpha_decay = self.calculate_alpha_decay(max_lag=5)
        correlation = ic  # Alias for backward compat
        
        # OFI statistics
        if self.signal_history:
            ofi_values = [item[1] for item in self.signal_history]
            ofi_values_float = [float(ofi) for ofi in ofi_values]
            ofi_mean = Decimal(str(statistics.mean(ofi_values_float)))
            ofi_std = Decimal(str(statistics.stdev(ofi_values_float))) if len(ofi_values_float) > 1 else Decimal('0')
            ofi_min = min(ofi_values)
            ofi_max = max(ofi_values)
        else:
            ofi_mean = ofi_std = ofi_min = ofi_max = None
        
        # Create snapshot
        snapshot = MetricsSnapshot(
            timestamp=datetime.utcnow(),
            window_size=len(self.signal_history),
            sharpe_ratio=sharpe_ratio,
            hit_rate=hit_rate,
            total_predictions=len(self.prediction_results),
            correct_predictions=sum(1 for r in self.prediction_results if r),
            incorrect_predictions=sum(1 for r in self.prediction_results if not r),
            max_drawdown=max_dd,
            current_drawdown=current_dd,
            win_loss_ratio=win_loss,
            avg_winning_prediction=avg_win,
            avg_losing_prediction=avg_loss,
            signal_persistence=persistence,
            price_correlation=correlation,
            information_coefficient=ic,
            alpha_decay=alpha_decay,
            ofi_mean=ofi_mean,
            ofi_std=ofi_std,
            ofi_min=ofi_min,
            ofi_max=ofi_max,
            # Diagnostic Data
            scatter_data=self.get_scatter_data(history_len=500),
            rolling_volatility=float(std_return * annualization_factor) if 'std_return' in locals() and 'annualization_factor' in locals() else None,
            recent_trades=list(self.recent_trades)
        )
        
        return snapshot
    
    def reset(self) -> None:
        """
        Reset all metrics and history.
        
        This clears all data, useful for restarting or testing.
        """
        self.signal_history.clear()
        self.prediction_results.clear()
        self.cumulative_returns.clear()
        self.peak_value = Decimal('1.0')
        self.max_drawdown = Decimal('0.0')
        self.total_predictions = 0
        self.correct_predictions = 0
        logger.debug("MetricsService reset")
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        snapshot = self.get_metrics_snapshot()
        sharpe = float(snapshot.sharpe_ratio) if snapshot.sharpe_ratio else 0.0
        hit_rate = float(snapshot.hit_rate) if snapshot.hit_rate else 0.0
        return (
            f"MetricsService(window_size={self.window_size}, "
            f"sharpe={sharpe:.2f}, hit_rate={hit_rate:.1f}%, "
            f"observations={len(self.signal_history)})"
        )
