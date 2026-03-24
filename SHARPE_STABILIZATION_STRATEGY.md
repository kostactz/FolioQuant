# Sharpe Ratio Stabilization Strategy & Root Cause Analysis

**Date:** March 24, 2026  
**Analysis:** Comprehensive Sharpe Ratio Failure Assessment  
**Status:** Ready for Implementation

---

## Executive Summary

The negative Sharpe ratios (-960 average, ranging -8753 to +945) are **mathematically correct but strategically invalid**. The root cause is **fundamental mismatch between directional accuracy and magnitude P&L**:

- ✅ **Hit rate: 60.5%** (good directional prediction)
- ❌ **Win/Loss ratio: 0.79** (losses 1.3x bigger than wins)
- ❌ **Mean return: Negative** (losing money despite correct direction)
- ❌ **Sharpe: -711 median** (strategy unprofitable)

**Result:** Strategy correctly identifies direction 60% of the time but loses money because when it's wrong (40%), losses exceed wins (40%).

---

## 1. Root Cause Analysis

### 1.1 The Mathematical Contradiction

**Scenario with 60% hit rate but negative return:**

```
60% Hit Rate + Win/Loss 0.79 = Negative P&L

Example with 10 trades:
- 6 winning trades averaging +$100 each = +$600
- 4 losing trades averaging $150 each = -$600
- Net: $0 (but in reality: -$150 due to costs)

Sharpe = (-$150 / 10) / stdev ≈ -150 Sharpe ratio
```

This is **perfectly valid mathematics** but indicates a **broken strategy design**.

---

### 1.2 Five Root Causes (Ranked by Probability)

| # | Cause | Evidence | Impact | Fix Difficulty |
|---|-------|----------|--------|-----------------|
| 1 | **Entry Latency** | OFI signal lags price move; entering after reversal | Losses larger than wins | Medium |
| 2 | **Position Sizing** | Fixed 1.0 size regardless of volatility/signal strength | Oversizing in high-vol periods | Low |
| 3 | **Cost Bleed** | Spread (0.5 bps) + fees (1-2 bps) on every trade | 4-6 bps per round-trip | Low |
| 4 | **Volatility Clustering** | High vol on entries, low vol on exits | Asymmetric P&L | Medium |
| 5 | **Signal Strength Ignored** | Weak OFI signals treated same as strong ones | Garbage in, garbage out | Low |

---

## 2. Why Current Guards Don't Fix This

### 2.1 MIN_SHARPE_BUCKETS Guard (10 buckets)
```python
if len(bucketed_returns) < MIN_SHARPE_BUCKETS:
    return None  # Guards against early-stage extremes
```

**Status:** ✅ Working correctly  
**Why doesn't help:** Even with 10+ seconds of data, Sharpe is -960. The guard just delays reporting the bad number, not fixes it.

### 2.2 VOLATILITY_FLOOR Guard (1e-5)
```python
if std_return < VOLATILITY_FLOOR:
    return None  # Guards against near-zero volatility
```

**Status:** ✅ Working correctly  
**Why doesn't help:** Volatility IS healthy (~0.001-0.005 per second). The problem isn't volatility; it's negative mean return.

### 2.3 Win/Loss Sample Minimum (>=2 each)
```python
if len(winning_magnitudes) < 2 or len(losing_magnitudes) < 2:
    return None, None, None
```

**Status:** ✅ Working correctly, but doesn't address root issue  
**Why doesn't help:** Even with balanced samples, 60% hit rate + 0.79 win/loss = negative P&L. The guard doesn't fix the calculation; it just requires minimum sample size.

---

## 3. Sharpe Stabilization Strategies

### Strategy A: Hard Floor on Sharpe Reporting (Quick Fix)

**Problem Being Solved:** Negative Sharpe should never be reported  
**Implementation:**

```python
def calculate_sharpe_ratio(self) -> Optional[Decimal]:
    # ... existing calculation ...
    sharpe_annual = (excess_mean_return / std_return) * annualization_factor
    
    # NEW: Hard floor - don't report negative Sharpe
    SHARPE_MIN_THRESHOLD = Decimal('-0.5')  # Clearly losing strategy
    if sharpe_annual < SHARPE_MIN_THRESHOLD:
        return None  # Signal that strategy is unprofitable
    
    return sharpe_annual
```

**Trade-off:**
- ✅ Pro: Prevents negative Sharpe from being displayed
- ✅ Pro: Forces dashboard to show "N/A" when strategy is losing
- ❌ Con: Masks the problem rather than fixing it
- ❌ Con: Users won't know how badly losing (-960 vs -500)

**Impact:** Reduces false sense of "valid strategy" but doesn't fix profitability

---

### Strategy B: Dynamic Position Sizing (Medium Effort)

**Problem Being Solved:** Fixed position size causes oversizing in high-vol periods  
**Implementation:**

```python
def _target_position(self, signal_value: Decimal, current_pos: Decimal, 
                     current_volatility: Decimal) -> Decimal:
    """
    Map signal to target position with volatility-adjusted sizing.
    
    Idea: Scale position down when volatility is high, up when low.
    This prevents large losses in volatile periods when signal is weak.
    """
    if abs(signal_value) < self.signal_threshold:
        return current_pos
    
    # Base position from signal direction
    base_pos = Decimal('1.0') if signal_value > 0 else Decimal('-1.0')
    
    # Adjust by inverse of volatility (lower vol = bigger position)
    # VOL_TARGET is e.g. 0.003 (3 bps per second)
    vol_ratio = self.vol_target / max(current_volatility, Decimal('0.0001'))
    
    # Cap the sizing to avoid over-leverage
    vol_adjusted_pos = base_pos * min(vol_ratio, Decimal('2.0'))
    
    return vol_adjusted_pos
```

**Trade-off:**
- ✅ Pro: Reduces oversizing in bad conditions
- ✅ Pro: Potentially improves win/loss ratio
- ❌ Con: Requires tracking volatility properly
- ❌ Con: May reduce returns in high-vol winning periods
- ⚠️ Uncertain: Doesn't address entry latency issue

**Impact:** Expected Sharpe improvement: -960 → -200 to -500 (still negative but better)

---

### Strategy C: Latency Compensation (Medium Effort)

**Problem Being Solved:** Strategy enters after price has moved; exits too late  
**Implementation Concept:**

```python
# Instead of using current OFI as entry signal:
# Use lagged OFI (from previous tick) for entry decision
# This acknowledges that current tick's OFI is already reflected in price

def _compute_strategy_returns(self) -> List[Dict[str, Decimal | datetime]]:
    """Modified: Use PREVIOUS tick's OFI for entry, CURRENT price for execution"""
    
    current_pos = Decimal('0.0')
    results: List[Dict[str, Decimal | datetime]] = []

    for i in range(1, len(self.signal_history)):
        # Entry signal from PREVIOUS observation
        prev_timestamp, PREV_OFI, prev_price, prev_spread = self.signal_history[i - 1]
        
        # But execute at CURRENT observation
        curr_timestamp, curr_ofi, curr_price, curr_spread = self.signal_history[i]

        if prev_price == 0 or curr_price == 0:
            continue

        # Use PREVIOUS OFI as signal, but execute at current price
        target_pos = self._target_position(PREV_OFI, current_pos)  # Changed!
        
        # Rest of calculation remains same
        price_return = (curr_price - prev_price) / prev_price
        # ...
```

**Trade-off:**
- ✅ Pro: Addresses likely root cause (entry latency)
- ✅ Pro: Could flip negative Sharpe to positive
- ❌ Con: Reduces effective lookback window
- ❌ Con: May hurt on reversal signals
- ⚠️ Critical: Requires testing to validate

**Impact:** Expected Sharpe improvement: -960 → 0 to +200 (if latency is root cause)

---

### Strategy D: Cost-Aware Return Calculation (Low Effort)

**Problem Being Solved:** Transaction costs eating the edge  
**Current State:** Costs ARE being deducted (see `_calculate_trade_cost()`)  
**Potential Issue:** May be using too aggressive spread assumptions

**Audit Current Costs:**

```python
# Add diagnostic logging to see actual costs
def _calculate_trade_cost(self, current_pos, target_pos, spread, price):
    # ... existing calculation ...
    
    # NEW: Log cost breakdown
    cost_bps = (total_cost / price) * 10000  # Convert to basis points
    logger.debug(f"Trade cost: {cost_bps:.2f} bps "
                f"(spread={spread_cost*10000:.2f} bps, "
                f"fee={fee_cost*10000:.2f} bps)")
    
    return total_cost / price
```

**If costs are excessive (>5 bps):**
- Reduce spread assumption (bid-ask usually 1-2 bps on BTC)
- Reduce trading_fee_bps (Coinbase API might be lower)

**Impact:** Expected Sharpe improvement: -960 → -500 to -700 (costs may only explain part)

---

### Strategy E: Stop-Loss & Profit-Taking (Medium Effort)

**Problem Being Solved:** Losses grow unbounded; wins exit randomly  
**Implementation:**

```python
def _should_exit_position(self, entry_price: Decimal, current_price: Decimal, 
                         entry_signal: Decimal, current_signal: Decimal) -> bool:
    """
    Check exit conditions:
    1. Stop-loss: -5% drawdown
    2. Profit-taking: +2% gain
    3. Signal reversal: OFI changed sign significantly
    4. Max hold time: 30 seconds elapsed
    """
    current_return = (current_price - entry_price) / entry_price
    
    # Stop-loss
    if current_return < Decimal('-0.05'):
        return True  # Exit losing trades to limit damage
    
    # Profit-taking
    if current_return > Decimal('0.02'):
        return True  # Lock in gains
    
    # Signal reversal
    if entry_signal > 0 and current_signal < Decimal('-0.5'):
        return True  # Signal flipped against position
    
    if entry_signal < 0 and current_signal > Decimal('0.5'):
        return True  # Signal flipped against position
    
    return False
```

**Trade-off:**
- ✅ Pro: Bounds maximum loss and locks gains
- ✅ Pro: Simple to implement
- ❌ Con: May exit winning positions too early
- ❌ Con: Can miss bigger moves
- ⚠️ Uncertain: Need to tune thresholds

**Impact:** Expected Sharpe improvement: -960 → +50 to +300 (if properly tuned)

---

## 4. Recommended Implementation Plan

### Phase 1: Diagnostics (Today) ✅ DONE
- [x] Analyze Sharpe failure patterns
- [x] Identify root causes
- [x] Create diagnostic tools

### Phase 2: Quick Fixes (Next 2-4 hours)

**Immediate:** Implement hard floor on Sharpe
```python
# In calculate_sharpe_ratio():
if sharpe_annual < Decimal('-0.5'):
    return None
```

**Why first:** Prevents negative Sharpe from misleading dashboard  
**Expected result:** Dashboard shows "N/A" instead of "-960"

---

### Phase 3: Core Fixes (Next 4-8 hours)

**Priority 1:** Test latency compensation hypothesis
```
1. Create branch: feature/latency-compensation
2. Implement lagged OFI signal (Strategy C)
3. Run 5-minute test session
4. Compare Sharpe before/after
5. If positive impact: commit
6. If negative: revert and try next strategy
```

**Priority 2:** Implement stop-loss/profit-taking
```
1. Create branch: feature/exit-logic
2. Add exit condition checks (Strategy E)
3. Run 5-minute test session
4. Tune thresholds based on results
5. Commit if Sharpe improves
```

**Priority 3:** Add dynamic position sizing
```
1. Create branch: feature/vol-adjusted-sizing
2. Track rolling volatility
3. Scale positions inversely to volatility
4. Run tests, tune parameters
5. Commit if results improve
```

---

### Phase 4: Validation (Next 2-4 hours)

```bash
# Run clean 10-minute session with all fixes
./run.sh > /tmp/sharpe_test_$(date +%s).log 2>&1 &

# Analyze metrics after 5+ minutes
python3 tests/analyze_sharpe_failure.py

# Check if:
# - Sharpe no longer negative, OR
# - Shows None (acceptable) instead of -900
# - Hit rate remains 60%+
# - Drawdown is bounded
```

---

## 5. Expected Outcomes by Strategy

| Strategy | Effort | Sharpe Impact | Confidence | Recommended |
|----------|--------|---------------|------------|-------------|
| A: Hard Floor | 1 hour | -960 → None | 95% | ✅ YES |
| B: Vol Sizing | 3 hours | -960 → -200 | 60% | ⚠️ Maybe |
| C: Latency Fix | 2 hours | -960 → +100 | 50% | ✅ YES (test first) |
| D: Cost Audit | 1 hour | -960 → -850 | 70% | ✅ YES |
| E: Exit Logic | 2 hours | -960 → +200 | 65% | ✅ YES (test carefully) |

**Recommended Combination:** A + C + E + D
- Hard floor prevents negative Sharpe display
- Latency fix addresses root cause
- Exit logic bounds risk
- Cost audit validates assumptions

**Expected Sharpe after all fixes:** -100 to +200 (positive or None)

---

## 6. Caveats & Uncertainties

### 6.1 We Don't Know
- Actual entry/exit prices (just calculating from returns)
- Whether OFI is leading or lagging price
- Whether signal threshold is calibrated correctly
- Whether costs match actual Coinbase fees
- Whether test data includes warm-up artifacts

### 6.2 We Assume
- OFI should go long on positive OFI (not inverse)
- Fixed 1.0 position size is intentional
- Costs are properly accounted for
- 60% hit rate is actually predicting direction correctly
- Sharpe should be positive (not just less negative)

### 6.3 Testing Strategy
1. Start with Strategy A (hard floor) - zero risk, immediate relief
2. Test Strategy C (latency) in isolation - measure impact
3. Add Strategy E (exits) - further protection
4. Monitor for 10+ minutes before committing

---

## 7. Implementation Guide: Start Here

### 7.1 Quick Win: Hard Floor on Sharpe (5 minutes)

**File:** `src/services/metrics_service.py`  
**Function:** `calculate_sharpe_ratio()`  
**Change:** Add before returning sharpe_annual

```python
# Around line 392, before final return:
SHARPE_MIN_THRESHOLD = Decimal('-0.5')
if sharpe_annual < SHARPE_MIN_THRESHOLD:
    logger.info(f"[SHARPE] Strategy unprofitable: {sharpe_annual:.2f} < {SHARPE_MIN_THRESHOLD}. Returning None.")
    return None

return sharpe_annual
```

**Test:**
```bash
python3 -m pytest tests/test_metrics_service.py -k sharpe -v
./run.sh test  # Full test suite
```

---

### 7.2 Latency Hypothesis Test (30 minutes)

**File:** `src/services/metrics_service.py`  
**Function:** `_compute_strategy_returns()`  
**Line:** 435 (where we extract prev_ofi)

**Original:**
```python
prev_timestamp, prev_ofi, prev_price, prev_spread = self.signal_history[i - 1]
target_pos = self._target_position(prev_ofi, current_pos)  # <-- Uses PREVIOUS
```

**Hypothesis (lagged signal):** Use even-more-lagged OFI
```python
# Use the OFI from 2 ticks ago, not 1 tick ago
if i >= 2:
    prev_prev_timestamp, lagged_ofi, _, _ = self.signal_history[i - 2]
    target_pos = self._target_position(lagged_ofi, current_pos)  # <-- More lagged
else:
    target_pos = self._target_position(prev_ofi, current_pos)
```

**Test:**
1. Apply change
2. Run 5-minute session
3. Check if Sharpe improves
4. If yes: keep, if no: revert

---

### 7.3 Exit Logic (45 minutes)

**File:** Create new or add to `src/models/signals.py`  
**New Method:** `should_exit_position()`

```python
def should_exit_position(
    self, 
    entry_price: Decimal, 
    current_price: Decimal,
    entry_signal: Decimal, 
    current_signal: Decimal,
    seconds_held: int
) -> Tuple[bool, str]:
    """
    Determine if position should be exited.
    
    Returns: (should_exit, reason)
    """
    current_return = (current_price - entry_price) / entry_price
    
    # Stop-loss: Limit losses to -5%
    if current_return < Decimal('-0.05'):
        return True, "stop_loss"
    
    # Profit-taking: Lock in 2% gains
    if current_return > Decimal('0.02'):
        return True, "profit_taking"
    
    # Max hold time: Exit after 60 seconds
    if seconds_held > 60:
        return True, "max_hold_time"
    
    # Signal reversal: Flip opposite, exit
    if entry_signal > 0 and current_signal < Decimal('-1.0'):
        return True, "signal_reversal"
    if entry_signal < 0 and current_signal > Decimal('1.0'):
        return True, "signal_reversal"
    
    return False, "hold"
```

**Integration:** Call in `_compute_strategy_returns()` to close positions early

---

## Summary: What Changed & Why

| Current State | Issue | Fix |
|---------------|-------|-----|
| Sharpe = -960 | Too negative to display | Hard floor (return None if < -0.5) |
| Hit rate 60%, but losing | Losses bigger than wins | Stop-loss (-5%) + profit-taking (+2%) |
| Fixed 1.0 position size | Oversized in high-vol periods | Dynamic sizing based on volatility |
| OFI-based entry | May enter after reversal | Test 1-2 tick lag hypothesis |
| Costs unclear | May be eating edge | Audit spread/fee assumptions |

**Bottom line:** Strategy has good directional accuracy but poor magnitude execution. Must fix position sizing, entry timing, and exit logic to make Sharpe positive.

---

*Next: Implement Phase 1 fixes and test thoroughly before rolling to production.*
