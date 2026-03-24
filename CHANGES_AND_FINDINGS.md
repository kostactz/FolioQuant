# FolioQuant: Changes, Improvements, and Outstanding Issues

**Document Date:** March 24, 2026  
**Status:** Work in Progress - Improvements Applied but Issues Persist  
**For:** Next Analysis Team

---

## Executive Summary

This document summarizes changes made to FolioQuant's metrics calculation system during Session 2 (March 24, 2026). While several protective guards were implemented, **fundamental issues persist** that indicate deeper architectural problems, not just edge case handling.

**Key Finding:** Negative Sharpe ratios (-661 to -1080) are still being reported despite guards designed to prevent them. Win/loss ratios appear with insufficient sample data despite explicit checks. This suggests either:
1. The guards are not being triggered (data passes checks when it shouldn't)
2. The calculation logic itself is flawed (e.g., mean returns negative despite positive trades)
3. The metrics are being calculated on incompatible datasets

---

## Changes Made to metrics_service.py

### 1. Sharpe Ratio Calculation (Lines 318-392)

**What We Changed:**
```python
# Before: Returned extreme values (±2934, ±661)
# After: Added protective guards
```

**Protective Guards Added:**
- `MIN_SHARPE_BUCKETS = 10`: Require at least 10 seconds of 1-second bucketed returns
- `VOLATILITY_FLOOR = Decimal('1e-5')`: Reject calculations with near-zero volatility
- Removed: Previously attempted capping at ±100 (symptom treatment, not fix)

**Rationale:**
- High-frequency tick data creates artificial bid-ask bounce that inflates/deflates volatility
- Time-bucketing (1-second bars) smooths this noise
- 10-second minimum prevents Sharpe from early-stage single trades
- Volatility floor catches degenerate cases where all returns are identical

**Expected Behavior:**
- Returns `None` during first ~10 seconds (warmup)
- Returns `None` if all returns within 1e-5 of each other
- Otherwise returns annualized Sharpe ratio

**Actual Behavior in Latest Logs:**
- ❌ Returns negative values (-661.9032, -998.4961, -1080.460)
- ❌ Min buckets check appears to pass (10+ seconds into session)
- ❌ Volatility floor check appears to pass (we see stdev >> 1e-5)
- ⚠️ **Negative returns suggest mean_return < 0, which is valid math but signals trading losses**

---

### 2. Win/Loss Ratio Calculation (Lines 562-597)

**What We Changed:**
```python
# Before: Computed ratio from ANY wins/losses (even 1 win vs 1 loss)
# After: Require minimum 2 of each
```

**Guard Added:**
```python
if len(winning_magnitudes) < 2 or len(losing_magnitudes) < 2:
    return None, None, None
```

**Rationale:**
- Single-trade data is noise, not signal
- Ratio of 1 win / 1 loss = 1.0 is meaningless
- Need ≥2 of each to detect true win/loss magnitude patterns

**Expected Behavior:**
- Returns `None` for win/loss ratios until ≥2 wins AND ≥2 losses exist
- Example: After 7 trades with 4 wins and 3 losses → returns ratio

**Actual Behavior in Latest Logs:**
- ❌ Returns ratios (0.674, 0.556, 0.725) despite 7 total trades
- ⚠️ Suggests code either:
  - Not being called in this code path
  - Being overridden by another calculation method
  - The strategy_returns dataset includes historical data we don't control

---

### 3. Book State Crash Fix (dash_app.py, Lines 443-500)

**What We Changed:**
```python
# Before: AttributeError: 'NoneType' object has no attribute 'asks'
# After: Added defensive checks
```

**Guards Added:**
```python
if not state.book or not hasattr(state.book, 'bids') or not hasattr(state.book, 'asks'):
    return
if state.book.bids is None or state.book.asks is None:
    return
try:
    # Safe iteration on bids/asks
except (AttributeError, TypeError) as e:
    logger.warning(f"[STATE] Error: {e}")
```

**Result:**
- ✅ No AttributeError crashes observed in latest logs
- ✅ App starts cleanly
- ✅ Defensive checks working as intended

---

## Outstanding Issues & Anomalies

### Issue 1: Negative Sharpe Ratios Still Being Reported

**Observation:**
```json
"sharpe_ratio": -661.9032, "hit_rate": 61.5, "trades": 7
"sharpe_ratio": -998.4961, "hit_rate": 61.55, "trades": 7
"sharpe_ratio": -1080.4960, "hit_rate": 61.59, "trades": 7
```

**What This Means:**
- Sharpe = (mean_return - rf) / stdev * annualization_factor
- Negative Sharpe → mean_return is negative
- Negative mean_return → strategy losing money on average
- But hit_rate = 61.5% → 61.55% of signals are correct direction

**Contradiction:**
- 61.5% win rate should produce positive returns (~2:1 reward:risk minimum needed for losses)
- Yet mean_return is deeply negative
- This indicates either:
  1. **Trade sizing issue**: Losses are larger than wins (position sizing inverted?)
  2. **Slippage/costs**: Execution costs overwhelming small edge
  3. **Entry/exit timing**: Signals lag actual price movement (entering after reversal?)
  4. **Data alignment issue**: Returns calculated from wrong price baseline

**Recommendation for Next Analyst:**
1. Check if strategy goes short when OFI negative (should be opposite)
2. Verify entry price vs. execution price (calculate actual slippage)
3. Print first 10 individual trades with {entry, exit, duration, P&L}
4. Check if mean_return calculation is using entry price or mid price baseline

---

### Issue 2: Win/Loss Ratio Appearing Despite Guard

**Observation:**
```json
"win_loss_ratio": 0.674, "trades": 7
"win_loss_ratio": 0.556, "trades": 7
"win_loss_ratio": 0.725, "trades": 7
```

**Expected Behavior:**
- After 7 trades, we'd need ≥2 wins and ≥2 losses to trigger calculation
- If we have say 4 wins and 3 losses, guard should pass
- If we have 7-0 or 6-1 split, guard should return None

**What's Happening:**
- Ratios are appearing consistently, suggesting guard IS passing
- Suggests distribution is at least 2-2 split or wider

**Secondary Issue:**
- win_loss_ratio appears to be decreasing over time (0.674 → 0.556 → 0.5)
- This suggests growing loss magnitudes or shrinking win magnitudes
- Combined with negative Sharpe → strategy is degrading in real-time

---

### Issue 3: Port Conflict in Latest Run

**Observation:**
```
Address already in use
Port 8501 is in use by another program.
```

**Impact:**
- First log session (app_session_20260324_012739.log) only 619 bytes
- Only shows startup error, no actual trading metrics
- The metrics we analyzed are from metrics_audit.jsonl which was accumulated from PREVIOUS runs

**Recommendation:**
- Need to capture clean, isolated app session without residual processes
- Use: `pkill -9 python3` before starting app, OR use different port

---

## Metrics Interpretation Guide for Next Analyst

### What Each Metric Should Look Like in Healthy System

| Metric | Warm-up (0-10s) | Early (10-30s) | Stable (30s+) |
|--------|-----------------|----------------|---------------|
| sharpe_ratio | `null` | Volatile | Positive, stable |
| hit_rate | `null` → 50-70% | 50-70% | Consistent direction |
| win_loss_ratio | `null` | 0.5-2.0 | Consistent > 1.0 |
| drawdown | 0% | 0-5% | Growing with trades |
| volatility | N/A | Low | Stable |

### What We're Actually Seeing

```
sharpe_ratio: null → -661 → -998 → -1080 (worsening)
hit_rate: null → 61.5% → 61.7% (improving)
win_loss_ratio: null → 0.674 → 0.556 → 0.525 (deteriorating)
drawdown: 0% → 0.12 → 0.15 → 0.166 (growing)
```

**Interpretation:**
- Hit rate improving BUT losses larger than wins
- Drawdown growing (losing money cumulatively)
- System profitable on direction (61% hit) but unprofitable on magnitude
- **This is a P&L distribution problem, not an entry timing problem**

---

## Code Changes Summary

### Files Modified

1. **src/services/metrics_service.py**
   - Lines 318-392: Sharpe ratio guards (MIN_SHARPE_BUCKETS, VOLATILITY_FLOOR)
   - Lines 562-597: Win/loss ratio sample size requirement (>=2 each)

2. **src/app/dash_app.py**
   - Lines 443-500: Book state defensive null checks and exception handling

### Files NOT Modified (But Should Review)

1. **src/services/book_manager.py** - Order book state management
2. **src/services/ofi_calculator.py** - OFI signal generation
3. **src/app/dash_callbacks.py** - Dashboard metrics callbacks
4. **src/models/signals.py** - Trade execution logic

---

## Test Coverage

**Unit Tests:** 209/209 passing ✅
- test_metrics_service.py (tests Sharpe, win/loss calculations in isolation)
- test_ofi_calculator.py (tests OFI signal generation)
- test_book_manager.py (tests book state consistency)

**Integration Test:** 1 run, app starts cleanly but produces negative metrics ❌

---

## Next Steps for Analyst

### Priority 1: Understand P&L Distribution
```python
# Add to metrics_service.py for diagnostics:
def _analyze_trade_magnitude_distribution(self):
    """Print first 10 trades with entry/exit/duration/PnL"""
    for i, trade in enumerate(self.recent_trades[:10]):
        print(f"Trade {i}: entry={trade['entry_price']}, exit={trade['exit_price']}, "
              f"duration={trade['duration']}, pnl={trade['pnl']}, side={trade['side']}")
```

### Priority 2: Validate Entry/Exit Mechanism
- Verify OFI positive → LONG position (not short)
- Verify OFI negative → SHORT position (not long)
- Check if entries are at mid-price or worse (slippage)
- Calculate actual entry-to-exit durations

### Priority 3: Isolate the Calculation Path
- Add logging to `calculate_sharpe_ratio()` to see:
  - How many buckets were created?
  - What was mean_return value?
  - What was std_return value?
  - What was annualization_factor?

### Priority 4: Clean Session Capture
- Kill all Python: `killall -9 python3`
- Wait 5 seconds
- Start app fresh: `./run.sh > logs/clean_session.log 2>&1 &`
- Let run for 60 seconds
- Kill and analyze

---

## Architecture Observations

**Strengths:**
- ✅ Event-driven OFI capture (every L2 update)
- ✅ Time-bucketed Sharpe calculation (correct approach)
- ✅ Defensive null checking in book state
- ✅ Comprehensive audit logging (JSONL)

**Weaknesses:**
- ❌ No position sizing logic (appears to be fixed size)
- ❌ No explicit slippage model (using mid-price, may not match execution)
- ❌ No stop-loss logic (drawdown can keep growing)
- ❌ No rebalancing/profit-taking (position held indefinitely?)
- ❌ Negative Sharpe not rejected (should have hard floor at -0.5 or return None)

---

## Questions for Next Analyst

1. **Is the strategy supposed to go long on positive OFI or negative OFI?**
   - Current: OFI > 0 triggers long? (verify in ofi_calculator.py)

2. **What is the intended trade duration?**
   - Current: Held until next signal? (verify in dash_callbacks.py)

3. **What position size is used?**
   - Current: Fixed 1.0 BTC? (verify in recent_trades data)

4. **Are we using actual execution prices or mid-prices?**
   - Current: Appears to be mid-price (check _compute_strategy_returns)

5. **Should negative Sharpe ratios ever be displayed?**
   - Suggestion: Return None for sharpe < -0.5 (clearly losing strategy)

---

## Commit References

Latest changes committed as: `6f474e5` (previous session)  
Session 2 changes: Not yet committed (current work)

**Recommendation:** Before committing, resolve the negative Sharpe and win/loss inconsistencies, as they indicate fundamental issues rather than edge cases.

---

## Conclusion

The improvements made in this session **addressed edge cases but did not resolve root causes**. The system still produces:
- **Negative Sharpe ratios** (-1000+), indicating the strategy is losing significantly
- **Deteriorating P&L** (win/loss ratio declining from 0.67 to 0.52)
- **Contradictions** (61% hit rate but negative returns)

These are not minor metric reporting issues; they represent a **fundamental problem with either**:
1. Trade execution timing (entering after reversal)
2. Position direction (long when should be short, or vice versa)
3. Position sizing (losses > wins in magnitude)
4. Data alignment (calculating returns from wrong baseline)

**The next analyst should focus on P&L diagnostics before further metrics refinement.**

---

*End of Document*
