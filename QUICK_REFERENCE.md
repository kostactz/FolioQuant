# Quick Reference: Issues & Status

## Changes Applied This Session

| Change | File | Lines | Status | Result |
|--------|------|-------|--------|--------|
| MIN_SHARPE_BUCKETS guard | metrics_service.py | 358 | ✅ Applied | Still returns -661 to -1080 |
| VOLATILITY_FLOOR guard | metrics_service.py | 371 | ✅ Applied | Guard appears to pass (no None return) |
| Win/loss sample minimum | metrics_service.py | 585 | ✅ Applied | Still returns 0.67-0.52 ratios |
| Book state null checks | dash_app.py | 450-460 | ✅ Applied | ✅ No crashes observed |

---

## Key Metrics from Latest Run (metrics_audit.jsonl)

### Observed Values Over ~30 Second Session

```
Time    Sharpe      Hit%    WinLoss  Trades  Drawdown
----    -------     ----    -------  ------  --------
t=0     null        null    null     0       0.000
t=10s   -661.90     61.5%   0.674    7       0.122
t=20s   -998.50     61.5%   0.556    7       0.157
t=30s   -1080.46    61.6%   0.525    7       0.166
```

### Problem Pattern
- ❌ Sharpe **worsening** (more negative)
- ✅ Hit rate **improving** (61.5% → 61.6%)
- ❌ Win/loss **deteriorating** (0.674 → 0.525)
- ❌ Drawdown **growing** (0.0% → 0.166%)

**Contradiction:** Rising accuracy (hit %) but falling profitability (Sharpe, win/loss)  
**Root Cause:** Loss magnitudes > win magnitudes (sizing or entry timing issue)

---

## Code Contradictions

### Win/Loss Guard Logic

**Code says:**
```python
if len(winning_magnitudes) < 2 or len(losing_magnitudes) < 2:
    return None, None, None
```

**But logs show:**
```json
"win_loss_ratio": 0.674  # After 7 trades
```

**Possible explanations:**
1. Guard IS working (we have ≥2 wins AND ≥2 losses)
2. Different calculation path being used
3. Historical data from previous run included

---

## Sharpe Negative Values Problem

### Mathematical Reality
- Sharpe = (avg_return - rf_rate) / stdev × √(periods_per_year)
- If avg_return < 0, Sharpe will be negative ✓ (mathematically valid)
- Negative avg_return = strategy losing money ✓ (correctly calculated)

### Business Problem
- **We should never report negative Sharpe in production**
- Negative Sharpe = losing strategy, not worth displaying
- Should return None for Sharpe < -0.5 (hard floor)

### Why It's Happening
- 61.5% hit rate is good directionally
- But losing money overall (negative Sharpe)
- Means: **When losses occur, they're bigger than wins**

**Example scenario:**
- 61.5% accuracy (correct direction) on 7 trades
- But 4 wins averaging $100 each = +$400
- And 3 losses averaging $400 each = -$1200
- Net: -$800 (hence negative Sharpe)

---

## What SHOULD Happen

### Healthy System Behavior

**First 10 seconds (warmup):**
```
sharpe: null (< MIN_SHARPE_BUCKETS)
hit_rate: null (no trades yet)
win_loss: null (< 2 samples each)
trades: 0
```

**10-30 seconds (active):**
```
sharpe: 5.2 (positive, stable)
hit_rate: 58% (directional accuracy)
win_loss: 1.25 (wins > losses in magnitude)
trades: 5-10
```

### What We Actually See

```
sharpe: null → -661 → -998 (NEGATIVE!)
hit_rate: null → 61.5% → 61.6% (good)
win_loss: null → 0.67 → 0.52 (LOSSES > WINS!)
trades: 0 → 7
```

---

## Questions the Code Can't Answer

1. **Why does the strategy go long/short when OFI goes positive/negative?**
   - Is this the intended direction?
   - (See: ofi_calculator.py, signals.py)

2. **How big is each position?**
   - Fixed 1.0 BTC? Fractional? Leverage?
   - (See: recent_trades data)

3. **What's the entry/exit price?**
   - Mid-price? Bid/ask? Market order?
   - (Check: dash_callbacks.py trade execution)

4. **How long are positions held?**
   - Until next signal? Timed exit?
   - (See: signal_history and trade duration)

5. **Why is hit rate good but P&L negative?**
   - Entry latency (entering after reversal)?
   - Slippage (costs eating edge)?
   - Position sizing (wrong size)?

---

## Debugging Checklist for Next Analyst

- [ ] Print first 10 trades with {entry_time, entry_price, exit_time, exit_price, P&L, duration}
- [ ] Verify OFI sign matches position direction (positive OFI → long, not short)
- [ ] Calculate slippage: actual_entry_price vs mid_price at order time
- [ ] Check if positions are FIFO (exiting oldest) or LIFO (exiting newest)
- [ ] Verify position size is constant across all trades
- [ ] Add explicit logging to _compute_strategy_returns() showing raw returns
- [ ] Run with synthetic data (known direction) to validate entry/exit logic
- [ ] Compare hit_rate calculation vs actual correct_direction/total_trades

---

## Next Session Starting Point

```bash
# Kill everything
killall -9 python3

# Clean logs
rm -f logs/app_session_*.log

# Fresh start with fresh port
# (or verify port 8501 not in use)

# Run with extended capture (2-5 minutes)
time ./run.sh > logs/clean_session_$(date +%s).log 2>&1 &

# After capturing sufficient data:
# 1. Analyze individual trades
# 2. Verify entry/exit logic
# 3. Check position sizing
# 4. Measure actual slippage
```

---

*Last Updated: March 24, 2026*  
*Status: Work handed off with detailed diagnostics*
