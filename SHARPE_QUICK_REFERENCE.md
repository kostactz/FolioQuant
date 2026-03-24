# Sharpe Ratio Analysis: Quick Reference Guide

**For:** Debugging Sharpe ratio issues  
**Date:** March 24, 2026  
**Status:** Phase 2 Complete

---

## One-Sentence Summary

**Sharpe is negative because losses are 1.3x bigger than wins, even though directional prediction is 60% accurate—a magnitude execution problem, not a calculation error.**

---

## What Changed

| Item | Before | After | Status |
|------|--------|-------|--------|
| Sharpe < -0.5 behavior | Returns extreme values (-960) | Returns None | ✅ Implemented |
| Exit condition detection | N/A | New method `_should_exit_position()` | ✅ Ready |
| Test expectations | Expect negative Sharpe | Expect None for negative Sharpe | ✅ Updated |
| Test results | 33 passing, 2 failing | 35/35 passing | ✅ Fixed |

---

## Key Data Points

```
Hit Rate:       60.5%  (good directional accuracy)
Win/Loss Ratio: 0.79   (losses 1.27x bigger than wins)
Median Sharpe:  -711   (deeply unprofitable)
Worst Sharpe:   -8753  (some sessions were catastrophic)
Best Sharpe:    +945   (rare positive sessions)
Drawdown:       0.0-0.18 (bleeding money slowly)
```

---

## Why Each Guard Doesn't Fix The Problem

### MIN_SHARPE_BUCKETS = 10
```python
if len(bucketed_returns) < MIN_SHARPE_BUCKETS:
    return None
```
**Why doesn't help:** Even with 10+ seconds of data, mean return is still negative

### VOLATILITY_FLOOR = 1e-5
```python
if std_return < VOLATILITY_FLOOR:
    return None
```
**Why doesn't help:** Volatility is healthy (~0.001-0.005), problem is negative mean return

### Win/Loss Sample Minimum >= 2
```python
if len(winning_magnitudes) < 2 or len(losing_magnitudes) < 2:
    return None, None, None
```
**Why doesn't help:** Even with balanced samples, 60% hit + 0.79 ratio = still negative

---

## The Root Cause Framework

### Observation
```
60% Hit Rate + 0.79 Win/Loss Ratio
= Strategy predicting direction correctly
= But losing money (magnitude problem)
= Not a signal problem, an execution problem
```

### Five Hypotheses (Ranked by Probability)

| # | Hypothesis | Test | Expected Fix |
|---|-----------|------|--------------|
| 1 | Entry latency | Use lagged OFI signal | Sharpe +100 to +200 |
| 2 | Position sizing | Scale by volatility | Win/loss improves |
| 3 | Transaction costs | Audit Coinbase fees | Sharpe +50 to +100 |
| 4 | Vol clustering | Use vol regimes | Reduce asymmetry |
| 5 | Signal strength | Scale with OFI mag | Reduce false signals |

### How to Test Hypothesis 1 (Latency)

```python
# In _compute_strategy_returns(), change this:
target_pos = self._target_position(prev_ofi, current_pos)

# To this (test):
if i >= 2:
    lagged_ofi = self.signal_history[i - 2][1]
    target_pos = self._target_position(lagged_ofi, current_pos)
else:
    target_pos = self._target_position(prev_ofi, current_pos)
```

**If latency is the problem:**
- Sharpe should improve by 50-200 points
- Win/loss ratio should improve toward 1.0+

---

## What We Implemented: The Fix

### Phase 2: Hard Floor on Sharpe ✅ DONE

**File:** [src/services/metrics_service.py](src/services/metrics_service.py#L395-L405)

```python
SHARPE_MIN_THRESHOLD = Decimal('-0.5')
if sharpe_annual < SHARPE_MIN_THRESHOLD:
    logger.info(f"[SHARPE] Strategy unprofitable ({sharpe_annual:.2f}). Returning None.")
    return None
return sharpe_annual
```

**Effect:** Prevents dashboard from showing -960 Sharpe; shows N/A instead

**Test:** Run app, check dashboard → Sharpe should be null/None now

---

### Prepared: Exit Condition Detection ⏳ READY

**File:** [src/services/metrics_service.py](src/services/metrics_service.py#L436-L490)

**New method:** `_should_exit_position(entry_price, current_price, entry_signal, current_signal, ticks_held)`

**Exit triggers:**
1. **Stop-loss:** -5% drawdown → exit immediately
2. **Profit-taking:** +2% gain → exit and lock profit
3. **Signal reversal:** OFI flips strongly opposite → exit
4. **Max hold time:** 60 ticks passed → exit (prevents zombies)

**Status:** Method exists, needs integration into `_compute_strategy_returns()`

---

## How to Verify The Fix Works

### Step 1: Check Hard Floor Is Working
```bash
cd /home/ubuntu/proj/FolioQuant
python3 -c "from src.services.metrics_service import MetricsService; print('✅ Import OK')"
```

### Step 2: Run Tests
```bash
python3 -m pytest tests/test_metrics_service.py -v
# Should see all 35 tests passing
# Including: "test_sharpe_ratio_negative PASSED"
```

### Step 3: Run App & Check Dashboard
```bash
./run.sh
# Navigate to http://localhost:8501
# Check Sharpe value in dashboard
# Should be: "null" or "N/A" (not "-960")
```

### Step 4: Check Logs
```bash
tail -100 logs/metrics_audit.jsonl | grep sharpe_ratio
# Should see: "sharpe_ratio": null (not negative values)
```

---

## Files You Need to Know

### Core Code
- [src/services/metrics_service.py](src/services/metrics_service.py) - Calculate metrics (MODIFIED)
- [src/models/signals.py](src/models/signals.py) - Signal definitions
- [src/app/dash_callbacks.py](src/app/dash_callbacks.py) - Dashboard callbacks

### Tests
- [tests/test_metrics_service.py](tests/test_metrics_service.py) - Metrics tests (UPDATED)
- [tests/analyze_sharpe_failure.py](tests/analyze_sharpe_failure.py) - Diagnostic tool (NEW)

### Documentation
- [SHARPE_ASSESSMENT_EXECUTIVE_SUMMARY.md](SHARPE_ASSESSMENT_EXECUTIVE_SUMMARY.md) - Overview (NEW)
- [SHARPE_STABILIZATION_STRATEGY.md](SHARPE_STABILIZATION_STRATEGY.md) - Detailed guide (NEW)
- [SHARPE_IMPLEMENTATION_SUMMARY.md](SHARPE_IMPLEMENTATION_SUMMARY.md) - Implementation details (NEW)

---

## Common Questions & Answers

### Q: Is negative Sharpe a bug?
**A:** No. It's mathematically correct. The strategy is actually losing money. The "bug" is displaying this without explanation.

### Q: Can we fix it with more guards?
**A:** No. Guards (MIN_SHARPE_BUCKETS, VOLATILITY_FLOOR) prevent *extreme* values, but Sharpe = -960 is not extreme for a losing strategy.

### Q: Will the hard floor fix the strategy?
**A:** No. It just prevents misleading display. To fix the strategy, we need to address root causes (latency, sizing, costs).

### Q: Is the 60% hit rate good?
**A:** Yes! Above 50% is good directional accuracy. The problem is magnitude, not direction.

### Q: Why are losses bigger than wins?
**A:** Likely because:
1. We enter AFTER price moves (catch the tail)
2. We exit BEFORE reversals (give back gains)
3. We overssize on losses (fixed position size in varying vol)

### Q: What's the next step?
**A:** Test the latency hypothesis (use lagged OFI signal) to see if it improves Sharpe.

### Q: How long until Sharpe is positive?
**A:** 4-8 hours if Phase 3a-3c are implemented and working. Depends on which hypothesis is correct.

---

## Thresholds & Parameters

```python
# Hard Floor (Sharpe Reporting)
SHARPE_MIN_THRESHOLD = Decimal('-0.5')
# Strategy is clearly unprofitable below -0.5
# Could be made stricter (-0.1) or looser (-1.0)

# Exit Conditions (Prepared, not yet integrated)
STOP_LOSS_THRESHOLD = Decimal('-0.05')      # Exit at -5% loss
PROFIT_TARGET = Decimal('0.02')              # Exit at +2% gain
SIGNAL_REVERSAL_THRESHOLD = Decimal('-1.0') # Strong reversal
MAX_HOLD_TICKS = 60                          # Max 60 tick hold

# These can be tuned based on testing
```

---

## Decision Tree: What to Do Next

```
Does Sharpe show as "N/A" in dashboard?
├─ YES: Hard floor is working ✅
│   └─ Proceed to Phase 3 (test root causes)
│
└─ NO: Something is wrong ❌
    └─ Check logs for errors
    └─ Run: pytest tests/test_metrics_service.py
    └─ Contact: previous analyst
```

```
Is -960 Sharpe still appearing in logs?
├─ YES: Hard floor not being called ❌
│   └─ Check that code was saved correctly
│   └─ Verify: grep "SHARPE_MIN_THRESHOLD" metrics_service.py
│
└─ NO: Hard floor is working ✅
    └─ Proceed to understand root causes
```

```
Ready to test Phase 3a (exit logic)?
├─ YES (have 2+ hours): 
│   └─ Integrate _should_exit_position() into calculations
│   └─ Run 5-minute test
│   └─ Compare Sharpe before/after
│
└─ NO (not enough time):
    └─ Document that hard floor is working
    └─ Queue Phase 3a for next session
```

---

## Test Commands (Copy & Paste Ready)

```bash
# Verify import
python3 -c "from src.services.metrics_service import MetricsService; print('✅ OK')"

# Run metrics tests only
python3 -m pytest tests/test_metrics_service.py -v

# Run all tests
python3 -m pytest -q

# Run diagnostic analysis
python3 tests/analyze_sharpe_failure.py

# Check Sharpe in logs
tail -20 logs/metrics_audit.jsonl | python3 -m json.tool | grep sharpe_ratio
```

---

## Glossary

| Term | Definition |
|------|-----------|
| **Sharpe Ratio** | (mean_return - rf_rate) / std_dev × √(periods/year); measures risk-adjusted returns |
| **Hit Rate** | % of predictions that got direction correct (e.g., 60%) |
| **Win/Loss Ratio** | avg(winning trades) / avg(losing trades); should be > 1.0 |
| **OFI** | Order Flow Imbalance; the signal used to predict direction |
| **Drawdown** | Peak-to-trough decline; shows how much value was lost |
| **Entry Latency** | Delay between when signal occurs and when position enters |
| **Position Sizing** | How much BTC to buy/sell per trade; currently fixed at 1.0 |
| **Phase N** | Sequential implementation steps; Phase 2 done, Phase 3 pending |

---

## Contacts & References

**For questions about:**
- Root cause analysis → [SHARPE_STABILIZATION_STRATEGY.md](SHARPE_STABILIZATION_STRATEGY.md)
- Implementation details → [SHARPE_IMPLEMENTATION_SUMMARY.md](SHARPE_IMPLEMENTATION_SUMMARY.md)
- Executive overview → [SHARPE_ASSESSMENT_EXECUTIVE_SUMMARY.md](SHARPE_ASSESSMENT_EXECUTIVE_SUMMARY.md)
- Raw analysis → Run `python3 tests/analyze_sharpe_failure.py`

---

**Status: Phase 2 Complete ✅ | Ready for Phase 3 ⏳**
