#!/usr/bin/env python3
"""
Diagnostic script to analyze Sharpe ratio calculation failures.

This script examines the metrics audit log to understand:
1. Why Sharpe ratios are negative (-900+)
2. Whether returns are being calculated incorrectly
3. Whether the strategy is actually losing money
4. Whether there's a calculation error vs. a real problem
"""

import json
import statistics
import math
from decimal import Decimal
from collections import defaultdict
from datetime import datetime


def analyze_audit_log():
    """Analyze metrics_audit.jsonl for Sharpe calculation patterns."""
    
    metrics = []
    
    with open('/home/ubuntu/proj/FolioQuant/logs/metrics_audit.jsonl', 'r') as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                metrics.append(record)
            except json.JSONDecodeError:
                continue
    
    if not metrics:
        print("❌ No metrics found in audit log")
        return
    
    print("=" * 80)
    print("SHARPE RATIO FAILURE ANALYSIS")
    print("=" * 80)
    print(f"\nTotal records analyzed: {len(metrics)}")
    
    # 1. Distribution of Sharpe values
    print("\n" + "=" * 80)
    print("1. SHARPE RATIO DISTRIBUTION")
    print("=" * 80)
    
    sharpe_values = [m.get('sharpe_ratio') for m in metrics if m.get('sharpe_ratio') is not None]
    print(f"   Non-null Sharpe ratios: {len(sharpe_values)} / {len(metrics)}")
    
    if sharpe_values:
        print(f"   Min Sharpe:    {min(sharpe_values):>12.2f}")
        print(f"   Max Sharpe:    {max(sharpe_values):>12.2f}")
        print(f"   Median Sharpe: {statistics.median(sharpe_values):>12.2f}")
        print(f"   Mean Sharpe:   {statistics.mean(sharpe_values):>12.2f}")
        print(f"   Stdev Sharpe:  {statistics.stdev(sharpe_values) if len(sharpe_values) > 1 else 'N/A':>12}")
        
        # Count by range
        ranges = {
            "< -1000": 0,
            "-1000 to -500": 0,
            "-500 to -100": 0,
            "-100 to 0": 0,
            "0 to 100": 0,
            "> 100": 0,
        }
        for s in sharpe_values:
            if s < -1000: ranges["< -1000"] += 1
            elif s < -500: ranges["-1000 to -500"] += 1
            elif s < -100: ranges["-500 to -100"] += 1
            elif s < 0: ranges["-100 to 0"] += 1
            elif s < 100: ranges["0 to 100"] += 1
            else: ranges["> 100"] += 1
        
        print("\n   Distribution by range:")
        for range_key, count in ranges.items():
            pct = 100 * count / len(sharpe_values) if sharpe_values else 0
            print(f"      {range_key:>15}: {count:>4} ({pct:>5.1f}%)")
    
    # 2. Correlation with other metrics
    print("\n" + "=" * 80)
    print("2. SHARPE vs OTHER METRICS CORRELATION")
    print("=" * 80)
    
    # Records with non-null Sharpe
    sharpe_records = [m for m in metrics if m.get('sharpe_ratio') is not None]
    
    if sharpe_records:
        hit_rates = [m.get('hit_rate', 0) for m in sharpe_records]
        win_losses = [m.get('win_loss_ratio') for m in sharpe_records if m.get('win_loss_ratio') is not None]
        drawdowns = [m.get('drawdown', 0) for m in sharpe_records]
        
        print(f"\n   Sharpe with valid hit_rate: {len(hit_rates)}")
        if hit_rates:
            print(f"      Hit rate range: {min(hit_rates):.1f}% - {max(hit_rates):.1f}%")
            print(f"      Hit rate avg:   {statistics.mean(hit_rates):.1f}%")
            
            # Correlation with Sharpe
            sharpe_vals = [m['sharpe_ratio'] for m in sharpe_records]
            correlation = calculate_correlation(sharpe_vals, hit_rates)
            print(f"      Sharpe-HitRate correlation: {correlation:>+.3f}")
            print(f"         ^ Negative means: better hit rate → worse Sharpe (CONTRADICTION)")
        
        print(f"\n   Sharpe with win/loss ratio: {len(win_losses)}")
        if win_losses:
            print(f"      Win/loss range: {min(win_losses):.3f} - {max(win_losses):.3f}")
            print(f"      Win/loss avg:   {statistics.mean(win_losses):.3f}")
            
            sharpe_vals = [m['sharpe_ratio'] for m in sharpe_records if m.get('win_loss_ratio') is not None]
            correlation = calculate_correlation(sharpe_vals, win_losses)
            print(f"      Sharpe-WinLoss correlation: {correlation:>+.3f}")
            print(f"         ^ Negative means: better win/loss → worse Sharpe")
        
        print(f"\n   Sharpe with drawdown: {len(drawdowns)}")
        if drawdowns:
            print(f"      Drawdown range: {min(drawdowns):.4f} - {max(drawdowns):.4f}")
            print(f"      Avg drawdown:   {statistics.mean(drawdowns):.4f}")
            
            sharpe_vals = [m['sharpe_ratio'] for m in sharpe_records]
            correlation = calculate_correlation(sharpe_vals, drawdowns)
            print(f"      Sharpe-Drawdown correlation: {correlation:>+.3f}")
            print(f"         ^ Positive means: bigger drawdown → worse Sharpe (EXPECTED)")
    
    # 3. Timeline analysis
    print("\n" + "=" * 80)
    print("3. SHARPE RATIO TREND OVER TIME")
    print("=" * 80)
    
    time_windows = defaultdict(list)
    for i, m in enumerate(metrics):
        window = i // 50  # Group into windows of 50 records
        sharpe = m.get('sharpe_ratio')
        if sharpe is not None:
            time_windows[window].append(sharpe)
    
    print(f"\n   Analyzing {len(time_windows)} time windows (50 records each):\n")
    for window in sorted(time_windows.keys())[:10]:  # Show first 10 windows
        sharpe_list = time_windows[window]
        record_num = window * 50
        print(f"   Window {window:>2} (records {record_num:>4}-{record_num+49:>4}): "
              f"avg={statistics.mean(sharpe_list):>10.1f}, "
              f"min={min(sharpe_list):>10.1f}, "
              f"max={max(sharpe_list):>10.1f}")
    
    # 4. Key observations
    print("\n" + "=" * 80)
    print("4. KEY OBSERVATIONS")
    print("=" * 80)
    
    print("\n   ✗ PROBLEM: Sharpe ratios are extremely negative (-900 to -1000)")
    print("   ✗ PROBLEM: Most Sharpe values cluster in -900 to -980 range")
    print(f"   ✓ INFO: Hit rates are solid (~{statistics.mean(hit_rates):.1f}% when Sharpe reported)")
    print(f"   ✓ INFO: Win/loss ratios are ~0.52-0.54 (losses > wins in magnitude)")
    print(f"   ✗ PROBLEM: Drawdown stuck at 0.1566 (strategy is bleeding slowly)")
    
    # 5. Root cause analysis
    print("\n" + "=" * 80)
    print("5. ROOT CAUSE ANALYSIS")
    print("=" * 80)
    
    print("""
   Q: Why is Sharpe so negative despite 61% hit rate?
   
   Mathematical Analysis:
   - Sharpe = (mean_return - rf) / stdev × √(periods)
   - Sharpe ≈ -960 means: mean_return << 0 (strategy losing money)
   - With 61% hit rate, we'd expect positive return IF win_size > loss_size
   - But we see win/loss ≈ 0.52, meaning: losses are 2x bigger than wins
   
   This creates a scenario:
   - 61% correct direction (good prediction)
   - But 2:1 loss:win magnitude (bad execution/sizing)
   - Net result: Negative P&L despite positive directional accuracy
   
   Possible Root Causes (in priority order):
   
   1. TRADE EXECUTION TIMING ISSUE
      - Entering position AFTER price has already moved in target direction
      - Example: OFI positive, strategy goes long, but price already rallied
      - Then price reverses, loss is large
      - Why hit rate is still good: early moves are in correct direction
      - Why losses > wins: entering late, exiting at bottom
      
   2. POSITION SIZING ISSUE  
      - Strategy uses fixed position size (e.g. 1.0 BTC for all trades)
      - When volatility is high, fixed size causes larger %-losses
      - When signal accuracy is low (early), losses are larger
      - When signal improves, wins are small relative to fixed size
      
   3. SLIPPAGE AND COSTS
      - Bid-ask spread cost (0.5 bps assuming)
      - No fee model in logs, but could be 1-2 bps
      - Total ~1-2 bps per round-trip = 2-4 bps per trade
      - Over 5-10 trades, this adds up and creates negative net return
      
   4. VOLATILITY CLUSTERING
      - Strategy enters on low OFI (weak signal)
      - High-volatility periods create large drawdowns
      - Low-volatility periods create small wins
      - Net: negative Sharpe from vol clustering
    """)
    
    print("\n" + "=" * 80)
    print("6. RECOMMENDED DIAGNOSTICS")
    print("=" * 80)
    
    print("""
   To fix this, we need to:
   
   ✓ Step 1: Print individual trade details
     - For each trade: entry_price, exit_price, entry_signal, duration, P&L
     - Identify if trades are entering late or exiting early
     
   ✓ Step 2: Measure actual slippage
     - Compare entry price to mid-price at signal time
     - Quantify bid-ask spread cost vs fee cost
     
   ✓ Step 3: Analyze signal strength vs outcome
     - Group trades by OFI magnitude at entry
     - Check if weak signals have worse P&L
     
   ✓ Step 4: Test position sizing
     - Try fractional sizing based on signal strength
     - Try dynamic sizing based on volatility
     
   ✓ Step 5: Implement protective logic
     - Add hard stop-loss (e.g., -5% max loss per trade)
     - Add profit-taking (e.g., +2% target)
     - Add maximum holding time (e.g., 30 seconds)
    """)


def calculate_correlation(x: list, y: list) -> float:
    """Calculate Pearson correlation between two lists."""
    if len(x) < 2 or len(y) < 2 or len(x) != len(y):
        return 0.0
    
    x_mean = statistics.mean(x)
    y_mean = statistics.mean(y)
    
    numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(len(x)))
    
    x_stdev = statistics.stdev(x) if len(x) > 1 else 0
    y_stdev = statistics.stdev(y) if len(y) > 1 else 0
    
    denominator = x_stdev * y_stdev
    
    if denominator == 0:
        return 0.0
    
    return numerator / (denominator * len(x))


if __name__ == '__main__':
    analyze_audit_log()
