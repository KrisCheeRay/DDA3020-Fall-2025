import polars as pl
import numpy as np
from pathlib import Path

def evaluate_predictions():
    print("="*80)
    print("EVALUATING PREDICTIONS AGAINST TRUE FORWARD RETURNS")
    print("="*80)
    
    # Load predictions
    pred_path = Path("submission.parquet")
    if not pred_path.exists():
        print("Error: submission.parquet not found. Run run_local_inference.py first.")
        return
    
    pred_df = pl.read_parquet(pred_path)
    print(f"\nLoaded predictions: {len(pred_df)} rows")
    print(pred_df.head())
    
    # Load test data
    test_path = Path("kaggle/input/hull-tactical-market-prediction/test.csv")
    test_df = pl.read_csv(test_path)
    
    # Extract lagged_forward_returns and shift to get true forward_returns
    # lagged_forward_returns at date_id=T is actually forward_returns at date_id=T-1
    # So we shift up by 1 to align
    test_df = test_df.with_columns([
        pl.col("lagged_forward_returns").shift(-1).alias("true_forward_returns")
    ])
    
    # Keep only date_id and true_forward_returns
    test_df = test_df.select(["date_id", "true_forward_returns"])
    
    # Load debug info if available
    debug_path = Path("submission_debug.parquet")
    has_debug = debug_path.exists()
    if has_debug:
        debug_df = pl.read_parquet(debug_path)
        print(f"\n✓ Debug info loaded: {len(debug_df)} rows")
    else:
        print("\n⚠ No debug info found (run run_local_inference.py to generate)")
    
    # Merge with predictions
    eval_df = pred_df.join(test_df, on="date_id", how="inner")
    
    # Merge with debug info if available
    if has_debug:
        eval_df = eval_df.join(
            debug_df.select(["date_id", "nhits_y_hat", "p8_raw", "s2_raw"]),
            on="date_id",
            how="left"
        )
    
    # Drop last row (has no future return)
    eval_df = eval_df.filter(pl.col("true_forward_returns").is_not_null())
    
    # Print detailed comparison
    if has_debug:
        print(f"\n{'date_id':<8} | {'N-HiTS':<10} | {'Policy':<10} | {'True Ret':<10} | {'PnL':<10} | {'S2':<10} | {'P8':<10}")
        print("-" * 90)
    else:
        print(f"\n{'date_id':<10} | {'Prediction':<12} | {'True Ret':<12} | {'Ret if Long':<12}")
        print("-" * 60)
    
    for row in eval_df.iter_rows(named=True):
        date_id = row["date_id"]
        pred = row["prediction"]
        true_ret = row["true_forward_returns"]
        
        # Calculate PnL if we follow the prediction
        # Prediction is position in [0, 2], 1.0 is neutral
        # Excess position = prediction - 1.0
        # PnL = excess_position * true_return
        excess_pos = pred - 1.0
        pnl = excess_pos * true_ret
        
        if has_debug and "nhits_y_hat" in row:
            nhits = row["nhits_y_hat"]
            s2 = row["s2_raw"]
            p8 = row["p8_raw"]
            print(f"{date_id:<8} | {nhits:<10.6f} | {pred:<10.6f} | {true_ret:<10.6f} | {pnl:<10.6f} | {s2:<10.6f} | {p8:<10.6f}")
        else:
            print(f"{date_id:<10} | {pred:<12.6f} | {true_ret:<12.6f} | {pnl:<12.6f}")
    
    # Calculate metrics
    predictions = eval_df["prediction"].to_numpy()
    true_returns = eval_df["true_forward_returns"].to_numpy()
    
    # Convert predictions to signals: position - 1.0 (so 1.0 = neutral, >1.0 = long, <1.0 = short)
    signals = predictions - 1.0
    
    # Calculate PnL
    pnls = signals * true_returns
    
    # Metrics
    total_pnl = pnls.sum()
    avg_pnl = pnls.mean()
    sharpe = pnls.mean() / (pnls.std() + 1e-8) * np.sqrt(252)  # Annualized
    
    # Correlation
    corr = np.corrcoef(signals, true_returns)[0, 1]
    
    # Hit rate (direction correct)
    hits = (signals * true_returns > 0).sum()
    hit_rate = hits / len(signals) * 100
    
    # N-HiTS metrics (if available)
    nhits_corr = None
    nhits_hits = None
    if has_debug and "nhits_y_hat" in eval_df.columns:
        nhits_preds = eval_df["nhits_y_hat"].to_numpy()
        nhits_corr = np.corrcoef(nhits_preds, true_returns)[0, 1]
        nhits_hits = (nhits_preds * true_returns > 0).sum()
        nhits_hit_rate = nhits_hits / len(nhits_preds) * 100
    
    print("\n" + "="*80)
    print("METRICS")
    print("="*80)
    print(f"Total PnL:       {total_pnl:.6f}")
    print(f"Average PnL:     {avg_pnl:.6f}")
    print(f"Sharpe Ratio:    {sharpe:.4f} (annualized)")
    print(f"\nPolicy Performance:")
    print(f"  Correlation:     {corr:.4f}")
    print(f"  Hit Rate:        {hit_rate:.2f}% ({hits}/{len(signals)})")
    print(f"  Signal Mean:     {signals.mean():.6f} (excess position)")
    print(f"  Signal Std:      {signals.std():.6f}")
    
    if nhits_corr is not None:
        print(f"\nN-HiTS Performance:")
        print(f"  Correlation:     {nhits_corr:.4f}")
        print(f"  Hit Rate:        {nhits_hit_rate:.2f}% ({nhits_hits}/{len(nhits_preds)})")
        print(f"  Mean Prediction: {nhits_preds.mean():.6f}")
        print(f"  Std Prediction:  {nhits_preds.std():.6f}")
    
    print(f"\nMarket:")
    print(f"  True Ret Mean:   {true_returns.mean():.6f}")
    print(f"  True Ret Std:    {true_returns.std():.6f}")
    
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    if corr > 0.3:
        print("✓ Strong positive correlation! Model has predictive power.")
    elif corr > 0.1:
        print("✓ Moderate positive correlation. Model is somewhat predictive.")
    elif corr > 0:
        print("⚠ Weak positive correlation. Model may have limited value.")
    else:
        print("✗ Negative correlation. Model is anti-predictive (consider reversing).")
    
    if hit_rate > 60:
        print("✓ High hit rate! Model is directionally accurate.")
    elif hit_rate > 50:
        print("✓ Above-chance hit rate. Model has some directional skill.")
    else:
        print("⚠ Below-chance hit rate. Model is not reliably directional.")
    
    if sharpe > 1.0:
        print("✓ Excellent Sharpe ratio! Strategy looks profitable.")
    elif sharpe > 0.5:
        print("✓ Decent Sharpe ratio. Strategy has potential.")
    elif sharpe > 0:
        print("⚠ Low Sharpe ratio. Strategy may not be robust.")
    else:
        print("✗ Negative Sharpe. Strategy loses money on average.")
    
    print("="*80)

if __name__ == "__main__":
    evaluate_predictions()

