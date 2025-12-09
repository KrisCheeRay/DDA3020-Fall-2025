import polars as pl
import os
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch

# Ensure we can import starter_notebook
sys.path.append(str(Path("kaggle/work").resolve()))

# Import predict function from starter_notebook
# This will also trigger artifact loading
from starter_notebook import predict, runtime, load_artifacts

def run_local_inference():
    print("="*50)
    print("Running Local Inference Simulation (Warm Start)")
    print("="*50)
    
    # Load artifacts explicitly first
    load_artifacts()
    
    # ============ WARM UP HISTORY FROM TRAIN ============
    train_path = Path("kaggle/input/hull-tactical-market-prediction/train.csv")
    if train_path.exists():
        print(f"Loading training data for warm start: {train_path}")
        # Load enough rows to cover the window
        # We need to apply same cleaning as inference
        train_df = pl.read_csv(train_path)
        
        # Sort by date
        if "date_id" in train_df.columns:
            train_df = train_df.sort("date_id")
            
        # Take last 60+ rows
        warmup_df = train_df.tail(100) 
        
        print(f"Injecting {len(warmup_df)} rows into history...")
        
        for row in warmup_df.iter_rows(named=True):
            # Simulate "prepare_input" logic manually to populate history
            # 1. Clean
            row_clean = row.copy()
            for c in runtime.feature_cols:
                if row_clean.get(c) is None:
                    row_clean[c] = 0.0
            
            # 2. Add to history
            runtime.history.append(row_clean)
            
            # 3. Also warmup return_history for Vol/Trend calculation
            if "market_forward_excess_returns" in row: # Train target is approx lagged_forward_returns for next day
                 # Note: In inference we look at 'lagged_forward_returns' (T-1). 
                 # In train, 'market_forward_excess_returns' is T.
                 # So we can treat it as the return stream.
                 val = row["market_forward_excess_returns"]
                 if val is not None:
                     runtime.return_history.append(float(val))
                     
        print(f"✓ History initialized. Current size: {len(runtime.history)}")
    else:
        print("⚠ Train file not found. Cannot warm start.")

    # ============ LOAD TEST DATA ============
    test_path = Path("kaggle/input/hull-tactical-market-prediction/test.csv")
    if not test_path.exists():
        print(f"Error: Test file not found at {test_path}")
        return
        
    print(f"Loading test data from {test_path}...")
    test_df = pl.read_csv(test_path)
    
    # Ensure date_id is sorted
    if "date_id" in test_df.columns:
        test_df = test_df.sort("date_id")
        
    print(f"Total rows: {len(test_df)}")
    
    # Simulate batch processing
    if "date_id" in test_df.columns:
        batches = test_df.partition_by("date_id", maintain_order=True)
        print(f"Processing {len(batches)} batches (dates)...")
    else:
        batches = [test_df.slice(i, 1) for i in range(len(test_df))]
        print(f"Processing {len(batches)} rows...")
        
    predictions = []
    debug_records = []  # Store all debug info
    
    # Enable debug mode to print intermediate values
    print("\n" + "="*80)
    print("DEBUG MODE: Printing intermediate predictions for first 5 batches")
    print("="*80)
    
    for i, batch in enumerate(tqdm(batches, desc="Inference")):
        try:
            # Call predict function
            # Since history is warmed up, this should trigger model inference immediately
            pred = predict(batch)
            
            date_id = batch["date_id"][0] if "date_id" in batch.columns else -1
            
            predictions.append({
                "date_id": date_id,
                "prediction": pred
            })
            
            # Save debug info for ALL batches
            if runtime.last_prediction_debug is not None:
                debug = runtime.last_prediction_debug.copy()
                debug["date_id"] = date_id
                debug_records.append(debug)
            
            # Print debug info for first 5 batches
            if i < 5:
                print(f"\n--- Batch {i+1} (date_id={date_id}) ---")
                if runtime.last_prediction_debug is not None:
                    debug = runtime.last_prediction_debug
                    print(f"  N-HiTS y_hat:    {debug['nhits_y_hat']:.6f}")
                    print(f"  Roll Vol:        {debug['roll_vol']:.6f}")
                    print(f"  Roll Trend:      {debug['roll_trend']:.6f}")
                    print(f"  y_hat_z:         {debug['y_hatz']:.4f}")
                    print(f"  yhat_ema:        {debug['yhat_ema']:.6f}")
                    print(f"  yhat_std:        {debug['yhat_std']:.6f}")
                    print(f"  Confidence:      {debug['conf_abs']:.4f}")
                    print(f"  P8 (raw):        {debug['p8_raw']:.6f}")
                    print(f"  S2 (raw):        {debug['s2_raw']:.6f}")
                    print(f"  Final Position:  {debug['final_action']:.6f}")
                else:
                    print(f"  Final Position: {pred:.6f} (no debug info)")
                
        except Exception as e:
            print(f"Error processing batch: {e}")
            predictions.append({
                "date_id": -1,
                "prediction": 0.8 # Fallback
            })
            
    # Convert to DataFrame
    result_df = pl.DataFrame(predictions)
    
    # Save submission
    out_path = "submission.parquet"
    result_df.write_parquet(out_path)
    
    # Save debug info
    if debug_records:
        debug_df = pl.DataFrame(debug_records)
        debug_path = "submission_debug.parquet"
        debug_df.write_parquet(debug_path)
        print(f"  Debug info saved to: {debug_path}")
    
    print("\n" + "="*50)
    print(f"Inference Complete. Saved to {out_path}")
    print("Stats:")
    print(result_df.describe())
    print("="*50)
    
    # Show distribution
    preds = result_df["prediction"].to_numpy()
    print(f"Mean: {preds.mean():.4f}")
    print(f"Std:  {preds.std():.4f}")
    print(f"Min:  {preds.min():.4f}")
    print(f"Max:  {preds.max():.4f}")
    
    unique_preds = len(np.unique(preds.round(4)))
    print(f"Unique prediction values: {unique_preds}")
    
    if unique_preds < 10:
        print("⚠ Warning: Very few unique predictions. Maybe stuck in Cold Start or constant fallback?")
    else:
        print("✓ Predictions show variance, Policy likely active.")
    
    # ============ SENSITIVITY TEST ============
    print("\n" + "="*80)
    print("SENSITIVITY TEST: Testing PolicyHead with extreme inputs")
    print("="*80)
    
    # Baseline: typical values from above
    baseline_state = np.array([
        0.001,      # nhits_y_hat: small positive
        0.009,      # roll_vol
        0.0005,     # roll_trend
        0.0,        # y_hatz
        0.001,      # yhat_ema
        0.004,      # yhat_std
        0.5,        # conf_abs
        1.77,       # P8
        0.0         # S2
    ], dtype=np.float32)
    
    test_cases = [
        ("Baseline (neutral)", baseline_state),
        ("S2 = +10 (Very Bearish)", baseline_state.copy()),
        ("S2 = -10 (Very Bullish)", baseline_state.copy()),
        ("N-HiTS = +0.05 (Strong Bull)", baseline_state.copy()),
        ("N-HiTS = -0.05 (Strong Bear)", baseline_state.copy()),
        ("P8 = 5.0 (Extreme P8)", baseline_state.copy()),
    ]
    
    # Modify test states
    test_cases[1][1][8] = 10.0   # S2 = +10
    test_cases[2][1][8] = -10.0  # S2 = -10
    test_cases[3][1][0] = 0.05   # nhits_y_hat = +0.05
    test_cases[4][1][0] = -0.05  # nhits_y_hat = -0.05
    test_cases[5][1][7] = 5.0    # P8 = 5.0
    
    print(f"\n{'Case':<30} | {'Position':<10} | {'Delta vs Baseline':<15}")
    print("-" * 60)
    
    baseline_pos = None
    for name, state in test_cases:
        state_t = torch.from_numpy(state).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
        pos = runtime.policy_head.get_action(state_t, deterministic=True).item()
        
        if baseline_pos is None:
            baseline_pos = pos
            delta_str = "-"
        else:
            delta = pos - baseline_pos
            delta_str = f"{delta:+.4f}"
        
        print(f"{name:<30} | {pos:<10.6f} | {delta_str:<15}")
    
    print("\n" + "="*80)
    print("If all deltas are near zero, PolicyHead may not be learning from inputs.")
    print("="*80)

if __name__ == "__main__":
    run_local_inference()