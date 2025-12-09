#!/usr/bin/env python
"""
Baseline Strategy: Always output 1.0 (neutral position)
This tests if the market is bullish - if this gets a good score, 
our model might just be riding the market beta.
"""

import os
import sys
from pathlib import Path

# Add kaggle_evaluation to path
_kaggle_eval_paths = [
    Path("/kaggle/input/hull-tactical-market-prediction/kaggle_evaluation"),
    Path("kaggle/input/hull-tactical-market-prediction/kaggle_evaluation"),
]

for _eval_path in _kaggle_eval_paths:
    if _eval_path.exists():
        sys.path.insert(0, str(_eval_path.parent))
        print(f"✓ Added kaggle_evaluation to path: {_eval_path.parent}")
        break

import kaggle_evaluation.default_inference_server
import polars as pl


def predict(test: pl.DataFrame) -> float:
    """
    Baseline strategy: Always return 1.0 (neutral)
    
    This is the simplest possible strategy:
    - No buying or selling
    - Just hold at neutral position
    - If this gets a good score, the market is trending up
    """
    return 1.0


# Create inference server
inference_server = kaggle_evaluation.default_inference_server.DefaultInferenceServer(predict)

if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    print("Running in Kaggle competition mode...")
    inference_server.serve()
else:
    print("="*60)
    print("BASELINE TEST: Always output 1.0 (neutral)")
    print("="*60)
    
    # Local testing
    kaggle_path = "/kaggle/input/hull-tactical-market-prediction/"
    local_path = "kaggle/input/hull-tactical-market-prediction/"
    
    if os.path.exists(kaggle_path):
        data_path = kaggle_path
    elif os.path.exists(local_path):
        data_path = local_path
    else:
        print("✗ Data path not found")
        sys.exit(1)
        
    print(f"Data path: {data_path}")
    inference_server.run_local_gateway((data_path,))
    
    print("\n" + "="*60)
    print("Baseline test complete!")
    print("Now compare this score with your model's score.")
    print("If baseline is close to your model, you're just riding beta.")
    print("="*60)

